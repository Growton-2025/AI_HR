
import os
import re
import json
import hashlib
import base64
import time
import asyncio
import logging
import threading
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from backend.api import schemas, deps
from backend.db.connection import (
    get_db_connection_context,
)
from backend.services.call_artifacts import transcript_preview
from backend.services.linkedin_normalize import normalize_linkedin
from backend.pipeline.query import (
    process_query_main,
    load_all_profiles_from_db,
    PROFILES_BY_ID,
    ALL_COMPANY_NAMES,
    TokenCostTracker,
    profiles_to_excel,
    SALES_TAXONOMY,
    SEGMENT_SYNONYMS,
    COMPANY_DETAILS_TAXONOMY,
    CULTURE_TAXONOMY,
    GEOGRAPHY_COUNTRY_TO_REGION_MAP,
    is_cache_initialized,
    initialize_cache,
    count_active_candidates_from_db,
)
from backend.services.candidate_pool import profile_passes_scope, VIEW_SCOPE_MASTER, POOL_SOURCE_RECRUITER_UPLOAD

router = APIRouter()
logger = logging.getLogger(__name__)
_analytics_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_ANALYTICS_CACHE_TTL = 60
_analytics_profile_cache_init_lock = asyncio.Lock()

# Cross-worker drift detection for the in-memory profile cache.
# PROFILES_BY_ID lives in one process; under gunicorn (WEB_CONCURRENCY=4) a write
# handled by one worker leaves the other three holding a pre-write snapshot with
# nothing to ever correct them. Headline counts no longer depend on that cache
# (they are read from SQL), but /stats, the candidate list and the analytics
# distributions still do — so cheaply notice the drift and rebuild in the
# background. Deliberately never blocks a request: initialize_cache() is a full
# reload measured in tens of seconds.
_PROFILE_DRIFT_CHECK_TTL = 60
_profile_drift_checked_at = 0.0
_profile_drift_reload_running = False
_profile_drift_lock = threading.Lock()


def _reload_profile_cache_if_drifted() -> None:
    """Compare the cached profile count against the DB and rebuild on mismatch."""
    global _profile_drift_reload_running
    try:
        from backend.pipeline import query as query_mod

        db_total = query_mod.count_all_candidates_from_db()
        cached_total = len(query_mod.PROFILES_BY_ID)
        if db_total is None or db_total == cached_total:
            return
        logger.warning(
            "Profile cache drift detected (cached=%s db=%s); rebuilding in background.",
            cached_total,
            db_total,
        )
        query_mod.initialize_cache()
        invalidate_candidate_analytics_cache()
        try:
            from backend.api.routes import browse as browse_mod

            browse_mod._invalidate_browse_cache()
        except Exception:
            pass
    except Exception as exc:
        logger.error("Profile cache drift check failed: %s", exc)
    finally:
        with _profile_drift_lock:
            _profile_drift_reload_running = False


def schedule_profile_cache_drift_check() -> None:
    """Fire-and-forget drift check, rate-limited to once per _PROFILE_DRIFT_CHECK_TTL."""
    global _profile_drift_checked_at, _profile_drift_reload_running
    # Nothing to compare against an unpopulated cache — a cold cache is already
    # handled by the explicit warm/init path, and rebuilding from here would only
    # race it. (This also keeps the background thread from perturbing global
    # cache state in tests that deliberately start from an empty cache.)
    if not PROFILES_BY_ID:
        return
    now = time.monotonic()
    with _profile_drift_lock:
        if _profile_drift_reload_running:
            return
        if now - _profile_drift_checked_at < _PROFILE_DRIFT_CHECK_TTL:
            return
        _profile_drift_checked_at = now
        _profile_drift_reload_running = True
    threading.Thread(
        target=_reload_profile_cache_if_drifted,
        name="profile-cache-drift-check",
        daemon=True,
    ).start()


def invalidate_candidate_analytics_cache() -> None:
    """Clear cached analytics summaries after candidate counts/statuses change."""
    _analytics_cache.clear()


def invalidate_candidate_count_caches(
    *,
    refresh_profile_ids: Optional[List[int]] = None,
    reload_profiles: bool = False,
) -> None:
    """Invalidate count-bearing caches and optionally refresh the in-memory profile cache."""
    invalidate_candidate_analytics_cache()
    try:
        from backend.api.routes import browse as browse_mod

        browse_mod._invalidate_browse_cache()
    except Exception:
        pass

    if reload_profiles:
        try:
            from backend.pipeline import query

            query.initialize_cache()
        except Exception:
            pass
        return

    if refresh_profile_ids:
        try:
            from backend.pipeline import query

            query.refresh_profiles_in_cache(refresh_profile_ids)
        except Exception:
            try:
                query.initialize_cache()
            except Exception:
                pass

@router.get("/candidates")
async def get_candidates(
    limit: int = 100,
    offset: int = 0,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Get paginated list of candidates scoped to the current user (recruiter = own pool only)."""
    if not is_cache_initialized():
        await asyncio.to_thread(initialize_cache)
    all_profiles = list(PROFILES_BY_ID.values())
    if (current_user.role or "").strip().lower() == "admin":
        scoped = [p for p in all_profiles if not p.get("is_archived")]
    else:
        scoped = [
            p
            for p in all_profiles
            if not p.get("is_archived")
            and p.get("owner_user_id") == current_user.id
        ]
    total = len(scoped)
    paginated = scoped[offset : offset + limit]

    # Return simplified version for listing
    simplified = []
    for p in paginated:
        primary_role = p.get('roles', [{}])[0] if p.get('roles') else {}
        simplified.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "linkedin": p.get("linkedin"),
            "location": p.get("location"),
            "headline": p.get("headline"),
            "total_experience_years": p.get("total_experience_years"),
            "max_people_managed": p.get("max_people_managed"),
            "current_title": primary_role.get("title"),
            "current_company": primary_role.get("company")
        })

    return {
        "candidates": simplified,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@router.get("/candidates/analytics")
async def get_candidate_analytics(current_user: schemas.User = Depends(deps.get_current_user)):
    """Get performance analytics for the recruiter/admin"""
    from backend.pipeline.query import get_analytics_summary

    # Headline counts below come from SQL and are always current; this only
    # keeps the cache-derived distributions from drifting on workers that did
    # not handle the write. Runs in a background thread — never delays this call.
    schedule_profile_cache_drift_check()

    if not is_cache_initialized() or not PROFILES_BY_ID:
        async with _analytics_profile_cache_init_lock:
            if not is_cache_initialized() or not PROFILES_BY_ID:
                await asyncio.to_thread(initialize_cache)

    if not PROFILES_BY_ID:
        active_count = await asyncio.to_thread(count_active_candidates_from_db)
        if active_count and active_count > 0:
            logger.error(
                "analytics profile cache unavailable after reload; active_candidates=%s",
                active_count,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "profile_cache_unavailable",
                    "message": "Candidate cache is warming or unavailable. Please retry shortly.",
                    "metadata": {
                        "active_candidates": active_count,
                        "cache_initialized": is_cache_initialized(),
                        "profile_count": len(PROFILES_BY_ID),
                    },
                },
            )

    # Admin analytics are org-wide and identical for every admin user (see
    # get_analytics_summary), so share one cache entry across all admins
    # instead of missing per-admin-id — recruiters still get their own entry
    # since their view is scoped to their own pool.
    role_l = (current_user.role or "").strip().lower()
    key = role_l if role_l == "admin" else f"{role_l}:{current_user.id}"
    cached = _analytics_cache.get(key)
    if cached and time.monotonic() - cached[0] < _ANALYTICS_CACHE_TTL:
        return cached[1]
    data = await get_analytics_summary(current_user.email, current_user.role, current_user.id)
    _analytics_cache[key] = (time.monotonic(), data)
    return data

@router.get("/candidates/{candidate_id}")
async def get_candidate(
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
    view_scope: Optional[str] = None,
    recruiter_filter_id: Optional[int] = None,
):
    """Get detailed candidate profile (scoped)."""
    prof = PROFILES_BY_ID.get(candidate_id)
    if not prof or prof.get("is_archived"):
        raise HTTPException(status_code=404, detail="Candidate not found")
    if (current_user.role or "").strip().lower() == "admin":
        scope = view_scope or VIEW_SCOPE_MASTER
        if not profile_passes_scope(
            prof,
            user_role=(current_user.role or "").strip().lower(),
            user_id=current_user.id,
            view_scope=scope,
            recruiter_filter_id=recruiter_filter_id,
        ):
            raise HTTPException(status_code=404, detail="Candidate not found")
    elif prof.get("owner_user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return prof

@router.post("/search")
async def search_candidates(request: schemas.SearchRequest, current_user: schemas.User = Depends(deps.get_current_user)):
    """
    Synchronous search endpoint - waits for complete results
    For real-time streaming, use WebSocket endpoint
    """
    session_id = request.session_id or hashlib.sha256(os.urandom(32)).hexdigest()
    tracker = TokenCostTracker()

    results = []
    status_messages = []
    verified_count = 0
    total_reviewed = 0
    potential_count = 0
    filter_debug = None

    try:
        async for item in process_query_main(
            request.query,
            session_id,
            tracker,
            screening_user_id=current_user.id,
            screening_role=current_user.role,
            source_type=request.source_type,
            source_role_id=request.source_role_id,
            use_web_search=bool(request.use_web_search),
        ):
            if isinstance(item, str):
                status_messages.append(item)
            elif isinstance(item, dict):
                msg_type = item.get("type")
                if msg_type == "complete":
                    results = item.get("data", [])
                    verified_count = int(item.get("verified_count") or 0)
                    total_reviewed = int(item.get("total_reviewed") or 0)
                    potential_count = int(item.get("potential_count") or max(0, len(results) - verified_count))
                    filter_debug = item.get("filter_debug")
                    break
                elif msg_type == "profile_chunk":
                    # Collect/update individual profiles as the stream refines shortlist status.
                    profile = item.get("data")
                    if profile:
                        profile_key = str(profile.get("id") or profile.get("linkedin") or profile.get("name") or len(results))
                        for idx, existing in enumerate(results):
                            existing_key = str(existing.get("id") or existing.get("linkedin") or existing.get("name") or idx)
                            if existing_key == profile_key:
                                results[idx] = {**existing, **profile}
                                break
                        else:
                            results.append(profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "candidates": results,
        "total": len(results),
        "query": request.query,
        "usage": {
            "total_tokens": tracker.total_tokens,
            "total_cost": round(tracker.total_cost, 6)
        },
        "verified_count": verified_count,
        "total_reviewed": total_reviewed,
        "potential_count": potential_count,
        "status_messages": status_messages,
        "filter_debug": filter_debug,
    }

@router.post("/export")
async def export_candidates(
    candidate_ids: List[int],
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Export selected candidates to Excel (returns base64). Scoped like browse."""

    allowed_ids: List[int] = []
    is_admin = (current_user.role or "").strip().lower() == "admin"
    for cid in candidate_ids:
        p = PROFILES_BY_ID.get(cid)
        if not p or p.get("is_archived"):
            continue
        if is_admin or p.get("owner_user_id") == current_user.id:
            allowed_ids.append(cid)
    selected = {cid: PROFILES_BY_ID[cid] for cid in allowed_ids}
    excel_bytes = profiles_to_excel(selected)

    return {
        "filename": f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "content": base64.b64encode(excel_bytes).decode('utf-8'),
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }


# --- WebSocket for Real-time Search Streaming ---

@router.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    WebSocket endpoint for real-time search streaming
    Sends progress updates and candidates as they're processed
    """
    await websocket.accept()

    # If the cache is still cold, warm it up while keeping the socket alive with
    # periodic status messages so the frontend fallback timer never fires.
    if not is_cache_initialized():
        await websocket.send_json({"type": "status", "message": "Loading candidate database..."})
        dot_task = None
        try:
            async def _keep_alive_dots():
                count = 0
                while not is_cache_initialized():
                    await asyncio.sleep(8)
                    count += 1
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Still loading candidate database{'.' * (count % 4 + 1)}"
                        })
                    except Exception:
                        break

            dot_task = asyncio.create_task(_keep_alive_dots())
            await asyncio.to_thread(initialize_cache)
        finally:
            if dot_task and not dot_task.done():
                dot_task.cancel()

    try:
        while True:
            # Receive search query
            data = await websocket.receive_json()
            query = data.get("query", "")
            session_id = data.get("session_id", hashlib.sha256(os.urandom(32)).hexdigest())
            token = data.get("token") or data.get("access_token")
            source_type = data.get("source_type")
            source_role_id = data.get("source_role_id")
            raw_use_web_search = data.get("use_web_search", data.get("useWebSearch", False))
            if isinstance(raw_use_web_search, str):
                use_web_search = raw_use_web_search.strip().lower() in {"1", "true", "yes", "on"}
            else:
                use_web_search = bool(raw_use_web_search)

            ws_user = deps.get_user_from_access_token(token)
            if not ws_user:
                await websocket.send_json({"type": "error", "message": "Authentication required"})
                continue

            if not query:
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue

            tracker = TokenCostTracker()
            
            pause_event = asyncio.Event()
            pause_event.set()
            
            send_lock = asyncio.Lock()

            async def command_listener():
                try:
                    while True:
                        msg = await websocket.receive_json()
                        action = msg.get("action")
                        if action == "pause":
                            pause_event.clear()
                        elif action == "resume":
                            pause_event.set()
                except Exception:
                    pass

            async def heartbeat_sender():
                try:
                    while True:
                        await asyncio.sleep(15)
                        try:
                            async with send_lock:
                                await websocket.send_json({"type": "ping"})
                        except Exception:
                            break
                except asyncio.CancelledError:
                    pass

            cmd_task = asyncio.create_task(command_listener())
            ping_task = asyncio.create_task(heartbeat_sender())

            try:
                async def send_event(payload: Dict[str, Any]) -> bool:
                    try:
                        async with send_lock:
                            await websocket.send_json(jsonable_encoder(payload))
                        return True
                    except (WebSocketDisconnect, RuntimeError):
                        return False

                # Use the pipeline generator
                async for item in process_query_main(
                    query,
                    session_id,
                    tracker,
                    screening_user_id=ws_user.id,
                    screening_role=ws_user.role,
                    source_type=source_type,
                    source_role_id=source_role_id,
                    pause_event=pause_event,
                    use_web_search=use_web_search,
                ):
                    if isinstance(item, str):
                        # Status message
                        if not await send_event({
                            "type": "status",
                            "message": item
                        }):
                            break

                    elif isinstance(item, dict):
                        msg_type = item.get("type")

                        if msg_type == "progress_start":
                            if not await send_event({
                                "type": "progress_start",
                                "total": item.get("total", 0)
                            }):
                                break

                        elif msg_type == "progress":
                            progress_payload: Dict[str, Any] = {
                                "type": "progress",
                                "current": item.get("current"),
                                "total": item.get("total"),
                            }
                            if item.get("phase"):
                                progress_payload["phase"] = item["phase"]
                            if item.get("message"):
                                progress_payload["message"] = item["message"]
                            if item.get("passed") is not None:
                                progress_payload["passed"] = item["passed"]
                            if not await send_event(progress_payload):
                                break

                        elif msg_type == "profile_chunk":
                            chunk_payload: Dict[str, Any] = {
                                "type": "candidate",
                                "data": item.get("data"),
                                "current": item.get("current"),
                                "total": item.get("total"),
                                "reviewed": item.get("reviewed"),
                                "verified": item.get("verified"),
                            }
                            if item.get("phase"):
                                chunk_payload["phase"] = item["phase"]
                            if not await send_event(chunk_payload):
                                break

                        elif msg_type == "candidate_batch":
                            # Batch of candidates from the scoring phase (before reasoning)
                            if not await send_event({
                                "type": "candidate_batch",
                                "phase": item.get("phase", "scoring"),
                                "data": item.get("data", []),
                                "reviewed": item.get("reviewed"),
                                "passed": item.get("passed"),
                                "total_pool": item.get("total_pool"),
                            }):
                                break

                        elif msg_type == "complete":
                            if not await send_event({
                                "type": "complete",
                                "candidates": item.get("data", []),
                                "total": len(item.get("data", [])),
                                "verified_count": item.get("verified_count", 0),
                                "total_reviewed": item.get("total_reviewed", 0),
                                "potential_count": item.get("potential_count", 0),
                                "filter_debug": item.get("filter_debug"),
                                "usage": {
                                    "total_tokens": tracker.total_tokens,
                                    "total_cost": round(tracker.total_cost, 6)
                                }
                            }):
                                break
                            break

            except Exception as e:
                # Log error but don't crash loop unless critical
                print(f"Error during search: {e}")
                # Try to send error to client if possible
                if not await send_event({
                    "type": "error",
                    "message": str(e)
                }):
                    break
            finally:
                cmd_task.cancel()
                ping_task.cancel()

    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError is raised by Starlette if we try to send after close
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"Unexpected WebSocket error: {e}")

@router.patch("/candidates/{candidate_id}")
async def update_candidate(candidate_id: int, data: Dict[str, Any], current_user: schemas.User = Depends(deps.get_current_user)):
    """Update candidate fields manually"""

    prof = PROFILES_BY_ID.get(candidate_id)
    if not prof or prof.get("is_archived"):
        raise HTTPException(status_code=404, detail="Candidate not found")
    if (current_user.role or "").strip().lower() != "admin" and prof.get("owner_user_id") is None:
        raise HTTPException(status_code=403, detail="Master library rows are read-only")
    if (current_user.role or "").strip().lower() != "admin" and prof.get("owner_user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 1. Update Database
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                for field, value in data.items():
                    # Basic whitelist for security
                    # 'phone' is the frontend alias for 'mobile_phone' — both must be allowed
                    if field not in ['email', 'mobile_phone', 'phone', 'linkedin', 'notes', 'name', 'first_name', 'last_name']:
                        continue

                    # Map frontend field name to actual DB column name
                    db_field = 'mobile_phone' if field == 'phone' else field

                    cur.execute(
                        f"UPDATE candidates SET {db_field} = %s, updated_at = NOW() WHERE id = %s",
                        (value, candidate_id),
                    )
                conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database update failed: {e}")

    # 2. Update Cache
    if candidate_id in PROFILES_BY_ID:
        profile = PROFILES_BY_ID[candidate_id]
        for field, value in data.items():
            profile[field] = value
            # Handle alias
            if field == 'phone': profile['mobile_phone'] = value
        PROFILES_BY_ID[candidate_id] = profile
    invalidate_candidate_count_caches(refresh_profile_ids=[candidate_id])

    # 3. A new phone number un-tags "wrong number" and resumes any paused call
    # cadence: the attempt where the wrong number was found is retried today.
    # Separate transaction so a failure here never breaks the contact update.
    phone_value = next(
        (str(data.get(f) or '').strip() for f in ('mobile_phone', 'phone') if str(data.get(f) or '').strip()),
        None,
    )
    if phone_value:
        try:
            cleared_wrong_number = False
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE candidates SET mobile_phone_wrong = FALSE WHERE id = %s AND COALESCE(mobile_phone_wrong, FALSE)",
                            (candidate_id,),
                        )
                        cleared_wrong_number = bool(cur.rowcount)
                        if cleared_wrong_number:
                            cur.execute(
                                """
                                INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
                                SELECT c.candidate_id, c.list_id, 'pending', CURRENT_DATE, c.task_title
                                FROM calls c
                                WHERE c.candidate_id = %s
                                  AND c.outcome = 'Wrong Number'
                                  AND c.created_at = (
                                      SELECT MAX(c2.created_at) FROM calls c2
                                      WHERE c2.candidate_id = c.candidate_id AND c2.list_id = c.list_id
                                  )
                                  AND NOT EXISTS (
                                      SELECT 1 FROM calls p
                                      WHERE p.candidate_id = c.candidate_id
                                        AND p.list_id = c.list_id
                                        AND p.status = 'pending'
                                  )
                                """,
                                (candidate_id,),
                            )
                    conn.commit()
            if cleared_wrong_number:
                from backend.api.routes.calls import invalidate_calls_cache
                invalidate_calls_cache()
        except Exception as e:
            print(f"WARNING: wrong-number reset failed for candidate {candidate_id}: {e}")

    return {"success": True, "data": data}


@router.get("/candidates/{candidate_id}/activity")
async def get_candidate_activity(
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    from backend.api.routes.calls import ensure_calls_schema_ready, get_call_list_owner

    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    try:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        COALESCE(c.completed_at, c.updated_at, c.created_at) AS occurred_at,
                        c.status,
                        c.outcome,
                        c.duration,
                        c.recording_url,
                        c.summary,
                        c.transcript,
                        c.notes,
                        c.plivo_virtual_number,
                        cand.mobile_phone,
                        COALESCE(c.likely_voicemail, FALSE)
                    FROM calls c
                    JOIN call_lists cl ON c.list_id = cl.id
                    JOIN candidates cand ON c.candidate_id = cand.id
                    WHERE c.candidate_id = %s
                      AND LOWER(COALESCE(cl.created_by, '')) = %s
                      AND c.status = 'completed'
                    ORDER BY COALESCE(c.completed_at, c.updated_at, c.created_at) DESC, c.id DESC
                    """,
                    (candidate_id, owner),
                )
                rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    items = []
    for row in rows:
        summary = (row[6] or row[8] or "").strip() or None
        items.append(
            {
                "id": row[0],
                "type": "call_completed",
                "occurred_at": row[1],
                "status": row[2],
                "outcome": row[3],
                "duration_seconds": row[4] or 0,
                "recording_url": row[5],
                "summary": summary,
                "transcript_preview": transcript_preview(row[7]),
                "notes": (row[8] or "").strip() or None,
                "from_number": row[9],
                "to_number": row[10],
                "source_url": row[5],
                "likely_voicemail": bool(row[11]),
            }
        )

    return {"items": items}


_LINKEDIN_HOST_RE = re.compile(r"linkedin\.com$", re.I)


@router.post("/candidates")
async def create_candidate(
    payload: schemas.CandidateCreate,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Add a single candidate directly (the alternative to CSV upload).

    Admins have no personal candidate pool of their own — same as the
    "Role owner's pool" search scope elsewhere on this page, an admin acting
    here is acting on behalf of the role's owning recruiter, so the new
    candidate is owned by that recruiter (recruitment_roles.user_id), not the
    admin account. This requires a role_id; an admin with no role context has
    nowhere to attach the candidate to.
    """
    first_name = payload.first_name.strip()
    last_name = payload.last_name.strip()
    city = payload.city.strip()
    title = payload.title.strip()
    linkedin_raw = payload.linkedin.strip()
    if not first_name or not last_name or not city or not title or not linkedin_raw:
        raise HTTPException(
            status_code=400,
            detail="First name, last name, LinkedIn URL, city, and title are required",
        )

    # Hard-validate the LinkedIn URL — stricter than CSV import, which only
    # soft-flags mismatches for review. A manually typed field is more
    # error-prone than a mapped CSV column.
    probe_url = linkedin_raw if linkedin_raw.startswith("http") else f"https://{linkedin_raw.lstrip('/')}"
    try:
        parsed = urlparse(probe_url)
    except Exception:
        parsed = None
    host = (parsed.netloc or "").lower() if parsed else ""
    if host.startswith("www."):
        host = host[4:]
    if not parsed or not _LINKEDIN_HOST_RE.search(host):
        raise HTTPException(status_code=400, detail="Enter a valid linkedin.com profile URL")

    normalized_li = normalize_linkedin(linkedin_raw)
    name = f"{first_name} {last_name}".strip()
    raw_fields = json.dumps({"import_company": payload.company_name}) if payload.company_name else "{}"
    is_admin = (current_user.role or "").strip().lower() == "admin"
    if is_admin and not payload.role_id:
        raise HTTPException(status_code=400, detail="Admins can only add a candidate from within a role")

    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                owner_id = current_user.id
                if is_admin:
                    cur.execute("SELECT user_id FROM recruitment_roles WHERE id = %s", (payload.role_id,))
                    role_row = cur.fetchone()
                    if not role_row:
                        raise HTTPException(status_code=404, detail="Role not found")
                    owner_id = role_row[0]

                cur.execute(
                    "SELECT id FROM candidates WHERE normalized_linkedin = %s AND owner_user_id = %s",
                    (normalized_li, owner_id),
                )
                if cur.fetchone():
                    raise HTTPException(status_code=409, detail="A candidate with this LinkedIn URL already exists")

                cur.execute(
                    """
                    INSERT INTO candidates (
                        name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
                        location, email, mobile_phone, notes, about, raw_fields,
                        owner_user_id, pool_source, created_by, status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb),
                        %s, %s, %s, 'To be started'
                    )
                    RETURNING id
                    """,
                    (
                        name, first_name, last_name, linkedin_raw, normalized_li, city, title,
                        payload.location or city, payload.email, payload.phone, payload.notes, payload.about, raw_fields,
                        owner_id, POOL_SOURCE_RECRUITER_UPLOAD, current_user.email or str(owner_id),
                    ),
                )
                candidate_id = cur.fetchone()[0]

                if payload.role_id:
                    cur.execute(
                        """
                        INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
                        VALUES (%s, %s, '--', '')
                        ON CONFLICT (role_id, candidate_id) DO NOTHING
                        """,
                        (payload.role_id, candidate_id),
                    )
                conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create candidate: {exc}")

    invalidate_candidate_count_caches(refresh_profile_ids=[candidate_id])
    if payload.role_id:
        try:
            from backend.api.routes.roles import invalidate_role_detail_cache_for_candidate
            invalidate_role_detail_cache_for_candidate(candidate_id)
        except Exception:
            pass

    return {"success": True, "candidate_id": candidate_id}

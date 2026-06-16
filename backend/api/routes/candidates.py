
import os
import hashlib
import base64
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from backend.api import schemas, deps
from backend.db.connection import (
    get_db_connection_context,
)
from backend.services.frejun_calls import transcript_preview
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
from backend.services.candidate_pool import profile_passes_scope, VIEW_SCOPE_MASTER

router = APIRouter()
logger = logging.getLogger(__name__)
_analytics_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_ANALYTICS_CACHE_TTL = 60
_analytics_profile_cache_init_lock = asyncio.Lock()


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

    key = f"{current_user.id}:{current_user.role}"
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
                            if not await send_event({
                                "type": "progress",
                                "current": item.get("current"),
                                "total": item.get("total")
                            }):
                                break

                        elif msg_type == "profile_chunk":
                            if not await send_event({
                                "type": "candidate",
                                "data": item.get("data"),
                                "current": item.get("current"),
                                "total": item.get("total"),
                                "reviewed": item.get("reviewed"),
                                "verified": item.get("verified"),
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
                        c.frejun_virtual_number,
                        cand.mobile_phone,
                        c.frejun_link,
                        c.frejun_summary_url
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
                "from_number": row[9],
                "to_number": row[10],
                "source_url": row[11] or row[12] or row[5],
            }
        )

    return {"items": items}


from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from backend.api import schemas, deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.api.routes.candidate_imports import build_upload_preview_response, commit_upload_file
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()

# Use lazy import to avoid circular dependency
from backend.pipeline.query import PROFILES_BY_ID, refresh_profiles_in_cache

_ROLE_DETAIL_CACHE: Dict[str, tuple] = {}
_ROLE_DETAIL_CACHE_TTL_SECONDS = 30


def invalidate_role_detail_cache(role_id: Optional[int] = None) -> None:
    """Drop cached role details after assignments change."""
    if role_id is None:
        _ROLE_DETAIL_CACHE.clear()
        return
    stale_keys = [
        key
        for key, cached in _ROLE_DETAIL_CACHE.items()
        if cached and cached[1].get("id") == role_id
    ]
    for key in stale_keys:
        _ROLE_DETAIL_CACHE.pop(key, None)

def fetch_candidates_from_db(candidate_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetch full candidate profiles from DB for IDs not in memory cache"""
    if not candidate_ids:
        return {}
    
    conn = get_db_connection()
    if not conn:
        return {}
        
    fetched_profiles = {}
    try:
        with conn.cursor() as cur:
            # 1. Fetch basic info
            cur.execute(f"SELECT id, name, linkedin, location, headline, about, total_experience_years, max_people_managed, email, mobile_phone FROM candidates WHERE id IN %s", (tuple(candidate_ids),))
            rows = cur.fetchall()
            
            for row in rows:
                fetched_profiles[row[0]] = {
                    "id": row[0],
                    "name": row[1],
                    "linkedin": row[2],
                    "location": row[3],
                    "headline": row[4],
                    "about": row[5],
                    "summary": row[5] or row[4] or "No summary available", # Fallback to headline
                    "total_experience_years": float(row[6]) if row[6] is not None else 0.0,
                    "max_people_managed": row[7] or 0,
                    "email": row[8],
                    "mobile_phone": row[9],
                    "roles": [] # Will populate below
                }
            
            # 2. Fetch roles and company details
            if fetched_profiles:
                cur.execute(f"""
                    SELECT 
                        r.candidate_id, r.title, r.details, r.duration_years,
                        c.name, c.funding_stage, c.revenue, c.business_model, 
                        c.product_service, c.customer_segment, c.customer_presence, 
                        c.culture_type, c.headquarters
                    FROM roles r
                    JOIN companies c ON r.company_id = c.id
                    WHERE r.candidate_id IN %s
                """, (tuple(fetched_profiles.keys()),))
                
                role_rows = cur.fetchall()
                for rr in role_rows:
                    cid = rr[0]
                    if cid in fetched_profiles:
                        company_details = {
                            "funding_stage": rr[5],
                            "revenue": rr[6],
                            "business_model": rr[7],
                            "product_service": rr[8],
                            "customer_segment": rr[9] if rr[9] is not None else [],
                            "customer_presence": rr[10] if rr[10] is not None else [],
                            "culture_type": rr[11],
                            "headquarters": rr[12],
                            "industry": rr[8] or ""
                        }
                        
                        fetched_profiles[cid]["roles"].append({
                            "title": rr[1],
                            "details": rr[2],
                            "duration_years": float(rr[3]) if rr[3] is not None else 0.0,
                            "company": rr[4],
                            "company_details": company_details
                        })

        # Debug logging
        for cid, profile in fetched_profiles.items():
            logger.info(f"DEBUG PROFILE {cid}: Name={profile.get('name')}, Summary={profile.get('summary')[:30] if profile.get('summary') else 'None'}, Roles={len(profile.get('roles', []))}")
            if profile.get('roles'):
                logger.info(f"  Role 0: {profile['roles'][0].get('title')} at {profile['roles'][0].get('company')}")
            else:
                logger.info("  No roles found for this candidate.")

        return fetched_profiles
    except Exception as e:
        logger.error(f"Error fetching candidates from DB: {e}")
        return {}
    finally:
        return_db_connection(conn)

def refresh_candidate_contact_info(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Refresh email and mobile_phone from the database for each candidate using a single batch query."""
    if not candidates:
        return candidates
    
    candidate_ids = [c.get("id") for c in candidates if c.get("id")]
    if not candidate_ids:
        return candidates

    conn = get_db_connection()
    if not conn:
        logger.warning("No DB connection for refreshing contact info")
        return candidates
    
    try:
        with conn.cursor() as cur:
            # Use IN clause for batch fetch
            query = "SELECT id, email, mobile_phone FROM candidates WHERE id IN %s"
            cur.execute(query, (tuple(candidate_ids),))
            rows = cur.fetchall()
            
            # Create a lookup map
            contact_map = {row[0]: {"email": row[1], "mobile_phone": row[2]} for row in rows}
            logger.info(f"📧 Fetched contact info for {len(rows)} candidates from DB: {contact_map}")
            
            # Apply to candidates
            for candidate in candidates:
                cid = candidate.get("id")
                if cid in contact_map:
                    candidate["email"] = contact_map[cid]["email"]
                    candidate["mobile_phone"] = contact_map[cid]["mobile_phone"]
                    
        return candidates
    except Exception as e:
        logger.error(f"Error refreshing contact info: {e}")
        return candidates
    finally:
        return_db_connection(conn)


def _json_safe(value: Any) -> Any:
    """Convert cached profile values into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _role_owner_id(current_user: schemas.User, owner_user_id: Optional[int]) -> int:
    if owner_user_id and (current_user.role or "").strip().lower() == "admin":
        return owner_user_id
    return current_user.id


def _is_admin(current_user: schemas.User) -> bool:
    return (current_user.role or "").strip().lower() == "admin"


def _resolve_role(
    cur,
    current_user: schemas.User,
    role_name: Optional[str] = None,
    role_id: Optional[int] = None,
):
    """Resolve an accessible role. Admins may target any role; recruiters only their own."""
    if role_id:
        if _is_admin(current_user):
            cur.execute(
                """
                SELECT r.id, r.name, r.user_id
                FROM recruitment_roles r
                WHERE r.id = %s
                """,
                (role_id,),
            )
        else:
            cur.execute(
                """
                SELECT r.id, r.name, r.user_id
                FROM recruitment_roles r
                WHERE r.id = %s AND r.user_id = %s
                """,
                (role_id, current_user.id),
            )
        return cur.fetchone()

    if not role_name:
        return None

    if _is_admin(current_user):
        cur.execute(
            """
            SELECT r.id, r.name, r.user_id
            FROM recruitment_roles r
            WHERE r.name = %s
            ORDER BY CASE WHEN r.user_id = %s THEN 0 ELSE 1 END, r.created_at DESC
            LIMIT 1
            """,
            (role_name, current_user.id),
        )
    else:
        cur.execute(
            """
            SELECT r.id, r.name, r.user_id
            FROM recruitment_roles r
            WHERE r.user_id = %s AND r.name = %s
            """,
            (current_user.id, role_name),
        )
    return cur.fetchone()


@router.get("", response_model=Dict[str, List[Dict[str, Any]]])
async def get_roles(
    current_user: schemas.User = Depends(deps.get_current_user),
    owner_user_id: Optional[int] = Query(None),
    view_scope: Optional[str] = Query(None),
):
    """Get all roles for the current user with candidate counts"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    roles_list = []
    try:
        with conn.cursor() as cur:
            user_role = (current_user.role or "").strip().lower()
            if user_role == "admin" and not owner_user_id:
                logger.info("Fetching all roles for admin scope: %s", view_scope or "default")
                cur.execute("""
                    SELECT r.id, r.name, r.job_description, COUNT(DISTINCT c.id) as candidate_count,
                           COUNT(DISTINCT cu.id) as upload_count,
                           MAX(cu.completed_at) as latest_upload_at,
                           r.user_id,
                           u.name as owner_name,
                           u.email as owner_email
                    FROM recruitment_roles r
                    LEFT JOIN users u ON u.id = r.user_id
                    LEFT JOIN recruitment_role_candidates rc ON r.id = rc.role_id
                    LEFT JOIN candidates c ON c.id = rc.candidate_id AND COALESCE(c.is_archived, FALSE) = FALSE
                    LEFT JOIN candidate_uploads cu ON cu.role_id = r.id
                    GROUP BY r.id, r.name, r.job_description, r.user_id, u.name, u.email
                    ORDER BY r.created_at DESC
                """)
            else:
                target_user_id = _role_owner_id(current_user, owner_user_id)
                logger.info(f"Fetching roles for user_id: {target_user_id}")
                cur.execute("""
                    SELECT r.id, r.name, r.job_description, COUNT(DISTINCT c.id) as candidate_count,
                           COUNT(DISTINCT cu.id) as upload_count,
                           MAX(cu.completed_at) as latest_upload_at,
                           r.user_id,
                           u.name as owner_name,
                           u.email as owner_email
                    FROM recruitment_roles r
                    LEFT JOIN users u ON u.id = r.user_id
                    LEFT JOIN recruitment_role_candidates rc ON r.id = rc.role_id
                    LEFT JOIN candidates c ON c.id = rc.candidate_id AND COALESCE(c.is_archived, FALSE) = FALSE
                    LEFT JOIN candidate_uploads cu ON cu.role_id = r.id
                    WHERE r.user_id = %s
                    GROUP BY r.id, r.name, r.job_description, r.user_id, u.name, u.email
                    ORDER BY r.created_at DESC
                """, (target_user_id,))
            
            rows = cur.fetchall()
            logger.info("Found %s roles", len(rows))
            for row in rows:
                logger.info(f"  Role: {row[1]}, candidate_count: {row[3]}")
                roles_list.append({
                    "id": row[0],
                    "name": row[1],
                    "job_description": row[2] or "",
                    "candidate_count": row[3],
                    "upload_count": row[4],
                    "latest_upload_at": row[5].isoformat() if row[5] else None,
                    "owner_user_id": row[6],
                    "owner_name": row[7] or "",
                    "owner_email": row[8] or "",
                })
    finally:
        return_db_connection(conn)
    
    logger.info(f"Returning roles: {roles_list}")
    return {"roles": roles_list}

@router.post("")
async def create_role(role: schemas.RoleCreate, current_user: schemas.User = Depends(deps.get_current_user)):
    """Create a new role for the current user"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            logger.info(f"Checking for role '{role.name}' for user_id: {current_user.id}")
            cur.execute("SELECT id FROM recruitment_roles WHERE user_id = %s AND name = %s", (current_user.id, role.name))
            if cur.fetchone():
                logger.warning(f"Role '{role.name}' already exists for user_id: {current_user.id}")
                raise HTTPException(status_code=400, detail="Role already exists")
            
            cur.execute("""
                INSERT INTO recruitment_roles (user_id, name, job_description)
                VALUES (%s, %s, %s) RETURNING id, name, job_description
            """, (current_user.id, role.name, (role.job_description or "").strip() or None))
            result = cur.fetchone()
            conn.commit()
            
            return {
                "message": f"Role '{result[1]}' created",
                "id": result[0],
                "name": result[1],
                "job_description": result[2] or "",
            }
    finally:
        return_db_connection(conn)

@router.delete("/{role_name}")
async def delete_role(
    role_name: str,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Delete an accessible role. Admins may delete any role; recruiters only their own."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not role:
                 raise HTTPException(status_code=404, detail="Role not found")
            
            # Cascade delete will handle candidates
            cur.execute("DELETE FROM recruitment_roles WHERE id = %s", (role[0],))
            conn.commit()
            
            return {"message": f"Role '{role_name}' deleted"}
    finally:
        return_db_connection(conn)

@router.get("/{role_name}")
async def get_role(
    role_name: str,
    role_id: Optional[int] = Query(None),
    refresh: bool = Query(False),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Get role details with candidates using one DB round trip and a short cache."""
    cache_key = f"{current_user.id}:{current_user.role}:{role_id or role_name}"
    cached = _ROLE_DETAIL_CACHE.get(cache_key)
    if not refresh and cached and time.monotonic() - cached[0] < _ROLE_DETAIL_CACHE_TTL_SECONDS:
        return cached[1]

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor() as cur:
            if role_id:
                role_filter = "r.id = %s"
                params = [role_id]
                if not _is_admin(current_user):
                    role_filter += " AND r.user_id = %s"
                    params.append(current_user.id)
                role_order = "r.created_at DESC"
            elif _is_admin(current_user):
                role_filter = "r.name = %s"
                params = [role_name]
                role_order = "CASE WHEN r.user_id = %s THEN 0 ELSE 1 END, r.created_at DESC"
                params.append(current_user.id)
            else:
                role_filter = "r.user_id = %s AND r.name = %s"
                params = [current_user.id, role_name]
                role_order = "r.created_at DESC"

            cur.execute(
                f"""
                WITH selected_role AS (
                    SELECT r.id, r.name, r.job_description, r.user_id
                    FROM recruitment_roles r
                    WHERE {role_filter}
                    ORDER BY {role_order}
                    LIMIT 1
                )
                SELECT sr.id, sr.name, sr.job_description, sr.user_id,
                       u.name, u.email,
                       (SELECT COUNT(*) FROM candidate_uploads cu WHERE cu.role_id = sr.id),
                       (SELECT MAX(cu.completed_at) FROM candidate_uploads cu WHERE cu.role_id = sr.id),
                       COUNT(c.id) OVER (),
                       c.id, rc.priority, rc.feedback,
                       c.name, c.linkedin, c.location, c.headline, c.about,
                       c.email,
                       COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), ''),
                       c.status
                FROM selected_role sr
                LEFT JOIN users u ON u.id = sr.user_id
                LEFT JOIN recruitment_role_candidates rc ON rc.role_id = sr.id
                LEFT JOIN candidates c ON c.id = rc.candidate_id
                    AND COALESCE(c.is_archived, FALSE) = FALSE
                ORDER BY c.id NULLS LAST
                """,
                tuple(params),
            )
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="Role not found")

            candidate_ids = [row[9] for row in rows if row[9] is not None]
            if refresh and candidate_ids:
                refresh_profiles_in_cache(candidate_ids)

            candidates = []
            for row in rows:
                candidate_id = row[9]
                if candidate_id is None:
                    continue

                cached_profile = PROFILES_BY_ID.get(candidate_id)
                if cached_profile:
                    candidate = cached_profile.copy()
                else:
                    # The global cache may still be warming. Return enough current
                    # DB data for the role table immediately instead of doing more
                    # remote round trips.
                    candidate = {
                        "id": candidate_id,
                        "name": row[12] or "",
                        "linkedin": row[13] or "",
                        "location": row[14] or "",
                        "headline": row[15] or "",
                        "about": row[16] or "",
                        "summary": row[16] or row[15] or "",
                        "email": row[17] or "",
                        "mobile_phone": row[18] or "",
                        "status": row[19] or "To be started",
                        "roles": [],
                    }
                candidate["priority"] = row[10]
                candidate["feedback"] = row[11]
                candidates.append(candidate)

            role = rows[0]
            response = {
                "id": role[0],
                "name": role[1],
                "job_description": role[2] or "",
                "candidate_count": role[8] or 0,
                "upload_count": role[6] or 0,
                "latest_upload_at": role[7].isoformat() if role[7] else None,
                "owner_user_id": role[3],
                "owner_name": role[4] or "",
                "owner_email": role[5] or "",
                "candidates": [_json_safe(candidate) for candidate in candidates],
            }
            _ROLE_DETAIL_CACHE[cache_key] = (time.monotonic(), response)
            return response
    finally:
        return_db_connection(conn)


@router.get("/{role_name}/contacts")
async def get_role_contacts(
    role_name: str,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Lightweight polling endpoint for Clay/import contact updates."""
    cache_key = f"{current_user.id}:{current_user.role}:{role_name}"
    cached_role = _ROLE_DETAIL_CACHE.get(cache_key)
    if cached_role and time.monotonic() - cached_role[0] < 300:
        contacts = []
        for candidate in cached_role[1].get("candidates", []):
            candidate_id = candidate.get("id")
            current = PROFILES_BY_ID.get(candidate_id) or candidate
            contacts.append(
                {
                    "id": candidate_id,
                    "email": current.get("email") or "",
                    "mobile_phone": current.get("mobile_phone") or current.get("phone") or "",
                }
            )
        return {"contacts": contacts}

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor() as cur:
            resolved_role = _resolve_role(cur, current_user, role_name=role_name)
            if not resolved_role:
                raise HTTPException(status_code=404, detail="Role not found")

            cur.execute(
                """
                SELECT c.id, c.email,
                       COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), '')
                FROM recruitment_role_candidates rc
                JOIN candidates c ON c.id = rc.candidate_id
                WHERE rc.role_id = %s
                  AND COALESCE(c.is_archived, FALSE) = FALSE
                """,
                (resolved_role[0],),
            )
            return {
                "contacts": [
                    {"id": row[0], "email": row[1] or "", "mobile_phone": row[2] or ""}
                    for row in cur.fetchall()
                ]
            }
    finally:
        return_db_connection(conn)


@router.patch("/{role_name}")
async def update_role(
    role_name: str,
    update: schemas.RoleUpdate,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Update editable role context used by role-aware AI columns."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        with conn.cursor() as cur:
            resolved_role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not resolved_role:
                raise HTTPException(status_code=404, detail="Role not found")

            cur.execute(
                """
                UPDATE recruitment_roles
                SET job_description = %s
                WHERE id = %s
                RETURNING id, name, job_description
                """,
                ((update.job_description or "").strip() or None, resolved_role[0]),
            )
            role = cur.fetchone()
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
            conn.commit()
            return {
                "id": role[0],
                "name": role[1],
                "job_description": role[2] or "",
                "message": "Role updated",
            }
    finally:
        return_db_connection(conn)


@router.post("/{role_name}/upload/preview")
async def role_upload_preview(
    role_name: str,
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            if not _resolve_role(cur, current_user, role_name=role_name, role_id=role_id):
                raise HTTPException(status_code=404, detail="Role not found")
    finally:
        return_db_connection(conn)
    return await build_upload_preview_response(file, use_llm=use_llm)


@router.post("/{role_name}/upload/commit")
async def role_upload_commit(
    role_name: str,
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    enrichment_mode: str = Form("none"),
    duplicate_policy: str = Form("upsert_existing"),
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
            role_id = role[0]
    finally:
        return_db_connection(conn)

    result = await commit_upload_file(
        file=file,
        mapping_json=mapping_json,
        enrichment_mode=enrichment_mode,
        duplicate_policy=duplicate_policy,
        current_user=current_user,
        role_id=role_id,
    )
    return {
        **result,
        "role_id": role_id,
        "role_name": role_name,
    }


@router.post("/{role_name}/assign")
async def assign_candidates(
    role_name: str,
    assignment: schemas.CandidateAssignment,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Assign candidates to an accessible role."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
            
            role_id = role[0]
            resolved_role_name = role[1]
            requested_by_id = {}
            for item in assignment.assignments:
                if item.candidate_id is not None:
                    requested_by_id[int(item.candidate_id)] = item

            requested_ids = list(requested_by_id.keys())
            if requested_ids:
                cur.execute(
                    """
                    SELECT candidate_id
                    FROM recruitment_role_candidates
                    WHERE role_id = %s AND candidate_id = ANY(%s)
                    """,
                    (role_id, requested_ids),
                )
                existing_ids = {int(row[0]) for row in cur.fetchall()}
            else:
                existing_ids = set()

            added_ids = []
            already_existing_ids = [cid for cid in requested_ids if cid in existing_ids]

            for cid in requested_ids:
                if cid in existing_ids:
                    continue
                item = requested_by_id[cid]
                try:
                    cur.execute(
                        """
                        INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (role_id, candidate_id) DO NOTHING
                        RETURNING candidate_id
                        """,
                        (role_id, cid, item.priority or '--', item.feedback or ''),
                    )
                    inserted = cur.fetchone()
                    if inserted:
                        added_ids.append(int(inserted[0]))
                    else:
                        already_existing_ids.append(cid)
                except Exception as e:
                     logger.error(f"Error assigning candidate {cid}: {e}")
            
            conn.commit()
            from backend.api.routes.candidates import invalidate_candidate_count_caches

            invalidate_candidate_count_caches()

            added_count = len(added_ids)
            already_existing_count = len(set(already_existing_ids))
            if added_count and already_existing_count:
                message = f"{added_count} added, {already_existing_count} already existed in '{resolved_role_name}'"
            elif added_count:
                message = f"{added_count} added to '{resolved_role_name}'"
            elif already_existing_count:
                message = f"{already_existing_count} already existed in '{resolved_role_name}'"
            else:
                message = f"No candidates added to '{resolved_role_name}'"
            
            return {
                "message": message,
                "assigned_ids": added_ids,
                "added_ids": added_ids,
                "already_existing_ids": sorted(set(already_existing_ids)),
                "added_count": added_count,
                "already_existing_count": already_existing_count,
            }
    finally:
        return_db_connection(conn)

@router.post("/{role_name}/candidates/{candidate_id}/feedback")
async def update_candidate_feedback(
    role_name: str,
    candidate_id: int,
    feedback: schemas.CandidateFeedback,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Update priority and feedback for a candidate in an accessible role."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not role:
                 raise HTTPException(status_code=404, detail="Role not found")
            
            role_id = role[0]
            
            cur.execute("""
                UPDATE recruitment_role_candidates 
                SET priority = %s, feedback = %s
                WHERE role_id = %s AND candidate_id = %s
            """, (feedback.priority, feedback.feedback, role_id, candidate_id))
            
            if cur.rowcount == 0:
                 raise HTTPException(status_code=404, detail="Candidate not assigned to this role")
                 
            conn.commit()
            return {"message": "Feedback updated"}
    finally:
        return_db_connection(conn)

@router.delete("/{role_name}/candidates/{candidate_id}")
async def remove_candidate_from_role(
    role_name: str,
    candidate_id: int,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Remove a candidate from an accessible role."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_name=role_name, role_id=role_id)
            if not role:
                 raise HTTPException(status_code=404, detail="Role not found")
            
            role_id = role[0]
            
            cur.execute("""
                DELETE FROM recruitment_role_candidates
                WHERE role_id = %s AND candidate_id = %s
            """, (role_id, candidate_id))
            
            conn.commit()
            from backend.api.routes.candidates import invalidate_candidate_count_caches

            invalidate_candidate_count_caches()
            return {"message": "Candidate removed from role"}
    finally:
        return_db_connection(conn)

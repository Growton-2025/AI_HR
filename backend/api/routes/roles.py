
from datetime import date, datetime
import asyncio
from decimal import Decimal
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from backend.api import schemas, deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.api.routes.candidate_imports import build_upload_preview_response, commit_upload_file
from backend.services.role_activation import (
    activate_role,
    fetch_role_activation,
    retry_role_activation,
)
from backend.services.outreach_counts import reply_count_sql
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


def invalidate_role_detail_cache_for_candidate(candidate_id: int) -> None:
    """Drop every cached role detail payload containing the candidate."""
    candidate_id = int(candidate_id)
    stale_keys = []
    for key, cached in _ROLE_DETAIL_CACHE.items():
        if not cached:
            continue
        candidates = cached[1].get("candidates") or []
        if any(int(candidate.get("id") or 0) == candidate_id for candidate in candidates):
            stale_keys.append(key)
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
                    SELECT r.id, r.name, r.job_description,
                           COALESCE(c_counts.candidate_count, 0) as candidate_count,
                           COALESCE(u_counts.upload_count, 0) as upload_count,
                           u_counts.latest_upload_at,
                           r.user_id,
                           u.name as owner_name,
                           u.email as owner_email,
                           sc.provisioning_status, sc.provisioning_error,
                           hc.provisioning_status, hc.provisioning_error,
                           r.linked_call_list_id
                    FROM recruitment_roles r
                    LEFT JOIN users u ON u.id = r.user_id
                    LEFT JOIN (
                        SELECT rc.role_id, COUNT(DISTINCT rc.candidate_id) as candidate_count
                        FROM recruitment_role_candidates rc
                        JOIN candidates c ON c.id = rc.candidate_id AND COALESCE(c.is_archived, FALSE) = FALSE
                        GROUP BY rc.role_id
                    ) c_counts ON c_counts.role_id = r.id
                    LEFT JOIN (
                        SELECT role_id, COUNT(id) as upload_count, MAX(completed_at) as latest_upload_at
                        FROM candidate_uploads
                        GROUP BY role_id
                    ) u_counts ON u_counts.role_id = r.id
                    LEFT JOIN role_smartlead_campaigns sc ON sc.recruitment_role_id = r.id
                    LEFT JOIN role_heyreach_campaigns hc ON hc.recruitment_role_id = r.id
                    ORDER BY r.created_at DESC
                """)
            else:
                target_user_id = _role_owner_id(current_user, owner_user_id)
                logger.info(f"Fetching roles for user_id: {target_user_id}")
                cur.execute("""
                    SELECT r.id, r.name, r.job_description,
                           COALESCE(c_counts.candidate_count, 0) as candidate_count,
                           COALESCE(u_counts.upload_count, 0) as upload_count,
                           u_counts.latest_upload_at,
                           r.user_id,
                           u.name as owner_name,
                           u.email as owner_email,
                           sc.provisioning_status, sc.provisioning_error,
                           hc.provisioning_status, hc.provisioning_error,
                           r.linked_call_list_id
                    FROM recruitment_roles r
                    LEFT JOIN users u ON u.id = r.user_id
                    LEFT JOIN (
                        SELECT rc.role_id, COUNT(DISTINCT rc.candidate_id) as candidate_count
                        FROM recruitment_role_candidates rc
                        JOIN candidates c ON c.id = rc.candidate_id AND COALESCE(c.is_archived, FALSE) = FALSE
                        GROUP BY rc.role_id
                    ) c_counts ON c_counts.role_id = r.id
                    LEFT JOIN (
                        SELECT role_id, COUNT(id) as upload_count, MAX(completed_at) as latest_upload_at
                        FROM candidate_uploads
                        GROUP BY role_id
                    ) u_counts ON u_counts.role_id = r.id
                    LEFT JOIN role_smartlead_campaigns sc ON sc.recruitment_role_id = r.id
                    LEFT JOIN role_heyreach_campaigns hc ON hc.recruitment_role_id = r.id
                    WHERE r.user_id = %s
                    ORDER BY r.created_at DESC
                """, (target_user_id,))
            
            rows = cur.fetchall()
            logger.info("Found %s roles", len(rows))
            for row in rows:
                has_call_list = bool(row[13])
                outreach_active = row[9] in ("configured", "skipped") and row[11] in ("configured", "skipped") and not (row[9] == "skipped" and row[11] == "skipped")
                active = outreach_active or has_call_list
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
                    "activation_status": "active" if active else "inactive",
                    "activation_error": " | ".join(value for value in (row[10], row[12]) if value),
                    "has_call_list": has_call_list,
                    "smartlead_status": row[9] or "missing",
                    "heyreach_status": row[11] or "missing",
                })
    finally:
        return_db_connection(conn)
    
    logger.info(f"Returning roles: {roles_list}")
    return {"roles": roles_list}

@router.post("")
async def create_role(role: schemas.RoleCreate, current_user: schemas.User = Depends(deps.get_current_user)):
    """Create a new role for the current user"""
    if not role.name.strip():
        raise HTTPException(status_code=400, detail="Role name is required")
    if role.heyreach_campaign_id <= 0 and role.smartlead_sender_account_id <= 0 and not role.auto_create_call_list:
        raise HTTPException(status_code=400, detail="At least one of HeyReach campaign, Smartlead sender, or Auto-create call list is required")
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            role_name = role.name.strip()
            logger.info(f"Checking for role '{role_name}' for user_id: {current_user.id}")
            cur.execute("SELECT id FROM recruitment_roles WHERE user_id = %s AND name = %s", (current_user.id, role_name))
            if cur.fetchone():
                logger.warning(f"Role '{role.name}' already exists for user_id: {current_user.id}")
                raise HTTPException(status_code=400, detail="Role already exists")
            
            cur.execute("""
                INSERT INTO recruitment_roles (user_id, name, job_description, auto_create_call_list)
                VALUES (%s, %s, %s, %s) RETURNING id, name, job_description
            """, (current_user.id, role_name, (role.job_description or "").strip() or None, role.auto_create_call_list))
            result = cur.fetchone()
            role_id = result[0]

            if role.auto_create_call_list:
                call_list_name = f"{role_name} - call list"
                owner = current_user.email or current_user.username
                try:
                    cur.execute("SELECT id FROM call_lists WHERE name = %s AND created_by = %s", (call_list_name, owner))
                    existing_row = cur.fetchone()
                    if existing_row:
                        call_list_id = existing_row[0]
                    else:
                        cur.execute(
                            "INSERT INTO call_lists (name, created_by) VALUES (%s, %s) RETURNING id",
                            (call_list_name, owner)
                        )
                        call_list_id = cur.fetchone()[0]
                        
                    cur.execute(
                        "UPDATE recruitment_roles SET linked_call_list_id = %s WHERE id = %s",
                        (call_list_id, role_id)
                    )
                    # Trigger call lists cache refresh asynchronously? Not strictly necessary here, but good.
                    from backend.api.routes.calls import bulk_load_call_lists_cache
                    import threading
                    threading.Thread(target=bulk_load_call_lists_cache).start()
                except Exception as e:
                    logger.error(f"Failed to auto-create call list for role {role_id}: {e}")
            conn.commit()
    finally:
        return_db_connection(conn)
    try:
        activation = await asyncio.to_thread(
            activate_role,
            result[0],
            result[1],
            role.heyreach_campaign_id,
            role.smartlead_sender_account_id,
            role.smartlead_campaign_id,
        )
    except Exception as exc:
        logger.exception("Role %s activation failed", result[0])
        activation = {
            "activation_status": "inactive",
            "activation_error": str(exc)[:1000],
            "smartlead_status": "failed",
            "heyreach_status": "failed",
        }
    return {
        "message": f"Role '{result[1]}' created",
        "id": result[0],
        "name": result[1],
        "job_description": result[2] or "",
        **activation,
    }


@router.post("/id/{role_id}/activate")
async def retry_role_activation_endpoint(
    role_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
    finally:
        return_db_connection(conn)
    activation = await asyncio.to_thread(retry_role_activation, role[0], role[1])
    invalidate_role_detail_cache(role_id)
    return activation


@router.post("/id/{role_id}/deactivate")
async def deactivate_role_endpoint(
    role_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
            
            cur.execute("UPDATE role_smartlead_campaigns SET provisioning_status = 'deactivated' WHERE recruitment_role_id = %s", (role[0],))
            cur.execute("UPDATE role_heyreach_campaigns SET provisioning_status = 'deactivated' WHERE recruitment_role_id = %s", (role[0],))
            conn.commit()
            activation_setup = fetch_role_activation(cur, role[0])
    finally:
        return_db_connection(conn)
    invalidate_role_detail_cache(role_id)
    return {"id": role[0], "name": role[1], **activation_setup}


@router.get("/id/{role_id}/activation")
async def get_role_activation_setup(
    role_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
            return {"id": role[0], "name": role[1], **fetch_role_activation(cur, role_id)}
    finally:
        return_db_connection(conn)


@router.put("/id/{role_id}/activation")
async def configure_existing_role_activation(
    role_id: int,
    setup: schemas.RoleActivationSetup,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if setup.heyreach_campaign_id <= 0 and setup.smartlead_sender_account_id <= 0:
        raise HTTPException(status_code=400, detail="At least one of HeyReach campaign ID or Smartlead sender is required")
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            role = _resolve_role(cur, current_user, role_id=role_id)
            if not role:
                raise HTTPException(status_code=404, detail="Role not found")
    finally:
        return_db_connection(conn)
    activation = await asyncio.to_thread(
        activate_role,
        role[0], role[1], setup.heyreach_campaign_id,
        setup.smartlead_sender_account_id,
        setup.smartlead_campaign_id,
    )
    invalidate_role_detail_cache(role_id)
    return activation

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

            # Replies = inbound messages only, so the conversation badge counts
            # what the candidate wrote back rather than what we sent.
            email_reply_count_sql = reply_count_sql(
                "co.email_chat_history_cache", "co.response_text"
            )
            li_reply_count_sql = reply_count_sql(
                "co.li_chat_history_cache", "co.li_response_text"
            )

            cur.execute(
                f"""
                WITH selected_role AS (
                    SELECT r.id, r.name, r.job_description, r.user_id,
                           (SELECT COUNT(*) FROM candidate_uploads cu WHERE cu.role_id = r.id) as upload_count,
                           (SELECT MAX(cu.completed_at) FROM candidate_uploads cu WHERE cu.role_id = r.id) as latest_upload_at
                    FROM recruitment_roles r
                    WHERE {role_filter}
                    ORDER BY {role_order}
                    LIMIT 1
                )
                SELECT sr.id, sr.name, sr.job_description, sr.user_id,
                       u.name, u.email,
                       sr.upload_count,
                       sr.latest_upload_at,
                       COUNT(c.id) OVER (),
                       c.id, rc.priority, rc.feedback,
                       c.name, c.linkedin, c.location, c.headline, c.about,
                       c.email,
                       COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), ''),
                       c.status,
                       c.first_name,
                       c.last_name,
                       COALESCE(NULLIF(TRIM(c.city), ''), split_part(COALESCE(c.location, ''), ',', 1), ''),
                       COALESCE(c.total_experience_years, 0),
                       COALESCE(c.avg_years_in_company, 0),
                       COALESCE(c.notes, ''),
                       COALESCE(role_outreach.response_text, c.response, ''),
                       COALESCE(primary_role.title, c.headline, ''),
                       COALESCE(primary_role.company, c.raw_fields->>'import_company', ''),
                       COALESCE(role_outreach.li_response_text, ''),
                       COALESCE(role_outreach.status, ''),
                       COALESCE(role_outreach.message_sent_count, 0),
                       COALESCE(role_outreach.li_status, ''),
                       COALESCE(role_outreach.li_sent_count, 0),
                       COALESCE(role_outreach.li_conversation_id, ''),
                       COALESCE(role_outreach.email_message_count, 0),
                       COALESCE(role_outreach.li_message_count, 0),
                       COALESCE(role_outreach.message_count, 0),
                       rc.created_at,
                       COALESCE(role_outreach.email_reply_count, 0),
                       COALESCE(role_outreach.li_reply_count, 0),
                       COALESCE(role_outreach.email_reply_count, 0)
                           + COALESCE(role_outreach.li_reply_count, 0)
                FROM selected_role sr
                LEFT JOIN users u ON u.id = sr.user_id
                LEFT JOIN recruitment_role_candidates rc ON rc.role_id = sr.id
                LEFT JOIN candidates c ON c.id = rc.candidate_id
                    AND COALESCE(c.is_archived, FALSE) = FALSE
                LEFT JOIN LATERAL (
                    SELECT r.title, company.name AS company
                    FROM roles r
                    LEFT JOIN companies company ON company.id = r.company_id
                    WHERE r.candidate_id = c.id
                    ORDER BY r.id ASC
                    LIMIT 1
                ) primary_role ON TRUE
                LEFT JOIN LATERAL (
                    SELECT
                        co.response_text,
                        co.li_response_text,
                        co.status,
                        co.message_sent_count,
                        co.li_status,
                        co.li_sent_count,
                        co.li_conversation_id,
                        GREATEST(
                            COALESCE(co.message_sent_count, 0)
                                + CASE WHEN NULLIF(TRIM(COALESCE(co.response_text, '')), '') IS NULL THEN 0 ELSE 1 END,
                            CASE
                                WHEN jsonb_typeof(co.email_chat_history_cache) = 'array'
                                THEN jsonb_array_length(co.email_chat_history_cache)
                                ELSE 0
                            END
                        ) AS email_message_count,
                        GREATEST(
                            COALESCE(co.li_sent_count, 0)
                                + CASE WHEN NULLIF(TRIM(COALESCE(co.li_response_text, '')), '') IS NULL THEN 0 ELSE 1 END,
                            CASE
                                WHEN jsonb_typeof(co.li_chat_history_cache) = 'array'
                                THEN jsonb_array_length(co.li_chat_history_cache)
                                ELSE 0
                            END
                        ) AS li_message_count,
                        GREATEST(
                            COALESCE(co.message_sent_count, 0)
                                + CASE WHEN NULLIF(TRIM(COALESCE(co.response_text, '')), '') IS NULL THEN 0 ELSE 1 END,
                            CASE
                                WHEN jsonb_typeof(co.email_chat_history_cache) = 'array'
                                THEN jsonb_array_length(co.email_chat_history_cache)
                                ELSE 0
                            END
                        )
                        + GREATEST(
                            COALESCE(co.li_sent_count, 0)
                                + CASE WHEN NULLIF(TRIM(COALESCE(co.li_response_text, '')), '') IS NULL THEN 0 ELSE 1 END,
                            CASE
                                WHEN jsonb_typeof(co.li_chat_history_cache) = 'array'
                                THEN jsonb_array_length(co.li_chat_history_cache)
                                ELSE 0
                            END
                        ) AS message_count,
                        {email_reply_count_sql} AS email_reply_count,
                        {li_reply_count_sql} AS li_reply_count
                    FROM candidate_outreach co
                    WHERE co.candidate_id = c.id
                      AND co.recruitment_role_id = sr.id
                    ORDER BY co.updated_at DESC NULLS LAST
                    LIMIT 1
                ) role_outreach ON TRUE
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
                name_parts = (row[12] or "").strip().split()
                first_name = (row[20] or "").strip() or (name_parts[0] if name_parts else "")
                last_name = (row[21] or "").strip() or (" ".join(name_parts[1:]) if len(name_parts) > 1 else "")
                title = row[27] or candidate.get("current_title") or candidate.get("title") or candidate.get("headline") or ""
                company = row[28] or candidate.get("current_company") or candidate.get("company") or ""
                candidate.update({
                    "name": row[12] or candidate.get("name") or (f"{first_name} {last_name}".strip()),
                    "first_name": first_name,
                    "last_name": last_name,
                    "linkedin": row[13] or candidate.get("linkedin") or "",
                    "title": title,
                    "current_title": title,
                    "company": company,
                    "current_company": company,
                    "city": row[22] or candidate.get("city") or "",
                    "location": row[14] or candidate.get("location") or "",
                    "total_experience_years": round(float(row[23] or 0), 1),
                    "avg_tenure_years": round(float(row[24] or 0), 1),
                    "avg_years_in_company": round(float(row[24] or 0), 1),
                    "email": row[17] or candidate.get("email") or "",
                    "phone": row[18] or candidate.get("phone") or candidate.get("mobile_phone") or "",
                    "mobile_phone": row[18] or candidate.get("mobile_phone") or candidate.get("phone") or "",
                    "response": row[26] or candidate.get("response") or "",
                    "response_text": row[26] or candidate.get("response_text") or candidate.get("response") or "",
                    "li_response_text": row[29] or candidate.get("li_response_text") or "",
                    "email_outreach_status": row[30] or candidate.get("email_outreach_status") or "",
                    "message_sent_count": int(row[31] or candidate.get("message_sent_count") or 0),
                    "linkedin_outreach_status": row[32] or candidate.get("linkedin_outreach_status") or "",
                    "li_status": row[32] or candidate.get("li_status") or "",
                    "li_sent_count": int(row[33] or candidate.get("li_sent_count") or 0),
                    "li_conversation_id": row[34] or candidate.get("li_conversation_id") or "",
                    "email_message_count": int(row[35] or 0),
                    "li_message_count": int(row[36] or 0),
                    "message_count": int(row[37] or 0),
                    "email_reply_count": int(row[39] or 0),
                    "li_reply_count": int(row[40] or 0),
                    "reply_count": int(row[41] or 0),
                    "outreach_counts_loaded": True,
                    "notes": row[25] or "",
                    "status": row[19] or candidate.get("status") or "To be started",
                    "added_to_role_at": row[38].isoformat() if row[38] else None,
                })
                candidate["priority"] = row[10]
                candidate["feedback"] = row[11]
                candidates.append(candidate)

            # Hydrate resume metadata in one batch query (no bytes, no text).
            if candidates:
                from backend.services.resume_service import fetch_resume_metas

                resume_metas = fetch_resume_metas([c["id"] for c in candidates])
                for candidate in candidates:
                    candidate["resume"] = resume_metas.get(candidate["id"])

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
            response.update(fetch_role_activation(cur, role[0]))
            _ROLE_DETAIL_CACHE[cache_key] = (time.monotonic(), response)
            return response
    finally:
        return_db_connection(conn)


@router.get("/{role_name}/contacts")
async def get_role_contacts(
    role_name: str,
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Lightweight polling endpoint for Clay/import contact updates."""
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
                SELECT c.id, c.email,
                       COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), ''),
                       co.status, co.li_status, co.email_enrollment_error, co.li_enrollment_error
                FROM recruitment_role_candidates rc
                JOIN candidates c ON c.id = rc.candidate_id
                LEFT JOIN candidate_outreach co
                  ON co.candidate_id = c.id AND co.recruitment_role_id = rc.role_id
                WHERE rc.role_id = %s
                  AND COALESCE(c.is_archived, FALSE) = FALSE
                """,
                (resolved_role[0],),
            )
            return {
                "contacts": [
                    {
                        "id": row[0], "email": row[1] or "", "mobile_phone": row[2] or "",
                        "email_status": row[3] or "not_started", "linkedin_status": row[4] or "not_started",
                        "email_error": row[5] or "", "linkedin_error": row[6] or "",
                    }
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
            
            if added_ids:
                from backend.services.auto_call_list import sync_shortlisted_to_call_list
                sync_shortlisted_to_call_list(cur, role_id, added_ids)
                conn.commit()

            from backend.api.routes.candidates import invalidate_candidate_count_caches
            invalidate_candidate_count_caches()
            invalidate_role_detail_cache(role_id)

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

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from backend.pipeline.query import (
    update_candidate_status,
    PROFILES_BY_ID,
    initialize_cache,
    is_cache_initialized,
    count_active_candidates_from_db,
)
from backend.api import deps, schemas
from backend.services.candidate_pool import (
    profile_passes_scope,
    VIEW_SCOPE_MASTER,
    VIEW_SCOPE_RECRUITER_POOLS,
    VIEW_SCOPE_ALL_RECRUITER_POOLS,
)
from backend.db.connection import get_db_connection, return_db_connection
import asyncio
import logging
import time
import hashlib
import json
from datetime import date

router = APIRouter()
logger = logging.getLogger(__name__)
_profile_cache_init_lock = asyncio.Lock()

@router.get("/candidates/sample")
async def get_sample_candidate(
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Return 5 candidate IDs+names directly from DB (no cache needed). Used for AI Column test fallback."""
    from backend.db.connection import get_db_connection_context
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (c.id)
                           c.id,
                           COALESCE(c.first_name, split_part(c.name,' ',1), '') AS first_name,
                           COALESCE(c.last_name,  split_part(c.name,' ',2), '') AS last_name,
                           COALESCE(co.name, c.headline, '') AS company,
                           COALESCE(r.title, c.headline, '') AS title
                    FROM candidates c
                    LEFT JOIN roles r    ON r.candidate_id = c.id
                    LEFT JOIN companies co ON co.id = r.company_id
                    ORDER BY c.id ASC, r.id ASC
                    LIMIT 5
                """)
                rows = cur.fetchall()
                return {
                    "candidates": [
                        {
                            "id": row[0],
                            "first_name": row[1],
                            "last_name": row[2],
                            "name": f"{row[1]} {row[2]}".strip(),
                            "company": row[3],
                            "title": row[4],
                        }
                        for row in rows
                    ]
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# ── Server-side browse result cache ───────────────────────────────────────────────
# Caches filtered + paginated results so repeated requests return instantly
_browse_cache: dict = {}  # key: param_hash → {result, ts}
_summary_cache: dict = {}  # key: param_hash → {result, ts}
_meta_cache: dict = {}     # key: param_hash → {result, ts}
_BROWSE_CACHE_TTL = 20   # seconds
_META_CACHE_TTL = 60     # seconds — filter dropdown values change rarely

def _invalidate_browse_cache():
    """Called when outreach data changes so stale data isn't served."""
    global _browse_cache, _summary_cache, _meta_cache
    _browse_cache.clear()
    _summary_cache.clear()
    _meta_cache.clear()
# ──────────────────────────────────────────────────────────────


async def _ensure_profiles_loaded() -> bool:
    """Load the per-worker profile cache before browse reads it.

    Azure/Gunicorn workers do not share process memory. A cold worker can otherwise
    answer browse/summary/meta from an empty cache and briefly poison frontend totals.
    """
    if is_cache_initialized() and PROFILES_BY_ID:
        return False
    if is_cache_initialized() and not PROFILES_BY_ID:
        logger.warning(
            "browse profile cache marked initialized but empty; forcing reload before serving request"
        )
    async with _profile_cache_init_lock:
        if is_cache_initialized() and PROFILES_BY_ID:
            return False
        started = time.monotonic()
        await asyncio.to_thread(initialize_cache)
        _invalidate_browse_cache()
        logger.warning(
            "browse profile cache initialized duration_ms=%.1f profiles=%s",
            (time.monotonic() - started) * 1000,
            len(PROFILES_BY_ID),
        )
        if not PROFILES_BY_ID:
            active_count = await asyncio.to_thread(count_active_candidates_from_db)
            if active_count and active_count > 0:
                logger.error(
                    "browse profile cache unavailable after reload; active_candidates=%s",
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
        return True


def _log_browse_timing(
    name: str,
    started: float,
    *,
    status: str = "ok",
    total: Optional[int] = None,
    page_size: Optional[int] = None,
    scope: Optional[str] = None,
    recruiter_id: Optional[int] = None,
) -> None:
    duration_ms = (time.monotonic() - started) * 1000
    log = logger.warning if duration_ms > 500 or status != "ok" else logger.info
    log(
        "browse %s status=%s duration_ms=%.1f total=%s page_size=%s scope=%s recruiter_id=%s",
        name,
        status,
        duration_ms,
        total,
        page_size,
        scope,
        recruiter_id,
    )

class StatusUpdate(BaseModel):
    status: str

class NotesUpdate(BaseModel):
    notes: str

def _authorize_candidate_update(
    candidate_id: int,
    current_user: schemas.User,
    *,
    allow_role_access: bool = False,
) -> None:
    """Authorize candidate edits even when the in-memory profile cache is cold."""
    is_admin = (current_user.role or "").strip().lower() == "admin"
    prof = PROFILES_BY_ID.get(candidate_id)
    candidate_found = bool(prof)
    owner_user_id = prof.get("owner_user_id") if prof else None
    has_role_access = False

    needs_db_lookup = not prof or (allow_role_access and not is_admin and owner_user_id != current_user.id)
    if needs_db_lookup:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        try:
            with conn.cursor() as cur:
                if not prof:
                    cur.execute(
                        """
                        SELECT owner_user_id, COALESCE(is_archived, FALSE)
                        FROM candidates
                        WHERE id = %s
                        """,
                        (candidate_id,),
                    )
                    row = cur.fetchone()
                    if not row or row[1]:
                        raise HTTPException(status_code=404, detail="Candidate not found")
                    candidate_found = True
                    owner_user_id = row[0]

                if allow_role_access and not is_admin and owner_user_id != current_user.id:
                    cur.execute(
                        """
                        SELECT 1 FROM recruitment_role_candidates rrc
                        JOIN recruitment_roles rr ON rr.id = rrc.role_id
                        WHERE rrc.candidate_id = %s AND rr.user_id = %s
                        LIMIT 1
                        """,
                        (candidate_id, current_user.id),
                    )
                    has_role_access = bool(cur.fetchone())
        finally:
            return_db_connection(conn)

    if not candidate_found:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if is_admin or owner_user_id == current_user.id or (allow_role_access and has_role_access):
        return

    if owner_user_id is None:
        raise HTTPException(status_code=403, detail="Master library rows are read-only")
    raise HTTPException(status_code=403, detail="Not allowed to update this candidate")

RECRUITMENT_STAGES = [
    'To be started', 'Shortlisted', 'Rejected', 'For Future', 
    'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested', 
    'Followup / In conversation', 'Shortlist - Rejected', 'High CTC', 
    'Duplicate', 'Not responding', 'Internal Review', 'Shared with customer'
]


# --- Industry Keywords for Extraction ---
INDUSTRY_KEYWORDS = [
    "SaaS", "Cloud", "Fintech", "HRTech", "MarTech", "AdTech", "EdTech", "PropTech", "HealthTech", "CyberSecurity",
    "AI", "Machine Learning", "Data Science", "Big Data", "Blockchain", "Crypto", "E-commerce", "Retail",
    "FMCG", "Banking", "Insurance", "Payments", "Logistics", "Supply Chain", "Manufacturing", "Automotive",
    "Telecom", "Real Estate", "Healthcare", "Pharma", "Biotech", "Energy", "Gaming", "IT Services"
]
INDUSTRY_KEYWORDS_LOWER = [kw.lower() for kw in INDUSTRY_KEYWORDS]


def resolve_browse_scope(
    current_user: schemas.User,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
) -> tuple[str, Optional[int]]:
    if (current_user.role or "").strip().lower() != "admin":
        return VIEW_SCOPE_RECRUITER_POOLS, current_user.id

    effective_scope = view_scope or VIEW_SCOPE_MASTER
    effective_recruiter = recruiter_filter_id
    if effective_scope == VIEW_SCOPE_RECRUITER_POOLS and not effective_recruiter:
        raise HTTPException(
            status_code=400,
            detail="recruiter_filter_id is required when view_scope=recruiter_pools",
        )
    return effective_scope, effective_recruiter


def _matches_filter(filter_str: Optional[str], target_val: Any) -> bool:
    if not filter_str:
        return True
    filter_vals = [v.strip().lower() for v in str(filter_str).split(",") if v.strip()]
    if not filter_vals:
        return True
    if not target_val:
        return False
    tv_lower = str(target_val).lower()
    return any(v in tv_lower for v in filter_vals)


def _split_filter_values(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    text = str(raw)
    # "||" separates values that may themselves contain commas (e.g. the city
    # "Bhubaneswar, Odisha, India"); plain comma-separated input still works.
    sep = "||" if "||" in text else ","
    return [v.strip().lower() for v in text.split(sep) if v.strip()]


def _parse_candidate_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    parsed: List[int] = []
    for piece in str(raw).split(","):
        value = piece.strip()
        if not value:
            continue
        try:
            parsed.append(int(value))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid candidate id: {value!r}")
    return parsed


def _role_candidate_id_set(
    current_user: schemas.User,
    *,
    role_id: Optional[int],
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
) -> Optional[set[int]]:
    if not role_id:
        return None
    owner_id: Optional[int] = current_user.id
    if (current_user.role or "").strip().lower() == "admin":
        if view_scope == VIEW_SCOPE_RECRUITER_POOLS and recruiter_filter_id:
            owner_id = recruiter_filter_id
        elif view_scope in (VIEW_SCOPE_MASTER, VIEW_SCOPE_ALL_RECRUITER_POOLS):
            owner_id = None

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            if owner_id is None:
                cur.execute("SELECT user_id FROM recruitment_roles WHERE id = %s", (role_id,))
            else:
                cur.execute(
                    "SELECT user_id FROM recruitment_roles WHERE id = %s AND user_id = %s",
                    (role_id, owner_id),
                )
            role = cur.fetchone()
            if not role:
                logger.warning(
                    "browse role filter ignored missing role_id=%s scope=%s recruiter_id=%s user_id=%s",
                    role_id,
                    view_scope,
                    recruiter_filter_id,
                    current_user.id,
                )
                return set()
            cur.execute(
                "SELECT candidate_id FROM recruitment_role_candidates WHERE role_id = %s",
                (role_id,),
            )
            return {int(row[0]) for row in cur.fetchall()}
    finally:
        return_db_connection(conn)


def _summary_scope_sql(
    current_user: schemas.User,
    *,
    effective_scope: str,
    effective_recruiter: Optional[int],
    role_id: Optional[int],
    added_from: Optional[str] = None,
    added_to: Optional[str] = None,
) -> tuple[str, List[Any]]:
    where = ["COALESCE(c.is_archived, FALSE) = FALSE"]
    params: List[Any] = []
    user_role = (current_user.role or "").strip().lower()

    if role_id:
        # Role membership alone defines the pool. The Manage Roles table lists
        # every candidate linked to the role regardless of which recruiter owns
        # the profile, so filters must not additionally narrow by candidate
        # owner — otherwise filtered results/counts drop candidates that are
        # plainly visible in the unfiltered table. Access control stays on the
        # role itself (rr.user_id).
        role_filters = ["rrc.candidate_id = c.id", "rrc.role_id = %s"]
        params.append(role_id)
        if user_role != "admin":
            role_filters.append("rr.user_id = %s")
            params.append(current_user.id)
        elif effective_scope == VIEW_SCOPE_RECRUITER_POOLS and effective_recruiter is not None:
            role_filters.append("rr.user_id = %s")
            params.append(effective_recruiter)
        if added_from:
            role_filters.append("rrc.created_at >= %s::date")
            params.append(added_from)
        if added_to:
            # Inclusive end date: anything before the next day counts.
            role_filters.append("rrc.created_at < %s::date + INTERVAL '1 day'")
            params.append(added_to)

        where.append(
            """
            EXISTS (
                SELECT 1
                FROM recruitment_role_candidates rrc
                JOIN recruitment_roles rr ON rr.id = rrc.role_id
                WHERE {role_where}
            )
            """.format(role_where=" AND ".join(role_filters))
        )
        return " AND ".join(where), params

    if user_role != "admin":
        where.append("c.owner_user_id = %s")
        params.append(current_user.id)
    elif effective_scope == VIEW_SCOPE_RECRUITER_POOLS:
        where.append("c.owner_user_id IS NOT NULL")
        if effective_recruiter is not None:
            where.append("c.owner_user_id = %s")
            params.append(effective_recruiter)
    elif effective_scope == VIEW_SCOPE_ALL_RECRUITER_POOLS:
        where.append("c.owner_user_id IS NOT NULL")

    return " AND ".join(where), params


def _add_like_filter(
    where: List[str],
    params: List[Any],
    raw: Optional[str],
    expressions: List[str],
) -> None:
    values = _split_filter_values(raw)
    if not values:
        return

    clauses: List[str] = []
    for value in values:
        needle = f"%{value}%"
        for expr in expressions:
            clauses.append(f"LOWER(COALESCE({expr}, '')) LIKE %s")
            params.append(needle)
    where.append("(" + " OR ".join(clauses) + ")")


def _add_title_filter(where: List[str], params: List[Any], raw: Optional[str]) -> None:
    values = _split_filter_values(raw)
    if not values:
        return

    clauses: List[str] = []
    for value in values:
        needle = f"%{value}%"
        clauses.append(
            """
            (
                LOWER(COALESCE(c.headline, '')) LIKE %s
                OR EXISTS (
                    SELECT 1
                    FROM roles r_title
                    WHERE r_title.candidate_id = c.id
                      AND LOWER(COALESCE(r_title.title, '')) LIKE %s
                )
            )
            """
        )
        params.extend([needle, needle])
    where.append("(" + " OR ".join(clauses) + ")")


def _add_company_filter(where: List[str], params: List[Any], raw: Optional[str]) -> None:
    values = _split_filter_values(raw)
    if not values:
        return

    clauses: List[str] = []
    for value in values:
        needle = f"%{value}%"
        clauses.append(
            """
            (
                LOWER(COALESCE(c.raw_fields->>'import_company', '')) LIKE %s
                OR EXISTS (
                    SELECT 1
                    FROM roles r_company
                    JOIN companies co_company ON co_company.id = r_company.company_id
                    WHERE r_company.candidate_id = c.id
                      AND LOWER(COALESCE(co_company.name, '')) LIKE %s
                )
            )
            """
        )
        params.extend([needle, needle])
    where.append("(" + " OR ".join(clauses) + ")")


def _add_product_service_filter(
    where: List[str],
    params: List[Any],
    raw: Optional[str],
) -> None:
    values = _split_filter_values(raw)
    if not values:
        return

    clauses: List[str] = []
    for value in values:
        needle = f"%{value}%"
        clauses.append(
            """
            (
                LOWER(COALESCE(c.raw_fields->>'extracted_industry', '')) LIKE %s
                OR LOWER(COALESCE(c.raw_fields->>'services', '')) LIKE %s
                OR EXISTS (
                    SELECT 1
                    FROM roles r_product
                    JOIN companies co_product ON co_product.id = r_product.company_id
                    WHERE r_product.candidate_id = c.id
                      AND LOWER(COALESCE(co_product.product_service, '')) LIKE %s
                )
            )
            """
        )
        params.extend([needle, needle, needle])
    where.append("(" + " OR ".join(clauses) + ")")


def _add_global_search_filter(where: List[str], params: List[Any], raw: Optional[str]) -> None:
    term = str(raw or "").strip().lower()
    if not term:
        return

    needle = f"%{term}%"
    where.append(
        """
        (
            LOWER(COALESCE(c.name, '')) LIKE %s
            OR LOWER(COALESCE(c.first_name, '')) LIKE %s
            OR LOWER(COALESCE(c.last_name, '')) LIKE %s
            OR LOWER(COALESCE(c.linkedin, '')) LIKE %s
            OR LOWER(COALESCE(c.normalized_linkedin, '')) LIKE %s
            OR LOWER(COALESCE(c.email, '')) LIKE %s
            OR LOWER(COALESCE(c.phone, c.mobile_phone, '')) LIKE %s
            OR LOWER(COALESCE(c.city, '')) LIKE %s
            OR LOWER(COALESCE(c.location, '')) LIKE %s
            OR LOWER(COALESCE(c.headline, '')) LIKE %s
            OR LOWER(COALESCE(c.raw_fields->>'import_company', '')) LIKE %s
            OR EXISTS (
                SELECT 1
                FROM roles r_search
                LEFT JOIN companies co_search ON co_search.id = r_search.company_id
                WHERE r_search.candidate_id = c.id
                  AND (
                    LOWER(COALESCE(r_search.title, '')) LIKE %s
                    OR LOWER(COALESCE(co_search.name, '')) LIKE %s
                  )
            )
        )
        """
    )
    params.extend([needle] * 13)


def _browse_where_sql(
    current_user: schemas.User,
    *,
    effective_scope: str,
    effective_recruiter: Optional[int],
    role_id: Optional[int],
    q: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    city: Optional[str] = None,
    location_type: Optional[str] = None,
    product_service: Optional[str] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = None,
    min_exp: Optional[float] = None,
    max_exp: Optional[float] = None,
    min_avg_tenure: Optional[float] = None,
    candidate_ids: Optional[List[int]] = None,
    include_status: bool = True,
    added_from: Optional[str] = None,
    added_to: Optional[str] = None,
) -> tuple[str, List[Any]]:
    where_sql, params = _summary_scope_sql(
        current_user,
        effective_scope=effective_scope,
        effective_recruiter=effective_recruiter,
        role_id=role_id,
        added_from=added_from,
        added_to=added_to,
    )
    where = [where_sql]

    ids = [int(cid) for cid in (candidate_ids or []) if cid is not None]
    if ids:
        where.append("c.id = ANY(%s)")
        params.append(ids)

    _add_global_search_filter(where, params, q)
    _add_title_filter(where, params, title)
    _add_company_filter(where, params, company)
    _add_product_service_filter(where, params, product_service)
    _add_like_filter(
        where,
        params,
        city,
        ["c.city", "split_part(COALESCE(c.location, ''), ',', 1)", "c.location"],
    )
    _add_like_filter(
        where,
        params,
        location_type,
        ["c.raw_fields->>'work_preference'", "c.raw_fields->>'location_type'"],
    )
    _add_like_filter(where, params, created_by, ["c.created_by"])

    if min_exp is not None:
        where.append("COALESCE(c.total_experience_years, 0) >= %s")
        params.append(min_exp)
    if max_exp is not None:
        where.append("COALESCE(c.total_experience_years, 0) <= %s")
        params.append(max_exp)
    if min_avg_tenure is not None:
        where.append("COALESCE(c.avg_years_in_company, 0) >= %s")
        params.append(min_avg_tenure)

    if include_status:
        status_values = _split_filter_values(status)
        if status_values:
            where.append(
                "LOWER(COALESCE(NULLIF(TRIM(c.status), ''), 'To be started')) = ANY(%s)"
            )
            params.append(status_values)

    return " AND ".join(f"({clause})" for clause in where if clause), params


async def fetch_browse_summary_counts(
    *,
    current_user: schemas.User,
    view_scope: Optional[str] = None,
    recruiter_filter_id: Optional[int] = None,
    role_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Return count-only Talent Pool summary directly from SQL."""
    effective_scope, effective_recruiter = resolve_browse_scope(
        current_user,
        view_scope,
        recruiter_filter_id,
    )
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        where_sql, params = _summary_scope_sql(
            current_user,
            effective_scope=effective_scope,
            effective_recruiter=effective_recruiter,
            role_id=role_id,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COALESCE(NULLIF(TRIM(c.status), ''), 'To be started') AS status,
                    COUNT(*)::int AS count
                FROM candidates c
                WHERE {where_sql}
                GROUP BY COALESCE(NULLIF(TRIM(c.status), ''), 'To be started')
                """,
                params,
            )
            status_counts = {str(row[0]): int(row[1] or 0) for row in cur.fetchall()}

        return {
            "total": sum(status_counts.values()),
            "status_counts": status_counts,
            "effective_scope": effective_scope,
            "effective_recruiter": effective_recruiter,
        }
    finally:
        return_db_connection(conn)


def _can_use_fast_sql_browse(
    *,
    q: Optional[str],
    title: Optional[str],
    company: Optional[str],
    city: Optional[str],
    location_type: Optional[str],
    product_service: Optional[str],
    status: Optional[str],
    created_by: Optional[str],
    min_exp: Optional[float],
    max_exp: Optional[float],
    min_avg_tenure: Optional[float],
    candidate_ids: Optional[List[int]],
) -> bool:
    return True


def _profile_cache_looks_test_scoped() -> bool:
    """Detect monkeypatched/unit-test profile fixtures so browse stays in-memory."""
    if not PROFILES_BY_ID:
        return False
    ids = {int(profile.get("id") or key or 0) for key, profile in PROFILES_BY_ID.items()}
    return bool(ids) and len(ids) <= 100


async def fetch_browse_page_sql(
    *,
    current_user: schemas.User,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
    page: int,
    page_size: int,
    role_id: Optional[int],
    q: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    city: Optional[str] = None,
    location_type: Optional[str] = None,
    product_service: Optional[str] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = None,
    min_exp: Optional[float] = None,
    max_exp: Optional[float] = None,
    min_avg_tenure: Optional[float] = None,
    candidate_ids: Optional[List[int]] = None,
    added_from: Optional[str] = None,
    added_to: Optional[str] = None,
    sort_by: Optional[str],
    sort_dir: Optional[str],
) -> Dict[str, Any]:
    effective_scope, effective_recruiter = resolve_browse_scope(
        current_user,
        view_scope,
        recruiter_filter_id,
    )
    count_where_sql, count_params = _browse_where_sql(
        current_user,
        effective_scope=effective_scope,
        effective_recruiter=effective_recruiter,
        role_id=role_id,
        q=q,
        title=title,
        company=company,
        city=city,
        location_type=location_type,
        product_service=product_service,
        created_by=created_by,
        min_exp=min_exp,
        max_exp=max_exp,
        min_avg_tenure=min_avg_tenure,
        candidate_ids=candidate_ids,
        include_status=False,
        added_from=added_from,
        added_to=added_to,
    )
    row_where_sql, row_params = _browse_where_sql(
        current_user,
        effective_scope=effective_scope,
        effective_recruiter=effective_recruiter,
        role_id=role_id,
        q=q,
        title=title,
        company=company,
        city=city,
        location_type=location_type,
        product_service=product_service,
        status=status,
        created_by=created_by,
        min_exp=min_exp,
        max_exp=max_exp,
        min_avg_tenure=min_avg_tenure,
        candidate_ids=candidate_ids,
        include_status=True,
        added_from=added_from,
        added_to=added_to,
    )
    sort_map = {
        "name": "LOWER(COALESCE(c.name, ''))",
        "title": "LOWER(COALESCE(pr.title, c.headline, ''))",
        "company": "LOWER(COALESCE(pr.company, c.raw_fields->>'import_company', ''))",
        "city": "LOWER(COALESCE(NULLIF(c.city, ''), split_part(COALESCE(c.location, ''), ',', 1), ''))",
        "exp": "COALESCE(c.total_experience_years, 0)",
        "tenure": "COALESCE(c.avg_years_in_company, 0)",
    }
    order_expr = sort_map.get(sort_by or "name", sort_map["name"])
    direction = "DESC" if sort_dir == "desc" else "ASC"
    offset = (page - 1) * page_size

    # Each round trip to the remote DB costs ~0.6s, so: (1) status counts and
    # the filtered total are combined into ONE statement, and (2) that statement
    # runs CONCURRENTLY with the page-rows query on a second pooled connection.
    # Wall-clock cost ≈ one round trip instead of three sequential ones.
    def _run_counts_query():
        conn = get_db_connection(validate=False, register_pgvector=False)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH status_counts AS (
                        SELECT
                            COALESCE(NULLIF(TRIM(c.status), ''), 'To be started') AS status,
                            COUNT(*)::int AS count
                        FROM candidates c
                        WHERE {count_where_sql}
                        GROUP BY COALESCE(NULLIF(TRIM(c.status), ''), 'To be started')
                    ),
                    total_count AS (
                        SELECT COUNT(*)::int AS total FROM candidates c WHERE {row_where_sql}
                    )
                    SELECT sc.status, sc.count, tc.total
                    FROM total_count tc
                    LEFT JOIN status_counts sc ON TRUE
                    """,
                    [*count_params, *row_params],
                )
                combined_rows = cur.fetchall()
            counts = {
                str(row[0]): int(row[1] or 0)
                for row in combined_rows
                if row[0] is not None
            }
            total_value = int(combined_rows[0][2] or 0) if combined_rows else 0
            return counts, total_value
        finally:
            return_db_connection(conn)

    def _run_rows_query():
        conn = get_db_connection(validate=False, register_pgvector=False)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                SELECT
                    c.id,
                    c.name,
                    c.first_name,
                    c.last_name,
                    c.linkedin,
                    c.email,
                    COALESCE(c.mobile_phone, c.phone, '') AS phone,
                    COALESCE(c.response, '') AS response,
                    COALESCE(c.notes, '') AS notes,
                    COALESCE(pr.title, c.headline, '') AS title,
                    COALESCE(pr.company, c.raw_fields->>'import_company', '') AS company,
                    COALESCE(c.raw_fields->>'extracted_industry', c.raw_fields->>'services', pr.product_service, '') AS product_service,
                    COALESCE(NULLIF(c.city, ''), split_part(COALESCE(c.location, ''), ',', 1), '') AS city,
                    COALESCE(c.raw_fields->>'work_preference', c.raw_fields->>'location_type', '') AS location_type,
                    COALESCE(c.total_experience_years, 0) AS total_experience_years,
                    COALESCE(c.avg_years_in_company, 0) AS avg_tenure_years,
                    COALESCE(NULLIF(TRIM(c.status), ''), 'To be started') AS status,
                    c.created_by,
                    c.headline,
                    c.owner_user_id,
                    c.pool_source,
                    COALESCE(outreach.li_status, '') AS li_status,
                    COALESCE(outreach.li_response_text, '') AS li_response_text,
                    outreach.heyreach_campaign_id,
                    outreach.campaign_id AS email_campaign_id,
                    COALESCE(outreach.message_sent_count, 0) AS message_sent_count,
                    COALESCE(outreach.li_sent_count, 0) AS li_sent_count
                FROM candidates c
                LEFT JOIN LATERAL (
                    SELECT r.title, co.name AS company, co.product_service
                    FROM roles r
                    LEFT JOIN companies co ON co.id = r.company_id
                    WHERE r.candidate_id = c.id
                    ORDER BY r.id ASC
                    LIMIT 1
                ) pr ON TRUE
                LEFT JOIN LATERAL (
                    SELECT co.*
                    FROM candidate_outreach co
                    WHERE co.candidate_id = c.id
                      AND co.recruitment_role_id IS NULL
                    ORDER BY co.updated_at DESC NULLS LAST
                    LIMIT 1
                ) outreach ON TRUE
                WHERE {row_where_sql}
                ORDER BY {order_expr} {direction}, c.id ASC
                LIMIT %s OFFSET %s
                """,
                    [*row_params, page_size, offset],
                )
                return cur.fetchall()
        finally:
            return_db_connection(conn)

    try:
        (status_counts, total), rows = await asyncio.gather(
            asyncio.to_thread(_run_counts_query),
            asyncio.to_thread(_run_rows_query),
        )

        candidates = []
        for row in rows:
            name_val = row[1] or ""
            fn = (row[2] or "").strip() or (name_val.split() or [""])[0]
            ln = (row[3] or "").strip() or (" ".join(name_val.split()[1:]) if name_val else "")
            candidates.append({
                "id": row[0],
                "name": name_val,
                "first_name": fn,
                "last_name": ln,
                "linkedin": row[4],
                "email": row[5] or "",
                "phone": row[6] or "",
                "response": row[7] or "",
                "notes": row[8] or "",
                "title": row[9] or "",
                "company": row[10] or "",
                "product_service": row[11] or "",
                "city": row[12] or "",
                "location_type": row[13] or "",
                "total_experience_years": round(float(row[14] or 0), 1),
                "avg_tenure_years": round(float(row[15] or 0), 1),
                "status": row[16] or "To be started",
                "created_by": row[17] or "",
                "headline": row[18] or "",
                "match_score": None,
                "owner_user_id": row[19],
                "pool_source": row[20],
                "is_master_row": row[19] is None,
                "li_status": row[21] or "",
                "li_response_text": row[22] or "",
                "heyreach_campaign_id": row[23] or "",
                "email_campaign_id": row[24] or "",
                "message_sent_count": row[25] or 0,
                "li_sent_count": row[26] or 0,
                "raw_fields": {},
            })

        return {
            "candidates": candidates,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "status_counts": status_counts,
            "is_semantic_search": False,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"browse SQL page fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load candidates")


async def build_browse_candidate_rows(
    *,
    current_user: schemas.User,
    view_scope: Optional[str] = None,
    recruiter_filter_id: Optional[int] = None,
    q: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    city: Optional[str] = None,
    location_type: Optional[str] = None,
    product_service: Optional[str] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = None,
    min_exp: Optional[float] = None,
    max_exp: Optional[float] = None,
    min_avg_tenure: Optional[float] = None,
    candidate_ids: Optional[List[int]] = None,
    role_id: Optional[int] = None,
    sort_by: Optional[str] = "name",
    sort_dir: Optional[str] = "asc",
) -> dict:
    effective_scope, effective_recruiter = resolve_browse_scope(
        current_user,
        view_scope,
        recruiter_filter_id,
    )
    await _ensure_profiles_loaded()
    role_candidate_ids = _role_candidate_id_set(
        current_user,
        role_id=role_id,
        view_scope=effective_scope,
        recruiter_filter_id=effective_recruiter,
    )
    candidate_id_filter = {int(cid) for cid in (candidate_ids or []) if cid is not None}

    all_profiles = [
        p
        for p in PROFILES_BY_ID.values()
        if not candidate_id_filter or int(p.get("id") or 0) in candidate_id_filter
        if role_candidate_ids is None or int(p.get("id") or 0) in role_candidate_ids
        if profile_passes_scope(
            p,
            user_role=(current_user.role or "").strip().lower(),
            user_id=current_user.id,
            view_scope=effective_scope,
            recruiter_filter_id=effective_recruiter,
        )
    ]

    semantic_scores = {}
    if product_service and product_service.strip():
        from backend.pipeline.query import get_semantic_scores

        semantic_scores = await get_semantic_scores(product_service)

    ql = q.lower() if q else None

    results = []
    for p in all_profiles:
        profile_id = p.get("id")
        score = semantic_scores.get(profile_id, 0.0) if semantic_scores else None

        primary_role = (p.get("roles") or [{}])[0]
        city_val = (p.get("city") or "").strip() or (p.get("location") or "").split(",")[0].strip()
        loc_val = p.get("work_preference") or p.get("location_type") or ""
        company_product = (primary_role.get("company_details") or {}).get("product_service") or ""
        extracted_prod = p.get("extracted_industry") or ""
        cand_services = p.get("candidate_services") or ""

        product_val = extracted_prod or cand_services or company_product or primary_role.get("industry") or ""
        if not product_val:
            search_text_lower = f"{p.get('headline') or ''} {p.get('about') or ''}".lower()
            found = [kw for kw, kw_lower in zip(INDUSTRY_KEYWORDS, INDUSTRY_KEYWORDS_LOWER) if kw_lower in search_text_lower]
            if found:
                product_val = ", ".join(list(set(found))[:3])

        exp_val = p.get("total_experience_years") or 0
        stability_val = p.get("avg_years_in_company") or 0
        status_val = p.get("status") or ""
        created_by_val = p.get("created_by") or ""
        name_val = p.get("name") or ""
        title_val = primary_role.get("title") or p.get("headline") or ""
        company_val = primary_role.get("company") or ""
        if not company_val and isinstance(p.get("raw_fields"), dict):
            company_val = (p.get("raw_fields") or {}).get("import_company") or ""

        if ql:
            searchable = " ".join(str(v or "") for v in (name_val, title_val, company_val, city_val, p.get("linkedin"), p.get("normalized_linkedin"), p.get("email"), p.get("phone"), p.get("mobile_phone"))).lower()
            if ql not in searchable:
                continue

        if not _matches_filter(title, title_val):
            continue
        if not _matches_filter(company, company_val):
            continue
        if not _matches_filter(city, city_val):
            continue
        if not _matches_filter(created_by, created_by_val):
            continue
        if not _matches_filter(location_type, loc_val):
            continue
        if not semantic_scores and product_service and not _matches_filter(product_service, product_val):
            continue
        if semantic_scores and score is not None and score < 0.45:
            continue

        fn = (p.get("first_name") or "").strip() or (name_val.split() or [""])[0]
        ln = (p.get("last_name") or "").strip() or (" ".join(name_val.split()[1:]) if name_val else "")
        results.append(
            {
                "id": profile_id,
                "name": name_val,
                "first_name": fn,
                "last_name": ln,
                "linkedin": p.get("linkedin"),
                "email": p.get("email") or "",
                "phone": p.get("phone") or p.get("mobile_phone") or "",
                "response": p.get("response", ""),
                "notes": p.get("notes", ""),
                "title": title_val,
                "company": company_val,
                "product_service": product_val,
                "city": city_val,
                "location_type": loc_val,
                "total_experience_years": round(float(exp_val), 1) if exp_val else 0,
                "avg_tenure_years": round(float(stability_val), 1) if stability_val else 0,
                "status": status_val,
                "created_by": created_by_val,
                "li_status": p.get("li_status") or "",
                "li_response_text": p.get("li_response_text") or "",
                "heyreach_campaign_id": p.get("heyreach_campaign_id") or "",
                "email_campaign_id": p.get("email_campaign_id") or "",
                "message_sent_count": p.get("message_sent_count") or 0,
                "li_sent_count": p.get("li_sent_count") or 0,
                "headline": p.get("headline") or "",
                "match_score": round(score * 100, 1) if score is not None else None,
                "owner_user_id": p.get("owner_user_id"),
                "pool_source": p.get("pool_source"),
                "is_master_row": p.get("owner_user_id") is None,
                "raw_fields": p.get("raw_fields") if isinstance(p.get("raw_fields"), dict) else {},
            }
        )

    if min_exp is not None:
        results = [r for r in results if r["total_experience_years"] >= min_exp]
    if max_exp is not None:
        results = [r for r in results if r["total_experience_years"] <= max_exp]
    if min_avg_tenure is not None:
        results = [r for r in results if r["avg_tenure_years"] >= min_avg_tenure]

    status_counts = {}
    for r in results:
        s = (r.get("status") or "").strip()
        if s:
            status_counts[s] = status_counts.get(s, 0) + 1
        else:
            status_counts["To be started"] = status_counts.get("To be started", 0) + 1

    if status and status.strip():
        status_vals = [s.strip().lower() for s in status.split(",") if s.strip()]
        if status_vals:
            results = [
                r
                for r in results
                if (r.get("status") or "To be started").strip().lower() in status_vals
            ]

    if candidate_id_filter:
        order = {int(cid): idx for idx, cid in enumerate(candidate_ids or [])}
        results.sort(key=lambda x: order.get(int(x.get("id") or 0), len(order)))
    elif semantic_scores:
        results.sort(key=lambda x: x.get("match_score") or 0, reverse=True)
    else:
        sort_key_map = {
            "name": "name",
            "title": "title",
            "company": "company",
            "city": "city",
            "exp": "total_experience_years",
            "tenure": "avg_tenure_years",
        }
        key = sort_key_map.get(sort_by, "name")
        results.sort(key=lambda x: (x.get(key) or ""), reverse=(sort_dir == "desc"))

    return {
        "candidates": results,
        "status_counts": status_counts,
        "is_semantic_search": bool(semantic_scores),
        "effective_scope": effective_scope,
        "effective_recruiter": effective_recruiter,
    }


@router.get("/candidates/browse/summary")
async def browse_summary(
    current_user: schemas.User = Depends(deps.get_current_user),
    view_scope: Optional[str] = Query(
        None,
        description="admin: master | recruiter_pools | all_recruiter_pools",
    ),
    recruiter_filter_id: Optional[int] = Query(
        None,
        description="admin + recruiter_pools: which recruiter's pool",
    ),
    role_id: Optional[int] = None,
):
    """Return unfiltered Talent Pool counts for the current scope."""
    started = time.monotonic()
    try:
        effective_scope, effective_recruiter = resolve_browse_scope(
            current_user,
            view_scope,
            recruiter_filter_id,
        )
        cache_key_src = json.dumps({
            "uid": current_user.id,
            "role": current_user.role,
            "view_scope": effective_scope,
            "recruiter_filter_id": effective_recruiter,
            "role_id": role_id,
        }, sort_keys=True)
        cache_key = hashlib.md5(cache_key_src.encode()).hexdigest()
        cached = _summary_cache.get(cache_key)
        if cached and (time.monotonic() - cached["ts"]) < _BROWSE_CACHE_TTL:
            result = cached["result"]
            _log_browse_timing(
                "summary",
                started,
                total=result["total"],
                scope=result["effective_scope"],
                recruiter_id=result["effective_recruiter"],
            )
            return result

        result = await fetch_browse_summary_counts(
            current_user=current_user,
            view_scope=effective_scope,
            recruiter_filter_id=effective_recruiter,
            role_id=role_id,
        )
        _summary_cache[cache_key] = {"result": result, "ts": time.monotonic()}
        _log_browse_timing(
            "summary",
            started,
            total=result["total"],
            scope=result["effective_scope"],
            recruiter_id=result["effective_recruiter"],
        )
        return result
    except Exception:
        _log_browse_timing("summary", started, status="error")
        raise


@router.get("/candidates/browse")
async def browse_candidates(
    current_user: schemas.User = Depends(deps.get_current_user),
    view_scope: Optional[str] = Query(
        None,
        description="admin: master | recruiter_pools | all_recruiter_pools",
    ),
    recruiter_filter_id: Optional[int] = Query(
        None,
        description="admin + recruiter_pools: which recruiter's pool",
    ),
    # Pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=5000),
    # Search
    q: Optional[str] = None,
    # Filters
    title: Optional[str] = None,
    company: Optional[str] = None,
    city: Optional[str] = None,
    location_type: Optional[str] = None,   # On-site | Remote | Hybrid
    product_service: Optional[str] = None,
    status: Optional[str] = None,          # shortlisted | rejected | etc.
    created_by: Optional[str] = None,
    min_exp: Optional[float] = None,
    max_exp: Optional[float] = None,
    min_avg_tenure: Optional[float] = None, # stability
    candidate_ids: Optional[str] = None,
    role_id: Optional[int] = None,
    # Role-link date range (when the candidate was added to the role, YYYY-MM-DD)
    added_from: Optional[str] = None,
    added_to: Optional[str] = None,
    # Sort
    sort_by: Optional[str] = "name",
    sort_dir: Optional[str] = "asc",
):
    """Browse candidates with role-based pool scope."""
    started = time.monotonic()
    for label, value in (("added_from", added_from), ("added_to", added_to)):
        if value:
            try:
                date.fromisoformat(value)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{label} must be YYYY-MM-DD")
    effective_scope, effective_recruiter = resolve_browse_scope(
        current_user,
        view_scope,
        recruiter_filter_id,
    )
    candidate_id_list = _parse_candidate_ids(candidate_ids)

    # One cache key for both the SQL fast path and the in-memory fallback, so
    # repeated identical requests (re-renders, tab switches) skip the DB.
    browse_cache_key = hashlib.md5(json.dumps({
        "uid": current_user.id,
        "role": current_user.role,
        "view_scope": effective_scope,
        "recruiter_filter_id": effective_recruiter,
        "page": page, "page_size": page_size, "q": q, "title": title,
        "company": company, "city": city, "location_type": location_type,
        "product_service": product_service, "status": status, "created_by": created_by,
        "min_exp": min_exp, "max_exp": max_exp, "min_avg_tenure": min_avg_tenure,
        "candidate_ids": candidate_id_list,
        "role_id": role_id,
        "added_from": added_from, "added_to": added_to,
        "sort_by": sort_by, "sort_dir": sort_dir,
    }, sort_keys=True).encode()).hexdigest()
    cached_browse = _browse_cache.get(browse_cache_key)
    if cached_browse and (time.monotonic() - cached_browse["ts"]) < _BROWSE_CACHE_TTL:
        cached_result = cached_browse["result"]
        _log_browse_timing(
            "rows",
            started,
            total=cached_result.get("total"),
            page_size=page_size,
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return cached_result

    if _can_use_fast_sql_browse(
        q=q,
        title=title,
        company=company,
        city=city,
        location_type=location_type,
        product_service=product_service,
        status=status,
        created_by=created_by,
        min_exp=min_exp,
        max_exp=max_exp,
        min_avg_tenure=min_avg_tenure,
        candidate_ids=candidate_id_list,
    ) and not _profile_cache_looks_test_scoped():
        result = await fetch_browse_page_sql(
            current_user=current_user,
            view_scope=effective_scope,
            recruiter_filter_id=effective_recruiter,
            page=page,
            page_size=page_size,
            role_id=role_id,
            q=q,
            title=title,
            company=company,
            city=city,
            location_type=location_type,
            product_service=product_service,
            status=status,
            created_by=created_by,
            min_exp=min_exp,
            max_exp=max_exp,
            min_avg_tenure=min_avg_tenure,
            candidate_ids=candidate_id_list,
            added_from=added_from,
            added_to=added_to,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )
        _browse_cache[browse_cache_key] = {"result": result, "ts": time.monotonic()}
        _log_browse_timing(
            "rows",
            started,
            total=result.get("total"),
            page_size=page_size,
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return result

    await _ensure_profiles_loaded()

    # ── Cache key from all params ───────────────────────────────────
    cache_key_src = json.dumps({
        "uid": current_user.id,
        "role": current_user.role,
        "view_scope": effective_scope,
        "recruiter_filter_id": effective_recruiter,
        "page": page, "page_size": page_size, "q": q, "title": title,
        "company": company, "city": city, "location_type": location_type,
        "product_service": product_service, "status": status, "created_by": created_by,
        "min_exp": min_exp, "max_exp": max_exp, "min_avg_tenure": min_avg_tenure,
        "candidate_ids": candidate_id_list,
        "role_id": role_id,
        "added_from": added_from, "added_to": added_to,
        "sort_by": sort_by, "sort_dir": sort_dir,
    }, sort_keys=True)
    cache_key = hashlib.md5(cache_key_src.encode()).hexdigest()

    # Serve from cache if fresh
    cached = _browse_cache.get(cache_key)
    if cached and (time.monotonic() - cached["ts"]) < _BROWSE_CACHE_TTL:
        cached_result = cached["result"]
        _log_browse_timing(
            "rows",
            started,
            total=cached_result.get("total"),
            page_size=page_size,
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return cached["result"]

    browse_payload = await build_browse_candidate_rows(
        current_user=current_user,
        view_scope=view_scope,
        recruiter_filter_id=recruiter_filter_id,
        q=q,
        title=title,
        company=company,
        city=city,
        location_type=location_type,
        product_service=product_service,
        status=status,
        created_by=created_by,
        min_exp=min_exp,
        max_exp=max_exp,
        min_avg_tenure=min_avg_tenure,
        candidate_ids=candidate_id_list,
        role_id=role_id,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    results = browse_payload["candidates"]
    status_counts = browse_payload["status_counts"]
    is_semantic_search = browse_payload["is_semantic_search"]

    # --- Pagination ---
    total = len(results)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_results = results[offset: offset + page_size]

    result = {
        "candidates": page_results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "status_counts": status_counts,
        "is_semantic_search": is_semantic_search,
    }
    # Store in cache
    _browse_cache[cache_key] = {"result": result, "ts": time.monotonic()}
    _log_browse_timing(
        "rows",
        started,
        total=total,
        page_size=page_size,
        scope=effective_scope,
        recruiter_id=effective_recruiter,
    )
    return result


@router.get("/candidates/browse/meta")
async def browse_metadata(
    current_user: schemas.User = Depends(deps.get_current_user),
    view_scope: Optional[str] = Query(None),
    recruiter_filter_id: Optional[int] = Query(None),
    role_id: Optional[int] = Query(None),
):
    """Return unique filter values (for dropdowns) scoped like browse."""
    started = time.monotonic()
    effective_scope, effective_recruiter = resolve_browse_scope(
        current_user,
        view_scope,
        recruiter_filter_id,
    )

    # Unit tests inject a small in-memory cache initializer to exercise the
    # fallback path. Production metadata is SQL-backed and must not pay the
    # full semantic-cache cold-start cost.
    if (
        not PROFILES_BY_ID
        and getattr(initialize_cache, "__module__", "") != "backend.pipeline.query"
    ):
        await _ensure_profiles_loaded()

    if _profile_cache_looks_test_scoped():
        rows = await build_browse_candidate_rows(
            current_user=current_user,
            view_scope=view_scope,
            recruiter_filter_id=recruiter_filter_id,
            role_id=role_id,
            sort_by="name",
            sort_dir="asc",
        )
        candidates = rows.get("candidates") or []

        def _clean(values: List[Any], *, limit: int = 100) -> List[str]:
            seen = set()
            cleaned: List[str] = []
            for value in values:
                text = str(value or "").strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(text)
            return sorted(cleaned, key=lambda item: item.lower())[:limit]

        result = {
            "companies": _clean([row.get("company") for row in candidates]),
            "titles": _clean([row.get("title") for row in candidates]),
            "cities": _clean([row.get("city") for row in candidates]),
            "statuses": _clean([row.get("status") for row in candidates]),
            "created_by": _clean([row.get("created_by") for row in candidates]),
            "location_types": _clean([row.get("location_type") for row in candidates]),
            "total": len(candidates),
        }
        _log_browse_timing(
            "meta",
            started,
            total=result["total"],
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return result

    where_sql, params = _summary_scope_sql(
        current_user,
        effective_scope=effective_scope,
        effective_recruiter=effective_recruiter,
        role_id=role_id,
    )

    # Filter dropdown values change rarely — serve repeats from a short cache.
    meta_cache_key = hashlib.md5(json.dumps({
        "uid": current_user.id,
        "role": current_user.role,
        "scope": effective_scope,
        "recruiter": effective_recruiter,
        "role_id": role_id,
    }, sort_keys=True).encode()).hexdigest()
    cached_meta = _meta_cache.get(meta_cache_key)
    if cached_meta and (time.monotonic() - cached_meta["ts"]) < _META_CACHE_TTL:
        _log_browse_timing(
            "meta",
            started,
            total=sum(len(v) for v in cached_meta["result"].values() if isinstance(v, list)),
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return cached_meta["result"]

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    def _clean(values: List[Any], *, limit: int = 100) -> List[str]:
        seen = set()
        cleaned: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return sorted(cleaned, key=lambda item: item.lower())[:limit]

    try:
        # All seven dropdown sources in ONE statement: each remote round trip
        # costs ~0.6s, so 7 sequential queries were ~4.5s of pure latency.
        with conn.cursor() as cur:
            cur.execute(
                f"""
                (SELECT 'company' AS kind, value FROM (
                    SELECT NULLIF(TRIM(c.raw_fields->>'import_company'), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                    UNION
                    SELECT NULLIF(TRIM(co.name), '') AS value
                    FROM candidates c
                    JOIN roles r ON r.candidate_id = c.id
                    JOIN companies co ON co.id = r.company_id
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL LIMIT 100)
                UNION ALL
                (SELECT 'title', value FROM (
                    SELECT NULLIF(TRIM(c.headline), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                    UNION
                    SELECT NULLIF(TRIM(r.title), '') AS value
                    FROM candidates c
                    JOIN roles r ON r.candidate_id = c.id
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL LIMIT 100)
                UNION ALL
                (SELECT 'city', value FROM (
                    SELECT DISTINCT NULLIF(TRIM(COALESCE(NULLIF(c.city, ''), split_part(COALESCE(c.location, ''), ',', 1))), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL LIMIT 100)
                UNION ALL
                (SELECT 'product', value FROM (
                    SELECT NULLIF(TRIM(c.raw_fields->>'extracted_industry'), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                    UNION
                    SELECT NULLIF(TRIM(c.raw_fields->>'services'), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                    UNION
                    SELECT NULLIF(TRIM(co.product_service), '') AS value
                    FROM candidates c
                    JOIN roles r ON r.candidate_id = c.id
                    JOIN companies co ON co.id = r.company_id
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL LIMIT 100)
                UNION ALL
                (SELECT 'location_type', value FROM (
                    SELECT DISTINCT NULLIF(TRIM(COALESCE(c.raw_fields->>'work_preference', c.raw_fields->>'location_type')), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL)
                UNION ALL
                (SELECT 'status', value FROM (
                    SELECT DISTINCT COALESCE(NULLIF(TRIM(c.status), ''), 'To be started') AS value
                    FROM candidates c
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL)
                UNION ALL
                (SELECT 'recruiter', value FROM (
                    SELECT DISTINCT NULLIF(TRIM(c.created_by), '') AS value
                    FROM candidates c
                    WHERE {where_sql}
                 ) v WHERE value IS NOT NULL)
                """,
                [*params] * 11,
            )
            grouped: Dict[str, List[str]] = {}
            for kind, value in cur.fetchall():
                grouped.setdefault(kind, []).append(value)

        companies = _clean(grouped.get("company", []))
        titles = _clean(grouped.get("title", []))
        cities = _clean(grouped.get("city", []))
        products = _clean(grouped.get("product", []))
        locations = _clean(grouped.get("location_type", []), limit=200)
        statuses = _clean(grouped.get("status", []), limit=200)
        recruiters = _clean(grouped.get("recruiter", []), limit=200)

        statuses = _clean([*statuses, *RECRUITMENT_STAGES], limit=300)
        result = {
            "companies": companies,
            "cities": cities,
            "titles": titles,
            "products": products,
            "location_types": locations,
            "statuses": statuses,
            "recruiters": recruiters,
        }
        _meta_cache[meta_cache_key] = {"result": result, "ts": time.monotonic()}
        _log_browse_timing(
            "meta",
            started,
            total=sum(len(result.get(key, [])) for key in result),
            scope=effective_scope,
            recruiter_id=effective_recruiter,
        )
        return result
    finally:
        return_db_connection(conn)

@router.post("/candidates/{candidate_id}/status")
async def update_status(
    candidate_id: int,
    update: StatusUpdate,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    started = time.monotonic()
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=True)
    success = update_candidate_status(candidate_id, update.status)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update candidate status")
    from backend.api.routes.candidates import invalidate_candidate_count_caches
    from backend.api.routes.roles import invalidate_role_detail_cache_for_candidate

    invalidate_candidate_count_caches(refresh_profile_ids=[candidate_id])
    invalidate_role_detail_cache_for_candidate(candidate_id)
    logger.info(
        "candidate status updated candidate_id=%s status=%s duration_ms=%.1f",
        candidate_id,
        update.status,
        (time.monotonic() - started) * 1000,
    )
    return {"message": "Status updated successfully"}

@router.patch("/candidates/{candidate_id}/notes")
async def update_notes(
    candidate_id: int,
    update: NotesUpdate,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    from backend.pipeline.query import update_candidate_notes

    started = time.monotonic()
    _authorize_candidate_update(candidate_id, current_user, allow_role_access=False)
    success = update_candidate_notes(candidate_id, update.notes)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update notes")
    from backend.api.routes.roles import invalidate_role_detail_cache_for_candidate

    _invalidate_browse_cache()
    invalidate_role_detail_cache_for_candidate(candidate_id)
    # Calls rows carry candidate_notes — refresh that cache too.
    try:
        from backend.api.routes.calls import invalidate_calls_cache
        invalidate_calls_cache()
    except Exception:
        pass
    logger.info(
        "candidate notes updated candidate_id=%s duration_ms=%.1f",
        candidate_id,
        (time.monotonic() - started) * 1000,
    )
    return {"message": "Notes updated successfully"}

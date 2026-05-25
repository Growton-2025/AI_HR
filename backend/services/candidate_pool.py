"""Candidate pool ownership, contact reuse, assignment, and browse scope."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from backend.services.linkedin_normalize import normalize_linkedin

logger = logging.getLogger(__name__)

POOL_SOURCE_LEGACY_MASTER = "legacy_master"
POOL_SOURCE_RECRUITER_UPLOAD = "recruiter_upload"
POOL_SOURCE_ADMIN_ASSIGNED = "admin_assigned"
POOL_SOURCE_CATALOG_FROM_UPLOAD = "catalog_from_upload"

VIEW_SCOPE_MASTER = "master"
VIEW_SCOPE_RECRUITER_POOLS = "recruiter_pools"
VIEW_SCOPE_ALL_RECRUITER_POOLS = "all_recruiter_pools"

REQUIRED_IMPORT_TARGETS = frozenset(
    {"first_name", "last_name", "linkedin", "city", "title"}
)

OPTIONAL_IMPORT_TARGETS = (
    "company_name",
    "email",
    "phone",
    "location",
    "notes",
    "headline",
    "about",
)

HEADER_ALIASES: Dict[str, set] = {
    "first_name": {"firstname", "first name", "given name", "fname", "fn"},
    "last_name": {"lastname", "last name", "surname", "lname", "ln"},
    "linkedin": {
        "linkedin",
        "linkedin url",
        "linkedin profile",
        "profile url",
        "li url",
        "person linkedin",
    },
    "city": {"city", "metro", "town", "current city"},
    "title": {"title", "role", "job title", "position", "designation"},
    "company_name": {"company", "company name", "employer", "organization"},
    "email": {"email", "email address", "work email"},
    "phone": {"phone", "mobile", "mobile phone", "phone number"},
    "location": {"location", "address"},
    "notes": {"notes", "comments", "remarks"},
}


def suggest_header_mapping(headers: List[str]) -> Dict[str, str]:
    """Map source header -> canonical target (best-effort heuristics)."""
    out: Dict[str, str] = {}
    for orig in headers:
        if not orig or not str(orig).strip():
            continue
        key = re.sub(r"[^a-z0-9]+", " ", orig.strip().lower()).strip()
        matched = None
        for target, aliases in HEADER_ALIASES.items():
            if key in aliases:
                matched = target
                break
            for a in aliases:
                if len(a) > 2 and (a in key or key in a):
                    matched = target
                    break
            if matched:
                break
        if matched:
            out[orig] = matched
    return out


def profile_passes_scope(
    profile: Dict[str, Any],
    *,
    user_role: str,
    user_id: int,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
) -> bool:
    if profile.get("is_archived"):
        return False
    oid = profile.get("owner_user_id")
    r = (user_role or "").strip().lower()
    if r == "admin":
        scope = view_scope or VIEW_SCOPE_MASTER
        if scope == VIEW_SCOPE_MASTER:
            return True
        if scope == VIEW_SCOPE_RECRUITER_POOLS:
            return oid is not None and (
                recruiter_filter_id is None or oid == recruiter_filter_id
            )
        if scope == VIEW_SCOPE_ALL_RECRUITER_POOLS:
            return oid is not None
        return True
    # Recruiters (and any non-admin): only rows they own (uploads + admin-assigned copies).
    return oid is not None and oid == user_id


def fetch_best_contact_for_normalized_li(
    cur, normalized_li: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    if not normalized_li:
        return None, None
    cur.execute(
        """
        SELECT email, COALESCE(mobile_phone, phone) AS ph
        FROM candidates
        WHERE normalized_linkedin = %s AND COALESCE(is_archived, FALSE) = FALSE
        ORDER BY
          CASE WHEN email IS NOT NULL AND email <> '' THEN 0 ELSE 1 END,
          CASE WHEN COALESCE(mobile_phone, phone) IS NOT NULL
               AND COALESCE(mobile_phone, phone) <> '' THEN 0 ELSE 1 END,
          id
        LIMIT 1
        """,
        (normalized_li,),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return row[0] or None, row[1] or None


def fetch_best_contacts_for_normalized_lis(
    cur, normalized_lis: Iterable[str]
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Batch contact reuse for imports without changing single-row preference rules."""
    keys = sorted({str(item).strip() for item in normalized_lis if str(item).strip()})
    if not keys:
        return {}
    cur.execute(
        """
        SELECT DISTINCT ON (normalized_linkedin)
               normalized_linkedin, email, COALESCE(mobile_phone, phone) AS ph
        FROM candidates
        WHERE normalized_linkedin = ANY(%s)
          AND COALESCE(is_archived, FALSE) = FALSE
        ORDER BY normalized_linkedin,
          CASE WHEN email IS NOT NULL AND email <> '' THEN 0 ELSE 1 END,
          CASE WHEN COALESCE(mobile_phone, phone) IS NOT NULL
               AND COALESCE(mobile_phone, phone) <> '' THEN 0 ELSE 1 END,
          id
        """,
        (keys,),
    )
    return {
        row[0]: (row[1] or None, row[2] or None)
        for row in cur.fetchall()
        if row and row[0]
    }


def load_master_row(cur, master_id: int) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        SELECT id, name, first_name, last_name, linkedin, normalized_linkedin, city,
               headline, about, location, email, COALESCE(mobile_phone, phone) AS ph,
               raw_fields, pool_source
        FROM candidates
        WHERE id = %s AND owner_user_id IS NULL AND COALESCE(is_archived, FALSE) = FALSE
        """,
        (master_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    rf = row[12] or {}
    if isinstance(rf, str):
        try:
            rf = json.loads(rf) if rf.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            rf = {}
    return {
        "id": row[0],
        "name": row[1],
        "first_name": row[2],
        "last_name": row[3],
        "linkedin": row[4],
        "normalized_linkedin": row[5],
        "city": row[6],
        "headline": row[7],
        "about": row[8],
        "location": row[9],
        "email": row[10],
        "phone": row[11],
        "raw_fields": rf,
        "pool_source": row[13],
    }


def upsert_master_catalog_row(
    cur,
    *,
    normalized_li: str,
    raw_linkedin: str,
    first_name: str,
    last_name: str,
    city: Optional[str],
    title: Optional[str],
    company_name: Optional[str],
    email: Optional[str],
    phone: Optional[str],
    location: Optional[str],
    notes: Optional[str],
    pool_source: str = POOL_SOURCE_CATALOG_FROM_UPLOAD,
    raw_fields_extra: Optional[Dict[str, Any]] = None,
    existing_id: Optional[int] = None,
    lookup_complete: bool = False,
) -> int:
    """Insert or update master library row keyed by normalized_linkedin."""
    name = (f"{first_name or ''} {last_name or ''}").strip() or "Unknown"
    headline = title or ""
    rf: Dict[str, Any] = {}
    if company_name:
        rf["import_company"] = company_name
    if raw_fields_extra:
        rf.update(raw_fields_extra)
    rf_json = json.dumps(rf) if rf else "{}"

    ex = (existing_id,) if existing_id else None
    if not ex and not lookup_complete and normalized_li:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id IS NULL AND normalized_linkedin = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (normalized_li,),
        )
        ex = cur.fetchone()
    if not ex and not lookup_complete and email:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id IS NULL AND email = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (email,),
        )
        ex = cur.fetchone()
    if not ex and not lookup_complete and first_name and last_name and company_name:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id IS NULL AND first_name = %s AND last_name = %s
              AND raw_fields->>'import_company' = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (first_name, last_name, company_name),
        )
        ex = cur.fetchone()
        
    loc = location or city

    if ex:
        mid = ex[0]
        cur.execute(
            """
            UPDATE candidates SET
              first_name = COALESCE(NULLIF(TRIM(%s), ''), first_name),
              last_name = COALESCE(NULLIF(TRIM(%s), ''), last_name),
              name = COALESCE(NULLIF(TRIM(%s), ''), name),
              linkedin = COALESCE(NULLIF(TRIM(%s), ''), linkedin),
              city = COALESCE(NULLIF(TRIM(%s), ''), city),
              headline = COALESCE(NULLIF(TRIM(%s), ''), headline),
              location = COALESCE(NULLIF(TRIM(%s), ''), location),
              email = COALESCE(NULLIF(TRIM(%s), ''), email),
              mobile_phone = COALESCE(NULLIF(TRIM(%s), ''), mobile_phone),
              raw_fields = COALESCE(raw_fields, '{}'::jsonb) || %s::jsonb,
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                first_name,
                last_name,
                name,
                raw_linkedin,
                city,
                headline,
                loc,
                email,
                phone,
                rf_json,
                mid,
            ),
        )
        return mid

    cur.execute(
        """
        INSERT INTO candidates (
          name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
          location, email, mobile_phone, raw_fields, pool_source, owner_user_id,
          created_by
        ) VALUES (
          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb),
          %s, NULL, 'catalog_upload'
        ) RETURNING id
        """,
        (
            name,
            first_name,
            last_name,
            raw_linkedin,
            normalized_li,
            city,
            headline,
            loc,
            email,
            phone,
            rf_json,
            pool_source,
        ),
    )
    return cur.fetchone()[0]


def upsert_recruiter_pool_row(
    cur,
    *,
    owner_id: int,
    master_id: int,
    normalized_li: str,
    raw_linkedin: str,
    first_name: str,
    last_name: str,
    city: Optional[str],
    title: Optional[str],
    company_name: Optional[str],
    email: Optional[str],
    phone: Optional[str],
    location: Optional[str],
    notes: Optional[str],
    pool_source: str,
    source_upload_id: Optional[int],
    assigned_by_user_id: Optional[int] = None,
    raw_fields_extra: Optional[Dict[str, Any]] = None,
    existing_id: Optional[int] = None,
    lookup_complete: bool = False,
) -> Tuple[int, str]:
    """
    Returns (candidate_id, 'inserted'|'updated').
    """
    name = (f"{first_name or ''} {last_name or ''}").strip() or "Unknown"
    headline = title or ""
    rf: Dict[str, Any] = {}
    if company_name:
        rf["import_company"] = company_name
    if raw_fields_extra:
        rf.update(raw_fields_extra)
    rf_json = json.dumps(rf) if rf else "{}"
    loc = location or city

    ex = (existing_id,) if existing_id else None
    if not ex and not lookup_complete and normalized_li:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id = %s AND normalized_linkedin = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (owner_id, normalized_li),
        )
        ex = cur.fetchone()
    if not ex and not lookup_complete and email:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id = %s AND email = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (owner_id, email),
        )
        ex = cur.fetchone()
    if not ex and not lookup_complete and first_name and last_name and company_name:
        cur.execute(
            """
            SELECT id FROM candidates
            WHERE owner_user_id = %s AND first_name = %s AND last_name = %s
              AND raw_fields->>'import_company' = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (owner_id, first_name, last_name, company_name),
        )
        ex = cur.fetchone()
    if ex:
        cid = ex[0]
        cur.execute(
            """
            UPDATE candidates SET
              source_master_candidate_id = COALESCE(source_master_candidate_id, %s),
              first_name = COALESCE(NULLIF(TRIM(%s), ''), first_name),
              last_name = COALESCE(NULLIF(TRIM(%s), ''), last_name),
              name = COALESCE(NULLIF(TRIM(%s), ''), name),
              linkedin = COALESCE(NULLIF(TRIM(%s), ''), linkedin),
              city = COALESCE(NULLIF(TRIM(%s), ''), city),
              headline = COALESCE(NULLIF(TRIM(%s), ''), headline),
              location = COALESCE(NULLIF(TRIM(%s), ''), location),
              email = COALESCE(NULLIF(TRIM(%s), ''), email),
              mobile_phone = COALESCE(NULLIF(TRIM(%s), ''), mobile_phone),
              notes = COALESCE(NULLIF(TRIM(%s), ''), notes),
              raw_fields = COALESCE(raw_fields, '{}'::jsonb) || %s::jsonb,
              source_upload_ids = CASE WHEN %s::int IS NOT NULL AND NOT (%s::int = ANY(COALESCE(source_upload_ids, '{}'::int[]))) 
                                    THEN array_append(COALESCE(source_upload_ids, '{}'::int[]), %s::int) 
                                    ELSE source_upload_ids END,
              assigned_by_user_id = COALESCE(%s, assigned_by_user_id),
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                master_id,
                first_name,
                last_name,
                name,
                raw_linkedin,
                city,
                headline,
                loc,
                email,
                phone,
                notes,
                rf_json,
                source_upload_id,
                source_upload_id,
                source_upload_id,
                assigned_by_user_id,
                cid,
            ),
        )
        return cid, "updated"

    cur.execute(
        """
        INSERT INTO candidates (
          name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
          location, email, mobile_phone, notes, raw_fields,
          owner_user_id, pool_source, source_master_candidate_id, source_upload_ids,
          assigned_by_user_id, created_by, status
        ) VALUES (
          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb),
          %s, %s, %s, CASE WHEN %s::int IS NOT NULL THEN ARRAY[%s::int] ELSE '{}'::int[] END, %s, %s, 'To be started'
        ) RETURNING id
        """,
        (
            name,
            first_name,
            last_name,
            raw_linkedin,
            normalized_li,
            city,
            headline,
            loc,
            email,
            phone,
            notes,
            rf_json,
            owner_id,
            pool_source,
            master_id,
            source_upload_id,
            source_upload_id,
            assigned_by_user_id,
            str(owner_id),
        ),
    )
    return cur.fetchone()[0], "inserted"


def assign_master_to_recruiter(
    cur,
    *,
    master_id: int,
    recruiter_user_id: int,
    admin_user_id: int,
) -> Tuple[int, str]:
    m = load_master_row(cur, master_id)
    if not m:
        raise ValueError("master_not_found")
    nli = m["normalized_linkedin"] or normalize_linkedin(m["linkedin"])
    if not nli:
        raise ValueError("master_missing_linkedin")

    email, phone = fetch_best_contact_for_normalized_li(cur, nli)
    email = email or m.get("email")
    phone = phone or m.get("phone")

    cur.execute(
        """
        SELECT id FROM candidates
        WHERE owner_user_id = %s AND normalized_linkedin = %s
          AND COALESCE(is_archived, FALSE) = FALSE
        LIMIT 1
        """,
        (recruiter_user_id, nli),
    )
    ex = cur.fetchone()
    if ex:
        cid = ex[0]
        cur.execute(
            """
            UPDATE candidates SET
              source_master_candidate_id = COALESCE(source_master_candidate_id, %s),
              first_name = COALESCE(NULLIF(TRIM(%s), ''), first_name),
              last_name = COALESCE(NULLIF(TRIM(%s), ''), last_name),
              name = COALESCE(NULLIF(TRIM(%s), ''), name),
              linkedin = COALESCE(NULLIF(TRIM(%s), ''), linkedin),
              city = COALESCE(NULLIF(TRIM(%s), ''), city),
              headline = COALESCE(NULLIF(TRIM(%s), ''), headline),
              location = COALESCE(NULLIF(TRIM(%s), ''), location),
              email = COALESCE(NULLIF(TRIM(%s), ''), email),
              mobile_phone = COALESCE(NULLIF(TRIM(%s), ''), mobile_phone),
              assigned_by_user_id = %s,
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                master_id,
                m.get("first_name"),
                m.get("last_name"),
                m.get("name"),
                m.get("linkedin"),
                m.get("city"),
                m.get("headline"),
                m.get("location"),
                email,
                phone,
                admin_user_id,
                cid,
            ),
        )
        return cid, "merged"

    name = m.get("name") or ""
    rf_dump = json.dumps(m.get("raw_fields") or {}, default=str)
    cur.execute(
        """
        INSERT INTO candidates (
          name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
          about, location, email, mobile_phone, raw_fields,
          owner_user_id, pool_source, source_master_candidate_id,
          assigned_by_user_id, created_by, status
        ) VALUES (
          %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb),
          %s, %s, %s, %s, %s, 'To be started'
        ) RETURNING id
        """,
        (
            name,
            m.get("first_name"),
            m.get("last_name"),
            m.get("linkedin"),
            nli,
            m.get("city"),
            m.get("headline"),
            m.get("about"),
            m.get("location"),
            email,
            phone,
            rf_dump,
            recruiter_user_id,
            POOL_SOURCE_ADMIN_ASSIGNED,
            master_id,
            admin_user_id,
            str(recruiter_user_id),
        ),
    )
    return cur.fetchone()[0], "inserted"


def assert_recruiter_can_touch_candidate(cur, user_id: int, candidate_id: int) -> None:
    cur.execute(
        """
        SELECT owner_user_id FROM candidates
        WHERE id = %s AND COALESCE(is_archived, FALSE) = FALSE
        """,
        (candidate_id,),
    )
    row = cur.fetchone()
    if not row:
        raise PermissionError("not_found")
    if row[0] != user_id:
        raise PermissionError("forbidden")


def assert_admin_or_recruiter_owner(
    cur, *, role: str, user_id: int, candidate_id: int
) -> Tuple[Optional[int], bool]:
    """Returns (owner_user_id, is_master)."""
    cur.execute(
        """
        SELECT owner_user_id FROM candidates
        WHERE id = %s AND COALESCE(is_archived, FALSE) = FALSE
        """,
        (candidate_id,),
    )
    row = cur.fetchone()
    if not row:
        raise PermissionError("not_found")
    oid = row[0]
    is_master = oid is None
    if (role or "").strip().lower() == "admin":
        return oid, is_master
    if oid != user_id:
        raise PermissionError("forbidden")
    return oid, is_master

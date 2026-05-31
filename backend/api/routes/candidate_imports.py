"""Recruiter CSV/XLSX import: preview, optional LLM mapping assist, commit with dual master write."""

import io
import json
import logging
import re
import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field

from backend.api import deps, schemas
from backend.db.connection import get_db_connection, return_db_connection
from backend.pipeline import query
from backend.services.candidate_pool import (
    REQUIRED_IMPORT_TARGETS,
    OPTIONAL_IMPORT_TARGETS,
    suggest_header_mapping,
    upsert_master_catalog_row,
    upsert_recruiter_pool_row,
    fetch_best_contacts_for_normalized_lis,
    POOL_SOURCE_RECRUITER_UPLOAD,
)
from backend.services.linkedin_normalize import normalize_linkedin

router = APIRouter()
logger = logging.getLogger(__name__)
_upload_threads: Dict[int, threading.Thread] = {}
_upload_threads_lock = threading.Lock()

IMPORT_TARGETS = ("ignore",) + tuple(sorted(REQUIRED_IMPORT_TARGETS)) + tuple(OPTIONAL_IMPORT_TARGETS) + ("custom",)


class SuggestMappingBody(BaseModel):
    headers: List[str] = Field(default_factory=list)


def _clean(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float):
        import math

        if math.isnan(val):
            return None
    s = str(val).strip()
    return s if s else None


def _read_frame(file_bytes: bytes, filename: str) -> pd.DataFrame:
    fn = filename.lower()
    if fn.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if fn.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise HTTPException(status_code=400, detail="Only CSV and XLSX are supported")


def _row_values(
    row: pd.Series, mapping: Dict[str, str]
) -> tuple[Dict[str, Optional[str]], Dict[str, Any]]:
    out: Dict[str, Optional[str]] = {}
    raw: Dict[str, Any] = {}
    for src, tgt in mapping.items():
        if src not in row.index:
            continue
        val = row.get(src)
        clean_val = _clean(val)
        if tgt == "ignore" or tgt == "custom" or not tgt:
            if clean_val is not None:
                raw[src] = clean_val
            continue
        out[tgt] = clean_val
    return out, raw


def _normalized_header_key(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(header or "").strip().lower()).strip()


def _is_history_upload_header(header: str, all_headers: Optional[List[str]] = None) -> bool:
    """Columns that should stay raw/custom for verified profile enrichment."""
    key = _normalized_header_key(header)
    headers_keyed = {_normalized_header_key(item) for item in (all_headers or [])}
    has_explicit_current_title = bool(
        headers_keyed
        & {
            "current role",
            "current title",
            "current job title",
            "job title",
            "designation",
        }
    )
    if re.fullmatch(r"company \d+ name", key) or re.fullmatch(r"company \d+", key):
        return True
    if re.fullmatch(r"education \d+ college name", key) or " college name" in key:
        return True
    if re.fullmatch(r"(start date|end date|details|degree name)( \d+)?", key):
        return True
    # Pandas de-duplicates repeated headers as Title.1 / Start date.1; after key normalization
    # they appear as "title 1", "start date 1", etc.
    if re.fullmatch(r"(title|start date|end date|details|degree name) \d+", key):
        return True
    # In Apify/LinkedIn-style sheets, bare "Title" is often Company 1 role history. If the
    # file also has an explicit current-role field, keep bare Title raw for enrichment.
    if key == "title" and has_explicit_current_title:
        return True
    return False


def _validate_mapping(mapping: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(mapping, dict):
        raise HTTPException(status_code=400, detail="mapping must be an object")

    clean_mapping: Dict[str, str] = {}
    for src, tgt in mapping.items():
        target = str(tgt or "ignore")
        if target not in IMPORT_TARGETS:
            target = "ignore"
        clean_mapping[str(src)] = target

    targets_used = {tgt for tgt in clean_mapping.values() if tgt and tgt != "ignore"}
    missing = REQUIRED_IMPORT_TARGETS - targets_used
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required field mappings: {sorted(missing)}",
        )
    return clean_mapping


async def _model_mapping(headers: List[str], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not headers:
        return {}
    try:
        from langchain_openai import ChatOpenAI
        from backend.pipeline.query import safe_json_loads

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            "Map spreadsheet column headers to canonical recruitment fields. "
            f"Headers JSON: {json.dumps(headers)}. "
            f"Sample rows JSON: {json.dumps(sample_rows[:5], default=str)}. "
            "Reply ONLY JSON object. Each key must be an exact header and each value must be an object "
            "with keys target, confidence, reason. target must be one of: "
            "first_name, last_name, linkedin, city, title, company_name, email, phone, "
            "location, notes, headline, about, ignore. Use ignore for columns that do not fit."
        )
        resp = await llm.ainvoke(prompt)
        parsed = safe_json_loads(resp.content, {})
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning("LLM mapping assist failed: %s", e)
        return {}


async def build_upload_preview_response(file: UploadFile, use_llm: bool = False) -> Dict[str, Any]:
    raw = await file.read()
    try:
        df = _read_frame(raw, file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    headers = [str(c) for c in df.columns.tolist()]
    sample = df.head(8).fillna("").to_dict(orient="records")
    deterministic = suggest_header_mapping(headers)
    model_raw = await _model_mapping(headers, sample) if use_llm else {}

    suggested: Dict[str, str] = {}
    mapping_details: Dict[str, Dict[str, Any]] = {}
    for header in headers:
        force_history_custom = _is_history_upload_header(header, headers)
        alias_target = "custom" if force_history_custom else deterministic.get(header)
        model_item = model_raw.get(header)
        model_target = None
        confidence = 0.0
        reason = ""
        if force_history_custom:
            model_item = None
        if isinstance(model_item, dict):
            raw_target = str(model_item.get("target") or "")
            if raw_target in IMPORT_TARGETS:
                model_target = raw_target
            try:
                confidence = float(model_item.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(model_item.get("reason") or "")
        elif isinstance(model_item, str) and model_item in IMPORT_TARGETS:
            model_target = model_item
            confidence = 0.7

        target = model_target or alias_target or "ignore"
        source = "history" if force_history_custom else "model" if model_target else "alias" if alias_target else "manual"
        suggested[header] = target
        mapping_details[header] = {
            "target": target,
            "source": source,
            "confidence": 1.0 if force_history_custom else confidence if model_target else (0.95 if alias_target else 0),
            "reason": (
                "Kept as custom/raw work-history data for verified enrichment"
                if force_history_custom
                else reason or ("Matched known header alias" if alias_target else "No confident match")
            ),
            "sample_values": [_clean(row.get(header)) or "" for row in sample[:4]],
        }

    used_targets = {v for v in suggested.values() if v and v != "ignore"}
    return {
        "filename": file.filename,
        "row_count": int(len(df.index)),
        "headers": headers,
        "sample_rows": sample,
        "suggested_mapping": suggested,
        "deterministic_mapping": deterministic,
        "mapping_details": mapping_details,
        "required_targets": sorted(REQUIRED_IMPORT_TARGETS),
        "optional_targets": list(OPTIONAL_IMPORT_TARGETS),
        "target_options": list(IMPORT_TARGETS),
        "missing_required": sorted(REQUIRED_IMPORT_TARGETS - used_targets),
    }


def _master_row_exists(
    cur,
    *,
    normalized_li: str,
    email: Optional[str],
    first_name: str,
    last_name: str,
    company_name: Optional[str],
) -> bool:
    if normalized_li:
        cur.execute(
            """
            SELECT 1 FROM candidates
            WHERE owner_user_id IS NULL AND normalized_linkedin = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (normalized_li,),
        )
        if cur.fetchone():
            return True
    if email:
        cur.execute(
            """
            SELECT 1 FROM candidates
            WHERE owner_user_id IS NULL AND email = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (email,),
        )
        if cur.fetchone():
            return True
    if first_name and last_name and company_name:
        cur.execute(
            """
            SELECT 1 FROM candidates
            WHERE owner_user_id IS NULL AND first_name = %s AND last_name = %s
              AND raw_fields->>'import_company' = %s
              AND COALESCE(is_archived, FALSE) = FALSE
            LIMIT 1
            """,
            (first_name, last_name, company_name),
        )
        return cur.fetchone() is not None
    return False


def _assign_imported_candidate_to_role(cur, *, role_id: Optional[int], candidate_id: int) -> bool:
    if not role_id:
        return False
    cur.execute(
        """
        INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
        VALUES (%s, %s, '--', '')
        ON CONFLICT (role_id, candidate_id) DO NOTHING
        """,
        (role_id, candidate_id),
    )
    return cur.rowcount > 0


def _identity_key(
    first_name: Optional[str], last_name: Optional[str], company_name: Optional[str]
) -> Optional[tuple[str, str, str]]:
    if not first_name or not last_name or not company_name:
        return None
    return (first_name, last_name, company_name)


def _prepare_import_rows(df: pd.DataFrame, mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for source_index, row in df.iterrows():
        vals, raw_vals = _row_values(row, mapping)
        linkedin = vals.get("linkedin") or ""
        prepared.append(
            {
                "source_index": source_index,
                "vals": vals,
                "raw_vals": raw_vals,
                "normalized_li": normalize_linkedin(linkedin) if linkedin else "",
            }
        )
    return prepared


def _set_cached_match(
    cache: Dict[str, Dict[Any, int]],
    candidate_id: int,
    *,
    normalized_li: Optional[str],
    email: Optional[str],
    first_name: Optional[str],
    last_name: Optional[str],
    company_name: Optional[str],
) -> None:
    if normalized_li:
        cache["linkedin"][normalized_li] = int(candidate_id)
    if email:
        cache["email"][email] = int(candidate_id)
    identity = _identity_key(first_name, last_name, company_name)
    if identity:
        cache["identity"][identity] = int(candidate_id)


def _cached_match_id(
    cache: Dict[str, Dict[Any, int]],
    *,
    normalized_li: Optional[str],
    email: Optional[str],
    first_name: Optional[str],
    last_name: Optional[str],
    company_name: Optional[str],
) -> Optional[int]:
    if normalized_li and normalized_li in cache["linkedin"]:
        return cache["linkedin"][normalized_li]
    if email and email in cache["email"]:
        return cache["email"][email]
    identity = _identity_key(first_name, last_name, company_name)
    if identity and identity in cache["identity"]:
        return cache["identity"][identity]
    return None


def _prefetch_candidate_matches(
    cur,
    *,
    owner_user_id: Optional[int],
    prepared_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[Any, int]]:
    """Load one import batch's current dedupe candidates into local match maps."""
    cache: Dict[str, Dict[Any, int]] = {"linkedin": {}, "email": {}, "identity": {}}
    norms = sorted({row["normalized_li"] for row in prepared_rows if row["normalized_li"]})
    emails = sorted({row["email"] for row in prepared_rows if row.get("email")})
    identities = sorted(
        {
            identity
            for row in prepared_rows
            for identity in [
                _identity_key(
                    row["vals"].get("first_name"),
                    row["vals"].get("last_name"),
                    row["vals"].get("company_name"),
                )
            ]
            if identity
        }
    )
    owner_sql = "owner_user_id IS NULL" if owner_user_id is None else "owner_user_id = %s"
    owner_params: List[Any] = [] if owner_user_id is None else [owner_user_id]

    if norms:
        cur.execute(
            f"""
            SELECT id, normalized_linkedin
            FROM candidates
            WHERE {owner_sql}
              AND normalized_linkedin = ANY(%s)
              AND COALESCE(is_archived, FALSE) = FALSE
            ORDER BY id
            """,
            (*owner_params, norms),
        )
        for candidate_id, normalized_li in cur.fetchall():
            if normalized_li:
                cache["linkedin"].setdefault(normalized_li, int(candidate_id))

    if emails:
        cur.execute(
            f"""
            SELECT id, email
            FROM candidates
            WHERE {owner_sql}
              AND email = ANY(%s)
              AND COALESCE(is_archived, FALSE) = FALSE
            ORDER BY id
            """,
            (*owner_params, emails),
        )
        for candidate_id, email in cur.fetchall():
            if email:
                cache["email"].setdefault(email, int(candidate_id))

    if identities:
        first_names = [item[0] for item in identities]
        last_names = [item[1] for item in identities]
        companies = [item[2] for item in identities]
        cur.execute(
            f"""
            SELECT c.id, c.first_name, c.last_name, c.raw_fields->>'import_company'
            FROM candidates c
            JOIN unnest(%s::text[], %s::text[], %s::text[])
              AS wanted(first_name, last_name, company_name)
              ON c.first_name = wanted.first_name
             AND c.last_name = wanted.last_name
             AND c.raw_fields->>'import_company' = wanted.company_name
            WHERE c.{owner_sql}
              AND COALESCE(c.is_archived, FALSE) = FALSE
            ORDER BY c.id
            """,
            (first_names, last_names, companies, *owner_params),
        )
        for candidate_id, first_name, last_name, company_name in cur.fetchall():
            identity = _identity_key(first_name, last_name, company_name)
            if identity:
                cache["identity"].setdefault(identity, int(candidate_id))
    return cache


def _write_upload_progress(cur, upload_id: int, **updates: Any) -> None:
    if not updates:
        return
    fields = [f"{key} = %s" for key in updates]
    cur.execute(
        f"UPDATE candidate_uploads SET {', '.join(fields)} WHERE id = %s",
        (*updates.values(), upload_id),
    )


def _mark_upload_failed(upload_id: int, message: str) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            _write_upload_progress(
                cur,
                upload_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error_message=(message or "Import failed")[:8000],
            )
        conn.commit()
    finally:
        return_db_connection(conn)


def _refresh_import_caches(touched_ids: List[int]) -> None:
    started = perf_counter()
    unique_touched = sorted({int(item) for item in touched_ids if item is not None})
    try:
        from backend.api.routes.candidates import invalidate_candidate_count_caches

        invalidate_candidate_count_caches(refresh_profile_ids=unique_touched)
    except Exception:
        logger.exception("candidate upload could not invalidate count caches")
    logger.info(
        "candidate upload cache refresh touched=%s duration_ms=%s",
        len(unique_touched),
        round((perf_counter() - started) * 1000),
    )


def _upload_status_payload(row: Any) -> Dict[str, Any]:
    error_message = row[12] or ""
    return {
        "upload_id": row[0],
        "owner_user_id": row[1],
        "filename": row[2],
        "row_count": row[3] or 0,
        "processed_count": row[4] or 0,
        "inserted": row[5] or 0,
        "updated": row[6] or 0,
        "skipped": row[7] or 0,
        "role_assigned_count": row[8] or 0,
        "status": row[9] or "pending",
        "role_id": row[10],
        "created_at": row[11].isoformat() if row[11] else None,
        "error_message": error_message,
        "errors": [line for line in error_message.splitlines() if line][:20],
        "completed_at": row[13].isoformat() if row[13] else None,
    }


def _raw_fields_for_import(
    company_name: Optional[str], raw_fields_extra: Optional[Dict[str, Any]]
) -> str:
    raw_fields: Dict[str, Any] = {}
    if company_name:
        raw_fields["import_company"] = company_name
    if raw_fields_extra:
        raw_fields.update(raw_fields_extra)
    return json.dumps(raw_fields) if raw_fields else "{}"


def _fast_new_import_rows(
    prepared_rows: List[Dict[str, Any]],
    *,
    master_matches: Dict[str, Dict[Any, int]],
    recruiter_matches: Dict[str, Dict[Any, int]],
    user_role: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split truly independent new LinkedIn rows into the bulk-insert path."""
    dedupe_counts: Dict[tuple[str, Any], int] = {}
    for row in prepared_rows:
        vals = row["vals"]
        keys = [
            ("linkedin", row["normalized_li"]) if row["normalized_li"] else None,
            ("email", row.get("email")) if row.get("email") else None,
            (
                "identity",
                _identity_key(
                    vals.get("first_name"),
                    vals.get("last_name"),
                    vals.get("company_name"),
                ),
            ),
        ]
        for key in keys:
            if key and key[1]:
                dedupe_counts[key] = dedupe_counts.get(key, 0) + 1

    fast: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    for row in prepared_rows:
        vals = row["vals"]
        fn = vals.get("first_name") or ""
        ln = vals.get("last_name") or ""
        company_name = vals.get("company_name")
        identity = _identity_key(fn, ln, company_name)
        row_keys = [
            ("linkedin", row["normalized_li"]) if row["normalized_li"] else None,
            ("email", row.get("email")) if row.get("email") else None,
            ("identity", identity) if identity else None,
        ]
        has_existing = _cached_match_id(
            master_matches,
            normalized_li=row["normalized_li"],
            email=row.get("email"),
            first_name=fn,
            last_name=ln,
            company_name=company_name,
        )
        recruiter_existing = (
            _cached_match_id(
                recruiter_matches,
                normalized_li=row["normalized_li"],
                email=row.get("email"),
                first_name=fn,
                last_name=ln,
                company_name=company_name,
            )
            if user_role != "admin"
            else None
        )
        is_unique = all(dedupe_counts.get(key, 0) == 1 for key in row_keys if key)
        if row["normalized_li"] and not has_existing and not recruiter_existing and is_unique:
            fast.append(row)
        else:
            fallback.append(row)
    return fast, fallback


def _is_verified_enrichment_mode(enrichment_mode: Optional[str]) -> bool:
    return str(enrichment_mode or "none").strip().lower() == "verified_profile"


def _bulk_insert_new_import_rows(
    cur,
    *,
    rows: List[Dict[str, Any]],
    owner_user_id: int,
    user_role: str,
    upload_id: int,
    role_id: Optional[int],
) -> tuple[List[int], List[int], int]:
    """Insert new master and recruiter rows in two round-trips for the common CSV path."""
    if not rows:
        return [], [], 0

    master_values = []
    for row in rows:
        vals = row["vals"]
        fn = vals.get("first_name") or ""
        ln = vals.get("last_name") or ""
        city = vals.get("city")
        title = vals.get("title") or ""
        loc = vals.get("location") or city
        name = (f"{fn} {ln}").strip() or "Unknown"
        master_values.append(
            (
                name,
                fn,
                ln,
                vals.get("linkedin") or "",
                row["normalized_li"],
                city,
                title,
                loc,
                row.get("email"),
                row.get("phone"),
                _raw_fields_for_import(vals.get("company_name"), row["raw_vals"]),
            )
        )
    master_rows = execute_values(
        cur,
        """
        INSERT INTO candidates (
          name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
          location, email, mobile_phone, raw_fields, pool_source, owner_user_id, created_by
        ) VALUES %s
        RETURNING id, normalized_linkedin
        """,
        master_values,
        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s::jsonb,'{}'::jsonb),'catalog_from_upload',NULL,'catalog_upload')",
        fetch=True,
    )
    master_ids_by_norm = {row[1]: int(row[0]) for row in master_rows}
    master_ids = [master_ids_by_norm[row["normalized_li"]] for row in rows]

    if user_role == "admin":
        role_count = _bulk_assign_role(cur, role_id=role_id, candidate_ids=master_ids)
        return master_ids, master_ids, role_count

    recruiter_values = []
    for row, master_id in zip(rows, master_ids):
        vals = row["vals"]
        fn = vals.get("first_name") or ""
        ln = vals.get("last_name") or ""
        city = vals.get("city")
        title = vals.get("title") or ""
        loc = vals.get("location") or city
        name = (f"{fn} {ln}").strip() or "Unknown"
        recruiter_values.append(
            (
                name,
                fn,
                ln,
                vals.get("linkedin") or "",
                row["normalized_li"],
                city,
                title,
                loc,
                row.get("email"),
                row.get("phone"),
                vals.get("notes"),
                _raw_fields_for_import(vals.get("company_name"), row["raw_vals"]),
                owner_user_id,
                master_id,
                upload_id,
                str(owner_user_id),
            )
        )
    recruiter_rows = execute_values(
        cur,
        """
        INSERT INTO candidates (
          name, first_name, last_name, linkedin, normalized_linkedin, city, headline,
          location, email, mobile_phone, notes, raw_fields, owner_user_id, pool_source,
          source_master_candidate_id, source_upload_ids, assigned_by_user_id, created_by, status
        ) VALUES %s
        RETURNING id, normalized_linkedin
        """,
        recruiter_values,
        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s::jsonb,'{}'::jsonb),%s,'recruiter_upload',%s,ARRAY[%s::int],NULL,%s,'To be started')",
        fetch=True,
    )
    recruiter_ids_by_norm = {row[1]: int(row[0]) for row in recruiter_rows}
    recruiter_ids = [recruiter_ids_by_norm[row["normalized_li"]] for row in rows]
    role_count = _bulk_assign_role(cur, role_id=role_id, candidate_ids=recruiter_ids)
    return master_ids, recruiter_ids, role_count


def _bulk_assign_role(cur, *, role_id: Optional[int], candidate_ids: List[int]) -> int:
    if not role_id or not candidate_ids:
        return 0
    cur.execute(
        """
        INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
        SELECT %s, candidate_id, '--', ''
        FROM unnest(%s::int[]) AS candidate_id
        ON CONFLICT (role_id, candidate_id) DO NOTHING
        RETURNING candidate_id
        """,
        (role_id, candidate_ids),
    )
    return len(cur.fetchall())


def _process_upload_rows(
    *,
    upload_id: int,
    df: pd.DataFrame,
    mapping: Dict[str, str],
    owner_user_id: int,
    user_role: str,
    role_id: Optional[int],
    enrichment_mode: str = "none",
) -> None:
    inserted = updated = skipped = role_assigned_count = 0
    processed_count = 0
    errors: List[str] = []
    enrichment_errors: List[str] = []
    touched_ids: List[int] = []
    batch_size = 10
    import_started = perf_counter()
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        _mark_upload_failed(upload_id, "Database connection failed")
        return

    try:
        conn.rollback()
        with conn.cursor() as cur:
            prepare_started = perf_counter()
            prepared_rows = _prepare_import_rows(df, mapping)
            contact_started = perf_counter()
            contacts = fetch_best_contacts_for_normalized_lis(
                cur, [row["normalized_li"] for row in prepared_rows]
            )
            for row in prepared_rows:
                vals = row["vals"]
                contact_email, contact_phone = contacts.get(
                    row["normalized_li"], (None, None)
                )
                row["email"] = contact_email or vals.get("email")
                row["phone"] = contact_phone or vals.get("phone")
            match_started = perf_counter()
            master_matches = _prefetch_candidate_matches(
                cur,
                owner_user_id=None,
                prepared_rows=prepared_rows,
            )
            recruiter_matches = (
                _prefetch_candidate_matches(
                    cur,
                    owner_user_id=owner_user_id,
                    prepared_rows=prepared_rows,
                )
                if user_role != "admin"
                else {"linkedin": {}, "email": {}, "identity": {}}
            )
            logger.info(
                "candidate upload prefetch upload_id=%s rows=%s prepare_ms=%s contacts_ms=%s dedupe_ms=%s",
                upload_id,
                len(prepared_rows),
                round((contact_started - prepare_started) * 1000),
                round((match_started - contact_started) * 1000),
                round((perf_counter() - match_started) * 1000),
            )

            writes_started = perf_counter()
            fast_rows, fallback_rows = _fast_new_import_rows(
                prepared_rows,
                master_matches=master_matches,
                recruiter_matches=recruiter_matches,
                user_role=user_role,
            )
            if fast_rows:
                try:
                    cur.execute("SAVEPOINT imp_fast_new")
                    fast_master_ids, fast_candidate_ids, fast_role_count = _bulk_insert_new_import_rows(
                        cur,
                        rows=fast_rows,
                        owner_user_id=owner_user_id,
                        user_role=user_role,
                        upload_id=upload_id,
                        role_id=role_id,
                    )
                    cur.execute("RELEASE SAVEPOINT imp_fast_new")
                    processed_count += len(fast_rows)
                    inserted += len(fast_rows)
                    role_assigned_count += fast_role_count
                    touched_ids.extend(fast_master_ids)
                    touched_ids.extend(fast_candidate_ids)
                    _write_upload_progress(
                        cur,
                        upload_id,
                        processed_count=processed_count,
                        inserted_count=inserted,
                        updated_count=updated,
                        skipped_count=skipped,
                        role_assigned_count=role_assigned_count,
                    )
                    conn.commit()
                except Exception:
                    logger.exception(
                        "candidate upload bulk insert failed; retrying row path upload_id=%s rows=%s",
                        upload_id,
                        len(fast_rows),
                    )
                    cur.execute("ROLLBACK TO SAVEPOINT imp_fast_new")
                    cur.execute("RELEASE SAVEPOINT imp_fast_new")
                    fallback_rows = fast_rows + fallback_rows

            for sequence, prepared in enumerate(fallback_rows, start=processed_count + 1):
                vals = prepared["vals"]
                raw_vals = prepared["raw_vals"]
                fn = vals.get("first_name") or ""
                ln = vals.get("last_name") or ""
                li = vals.get("linkedin") or ""
                norm = prepared["normalized_li"]
                email = prepared.get("email")
                phone = prepared.get("phone")
                company_name = vals.get("company_name")
                processed_count = sequence

                if not norm and not email and not _identity_key(fn, ln, company_name):
                    skipped += 1
                else:
                    savepoint = f"imp_row_{sequence}"
                    try:
                        # One bad row must not abort rows already processed for this import.
                        cur.execute(f"SAVEPOINT {savepoint}")
                        master_existing_id = _cached_match_id(
                            master_matches,
                            normalized_li=norm,
                            email=email,
                            first_name=fn,
                            last_name=ln,
                            company_name=company_name,
                        )
                        mid = upsert_master_catalog_row(
                            cur,
                            normalized_li=norm,
                            raw_linkedin=li,
                            first_name=fn,
                            last_name=ln,
                            city=vals.get("city"),
                            title=vals.get("title"),
                            company_name=company_name,
                            email=email,
                            phone=phone,
                            location=vals.get("location"),
                            notes=None,
                            raw_fields_extra=raw_vals,
                            existing_id=master_existing_id,
                            lookup_complete=True,
                        )
                        _set_cached_match(
                            master_matches,
                            int(mid),
                            normalized_li=norm,
                            email=email,
                            first_name=fn,
                            last_name=ln,
                            company_name=company_name,
                        )
                        cid = mid
                        op = "updated" if master_existing_id else "inserted"
                        if user_role != "admin":
                            recruiter_existing_id = _cached_match_id(
                                recruiter_matches,
                                normalized_li=norm,
                                email=email,
                                first_name=fn,
                                last_name=ln,
                                company_name=company_name,
                            )
                            cid, op = upsert_recruiter_pool_row(
                                cur,
                                owner_id=owner_user_id,
                                master_id=mid,
                                normalized_li=norm,
                                raw_linkedin=li,
                                first_name=fn,
                                last_name=ln,
                                city=vals.get("city"),
                                title=vals.get("title"),
                                company_name=company_name,
                                email=email,
                                phone=phone,
                                location=vals.get("location"),
                                notes=vals.get("notes"),
                                pool_source=POOL_SOURCE_RECRUITER_UPLOAD,
                                source_upload_id=upload_id,
                                assigned_by_user_id=None,
                                raw_fields_extra=raw_vals,
                                existing_id=recruiter_existing_id,
                                lookup_complete=True,
                            )
                            _set_cached_match(
                                recruiter_matches,
                                int(cid),
                                normalized_li=norm,
                                email=email,
                                first_name=fn,
                                last_name=ln,
                                company_name=company_name,
                            )
                        if _assign_imported_candidate_to_role(
                            cur, role_id=role_id, candidate_id=int(cid)
                        ):
                            role_assigned_count += 1
                        touched_ids.extend([int(mid), int(cid)])
                        if norm:
                            contacts[norm] = (email, phone)
                        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        if op == "inserted":
                            inserted += 1
                        else:
                            updated += 1
                    except Exception as row_exc:
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                            cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        except Exception:
                            conn.rollback()
                            raise
                        errors.append(f"row {prepared['source_index']}: {row_exc}")
                        skipped += 1

                if sequence % batch_size == 0 or sequence == len(prepared_rows):
                    _write_upload_progress(
                        cur,
                        upload_id,
                        processed_count=processed_count,
                        inserted_count=inserted,
                        updated_count=updated,
                        skipped_count=skipped,
                        role_assigned_count=role_assigned_count,
                        error_message="\n".join(errors[:20]) if errors else None,
                    )
                    conn.commit()

            with conn.cursor() as cur:
                _write_upload_progress(
                    cur,
                    upload_id,
                    processed_count=processed_count,
                    inserted_count=inserted,
                    updated_count=updated,
                    skipped_count=skipped,
                    role_assigned_count=role_assigned_count,
                    status="refreshing",
                    error_message="\n".join(errors[:20]) if errors else None,
                )
            conn.commit()
            logger.info(
                "candidate upload writes upload_id=%s rows=%s inserted=%s updated=%s skipped=%s duration_ms=%s",
                upload_id,
                len(prepared_rows),
                inserted,
                updated,
                skipped,
                round((perf_counter() - writes_started) * 1000),
            )
    except Exception as exc:
        conn.rollback()
        logger.exception("candidate upload worker failed upload_id=%s", upload_id)
        _mark_upload_failed(upload_id, str(exc))
        return
    finally:
        return_db_connection(conn)

    _refresh_import_caches(touched_ids)
    if _is_verified_enrichment_mode(enrichment_mode):
        conn = get_db_connection(validate=False, register_pgvector=False)
        if conn:
            try:
                with conn.cursor() as cur:
                    _write_upload_progress(
                        cur,
                        upload_id,
                        status="enriching",
                        error_message="\n".join((errors + enrichment_errors)[:20]) if (errors or enrichment_errors) else None,
                    )
                conn.commit()
            finally:
                return_db_connection(conn)
        try:
            from backend.services.import_enrichment import enrich_candidate_profiles

            enrichment_result = enrich_candidate_profiles(touched_ids, allow_web=True)
            enrichment_errors.extend(enrichment_result.get("errors") or [])
        except Exception as enrich_exc:
            logger.exception("candidate upload enrichment failed upload_id=%s", upload_id)
            enrichment_errors.append(f"enrichment: {enrich_exc}")
        _refresh_import_caches(touched_ids)

    final_status = "completed_with_errors" if errors else "completed"
    if enrichment_errors:
        final_status = "completed_with_errors"
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        _mark_upload_failed(upload_id, "Database connection failed while finalizing import")
        return
    try:
        with conn.cursor() as cur:
            _write_upload_progress(
                cur,
                upload_id,
                processed_count=processed_count,
                inserted_count=inserted,
                updated_count=updated,
                skipped_count=skipped,
                role_assigned_count=role_assigned_count,
                status=final_status,
                completed_at=datetime.now(timezone.utc),
                error_message="\n".join((errors + enrichment_errors)[:20]) if (errors or enrichment_errors) else None,
            )
        conn.commit()
        logger.info(
            "candidate upload finished upload_id=%s status=%s total_ms=%s",
            upload_id,
            final_status,
            round((perf_counter() - import_started) * 1000),
        )
    finally:
        return_db_connection(conn)


def _spawn_upload_thread(
    *,
    upload_id: int,
    df: pd.DataFrame,
    mapping: Dict[str, str],
    owner_user_id: int,
    user_role: str,
    role_id: Optional[int],
    enrichment_mode: str = "none",
) -> None:
    def run() -> None:
        try:
            _process_upload_rows(
                upload_id=upload_id,
                df=df,
                mapping=mapping,
                owner_user_id=owner_user_id,
                user_role=user_role,
                role_id=role_id,
                enrichment_mode=enrichment_mode,
            )
        finally:
            with _upload_threads_lock:
                _upload_threads.pop(upload_id, None)

    worker = threading.Thread(
        target=run,
        daemon=True,
        name=f"candidate-upload-{upload_id}",
    )
    with _upload_threads_lock:
        _upload_threads[upload_id] = worker
    worker.start()


@router.post("/candidates/upload/preview")
async def upload_preview(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if current_user.role == "admin":
        raise HTTPException(status_code=403, detail="Admins import via recruiter accounts")
    return await build_upload_preview_response(file, use_llm=use_llm)


@router.post("/candidates/upload/suggest-mapping")
async def suggest_mapping_llm(
    body: SuggestMappingBody,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if current_user.role == "admin":
        raise HTTPException(status_code=403, detail="Recruiter only")
    headers = body.headers
    base = suggest_header_mapping(headers)
    if not headers:
        return {"suggested_mapping": base}
    try:
        from langchain_openai import ChatOpenAI
        from backend.pipeline.query import safe_json_loads

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            "Map spreadsheet column headers to canonical recruitment fields. "
            f"Headers JSON: {json.dumps(headers)}. "
            "Reply ONLY JSON object mapping each header string to one of: "
            "first_name, last_name, linkedin, city, title, company_name, email, phone, "
            "location, notes, headline, about, or ignore."
        )
        resp = await llm.ainvoke(prompt)
        llm_map = safe_json_loads(resp.content, {})
        if isinstance(llm_map, dict):
            for k, v in llm_map.items():
                if k in headers and isinstance(v, str):
                    base[k] = v
    except Exception as e:
        logger.warning("LLM mapping failed: %s", e)
    return {"suggested_mapping": base}


async def commit_upload_file(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    enrichment_mode: str = Form("none"),
    current_user: schemas.User = None,
    role_id: Optional[int] = None,
) -> Dict[str, Any]:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        mapping = json.loads(mapping_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid mapping JSON")

    mapping = _validate_mapping(mapping)
    enrichment_mode = "verified_profile" if _is_verified_enrichment_mode(enrichment_mode) else "none"

    parse_started = perf_counter()
    raw = await file.read()
    try:
        df = _read_frame(raw, file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    row_count_total = int(len(df.index))
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidate_uploads (
                  owner_user_id, filename, file_headers, mapping, row_count,
                  processed_count, status, role_id
                ) VALUES (%s, %s, %s, %s, %s, 0, 'processing', %s)
                RETURNING id
                """,
                (
                    current_user.id,
                    file.filename or "upload",
                    json.dumps(list(df.columns)),
                    json.dumps(mapping),
                    len(df.index),
                    role_id,
                ),
            )
            upload_row = cur.fetchone()
            upload_id = upload_row[0]
            conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.exception("upload commit failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)

    logger.info(
        "candidate upload accepted upload_id=%s rows=%s parse_ms=%s",
        upload_id,
        row_count_total,
        round((perf_counter() - parse_started) * 1000),
    )
    _spawn_upload_thread(
        upload_id=upload_id,
        df=df,
        mapping=mapping,
        owner_user_id=int(current_user.id),
        user_role=(current_user.role or "").strip().lower(),
        role_id=role_id,
        enrichment_mode=enrichment_mode,
    )

    return {
        "upload_id": upload_id,
        "row_count": row_count_total,
        "processed_count": 0,
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "role_assigned_count": 0,
        "status": "processing",
        "enrichment_mode": enrichment_mode,
        "errors": [],
    }


@router.post("/candidates/upload/commit")
async def upload_commit(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    enrichment_mode: str = Form("none"),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if current_user.role == "admin":
        raise HTTPException(status_code=403, detail="Recruiter only")
    return await commit_upload_file(
        file=file,
        mapping_json=mapping_json,
        enrichment_mode=enrichment_mode,
        current_user=current_user,
    )


@router.get("/candidates/uploads")
async def list_uploads(
    current_user: schemas.User = Depends(deps.get_current_user),
    limit: int = 20,
):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            if current_user.role == "admin":
                cur.execute(
                    """
                    SELECT id, owner_user_id, filename, row_count, processed_count,
                           inserted_count, updated_count, skipped_count, role_assigned_count,
                           status, created_at, completed_at, role_id
                    FROM candidate_uploads
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, owner_user_id, filename, row_count, processed_count,
                           inserted_count, updated_count, skipped_count, role_assigned_count,
                           status, created_at, completed_at, role_id
                    FROM candidate_uploads
                    WHERE owner_user_id = %s
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (current_user.id, limit),
                )
            rows = cur.fetchall()
        return {
            "uploads": [
                {
                    "id": r[0],
                    "owner_user_id": r[1],
                    "filename": r[2],
                    "row_count": r[3],
                    "processed_count": r[4] if r[4] is not None else 0,
                    "inserted_count": r[5],
                    "updated_count": r[6],
                    "skipped_count": r[7] if r[7] is not None else 0,
                    "role_assigned_count": r[8] if r[8] is not None else 0,
                    "status": r[9],
                    "created_at": r[10].isoformat() if r[10] else None,
                    "completed_at": r[11].isoformat() if r[11] else None,
                    "role_id": r[12],
                }
                for r in rows
            ]
        }
    finally:
        return_db_connection(conn)


@router.get("/candidates/uploads/{upload_id}")
async def get_upload_status(
    upload_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_user_id, filename, row_count, processed_count,
                       inserted_count, updated_count, skipped_count, role_assigned_count,
                       status, role_id, created_at, error_message, completed_at
                FROM candidate_uploads
                WHERE id = %s
                """,
                (upload_id,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Upload not found")
        if current_user.role != "admin" and int(row[1]) != int(current_user.id):
            raise HTTPException(status_code=404, detail="Upload not found")
        return _upload_status_payload(row)
    finally:
        return_db_connection(conn)

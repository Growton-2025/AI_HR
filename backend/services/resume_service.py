"""Resume lifecycle: create → background parse → fill-gaps write-back.

The write-back deliberately does NOT reuse import_enrichment's
_persist_candidate_enrichment — that function deletes and rebuilds the
candidate's roles/education/experience child tables, which would wipe
LinkedIn-enriched history. A resume is allowed to fill empty fields only;
conflicts are recorded in candidate_resumes.proposed_changes for the UI.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from backend.db.connection import get_db_connection, return_db_connection
from backend.services import resume_parse
from backend.services.resume_storage import get_resume_store

logger = logging.getLogger(__name__)

RESUME_MAX_MB = int(os.getenv("RESUME_MAX_MB", "10"))

# Bounded: each parse thread holds a DB connection at times and DB_POOL_MAX
# is small (default 8). Unbounded threads here can starve the whole app.
_parse_semaphore = threading.BoundedSemaphore(int(os.getenv("RESUME_MAX_CONCURRENT_PARSES", "3")))
_parse_threads: Dict[int, threading.Thread] = {}
_parse_threads_lock = threading.Lock()

_META_COLUMNS = (
    "id, candidate_id, filename, content_type, size_bytes, storage_backend, storage_key, "
    "parse_status, parse_error, summary, proposed_changes, applied_fields, text_char_count, "
    "created_at, parsed_at, uploaded_by_user_id, parsed_json"
)


def _meta_row_to_dict(row) -> Dict[str, Any]:
    def _jsonish(value, default):
        if value is None:
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    return {
        "id": row[0],
        "candidate_id": row[1],
        "filename": row[2],
        "content_type": row[3],
        "size_bytes": row[4],
        "storage_backend": row[5],
        "storage_key": row[6],
        "parse_status": row[7],
        "parse_error": row[8],
        "summary": row[9],
        "proposed_changes": _jsonish(row[10], []),
        "applied_fields": _jsonish(row[11], []),
        "text_char_count": row[12] or 0,
        "uploaded_at": row[13].isoformat() if row[13] else None,
        "parsed_at": row[14].isoformat() if row[14] else None,
        "uploaded_by_user_id": row[15],
        "parsed_json": _jsonish(row[16], {}),
    }


def create_resume(
    candidate_id: int,
    *,
    filename: str,
    content_type: str,
    data: bytes,
    uploaded_by_user_id: Optional[int],
) -> Dict[str, Any]:
    """Insert the row, store the bytes, retire any previous current resume."""
    checksum = hashlib.sha256(data).hexdigest()
    store = get_resume_store()
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE candidate_resumes SET is_current = FALSE, updated_at = NOW() "
                "WHERE candidate_id = %s AND is_current",
                (candidate_id,),
            )
            cur.execute(
                f"""
                INSERT INTO candidate_resumes
                    (candidate_id, filename, content_type, size_bytes, checksum_sha256,
                     storage_backend, parse_status, uploaded_by_user_id)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s)
                RETURNING {_META_COLUMNS}
                """,
                (candidate_id, filename[:512], content_type[:255], len(data),
                 checksum, store.backend_name, uploaded_by_user_id),
            )
            row = cur.fetchone()
        conn.commit()
    finally:
        return_db_connection(conn)

    resume_id = row[0]
    storage_key = store.put(
        resume_id=resume_id, candidate_id=candidate_id,
        filename=filename, content_type=content_type, data=data,
    )
    _update_resume(resume_id, storage_key=storage_key)
    meta = _meta_row_to_dict(row)
    meta["storage_key"] = storage_key
    return meta


def _update_resume(resume_id: int, **fields: Any) -> None:
    if not fields:
        return
    json_fields = {"parsed_json", "proposed_changes", "applied_fields"}
    sets, params = [], []
    for key, value in fields.items():
        if key in json_fields:
            sets.append(f"{key} = %s::jsonb")
            params.append(json.dumps(value, ensure_ascii=False, default=str))
        else:
            sets.append(f"{key} = %s")
            params.append(value)
    sets.append("updated_at = NOW()")
    params.append(resume_id)
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE candidate_resumes SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
    finally:
        return_db_connection(conn)


def fetch_resume_meta(candidate_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_META_COLUMNS} FROM candidate_resumes "
                "WHERE candidate_id = %s AND is_current LIMIT 1",
                (candidate_id,),
            )
            row = cur.fetchone()
        return _meta_row_to_dict(row) if row else None
    finally:
        return_db_connection(conn)


def fetch_resume_metas(candidate_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Batch metadata for grid hydration — one query, no bytes, no text."""
    ids = sorted({int(i) for i in candidate_ids if i is not None})
    if not ids:
        return {}
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT candidate_id, id, filename, content_type, size_bytes, parse_status, created_at "
                "FROM candidate_resumes WHERE candidate_id = ANY(%s) AND is_current",
                (ids,),
            )
            rows = cur.fetchall()
        return {
            row[0]: {
                "id": row[1],
                "filename": row[2],
                "content_type": row[3],
                "size_bytes": row[4],
                "parse_status": row[5],
                "uploaded_at": row[6].isoformat() if row[6] else None,
            }
            for row in rows
        }
    except Exception as exc:
        logger.warning("fetch_resume_metas failed (resume column will render empty): %s", exc)
        return {}
    finally:
        return_db_connection(conn)


def fetch_resume_text(candidate_id: int) -> str:
    conn = get_db_connection()
    if not conn:
        return ""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT extracted_text FROM candidate_resumes "
                "WHERE candidate_id = %s AND is_current LIMIT 1",
                (candidate_id,),
            )
            row = cur.fetchone()
        return row[0] or "" if row else ""
    finally:
        return_db_connection(conn)


def fetch_resume_file(candidate_id: int) -> Optional[Tuple[Dict[str, Any], bytes]]:
    meta = fetch_resume_meta(candidate_id)
    if not meta:
        return None
    store = get_resume_store(meta.get("storage_backend"))
    data = store.get(resume_id=meta["id"], storage_key=meta.get("storage_key"))
    return meta, data


# ---------------------------------------------------------------------------
# Fill-gaps write-back


def _current_candidate_state(cur, candidate_id: int) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        SELECT first_name, last_name, email, phone, mobile_phone, linkedin, city, location,
               headline, about, skills, licenses_and_certifications,
               total_experience_years, avg_years_in_company, max_people_managed,
               years_team_management, raw_fields,
               COALESCE(email_locked_by_user, FALSE), COALESCE(mobile_phone_locked_by_user, FALSE)
        FROM candidates WHERE id = %s
        """,
        (candidate_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    keys = (
        "first_name", "last_name", "email", "phone", "mobile_phone", "linkedin", "city",
        "location", "headline", "about", "skills", "licenses_and_certifications",
        "total_experience_years", "avg_years_in_company", "max_people_managed",
        "years_team_management", "raw_fields", "email_locked", "mobile_phone_locked",
    )
    return dict(zip(keys, row))


def apply_resume_to_candidate(
    candidate_id: int, parsed: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fill empty candidate fields from the parse. Returns (applied, proposed)."""
    text_fields = {
        "first_name": parsed.get("first_name"),
        "last_name": parsed.get("last_name"),
        "email": parsed.get("email"),
        "phone": parsed.get("phone"),
        "mobile_phone": parsed.get("phone"),
        "linkedin": parsed.get("linkedin"),
        "city": parsed.get("city"),
        "location": parsed.get("location"),
        "headline": parsed.get("headline"),
        "about": parsed.get("summary"),
        "skills": ", ".join(parsed.get("skills") or []) or None,
        "licenses_and_certifications": ", ".join(parsed.get("certifications") or []) or None,
    }
    numeric_fields = {
        "total_experience_years": parsed.get("total_experience_years"),
        "avg_years_in_company": parsed.get("avg_years_in_company"),
        "max_people_managed": parsed.get("max_people_managed"),
        "years_team_management": parsed.get("years_team_management"),
    }

    applied: List[Dict[str, Any]] = []
    proposed: List[Dict[str, Any]] = []
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            current = _current_candidate_state(cur, candidate_id)
            if current is None:
                raise ValueError(f"Candidate {candidate_id} not found")

            sets, params = [], []
            for column, resume_value in text_fields.items():
                value = str(resume_value or "").strip()
                if not value:
                    continue
                if column == "email" and current["email_locked"]:
                    continue
                if column == "mobile_phone" and current["mobile_phone_locked"]:
                    continue
                existing = str(current.get(column) or "").strip()
                if existing:
                    if existing.lower() != value.lower():
                        proposed.append({"field": column, "current_value": existing, "resume_value": value})
                    continue
                sets.append(f"{column} = %s")
                params.append(value)
                applied.append({"field": column, "value": value})

            for column, resume_value in numeric_fields.items():
                if resume_value is None:
                    continue
                existing = current.get(column)
                if existing is not None and float(existing) != 0.0:
                    if abs(float(existing) - float(resume_value)) > 0.05:
                        proposed.append(
                            {"field": column, "current_value": float(existing), "resume_value": float(resume_value)}
                        )
                    continue
                sets.append(f"{column} = %s")
                params.append(resume_value)
                applied.append({"field": column, "value": resume_value})

            # Merge one top-level raw_fields key; never touch raw_fields['enrichment'].
            raw = current.get("raw_fields")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            raw = raw if isinstance(raw, dict) else {}
            raw["resume"] = {
                "headline": parsed.get("headline") or "",
                "summary": parsed.get("summary") or "",
                "skills": parsed.get("skills") or [],
                "confidence": parsed.get("confidence") or "low",
            }
            sets.append("raw_fields = %s::jsonb")
            params.append(json.dumps(raw, ensure_ascii=False, default=str))
            sets.append("updated_at = NOW()")

            params.append(candidate_id)
            cur.execute(f"UPDATE candidates SET {', '.join(sets)} WHERE id = %s", params)

            _seed_children_if_empty(cur, candidate_id, parsed, applied)
        conn.commit()
        return applied, proposed
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_connection(conn)


def _seed_children_if_empty(cur, candidate_id: int, parsed: Dict[str, Any], applied: List[Dict[str, Any]]) -> None:
    """Seed roles/education from the resume ONLY when the candidate has none.

    Never deletes or rewrites existing rows — enriched history is untouchable.
    """
    cur.execute("SELECT COUNT(*) FROM roles WHERE candidate_id = %s", (candidate_id,))
    role_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM education WHERE candidate_id = %s", (candidate_id,))
    education_count = cur.fetchone()[0]
    if role_count or education_count:
        return

    seeded_roles = 0
    for role in (parsed.get("roles") or [])[:12]:
        company = role.get("company") or ""
        title = role.get("title") or ""
        if not company:
            continue
        cur.execute(
            "INSERT INTO companies (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (company[:255],),
        )
        company_id = cur.fetchone()[0]
        start = resume_parse.parse_profile_date(role.get("start_date"))
        end = resume_parse.parse_profile_date(role.get("end_date"), default_current=bool(role.get("is_current")))
        duration = resume_parse.years_from_months(resume_parse.months_between(start, end))
        cur.execute(
            "INSERT INTO roles (candidate_id, company_id, title, details, duration_years) VALUES (%s, %s, %s, %s, %s)",
            (candidate_id, company_id, title[:255], (role.get("description") or "")[:2000], duration or None),
        )
        seeded_roles += 1

    seeded_education = 0
    for entry in (parsed.get("education") or [])[:8]:
        institution = entry.get("institution") or ""
        if not institution:
            continue
        start = resume_parse.parse_profile_date(entry.get("start_date"))
        end = resume_parse.parse_profile_date(entry.get("end_date"))
        cur.execute(
            "INSERT INTO education (candidate_id, college, degree, start_date, end_date, details) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                candidate_id,
                institution[:255],
                (entry.get("degree") or "")[:255],
                start.date() if start else None,
                end.date() if end else None,
                (entry.get("field") or "")[:500] or None,
            ),
        )
        seeded_education += 1

    if seeded_roles:
        applied.append({"field": "roles", "value": f"seeded {seeded_roles} role(s) from resume"})
    if seeded_education:
        applied.append({"field": "education", "value": f"seeded {seeded_education} education entrie(s) from resume"})


# ---------------------------------------------------------------------------
# Background processing


def process_resume(resume_id: int, candidate_id: int, *, apply_fields: bool = True) -> None:
    with _parse_semaphore:
        model = os.getenv("RESUME_PARSE_OPENAI_MODEL", "gpt-4o-mini")
        try:
            meta = None
            conn = get_db_connection()
            if not conn:
                raise RuntimeError("Database connection failed")
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename, storage_backend, storage_key FROM candidate_resumes WHERE id = %s",
                        (resume_id,),
                    )
                    meta = cur.fetchone()
            finally:
                return_db_connection(conn)
            if not meta:
                raise ValueError(f"Resume {resume_id} not found")
            filename, storage_backend, storage_key = meta

            _update_resume(resume_id, parse_status="extracting")
            data = get_resume_store(storage_backend).get(resume_id=resume_id, storage_key=storage_key)
            text, text_status = resume_parse.extract_text(data, filename)
            _update_resume(resume_id, extracted_text=text, text_char_count=len(text))

            if text_status == "low_text":
                _update_resume(resume_id, parse_status="low_text", parse_model=model)
                return

            _update_resume(resume_id, parse_status="parsing")
            parsed = resume_parse.parse_resume(text)
            parsed = resume_parse.recompute_totals(parsed)
            summary = resume_parse.summarize_resume(parsed)

            applied: List[Dict[str, Any]] = []
            proposed: List[Dict[str, Any]] = []
            if apply_fields:
                applied, proposed = apply_resume_to_candidate(candidate_id, parsed)

            from datetime import datetime, timezone

            _update_resume(
                resume_id,
                parsed_json=parsed,
                summary=summary,
                applied_fields=applied,
                proposed_changes=proposed,
                parse_status="complete",
                parse_error=None,
                parse_model=model,
                parsed_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.exception("Resume %s parse failed", resume_id)
            try:
                _update_resume(resume_id, parse_status="failed", parse_error=str(exc)[:2000])
            except Exception:
                logger.exception("Resume %s could not record failure", resume_id)
        finally:
            # Without this the in-memory profile cache is stale and Smart AI
            # columns never see the resume until a process restart.
            try:
                from backend.pipeline import query

                query.refresh_profiles_in_cache([candidate_id])
            except Exception:
                logger.exception("Resume %s: profile cache refresh failed", resume_id)


def spawn_parse_thread(resume_id: int, candidate_id: int) -> None:
    with _parse_threads_lock:
        existing = _parse_threads.get(resume_id)
        if existing and existing.is_alive():
            return
        thread = threading.Thread(
            target=process_resume,
            args=(resume_id, candidate_id),
            name=f"resume-parse-{resume_id}",
            daemon=True,
        )
        _parse_threads[resume_id] = thread
        thread.start()

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List
import re
import json
import os
import logging

from openai import OpenAI

from backend.api import deps, schemas
from backend.db.connection import get_db_connection_context
from backend.services.clay import trigger_clay
from backend.services.linkedin_normalize import normalize_linkedin
from backend.services.candidate_pool import (
    fetch_best_contact_for_normalized_li,
    assert_admin_or_recruiter_owner,
)
from backend.pipeline import query

router = APIRouter()
logger = logging.getLogger(__name__)

# ─── OpenAI client for AI Column features ────────────────────────────────────
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class AiColumnTestRequest(BaseModel):
    candidate_id:    int
    prompt_template: str
    use_web_search:  bool = False

class AiColumnRunRequest(BaseModel):
    candidate_ids:   List[int]
    prompt_template: str
    column_name:     str
    use_web_search:  bool = False

# ─── Helper: resolve a candidate dict from PROFILES_BY_ID ────────────────────
_CANDIDATE_FIELD_MAP = {
    "first_name":             "first_name",
    "last_name":              "last_name",
    "name":                   "name",
    "company":                "company",
    "title":                  "title",
    "headline":               "headline",
    "linkedin":               "linkedin",
    "location":               "location",
    "city":                   "city",
    "email":                  "email",
    "phone":                  "phone",
    "about":                  "about",
    "total_experience_years": "total_experience_years",
}

def _build_candidate_context(profile: dict) -> dict:
    """Build a flat context dict from a candidate profile for prompt injection."""
    ctx = {}
    for token, field in _CANDIDATE_FIELD_MAP.items():
        val = profile.get(field)
        # For company, try the most-recent role first
        if field == "company" and not val:
            roles = profile.get("roles", [])
            val = roles[0].get("company", "") if roles else ""
        if field == "title" and not val:
            roles = profile.get("roles", [])
            val = roles[0].get("title", "") if roles else ""
        ctx[token] = str(val or "N/A")
    return ctx

def _fill_prompt(template: str, ctx: dict) -> str:
    """Replace {variable} tokens in the template with actual values."""
    result = template
    for key, value in ctx.items():
        result = result.replace(f"{{{key}}}", value)
    return result

def _call_openai(filled_prompt: str, use_web_search: bool) -> str:
    """
    Call OpenAI with or without live web search using the Responses API.
    Returns the model's output as a plain string.
    """
    try:
        if use_web_search:
            response = _openai_client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search_preview"}],
                input=filled_prompt,
            )
            return response.output_text
        else:
            response = _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": filled_prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        raise

def _update_ai_column_in_db(candidate_id: int, column_name: str, value: str):
    """Persist a single AI column result into candidates.raw_fields->ai_columns."""
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            logger.error("No DB connection for ai_column update candidate=%s", candidate_id)
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE candidates
                    SET raw_fields = jsonb_set(
                        jsonb_set(
                            COALESCE(raw_fields, '{}'::jsonb),
                            ARRAY['ai_columns'],
                            COALESCE(raw_fields->'ai_columns', '{}'::jsonb),
                            true
                        ),
                        ARRAY['ai_columns', %s],
                        to_jsonb(%s::text),
                        true
                    ),
                    updated_at = NOW()
                    WHERE id = %s
                    """,
                    (column_name, value, candidate_id),
                )
            conn.commit()
        except Exception as e:
            logger.error("DB update failed for ai_column candidate=%s: %s", candidate_id, e)
            raise


def _fetch_profile_from_db(candidate_id: int) -> dict | None:
    """Fallback: fetch a single candidate profile directly from DB if not in cache."""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (c.id)
                        c.id, c.name, c.linkedin, c.headline, c.about,
                        c.location, c.city,
                        c.first_name, c.last_name,
                        c.total_experience_years,
                        c.raw_fields,
                        r.title AS role_title,
                        co.name AS role_company
                    FROM candidates c
                    LEFT JOIN roles r ON r.candidate_id = c.id
                    LEFT JOIN companies co ON co.id = r.company_id
                    WHERE c.id = %s
                    ORDER BY c.id, r.id ASC
                    LIMIT 1
                """, (candidate_id,))
                row = cur.fetchone()
                if not row:
                    return None
                raw_fields = row[10]
                if isinstance(raw_fields, str):
                    try:
                        raw_fields = json.loads(raw_fields)
                    except Exception:
                        raw_fields = {}
                first_name = row[7] or (row[1] or "").split()[0] if row[1] else ""
                last_name  = row[8] or (" ".join((row[1] or "").split()[1:])) if row[1] else ""
                return {
                    "id":                     row[0],
                    "name":                   row[1] or "",
                    "linkedin":               row[2] or "",
                    "headline":               row[3] or "",
                    "about":                  row[4] or "",
                    "location":               row[5] or "",
                    "city":                   row[6] or "",
                    "first_name":             first_name,
                    "last_name":              last_name,
                    "total_experience_years": float(row[9]) if row[9] else 0.0,
                    "raw_fields":             raw_fields if isinstance(raw_fields, dict) else {},
                    "title":                  row[11] or row[3] or "",  # role_title or headline
                    "company":                row[12] or "",
                    "email":                  (raw_fields or {}).get("email", "") if isinstance(raw_fields, dict) else "",
                    "phone":                  (raw_fields or {}).get("phone", "") if isinstance(raw_fields, dict) else "",
                    "status":                 (raw_fields or {}).get("status", "") if isinstance(raw_fields, dict) else "",
                }
    except Exception as e:
        logger.error("_fetch_profile_from_db failed for candidate %s: %s", candidate_id, e)
        return None



def _run_ai_column_batch(candidate_ids: List[int], prompt_template: str, column_name: str, use_web_search: bool):
    """Background task: iterate ALL candidates, falling back to DB if not in cache."""
    total = len(candidate_ids)
    done = 0
    skipped = 0
    failed = 0
    for cid in candidate_ids:
        try:
            # Try in-memory cache first (fast), fall back to direct DB fetch
            profile = query.PROFILES_BY_ID.get(cid) or _fetch_profile_from_db(cid)
            if not profile:
                logger.warning("ai_column: candidate %s not found in cache or DB, skipping", cid)
                skipped += 1
                continue
            ctx = _build_candidate_context(profile)
            filled = _fill_prompt(prompt_template, ctx)
            result = _call_openai(filled, use_web_search)
            _update_ai_column_in_db(cid, column_name, result)
            query.refresh_profiles_in_cache([cid])
            done += 1
            logger.info("ai_column '%s' written [%d/%d] candidate=%s", column_name, done, total, cid)
        except Exception as e:
            failed += 1
            logger.error("ai_column batch failed for candidate %s: %s", cid, e)
    logger.info("ai_column '%s' batch complete: done=%d skipped=%d failed=%d",
                column_name, done, skipped, failed)


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ai-column/test")
async def ai_column_test(
    body: AiColumnTestRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Test an AI column prompt against a single candidate and return the result."""
    # Try cache first, fall back to direct DB fetch
    profile = query.PROFILES_BY_ID.get(body.candidate_id) or _fetch_profile_from_db(body.candidate_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate not found in cache or DB")

    ctx = _build_candidate_context(profile)
    filled = _fill_prompt(body.prompt_template, ctx)
    logger.info("ai_column test — user=%s cand=%s web=%s prompt=%s…",
                current_user.id, body.candidate_id, body.use_web_search, filled[:80])

    try:
        result = _call_openai(filled, body.use_web_search)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    return {"result": result, "filled_prompt": filled}


@router.post("/ai-column/run")
async def ai_column_run(
    body: AiColumnRunRequest,
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Kick off a batch AI column job in the background."""
    if not body.column_name.strip():
        raise HTTPException(status_code=400, detail="column_name is required")
    if not body.prompt_template.strip():
        raise HTTPException(status_code=400, detail="prompt_template is required")
    if not body.candidate_ids:
        raise HTTPException(status_code=400, detail="candidate_ids must not be empty")

    # Validate column name — allow letters, digits, spaces, and common punctuation
    if not re.match(r'^[\w\s\-\+\&\(\)\.\,\/]+$', body.column_name.strip()):
        raise HTTPException(status_code=400, detail="column_name contains invalid characters")

    logger.info("ai_column run — user=%s candidates=%s col='%s' web=%s",
                current_user.id, len(body.candidate_ids), body.column_name, body.use_web_search)

    background_tasks.add_task(
        _run_ai_column_batch,
        body.candidate_ids,
        body.prompt_template,
        body.column_name.strip(),
        body.use_web_search,
    )
    return {
        "status":   "started",
        "message":  f"AI column '{body.column_name}' is being generated for {len(body.candidate_ids)} candidates",
        "count":    len(body.candidate_ids),
    }


def clean_val(val):
    if val is None:
        return None
    cleaned = str(val).strip().strip("\ufeff\u200b").strip()
    if not cleaned or cleaned.lower() in [
        "none found",
        "not found",
        "undefined",
        "null",
        "",
        "n/a",
    ]:
        return None
    return cleaned


@router.post("/enrich/{candidate_id}")
async def enrich_candidate(
    candidate_id: int,
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """DB contact reuse first; call Clay only for missing email/phone."""
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT first_name, last_name, name, linkedin, email,
                       COALESCE(mobile_phone, phone) AS ph,
                       normalized_linkedin, owner_user_id
                FROM candidates WHERE id = %s AND COALESCE(is_archived, FALSE) = FALSE
                """,
                (candidate_id,),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Candidate not found")

            (
                first_name,
                last_name,
                full_name,
                linkedin_url,
                existing_email,
                existing_phone,
                norm_db,
                owner_uid,
            ) = row

            try:
                assert_admin_or_recruiter_owner(
                    cur,
                    role=current_user.role,
                    user_id=current_user.id,
                    candidate_id=candidate_id,
                )
            except PermissionError as pe:
                if str(pe) == "not_found":
                    raise HTTPException(status_code=404, detail="Candidate not found")
                raise HTTPException(status_code=403, detail="Forbidden")

            if (current_user.role or "").strip().lower() != "admin" and owner_uid is None:
                raise HTTPException(
                    status_code=403,
                    detail="Master library is read-only for recruiters",
                )

            norm = norm_db or normalize_linkedin(linkedin_url)
            db_email, db_phone = fetch_best_contact_for_normalized_li(cur, norm)
            merged_email = existing_email or db_email
            merged_phone = existing_phone or db_phone

            need_email = not (merged_email and str(merged_email).strip())
            need_phone = not (merged_phone and str(merged_phone).strip())

            if not need_email and not need_phone:
                cur.execute(
                    """
                    UPDATE candidates SET email = COALESCE(email, %s),
                      mobile_phone = COALESCE(mobile_phone, %s),
                      updated_at = NOW()
                    WHERE id = %s
                    """,
                    (db_email, db_phone, candidate_id),
                )
                conn.commit()
                query.update_candidate_contact(
                    linkedin_url or "",
                    merged_email,
                    merged_phone,
                    normalized_linkedin=norm,
                )
                logger.info("Skipping Clay for candidate %s (contact complete)", candidate_id)
                return {
                    "status": "cached",
                    "message": f"Already enriched: {full_name}",
                    "email": merged_email,
                    "phone": merged_phone,
                }

            if db_email or db_phone:
                cur.execute(
                    """
                    UPDATE candidates SET email = COALESCE(email, %s),
                      mobile_phone = COALESCE(mobile_phone, %s),
                      updated_at = NOW()
                    WHERE id = %s
                    """,
                    (db_email, db_phone, candidate_id),
                )
                conn.commit()
                query.update_candidate_contact(
                    linkedin_url or "",
                    db_email or merged_email,
                    db_phone or merged_phone,
                    normalized_linkedin=norm,
                )

            if not first_name or not last_name:
                parts = (full_name or "").split(" ", 1)
                first_name = parts[0]
                last_name = parts[1] if len(parts) > 1 else ""

            logger.info("Calling Clay for candidate %s", candidate_id)
            background_tasks.add_task(trigger_clay, first_name, last_name, linkedin_url)
            return {
                "status": "processing",
                "message": f"Enrichment started for {first_name} {last_name}",
            }


@router.post("/results")
async def receive_results(request: Request):
    """Clay callback — fan out contact updates by normalized LinkedIn."""
    data = await request.json()

    first = clean_val(data.get("first_name")) or "N/A"
    last = clean_val(data.get("last_name")) or "N/A"
    email = clean_val(data.get("result_email"))
    phone = clean_val(data.get("mobile_phone"))
    li_url = clean_val(data.get("linkedin_url"))

    logger.info(
        "Clay result %s %s linkedin=%s has_email=%s has_phone=%s",
        first,
        last,
        li_url or "N/A",
        bool(email),
        bool(phone),
    )

    norm = normalize_linkedin(li_url)
    if not norm:
        logger.warning("Clay callback ignored: missing or invalid linkedin_url")
        return {
            "status": "invalid_linkedin",
            "matched_candidates": 0,
            "updated_candidates": 0,
        }

    matched_ids = []
    if email or phone:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=503, detail="Database connection failed")
            with conn.cursor() as cur:
                # normalized_linkedin was added after some legacy imports. Compare the
                # raw URL in Python too so those candidates and legacy duplicate rows
                # still receive Clay results.
                cur.execute(
                    """
                    SELECT id, linkedin, normalized_linkedin
                    FROM candidates
                    WHERE COALESCE(is_archived, FALSE) = FALSE
                      AND (normalized_linkedin = %s OR linkedin IS NOT NULL)
                    """,
                    (norm,),
                )
                matched_ids = [
                    row[0]
                    for row in cur.fetchall()
                    if row[2] == norm or normalize_linkedin(row[1]) == norm
                ]

                if matched_ids:
                    cur.execute(
                        """
                        UPDATE candidates SET
                          email = CASE WHEN %s IS NOT NULL AND (
                            email IS NULL OR TRIM(email) = '' OR
                            LOWER(TRIM(email)) IN ('n/a', 'not available', 'not found', 'none found', 'undefined', 'null')
                          ) THEN %s ELSE email END,
                          mobile_phone = CASE WHEN %s IS NOT NULL AND (
                            NULLIF(TRIM(COALESCE(mobile_phone, phone)), '') IS NULL OR
                            LOWER(TRIM(COALESCE(mobile_phone, phone))) IN ('n/a', 'not available', 'not found', 'none found', 'undefined', 'null')
                          ) THEN %s ELSE mobile_phone END,
                          normalized_linkedin = CASE
                            WHEN normalized_linkedin IS NULL THEN %s
                            ELSE normalized_linkedin
                          END,
                          updated_at = NOW()
                        WHERE id = ANY(%s)
                        """,
                        (email, email, phone, phone, norm, matched_ids),
                    )
                    conn.commit()

    query.update_candidate_contact(
        li_url or "",
        email,
        phone,
        normalized_linkedin=norm,
    )
    if matched_ids:
        query.refresh_profiles_in_cache(matched_ids)
    try:
        from backend.api.routes import browse as browse_mod

        browse_mod._invalidate_browse_cache()
    except Exception:
        pass

    if not email and not phone:
        logger.warning("Clay callback contained no email or phone for linkedin=%s", li_url)
        callback_status = "no_contact"
    elif not matched_ids:
        logger.warning("Clay callback matched no candidate for linkedin=%s", li_url)
        callback_status = "no_match"
    else:
        callback_status = "updated"

    return {
        "status": callback_status,
        "matched_candidates": len(matched_ids),
        "updated_candidates": len(matched_ids) if email or phone else 0,
        "has_email": bool(email),
        "has_phone": bool(phone),
    }

"""Recruiter CSV/XLSX import: preview, optional LLM mapping assist, commit with dual master write."""

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
    fetch_best_contact_for_normalized_li,
    POOL_SOURCE_RECRUITER_UPLOAD,
)
from backend.services.linkedin_normalize import normalize_linkedin

router = APIRouter()
logger = logging.getLogger(__name__)

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
        alias_target = deterministic.get(header)
        model_item = model_raw.get(header)
        model_target = None
        confidence = 0.0
        reason = ""
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
        source = "model" if model_target else "alias" if alias_target else "manual"
        suggested[header] = target
        mapping_details[header] = {
            "target": target,
            "source": source,
            "confidence": confidence if model_target else (0.95 if alias_target else 0),
            "reason": reason or ("Matched known header alias" if alias_target else "No confident match"),
            "sample_values": [_clean(row.get(header)) or "" for row in sample[:4]],
        }

    used_targets = {v for v in suggested.values() if v and v != "ignore"}
    return {
        "filename": file.filename,
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

    raw = await file.read()
    try:
        df = _read_frame(raw, file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    inserted = updated = skipped = 0
    errors: List[str] = []
    touched_ids: List[int] = []
    role_assigned_count = 0
    row_count_total = int(len(df.index))
    unique_norms_in_file: set = set()
    user_role = (current_user.role or "").strip().lower()

    try:
        # Pooled connections can be left in "aborted" state after a prior error.
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidate_uploads (
                  owner_user_id, filename, file_headers, mapping, row_count, status, role_id
                ) VALUES (%s, %s, %s, %s, %s, 'processing', %s)
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

            row_spi = 0
            for idx, row in df.iterrows():
                vals, raw_vals = _row_values(row, mapping)
                fn = vals.get("first_name") or ""
                ln = vals.get("last_name") or ""
                li = vals.get("linkedin") or ""
                city = vals.get("city")
                title = vals.get("title")
                company_name = vals.get("company_name")
                norm = normalize_linkedin(li) if li else ""
                
                if not norm and not vals.get("email") and not (fn and ln and vals.get("company_name")):
                    # Can't dedup, skip
                    skipped += 1
                    continue
                    
                if norm:
                    unique_norms_in_file.add(norm)

                row_spi += 1
                sp = f"imp_row_{row_spi}"
                try:
                    # One bad row must not abort the whole import transaction.
                    cur.execute(f"SAVEPOINT {sp}")
                    email, phone = fetch_best_contact_for_normalized_li(cur, norm)
                    email = email or vals.get("email")
                    phone = phone or vals.get("phone")
                    master_existed = _master_row_exists(
                        cur,
                        normalized_li=norm,
                        email=email,
                        first_name=fn,
                        last_name=ln,
                        company_name=vals.get("company_name"),
                    )

                    mid = upsert_master_catalog_row(
                        cur,
                        normalized_li=norm,
                        raw_linkedin=li,
                        first_name=fn,
                        last_name=ln,
                        city=city,
                        title=title,
                        company_name=vals.get("company_name"),
                        email=email,
                        phone=phone,
                        location=vals.get("location"),
                        notes=None,
                        raw_fields_extra=raw_vals,
                    )
                    cid = mid
                    op = "updated" if master_existed else "inserted"
                    if user_role != "admin":
                        cid, op = upsert_recruiter_pool_row(
                            cur,
                            owner_id=current_user.id,
                            master_id=mid,
                            normalized_li=norm,
                            raw_linkedin=li,
                            first_name=fn,
                            last_name=ln,
                            city=city,
                            title=title,
                            company_name=vals.get("company_name"),
                            email=email,
                            phone=phone,
                            location=vals.get("location"),
                            notes=vals.get("notes"),
                            pool_source=POOL_SOURCE_RECRUITER_UPLOAD,
                            source_upload_id=upload_id,
                            assigned_by_user_id=None,
                            raw_fields_extra=raw_vals,
                        )
                    if _assign_imported_candidate_to_role(cur, role_id=role_id, candidate_id=int(cid)):
                        role_assigned_count += 1
                    touched_ids.extend([int(mid), int(cid)])
                    cur.execute(f"RELEASE SAVEPOINT {sp}")
                    if op == "inserted":
                        inserted += 1
                    else:
                        updated += 1
                except Exception as row_exc:
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                    except Exception:
                        conn.rollback()
                        raise
                    errors.append(f"row {idx}: {row_exc}")
                    skipped += 1

            cur.execute(
                """
                UPDATE candidate_uploads SET
                  inserted_count = %s,
                  updated_count = %s,
                  skipped_count = %s,
                  status = %s,
                  completed_at = %s,
                  error_message = %s
                WHERE id = %s
                """,
                (
                    inserted,
                    updated,
                    skipped,
                    "completed" if not errors else "completed_with_errors",
                    datetime.now(timezone.utc),
                    "\n".join(errors[:20]) if errors else None,
                    upload_id,
                ),
            )
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

    try:
        from backend.api.routes import browse as browse_mod

        browse_mod._invalidate_browse_cache()
    except Exception:
        pass

    unique_touched = sorted({i for i in touched_ids if i is not None})
    if unique_touched:
        try:
            merged = query.refresh_profiles_in_cache(unique_touched)
            if merged < len(unique_touched):
                logger.warning(
                    "refresh_profiles_in_cache merged %s/%s; falling back to full cache reload",
                    merged,
                    len(unique_touched),
                )
                query.initialize_cache()
        except Exception:
            logger.exception("refresh_profiles_in_cache failed; falling back to full cache reload")
            query.initialize_cache()

    return {
        "upload_id": upload_id,
        "row_count": row_count_total,
        "unique_linkedin_in_file": len(unique_norms_in_file),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "role_assigned_count": role_assigned_count,
        "errors": errors[:20],
    }


@router.post("/candidates/upload/commit")
async def upload_commit(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if current_user.role == "admin":
        raise HTTPException(status_code=403, detail="Recruiter only")
    return await commit_upload_file(file=file, mapping_json=mapping_json, current_user=current_user)


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
                    SELECT id, owner_user_id, filename, row_count, inserted_count,
                           updated_count, skipped_count, status, created_at, completed_at, role_id
                    FROM candidate_uploads
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, owner_user_id, filename, row_count, inserted_count,
                           updated_count, skipped_count, status, created_at, completed_at, role_id
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
                    "inserted_count": r[4],
                    "updated_count": r[5],
                    "skipped_count": r[6] if r[6] is not None else 0,
                    "status": r[7],
                    "created_at": r[8].isoformat() if r[8] else None,
                    "completed_at": r[9].isoformat() if r[9] else None,
                    "role_id": r[10],
                }
                for r in rows
            ]
        }
    finally:
        return_db_connection(conn)

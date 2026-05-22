from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel, Field

from backend.api import deps, schemas
from backend.api.routes.browse import build_browse_candidate_rows
from backend.db.connection import get_db_connection_context
from backend.pipeline import query
from backend.services.ai_columns import (
    build_candidate_context,
    build_field_catalog,
    default_output_schema,
    evaluate_required_fields,
    extract_prompt_tokens,
    fill_prompt_template,
    get_profiles_for_scope,
    labelize_key,
    map_raw_outputs_to_schema_keys,
    normalize_output_key,
    normalize_output_schema,
    safe_json,
    summarize_only_run_if,
)
from backend.services.candidate_pool import (
    VIEW_SCOPE_MASTER,
    VIEW_SCOPE_RECRUITER_POOLS,
)
from backend.services.ai_column_presets import list_ai_column_presets


router = APIRouter()
logger = logging.getLogger(__name__)

# Building the field catalog walks every profile's raw_fields; master-scope pools can be huge
# and block the worker for minutes (appears as a crash / dev-server restart). Sampling keeps
# Clay-style generate + field-catalog responsive while still surfacing representative raw keys.
_MAX_PROFILES_FOR_FIELD_CATALOG = int(os.getenv("AI_COLUMN_FIELD_CATALOG_PROFILE_CAP", "80"))
# Listing cells for too many candidate ids in one query can stall Postgres and the event loop.
_MAX_CANDIDATE_IDS_PER_AI_COLUMNS_LIST = int(os.getenv("AI_COLUMN_LIST_CANDIDATE_ID_CAP", "400"))
# Active runs older than this are usually left behind by a crashed/restarted dev worker.
_STALE_ACTIVE_RUN_AFTER_MINUTES = int(os.getenv("AI_COLUMN_STALE_ACTIVE_RUN_MINUTES", "240"))

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
_run_threads: dict[int, threading.Thread] = {}
_run_threads_lock = threading.Lock()
_list_cache: dict[tuple[Any, ...], tuple[float, List[Dict[str, Any]]]] = {}
_LIST_CACHE_TTL_SECONDS = 2.0


def _clear_list_cache() -> None:
    _list_cache.clear()


class AiColumnOutput(BaseModel):
    key: str
    label: str
    type: str = "text"
    primary: bool = False


class AiColumnGenerateRequest(BaseModel):
    goal: str
    view_scope: Optional[str] = None
    recruiter_filter_id: Optional[int] = None
    prefer_web_search: bool = False


class AiColumnSaveRequest(BaseModel):
    id: Optional[int] = None
    name: str
    prompt_template: str
    mode: str = "auto"
    output_schema: List[AiColumnOutput] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    only_run_if: Dict[str, Any] = Field(default_factory=dict)
    context_inputs: Dict[str, Any] = Field(default_factory=dict)
    view_scope: Optional[str] = None
    recruiter_filter_id: Optional[int] = None


class AiColumnTestRequest(BaseModel):
    candidate_id: int
    prompt_template: str
    mode: str = "auto"
    output_schema: List[AiColumnOutput] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    context_inputs: Dict[str, Any] = Field(default_factory=dict)
    view_scope: Optional[str] = None
    recruiter_filter_id: Optional[int] = None
    role_id: Optional[int] = None


class AiColumnRunRequest(BaseModel):
    column_definition_id: int
    selection_mode: str = "selected_ids"
    selected_ids: List[int] = Field(default_factory=list)
    view_scope: Optional[str] = None
    recruiter_filter_id: Optional[int] = None
    role_id: Optional[int] = None
    global_search: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    city: Optional[str] = None
    location_type: Optional[str] = None
    product_service: Optional[str] = None
    status: Optional[str] = None
    created_by: Optional[str] = None
    min_exp: Optional[float] = None
    max_exp: Optional[float] = None
    min_avg_tenure: Optional[float] = None
    sort_by: Optional[str] = "name"
    sort_dir: Optional[str] = "asc"


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _format_full_row_context(context: Dict[str, str], *, max_fields: int = 220, max_value_len: int = 900) -> str:
    def priority(item: tuple[str, str]) -> tuple[int, str]:
        key, _ = item
        if key.startswith("candidate."):
            return (0, key)
        if key.startswith("role."):
            return (1, key)
        if key.startswith("raw."):
            return (2, key)
        if key.startswith("ai."):
            return (3, key)
        if key.startswith("row."):
            return (4, key)
        return (5, key)

    payload: Dict[str, str] = {}
    for key, raw_value in sorted((context or {}).items(), key=priority):
        value = str(raw_value or "").strip()
        if not value:
            continue
        if len(value) > max_value_len:
            value = f"{value[:max_value_len]}..."
        payload[key] = value
        if len(payload) >= max_fields:
            break
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _slugify(name: str) -> str:
    return normalize_output_key(name)


def _resolve_scope(
    current_user: schemas.User,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
) -> tuple[str, Optional[int]]:
    if (current_user.role or "").strip().lower() == "admin":
        return view_scope or VIEW_SCOPE_MASTER, recruiter_filter_id
    return VIEW_SCOPE_RECRUITER_POOLS, current_user.id


def _limit_profiles_for_field_catalog(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return up to N evenly spaced profiles so build_field_catalog stays O(cap) not O(pool size)."""
    plist = list(profiles or [])
    n = len(plist)
    cap = max(8, _MAX_PROFILES_FOR_FIELD_CATALOG)
    if n <= cap:
        return plist
    idxs = sorted({min(n - 1, int(round(i * (n - 1) / (cap - 1)))) for i in range(cap)})
    sampled = [plist[i] for i in idxs]
    logger.info(
        "AI column field catalog: sampling %s of %s profiles (cap=%s)",
        len(sampled),
        n,
        cap,
    )
    return sampled


def _definition_where_clause(current_user: schemas.User) -> str:
    if (current_user.role or "").strip().lower() == "admin":
        return "(owner_user_id = %s OR owner_user_id IS NULL)"
    return "owner_user_id = %s"


def _definition_owner_clause(current_user: schemas.User, alias: str = "d") -> str:
    owner = f"{alias}.owner_user_id"
    if (current_user.role or "").strip().lower() == "admin":
        return f"({owner} = %s OR {owner} IS NULL)"
    return f"{owner} = %s"


def _fetch_role_context(role_id: Optional[int], current_user: schemas.User) -> Dict[str, Any]:
    if not role_id:
        return {}
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return {}
        with conn.cursor() as cur:
            if (current_user.role or "").strip().lower() == "admin":
                cur.execute(
                    "SELECT id, name, job_description FROM recruitment_roles WHERE id = %s",
                    (role_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, job_description
                    FROM recruitment_roles
                    WHERE id = %s AND user_id = %s
                    """,
                    (role_id, current_user.id),
                )
            row = cur.fetchone()
    if not row:
        return {}
    return {
        "id": int(row[0]),
        "name": row[1] or "",
        "job_description": row[2] or "",
    }


def _fail_stale_active_runs(
    cur: Any,
    current_user: schemas.User,
    *,
    resolved_scope: str,
    resolved_recruiter: Optional[int],
) -> int:
    """Close abandoned queued/running runs so the UI does not poll forever."""
    timeout_minutes = max(10, _STALE_ACTIVE_RUN_AFTER_MINUTES)
    cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
    cur.execute(
        f"""
        UPDATE ai_column_runs r
        SET status = 'failed',
            failed = GREATEST(r.failed, GREATEST(r.total - r.completed - r.skipped, 0)),
            completed_at = COALESCE(r.completed_at, NOW()),
            updated_at = NOW()
        FROM ai_column_definitions d
        WHERE r.column_definition_id = d.id
          AND r.status IN ('queued', 'running')
          AND COALESCE(r.started_at, r.created_at, r.updated_at) < %s
          AND {_definition_owner_clause(current_user)}
          AND d.view_scope = %s
          AND COALESCE(d.recruiter_filter_id, 0) = COALESCE(%s, 0)
          AND COALESCE(d.is_archived, FALSE) = FALSE
        RETURNING r.id
        """,
        (cutoff, current_user.id, resolved_scope, resolved_recruiter),
    )
    run_ids = [int(row[0]) for row in cur.fetchall()]
    if not run_ids:
        return 0
    cur.execute(
        """
        UPDATE ai_column_cells
        SET status = 'failed',
            error_message = 'Run timed out before this row completed.',
            completed_at = NOW(),
            updated_at = NOW()
        WHERE last_run_id = ANY(%s)
          AND status IN ('queued', 'running')
        """,
        (run_ids,),
    )
    logger.warning("Marked %s stale AI column run(s) as failed", len(run_ids))
    return len(run_ids)


def _fetch_visible_definitions(
    current_user: schemas.User,
    *,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
    candidate_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    resolved_scope, resolved_recruiter = _resolve_scope(current_user, view_scope, recruiter_filter_id)
    candidate_ids = [int(cid) for cid in (candidate_ids or []) if cid is not None]
    if len(candidate_ids) > _MAX_CANDIDATE_IDS_PER_AI_COLUMNS_LIST:
        logger.warning(
            "AI columns list: truncating candidate_ids from %s to %s",
            len(candidate_ids),
            _MAX_CANDIDATE_IDS_PER_AI_COLUMNS_LIST,
        )
        candidate_ids = candidate_ids[:_MAX_CANDIDATE_IDS_PER_AI_COLUMNS_LIST]
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            stale_count = _fail_stale_active_runs(
                cur,
                current_user,
                resolved_scope=resolved_scope,
                resolved_recruiter=resolved_recruiter,
            )
            if stale_count:
                conn.commit()
            cur.execute(
                f"""
                SELECT d.id, d.name, d.slug, d.owner_user_id, d.view_scope, d.recruiter_filter_id,
                       d.prompt_template, d.mode, d.output_schema, d.required_fields, d.only_run_if,
                       d.context_inputs,
                       d.created_at, d.updated_at,
                       r.id, r.total, r.completed, r.failed, r.skipped, r.status, r.started_at, r.completed_at
                FROM ai_column_definitions d
                LEFT JOIN LATERAL (
                    SELECT id, total, completed, failed, skipped, status, started_at, completed_at
                    FROM ai_column_runs
                    WHERE column_definition_id = d.id
                    ORDER BY id DESC
                    LIMIT 1
                ) r ON TRUE
                WHERE {_definition_where_clause(current_user)}
                  AND d.view_scope = %s
                  AND COALESCE(d.recruiter_filter_id, 0) = COALESCE(%s, 0)
                  AND COALESCE(d.is_archived, FALSE) = FALSE
                ORDER BY d.created_at ASC
                """,
                (current_user.id, resolved_scope, resolved_recruiter),
            )
            rows = cur.fetchall()

            cell_rows = []
            if candidate_ids:
                cur.execute(
                    """
                    SELECT c.column_definition_id, c.candidate_id, c.primary_output, c.outputs, c.status,
                           c.error_message, c.completed_at, c.updated_at
                    FROM ai_column_cells c
                    WHERE c.column_definition_id = ANY(%s) AND c.candidate_id = ANY(%s)
                    """,
                    ([row[0] for row in rows] or [0], candidate_ids),
                )
                cell_rows = cur.fetchall()

    cells_by_definition: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in cell_rows:
        def_id = int(row[0])
        candidate_id = int(row[1])
        cells_by_definition.setdefault(def_id, {})[candidate_id] = {
            "candidate_id": candidate_id,
            "primary_output": row[2] or "",
            "outputs": safe_json(row[3]),
            "status": row[4] or "idle",
            "error_message": row[5] or "",
            "completed_at": row[6].isoformat() if row[6] else None,
            "updated_at": row[7].isoformat() if row[7] else None,
        }

    definitions = []
    for row in rows:
        definitions.append(
            {
                "id": int(row[0]),
                "name": row[1],
                "slug": row[2],
                "owner_user_id": row[3],
                "view_scope": row[4],
                "recruiter_filter_id": row[5],
                "prompt_template": row[6],
                "mode": row[7] or "auto",
                "output_schema": normalize_output_schema(_parse_jsonish(row[8], [])),
                "required_fields": _parse_jsonish(row[9], []),
                "only_run_if": _parse_jsonish(row[10], {}),
                "context_inputs": _parse_jsonish(row[11], {}),
                "created_at": row[12].isoformat() if row[12] else None,
                "updated_at": row[13].isoformat() if row[13] else None,
                "latest_run": (
                    {
                        "id": row[14],
                        "total": row[15] or 0,
                        "completed": row[16] or 0,
                        "failed": row[17] or 0,
                        "skipped": row[18] or 0,
                        "status": row[19] or "idle",
                        "started_at": row[20].isoformat() if row[20] else None,
                        "completed_at": row[21].isoformat() if row[21] else None,
                    }
                    if row[14]
                    else None
                ),
                "cells_by_candidate": cells_by_definition.get(int(row[0]), {}),
            }
        )
    return definitions


def _fetch_profile(candidate_id: int) -> Optional[Dict[str, Any]]:
    profile = query.PROFILES_BY_ID.get(candidate_id)
    if profile:
        return profile
    from backend.api.routes.enrichment import _fetch_profile_from_db

    return _fetch_profile_from_db(candidate_id)


def _ai_context_values_from_definitions(definitions: List[Dict[str, Any]], candidate_id: int) -> Dict[str, str]:
    """Build ai.* tokens for one row from definitions that already include cells_by_candidate."""
    ai_values: Dict[str, str] = {}
    for definition in definitions:
        slug = definition.get("slug") or _slugify(definition.get("name", ""))
        cell = (definition.get("cells_by_candidate") or {}).get(candidate_id)
        if not cell:
            continue
        outputs = cell.get("outputs") or {}
        for output in definition.get("output_schema") or []:
            ai_values[f"ai.{slug}.{output['key']}"] = str(outputs.get(output["key"]) or "")
        ai_values[f"ai.{slug}.primary"] = str(cell.get("primary_output") or "")
    return ai_values


def _fetch_ai_context_for_candidate(candidate_id: int, current_user: schemas.User, view_scope: Optional[str], recruiter_filter_id: Optional[int]) -> Dict[str, str]:
    definitions = _fetch_visible_definitions(
        current_user,
        view_scope=view_scope,
        recruiter_filter_id=recruiter_filter_id,
        candidate_ids=[candidate_id],
    )
    return _ai_context_values_from_definitions(definitions, candidate_id)


def _json_block_to_dict(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def _call_openai_for_json(system_prompt: str, user_prompt: str, *, use_web: bool = False) -> Dict[str, Any]:
    if not _openai_client:
        return {}
    try:
        request_timeout = 75.0 if use_web else 35.0
        if use_web:
            response = _openai_client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search_preview"}],
                input=f"{system_prompt}\n\n{user_prompt}",
                timeout=request_timeout,
            )
            return _json_block_to_dict(response.output_text or "")
        response = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=request_timeout,
        )
        content = ""
        choices = getattr(response, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", None) or ""
        return _json_block_to_dict(content)
    except Exception as exc:
        logger.warning("AI columns OpenAI JSON call failed: %s", exc)
        return {}


def _generate_config(
    goal: str,
    field_catalog: List[Dict[str, Any]],
    *,
    prefer_web_search: bool = False,
) -> Dict[str, Any]:
    trimmed_goal = (goal or "").strip()
    if not trimmed_goal:
        raise HTTPException(status_code=400, detail="goal is required")

    flattened_fields = []
    for group in field_catalog:
        for item in group.get("items", [])[:25]:
            label = str(item.get("label") or item.get("key") or "?")
            token = str(item.get("token") or (f"{{{item.get('key', '')}}}"))
            flattened_fields.append(f"{label} -> {token}")

    system_prompt = (
        "You design Clay-style AI column configs for recruitment tables. "
        "Return JSON only with keys: name, prompt_template, mode, output_schema, required_fields. "
        "mode must be one of auto, content, web_research. "
        "output_schema must be an array of objects with key, label, type, primary."
    )
    user_prompt = (
        f"Goal: {trimmed_goal}\n"
        "Available fields:\n"
        + "\n".join(flattened_fields[:80])
        + "\nUse field tokens naturally inside the prompt if relevant."
    )
    generated = _call_openai_for_json(system_prompt, user_prompt, use_web=False)

    output_schema = normalize_output_schema(generated.get("output_schema") or default_output_schema(trimmed_goal))
    prompt_template = str(generated.get("prompt_template") or trimmed_goal).strip()
    _raw_rf = generated.get("required_fields")
    if isinstance(_raw_rf, list):
        req_iter = [x for x in _raw_rf if isinstance(x, str) and x.strip()]
    else:
        req_iter = []
    if not req_iter:
        req_iter = [t for t in extract_prompt_tokens(prompt_template) if isinstance(t, str)]
    required_fields = list(dict.fromkeys(req_iter))
    tokens_in_prompt = set(extract_prompt_tokens(prompt_template))
    # Models often return required_fields like candidate.email/phone even when the prompt
    # only uses LinkedIn + name — that skips rows without email. Keep only keys referenced in the prompt.
    required_fields = [f for f in required_fields if f in tokens_in_prompt]
    mode = str(generated.get("mode") or "").strip().lower()
    if mode not in {"auto", "content", "web_research"}:
        mode = "auto"
    goal_l = trimmed_goal.lower()
    explicit_web_terms = (
        "use web",
        "search web",
        "search the web",
        "browse web",
        "browse the web",
        "research online",
        "search online",
        "latest news",
        "recent news",
        "company website",
        "website research",
    )
    if mode == "web_research" and not any(term in goal_l for term in explicit_web_terms):
        mode = "auto"
    if prefer_web_search:
        mode = "web_research"
    name = str(generated.get("name") or "").strip()
    if not name:
        primary = next((item["label"] for item in output_schema if item.get("primary")), None)
        name = primary or "AI Result"
    return {
        "name": name[:255],
        "prompt_template": prompt_template,
        "mode": mode,
        "output_schema": output_schema,
        "required_fields": required_fields,
        "only_run_if": {
            "required_fields": required_fields,
            "summary": summarize_only_run_if(required_fields),
        },
    }


def _run_ai_task(
    *,
    prompt_template: str,
    mode: str,
    output_schema: List[Dict[str, Any]],
    context: Dict[str, str],
) -> Dict[str, Any]:
    rendered_prompt = fill_prompt_template(prompt_template, context)
    full_row_context = _format_full_row_context(context)
    normalized_outputs = normalize_output_schema(output_schema)
    primary_output_key = next((item["key"] for item in normalized_outputs if item.get("primary")), normalized_outputs[0]["key"])

    if not _openai_client:
        fallback_outputs = {item["key"]: "" for item in normalized_outputs}
        fallback_outputs[primary_output_key] = "AI service is not configured."
        return {
            "primary_output": fallback_outputs[primary_output_key],
            "outputs": fallback_outputs,
            "details": {
                "response": fallback_outputs[primary_output_key],
                "outputs": fallback_outputs,
                "reasoning": "OPENAI_API_KEY is missing, so the AI column could not run.",
                "confidence": "low",
                "steps": [],
                "sources": [],
                "rendered_prompt": rendered_prompt,
                "row_context_keys": list((context or {}).keys()),
                "raw_model_output": {},
            },
        }

    system_prompt = (
        "You are an expert recruiter operations analyst. Return valid JSON only with keys "
        "outputs, reasoning, confidence, steps, sources. "
        "outputs must be an object whose keys exactly match the requested output keys. "
        "confidence must be one of low, medium, high. steps must be an array of short strings. "
        "sources must be an array of objects with optional title, url, note."
    )
    content_first_system_prompt = (
        f"{system_prompt} "
        "When web access is disabled, answer strictly from the candidate row context already present in the task. "
        "If the row context is insufficient, leave outputs blank instead of guessing and explain what information is missing."
    )
    output_hint = ", ".join([f"{item['key']} ({item['label']})" for item in normalized_outputs])
    user_prompt = (
        f"Requested outputs: {output_hint}\n"
        "Use the full row context as the source of truth for this candidate. "
        "The user prompt may mention only part of the row; use any relevant row data when answering.\n\n"
        f"Full row context JSON:\n{full_row_context}\n\n"
        f"User prompt:\n{rendered_prompt or prompt_template}"
    )
    if mode == "web_research":
        structured = _call_openai_for_json(system_prompt, user_prompt, use_web=True)
    elif mode == "content":
        structured = _call_openai_for_json(content_first_system_prompt, user_prompt, use_web=False)
    else:
        structured = _call_openai_for_json(content_first_system_prompt, user_prompt, use_web=False)
        auto_outputs = structured.get("outputs") if isinstance(structured.get("outputs"), dict) else {}
        has_row_answer = any(str(auto_outputs.get(item["key"]) or "").strip() for item in normalized_outputs)
        if not has_row_answer:
            structured = _call_openai_for_json(system_prompt, user_prompt, use_web=True)

    outputs = structured.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}

    schema_keys = [item["key"] for item in normalized_outputs]
    normalized_values = map_raw_outputs_to_schema_keys(outputs, schema_keys)

    if not any(normalized_values.values()):
        fallback_text = structured.get("result") or structured.get("response") or "No structured response returned."
        fb = str(fallback_text).strip()
        for item in normalized_outputs:
            normalized_values[item["key"]] = fb

    primary_output = normalized_values.get(primary_output_key) or next(iter(normalized_values.values()), "")
    details = {
        "response": primary_output,
        "outputs": normalized_values,
        "reasoning": str(structured.get("reasoning") or "").strip(),
        "confidence": str(structured.get("confidence") or "medium").strip().lower(),
        "steps": structured.get("steps") if isinstance(structured.get("steps"), list) else [],
        "sources": structured.get("sources") if isinstance(structured.get("sources"), list) else [],
        "rendered_prompt": rendered_prompt,
        "row_context_keys": list((context or {}).keys()),
        "raw_model_output": structured,
    }
    return {
        "primary_output": primary_output,
        "outputs": normalized_values,
        "details": details,
    }


def _upsert_ai_column_cell(
    *,
    column_definition_id: int,
    candidate_id: int,
    primary_output: str,
    outputs: Dict[str, Any],
    details: Dict[str, Any],
    status: str,
    error_message: str,
    run_id: Optional[int] = None,
    cur: Optional[Any] = None,
) -> None:
    if cur:
        cur.execute(
            """
            INSERT INTO ai_column_cells (
                column_definition_id, candidate_id, primary_output, outputs, details,
                status, error_message, started_at, completed_at, last_run_id, updated_at
            )
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, NOW(),
                    CASE WHEN %s IN ('completed', 'failed', 'skipped') THEN NOW() ELSE NULL END,
                    %s, NOW())
            ON CONFLICT (column_definition_id, candidate_id)
            DO UPDATE SET
                primary_output = EXCLUDED.primary_output,
                outputs = EXCLUDED.outputs,
                details = EXCLUDED.details,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                last_run_id = EXCLUDED.last_run_id,
                updated_at = NOW()
            """,
            (
                column_definition_id,
                candidate_id,
                primary_output,
                json.dumps(outputs or {}, ensure_ascii=False),
                json.dumps(details or {}, ensure_ascii=False),
                status,
                error_message,
                status,
                run_id,
            ),
        )
        return

    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_column_cells (
                    column_definition_id, candidate_id, primary_output, outputs, details,
                    status, error_message, started_at, completed_at, last_run_id, updated_at
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, NOW(),
                        CASE WHEN %s IN ('completed', 'failed', 'skipped') THEN NOW() ELSE NULL END,
                        %s, NOW())
                ON CONFLICT (column_definition_id, candidate_id)
                DO UPDATE SET
                    primary_output = EXCLUDED.primary_output,
                    outputs = EXCLUDED.outputs,
                    details = EXCLUDED.details,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    last_run_id = EXCLUDED.last_run_id,
                    updated_at = NOW()
                """,
                (
                    column_definition_id,
                    candidate_id,
                    primary_output,
                    json.dumps(outputs or {}, ensure_ascii=False),
                    json.dumps(details or {}, ensure_ascii=False),
                    status,
                    error_message,
                    status,
                    run_id,
                ),
            )
            conn.commit()


def _update_run_progress(run_id: int, cur: Optional[Any] = None, **updates: Any) -> None:
    if not updates:
        return
    fields = []
    values = []
    for key, value in updates.items():
        fields.append(f"{key} = %s")
        values.append(value)
    values.append(run_id)
    
    if cur:
        cur.execute(
            f"UPDATE ai_column_runs SET {', '.join(fields)}, updated_at = NOW() WHERE id = %s",
            values,
        )
        return

    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE ai_column_runs SET {', '.join(fields)}, updated_at = NOW() WHERE id = %s",
                values,
            )
            conn.commit()


def _seed_queued_ai_column_cells(run_id: int, column_definition_id: int, candidate_ids: List[int]) -> None:
    """Create/update one cell per candidate as queued so the Talent Pool grid shows progress while the run executes."""
    if not candidate_ids:
        return
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_column_cells (
                    column_definition_id, candidate_id, primary_output, outputs, details,
                    status, error_message, started_at, completed_at, last_run_id, updated_at
                )
                SELECT %s, cid, '', '{}'::jsonb, '{}'::jsonb, 'queued', '', NULL, NULL, %s, NOW()
                FROM unnest(%s::int[]) AS cid
                ON CONFLICT (column_definition_id, candidate_id)
                DO UPDATE SET
                    primary_output = EXCLUDED.primary_output,
                    outputs = EXCLUDED.outputs,
                    details = EXCLUDED.details,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    last_run_id = EXCLUDED.last_run_id,
                    updated_at = NOW()
                """,
                (column_definition_id, run_id, candidate_ids),
            )
            conn.commit()


def _mark_incomplete_run_cells_failed(run_id: int, column_definition_id: int, message: str) -> None:
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ai_column_cells
                SET status = 'failed',
                    error_message = %s,
                    completed_at = NOW(),
                    updated_at = NOW()
                WHERE column_definition_id = %s
                  AND last_run_id = %s
                  AND status IN ('queued', 'running')
                """,
                (message[:8000], column_definition_id, run_id),
            )
            conn.commit()


def _fetch_definition_by_id(column_definition_id: int, current_user: Optional[schemas.User] = None) -> Optional[Dict[str, Any]]:
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return None
        with conn.cursor() as cur:
            if current_user:
                cur.execute(
                    f"""
                    SELECT id, name, slug, owner_user_id, view_scope, recruiter_filter_id,
                           prompt_template, mode, output_schema, required_fields, only_run_if,
                           context_inputs, is_archived
                    FROM ai_column_definitions
                    WHERE id = %s AND {_definition_where_clause(current_user)}
                    """,
                    (column_definition_id, current_user.id),
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, slug, owner_user_id, view_scope, recruiter_filter_id,
                           prompt_template, mode, output_schema, required_fields, only_run_if,
                           context_inputs, is_archived
                    FROM ai_column_definitions
                    WHERE id = %s
                    """,
                    (column_definition_id,),
                )
            row = cur.fetchone()
    if not row or row[12]:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "slug": row[2],
        "owner_user_id": row[3],
        "view_scope": row[4],
        "recruiter_filter_id": row[5],
        "prompt_template": row[6],
        "mode": row[7],
        "output_schema": normalize_output_schema(_parse_jsonish(row[8], [])),
        "required_fields": _parse_jsonish(row[9], []),
        "only_run_if": _parse_jsonish(row[10], {}),
        "context_inputs": _parse_jsonish(row[11], {}),
    }


def _is_run_cancelled(run_id: int, column_definition_id: int) -> bool:
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return False
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.status, COALESCE(d.is_archived, FALSE)
                FROM ai_column_runs r
                JOIN ai_column_definitions d ON d.id = r.column_definition_id
                WHERE r.id = %s AND r.column_definition_id = %s
                """,
                (run_id, column_definition_id),
            )
            row = cur.fetchone()
    if not row:
        return True
    return (row[0] or "").lower() == "canceled" or bool(row[1])


async def _resolve_run_candidate_ids(current_user: schemas.User, body: AiColumnRunRequest) -> List[int]:
    if body.selection_mode == "selected_ids":
        return sorted({int(cid) for cid in body.selected_ids if cid is not None})
    if body.selection_mode != "all_filtered":
        raise HTTPException(status_code=400, detail="Unsupported selection_mode")

    payload = await build_browse_candidate_rows(
        current_user=current_user,
        view_scope=body.view_scope,
        recruiter_filter_id=body.recruiter_filter_id,
        q=body.global_search,
        title=body.title,
        company=body.company,
        city=body.city,
        location_type=body.location_type,
        product_service=body.product_service,
        status=body.status,
        created_by=body.created_by,
        min_exp=body.min_exp,
        max_exp=body.max_exp,
        min_avg_tenure=body.min_avg_tenure,
        sort_by=body.sort_by,
        sort_dir=body.sort_dir,
    )
    return [int(row["id"]) for row in payload.get("candidates", [])]


def _process_ai_run(
    run_id: int,
    current_user: schemas.User,
    candidate_ids: List[int],
    column_definition_id: int,
    role_id: Optional[int] = None,
) -> None:
    """
    Run AI for each candidate. Must not hold a pooled DB connection across OpenAI calls:
    nested _fetch_visible_definitions while a connection is checked out exhausts the pool
    under concurrent runs and breaks GET /api/ai-columns with 500.
    """
    completed = failed = skipped = 0
    try:
        definition = _fetch_definition_by_id(column_definition_id)
        if not definition:
            _update_run_progress(run_id, status="failed", completed_at=datetime.utcnow())
            return

        _update_run_progress(run_id, status="running", started_at=datetime.utcnow())
        _seed_queued_ai_column_cells(run_id, column_definition_id, candidate_ids)
        role_context = _fetch_role_context(role_id, current_user)

        try:
            definitions_for_ai_ctx = _fetch_visible_definitions(
                current_user,
                view_scope=definition.get("view_scope"),
                recruiter_filter_id=definition.get("recruiter_filter_id"),
                candidate_ids=candidate_ids,
            )
        except Exception:
            logger.exception(
                "AI column run could not load prior AI-column context; continuing with row context only "
                "(run_id=%s column_definition_id=%s)",
                run_id,
                column_definition_id,
            )
            definitions_for_ai_ctx = []

        for idx, candidate_id in enumerate(candidate_ids):
            if _is_run_cancelled(run_id, column_definition_id):
                logger.info("AI column run canceled (run_id=%s column_definition_id=%s)", run_id, column_definition_id)
                _update_run_progress(
                    run_id,
                    status="canceled",
                    completed_at=datetime.utcnow(),
                    completed=completed,
                    failed=failed,
                    skipped=skipped,
                )
                return

            profile = _fetch_profile(candidate_id)
            if not profile:
                failed += 1
                _upsert_ai_column_cell(
                    column_definition_id=column_definition_id,
                    candidate_id=candidate_id,
                    primary_output="",
                    outputs={},
                    details={"response": "", "reasoning": "", "confidence": "low", "steps": [], "sources": []},
                    status="failed",
                    error_message="Candidate not found",
                    run_id=run_id,
                )
                if (idx + 1) % 5 == 0 or idx == len(candidate_ids) - 1:
                    _update_run_progress(run_id, completed=completed, failed=failed, skipped=skipped)
                continue

            ai_values = _ai_context_values_from_definitions(definitions_for_ai_ctx, candidate_id)
            context = build_candidate_context(
                profile,
                ai_values=ai_values,
                role_context=role_context,
                context_inputs=definition.get("context_inputs") or {},
            )
            required_fields = list(definition.get("required_fields") or [])
            if required_fields:
                ok_to_run, missing = evaluate_required_fields(required_fields, context)
            else:
                ok_to_run, missing = True, []
            if not ok_to_run:
                skipped += 1
                missing_label = ", ".join(missing[:20])
                if len(missing) > 20:
                    missing_label += ", …"
                skip_msg = f"Missing required fields: {missing_label}" if missing_label else "Missing required fields"
                _upsert_ai_column_cell(
                    column_definition_id=column_definition_id,
                    candidate_id=candidate_id,
                    primary_output="",
                    outputs={},
                    details={
                        "response": "", "reasoning": "", "confidence": "low", "steps": [], "sources": [],
                        "missing_required_fields": missing,
                    },
                    status="skipped",
                    error_message=skip_msg[:8000],
                    run_id=run_id,
                )
                if (idx + 1) % 5 == 0 or idx == len(candidate_ids) - 1:
                    _update_run_progress(run_id, completed=completed, failed=failed, skipped=skipped)
                continue

            _upsert_ai_column_cell(
                column_definition_id=column_definition_id,
                candidate_id=candidate_id,
                primary_output="",
                outputs={},
                details={"response": "", "reasoning": "", "confidence": "low", "steps": [], "sources": []},
                status="running",
                error_message="",
                run_id=run_id,
            )
            if (idx + 1) % 5 == 0 or idx == len(candidate_ids) - 1:
                _update_run_progress(run_id, completed=completed, failed=failed, skipped=skipped)

            try:
                result = _run_ai_task(
                    prompt_template=definition["prompt_template"],
                    mode=definition.get("mode") or "auto",
                    output_schema=definition.get("output_schema") or [],
                    context=context,
                )
                completed += 1
                _upsert_ai_column_cell(
                    column_definition_id=column_definition_id,
                    candidate_id=candidate_id,
                    primary_output=result["primary_output"],
                    outputs=result["outputs"],
                    details=result["details"],
                    status="completed",
                    error_message="",
                    run_id=run_id,
                )
            except Exception as exc:
                logger.exception("AI column run failed for candidate %s", candidate_id)
                failed += 1
                _upsert_ai_column_cell(
                    column_definition_id=column_definition_id,
                    candidate_id=candidate_id,
                    primary_output="",
                    outputs={},
                    details={"response": "", "reasoning": "", "confidence": "low", "steps": [], "sources": []},
                    status="failed",
                    error_message=str(exc),
                    run_id=run_id,
                )
            if (idx + 1) % 5 == 0 or idx == len(candidate_ids) - 1:
                _update_run_progress(run_id, completed=completed, failed=failed, skipped=skipped)

        final_status = "completed" if failed == 0 else ("completed_with_errors" if completed or skipped else "failed")
        _update_run_progress(
            run_id,
            status=final_status,
            completed_at=datetime.utcnow(),
            completed=completed,
            failed=failed,
            skipped=skipped,
        )
    except Exception:
        logger.exception("AI column run crashed (run_id=%s column_definition_id=%s)", run_id, column_definition_id)
        _mark_incomplete_run_cells_failed(run_id, column_definition_id, "Run failed before this row completed.")
        _update_run_progress(
            run_id,
            status="failed",
            completed_at=datetime.utcnow(),
            completed=completed,
            failed=failed,
            skipped=skipped,
        )
    finally:
        try:
            query.initialize_cache()
        except Exception:
            pass
        with _run_threads_lock:
            _run_threads.pop(run_id, None)


def _spawn_run_thread(
    run_id: int,
    current_user: schemas.User,
    candidate_ids: List[int],
    column_definition_id: int,
    role_id: Optional[int] = None,
) -> None:
    worker = threading.Thread(
        target=_process_ai_run,
        args=(run_id, current_user, candidate_ids, column_definition_id, role_id),
        daemon=True,
        name=f"ai-column-run-{run_id}",
    )
    with _run_threads_lock:
        _run_threads[run_id] = worker
    worker.start()


def _parse_candidate_ids_query(raw: Optional[str]) -> List[int]:
    """Parse comma-separated candidate ids; reject garbage so the client gets 400 instead of 500."""
    out: List[int] = []
    if not raw or not str(raw).strip():
        return out
    for item in str(raw).split(","):
        piece = item.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid candidate id in candidate_ids query: {piece!r}",
            ) from None
    return out


@router.get("/ai-columns")
def list_ai_columns(
    candidate_ids: Optional[str] = Query(None),
    view_scope: Optional[str] = Query(None),
    recruiter_filter_id: Optional[int] = Query(None),
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    candidate_id_list = _parse_candidate_ids_query(candidate_ids)
    cache_key = (
        current_user.id,
        current_user.role,
        view_scope or "",
        recruiter_filter_id or 0,
        role_id or 0,
        tuple(candidate_id_list),
    )
    cached = _list_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _LIST_CACHE_TTL_SECONDS:
        return {"columns": cached[1]}
    try:
        columns = _fetch_visible_definitions(
            current_user,
            view_scope=view_scope,
            recruiter_filter_id=recruiter_filter_id,
            candidate_ids=candidate_id_list,
        )
        _list_cache[cache_key] = (time.monotonic(), columns)
        return {"columns": columns}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GET /ai-columns failed (n_candidate_ids=%s)", len(candidate_id_list))
        raise HTTPException(
            status_code=500,
            detail=(str(exc) or "Failed to load AI columns")[:800],
        ) from exc


@router.get("/ai-columns/field-catalog")
def get_ai_column_field_catalog(
    view_scope: Optional[str] = Query(None),
    recruiter_filter_id: Optional[int] = Query(None),
    role_id: Optional[int] = Query(None),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    resolved_scope, resolved_recruiter = _resolve_scope(current_user, view_scope, recruiter_filter_id)
    profiles = get_profiles_for_scope(
        query.PROFILES_BY_ID,
        user_role=(current_user.role or "").strip().lower(),
        user_id=current_user.id,
        view_scope=resolved_scope,
        recruiter_filter_id=resolved_recruiter,
    )
    defs = _fetch_visible_definitions(
        current_user,
        view_scope=resolved_scope,
        recruiter_filter_id=resolved_recruiter,
    )
    profiles = _limit_profiles_for_field_catalog(profiles)
    return {
        "groups": build_field_catalog(profiles, ai_columns=defs),
        "role_context": _fetch_role_context(role_id, current_user),
    }


@router.get("/ai-columns/presets")
def get_ai_column_presets(_current_user: schemas.User = Depends(deps.get_current_user)):
    return {"presets": list_ai_column_presets()}


@router.post("/ai-columns/generate")
def generate_ai_column_config(
    body: AiColumnGenerateRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    try:
        resolved_scope, resolved_recruiter = _resolve_scope(current_user, body.view_scope, body.recruiter_filter_id)
        profiles = get_profiles_for_scope(
            query.PROFILES_BY_ID,
            user_role=(current_user.role or "").strip().lower(),
            user_id=current_user.id,
            view_scope=resolved_scope,
            recruiter_filter_id=resolved_recruiter,
        )
        defs = _fetch_visible_definitions(
            current_user,
            view_scope=resolved_scope,
            recruiter_filter_id=resolved_recruiter,
        )
        profiles = _limit_profiles_for_field_catalog(profiles)
        field_catalog = build_field_catalog(profiles, ai_columns=defs)
        return _generate_config(body.goal, field_catalog, prefer_web_search=bool(body.prefer_web_search))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /ai-columns/generate failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc)[:800] if str(exc) else "AI column generation failed",
        ) from exc


@router.post("/ai-columns")
def save_ai_column_definition(
    body: AiColumnSaveRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    resolved_scope, resolved_recruiter = _resolve_scope(current_user, body.view_scope, body.recruiter_filter_id)
    slug = _slugify(body.name)
    output_schema = normalize_output_schema([_model_to_dict(item) for item in body.output_schema])
    # Don't auto-extract required_fields from prompt tokens.
    # Required fields are an explicit run-guard the user opts into.
    required_fields = body.required_fields or []
    only_run_if = body.only_run_if or {
        "required_fields": required_fields,
        "summary": summarize_only_run_if(required_fields),
    }
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            if body.id:
                cur.execute(
                    f"""
                    UPDATE ai_column_definitions
                    SET name = %s,
                        slug = %s,
                        owner_user_id = %s,
                        view_scope = %s,
                        recruiter_filter_id = %s,
                        prompt_template = %s,
                        mode = %s,
                        output_schema = %s::jsonb,
                        required_fields = %s::jsonb,
                        only_run_if = %s::jsonb,
                        context_inputs = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                      AND {_definition_where_clause(current_user)}
                      AND COALESCE(is_archived, FALSE) = FALSE
                    RETURNING id
                    """,
                    (
                        body.name.strip(),
                        slug,
                        current_user.id,
                        resolved_scope,
                        resolved_recruiter,
                        body.prompt_template.strip(),
                        body.mode,
                        json.dumps(output_schema, ensure_ascii=False),
                        json.dumps(required_fields, ensure_ascii=False),
                        json.dumps(only_run_if, ensure_ascii=False),
                        json.dumps(body.context_inputs or {}, ensure_ascii=False),
                        body.id,
                        current_user.id,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="AI column not found")
                definition_id = int(row[0])
            else:
                cur.execute(
                    """
                    INSERT INTO ai_column_definitions (
                        name, slug, owner_user_id, view_scope, recruiter_filter_id,
                        prompt_template, mode, output_schema, required_fields, only_run_if, context_inputs
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                    ON CONFLICT (COALESCE(owner_user_id, 0), view_scope, COALESCE(recruiter_filter_id, 0), slug) WHERE COALESCE(is_archived, FALSE) = FALSE
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        prompt_template = EXCLUDED.prompt_template,
                        mode = EXCLUDED.mode,
                        output_schema = EXCLUDED.output_schema,
                        required_fields = EXCLUDED.required_fields,
                        only_run_if = EXCLUDED.only_run_if,
                        context_inputs = EXCLUDED.context_inputs
                    RETURNING id
                    """,
                    (
                        body.name.strip(),
                        slug,
                        current_user.id,
                        resolved_scope,
                        resolved_recruiter,
                        body.prompt_template.strip(),
                        body.mode,
                        json.dumps(output_schema, ensure_ascii=False),
                        json.dumps(required_fields, ensure_ascii=False),
                        json.dumps(only_run_if, ensure_ascii=False),
                        json.dumps(body.context_inputs or {}, ensure_ascii=False),
                    ),
                )
                definition_id = int(cur.fetchone()[0])
            conn.commit()
    _clear_list_cache()
    return {"id": definition_id, "name": body.name.strip(), "slug": slug}


@router.post("/ai-columns/test")
def test_ai_column(
    body: AiColumnTestRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    profile = _fetch_profile(body.candidate_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate not found")
    ai_values = _fetch_ai_context_for_candidate(
        body.candidate_id,
        current_user,
        body.view_scope,
        body.recruiter_filter_id,
    )
    context = build_candidate_context(
        profile,
        ai_values=ai_values,
        role_context=_fetch_role_context(body.role_id, current_user),
        context_inputs=body.context_inputs or {},
    )
    # Only guard on required_fields if the user explicitly set them.
    # Prompt tokens are NOT auto-required — missing fields just render as empty
    # strings (Clay behavior: always run every selected row).
    required_fields = list(body.required_fields or [])
    if required_fields:
        ok_to_run, missing = evaluate_required_fields(required_fields, context)
        if not ok_to_run:
            return {
                "status": "skipped",
                "missing_required_fields": missing,
                "preview": {
                    "primary_output": "",
                    "outputs": {},
                    "details": {
                        "response": "",
                        "reasoning": "",
                        "confidence": "low",
                        "steps": [],
                        "sources": [],
                        "rendered_prompt": fill_prompt_template(body.prompt_template, context),
                    },
                },
            }
    result = _run_ai_task(
        prompt_template=body.prompt_template,
        mode=body.mode,
        output_schema=[_model_to_dict(item) for item in body.output_schema],
        context=context,
    )
    return {"status": "completed", "preview": result}


@router.post("/ai-columns/run")
async def run_ai_column(
    body: AiColumnRunRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    definition = _fetch_definition_by_id(body.column_definition_id, current_user=current_user)
    if not definition:
        raise HTTPException(status_code=404, detail="AI column not found")
    candidate_ids = await _resolve_run_candidate_ids(current_user, body)
    if body.selection_mode == "selected_ids" and not candidate_ids:
        raise HTTPException(
            status_code=400,
            detail="No candidate ids selected. Select rows in the table, then run the column again.",
        )
    if body.selection_mode == "all_filtered" and not candidate_ids:
        raise HTTPException(
            status_code=400,
            detail="No candidates match the current filters for this run.",
        )
    payload = _model_to_dict(body)
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_column_runs (
                    column_definition_id, selection_mode, selection_payload, total,
                    completed, failed, skipped, status, created_at, updated_at
                )
                VALUES (%s, %s, %s::jsonb, %s, 0, 0, 0, 'queued', NOW(), NOW())
                RETURNING id
                """,
                (
                    body.column_definition_id,
                    body.selection_mode,
                    json.dumps(payload, ensure_ascii=False),
                    len(candidate_ids),
                ),
            )
            run_id = int(cur.fetchone()[0])
            conn.commit()
    _clear_list_cache()
    _spawn_run_thread(run_id, current_user, candidate_ids, body.column_definition_id, body.role_id)
    return {"run_id": run_id, "status": "queued", "total": len(candidate_ids)}


@router.get("/ai-columns/{column_definition_id}/cells/{candidate_id}")
def get_ai_column_cell_detail(
    column_definition_id: int,
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    definition = _fetch_definition_by_id(column_definition_id, current_user=current_user)
    if not definition:
        raise HTTPException(status_code=404, detail="AI column not found")
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT primary_output, outputs, details, status, error_message, completed_at, updated_at
                FROM ai_column_cells
                WHERE column_definition_id = %s AND candidate_id = %s
                """,
                (column_definition_id, candidate_id),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="AI column cell not found")
    return {
        "column_definition_id": column_definition_id,
        "candidate_id": candidate_id,
        "primary_output": row[0] or "",
        "outputs": _parse_jsonish(row[1], {}),
        "details": _parse_jsonish(row[2], {}),
        "status": row[3] or "idle",
        "error_message": row[4] or "",
        "completed_at": row[5].isoformat() if row[5] else None,
        "updated_at": row[6].isoformat() if row[6] else None,
    }


@router.delete("/ai-columns/{column_definition_id}")
def delete_ai_column(
    column_definition_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """
    Hide an AI column immediately and cancel active runs.

    Soft-archive avoids racing background workers that may still be writing cells.
    Archived definitions are filtered from all list/fetch routes.
    """
    where_def = _definition_where_clause(current_user)
    own_params = (column_definition_id, current_user.id)
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id FROM ai_column_definitions
                        WHERE id = %s AND {where_def}
                        """,
                        own_params,
                    )
                    if cur.fetchone() is None:
                        raise HTTPException(status_code=404, detail="AI column not found")

                    cur.execute(
                        """
                        UPDATE ai_column_runs
                        SET status = 'canceled',
                            completed_at = COALESCE(completed_at, NOW()),
                            updated_at = NOW()
                        WHERE column_definition_id = %s
                          AND status IN ('queued', 'running')
                        """,
                        (column_definition_id,),
                    )
                    cur.execute(
                        f"""
                        UPDATE ai_column_definitions
                        SET is_archived = TRUE,
                            updated_at = NOW()
                        WHERE id = %s AND {where_def}
                          AND COALESCE(is_archived, FALSE) = FALSE
                        RETURNING id
                        """,
                        own_params,
                    )
                    row = cur.fetchone()
                conn.commit()
            except HTTPException:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("DELETE /ai-columns/%s failed", column_definition_id)
        raise HTTPException(
            status_code=500,
            detail=(str(exc) or "Delete failed")[:800],
        ) from exc
    if not row:
        logger.warning(
            "DELETE /ai-columns/%s: ownership check passed but definition row missing after deletes",
            column_definition_id,
        )
        raise HTTPException(status_code=404, detail="AI column not found")
    _clear_list_cache()
    logger.info("DELETE /ai-columns/%s ok (user_id=%s)", column_definition_id, current_user.id)
    return {"deleted": True}

import asyncio
import logging
import os
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
from datetime import time as dt_time
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from backend.api import deps, schemas
from backend.db.connection import get_db_connection, return_db_connection, get_db_connection_context
from backend.services.call_artifacts import (
    extract_transcript_text,
    is_placeholder_summary,
    transcript_preview,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Call cadence ─────────────────────────────────────────────────────────────
# 5 attempts: Day 1 (second half), Day 2 (first half), Day 4 (second half),
# Day 7 (first half), Day 10 (second half). After the 5th failed attempt the
# system auto-marks the call "Unreachable" and stops scheduling.
FIRST_ATTEMPT_TITLE = "Call 1 - Day 1 - Second Half"

# (title_prefix, delay_days_to_next, next_title)
CALL_SEQUENCE_STEPS = [
    ("Call 1", 1, "Call 2 - Day 2 - First Half"),
    ("Call 2", 2, "Call 3 - Day 4 - Second Half"),
    ("Call 3", 3, "Call 4 - Day 7 - First Half"),
    ("Call 4", 3, "Call 5 - Day 10 - Second Half"),
]
FINAL_ATTEMPT_PREFIX = "Call 5"
UNREACHABLE_OUTCOME = "Unreachable"

# Recruiter outcomes that mean "the call did not connect" — cadence continues.
# Legacy outcomes are kept so historical rows keep progressing.
FAILED_CALL_OUTCOMES = {
    "Not Connected",
    "Not Connected - Not Reachable",
    "Left Voicemail",
    "No Answer",
}
# Connected outcomes that end the cadence (candidate leaves the calling list).
TERMINAL_CALL_OUTCOMES = {
    "Connected - Interested",
    "Connected - Not Interested",
}
FOLLOWUP_CALL_OUTCOMES = {"Connected - Follow-up", "Connected - Follow up later"}
FOLLOWUP_TASK_TITLE = "Follow-up Call"
WRONG_NUMBER_OUTCOME = "Wrong Number"
NOT_INTERESTED_OUTCOME = "Connected - Not Interested"
# Must match the RECRUITMENT_STAGES value in frontend/src/components/StatusDropdown.jsx.
NOT_INTERESTED_CANDIDATE_STATUS = "Not Interested"

# Candidate-status values that mean "this candidate is done" — once set, from
# ANY UI surface (Roles, Talent Pool, the Calls list's own status dropdown, or
# the call-log modal's "Candidate Status" dropdown — all of these hit the same
# POST /candidates/{id}/status endpoint), any other still-pending call rows for
# them stop showing up in the active calling loop. They're resolved to
# 'completed' rather than deleted, so they stay visible as history.
# Must match RECRUITMENT_STAGES values in frontend/src/components/StatusDropdown.jsx.
TERMINAL_CANDIDATE_STATUSES = {
    "not interested",
    "high ctc",
    "shared with customer",
    "for future",
    "shortlist - rejected",
    "duplicate",
    "rejected",
}

# ── Slicer (date-range / outcome) filters for the Calls workspace ────────────
# "Wrong Number" belongs to no group on purpose: those rows only show under
# All Outcomes.
OUTCOME_GROUPS: Dict[str, frozenset] = {
    "connected": frozenset(TERMINAL_CALL_OUTCOMES),
    "followup": frozenset(FOLLOWUP_CALL_OUTCOMES),
    "not_connected": frozenset(FAILED_CALL_OUTCOMES | {UNREACHABLE_OUTCOME}),
}
RANGE_PRESETS = {"today", "yesterday", "last7", "last30", "custom"}
# Inclusive day-offsets back from today for the non-custom presets.
_RANGE_PRESET_DAYS = {"today": 0, "yesterday": 1, "last7": 6, "last30": 29}


def validate_slicer_params(range_, date_from, date_to, outcome_group):
    if range_ and range_ not in RANGE_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown range preset: {range_}")
    if outcome_group and outcome_group not in OUTCOME_GROUPS:
        raise HTTPException(status_code=400, detail=f"Unknown outcome group: {outcome_group}")
    if range_ == "custom":
        if date_from is None or date_to is None:
            raise HTTPException(status_code=400, detail="Custom range requires date_from and date_to")
        if date_from > date_to:
            raise HTTPException(status_code=400, detail="date_from must not be after date_to")


def resolve_range_bounds(range_, date_from, date_to, today: date):
    """Inclusive (start, end) date bounds for a range preset, or None."""
    if not range_:
        return None
    if range_ == "custom":
        return (date_from, date_to)
    if range_ == "yesterday":
        return (today - timedelta(days=1), today - timedelta(days=1))
    return (today - timedelta(days=_RANGE_PRESET_DAYS[range_]), today)


def call_matches_slicer(call, bounds, outcome_set, *, use_completed_at: bool):
    """In-memory-cache-path predicate. MUST stay behaviorally identical to the
    SQL produced by build_range_sql / the outcome ANY() clause."""
    if outcome_set is not None and call.get("outcome") not in outcome_set:
        return False
    if bounds is None:
        return True
    if use_completed_at:
        value = call.get("completed_at")
        if isinstance(value, datetime):
            value = value.date()
    else:
        value = call.get("due_date")
    return value is not None and bounds[0] <= value <= bounds[1]


def build_range_sql(range_, date_from, date_to, *, use_completed_at: bool, col_prefix: str = "c."):
    """SQL fragment (leading " AND ...") + params mirroring call_matches_slicer.
    due_date is a DATE (inclusive compare); completed_at is a TIMESTAMP, so the
    upper bound is half-open on the next day (same idiom as browse.py)."""
    if not range_:
        return "", []
    col = f"{col_prefix}completed_at" if use_completed_at else f"{col_prefix}due_date"
    params: List[Any] = []
    if range_ == "custom":
        lo, hi = "%s::date", "%s::date"
        params = [date_from.isoformat(), date_to.isoformat()]
    elif range_ == "today":
        lo, hi = "CURRENT_DATE", "CURRENT_DATE"
    elif range_ == "yesterday":
        lo, hi = "CURRENT_DATE - 1", "CURRENT_DATE - 1"
    else:
        lo, hi = f"CURRENT_DATE - {_RANGE_PRESET_DAYS[range_]}", "CURRENT_DATE"
    if use_completed_at:
        return f" AND {col} >= {lo} AND {col} < {hi} + INTERVAL '1 day'", params
    return f" AND {col} >= {lo} AND {col} <= {hi}", params


def next_sequence_step(task_title: Optional[str]):
    """Returns (delay_days, next_title) for the attempt after task_title,
    or None when there is no scheduled next attempt (final attempt, follow-up
    tasks, or unrecognized titles)."""
    current_title = task_title or "Call 1"
    for step_prefix, step_delay, next_title in CALL_SEQUENCE_STEPS:
        if current_title.startswith(step_prefix):
            return step_delay, next_title
    return None

_calls_schema_ready = False
_calls_schema_lock = threading.Lock()

# In-memory Calls cache
_calls_cache: Optional[List[dict]] = None
_calls_lock = threading.RLock()
_call_lists_cache: Optional[List[dict]] = None
_cache_refresh_lock = threading.Lock()
_cache_refreshing = False
# Monotonically increasing counter — incremented on every eviction.
# warm_call_caches() captures this before its DB round-trip and
# discards results if the counter changed (eviction happened mid-flight).
_cache_generation: int = 0
# When the caches were last warmed from the DB. Gunicorn workers do NOT share
# memory: a mutation handled by worker A only invalidates A's cache, so worker
# B would serve stale counts forever. Serving memory only while fresh makes a
# stale worker fall back to the DB (always correct) and re-warm itself.
_cache_warmed_at: float = 0.0
_CACHE_FRESH_SECONDS = int(os.getenv("CALLS_CACHE_FRESH_SECONDS", "15"))


def calls_cache_is_fresh() -> bool:
    return (time.time() - _cache_warmed_at) < _CACHE_FRESH_SECONDS

# In-memory Stats cache
_stats_cache: dict[str, dict] = {}
_stats_cache_ts: dict[str, float] = {}
_STATS_TTL = 30 # seconds

CALLS_SELECT_QUERY = """
    SELECT
        c.id,
        c.candidate_id,
        c.list_id,
        c.status,
        c.outcome,
        c.notes,
        c.duration,
        c.due_date,
        c.created_at,
        c.task_title,
        cand.name AS candidate_name,
        cand.headline AS candidate_title,
        cand.mobile_phone AS candidate_phone,
        c.completed_at,
        c.recording_url,
        c.transcript,
        c.summary,
        cl.created_by,
        c.plivo_status,
        c.plivo_call_uuid,
        c.plivo_transaction_id,
        c.plivo_virtual_number,
        c.plivo_endpoint_username,
        c.plivo_recruiter_email,
        c.recording_source,
        c.recording_synced_at,
        c.due_time,
        COALESCE(cand.mobile_phone_wrong, FALSE),
        cand.notes,
        COALESCE(NULLIF(TRIM(cand.status), ''), 'To be started'),
        cand.linkedin,
        c.sentiment,
        c.sentiment_reason,
        COALESCE(c.likely_voicemail, FALSE),
        COALESCE(cand.cadence_paused, FALSE)
    FROM calls c
    JOIN call_lists cl ON c.list_id = cl.id
    JOIN candidates cand ON c.candidate_id = cand.id
"""


def call_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "candidate_id": row[1],
        "list_id": row[2],
        "status": row[3],
        "outcome": row[4],
        "notes": row[5],
        "duration": row[6],
        "due_date": row[7],
        "created_at": row[8],
        "task_title": row[9],
        "candidate_name": row[10],
        "candidate_title": row[11],
        "candidate_company": "",
        "candidate_phone": row[12],
        "completed_at": row[13],
        "recording_url": row[14],
        "transcript": row[15],
        "summary": row[16],
        "created_by": (row[17] or "").strip().lower(),
        "plivo_status": row[18],
        "plivo_call_uuid": row[19],
        "plivo_transaction_id": row[20],
        "plivo_virtual_number": row[21],
        "plivo_endpoint_username": row[22],
        "plivo_recruiter_email": row[23],
        "recording_source": row[24],
        "recording_synced_at": row[25],
        "due_time": row[26],
        "candidate_phone_wrong": bool(row[27]),
        "candidate_notes": row[28],
        "candidate_status": row[29],
        "candidate_linkedin": row[30],
        "sentiment": row[31],
        "sentiment_reason": row[32],
        "likely_voicemail": bool(row[33]),
        "cadence_paused": bool(row[34]),
    }


def invalidate_calls_cache():
    """Clear the stats cache and kick off a background cache refresh.
    Always uses a fresh pooled connection - never the write connection
    (which may still have open cursors or be mid-transaction).
    """
    global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts, _cache_generation
    with _calls_lock:
        _cache_generation += 1
        _calls_cache = None
        _call_lists_cache = None
        _stats_cache = {}
        _stats_cache_ts = {}
    refresh_call_caches_async()


def evict_call_from_cache(call_id: int):
    """Synchronously remove a single deleted call from _calls_cache so that
    GET /calls, GET /calls/lists, and GET /calls/stats immediately reflect the
    deletion — before the async background refresh finishes.
    Also bumps _cache_generation so any in-flight background refresh
    that read stale (pre-delete) data will discard its results.
    """
    global _calls_cache, _stats_cache, _stats_cache_ts, _cache_generation
    with _calls_lock:
        _cache_generation += 1
        if _calls_cache is not None:
            _calls_cache = [c for c in _calls_cache if c.get("id") != call_id]
        _stats_cache = {}
        _stats_cache_ts = {}


def evict_call_list_from_cache(list_id: int):
    """Synchronously remove a deleted call list and all its calls from the
    in-memory caches so every GET endpoint immediately sees up-to-date data.
    Also bumps _cache_generation so any in-flight background refresh
    that read stale (pre-delete) data will discard its results.
    """
    global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts, _cache_generation
    with _calls_lock:
        _cache_generation += 1
        if _calls_cache is not None:
            _calls_cache = [c for c in _calls_cache if c.get("list_id") != list_id]
        if _call_lists_cache is not None:
            _call_lists_cache = [l for l in _call_lists_cache if l.get("id") != list_id]
        _stats_cache = {}
        _stats_cache_ts = {}

def get_call_list_owner(current_user: schemas.User) -> str:
    owner = (current_user.email or current_user.username or "").strip().lower()
    if not owner:
        raise HTTPException(status_code=400, detail="Current user is missing an email")
    return owner


def load_calls_cache_data(conn) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(f"{CALLS_SELECT_QUERY} ORDER BY c.due_date ASC, c.created_at DESC")
        rows = cur.fetchall()
        return [call_row_to_dict(row) for row in rows]


def load_call_lists_cache_data(conn) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, created_at, LOWER(COALESCE(created_by, ''))
            FROM call_lists
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "created_by": row[3],
            }
            for row in rows
        ]


def build_call_initiation_error(
    code: str,
    message: str,
    *,
    action_label: Optional[str] = None,
    action_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "action_label": action_label,
        "action_url": action_url,
        "metadata": metadata or {},
    }

def ensure_list_exists_for_owner(cur, list_id: int, owner: str):
    cur.execute(
        """
        SELECT id
        FROM call_lists
        WHERE id = %s
          AND LOWER(COALESCE(created_by, '')) = %s
        """,
        (list_id, owner),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Call list not found")

def bulk_load_calls_cache(shared_conn=None):
    """Warms the calls cache from DB."""
    global _calls_cache
    
    def _do_load(conn):
        global _calls_cache
        try:
            data = load_calls_cache_data(conn)
            with _calls_lock:
                _calls_cache = data
            print(f"DEBUG: Bulk-warmed {len(data)} calls into memory.")
        except Exception as e:
            print(f"WARNING: Failed to warm calls cache: {e}")

    if shared_conn:
        _do_load(shared_conn)
    else:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if conn:
                _do_load(conn)


def bulk_load_call_lists_cache(shared_conn=None):
    """Warms the call lists cache from DB."""
    global _call_lists_cache

    def _do_load(conn):
        global _call_lists_cache
        try:
            data = load_call_lists_cache_data(conn)
            with _calls_lock:
                _call_lists_cache = data
            print(f"DEBUG: Bulk-warmed {len(data)} call lists into memory.")
        except Exception as e:
            print(f"WARNING: Failed to warm call lists cache: {e}")

    if shared_conn:
        _do_load(shared_conn)
    else:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if conn:
                _do_load(conn)


def warm_call_caches(shared_conn=None):
    global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts

    def _warm_from_connection(conn):
        global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts, _cache_generation, _cache_warmed_at
        # Snapshot generation BEFORE reading from DB.
        # If an eviction (delete) happens while we're querying,
        # _cache_generation will be > gen_before, so we discard stale results.
        with _calls_lock:
            gen_before = _cache_generation

        call_lists_data = load_call_lists_cache_data(conn)
        calls_data = load_calls_cache_data(conn)

        with _calls_lock:
            if _cache_generation != gen_before:
                # An eviction happened while we were reading — our data is stale.
                # Signal the runner to re-schedule a fresh refresh.
                print("DEBUG: Background refresh discarded (eviction mid-flight). Will rerun.")
                return True  # signal: needs rerun
            _call_lists_cache = call_lists_data
            _calls_cache = calls_data
            _stats_cache = {}
            _stats_cache_ts = {}
            _cache_warmed_at = time.time()
        print(f"DEBUG: Bulk-warmed {len(call_lists_data)} call lists and {len(calls_data)} calls into memory.")
        return False  # signal: all good

    ensure_calls_schema_ready()
    rerun_needed = False
    if shared_conn:
        rerun_needed = _warm_from_connection(shared_conn) or False
    else:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if conn:
                rerun_needed = _warm_from_connection(conn) or False
    return rerun_needed


def refresh_call_caches_async():
    global _cache_refreshing
    with _cache_refresh_lock:
        if _cache_refreshing:
            return
        _cache_refreshing = True

    def _runner():
        global _cache_refreshing
        rerun = False
        try:
            rerun = warm_call_caches() or False
        finally:
            with _cache_refresh_lock:
                _cache_refreshing = False
        # If warm_call_caches detected an eviction mid-flight, schedule a fresh run.
        # We do this AFTER clearing _cache_refreshing so the new run isn't blocked.
        if rerun:
            refresh_call_caches_async()

    threading.Thread(target=_runner, daemon=True).start()


def get_cached_call_lists(owner: str) -> Optional[List[dict]]:
    with _calls_lock:
        if _call_lists_cache is None or _calls_cache is None:
            return None
        # Stale-while-revalidate: a stale cache is still served (the caller
        # kicks a background rewarm) — blocking on the remote DB here made
        # every page load pay ~0.7s per request.

        pending_counts: dict[int, int] = {}
        completed_counts: dict[int, int] = {}
        total_counts: dict[int, int] = {}
        for call in _calls_cache:
            if call.get("created_by") != owner:
                continue
            list_id = call.get("list_id")
            if list_id is None:
                continue
            total_counts[list_id] = total_counts.get(list_id, 0) + 1
            if call.get("status") == "pending":
                pending_counts[list_id] = pending_counts.get(list_id, 0) + 1
            elif call.get("status") == "completed":
                completed_counts[list_id] = completed_counts.get(list_id, 0) + 1

        return [
            {
                "id": item["id"],
                "name": item["name"],
                "created_at": item["created_at"],
                "candidate_count": pending_counts.get(item["id"], 0),
                "pending_count": pending_counts.get(item["id"], 0),
                "completed_count": completed_counts.get(item["id"], 0),
                "total_count": total_counts.get(item["id"], 0),
            }
            for item in _call_lists_cache
            if item.get("created_by") == owner
        ]


def get_calls_db_connection():
    return get_db_connection(validate=True, register_pgvector=False)


# ── Inbound call persistence (called from the Plivo webhooks) ────────────────
# Every write keys on plivo_call_uuid: Plivo retries callbacks and may deliver
# the same one repeatedly, so these must be safe to run more than once.

def _inbound_conn():
    ensure_calls_schema_ready()
    return get_db_connection(validate=False, register_pgvector=False)


def record_inbound_call(call_uuid, from_number, to_number, call_status) -> None:
    """Upsert the inbound row and match the caller to a candidate by number."""
    if not call_uuid:
        return
    conn = _inbound_conn()
    if not conn:
        return
    from backend.integrations import plivo_service
    try:
        normalized = plivo_service.normalize_number(from_number) or from_number
        digits = "".join(ch for ch in (normalized or "") if ch.isdigit())[-10:]
        with conn.cursor() as cur:
            candidate_id = None
            if digits:
                # Match on the last 10 digits so +91/0 prefixes and spacing
                # variations still resolve to the same person.
                cur.execute(
                    """
                    SELECT id FROM candidates
                    WHERE RIGHT(REGEXP_REPLACE(COALESCE(NULLIF(TRIM(mobile_phone), ''),
                                                        NULLIF(TRIM(phone), '')), '[^0-9]', '', 'g'), 10) = %s
                      AND COALESCE(is_archived, FALSE) = FALSE
                    ORDER BY id LIMIT 1
                    """,
                    (digits,),
                )
                row = cur.fetchone()
                candidate_id = row[0] if row else None
            cur.execute(
                """
                INSERT INTO inbound_calls (candidate_id, from_number, to_number,
                                           plivo_call_uuid, call_status, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (plivo_call_uuid) DO UPDATE
                   SET call_status = EXCLUDED.call_status,
                       candidate_id = COALESCE(inbound_calls.candidate_id, EXCLUDED.candidate_id)
                """,
                (candidate_id, normalized or from_number, to_number, call_uuid, call_status),
            )
        conn.commit()
        invalidate_calls_cache()
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to record inbound call %s: %s", call_uuid, exc)
    finally:
        return_db_connection(conn)


def record_inbound_dial_result(call_uuid, dial_status, b_leg_uuid) -> None:
    """Who (if anyone) answered the ring-all leg."""
    if not call_uuid:
        return
    conn = _inbound_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            answered = dial_status == "completed" and bool(b_leg_uuid)
            cur.execute(
                """
                UPDATE inbound_calls
                   SET dial_status = %s,
                       status = CASE WHEN %s THEN 'answered' ELSE status END,
                       answered_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE answered_at END
                 WHERE plivo_call_uuid = %s
                """,
                (dial_status, answered, answered, call_uuid),
            )
        conn.commit()
        invalidate_calls_cache()
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to record dial result for %s: %s", call_uuid, exc)
    finally:
        return_db_connection(conn)


def attach_inbound_recording(call_uuid, recording_url) -> None:
    if not call_uuid or not recording_url:
        return
    conn = _inbound_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE inbound_calls SET recording_url = %s WHERE plivo_call_uuid = %s",
                (recording_url, call_uuid),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to attach voicemail to %s: %s", call_uuid, exc)
    finally:
        return_db_connection(conn)


def finalize_inbound_call(call_uuid, call_status, hangup_cause, duration) -> None:
    if not call_uuid:
        return
    conn = _inbound_conn()
    if not conn:
        return
    try:
        try:
            duration_int = int(duration or 0)
        except (TypeError, ValueError):
            duration_int = 0
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE inbound_calls
                   SET call_status = %s, hangup_cause = %s, duration = %s
                 WHERE plivo_call_uuid = %s
                """,
                (call_status, hangup_cause, duration_int, call_uuid),
            )
        conn.commit()
        invalidate_calls_cache()
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to finalize inbound call %s: %s", call_uuid, exc)
    finally:
        return_db_connection(conn)


def fetch_call_by_id(cur, call_id: int, owner: Optional[str] = None) -> Optional[dict]:
    query = f"{CALLS_SELECT_QUERY} WHERE c.id = %s"
    params: list[Any] = [call_id]
    if owner:
        query += " AND LOWER(COALESCE(cl.created_by, '')) = %s"
        params.append(owner)
    cur.execute(query, params)
    row = cur.fetchone()
    return call_row_to_dict(row) if row else None


def call_artifacts_need_repair(call_data: Optional[dict]) -> bool:
    if not isinstance(call_data, dict):
        return False

    transcript = (call_data.get("transcript") or "").strip()
    summary = (call_data.get("summary") or "").strip()

    if not transcript:
        return True
    if not summary:
        return True
    if is_placeholder_summary(summary):
        return True
    return False


class CallListCreate(BaseModel):
    name: str


class CallListResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    candidate_count: int = 0  # pending count — kept for backward compat
    pending_count: int = 0
    completed_count: int = 0
    total_count: int = 0


class AddCandidatesRequest(BaseModel):
    candidate_ids: List[int]
    list_id: int


class CallInitiateRequest(BaseModel):
    call_id: int
    dial_mode: str = "voip"
    plivo_username: Optional[str] = None


class CallUpdate(BaseModel):
    status: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None
    duration: Optional[int] = None
    due_date: Optional[date] = None
    due_time: Optional[dt_time] = None
    task_title: Optional[str] = None
    recording_url: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    # Legacy fields kept for backwards compatibility — no longer used.
    followup_due_date: Optional[date] = None
    followup_due_time: Optional[dt_time] = None


class CallResponse(BaseModel):
    id: int
    candidate_id: int
    list_id: int
    status: str
    outcome: Optional[str]
    notes: Optional[str]
    duration: int
    due_date: date
    created_at: datetime
    task_title: Optional[str]
    candidate_name: str
    candidate_title: Optional[str]
    candidate_company: Optional[str]
    candidate_phone: Optional[str]
    completed_at: Optional[datetime] = None
    recording_url: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    plivo_status: Optional[str] = None
    plivo_call_uuid: Optional[str] = None
    plivo_transaction_id: Optional[str] = None
    plivo_virtual_number: Optional[str] = None
    plivo_endpoint_username: Optional[str] = None
    plivo_recruiter_email: Optional[str] = None
    recording_source: Optional[str] = None
    recording_synced_at: Optional[datetime] = None
    due_time: Optional[dt_time] = None
    candidate_phone_wrong: Optional[bool] = False
    candidate_notes: Optional[str] = None
    candidate_status: Optional[str] = None
    candidate_linkedin: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_reason: Optional[str] = None
    likely_voicemail: Optional[bool] = False
    cadence_paused: Optional[bool] = False


def ensure_calls_schema_ready(force: bool = False):
    global _calls_schema_ready

    if _calls_schema_ready and not force:
        return

    with _calls_schema_lock:
        if _calls_schema_ready and not force:
            return

        conn = get_calls_db_connection()
        if not conn:
            raise RuntimeError("Database connection failed")

        cur = None
        try:
            legacy_provider_prefix = "fre" + "jun"
            cur = conn.cursor()

            # Fast path: the remote DB costs ~0.6s per statement, and the full
            # DDL below is ~50 statements (~30s). One sentinel round-trip checks
            # whether the NEWEST migration columns already exist — if so the
            # whole barrage is skipped. Keep the sentinel columns in sync with
            # the latest ALTERs whenever new DDL is added below.
            if not force:
                cur.execute("""
                    SELECT
                        EXISTS (SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'calls' AND column_name = 'due_time')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'candidates' AND column_name = 'mobile_phone_wrong')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'calls' AND column_name = 'likely_voicemail')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'candidates' AND column_name = 'cadence_paused')
                        AND EXISTS (SELECT 1 FROM information_schema.tables
                                    WHERE table_name = 'inbound_calls')
                        AND EXISTS (SELECT 1 FROM information_schema.tables
                                    WHERE table_name = 'plivo_endpoints')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'plivo_endpoints' AND column_name = 'in_call_since')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'calls' AND column_name = 'dial_token')
                        AND EXISTS (SELECT 1 FROM information_schema.tables
                                    WHERE table_name = 'plivo_app_state')
                        AND EXISTS (SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'plivo_endpoints' AND column_name = 'env_key')
                        AND NOT EXISTS (SELECT 1 FROM pg_constraint
                                        WHERE conname = 'unique_candidate_list')
                """)
                sentinel_row = cur.fetchone()
                if sentinel_row and sentinel_row[0]:
                    conn.rollback()
                    _calls_schema_ready = True
                    return

            cur.execute("""
                CREATE TABLE IF NOT EXISTS call_lists (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(255)
                );
            """)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'call_lists_name_user_unique'
                    ) THEN
                        ALTER TABLE call_lists DROP CONSTRAINT IF EXISTS call_lists_name_key;
                        ALTER TABLE call_lists ADD CONSTRAINT call_lists_name_user_unique UNIQUE (name, created_by);
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id SERIAL PRIMARY KEY,
                    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                    list_id INTEGER NOT NULL REFERENCES call_lists(id) ON DELETE CASCADE,
                    status VARCHAR(50) DEFAULT 'pending',
                    outcome VARCHAR(100),
                    notes TEXT,
                    duration INTEGER DEFAULT 0,
                    task_title VARCHAR(255),
                    due_date DATE DEFAULT CURRENT_DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
            """)
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS outcome VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS notes TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS duration INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS task_title VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS due_date DATE DEFAULT CURRENT_DATE;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_url TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS transcript TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS summary TEXT;")
            # AI-derived call sentiment (Positive/Neutral/Negative) + a one-line reason,
            # generated alongside the summary — distinct from `outcome`, which is the
            # recruiter's own manual call disposition.
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS sentiment VARCHAR(20);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS sentiment_reason TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_status VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_call_uuid VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_transaction_id VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_recruiter_email VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_virtual_number VARCHAR(50);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_endpoint_username VARCHAR(255);")
            # Per-dial-attempt identifier. Attribution used to match the most
            # recently updated row for a SIP username, which picks the wrong row
            # whenever two attempts share a username (two recruiters on the
            # shared fallback endpoint, or one recruiter redialling quickly) and
            # silently writes a recording onto the wrong candidate.
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS dial_token VARCHAR(64);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_source VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_synced_at TIMESTAMP;")
            # Follow-up calls carry an exact slot (date + time); cadence calls only a date.
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS due_time TIME;")
            # Post-call heuristic (transcript phrases + short one-sided duration) — Plivo
            # doesn't support real AMD on this app's browser-SDK/XML-Dial call flow, so
            # this is a suggestion the recruiter confirms, not a live detection signal.
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS likely_voicemail BOOLEAN DEFAULT FALSE;")
            # Wrong-number tag lives on the candidate so every call list sees it.
            cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS mobile_phone_wrong BOOLEAN DEFAULT FALSE;")
            # Manual "Stop cadence" toggle — same candidate-level scope as
            # mobile_phone_wrong, gates the next-attempt insert the same way.
            cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS cadence_paused BOOLEAN DEFAULT FALSE;")
            cur.execute(f"""
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_status') THEN
                        UPDATE calls SET plivo_status = COALESCE(NULLIF(plivo_status, ''), {legacy_provider_prefix}_status);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = 'call_status') THEN
                        UPDATE calls SET plivo_status = COALESCE(NULLIF(plivo_status, ''), call_status);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_call_id') THEN
                        UPDATE calls SET plivo_call_uuid = COALESCE(NULLIF(plivo_call_uuid, ''), {legacy_provider_prefix}_call_id);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_event_id') THEN
                        UPDATE calls SET plivo_call_uuid = COALESCE(NULLIF(plivo_call_uuid, ''), {legacy_provider_prefix}_event_id);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_transaction_id') THEN
                        UPDATE calls SET plivo_transaction_id = COALESCE(NULLIF(plivo_transaction_id, ''), {legacy_provider_prefix}_transaction_id);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_recruiter_email') THEN
                        UPDATE calls SET plivo_recruiter_email = COALESCE(NULLIF(plivo_recruiter_email, ''), {legacy_provider_prefix}_recruiter_email);
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'calls' AND column_name = '{legacy_provider_prefix}_virtual_number') THEN
                        UPDATE calls SET plivo_virtual_number = COALESCE(NULLIF(plivo_virtual_number, ''), {legacy_provider_prefix}_virtual_number);
                    END IF;
                END $$;
            """)
            # ── Inbound calls ────────────────────────────────────────────────
            # Deliberately NOT rows in `calls`: that table is an outbound cadence
            # task list, and Due Today counts, call-list counts and the
            # next-attempt sequencing all assume outbound semantics. Inbound
            # events living there would corrupt every one of those.
            # plivo_call_uuid is UNIQUE because Plivo retries webhooks and may
            # deliver the same callback more than once — it is the idempotency key.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inbound_calls (
                    id SERIAL PRIMARY KEY,
                    candidate_id INTEGER REFERENCES candidates(id) ON DELETE SET NULL,
                    from_number VARCHAR(32) NOT NULL,
                    to_number VARCHAR(32),
                    plivo_call_uuid VARCHAR(128) UNIQUE,
                    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    answered_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    answered_at TIMESTAMP,
                    duration INTEGER DEFAULT 0,
                    hangup_cause VARCHAR(120),
                    call_status VARCHAR(50),
                    dial_status VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'pending',
                    note TEXT,
                    resolved_at TIMESTAMP,
                    resolved_by VARCHAR(255),
                    resolved_call_id INTEGER REFERENCES calls(id) ON DELETE SET NULL,
                    recording_url TEXT,
                    transcript TEXT,
                    created_by VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ix_inbound_calls_status ON inbound_calls (status, received_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_inbound_calls_candidate ON inbound_calls (candidate_id);")

            # One SIP endpoint per recruiter. A single shared endpoint cannot be
            # forked to ("ring everyone") and gives no way to tell who answered.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS plivo_endpoints (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    endpoint_id VARCHAR(128),
                    username VARCHAR(128) NOT NULL,
                    password VARCHAR(128) NOT NULL,
                    app_id VARCHAR(128),
                    last_registered_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Set while this recruiter is on a call, so inbound "ring everyone"
            # skips them instead of ringing over a live conversation.
            cur.execute("ALTER TABLE plivo_endpoints ADD COLUMN IF NOT EXISTS in_call_since TIMESTAMP;")

            # Which Plivo Application each environment owns. Previously two JSON
            # files under data/, which are gitignored and wiped by every Azure
            # deploy — so hosted always started blank and created a brand new
            # Application per deploy (20 accumulated), while endpoints stayed
            # bound to older ones carrying dead tunnel URLs.
            #
            # env_key keeps local and hosted from claiming each other's app:
            # this Postgres is shared between them.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS plivo_app_state (
                    kind VARCHAR(32) NOT NULL,
                    env_key VARCHAR(255) NOT NULL,
                    app_id VARCHAR(128) NOT NULL,
                    answer_url TEXT,
                    username VARCHAR(128),
                    password VARCHAR(128),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (kind, env_key)
                );
            """)

            # Endpoints are bound to an app_id, and app_ids are per-environment,
            # so the same recruiter needs a distinct endpoint per environment.
            # Without this a laptop-created endpoint gets handed to hosted along
            # with the laptop's ngrok answer URL.
            cur.execute("ALTER TABLE plivo_endpoints ADD COLUMN IF NOT EXISTS env_key VARCHAR(255) NOT NULL DEFAULT 'legacy';")
            cur.execute("ALTER TABLE plivo_endpoints DROP CONSTRAINT IF EXISTS plivo_endpoints_user_id_key;")
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_plivo_endpoints_user_env
                ON plivo_endpoints (user_id, env_key);
            """)

            cur.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at() RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS sync_updated_at_calls ON calls;
                CREATE TRIGGER sync_updated_at_calls
                BEFORE UPDATE ON calls
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            """)
            cur.execute("""
                DROP INDEX IF EXISTS idx_calls_candidate_list_unique;
            """)
            # Legacy one-call-per-candidate-per-list constraint — incompatible
            # with the multi-attempt cadence (Call 1..5 rows per candidate).
            cur.execute("ALTER TABLE calls DROP CONSTRAINT IF EXISTS unique_candidate_list;")
            cur.execute("DROP INDEX IF EXISTS unique_candidate_list;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_list_id ON calls(list_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_candidate_id ON calls(candidate_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_status_due_date ON calls(status, due_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_created_at ON calls(created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_call_lists_created_by ON call_lists(created_by);")
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_call_id;")
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_event_id;")
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_transaction_id;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_call_uuid ON calls(plivo_call_uuid);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_transaction_id ON calls(plivo_transaction_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_endpoint_username ON calls(plivo_endpoint_username);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_dial_token ON calls(dial_token);")
            cur.execute(f"""
                ALTER TABLE calls
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_status,
                    DROP COLUMN IF EXISTS call_status,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_call_id,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_event_id,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_transaction_id,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_recruiter_email,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_virtual_number,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_summary_url,
                    DROP COLUMN IF EXISTS {legacy_provider_prefix}_link;
            """)
            cur.execute(f"DROP TABLE IF EXISTS {legacy_provider_prefix}_oauth_credentials;")
            conn.commit()
            _calls_schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            if cur:
                cur.close()
            return_db_connection(conn)


@router.get("/lists", response_model=List[CallListResponse])
def get_call_lists(current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    cached_lists = get_cached_call_lists(owner)
    if cached_lists is not None:
        if not calls_cache_is_fresh():
            refresh_call_caches_async()
        return cached_lists

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        refresh_call_caches_async()

        cur = conn.cursor()
        cur.execute("""
            SELECT cl.id, cl.name, cl.created_at,
                   COUNT(CASE WHEN c.status = 'pending' THEN c.id END) AS pending_count,
                   COUNT(CASE WHEN c.status = 'completed' THEN c.id END) AS completed_count,
                   COUNT(c.id) AS total_count
            FROM call_lists cl
            LEFT JOIN calls c ON cl.id = c.list_id
            WHERE LOWER(COALESCE(cl.created_by, '')) = %s
            GROUP BY cl.id, cl.name, cl.created_at
            ORDER BY cl.created_at DESC
        """, (owner,))
        return [
            {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "candidate_count": row[3],
                "pending_count": row[3],
                "completed_count": row[4],
                "total_count": row[5],
            }
            for row in cur.fetchall()
        ]
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.post("/lists", response_model=CallListResponse)
def create_call_list(request: CallListCreate, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    list_name = (request.name or "").strip()
    if not list_name:
        raise HTTPException(status_code=400, detail="List name is required")

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        # Case-insensitive duplicate check + insert in ONE round trip — each
        # statement costs ~0.6s against the remote DB.
        cur.execute(
            """
            WITH existing AS (
                SELECT id FROM call_lists
                WHERE LOWER(TRIM(name)) = LOWER(%s)
                  AND LOWER(COALESCE(created_by, '')) = %s
                LIMIT 1
            ), inserted AS (
                INSERT INTO call_lists (name, created_by)
                SELECT %s, %s
                WHERE NOT EXISTS (SELECT 1 FROM existing)
                RETURNING id, name, created_at
            )
            SELECT id, name, created_at FROM inserted
            """,
            (list_name, owner, list_name, owner),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="A list with this name already exists")
        conn.commit()
        result = {"id": row[0], "name": row[1], "created_at": row[2], "candidate_count": 0, "pending_count": 0, "completed_count": 0, "total_count": 0}
        if cur:
            cur.close()
        return_db_connection(conn)
        conn = None
        cur = None
        invalidate_calls_cache()
        return result
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        error_msg = str(e).lower()
        if "unique" in error_msg and "name" in error_msg:
            raise HTTPException(status_code=400, detail="A list with this name already exists")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


@router.post("/add-candidates")
def add_candidates_to_list(
    request: AddCandidatesRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        if not request.candidate_ids:
            return {"success": True, "added_count": 0}

        cur = conn.cursor()
        # Ownership check + duplicate count + insert in ONE round trip — each
        # statement costs ~0.6s against the remote DB. CTEs read a consistent
        # snapshot, so `duplicates` counts rows as they were BEFORE the insert.
        cur.execute(
            """
            WITH target_list AS (
                SELECT id FROM call_lists
                WHERE id = %s
                  AND LOWER(COALESCE(created_by, '')) = %s
            ),
            duplicates AS (
                SELECT COUNT(DISTINCT candidate_id) AS dup_count
                FROM calls
                WHERE list_id = %s
                  AND candidate_id = ANY(%s::int[])
                  AND status = 'pending'
            ),
            callable_ids AS (
                SELECT c_id
                FROM UNNEST(%s::int[]) AS c_id
                WHERE EXISTS (
                    SELECT 1 FROM candidates cand
                    WHERE cand.id = c_id
                      AND COALESCE(NULLIF(TRIM(cand.mobile_phone), ''),
                                   NULLIF(TRIM(cand.phone), '')) IS NOT NULL
                )
            ),
            inserted AS (
                INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
                SELECT DISTINCT c_id, %s, 'pending', CURRENT_DATE, %s
                FROM callable_ids
                WHERE EXISTS (SELECT 1 FROM target_list)
                  AND (SELECT dup_count FROM duplicates) = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM calls existing
                      WHERE existing.candidate_id = c_id AND existing.list_id = %s
                        AND existing.status = 'pending'
                  )
                RETURNING id
            )
            SELECT
                (SELECT COUNT(*) FROM target_list) AS list_found,
                (SELECT dup_count FROM duplicates) AS duplicate_count,
                (SELECT COUNT(*) FROM inserted) AS inserted_count,
                (SELECT COUNT(DISTINCT c_id) FROM UNNEST(%s::int[]) AS c_id) AS requested_count,
                (SELECT COUNT(DISTINCT c_id) FROM callable_ids) AS callable_count
            """,
            (
                request.list_id, owner,
                request.list_id, request.candidate_ids,
                request.candidate_ids,
                request.list_id, FIRST_ATTEMPT_TITLE, request.list_id,
                request.candidate_ids,
            ),
        )
        (
            list_found,
            duplicate_count,
            inserted_count,
            requested_count,
            callable_count,
        ) = cur.fetchone()

        if not list_found:
            raise HTTPException(status_code=404, detail="Call list not found")

        duplicate_count = duplicate_count or 0
        if duplicate_count > 0:
            requested_count = len(set(request.candidate_ids))
            if requested_count == 1:
                detail = "Candidate is already in this call list"
            elif duplicate_count >= requested_count:
                detail = "All selected candidates are already in this call list"
            else:
                detail = f"{duplicate_count} of {requested_count} selected candidates are already in this call list"
            raise HTTPException(status_code=400, detail=detail)

        # Candidates with no phone number are silently unusable in a call list —
        # they render as "N/A" rows the recruiter can never dial. Skip them at
        # insert time and report how many, so the count is explainable rather
        # than the list quietly coming up short.
        skipped_no_phone = max(0, (requested_count or 0) - (callable_count or 0))
        if not inserted_count and skipped_no_phone:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No phone number available for the selected candidate"
                    if skipped_no_phone == 1
                    else f"None of the {skipped_no_phone} selected candidates have a phone number"
                ),
            )

        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
        return {
            "success": True,
            "added_count": inserted_count,
            "skipped_no_phone": skipped_no_phone,
        }
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


@router.get("", response_model=List[CallResponse])
def get_calls(
    status: Optional[str] = None,
    list_id: Optional[int] = None,
    due_filter: Optional[str] = None,
    range_: Optional[str] = Query(None, alias="range"),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    outcome_group: Optional[str] = None,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    validate_slicer_params(range_, date_from, date_to, outcome_group)
    owner = get_call_list_owner(current_user)
    # Completed views slice on completion date; pending/due views on due_date.
    slice_on_completed = status == "completed"
    outcome_set = OUTCOME_GROUPS.get(outcome_group) if outcome_group else None

    # ── HIGH PERFORMANCE IN-MEMORY FILTERING ──
    # Stale-while-revalidate: the remote DB costs ~0.7s per request, so any
    # populated cache is served instantly; when it is past its freshness
    # window a background rewarm is kicked off (bounded staleness — a sibling
    # gunicorn worker's mutation shows up after the ~1-2s rewarm lands).
    # In-process mutations evict/None the cache, which forces the SQL path.
    cached_result = None
    cache_stale = False
    with _calls_lock:
        if _calls_cache is not None:
            cache_stale = not calls_cache_is_fresh()
            data = [c for c in _calls_cache if c.get("created_by") == owner]
            if status:
                data = [c for c in data if c.get("status") == status]
            if list_id:
                data = [c for c in data if c.get("list_id") == list_id]
            
            if due_filter == "today":
                today = date.today()
                data = [c for c in data if c.get("due_date") and c["due_date"] <= today]
            elif due_filter == "upcoming":
                today = date.today()
                data = [c for c in data if c.get("due_date") and c["due_date"] > today]

            bounds = resolve_range_bounds(range_, date_from, date_to, date.today())
            if bounds is not None or outcome_set is not None:
                data = [
                    c for c in data
                    if call_matches_slicer(c, bounds, outcome_set, use_completed_at=slice_on_completed)
                ]

            # Custom Sorting for UX consistency
            if status == "completed":
                data.sort(key=lambda x: x.get("completed_at") or datetime.min, reverse=True)
            else:
                # Key tip: Python's sort is stable, we sort by created_at then due_date+due_time
                data.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)
                data.sort(key=lambda x: (x.get("due_date") or date.min, x.get("due_time") or dt_time.min))

            cached_result = data

    if cached_result is not None:
        if cache_stale:
            refresh_call_caches_async()
        return cached_result

    # ── FALLBACK TO DB IF CACHE EMPTY ──
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        refresh_call_caches_async()
        
        cur = conn.cursor()
        query = f"{CALLS_SELECT_QUERY} WHERE LOWER(COALESCE(cl.created_by, '')) = %s"
        params: list[Any] = [owner]

        if status:
            query += " AND c.status = %s"
            params.append(status)
        if list_id:
            query += " AND c.list_id = %s"
            params.append(list_id)

        if due_filter == "today":
            query += " AND c.due_date <= CURRENT_DATE"
        elif due_filter == "upcoming":
            query += " AND c.due_date > CURRENT_DATE"

        range_sql, range_params = build_range_sql(
            range_, date_from, date_to, use_completed_at=slice_on_completed
        )
        query += range_sql
        params.extend(range_params)
        if outcome_set is not None:
            query += " AND c.outcome = ANY(%s)"
            params.append(sorted(outcome_set))

        if status == "completed":
            query += " ORDER BY c.completed_at DESC NULLS LAST"
        else:
            query += " ORDER BY c.due_date ASC NULLS FIRST, c.due_time ASC NULLS FIRST, c.created_at DESC NULLS LAST"

        cur.execute(query, params)
        rows = cur.fetchall()
        return [call_row_to_dict(row) for row in rows]
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur: cur.close()
        return_db_connection(conn)


@router.patch("/{call_id}")
def update_call(call_id: int, request: CallUpdate, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    # A follow-up outcome must carry the exact slot the candidate reappears in.
    if (
        (request.outcome or "").strip() in FOLLOWUP_CALL_OUTCOMES
        and (request.due_date is None or request.due_time is None)
    ):
        raise HTTPException(
            status_code=400,
            detail="Follow-up date and time are required for a follow-up outcome",
        )

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        fields = []
        params = []
        if request.status is not None:
            fields.append("status = %s")
            params.append(request.status)
            if request.status == "completed":
                fields.append("completed_at = NOW()")
            else:
                fields.append("completed_at = NULL")
        if request.outcome is not None:
            fields.append("outcome = %s")
            params.append(request.outcome)
        if request.notes is not None:
            fields.append("notes = %s")
            params.append(request.notes)
        if request.duration is not None:
            fields.append("duration = %s")
            params.append(request.duration)
        if request.due_date is not None:
            fields.append("due_date = %s")
            params.append(request.due_date)
        if request.due_time is not None:
            fields.append("due_time = %s")
            params.append(request.due_time)
        if request.task_title is not None:
            fields.append("task_title = %s")
            params.append(request.task_title.strip() or None)
        if request.recording_url is not None:
            fields.append("recording_url = %s")
            params.append(request.recording_url.strip() or None)
        if request.transcript is not None:
            fields.append("transcript = %s")
            params.append(request.transcript.strip() or None)
        if request.summary is not None:
            fields.append("summary = %s")
            params.append(request.summary.strip() or None)

        if not fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        params.extend([call_id, owner])
        cur.execute(
            f"""
            UPDATE calls
            SET {', '.join(fields)}, updated_at = NOW()
            WHERE id = %s
              AND list_id IN (
                  SELECT id FROM call_lists
                  WHERE LOWER(COALESCE(created_by, '')) = %s
              )
            RETURNING candidate_id, list_id, status, outcome, task_title
            """,
            params,
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Call task not found")

        candidate_id, list_id, current_status, current_outcome, task_title = row

        scheduled_next_title = None
        auto_unreachable = False
        wrong_number_tagged = False
        not_interested_removed = False

        if current_status == "completed" and current_outcome:
            if current_outcome in FAILED_CALL_OUTCOMES:
                current_title = task_title or "Call 1"
                if current_title.startswith(FINAL_ATTEMPT_PREFIX):
                    # 5th failed attempt — system tags the call Unreachable and
                    # the cadence stops (candidate moves to the cooldown pool).
                    cur.execute(
                        "UPDATE calls SET outcome = %s WHERE id = %s",
                        (UNREACHABLE_OUTCOME, call_id),
                    )
                    auto_unreachable = True
                else:
                    step = next_sequence_step(task_title)
                    if step:
                        # A wrong-number tag or a manual "Stop cadence" both pause
                        # scheduling until cleared (phone update, or "Continue").
                        # A missing number stops it too: the next attempt would
                        # just be another undiallable "N/A" row in Due Today.
                        cur.execute(
                            """
                            SELECT COALESCE(mobile_phone_wrong, FALSE),
                                   COALESCE(cadence_paused, FALSE),
                                   COALESCE(NULLIF(TRIM(mobile_phone), ''),
                                            NULLIF(TRIM(phone), '')) IS NULL AS no_phone
                            FROM candidates WHERE id = %s
                            """,
                            (candidate_id,),
                        )
                        phone_row = cur.fetchone()
                        blocked = bool(phone_row) and any(phone_row[:3])
                        if not blocked:
                            delay_days, next_title = step
                            cur.execute(
                                """
                                INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
                                VALUES (%s, %s, 'pending', CURRENT_DATE + %s * INTERVAL '1 day', %s)
                                """,
                                (candidate_id, list_id, delay_days, next_title)
                            )
                            scheduled_next_title = next_title
            elif current_outcome in FOLLOWUP_CALL_OUTCOMES:
                # Each follow-up gets its own new row — same as every other
                # retry — so this row's outcome/recording/transcript stays
                # intact as history instead of being overwritten when the
                # follow-up call happens (initiate_call reuses plivo_transaction_id
                # for the SAME row, which previously clobbered the prior recording).
                cur.execute(
                    """
                    INSERT INTO calls (candidate_id, list_id, status, due_date, due_time, task_title)
                    VALUES (%s, %s, 'pending', %s, %s, %s)
                    """,
                    (candidate_id, list_id, request.due_date, request.due_time, FOLLOWUP_TASK_TITLE),
                )
                scheduled_next_title = FOLLOWUP_TASK_TITLE
            elif current_outcome == WRONG_NUMBER_OUTCOME:
                # Tag the number on the candidate so every list sees it; the
                # cadence pauses until an alternate number is saved.
                cur.execute(
                    "UPDATE candidates SET mobile_phone_wrong = TRUE, updated_at = NOW() WHERE id = %s",
                    (candidate_id,),
                )
                wrong_number_tagged = True
            elif current_outcome == NOT_INTERESTED_OUTCOME:
                # Not interested ends the relationship outright, immediately —
                # not just "no next attempt scheduled". Set a terminal status
                # everywhere the candidate is visible and resolve any other
                # still-pending attempts for them (other call lists included) as
                # completed too, so they stop appearing in the active calling
                # loop but keep a visible record in Completed instead of just
                # vanishing.
                cur.execute(
                    "UPDATE candidates SET status = %s, updated_at = NOW() WHERE id = %s",
                    (NOT_INTERESTED_CANDIDATE_STATUS, candidate_id),
                )
                cur.execute(
                    """
                    UPDATE calls
                    SET status = 'completed', outcome = %s, completed_at = NOW(), updated_at = NOW()
                    WHERE candidate_id = %s
                      AND status = 'pending'
                      AND list_id IN (
                          SELECT id FROM call_lists
                          WHERE LOWER(COALESCE(created_by, '')) = %s
                      )
                    """,
                    (NOT_INTERESTED_OUTCOME, candidate_id, owner),
                )
                not_interested_removed = True
            # Connected - Interested ends the cadence: the call completes and
            # nothing new is scheduled.

        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
        if not_interested_removed:
            try:
                from backend.api.routes.candidates import invalidate_candidate_count_caches
                from backend.api.routes.roles import invalidate_role_detail_cache_for_candidate
                invalidate_candidate_count_caches(refresh_profile_ids=[candidate_id])
                invalidate_role_detail_cache_for_candidate(candidate_id)
            except Exception:
                pass
        return {
            "success": True,
            "scheduled_next_title": scheduled_next_title,
            "auto_unreachable": auto_unreachable,
            "wrong_number_tagged": wrong_number_tagged,
            "not_interested_removed": not_interested_removed,
        }
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


class CadenceToggleRequest(BaseModel):
    paused: bool


@router.post("/candidates/{candidate_id}/cadence")
async def set_candidate_cadence_paused(
    candidate_id: int,
    request: CadenceToggleRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Manual Stop/Continue cadence toggle — gates the next-attempt insert in
    update_call the same way the Wrong Number tag already does. Does not touch
    any currently-pending call row; it only pauses future auto-scheduling."""
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE candidates
            SET cadence_paused = %s, updated_at = NOW()
            WHERE id = %s
              AND EXISTS (
                  SELECT 1 FROM calls c
                  JOIN call_lists cl ON c.list_id = cl.id
                  WHERE c.candidate_id = candidates.id
                    AND LOWER(COALESCE(cl.created_by, '')) = %s
              )
            """,
            (request.paused, candidate_id, owner),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Candidate not found in your call lists")
        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
        return {"success": True, "paused": request.paused}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


class InboundResolveRequest(BaseModel):
    note: Optional[str] = None
    call_id: Optional[int] = None


class InboundManualRequest(BaseModel):
    from_number: str
    candidate_id: Optional[int] = None
    note: Optional[str] = None


@router.get("/inbound")
def list_inbound_calls(
    status: Optional[str] = None,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Inbound callbacks, newest first, joined to the matched candidate."""
    ensure_calls_schema_ready()
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        where, params = "", []
        normalized = (status or "").strip().lower()
        if normalized == "pending":
            where = "WHERE ic.status IN ('pending', 'answered')"
        elif normalized == "resolved":
            where = "WHERE ic.status = 'resolved'"
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT ic.id, ic.from_number, ic.received_at, ic.status, ic.note,
                       ic.duration, ic.recording_url, ic.dial_status, ic.answered_at,
                       ic.candidate_id, c.name,
                       COALESCE(NULLIF(TRIM(c.headline), ''), ''),
                       COALESCE(c.raw_fields->>'import_company', ''),
                       COALESCE(NULLIF(TRIM(c.status), ''), 'To be started'),
                       u.name
                  FROM inbound_calls ic
                  LEFT JOIN candidates c ON c.id = ic.candidate_id
                  LEFT JOIN users u ON u.id = ic.answered_by_user_id
                  {where}
                 ORDER BY ic.received_at DESC
                 LIMIT 200
                """,
                params,
            )
            rows = cur.fetchall()
        items = [
            {
                "id": r[0], "from_number": r[1],
                "received_at": r[2].isoformat() if r[2] else None,
                "status": r[3], "note": r[4] or "", "duration": r[5] or 0,
                "recording_url": r[6], "dial_status": r[7],
                "answered_at": r[8].isoformat() if r[8] else None,
                "candidate_id": r[9], "candidate_name": r[10],
                "candidate_title": r[11], "candidate_company": r[12],
                "candidate_status": r[13], "answered_by": r[14],
                # No candidate match means the number isn't in the pool — shown
                # flagged rather than dropped, so a real callback is never lost.
                "is_unknown": r[9] is None,
            }
            for r in rows
        ]
        return {"items": items, "pending_count": sum(1 for i in items if i["status"] != "resolved")}
    finally:
        return_db_connection(conn)


@router.post("/inbound/{inbound_id}/callback-task")
def create_callback_task(
    inbound_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Return a real `calls` task to dial this callback through.

    "1-Click Call Back" used to hand the CallingModal the *candidate* id as if
    it were a call-task id, so /calls/initiate answered 404 "Call task not
    found" and the callback could never be placed. The outbound dial path needs
    an actual `calls` row, so reuse the candidate's open task when there is one
    and create a task otherwise.
    """
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT candidate_id FROM inbound_calls WHERE id = %s", (inbound_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Inbound call not found")
            candidate_id = row[0]
            if not candidate_id:
                raise HTTPException(
                    status_code=400,
                    detail="This number is not in the talent pool — dial it manually.",
                )

            # Prefer an open task so the callback closes out real cadence work
            # instead of leaving a duplicate behind.
            #
            # Both lookups are scoped to the caller's own lists. fetch_call_by_id
            # below filters by list owner, so a task sitting in a colleague's
            # list would come back as null and leave the dialer with no call to
            # place — candidates routinely appear in several recruiters' lists.
            cur.execute(
                """
                SELECT c.id FROM calls c
                  JOIN call_lists cl ON cl.id = c.list_id
                 WHERE c.candidate_id = %s AND c.status IN ('pending', 'in_progress')
                   AND LOWER(COALESCE(cl.created_by, '')) = %s
                 ORDER BY c.due_date ASC NULLS LAST, c.created_at ASC
                 LIMIT 1
                """,
                (candidate_id, owner),
            )
            found = cur.fetchone()
            call_id = found[0] if found else None

            if not call_id:
                # Fall back to the candidate's most recent list *of ours* so the
                # new task stays with their existing calling history.
                cur.execute(
                    """
                    SELECT c.list_id FROM calls c
                      JOIN call_lists cl ON cl.id = c.list_id
                     WHERE c.candidate_id = %s
                       AND LOWER(COALESCE(cl.created_by, '')) = %s
                     ORDER BY c.created_at DESC LIMIT 1
                    """,
                    (candidate_id, owner),
                )
                list_row = cur.fetchone()
                list_id = list_row[0] if list_row else None

                if not list_id:
                    # `calls.list_id` is NOT NULL, so a candidate who has never
                    # been called needs somewhere to live.
                    cur.execute(
                        "SELECT id FROM call_lists WHERE name = %s AND LOWER(COALESCE(created_by, '')) = %s LIMIT 1",
                        ("Inbound Callbacks", owner),
                    )
                    existing_list = cur.fetchone()
                    if existing_list:
                        list_id = existing_list[0]
                    else:
                        cur.execute(
                            "INSERT INTO call_lists (name, created_by) VALUES (%s, %s) RETURNING id",
                            ("Inbound Callbacks", owner),
                        )
                        list_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO calls (candidate_id, list_id, status, task_title, due_date)
                    VALUES (%s, %s, 'pending', %s, CURRENT_DATE)
                    RETURNING id
                    """,
                    (candidate_id, list_id, "Inbound callback"),
                )
                call_id = cur.fetchone()[0]

            call = fetch_call_by_id(cur, call_id, owner)
        conn.commit()
        invalidate_calls_cache()
        return {"success": True, "call": call}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.exception("Failed to build a callback task for inbound %s", inbound_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)


@router.post("/inbound/{inbound_id}/resolve")
def resolve_inbound_call(
    inbound_id: int,
    request: InboundResolveRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Mark a callback done.

    Deliberately outcome-independent: per the product owner, "the moment you
    complete the callback, irrespective of whether it is connected or not, it
    should be marked as callback completed and the number here will reduce."
    So this never inspects the call outcome — attempting the callback clears it.
    """
    ensure_calls_schema_ready()
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE inbound_calls
                   SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP,
                       resolved_by = %s, resolved_call_id = COALESCE(%s, resolved_call_id),
                       note = COALESCE(NULLIF(%s, ''), note)
                 WHERE id = %s
             RETURNING id
                """,
                (current_user.email, request.call_id, (request.note or "").strip(), inbound_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Inbound call not found")
        conn.commit()
        invalidate_calls_cache()
        return {"success": True}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)


@router.post("/inbound/manual")
def log_inbound_call_manually(
    request: InboundManualRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Backs '+ Log Incoming Call' — a callback that reached the recruiter
    outside the system still belongs in the queue."""
    ensure_calls_schema_ready()
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        from backend.integrations import plivo_service

        number = plivo_service.normalize_number(request.from_number) or request.from_number
        with conn.cursor() as cur:
            candidate_id = request.candidate_id
            if candidate_id is None:
                digits = "".join(ch for ch in number if ch.isdigit())[-10:]
                if digits:
                    cur.execute(
                        """
                        SELECT id FROM candidates
                        WHERE RIGHT(REGEXP_REPLACE(COALESCE(NULLIF(TRIM(mobile_phone), ''),
                                                            NULLIF(TRIM(phone), '')), '[^0-9]', '', 'g'), 10) = %s
                          AND COALESCE(is_archived, FALSE) = FALSE
                        ORDER BY id LIMIT 1
                        """,
                        (digits,),
                    )
                    row = cur.fetchone()
                    candidate_id = row[0] if row else None
            cur.execute(
                """
                INSERT INTO inbound_calls (candidate_id, from_number, status, note, created_by)
                VALUES (%s, %s, 'pending', %s, %s)
                RETURNING id
                """,
                (candidate_id, number, (request.note or "").strip() or None, current_user.email),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        invalidate_calls_cache()
        return {"success": True, "id": new_id, "candidate_id": candidate_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)


def inbound_pending_count() -> int:
    """Unresolved inbound callbacks — drives the sidebar bubble.

    Folded into /stats rather than given its own endpoint: Calls.jsx already
    polls stats every 15s, and this codebase has a history of request storms.
    The value rides the stats cache, so cache hits cost nothing.
    """
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM inbound_calls WHERE status <> 'resolved'")
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
    except Exception:
        return 0
    finally:
        return_db_connection(conn)


@router.get("/stats")
def get_call_stats(
    range_: Optional[str] = Query(None, alias="range"),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    outcome_group: Optional[str] = None,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    validate_slicer_params(range_, date_from, date_to, outcome_group)
    owner = get_call_list_owner(current_user)
    outcome_set = OUTCOME_GROUPS.get(outcome_group) if outcome_group else None
    # Stats are cached per owner + slicer params (invalidation always clears
    # the whole dict, so the composite key needs no targeted eviction).
    stats_key = f"{owner}|{range_ or ''}|{date_from or ''}|{date_to or ''}|{outcome_group or ''}"

    # ── INSTANT STATS CACHE ──
    # Stale-while-revalidate (see get_calls): serve any populated cache
    # immediately and rewarm in the background when past the fresh window.
    now = time.time()
    if (
        stats_key in _stats_cache
        and (now - _stats_cache_ts.get(stats_key, 0) < _STATS_TTL)
    ):
        if not calls_cache_is_fresh():
            refresh_call_caches_async()
        return _stats_cache[stats_key]

    stats_cache_stale = False
    with _calls_lock:
        if _calls_cache is not None and _call_lists_cache is not None:
            stats_cache_stale = not calls_cache_is_fresh()
            today = date.today()
            bounds = resolve_range_bounds(range_, date_from, date_to, today)
            due_today = 0
            upcoming = 0
            completed = 0

            for call in _calls_cache:
                if call.get("created_by") != owner:
                    continue
                status = call.get("status")
                if status == "completed":
                    # Completed counts slice on completion date AND outcome.
                    if call_matches_slicer(call, bounds, outcome_set, use_completed_at=True):
                        completed += 1
                    continue
                # Only pending calls are bucketed into due_today / upcoming.
                # This MUST match the SQL path below (status = 'pending')
                # otherwise the counts flicker whenever the stats request
                # switches between the in-memory cache and the DB fallback.
                if status != "pending":
                    continue
                due_date = call.get("due_date")
                if not due_date:
                    continue
                # Pending rows have no outcome yet, so the outcome filter is
                # NOT applied here (it would always zero these buckets) —
                # only the date range slices pending calls.
                if not call_matches_slicer(call, bounds, None, use_completed_at=False):
                    continue
                if due_date <= today:
                    due_today += 1
                else:
                    upcoming += 1

            active_lists = sum(1 for item in _call_lists_cache if item.get("created_by") == owner)
            stats = {
                "due_today": due_today,
                "upcoming": upcoming,
                "completed": completed,
                "active_lists": active_lists,
                "inbound_pending": inbound_pending_count(),
            }
            if len(_stats_cache) > 64:
                _stats_cache.clear()
                _stats_cache_ts.clear()
            _stats_cache[stats_key] = stats
            _stats_cache_ts[stats_key] = now
            if stats_cache_stale:
                refresh_call_caches_async()
            return stats

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        refresh_call_caches_async()

        cur = conn.cursor()
        # Slicer predicates mirror the in-memory path above: pending buckets
        # slice on due_date (never on outcome — pending rows have none),
        # completed slices on completed_at and outcome.
        due_sql, due_params = build_range_sql(range_, date_from, date_to, use_completed_at=False, col_prefix="")
        comp_sql, comp_params = build_range_sql(range_, date_from, date_to, use_completed_at=True, col_prefix="")
        outcome_sql = " AND outcome = ANY(%s)" if outcome_set is not None else ""
        outcome_params = [sorted(outcome_set)] if outcome_set is not None else []
        query_params = (
            [owner]
            + due_params
            + due_params
            + comp_params + outcome_params
        )
        cur.execute(f"""
            WITH owned_lists AS (
                SELECT id
                FROM call_lists
                WHERE LOWER(COALESCE(created_by, '')) = %s
            ),
            call_counts AS (
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date <= CURRENT_DATE{due_sql}) AS due_today,
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date > CURRENT_DATE{due_sql}) AS upcoming,
                    COUNT(*) FILTER (WHERE status = 'completed'{comp_sql}{outcome_sql}) AS completed
                FROM calls
                WHERE list_id IN (SELECT id FROM owned_lists)
            ),
            list_counts AS (
                SELECT COUNT(*) AS total_lists
                FROM owned_lists
            )
            SELECT
                call_counts.due_today,
                call_counts.upcoming,
                call_counts.completed,
                list_counts.total_lists
            FROM call_counts
            CROSS JOIN list_counts
        """, query_params)
        row = cur.fetchone()
        stats = {
            "due_today": row[0] or 0,
            "upcoming": row[1] or 0,
            "completed": row[2] or 0,
            "active_lists": row[3] or 0,
            "inbound_pending": inbound_pending_count(),
        }
        if len(_stats_cache) > 64:
            _stats_cache.clear()
            _stats_cache_ts.clear()
        _stats_cache[stats_key] = stats
        _stats_cache_ts[stats_key] = now
        return stats
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.delete("/{call_id}")
def delete_call(call_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM calls
            WHERE id = %s
              AND list_id IN (
                  SELECT id FROM call_lists
                  WHERE LOWER(COALESCE(created_by, '')) = %s
              )
            RETURNING list_id
            """,
            (call_id, owner),
        )
        deleted = cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Call task not found")

        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        evict_call_from_cache(call_id)
        refresh_call_caches_async()
        return {"success": True, "list_id": deleted[0]}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


@router.delete("/lists/{list_id}")
def delete_call_list(list_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM calls c
            JOIN call_lists cl ON c.list_id = cl.id
            WHERE c.list_id = %s
              AND LOWER(COALESCE(cl.created_by, '')) = %s
            """,
            (list_id, owner),
        )
        deleted_call_count = cur.fetchone()[0] or 0

        cur.execute(
            """
            DELETE FROM call_lists
            WHERE id = %s
              AND LOWER(COALESCE(created_by, '')) = %s
            RETURNING id
            """,
            (list_id, owner),
        )
        deleted = cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Call list not found")

        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        evict_call_list_from_cache(list_id)
        refresh_call_caches_async()
        return {"success": True, "deleted_call_count": deleted_call_count}
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)


@router.post("/initiate")
def initiate_call(
    request: CallInitiateRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)
    call_id = request.call_id

    candidate_name = None
    candidate_phone = None
    candidate_id = None
    recruiter_email = None
    transaction_id = None
    endpoint_username = (request.plivo_username or "").strip()
    configured_recruiter_email = (current_user.email or "").strip()

    try:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT candidate_id, list_id, plivo_recruiter_email, plivo_transaction_id
                    FROM calls
                    WHERE id = %s
                    """,
                    (call_id,),
                )
                call_row = cur.fetchone()
                if not call_row:
                    raise HTTPException(status_code=404, detail="Call task not found")

                candidate_id = call_row[0]
                list_id = call_row[1]
                recruiter_email = (call_row[2] or "").strip()
                transaction_id = (call_row[3] or "").strip() or f"call:{call_id}"

                cur.execute(
                    """
                    SELECT LOWER(COALESCE(created_by, ''))
                    FROM call_lists
                    WHERE id = %s
                      AND LOWER(COALESCE(created_by, '')) = %s
                    """,
                    (list_id, owner),
                )
                list_row = cur.fetchone()
                if not list_row:
                    raise HTTPException(status_code=404, detail="Call task not found")

                list_owner = (list_row[0] or "").strip()
                recruiter_email = recruiter_email or configured_recruiter_email or list_owner

                cur.execute(
                    """
                    SELECT name, mobile_phone
                    FROM candidates
                    WHERE id = %s
                    """,
                    (candidate_id,),
                )
                candidate_row = cur.fetchone()
                if not candidate_row:
                    raise HTTPException(status_code=404, detail="Candidate not found for call task")

                candidate_name = candidate_row[0]
                candidate_phone = candidate_row[1]

                cur.execute(
                    """
                    UPDATE calls
                    SET
                        plivo_transaction_id = COALESCE(NULLIF(plivo_transaction_id, ''), %s),
                        plivo_recruiter_email = COALESCE(NULLIF(plivo_recruiter_email, ''), %s),
                        plivo_endpoint_username = COALESCE(NULLIF(%s, ''), plivo_endpoint_username),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (transaction_id, recruiter_email or None, endpoint_username or None, call_id),
                )
                conn.commit()
    except psycopg2.Error as exc:
        logger.exception("Database error preparing call initiation for call %s", call_id)
        raise HTTPException(
            status_code=500,
            detail=build_call_initiation_error(
                "call_lookup_failed",
                "Unable to load this call task right now. Please retry.",
                action_label="Retry VoIP",
                metadata={"call_id": call_id, "db_error": exc.__class__.__name__},
            ),
        ) from exc

    if not candidate_phone:
        raise HTTPException(status_code=400, detail="Candidate has no phone number")
    if not recruiter_email:
        raise HTTPException(status_code=400, detail="Recruiter email missing for Plivo call initiation")
    if (request.dial_mode or "voip").strip().lower() != "voip":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "voip_only",
                "message": "This Calls flow supports Browser VoIP only.",
                "action_label": None,
                "action_url": None,
                "metadata": {"requested_dial_mode": request.dial_mode},
            },
        )

    from backend.integrations import plivo_service
    logger.info(
        "Prepared Plivo browser call for %s by recruiter %s using endpoint %s",
        candidate_phone,
        recruiter_email,
        endpoint_username or "unknown",
    )
    returned_virtual_number = plivo_service.normalize_number(plivo_service.PLIVO_NUMBER)
    returned_status = "pending"
    # Fresh per attempt, never reused: a redial must not be satisfiable by the
    # previous attempt's token. Hex only — Plivo requires X-PH-* header values
    # to be alphanumeric.
    dial_token = uuid.uuid4().hex

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE calls
            SET
                plivo_transaction_id = COALESCE(NULLIF(plivo_transaction_id, ''), %s),
                plivo_recruiter_email = COALESCE(NULLIF(plivo_recruiter_email, ''), %s),
                plivo_virtual_number = COALESCE(NULLIF(plivo_virtual_number, ''), %s),
                plivo_endpoint_username = COALESCE(NULLIF(%s, ''), plivo_endpoint_username),
                plivo_status = %s,
                dial_token = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                transaction_id,
                recruiter_email,
                returned_virtual_number,
                endpoint_username or None,
                returned_status,
                # Overwritten, not COALESCEd: each attempt needs its own token,
                # or a redial would be attributed to the first attempt's row.
                dial_token,
                call_id,
            ),
        )
        updated_call = fetch_call_by_id(cur, call_id, owner)
        conn.commit()
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
    except Exception as exc:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if cur:
            cur.close()
        if conn:
            return_db_connection(conn)

    return {
        "success": True,
        "message": "Call initiated",
        "dial_mode": "voip",
        "plivo_data": {
            "transaction_id": transaction_id,
            "virtual_number": returned_virtual_number,
            "endpoint_username": endpoint_username or None,
            "status": returned_status,
            "dial_token": dial_token,
        },
        "call": updated_call,
    }


@router.post("/{call_id}/sync-recording", response_model=CallResponse)
async def sync_call_recording(
    call_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    call = None
    try:
        cur = conn.cursor()
        call = fetch_call_by_id(cur, call_id, owner)
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)

    if not call:
        raise HTTPException(status_code=404, detail="Call task not found")

    if call.get("transcript") or call.get("summary"):
        logger.info(f"Returning cached insights for Call {call_id}")
        return call

    call_uuid = call.get("plivo_call_uuid") or call.get("plivo_transaction_id")
    from backend.integrations import plivo_service
    
    record_url = None
    if call_uuid and call_uuid in plivo_service.recordings:
        record_url = plivo_service.recordings[call_uuid]
        logger.info(f"Syncing call using Plivo recording for UUID: {call_uuid}")
    elif call_uuid and call.get("completed_at"):
        # The in-memory recordings map is lost on restart and the webhook can
        # be missed entirely (ngrok down) — recover via Plivo's REST API.
        record_url = await asyncio.to_thread(plivo_service.lookup_recording_url, call_uuid)
        if record_url:
            plivo_service.recordings[call_uuid] = record_url
            logger.info(f"Recovered recording for UUID {call_uuid} via Plivo REST lookup")

    if record_url:
        # Pass the recruiter-timed duration (already on this row) so a long
        # call routed through this manual-sync fallback also skips
        # gpt-4o-transcribe's silent-truncation risk on longer audio, same as
        # the webhook path does.
        await plivo_service.process_call_insights(call_uuid, record_url, duration_seconds=call.get("duration"))

        conn = get_calls_db_connection()
        if conn:
            cur = conn.cursor()
            updated_call = fetch_call_by_id(cur, call_id, owner)
            cur.close()
            return_db_connection(conn)
            invalidate_calls_cache()
            return updated_call

    raise HTTPException(
         status_code=404,
         detail="No Plivo recording callback has arrived for this call yet"
    )

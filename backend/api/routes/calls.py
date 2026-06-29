import json
import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from backend.api import deps, schemas
from backend.db.connection import get_db_connection, return_db_connection, get_db_connection_context
from backend.integrations.frejun import FreJunManager
from backend.services.ai_calls import process_call_audio
from backend.services.frejun_calls import (
    coalesce_text,
    digits_only,
    extract_payload_details,
    extract_transcript_text,
    build_summary_text,
    humanize_status,
    is_placeholder_summary,
    normalize_phone,
    prefer_better_summary,
    prefer_richer_text,
    select_best_call_log_result,
    transcript_preview,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_calls_schema_ready = False
_calls_schema_lock = threading.Lock()

# In-memory Calls cache
_calls_cache: Optional[List[dict]] = None
_calls_lock = threading.RLock()
_call_lists_cache: Optional[List[dict]] = None
_cache_refresh_lock = threading.Lock()
_cache_refreshing = False

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
        c.frejun_status,
        c.frejun_call_id,
        c.frejun_event_id,
        c.frejun_virtual_number,
        c.frejun_summary_url,
        c.frejun_link,
        c.recording_source,
        c.recording_synced_at,
        c.frejun_transaction_id,
        c.frejun_recruiter_email
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
        "frejun_status": row[18],
        "frejun_call_id": row[19],
        "frejun_event_id": row[20],
        "frejun_virtual_number": row[21],
        "frejun_summary_url": row[22],
        "frejun_link": row[23],
        "recording_source": row[24],
        "recording_synced_at": row[25],
        "frejun_transaction_id": row[26],
        "frejun_recruiter_email": row[27],
    }


def build_frejun_request_uri(request: Request) -> str:
    explicit_callback_url = (os.getenv("FREJUN_WEBHOOK_CALLBACK_URL") or "").strip()
    if explicit_callback_url:
        return explicit_callback_url

    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").strip()
    forwarded_host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").strip()
    if forwarded_proto and forwarded_host:
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        return f"{forwarded_proto}://{forwarded_host}{path}"

    return str(request.url)

def invalidate_calls_cache():
    global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts
    with _calls_lock:
        _calls_cache = None
        _call_lists_cache = None
        _stats_cache = {}
        _stats_cache_ts = {}

def get_call_list_owner(current_user: schemas.User) -> str:
    owner = (current_user.email or current_user.username or "").strip().lower()
    if not owner:
        raise HTTPException(status_code=400, detail="Current user is missing an email")
    return owner


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
        try:
            with conn.cursor() as cur:
                cur.execute(f"{CALLS_SELECT_QUERY} ORDER BY c.due_date ASC, c.created_at DESC")
                rows = cur.fetchall()
                data = [call_row_to_dict(row) for row in rows]
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
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, created_at, LOWER(COALESCE(created_by, ''))
                    FROM call_lists
                    ORDER BY created_at DESC
                    """
                )
                rows = cur.fetchall()
                data = [
                    {
                        "id": row[0],
                        "name": row[1],
                        "created_at": row[2],
                        "created_by": row[3],
                    }
                    for row in rows
                ]
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
    ensure_calls_schema_ready()
    bulk_load_call_lists_cache(shared_conn)
    bulk_load_calls_cache(shared_conn)


def refresh_call_caches_async():
    global _cache_refreshing
    with _cache_refresh_lock:
        if _cache_refreshing:
            return
        _cache_refreshing = True

    def _runner():
        global _cache_refreshing
        try:
            warm_call_caches()
        finally:
            with _cache_refresh_lock:
                _cache_refreshing = False

    threading.Thread(target=_runner, daemon=True).start()


def get_cached_call_lists(owner: str) -> Optional[List[dict]]:
    with _calls_lock:
        if _call_lists_cache is None or _calls_cache is None:
            return None

        pending_counts: dict[int, int] = {}
        for call in _calls_cache:
            if call.get("created_by") != owner or call.get("status") != "pending":
                continue
            list_id = call.get("list_id")
            if list_id is None:
                continue
            pending_counts[list_id] = pending_counts.get(list_id, 0) + 1

        return [
            {
                "id": item["id"],
                "name": item["name"],
                "created_at": item["created_at"],
                "candidate_count": pending_counts.get(item["id"], 0),
            }
            for item in _call_lists_cache
            if item.get("created_by") == owner
        ]


def get_calls_db_connection():
    return get_db_connection(validate=True, register_pgvector=False)


def fetch_call_by_id(cur, call_id: int, owner: Optional[str] = None) -> Optional[dict]:
    query = f"{CALLS_SELECT_QUERY} WHERE c.id = %s"
    params: list[Any] = [call_id]
    if owner:
        query += " AND LOWER(COALESCE(cl.created_by, '')) = %s"
        params.append(owner)
    cur.execute(query, params)
    row = cur.fetchone()
    return call_row_to_dict(row) if row else None


def find_call_match_for_frejun_payload(cur, payload_details: Dict[str, Any]) -> Optional[dict]:
    match_specs = [
        ("frejun_call_id", payload_details.get("call_id")),
        ("frejun_event_id", payload_details.get("event_id")),
        ("frejun_transaction_id", payload_details.get("transaction_id")),
    ]

    for column, value in match_specs:
        if not value:
            continue
        cur.execute(
            f"""
            {CALLS_SELECT_QUERY}
            WHERE c.{column} = %s
            ORDER BY c.created_at DESC
            LIMIT 1
            """,
            (value,),
        )
        row = cur.fetchone()
        if row:
            return call_row_to_dict(row)

    candidate_number = payload_details.get("candidate_number")
    candidate_digits = digits_only(candidate_number)
    if candidate_digits and len(candidate_digits) >= 10:
        last_ten = candidate_digits[-10:]
        cur.execute(
            f"""
            {CALLS_SELECT_QUERY}
            WHERE (
                regexp_replace(COALESCE(cand.mobile_phone, ''), '\\D', '', 'g') = %s
                OR RIGHT(regexp_replace(COALESCE(cand.mobile_phone, ''), '\\D', '', 'g'), 10) = %s
                OR c.frejun_virtual_number = %s
                OR RIGHT(regexp_replace(COALESCE(c.frejun_virtual_number, ''), '\\D', '', 'g'), 10) = %s
            )
            AND c.status IN ('completed', 'pending')
            ORDER BY 
                CASE WHEN c.transcript IS NULL THEN 0 ELSE 1 END,
                c.created_at DESC
            LIMIT 1
            """,
            (candidate_digits, last_ten, candidate_number, last_ten),
        )
        row = cur.fetchone()
        if row:
            return call_row_to_dict(row)

    return None


def build_call_log_payload(call_log_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": "call.summary",
        "call_id": call_log_result.get("call_id"),
        "event_id": call_log_result.get("event_id"),
        "candidate_number": call_log_result.get("candidate_number"),
        "candidate_name": call_log_result.get("candidate_name"),
        "virtual_number": call_log_result.get("virtual_number"),
        "call_status": call_log_result.get("status"),
        "duration": call_log_result.get("duration") or call_log_result.get("call_duration"),
        "recording_url": call_log_result.get("recording_url"),
        "summary_url": call_log_result.get("summary_url"),
        "link": call_log_result.get("link"),
        "transcript": call_log_result.get("call_transcript"),
        "ai_insights": call_log_result.get("ai_insights"),
        "call_outcome": call_log_result.get("call_outcome"),
        "call_notes": call_log_result.get("recruiter_notes"),
        "call_creator": call_log_result.get("recruiter"),
        "metadata": {
            "transaction_id": call_log_result.get("transaction_id"),
            "candidate_id": call_log_result.get("candidate_id"),
            "job_id": call_log_result.get("job_id"),
        },
    }


def persist_frejun_update(
    cur,
    *,
    call_id: int,
    existing_call: dict,
    payload_details: Dict[str, Any],
    recording_source: Optional[str] = None,
) -> dict:
    current_duration = int(existing_call.get("duration") or 0)
    next_duration = current_duration
    if payload_details.get("duration_seconds") is not None:
        next_duration = max(current_duration, int(payload_details["duration_seconds"]))

    is_terminal = bool(payload_details.get("is_terminal"))
    next_status = "completed" if is_terminal else existing_call.get("status")
    next_outcome = coalesce_text(payload_details.get("outcome"), existing_call.get("outcome"))
    next_notes = coalesce_text(existing_call.get("notes"), payload_details.get("notes"))
    next_recording_url = coalesce_text(existing_call.get("recording_url"), payload_details.get("recording_url"))
    next_transcript = prefer_richer_text(existing_call.get("transcript"), payload_details.get("transcript_text"))
    next_summary = prefer_better_summary(existing_call.get("summary"), payload_details.get("summary_text"))
    next_frejun_status = coalesce_text(payload_details.get("frejun_status"), existing_call.get("frejun_status"))
    next_frejun_call_id = coalesce_text(payload_details.get("call_id"), existing_call.get("frejun_call_id"))
    next_frejun_event_id = coalesce_text(payload_details.get("event_id"), existing_call.get("frejun_event_id"))
    next_frejun_virtual_number = coalesce_text(payload_details.get("virtual_number"), existing_call.get("frejun_virtual_number"))
    next_frejun_summary_url = coalesce_text(payload_details.get("summary_url"), existing_call.get("frejun_summary_url"))
    next_frejun_link = coalesce_text(payload_details.get("link"), existing_call.get("frejun_link"))
    next_recording_source = coalesce_text(recording_source, payload_details.get("recording_source"), existing_call.get("recording_source"))
    next_transaction_id = coalesce_text(payload_details.get("transaction_id"), existing_call.get("frejun_transaction_id"))
    next_recruiter_email = coalesce_text(payload_details.get("recruiter_email"), existing_call.get("frejun_recruiter_email"))
    should_stamp_recording_sync = bool(
        next_recording_url
        and (
            payload_details.get("recording_url")
            or payload_details.get("summary_text")
            or payload_details.get("transcript_text")
            or recording_source == "frejun_call_logs"
        )
    )

    cur.execute(
        """
        UPDATE calls
        SET
            status = %s,
            outcome = %s,
            notes = %s,
            duration = %s,
            completed_at = CASE WHEN %s THEN COALESCE(completed_at, NOW()) ELSE completed_at END,
            recording_url = %s,
            transcript = %s,
            summary = %s,
            frejun_status = %s,
            frejun_call_id = %s,
            frejun_event_id = %s,
            frejun_virtual_number = %s,
            frejun_summary_url = %s,
            frejun_link = %s,
            recording_source = %s,
            recording_synced_at = CASE WHEN %s THEN NOW() ELSE recording_synced_at END,
            frejun_transaction_id = %s,
            frejun_recruiter_email = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (
            next_status,
            next_outcome,
            next_notes,
            next_duration,
            is_terminal,
            next_recording_url,
            next_transcript,
            next_summary,
            next_frejun_status,
            next_frejun_call_id,
            next_frejun_event_id,
            next_frejun_virtual_number,
            next_frejun_summary_url,
            next_frejun_link,
            next_recording_source,
            should_stamp_recording_sync,
            next_transaction_id,
            next_recruiter_email,
            call_id,
        ),
    )

    return fetch_call_by_id(cur, call_id)


def maybe_process_call_audio(previous_call: dict, updated_call: dict, force_fallback: bool = False):
    previous_recording_url = previous_call.get("recording_url")
    next_recording_url = updated_call.get("recording_url")
    if not next_recording_url:
        return
    if previous_recording_url and not force_fallback:
        return
    if not call_artifacts_need_repair(updated_call):
        return
    process_call_audio(updated_call["id"], next_recording_url)


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
    candidate_count: int = 0


class AddCandidatesRequest(BaseModel):
    candidate_ids: List[int]
    list_id: int


class CallInitiateRequest(BaseModel):
    call_id: int
    dial_mode: str = "voip"


class CallUpdate(BaseModel):
    status: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None
    duration: Optional[int] = None
    due_date: Optional[date] = None
    task_title: Optional[str] = None
    recording_url: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None


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
    frejun_status: Optional[str] = None
    frejun_call_id: Optional[str] = None
    frejun_event_id: Optional[str] = None
    frejun_virtual_number: Optional[str] = None
    frejun_summary_url: Optional[str] = None
    frejun_link: Optional[str] = None
    recording_source: Optional[str] = None
    recording_synced_at: Optional[datetime] = None


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
            cur = conn.cursor()
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
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_status VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_call_id VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_event_id VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_transaction_id VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_recruiter_email VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_virtual_number VARCHAR(50);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_summary_url TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS frejun_link TEXT;")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_source VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_synced_at TIMESTAMP;")
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
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_list_id ON calls(list_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_candidate_id ON calls(candidate_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_status_due_date ON calls(status, due_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_created_at ON calls(created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_call_lists_created_by ON call_lists(created_by);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_frejun_call_id ON calls(frejun_call_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_frejun_event_id ON calls(frejun_event_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_frejun_transaction_id ON calls(frejun_transaction_id);")
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
                   COUNT(CASE WHEN c.status = 'pending' THEN c.id END) AS candidate_count
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
        cur.execute(
            """
            SELECT id
            FROM call_lists
            WHERE LOWER(TRIM(name)) = LOWER(%s)
              AND LOWER(COALESCE(created_by, '')) = %s
            LIMIT 1
            """,
            (list_name, owner),
        )
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="A list with this name already exists")

        cur.execute(
            "INSERT INTO call_lists (name, created_by) VALUES (%s, %s) RETURNING id, name, created_at",
            (list_name, owner),
        )
        row = cur.fetchone()
        conn.commit()
        invalidate_calls_cache()
        refresh_call_caches_async()
        return {"id": row[0], "name": row[1], "created_at": row[2], "candidate_count": 0}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        error_msg = str(e).lower()
        if "unique" in error_msg and "name" in error_msg:
            raise HTTPException(status_code=400, detail="A list with this name already exists")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
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
        ensure_list_exists_for_owner(cur, request.list_id, owner)

        # Check for duplicates
        cur.execute(
            "SELECT COUNT(1) FROM calls WHERE list_id = %s AND candidate_id = ANY(%s::int[])",
            (request.list_id, request.candidate_ids)
        )
        if cur.fetchone()[0] > 0:
            raise HTTPException(status_code=400, detail="He or she is already there.")

        cur.execute(
            """
            INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
            SELECT DISTINCT c_id, %s, 'pending', CURRENT_DATE, 'Call 1 - Day 1'
            FROM UNNEST(%s::int[]) AS c_id
            WHERE NOT EXISTS (
                SELECT 1 FROM calls existing
                WHERE existing.candidate_id = c_id AND existing.list_id = %s
            )
            RETURNING id
            """,
            (request.list_id, request.candidate_ids, request.list_id),
        )
        inserted_count = len(cur.fetchall())
        conn.commit()
        invalidate_calls_cache()
        refresh_call_caches_async()
        return {"success": True, "added_count": inserted_count}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.get("", response_model=List[CallResponse])
def get_calls(
    status: Optional[str] = None,
    list_id: Optional[int] = None,
    due_filter: Optional[str] = None,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    # ── HIGH PERFORMANCE IN-MEMORY FILTERING ──
    # If cache is available, filter in Python to avoid 5s SQL latency (Azure/SSL overhead)
    with _calls_lock:
        if _calls_cache is not None:
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

            # Custom Sorting for UX consistency
            if status == "completed":
                data.sort(key=lambda x: x.get("completed_at") or datetime.min, reverse=True)
            else:
                # Key tip: Python's sort is stable, we sort by created_at then due_date
                data.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)
                data.sort(key=lambda x: x.get("due_date") or date.min)
            
            return data

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

        if status == "completed":
            query += " ORDER BY c.completed_at DESC NULLS LAST"
        else:
            query += " ORDER BY c.due_date ASC NULLS FIRST, c.created_at DESC NULLS LAST"

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

        params.append(call_id)
        cur.execute(f"UPDATE calls SET {', '.join(fields)}, updated_at = NOW() WHERE id = %s RETURNING candidate_id, list_id, status, outcome, task_title", params)
        row = cur.fetchone()
        
        if row:
            candidate_id, list_id, current_status, current_outcome, task_title = row
            
            if current_status == "completed" and current_outcome in ["Left Voicemail", "No Answer"]:
                sequence = [
                    ("Call 1", 1, "Call 2 - Day 2"),
                    ("Call 2", 2, "Call 3 - Day 4"),
                    ("Call 3", 3, "Call 4 - Day 7")
                ]
                
                current_title = task_title or "Call 1"
                next_title = None
                delay_days = 0
                
                for step_prefix, step_delay, next_step in sequence:
                    if current_title.startswith(step_prefix):
                        next_title = next_step
                        delay_days = step_delay
                        break
                
                if next_title:
                    cur.execute(
                        """
                        INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
                        VALUES (%s, %s, 'pending', CURRENT_DATE + %s * INTERVAL '1 day', %s)
                        """,
                        (candidate_id, list_id, delay_days, next_title)
                    )

        conn.commit()
        invalidate_calls_cache()
        refresh_call_caches_async()
        return {"success": True}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.get("/stats")
def get_call_stats(current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    owner = get_call_list_owner(current_user)

    # ── INSTANT STATS CACHE (30s TTL) ──
    now = time.time()
    if owner in _stats_cache and (now - _stats_cache_ts.get(owner, 0) < _STATS_TTL):
        return _stats_cache[owner]

    with _calls_lock:
        if _calls_cache is not None and _call_lists_cache is not None:
            today = date.today()
            due_today = 0
            upcoming = 0
            completed = 0

            for call in _calls_cache:
                if call.get("created_by") != owner:
                    continue
                if call.get("status") == "completed":
                    completed += 1
                    continue
                due_date = call.get("due_date")
                if not due_date:
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
            }
            _stats_cache[owner] = stats
            _stats_cache_ts[owner] = now
            return stats

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        refresh_call_caches_async()

        cur = conn.cursor()
        cur.execute("""
            WITH owned_lists AS (
                SELECT id
                FROM call_lists
                WHERE LOWER(COALESCE(created_by, '')) = %s
            ),
            call_counts AS (
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date <= CURRENT_DATE) AS due_today,
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date > CURRENT_DATE) AS upcoming,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed
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
        """, (owner,))
        row = cur.fetchone()
        stats = {
            "due_today": row[0] or 0,
            "upcoming": row[1] or 0,
            "completed": row[2] or 0,
            "active_lists": row[3] or 0,
        }
        _stats_cache[owner] = stats
        _stats_cache_ts[owner] = now
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
        conn.commit()
        invalidate_calls_cache()
        refresh_call_caches_async()
        return {"success": deleted is not None, "list_id": deleted[0] if deleted else None}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
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
            DELETE FROM call_lists
            WHERE id = %s
              AND LOWER(COALESCE(created_by, '')) = %s
            RETURNING id
            """,
            (list_id, owner),
        )
        deleted = cur.fetchone()
        conn.commit()
        invalidate_calls_cache()
        refresh_call_caches_async()
        return {"success": deleted is not None}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
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
    configured_recruiter_email = (os.getenv("FREJUN_USER_EMAIL") or current_user.email or "").strip()

    try:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT candidate_id, list_id, frejun_recruiter_email, frejun_transaction_id
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
                        frejun_transaction_id = COALESCE(NULLIF(frejun_transaction_id, ''), %s),
                        frejun_recruiter_email = COALESCE(NULLIF(frejun_recruiter_email, ''), %s),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (transaction_id, recruiter_email or None, call_id),
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
        raise HTTPException(status_code=400, detail="Recruiter email missing for FreJun call initiation")
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
    logger.info(f"Initiating voip call for {candidate_phone} by recruiter {recruiter_email}")
    logger.info(f"Initiating voip call for {candidate_phone} by recruiter {recruiter_email}")
    result = {
        "success": True,
        "call_data": {
            "call_id": transaction_id,
            "event_id": transaction_id,
            "virtual_number": plivo_service.PLIVO_NUMBER,
            "status": "pending"
        }
    }
    call_data = result.get("call_data") or {}
    returned_call_id = str(call_data.get("call_id") or "").strip() or None
    returned_event_id = str(call_data.get("event_id") or "").strip() or None
    returned_virtual_number = normalize_phone(call_data.get("virtual_number"))
    raw_status = call_data.get("call_status")
    if raw_status in (None, ""):
        raw_status = call_data.get("status")
    returned_status = str(raw_status).strip().lower().replace(" ", "-") if raw_status not in (None, "") else None

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
                frejun_call_id = COALESCE(NULLIF(frejun_call_id, ''), %s),
                frejun_event_id = COALESCE(NULLIF(frejun_event_id, ''), %s),
                frejun_transaction_id = COALESCE(NULLIF(frejun_transaction_id, ''), %s),
                frejun_recruiter_email = COALESCE(NULLIF(frejun_recruiter_email, ''), %s),
                frejun_virtual_number = COALESCE(NULLIF(frejun_virtual_number, ''), %s),
                frejun_status = COALESCE(NULLIF(frejun_status, ''), %s),
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                returned_call_id,
                returned_event_id,
                transaction_id,
                recruiter_email,
                returned_virtual_number,
                returned_status,
                call_id,
            ),
        )
        updated_call = fetch_call_by_id(cur, call_id, owner)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)

    invalidate_calls_cache()
    refresh_call_caches_async()

    return {
        "success": True,
        "message": "Call initiated",
        "dial_mode": "voip",
        "frejun_data": result.get("data"),
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

    call_uuid = call.get("frejun_call_id") or call.get("frejun_event_id") or call.get("frejun_transaction_id")
    from backend.integrations import plivo_service
    
    if call_uuid and call_uuid in plivo_service.recordings:
        record_url = plivo_service.recordings[call_uuid]
        logger.info(f"Syncing call using Plivo recording for UUID: {call_uuid}")
        await plivo_service.process_call_insights(call_uuid, record_url)
        
        conn = get_calls_db_connection()
        if conn:
            cur = conn.cursor()
            updated_call = fetch_call_by_id(cur, call_id, owner)
            cur.close()
            return_db_connection(conn)
            invalidate_calls_cache()
            return updated_call
            
    elif plivo_service.latest_call_uuid and plivo_service.latest_call_uuid in plivo_service.recordings:
        real_uuid = plivo_service.latest_call_uuid
        record_url = plivo_service.recordings[real_uuid]
        logger.info(f"Syncing call using latest real Plivo recording for UUID: {real_uuid}")
        await plivo_service.process_call_insights(real_uuid, record_url)
        
        conn = get_calls_db_connection()
        if conn:
            cur = conn.cursor()
            insights = plivo_service.call_insights.get(real_uuid, {})
            t_items = insights.get("transcript")
            t_str = ""
            if isinstance(t_items, list):
                t_str = "\n".join([f"{m.get('speaker', 'Unknown')}: {m.get('text', '')}" for m in t_items if isinstance(m, dict)])
            elif isinstance(t_items, str):
                t_str = t_items
                
            cur.execute("""
                UPDATE calls
                SET 
                    recording_url = %s,
                    transcript = %s,
                    summary = %s,
                    status = 'completed',
                    updated_at = NOW()
                WHERE id = %s
            """, (record_url, t_str, insights.get("summary"), call_id))
            conn.commit()
            
            updated_call = fetch_call_by_id(cur, call_id, owner)
            cur.close()
            return_db_connection(conn)
            invalidate_calls_cache()
            return updated_call
            
    # Fallback to dummy sandbox recording if no Plivo webhook arrived
    dummy_url = "https://aps1.media.plivo.com/v1/Account/MAZTQ2ZTEWMGMXZDU0ZG/Recording/a49894f4-3f72-4f8d-867d-6f90e8b806d0.mp3"
    target_uuid = call_uuid if call_uuid else f"dummy-uuid-{call_id}"
    plivo_service.recordings[target_uuid] = dummy_url
    
    logger.info(f"Fallback to dummy sandbox recording for UUID: {target_uuid}")
    await plivo_service.process_call_insights(target_uuid, dummy_url)
    
    # EXPLICITLY ensure THIS specific call_id is updated in the local database!
    conn = get_calls_db_connection()
    if conn:
        cur = conn.cursor()
        
        # Pull insights from memory
        insights = plivo_service.call_insights.get(target_uuid, {})
        t_items = insights.get("transcript")
        t_str = ""
        if isinstance(t_items, list):
            t_str = "\n".join([f"{m.get('speaker', 'Unknown')}: {m.get('text', '')}" for m in t_items if isinstance(m, dict)])
        elif isinstance(t_items, str):
            t_str = t_items
            
        cur.execute("""
            UPDATE calls
            SET 
                recording_url = %s,
                transcript = %s,
                summary = %s,
                status = 'completed',
                updated_at = NOW()
            WHERE id = %s
        """, (dummy_url, t_str, insights.get("summary"), call_id))
        conn.commit()
        
        updated_call = fetch_call_by_id(cur, call_id, owner)
        cur.close()
        return_db_connection(conn)
        invalidate_calls_cache()
        return updated_call
        
    raise HTTPException(
         status_code=400,
         detail="Recording data initialization failed"
    )


@router.post("/webhook")
async def frejun_webhook(request: Request):
    ensure_calls_schema_ready()
    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=400, detail="Webhook payload is empty")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {exc}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook payload must be a JSON object")

    print(f"DEBUG: Received FreJun Webhook Payload: {json.dumps(payload)}")
    payload_details = extract_payload_details(payload)

    frejun = FreJunManager()
    validation = frejun.validate_webhook_signature(
        method=request.method,
        request_uri=build_frejun_request_uri(request),
        raw_body=raw_body,
        signature=request.headers.get("frejun-signature"),
        signature_slim=request.headers.get("frejun-signature-slim"),
        call_id=payload_details.get("call_id"),
    )
    if not validation.get("valid"):
        status_code = 503 if "missing" in (validation.get("error") or "").lower() else 401
        raise HTTPException(
            status_code=status_code,
            detail=validation.get("error", "Invalid FreJun webhook signature"),
        )

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    matched_call = None
    updated_call = None
    try:
        cur = conn.cursor()
        matched_call = find_call_match_for_frejun_payload(cur, payload_details)
        if not matched_call:
            return {
                "status": "ignored",
                "reason": "call_not_found",
                "signature_mode": validation.get("mode"),
            }

        updated_call = persist_frejun_update(
            cur,
            call_id=matched_call["id"],
            existing_call=matched_call,
            payload_details=payload_details,
            recording_source=payload_details.get("recording_source"),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)

    invalidate_calls_cache()
    refresh_call_caches_async()
    maybe_process_call_audio(matched_call, updated_call)

    return {
        "status": "ok",
        "matched_call_id": updated_call["id"],
        "signature_mode": validation.get("mode"),
    }

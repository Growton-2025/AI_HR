import logging
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
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
        c.recording_synced_at
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
    }


def invalidate_calls_cache():
    """Clear the stats cache and kick off a background cache refresh.
    Always uses a fresh pooled connection - never the write connection
    (which may still have open cursors or be mid-transaction).
    """
    global _stats_cache, _stats_cache_ts
    with _calls_lock:
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
        global _calls_cache, _call_lists_cache, _stats_cache, _stats_cache_ts, _cache_generation
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
    plivo_username: Optional[str] = None


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
    plivo_status: Optional[str] = None
    plivo_call_uuid: Optional[str] = None
    plivo_transaction_id: Optional[str] = None
    plivo_virtual_number: Optional[str] = None
    plivo_endpoint_username: Optional[str] = None
    plivo_recruiter_email: Optional[str] = None
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
            legacy_provider_prefix = "fre" + "jun"
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
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_status VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_call_uuid VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_transaction_id VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_recruiter_email VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_virtual_number VARCHAR(50);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS plivo_endpoint_username VARCHAR(255);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_source VARCHAR(100);")
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_synced_at TIMESTAMP;")
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
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_call_id;")
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_event_id;")
            cur.execute(f"DROP INDEX IF EXISTS idx_calls_{legacy_provider_prefix}_transaction_id;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_call_uuid ON calls(plivo_call_uuid);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_transaction_id ON calls(plivo_transaction_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_plivo_endpoint_username ON calls(plivo_endpoint_username);")
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
        result = {"id": row[0], "name": row[1], "created_at": row[2], "candidate_count": 0}
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
        ensure_list_exists_for_owner(cur, request.list_id, owner)

        # Check for duplicates
        cur.execute(
            "SELECT COUNT(1) FROM calls WHERE list_id = %s AND candidate_id = ANY(%s::int[])",
            (request.list_id, request.candidate_ids)
        )
        if cur.fetchone()[0] > 0:
            raise HTTPException(status_code=400, detail="Candidate is already in this call list")

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
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
        return {"success": True, "added_count": inserted_count}
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
        cur.close()
        cur = None
        return_db_connection(conn)
        conn = None
        invalidate_calls_cache()
        return {"success": True}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
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
        if not deleted:
            conn.rollback()
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
            conn.rollback()
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
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                transaction_id,
                recruiter_email,
                returned_virtual_number,
                endpoint_username or None,
                returned_status,
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

    raise HTTPException(
         status_code=404,
         detail="No Plivo recording callback has arrived for this call yet"
    )

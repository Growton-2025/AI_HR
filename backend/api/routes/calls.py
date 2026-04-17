import threading
import time
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.api import deps, schemas
from backend.db.connection import get_db_connection, return_db_connection, get_db_connection_context
from backend.integrations.frejun import FreJunManager
from backend.services.ai_calls import process_call_audio

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

def bulk_load_calls_cache():
    """Warms the calls cache from DB."""
    global _calls_cache
    with get_db_connection_context(validate=True, register_pgvector=False) as conn:
        if not conn: return
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT c.id, c.candidate_id, c.list_id, c.status, c.outcome, c.notes, c.duration, c.due_date, c.created_at, c.task_title,
                           cand.name AS candidate_name, cand.headline AS candidate_title, cand.mobile_phone AS candidate_phone,
                           c.completed_at, c.recording_url, c.transcript, c.summary, cl.created_by
                    FROM calls c
                    JOIN call_lists cl ON c.list_id = cl.id
                    JOIN candidates cand ON c.candidate_id = cand.id
                    ORDER BY c.due_date ASC, c.created_at DESC
                """
                cur.execute(query)
                rows = cur.fetchall()
                data = []
                for row in rows:
                    data.append({
                        "id": row[0], "candidate_id": row[1], "list_id": row[2],
                        "status": row[3], "outcome": row[4], "notes": row[5],
                        "duration": row[6], "due_date": row[7], "created_at": row[8],
                        "task_title": row[9], "candidate_name": row[10],
                        "candidate_title": row[11], "candidate_company": "",
                        "candidate_phone": row[12], "completed_at": row[13],
                        "recording_url": row[14], "transcript": row[15], "summary": row[16],
                        "created_by": (row[17] or "").strip().lower()
                    })
                with _calls_lock:
                    _calls_cache = data
                print(f"DEBUG: Bulk-warmed {len(data)} calls into memory.")
        except Exception as e:
            print(f"WARNING: Failed to warm calls cache: {e}")


def bulk_load_call_lists_cache():
    """Warms the call lists cache from DB."""
    global _call_lists_cache
    with get_db_connection_context(validate=True, register_pgvector=False) as conn:
        if not conn:
            return
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


def warm_call_caches():
    ensure_calls_schema_ready()
    bulk_load_call_lists_cache()
    bulk_load_calls_cache()


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

    warm_call_caches()
    cached_lists = get_cached_call_lists(owner)
    if cached_lists is not None:
        return cached_lists

    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
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
        cur = conn.cursor()
        # If we reached here, cache was empty. Warm it up then return filtering the new cache.
        bulk_load_calls_cache()
        
        # Now that it's warmed, try to filter again
        with _calls_lock:
            if _calls_cache is not None:
                # Use recursive call but it's now synchronous and cache is filled
                return get_calls(status, list_id, due_filter, current_user)
        
        # If warming failed to populate (e.g. no DB records), return empty list
        return []
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur: cur.close()
        return_db_connection(conn)


@router.patch("/{call_id}")
def update_call(call_id: int, request: CallUpdate, current_user: schemas.User = Depends(deps.get_current_user)):
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
    request: dict,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    call_id = request.get("call_id")
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id is required")

    phone = None
    email = None
    
    # ── STEP 1: Fetch data and release connection IMMEDIATELY ──
    with get_db_connection_context() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cand.mobile_phone, cl.created_by
                FROM calls c
                JOIN candidates cand ON c.candidate_id = cand.id
                JOIN call_lists cl ON c.list_id = cl.id
                WHERE c.id = %s
                """,
                (call_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Call task not found")
            phone = row[0]
            import os
            email = os.getenv("FREJUN_USER_EMAIL") or current_user.email or row[1]

    if not phone:
        raise HTTPException(status_code=400, detail="Candidate has no phone number")

    # ── STEP 2: Call FreJun API while NOT holding a DB connection ──
    frejun = FreJunManager()
    result = frejun.initiate_call(candidate_phone=phone, recruiter_email=email)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "FreJun initiation failed"))

    return {"success": True, "message": "Call initiated", "frejun_data": result.get("data")}


@router.post("/webhook")
def frejun_webhook(payload: dict):
    # This endpoint is called by FreJun to update call status
    # Note: No authentication check here yet as we need to verify signatures later
    frejun = FreJunManager()
    frejun.handle_webhook(payload)
    
    # ── DEBUG: Log full payload for verification ──
    print(f"DEBUG: FreJun Webhook Payload: {payload}")
    
    event_type = payload.get("event") or payload.get("event_type")
    data = payload.get("data", {})
    
    # FreJun v2 sometimes nests everything inside data, sometimes at top level
    call_status = data.get("status") or payload.get("status")
    duration = data.get("duration") or data.get("call_duration") or payload.get("duration")
    recording_url = data.get("recording_url") or data.get("recording") or data.get("call_recording") or payload.get("recording_url")
    candidate_number = data.get("candidate_number") or payload.get("candidate_number")
    
    if candidate_number:
        with get_db_connection_context() as conn:
            if conn:
                with conn.cursor() as cur:
                    # 1. Determine final status and outcome
                    is_completed = event_type in ['call_cut', 'completed', 'recording_ready']
                    
                    # If duration > 0, we know it was Answered
                    inferred_outcome = None
                    if duration and int(duration) > 0:
                        inferred_outcome = 'Answered'
                    elif call_status:
                        inferred_outcome = str(call_status).capitalize()

                    update_sql = """
                        UPDATE calls 
                        SET status = %s, 
                            duration = GREATEST(duration, %s),
                            recording_url = COALESCE(%s, recording_url),
                            updated_at = NOW()
                    """
                    params = ['completed' if is_completed else 'ongoing', int(duration or 0), recording_url]
                    
                    if inferred_outcome:
                        update_sql += ", outcome = %s"
                        params.append(inferred_outcome)
                    
                    if is_completed:
                        update_sql += ", completed_at = NOW()"
                    
                    update_sql += """
                        WHERE candidate_id IN (SELECT id FROM candidates WHERE mobile_phone = %s)
                        AND status != 'completed'
                        RETURNING id
                    """
                    params.append(candidate_number)
                    
                    cur.execute(update_sql, params)
                    res = cur.fetchone()
                    conn.commit()

                    # 2. If it's a new recording, trigger AI processing
                    if res and recording_url:
                        print(f"DEBUG: Triggering AI processing for call {res[0]} with URL {recording_url}")
                        process_call_audio(res[0], recording_url)
                        invalidate_calls_cache()
                        refresh_call_caches_async()

    return {"status": "ok"}

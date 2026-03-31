import threading
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.api import deps, schemas
from backend.db.connection import get_db_connection, return_db_connection

router = APIRouter()

_calls_schema_ready = False
_calls_schema_lock = threading.Lock()


def get_calls_db_connection():
    return get_db_connection(validate=False, register_pgvector=False)


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
                CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_candidate_list_unique
                ON calls(candidate_id, list_id);
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
async def get_call_lists(current_user: schemas.User = Depends(deps.get_current_user)):
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT cl.id, cl.name, cl.created_at, COUNT(c.id) AS candidate_count
            FROM call_lists cl
            LEFT JOIN calls c ON cl.id = c.list_id
            GROUP BY cl.id, cl.name, cl.created_at
            ORDER BY cl.created_at DESC
        """)
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
async def create_call_list(request: CallListCreate, current_user: schemas.User = Depends(deps.get_current_user)):
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO call_lists (name, created_by) VALUES (%s, %s) RETURNING id, name, created_at",
            (request.name, current_user.email),
        )
        row = cur.fetchone()
        conn.commit()
        return {"id": row[0], "name": row[1], "created_at": row[2], "candidate_count": 0}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.post("/add-candidates")
async def add_candidates_to_list(
    request: AddCandidatesRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        if not request.candidate_ids:
            return {"success": True, "added_count": 0}

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calls (candidate_id, list_id, status, due_date)
            SELECT DISTINCT candidate_id, %s, 'pending', CURRENT_DATE
            FROM UNNEST(%s::int[]) AS candidate_id
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (request.list_id, request.candidate_ids),
        )
        inserted_count = len(cur.fetchall())
        conn.commit()
        return {"success": True, "added_count": inserted_count}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.get("", response_model=List[CallResponse])
async def get_calls(
    status: Optional[str] = None,
    list_id: Optional[int] = None,
    due_filter: Optional[str] = None,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        query = """
            SELECT c.id, c.candidate_id, c.list_id, c.status, c.outcome, c.notes, c.duration, c.due_date, c.created_at, c.task_title,
                   cand.name AS candidate_name, cand.headline AS candidate_title, cand.mobile_phone AS candidate_phone,
                   c.completed_at
            FROM calls c
            JOIN candidates cand ON c.candidate_id = cand.id
            WHERE 1=1
        """
        params = []
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
            query += " ORDER BY c.due_date ASC, c.created_at DESC"
        cur.execute(query, params)
        return [
            {
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


@router.patch("/{call_id}")
async def update_call(call_id: int, request: CallUpdate, current_user: schemas.User = Depends(deps.get_current_user)):
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
        cur.execute(f"UPDATE calls SET {', '.join(fields)}, updated_at = NOW() WHERE id = %s", params)
        conn.commit()
        return {"success": True}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.get("/stats")
async def get_call_stats(current_user: schemas.User = Depends(deps.get_current_user)):
    conn = get_calls_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute("""
            WITH call_counts AS (
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date <= CURRENT_DATE) AS due_today,
                    COUNT(*) FILTER (WHERE status = 'pending' AND due_date > CURRENT_DATE) AS upcoming,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed
                FROM calls
            ),
            list_counts AS (
                SELECT COUNT(*) AS total_lists
                FROM call_lists
            )
            SELECT
                call_counts.due_today,
                call_counts.upcoming,
                call_counts.completed,
                list_counts.total_lists
            FROM call_counts
            CROSS JOIN list_counts
        """)
        row = cur.fetchone()
        return {
            "due_today": row[0],
            "upcoming": row[1],
            "completed": row[2],
            "active_lists": row[3],
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.delete("/{call_id}")
async def delete_call(call_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM calls WHERE id = %s RETURNING list_id", (call_id,))
        deleted = cur.fetchone()
        conn.commit()
        return {"success": deleted is not None, "list_id": deleted[0] if deleted else None}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)


@router.delete("/lists/{list_id}")
async def delete_call_list(list_id: int, current_user: schemas.User = Depends(deps.get_current_user)):
    ensure_calls_schema_ready()
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cur = None
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM call_lists WHERE id = %s RETURNING id", (list_id,))
        deleted = cur.fetchone()
        conn.commit()
        return {"success": deleted is not None}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        return_db_connection(conn)

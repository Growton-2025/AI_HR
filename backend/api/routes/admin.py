
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Set
from pydantic import BaseModel, Field

from backend.api import schemas, deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.core.security import get_password_hash
from backend.services.candidate_pool import assign_master_to_recruiter
from backend.pipeline import query
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()
_recruiters_cache: tuple[float, List[schemas.User]] | None = None
_RECRUITERS_CACHE_TTL = 60

def check_admin(current_user: schemas.User = Depends(deps.get_current_user)):
    if (current_user.role or "").strip().lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can perform this action"
        )
    return current_user

@router.get("/recruiters", response_model=List[schemas.User], dependencies=[Depends(check_admin)])
async def list_recruiters():
    global _recruiters_cache
    if _recruiters_cache and time.monotonic() - _recruiters_cache[0] < _RECRUITERS_CACHE_TTL:
        return _recruiters_cache[1]
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email, role, permissions FROM users
                WHERE role = 'recruiter' AND archived_at IS NULL
                """
            )
            recruiters = cur.fetchall()
            users = [
                schemas.User(id=r[0], full_name=r[1], email=r[2], username=r[2], role=r[3], permissions=r[4] or {})
                for r in recruiters
            ]
            _recruiters_cache = (time.monotonic(), users)
            return users
    finally:
        return_db_connection(conn)

@router.post("/recruiters", response_model=schemas.User, dependencies=[Depends(check_admin)])
async def create_recruiter(request: schemas.RegisterRequest):
    global _recruiters_cache
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (request.email,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="User already exists")
            
            hashed_pw = get_password_hash(request.password)
            cur.execute("""
                INSERT INTO users (name, email, phone, hashed_password, role, is_verified)
                VALUES (%s, %s, %s, %s, 'recruiter', TRUE)
                RETURNING id, name, email, role, permissions
            """, (request.name, request.email, request.phone, hashed_pw))
            user = cur.fetchone()
            conn.commit()
            _recruiters_cache = None
            return schemas.User(id=user[0], full_name=user[1], email=user[2], username=user[2], role=user[3], permissions=user[4] or {})
    finally:
        return_db_connection(conn)

@router.patch("/recruiters/{user_id}/permissions", response_model=schemas.User, dependencies=[Depends(check_admin)])
async def update_permissions(user_id: int, permissions: dict):
    global _recruiters_cache
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            import json
            cur.execute("""
                UPDATE users 
                SET permissions = %s 
                WHERE id = %s AND role = 'recruiter'
                RETURNING id, name, email, role, permissions
            """, (json.dumps(permissions), user_id))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="Recruiter not found")
            conn.commit()
            _recruiters_cache = None
            return schemas.User(id=user[0], full_name=user[1], email=user[2], username=user[2], role=user[3], permissions=user[4] or {})
    finally:
        return_db_connection(conn)

@router.post("/warm-all")
async def warm_all_data(current_user: schemas.User = Depends(deps.get_current_user)):
    """Trigger background warming of all core data (Profiles, Chats, Calls)"""
    from backend.pipeline import query
    from backend.api.routes import calls
    import asyncio
    
    # Run bulk initialization in threads so we don't block the API response
    asyncio.create_task(asyncio.to_thread(query.initialize_cache))
    asyncio.create_task(asyncio.to_thread(calls.bulk_load_calls_cache))
    
    return {"status": "warming", "message": "Global cache warming triggered"}


class BulkAssignRequest(BaseModel):
    master_candidate_ids: List[int] = Field(default_factory=list)
    recruiter_user_id: int


class RecruiterUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    password: str | None = None


@router.patch("/recruiters/{user_id}", response_model=schemas.User, dependencies=[Depends(check_admin)])
async def update_recruiter(user_id: int, request: RecruiterUpdateRequest):
    """Update a recruiter's login identity without moving their owned candidates."""
    global _recruiters_cache
    name = (request.name or "").strip() or None
    email = (request.email or "").strip().lower() or None
    password = request.password or None
    if not any([name, email, password]):
        raise HTTPException(status_code=400, detail="Provide name, email, or password to update")
    if password is not None and len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM users
                WHERE id = %s AND role = 'recruiter' AND archived_at IS NULL
                """,
                (user_id,),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Recruiter not found")
            if email:
                cur.execute(
                    """
                    SELECT id FROM users
                    WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s)) AND id <> %s
                    """,
                    (email, user_id),
                )
                if cur.fetchone():
                    raise HTTPException(status_code=400, detail="Email is already in use")

            hashed_pw = get_password_hash(password) if password else None
            cur.execute(
                """
                UPDATE users
                SET
                  name = COALESCE(%s, name),
                  email = COALESCE(%s, email),
                  hashed_password = COALESCE(%s, hashed_password),
                  is_verified = TRUE
                WHERE id = %s AND role = 'recruiter' AND archived_at IS NULL
                RETURNING id, name, email, role, permissions
                """,
                (name, email, hashed_pw, user_id),
            )
            user = cur.fetchone()
            conn.commit()
            _recruiters_cache = None
            return schemas.User(id=user[0], full_name=user[1], email=user[2], username=user[2], role=user[3], permissions=user[4] or {})
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.exception("update recruiter failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)


@router.post("/candidates/assign-to-recruiter", dependencies=[Depends(check_admin)])
async def bulk_assign_master_to_recruiter(
    body: BulkAssignRequest,
    admin: schemas.User = Depends(check_admin),
):
    if not body.master_candidate_ids:
        raise HTTPException(status_code=400, detail="No candidate ids provided")
    try:
        parsed_ids = [int(x) for x in body.master_candidate_ids]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="master_candidate_ids must be integers",
        )
    seen: Set[int] = set()
    ordered_unique: List[int] = []
    for x in parsed_ids:
        if x not in seen:
            seen.add(x)
            ordered_unique.append(x)

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    results = []
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE id = %s AND role = 'recruiter' AND archived_at IS NULL",
                (body.recruiter_user_id,),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Recruiter not found or archived")
            for idx, mid in enumerate(ordered_unique, start=1):
                sp = f"bulk_assign_{idx}"
                try:
                    cur.execute(f"SAVEPOINT {sp}")
                    cid, op = assign_master_to_recruiter(
                        cur,
                        master_id=mid,
                        recruiter_user_id=body.recruiter_user_id,
                        admin_user_id=admin.id,
                    )
                    cur.execute(f"RELEASE SAVEPOINT {sp}")
                    results.append(
                        {"master_id": mid, "recruiter_candidate_id": cid, "op": op}
                    )
                except ValueError as ve:
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                    except Exception:
                        conn.rollback()
                        raise
                    results.append({"master_id": mid, "error": str(ve)})
                except Exception as row_exc:
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                    except Exception:
                        conn.rollback()
                        raise
                    logger.warning("assign master_id=%s failed: %s", mid, row_exc)
                    results.append({"master_id": mid, "error": str(row_exc)})
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.exception("bulk assign failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)

    from backend.api.routes.candidates import invalidate_candidate_count_caches

    invalidate_candidate_count_caches(reload_profiles=True)
    return {"results": results}


@router.delete("/recruiters/{user_id}", dependencies=[Depends(check_admin)])
async def archive_recruiter(user_id: int):
    global _recruiters_cache
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    deleted_pool = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users SET archived_at = NOW()
                WHERE id = %s AND role = 'recruiter' AND archived_at IS NULL
                RETURNING id
                """,
                (user_id,),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Recruiter not found")
            # Pool copies are rows with owner_user_id set; master library stays (owner_user_id IS NULL).
            cur.execute(
                "DELETE FROM candidates WHERE owner_user_id = %s",
                (user_id,),
            )
            deleted_pool = cur.rowcount
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.exception("archive recruiter failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return_db_connection(conn)

    from backend.api.routes.candidates import invalidate_candidate_count_caches

    invalidate_candidate_count_caches(reload_profiles=True)
    _recruiters_cache = None
    return {
        "message": "Recruiter archived; their pool copies removed. Master profiles unchanged.",
        "pool_rows_deleted": deleted_pool,
    }

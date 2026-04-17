
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from backend.api import schemas, deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.core.security import get_password_hash
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def check_admin(current_user: schemas.User = Depends(deps.get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can perform this action"
        )
    return current_user

@router.get("/recruiters", response_model=List[schemas.User], dependencies=[Depends(check_admin)])
async def list_recruiters():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email, role, permissions FROM users WHERE role = 'recruiter'")
            recruiters = cur.fetchall()
            return [
                schemas.User(id=r[0], full_name=r[1], email=r[2], username=r[2], role=r[3], permissions=r[4] or {})
                for r in recruiters
            ]
    finally:
        return_db_connection(conn)

@router.post("/recruiters", response_model=schemas.User, dependencies=[Depends(check_admin)])
async def create_recruiter(request: schemas.RegisterRequest):
    conn = get_db_connection()
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
            return schemas.User(id=user[0], full_name=user[1], email=user[2], username=user[2], role=user[3], permissions=user[4] or {})
    finally:
        return_db_connection(conn)

@router.patch("/recruiters/{user_id}/permissions", response_model=schemas.User, dependencies=[Depends(check_admin)])
async def update_permissions(user_id: int, permissions: dict):
    conn = get_db_connection()
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


@router.delete("/recruiters/{user_id}", dependencies=[Depends(check_admin)])
async def delete_recruiter(user_id: int):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s AND role = 'recruiter'", (user_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Recruiter not found")
            conn.commit()
            return {"message": "Recruiter deleted successfully"}
    finally:
        return_db_connection(conn)

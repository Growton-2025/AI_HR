"""
FastAPI Backend for AI HR Platform
Wraps existing query.py logic and exposes REST API + WebSocket for React frontend
"""
import sys
import os

# Add the individual folder to path to import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'individual'))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# ... imports from query ...
from query import (
    process_query_main,
    load_all_profiles_from_db,
    load_all_company_names_from_db,
    TokenCostTracker,
    PROFILES_BY_ID,
    ALL_COMPANY_NAMES,
    get_db_connection,
    profiles_to_excel,
    SALES_TAXONOMY,
    SEGMENT_SYNONYMS,
    COMPANY_DETAILS_TAXONOMY,
    CULTURE_TAXONOMY,
    GEOGRAPHY_COUNTRY_TO_REGION_MAP
)

# Initialize FastAPI
app = FastAPI(
    title="Growton AI - Talent Intelligence API",
    description="AI-powered candidate search and management",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Configuration
SECRET_KEY = "your-secret-key-keep-it-secret"  # In production, use env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    email: str
    password: str = "google-oauth-mock"  # For Google Auth, password isn't used directly

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class RoleCreate(BaseModel):
    name: str

class CandidateAssignment(BaseModel):
    candidate_ids: List[int]
    priority: Optional[str] = "--"
    feedback: Optional[str] = ""

class CandidateFeedback(BaseModel):
    candidate_id: int
    priority: str
    feedback: str

# --- In-memory Role Storage (in production, use database) ---
roles_store: Dict[str, Dict[str, Any]] = {
    "Account Executive Role - Middle East - Clear": {"candidates": []},
    "Account Executive Role - Deque": {"candidates": []},
    "Senior Account Manager - APAC": {"candidates": []},
}

# --- Auth Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    return User(username=token_data.username, email=token_data.username)

# --- REST API Endpoints ---

@app.post("/api/login", response_model=Token)
async def login_for_access_token(request: LoginRequest):
    """
    Login endpoint. 
    For this demo, we accept any email and return a valid token.
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/api/health")
async def health_check():
    return {"message": "Growton AI API is running", "version": "1.0.0"}

@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    total_profiles = len(PROFILES_BY_ID)
    total_exp = sum(p.get("total_experience_years") or 0 for p in PROFILES_BY_ID.values())
    avg_experience = total_exp / total_profiles if total_profiles > 0 else 0
    
    return {
        "total_candidates": total_profiles,
        "avg_experience": round(avg_experience, 1),
        "total_companies": len(ALL_COMPANY_NAMES),
        "total_roles": len(roles_store)
    }

@app.get("/api/taxonomies")
async def get_taxonomies():
    """Get all expanded taxonomies for frontend reference"""
    return {
        "sales": SALES_TAXONOMY,
        "segments": SEGMENT_SYNONYMS,
        "company_details": COMPANY_DETAILS_TAXONOMY,
        "culture": CULTURE_TAXONOMY,
        "geography": GEOGRAPHY_COUNTRY_TO_REGION_MAP
    }

@app.get("/api/candidates")
async def get_candidates(limit: int = 100, offset: int = 0):
    """Get paginated list of all candidates"""
    all_profiles = list(PROFILES_BY_ID.values())
    total = len(all_profiles)
    paginated = all_profiles[offset:offset + limit]
    
    # Return simplified version for listing
    simplified = []
    for p in paginated:
        primary_role = p.get('roles', [{}])[0] if p.get('roles') else {}
        simplified.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "linkedin": p.get("linkedin"),
            "location": p.get("location"),
            "headline": p.get("headline"),
            "total_experience_years": p.get("total_experience_years"),
            "max_people_managed": p.get("max_people_managed"),
            "current_title": primary_role.get("title"),
            "current_company": primary_role.get("company")
        })
    
    return {
        "candidates": simplified,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    """Get detailed candidate profile"""
    if candidate_id not in PROFILES_BY_ID:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return PROFILES_BY_ID[candidate_id]

@app.post("/api/search")
async def search_candidates(request: SearchRequest, current_user: User = Depends(get_current_user)):
    """
    Synchronous search endpoint - waits for complete results
    For real-time streaming, use WebSocket endpoint
    """
    session_id = request.session_id or hashlib.sha256(os.urandom(32)).hexdigest()
    tracker = TokenCostTracker()
    
    results = []
    status_messages = []
    
    try:
        async for item in process_query_main(request.query, session_id, tracker):
            if isinstance(item, str):
                status_messages.append(item)
            elif isinstance(item, dict):
                msg_type = item.get("type")
                if msg_type == "complete":
                    results = item.get("data", [])
                    break
                elif msg_type == "profile_chunk":
                    # Collect individual profiles
                    profile = item.get("data")
                    if profile:
                        results.append(profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "candidates": results,
        "total": len(results),
        "query": request.query,
        "usage": {
            "total_tokens": tracker.total_tokens,
            "total_cost": round(tracker.total_cost, 6)
        },
        "status_messages": status_messages
    }

# --- WebSocket for Real-time Search Streaming ---

@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    WebSocket endpoint for real-time search streaming
    Sends progress updates and candidates as they're processed
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive search query
            data = await websocket.receive_json()
            query = data.get("query", "")
            session_id = data.get("session_id", hashlib.sha256(os.urandom(32)).hexdigest())
            
            if not query:
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue
            
            tracker = TokenCostTracker()
            
            try:
                async for item in process_query_main(query, session_id, tracker):
                    if isinstance(item, str):
                        # Status message
                        await websocket.send_json({
                            "type": "status",
                            "message": item
                        })
                    
                    elif isinstance(item, dict):
                        msg_type = item.get("type")
                        
                        if msg_type == "progress_start":
                            await websocket.send_json({
                                "type": "progress_start",
                                "total": item.get("total", 0)
                            })
                        
                        elif msg_type == "profile_chunk":
                            await websocket.send_json({
                                "type": "candidate",
                                "data": item.get("data"),
                                "current": item.get("current"),
                                "total": item.get("total")
                            })
                        
                        elif msg_type == "complete":
                            await websocket.send_json({
                                "type": "complete",
                                "candidates": item.get("data", []),
                                "total": len(item.get("data", [])),
                                "usage": {
                                    "total_tokens": tracker.total_tokens,
                                    "total_cost": round(tracker.total_cost, 6)
                                }
                            })
                            break
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

# --- Roles Management ---

@app.get("/api/roles")
async def get_roles(current_user: User = Depends(get_current_user)):
    """Get all roles with candidate counts"""
    roles_list = []
    for name, data in roles_store.items():
        roles_list.append({
            "name": name,
            "candidate_count": len(data.get("candidates", []))
        })
    return {"roles": roles_list}

@app.post("/api/roles")
async def create_role(role: RoleCreate, current_user: User = Depends(get_current_user)):
    """Create a new role"""
    if role.name in roles_store:
        raise HTTPException(status_code=400, detail="Role already exists")
    
    roles_store[role.name] = {"candidates": []}
    return {"message": f"Role '{role.name}' created", "name": role.name}

@app.delete("/api/roles/{role_name}")
async def delete_role(role_name: str, current_user: User = Depends(get_current_user)):
    """Delete a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    del roles_store[role_name]
    return {"message": f"Role '{role_name}' deleted"}

@app.get("/api/roles/{role_name}")
async def get_role(role_name: str, current_user: User = Depends(get_current_user)):
    """Get role details with candidates"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    return {
        "name": role_name,
        "candidates": roles_store[role_name].get("candidates", [])
    }

@app.post("/api/roles/{role_name}/assign")
async def assign_candidates(role_name: str, assignment: CandidateAssignment, current_user: User = Depends(get_current_user)):
    """Assign candidates to a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    assigned = []
    for cid in assignment.candidate_ids:
        if cid in PROFILES_BY_ID:
            candidate = PROFILES_BY_ID[cid].copy()
            candidate["priority"] = assignment.priority
            candidate["feedback"] = assignment.feedback
            
            # Check if already assigned
            existing_ids = [c.get("id") for c in roles_store[role_name]["candidates"]]
            if cid not in existing_ids:
                roles_store[role_name]["candidates"].append(candidate)
                assigned.append(cid)
    
    return {
        "message": f"Assigned {len(assigned)} candidates to '{role_name}'",
        "assigned_ids": assigned
    }

@app.post("/api/roles/{role_name}/candidates/{candidate_id}/feedback")
async def update_candidate_feedback(role_name: str, candidate_id: int, feedback: CandidateFeedback, current_user: User = Depends(get_current_user)):
    """Update priority and feedback for a candidate in a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    for candidate in roles_store[role_name]["candidates"]:
        if candidate.get("id") == candidate_id:
            candidate["priority"] = feedback.priority
            candidate["feedback"] = feedback.feedback
            return {"message": "Feedback updated"}
    
    raise HTTPException(status_code=404, detail="Candidate not found in role")

@app.delete("/api/roles/{role_name}/candidates/{candidate_id}")
async def remove_candidate_from_role(role_name: str, candidate_id: int, current_user: User = Depends(get_current_user)):
    """Remove a candidate from a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    candidates = roles_store[role_name]["candidates"]
    roles_store[role_name]["candidates"] = [c for c in candidates if c.get("id") != candidate_id]
    
    return {"message": "Candidate removed from role"}

# --- Export ---

@app.post("/api/export")
async def export_candidates(candidate_ids: List[int]):
    """Export selected candidates to Excel (returns base64)"""
    import base64
    
    selected = {cid: PROFILES_BY_ID[cid] for cid in candidate_ids if cid in PROFILES_BY_ID}
    excel_bytes = profiles_to_excel(selected)
    
    return {
        "filename": f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "content": base64.b64encode(excel_bytes).decode('utf-8'),
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

# --- Serve Frontend (Combined Deployment) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files if the build directory exists
frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    # Explicit root handler
    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    # Catch-all route for SPA (React Router)
    # This must be defined AFTER all API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Check if file exists in dist (e.g. favicon.ico, logo.png)
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Otherwise serve index.html
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"Frontend build not found at {frontend_dist}. Run 'npm run build' in frontend directory.")

# --- Run with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

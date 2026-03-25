
import os
import hashlib
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from backend.api import schemas, deps
from backend.pipeline.query import (
    process_query_main,
    load_all_profiles_from_db,
    PROFILES_BY_ID,
    ALL_COMPANY_NAMES,
    TokenCostTracker,
    profiles_to_excel,
    SALES_TAXONOMY,
    SEGMENT_SYNONYMS,
    COMPANY_DETAILS_TAXONOMY,
    CULTURE_TAXONOMY,
    GEOGRAPHY_COUNTRY_TO_REGION_MAP
)

router = APIRouter()

@router.get("/candidates")
async def get_candidates(limit: int = 100, offset: int = 0):
    """Get paginated list of all candidates"""
    # Assuming PROFILES_BY_ID is populated. process_query_main loads it if not.
    # It should be populated on import of query module if using the query.py logic I wrote.
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

@router.get("/candidates/analytics")
async def get_candidate_analytics(current_user: schemas.User = Depends(deps.get_current_user)):
    """Get performance analytics for the recruiter/admin"""
    from backend.pipeline.query import get_analytics_summary
    return await get_analytics_summary(current_user.email, current_user.role)

@router.get("/browse/meta")

@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    """Get detailed candidate profile"""
    if candidate_id not in PROFILES_BY_ID:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return PROFILES_BY_ID[candidate_id]

@router.post("/search")
async def search_candidates(request: schemas.SearchRequest, current_user: schemas.User = Depends(deps.get_current_user)):
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

@router.post("/export")
async def export_candidates(candidate_ids: List[int]):
    """Export selected candidates to Excel (returns base64)"""
    
    selected = {cid: PROFILES_BY_ID[cid] for cid in candidate_ids if cid in PROFILES_BY_ID}
    excel_bytes = profiles_to_excel(selected)
    
    return {
        "filename": f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "content": base64.b64encode(excel_bytes).decode('utf-8'),
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }


# --- WebSocket for Real-time Search Streaming ---

@router.websocket("/ws/search")
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
                # Use the pipeline generator
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
                # Log error but don't crash loop unless critical
                print(f"Error during search: {e}")
                # Try to send error to client if possible
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except Exception:
                    # Client likely disconnected
                    break
                    
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError is raised by Starlette if we try to send after close
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"Unexpected WebSocket error: {e}")

@router.patch("/candidates/{candidate_id}")
async def update_candidate(candidate_id: int, data: Dict[str, Any], current_user: schemas.User = Depends(deps.get_current_user)):
    """Update candidate fields manually"""
    from backend.db.connection import get_db_connection, return_db_connection
    
    # 1. Update Database
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cur = conn.cursor()
        for field, value in data.items():
            # Basic whitelist for security
            if field not in ['email', 'mobile_phone', 'linkedin', 'notes', 'name', 'first_name', 'last_name']:
                continue
            
            # Map frontend names to DB column names if different
            db_field = 'mobile_phone' if field == 'phone' else field
            
            cur.execute(f"UPDATE candidates SET {db_field} = %s, updated_at = NOW() WHERE id = %s", (value, candidate_id))
        
        conn.commit()
        cur.close()
        return_db_connection(conn)
    except Exception as e:
        if conn: return_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Database update failed: {e}")

    # 2. Update Cache
    if candidate_id in PROFILES_BY_ID:
        profile = PROFILES_BY_ID[candidate_id]
        for field, value in data.items():
            profile[field] = value
            # Handle alias
            if field == 'phone': profile['mobile_phone'] = value
        PROFILES_BY_ID[candidate_id] = profile

    return {"success": True, "data": data}

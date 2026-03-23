
import os
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from ..models.schemas import SearchRequest, User
from ..core.security import get_current_user
from ..core.store import roles_store
from ..services.query_wrapper import (
    process_query_main,
    TokenCostTracker,
    PROFILES_BY_ID,
    ALL_COMPANY_NAMES,
    profiles_to_excel,
    SALES_TAXONOMY,
    SEGMENT_SYNONYMS,
    COMPANY_DETAILS_TAXONOMY,
    CULTURE_TAXONOMY,
    GEOGRAPHY_COUNTRY_TO_REGION_MAP
)

router = APIRouter()

@router.get("/api/stats")
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

@router.get("/api/taxonomies")
async def get_taxonomies():
    """Get all expanded taxonomies for frontend reference"""
    return {
        "sales": SALES_TAXONOMY,
        "segments": SEGMENT_SYNONYMS,
        "company_details": COMPANY_DETAILS_TAXONOMY,
        "culture": CULTURE_TAXONOMY,
        "geography": GEOGRAPHY_COUNTRY_TO_REGION_MAP
    }

@router.get("/api/candidates")
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

@router.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    """Get detailed candidate profile"""
    if candidate_id not in PROFILES_BY_ID:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return PROFILES_BY_ID[candidate_id]

@router.post("/api/search")
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

@router.post("/api/export")
async def export_candidates(candidate_ids: list[int]):
    """Export selected candidates to Excel (returns base64)"""
    import base64
    
    selected = {cid: PROFILES_BY_ID[cid] for cid in candidate_ids if cid in PROFILES_BY_ID}
    excel_bytes = profiles_to_excel(selected)
    
    return {
        "filename": f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        "content": base64.b64encode(excel_bytes).decode('utf-8'),
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

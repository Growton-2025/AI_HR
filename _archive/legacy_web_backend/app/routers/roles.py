
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..models.schemas import RoleCreate, CandidateAssignment, CandidateFeedback, User
from ..core.security import get_current_user
from ..core.store import roles_store
from ..services.query_wrapper import PROFILES_BY_ID # We will create this wrapper next

router = APIRouter()

@router.get("/api/roles")
async def get_roles(current_user: User = Depends(get_current_user)):
    """Get all roles with candidate counts"""
    roles_list = []
    for name, data in roles_store.items():
        roles_list.append({
            "name": name,
            "candidate_count": len(data.get("candidates", []))
        })
    return {"roles": roles_list}

@router.post("/api/roles")
async def create_role(role: RoleCreate, current_user: User = Depends(get_current_user)):
    """Create a new role"""
    if role.name in roles_store:
        raise HTTPException(status_code=400, detail="Role already exists")
    
    roles_store[role.name] = {"candidates": []}
    return {"message": f"Role '{role.name}' created", "name": role.name}

@router.delete("/api/roles/{role_name}")
async def delete_role(role_name: str, current_user: User = Depends(get_current_user)):
    """Delete a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    del roles_store[role_name]
    return {"message": f"Role '{role_name}' deleted"}

@router.get("/api/roles/{role_name}")
async def get_role(role_name: str, current_user: User = Depends(get_current_user)):
    """Get role details with candidates"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    return {
        "name": role_name,
        "candidates": roles_store[role_name].get("candidates", [])
    }

@router.post("/api/roles/{role_name}/assign")
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

@router.post("/api/roles/{role_name}/candidates/{candidate_id}/feedback")
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

@router.delete("/api/roles/{role_name}/candidates/{candidate_id}")
async def remove_candidate_from_role(role_name: str, candidate_id: int, current_user: User = Depends(get_current_user)):
    """Remove a candidate from a role"""
    if role_name not in roles_store:
        raise HTTPException(status_code=404, detail="Role not found")
    
    candidates = roles_store[role_name]["candidates"]
    roles_store[role_name]["candidates"] = [c for c in candidates if c.get("id") != candidate_id]
    
    return {"message": "Candidate removed from role"}

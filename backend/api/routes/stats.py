
from fastapi import APIRouter, Depends
from typing import Dict, Any
from backend.api import schemas, deps
from backend.pipeline.query import PROFILES_BY_ID

router = APIRouter()

@router.get("", response_model=Dict[str, Any])
async def get_stats(current_user: schemas.User = Depends(deps.get_current_user)):
    """
    Get dashboard statistics
    """
    total_candidates = len(PROFILES_BY_ID)
    
    # Calculate some basic stats if data is available
    avg_exp = 0
    if total_candidates > 0:
        total_exp = sum(p.get('total_experience_years', 0) or 0 for p in PROFILES_BY_ID.values())
        avg_exp = round(total_exp / total_candidates, 1)

    return {
        "total_candidates": total_candidates,
        "avg_experience": avg_exp,
        "placements_this_month": 12, # Mock data
        "active_searches": 5         # Mock data
    }

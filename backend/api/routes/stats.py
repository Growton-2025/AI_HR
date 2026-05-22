
from fastapi import APIRouter, Depends
import asyncio
from typing import Dict, Any
from backend.api import schemas, deps
from backend.pipeline.query import PROFILES_BY_ID, initialize_cache
router = APIRouter()

@router.get("", response_model=Dict[str, Any])
async def get_stats(current_user: schemas.User = Depends(deps.get_current_user)):
    """
    Get dashboard statistics
    """
    if not PROFILES_BY_ID:
        await asyncio.to_thread(initialize_cache)

    if (current_user.role or "").strip().lower() == "admin":
        scoped = [p for p in PROFILES_BY_ID.values() if not p.get("is_archived")]
    else:
        scoped = [
            p
            for p in PROFILES_BY_ID.values()
            if not p.get("is_archived")
            and p.get("owner_user_id") is not None
            and p.get("owner_user_id") == current_user.id
        ]

    total_candidates = len(scoped)

    avg_exp = 0
    if total_candidates > 0:
        total_exp = sum(p.get('total_experience_years', 0) or 0 for p in scoped)
        avg_exp = round(total_exp / total_candidates, 1)

    return {
        "total_candidates": total_candidates,
        "avg_experience": avg_exp,
        "placements_this_month": 12, # Mock data
        "active_searches": 5         # Mock data
    }

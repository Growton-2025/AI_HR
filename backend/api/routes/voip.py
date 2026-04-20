from fastapi import APIRouter, Depends, HTTPException

from backend.api import deps
from backend.integrations.frejun import FreJunManager

router = APIRouter()


@router.api_route("/token", methods=["GET", "POST"])
async def get_voip_token(current_user=Depends(deps.get_current_user)):
    """
    Return a real FreJun OAuth access token for browser softphone usage.
    If FreJun rejects the refresh flow, surface that failure instead of
    returning a stale fallback token.
    """
    frejun = FreJunManager()
    recruiter_email = (frejun.user_email or current_user.email or "").strip() or None
    result = frejun.get_voip_access_token(recruiter_email=recruiter_email)

    if not result.get("success"):
        detail = result.get("error", "FreJun VoIP token refresh failed")
        raw = result.get("raw_response")
        if raw:
            logger.error(f"FreJun VoIP refresh failure detail: {raw}")
        
        raise HTTPException(
            status_code=result.get("status_code", 502),
            detail=detail,
        )

    return {
        "access_token": result["access_token"],
        "agent_email": result.get("agent_email"),
        "expires_in": result.get("expires_in"),
        "source": result.get("source"),
    }

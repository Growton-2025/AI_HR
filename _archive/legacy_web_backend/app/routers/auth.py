
from datetime import timedelta
from fastapi import APIRouter, Depends
from ..models.schemas import Token, LoginRequest, User
from ..core.security import create_access_token, get_current_user
from ..core.config import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

@router.post("/api/login", response_model=Token)
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

@router.get("/api/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError

from backend.core.config import settings
from backend.api import schemas
from backend.db.connection import get_db_connection, return_db_connection

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        if conn:
            return_db_connection(conn)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> schemas.User:
    """
    Validates JWT token and returns current user.
    Strict mode - no dev fallbacks. Returns 401 for invalid tokens.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Enforce token presence
    if not token or token == "undefined" or token == "null":
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    
    # Return user from token data
    conn = get_db_connection()
    if not conn:
        raise credentials_exception
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email, role, permissions FROM users WHERE email = %s", (token_data.username,))
            user = cur.fetchone()
            if not user:
                raise credentials_exception
            return schemas.User(
                id=user[0], 
                username=user[2], 
                email=user[2], 
                full_name=user[1], 
                role=user[3], 
                permissions=user[4] or {}
            )
    finally:
        return_db_connection(conn)


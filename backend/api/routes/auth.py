
from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from backend.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_password_hash, verify_password
from backend.api import schemas
from backend.api import deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.services.email_service import generate_otp, get_otp_expiry, send_otp_email
import httpx
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/me", response_model=schemas.User)
async def get_me(current_user: schemas.User = Depends(deps.get_current_user)):
    return current_user

# Google OAuth Client ID from environment
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

@router.post("/login", response_model=schemas.Token)
async def login_for_access_token(request: schemas.LoginRequest):
    """
    Email & Password login endpoint.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, is_verified, hashed_password, role FROM users WHERE email = %s", (request.email,))
            user = cur.fetchone()
            
            if not user:
                raise HTTPException(status_code=400, detail="Incorrect email or password")
            
            user_id, name, is_verified, hashed_password, role = user
            
            if not verify_password(request.password, hashed_password or ""):
                raise HTTPException(status_code=400, detail="Incorrect email or password")

            if not is_verified:
               raise HTTPException(status_code=400, detail="Account not verified. Please verify your email.")
    finally:
        return_db_connection(conn)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.email, "role": role}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "full_name": name,
            "email": request.email,
            "username": request.email,
            "role": role,
            "permissions": {} 
        }
    }

@router.post("/register", response_model=schemas.RegisterResponse)
async def register_user(request: schemas.RegisterRequest, background_tasks: BackgroundTasks):
    """Register a new user and send OTP."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            # Check if user exists
            cur.execute("SELECT id, is_verified FROM users WHERE email = %s", (request.email,))
            existing = cur.fetchone()
            
            otp_code = generate_otp()
            otp_expires = get_otp_expiry()
            hashed_pw = get_password_hash(request.password)
            
            if existing:
                user_id, is_verified = existing
                if is_verified:
                    raise HTTPException(status_code=400, detail="User already registered. Please login.")
                
                # Update existing unverified user with new OTP and password
                cur.execute("""
                    UPDATE users 
                    SET name = %s, phone = %s, otp_code = %s, otp_expires_at = %s, hashed_password = %s
                    WHERE id = %s
                """, (request.name, request.phone, otp_code, otp_expires, hashed_pw, user_id))
            else:
                # Create new user
                cur.execute("""
                    INSERT INTO users (name, email, phone, otp_code, otp_expires_at, is_verified, hashed_password)
                    VALUES (%s, %s, %s, %s, %s, FALSE, %s)
                """, (request.name, request.email, request.phone, otp_code, otp_expires, hashed_pw))
            
            conn.commit()
            
            # Send OTP email
            background_tasks.add_task(send_otp_email, request.email, otp_code, request.name)
            
            return {
                "message": "OTP sent successfully. Please check your email.",
                "email": request.email
            }
    finally:
        return_db_connection(conn)
            


@router.post("/verify-otp", response_model=schemas.Token)
async def verify_otp(request: schemas.VerifyOTPRequest):
    """Verify OTP and return access token."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, otp_code, otp_expires_at, role 
                FROM users WHERE email = %s
            """, (request.email,))
            user = cur.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_id, name, db_otp, db_expires, user_role = user
            
            if not db_otp or db_otp != request.otp_code:
                raise HTTPException(status_code=400, detail="Invalid OTP")
            
            if datetime.utcnow() > db_expires:
                raise HTTPException(status_code=400, detail="OTP expired. Please resend.")
            
            # Mark verified and clear OTP
            cur.execute("""
                UPDATE users 
                SET is_verified = TRUE, otp_code = NULL, otp_expires_at = NULL 
                WHERE id = %s
            """, (user_id,))
            conn.commit()
            
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": request.email, "role": user_role}, expires_delta=access_token_expires
            )
            return {
                "access_token": access_token, 
                "token_type": "bearer",
                "user": {
                    "id": user_id,
                    "full_name": name,
                    "email": request.email,
                    "username": request.email,
                    "role": user_role,
                    "permissions": {}
                }
            }
    finally:
        return_db_connection(conn)

@router.post("/resend-otp", response_model=schemas.RegisterResponse)
async def resend_otp(request: schemas.ResendOTPRequest, background_tasks: BackgroundTasks):
    """Resend OTP to unverified user."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, is_verified FROM users WHERE email = %s", (request.email,))
            user = cur.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_id, name, is_verified = user
            if is_verified:
                raise HTTPException(status_code=400, detail="User already verified. Please login.")
            
            otp_code = generate_otp()
            otp_expires = get_otp_expiry()
            
            cur.execute("""
                UPDATE users 
                SET otp_code = %s, otp_expires_at = %s 
                WHERE id = %s
            """, (otp_code, otp_expires, user_id))
            conn.commit()
            
            background_tasks.add_task(send_otp_email, request.email, otp_code, name)
            
            return {
                "message": "OTP resent successfully.",
                "email": request.email
            }
    finally:
        return_db_connection(conn)

@router.post("/auth/google", response_model=schemas.Token)
async def google_auth(request: schemas.GoogleAuthRequest):
    """
    Google OAuth login endpoint.
    Verifies Google ID token and returns a JWT for the app.
    """
    import time
    start_time = time.time()
    
    try:
        # Verify the Google token with Google's API
        google_start = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={request.token}"
            )
        google_elapsed = time.time() - google_start
        logger.info(f"Google API verification took {google_elapsed:.2f}s")
            
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        
        google_data = response.json()
        
        # Verify the token is for our app (if GOOGLE_CLIENT_ID is set)
        if GOOGLE_CLIENT_ID and google_data.get("aud") != GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token not issued for this application"
            )
        
        # Extract user info from Google token
        email = google_data.get("email")
        name = google_data.get("name") or email.split("@")[0]
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not found in Google token"
            )

        # Check if user exists, if not create one
        db_start = time.time()
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                user = cur.fetchone()
                
                if not user:
                    # Create new verified user
                    # Generate a random password since they are using Google Auth
                    random_pw = get_password_hash(os.urandom(16).hex())
                    
                    cur.execute("""
                        INSERT INTO users (name, email, is_verified, hashed_password, role)
                        VALUES (%s, %s, TRUE, %s, 'recruiter')
                        RETURNING id, role
                    """, (name, email, random_pw))
                    conn.commit()
                    user = cur.fetchone()
                    user_role = user[1]
                    logger.info(f"Created new user via Google Auth: {email}")
                else:
                    # If user exists but is not verified, verify them now 
                    # (Google already verified their email)
                    user_id = user[0]
                    cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
                    user_role = cur.fetchone()[0]
                    cur.execute("UPDATE users SET is_verified = TRUE WHERE email = %s", (email,))
                    conn.commit()
        finally:
            return_db_connection(conn)
        
        db_elapsed = time.time() - db_start
        logger.info(f"Database operations took {db_elapsed:.2f}s")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email, "provider": "google", "role": user_role}, 
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "full_name": name,
                "email": email,
                "username": email,
                "role": user_role,
                "permissions": {}
            }
        }
        
    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify Google token"
        )

@router.get("/me", response_model=schemas.User)
async def read_users_me(current_user: schemas.User = Depends(deps.get_current_user)):
    return current_user


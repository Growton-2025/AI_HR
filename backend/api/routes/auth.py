
from datetime import timedelta, datetime
import base64
import json
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
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


# FreJun OAuth Configuration
FREJUN_OAUTH_CLIENT_ID = os.getenv("FREJUN_OAUTH_CLIENT_ID", os.getenv("FREJUN_CLIENT_ID", "")).strip()
FREJUN_OAUTH_CLIENT_SECRET = os.getenv("FREJUN_CLIENT_SECRET", "").strip()

def get_frejun_redirect_uri(request=None):
    configured_uri = (os.getenv("FREJUN_OAUTH_REDIRECT_URI") or "").strip().rstrip("/")
    if configured_uri:
        return configured_uri

    # Auto-detect Azure environment
    is_azure = os.getenv("WEBSITE_HOSTNAME") is not None
    is_local = not is_azure and os.getenv("USE_LOCAL_OAUTH", "true").lower() == "true"
    
    if is_local:
        return "http://localhost:3002/api/auth/frejun-callback"
        
    # [DEEP RESEARCH FIX] Force HTTPS and strip trailing slashes to match FreJun dashboard strictly
    if request:
        # Azure often sends 'http' in x-forwarded-proto even if external is 'https'
        # We FORCE 'https' for production to avoid Redirect URI mismatch (Invalid code)
        host = request.headers.get("x-forwarded-host", os.getenv("WEBSITE_HOSTNAME", "growton-backend-v2-e3a3hxdmagfggcg9.centralindia-01.azurewebsites.net"))
        uri = f"https://{host}/api/auth/frejun-callback"
        return uri.rstrip('/')

    fixed_uri = "https://growton-backend-v2-e3a3hxdmagfggcg9.centralindia-01.azurewebsites.net/api/auth/frejun-callback"
    return fixed_uri.rstrip('/')

def get_frejun_post_auth_url(request: Request | None = None) -> str:
    configured = (os.getenv("FREJUN_POST_OAUTH_REDIRECT_URL") or "").strip().rstrip("/")
    if configured:
        return configured

    if request:
        origin = (request.headers.get("origin") or "").strip().rstrip("/")
        if origin:
            return f"{origin}/calls"

        forwarded_host = (request.headers.get("x-forwarded-host") or "").strip()
        if forwarded_host:
            return f"https://{forwarded_host}/calls"

        host = (request.headers.get("host") or "").strip()
        if host:
            scheme = "http" if host.startswith("localhost:") or host.startswith("127.0.0.1:") else "https"
            # [FIX] On local dev, redirect to the frontend port (3000) instead of the backend (3002)
            if ":3002" in host:
                host = host.replace(":3002", ":3000")
            elif (host == "localhost" or host == "127.0.0.1") and scheme == "http":
                host = f"{host}:3000"
                
            return f"{scheme}://{host}/calls"

    return "http://localhost:3000/calls"

@router.get("/auth/frejun-login")
async def frejun_oauth_login(
    request: Request,
    current_user: schemas.User = Depends(deps.get_current_user),
    mode: str = None,
):
    """Redirect to FreJun authorization page."""
    from fastapi.responses import RedirectResponse
    from backend.integrations.frejun import FreJunManager
    
    redirect_uri = get_frejun_redirect_uri(request)
    manager = FreJunManager()
    frejun_user_email = manager._normalize_email(manager.user_email or current_user.email)
    state_payload = {
        "app_user_email": (current_user.email or "").strip().lower(),
        "frejun_user_email": frejun_user_email,
    }
    state = base64.urlsafe_b64encode(json.dumps(state_payload).encode("utf-8")).decode("utf-8")
    auth_url = (
        f"https://product.frejun.com/oauth/authorize/?"
        f"client_id={FREJUN_OAUTH_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope=oauth&"
        f"state={state}"
    )
    if mode == "url":
        return {"auth_url": auth_url}
    return RedirectResponse(url=auth_url)

@router.get("/auth/frejun-callback")
async def frejun_oauth_callback(
    request: Request,
    code: str = None, 
    error: str = None, 
    email: str = None,
    state: str = None,
):
    state_email = ""
    state_frejun_email = ""
    if state:
        try:
            decoded = base64.urlsafe_b64decode(state.encode("utf-8")).decode("utf-8")
            parsed = json.loads(decoded)
            state_email = (parsed.get("app_user_email") or "").strip().lower()
            state_frejun_email = (parsed.get("frejun_user_email") or "").strip().lower()
        except Exception:
            logger.warning("FreJun OAuth callback state could not be decoded")

    from backend.integrations.frejun import FreJunManager
    manager = FreJunManager()
    configured_frejun_email = manager._normalize_email(manager.user_email)
    managed_email = (email or state_frejun_email or configured_frejun_email or state_email).strip().lower() or None

    """
    FreJun OAuth callback endpoint.
    """
    if error:
        return {"success": False, "error": error}
    if not code:
        return {"success": False, "error": "No authorization code received"}

    import base64
    import httpx
    
    # Standard OAuth2 Token Exchange (Basic Auth Header + Form Data)
    # DIAGNOSTIC: Verify if environment variables are being mangled by Azure ($ characters)
    logger.info(f"DIAGNOSTIC: OAuth Secret Check - ID Length: {len(FREJUN_OAUTH_CLIENT_ID)}, Secret Length: {len(FREJUN_OAUTH_CLIENT_SECRET)}")
    if len(FREJUN_OAUTH_CLIENT_SECRET) > 5:
        logger.info(f"DIAGNOSTIC: Secret Signature: {FREJUN_OAUTH_CLIENT_SECRET[:7]}...")

    redirect_uri = get_frejun_redirect_uri(request)
    logger.info(f"DIAGNOSTIC: Using Redirect URI: {redirect_uri}")

    auth_str = f"{FREJUN_OAUTH_CLIENT_ID}:{FREJUN_OAUTH_CLIENT_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    token_url = "https://api.frejun.com/api/v2/oauth/token/"
    form_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(token_url, data=form_data, headers=headers)

        body = resp.json() if resp.headers.get("content-type","").startswith("application/json") else {}
        logger.info(f"FreJun token exchange status: {resp.status_code}")

        # Support deep access if API wraps response in 'data'
        data_block = body.get("data", {}) if isinstance(body, dict) else {}
        access_token = body.get("access_token") or data_block.get("access_token", "")
        refresh_token = body.get("refresh_token") or data_block.get("refresh_token", "")

        if resp.status_code in (200, 201) and access_token:
            # 1. Save to the new durable DB store (The "Forever Online" bridge)
            expires_in = body.get("expires_in") or data_block.get("expires_in") or 21600
            try:
                manager._save_managed_token(access_token, refresh_token, int(expires_in), managed_email)
                logger.info(f"✅ FreJun OAuth: fresh token saved to database storage")
            except Exception as db_err:
                logger.error(f"Failed to persist FreJun token to DB: {db_err}")

            # Return a simple HTML success page
            post_auth_url = get_frejun_post_auth_url(request)
            html = f"""<!DOCTYPE html>
<html>
<head><title>FreJun Token</title>
<style>body{{font-family:monospace;padding:32px;background:#f0fdf4;}} pre{{background:#fff;padding:16px;border-radius:8px;word-break:break-all;border:1px solid #bbf7d0;max-width:900px;}} h2{{color:#166534;}}</style>
</head>
<body>
<h2>✅ FreJun OAuth Successful!</h2>
<p><strong>Tokens established and persistence bridge is LIVE.</strong></p>
<pre id="token">Access Token: {access_token[:10]}...</pre>
<p><strong>Mapped App User:</strong> {managed_email or "not provided"}</p>
<p>The system will now maintain this connection automatically. <strong>Hard-refresh</strong> your dashboard to start calling.</p>
<p><a href="{post_auth_url}">→ Go back to Calls</a></p>
</body>
</html>"""
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html)
        else:
            logger.error(f"FreJun token exchange failed: {resp.status_code} {resp.text}")
            return {
                "success": False, 
                "status": resp.status_code, 
                "error": "Token exchange failed. Verify your Client ID and Secret match the dashboard.",
                "raw_response": resp.text[:500]
            }

    except Exception as e:
        logger.exception("FreJun OAuth callback error")
        return {"success": False, "error": str(e)}

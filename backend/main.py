
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import base64
import requests as http_requests

from backend.core.config import settings
from backend.api.routes import auth, roles, candidates, stats, outreach, admin, browse, calls, voip

from contextlib import asynccontextmanager
import asyncio
from backend.api import deps
from backend.pipeline import query
from backend.db.connection import get_db_connection_context

async def warm_calls_backend():
    # Prime the DB-backed calls routes once so the first real page load
    # does not spend 10s+ initializing the pool and schema.
    try:
        await asyncio.to_thread(calls.warm_call_caches)
    except Exception as e:
        print(f"CALLS WARMUP FAILED: {e}")

async def warm_profiles_backend():
    # Warm the in-memory candidate cache before serving requests so analytics,
    # dashboard cards, and Talent Pool do not come up empty after a cold start.
    try:
        await asyncio.to_thread(query.initialize_cache)
    except Exception as e:
        print(f"PROFILE WARMUP FAILED: {e}")

async def warm_backend_caches():
    # Sequence cold-start DB work so startup does not stampede the shared pool
    # with multiple large warmers at the same time.
    await warm_calls_backend()
    await warm_profiles_backend()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Backend is starting up...")

    # Run warmup tasks in the background so they don't block the application
    # from starting and responding to health checks.
    # We use a try-except here to catch any immediate setup errors.
    try:
        # Schedule a single warmup task so DB-heavy cold-start work stays sequential.
        asyncio.create_task(warm_backend_caches())
    except Exception as e:
        print(f"CRITICAL: Background warmup task scheduling failed: {e}")

    yield

    try:
        from backend.db.connection import close_all_connections
        close_all_connections()
    except:
        pass

async def load_data_async():
    await asyncio.sleep(0.5) # Wait for the app to be fully up
    print("Starting background data loading...")
    try:
        await asyncio.to_thread(calls.ensure_calls_schema_ready)
        await asyncio.to_thread(calls.bulk_load_calls_cache)
        await asyncio.to_thread(query.initialize_cache)
        print("Background data loading complete.")
    except Exception as e:
        print(f"DATABASE CONNECTION FAILED on startup: {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="AI-powered candidate search and management",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Growton AI Backend is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/ping")
async def ping_check():
    return {"message": "pong"}

@app.get("/api/debug_db")
async def debug_db():
    from backend.pipeline.query import redis_client

    results = {"db": "testing...", "redis": "testing..."}

    # Test DB
    try:
        with get_db_connection_context(
            max_retries=1,
            validate=True,
            register_pgvector=False,
        ) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    results["db"] = "ok"
            else:
                results["db"] = "failed (no connection)"
    except Exception as e:
        results["db"] = f"failed: {str(e)}"

    # Test Redis
    try:
        if redis_client:
            redis_client.ping()
            results["redis"] = "ok"
        else:
            results["redis"] = "not initialized"
    except Exception as e:
        results["redis"] = f"failed: {str(e)}"

    return results

# Include API Routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(roles.router, prefix="/api/roles", tags=["roles"])
app.include_router(browse.router, prefix="/api", tags=["browse"])
app.include_router(candidates.router, prefix="/api", tags=["candidates"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(outreach.router, prefix="/api/outreach", tags=["outreach"])
app.include_router(calls.router, prefix="/api/calls", tags=["calls"])
app.include_router(voip.router, prefix="/api/voip", tags=["voip"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(browse.router, prefix="/api", tags=["browse"])
from backend.api.routes import enrichment
# Mount at /api for frontend calls to /api/enrich/{id}
app.include_router(enrichment.router, prefix="/api", tags=["enrichment"])
# Also mount at root for Clay callback to /results (Clay calls https://ngrok-url/results)
app.include_router(enrichment.router, tags=["enrichment-callback"])
# VoIP Token Endpoint (Directly in main.py for maximum stability)
FALLBACK_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvcmdfaWQiOjQ1OTUyLCJzY29wZSI6Im9hdXRoIiwicmVmcmVzaCI6ZmFsc2UsInRva2VuX3R5cGUiOiJhY2Nlc3MiLCJqdGkiOiIzMjI4NWQ4NC1iMWQ2LTRiYzQtYjY3YS1jNTViZWY2MDk2YWUiLCJpYXQiOjE3NzY1NzU5MDYsImV4cCI6MTc3NjU5NzUwNn0.W7sInfb7z0NtccD2Q4uT0nI1I-Wnh9RTXgmo_2V327npEHjcu6OQhX0MvN4NIEB76ELOXWoZoZXjsNxmO0RYDYxqLAG18-BLM4jgczAbKy2OeSaRfTfpe0eDcYoQ4FZRP1jgvlcWhTwm498BJjkL4h8vCAlb4rW-KcdQn8sZGo05ZBy6ebjFnQTXtUoS0155uePWdVw0J5dQpw0Y2kzgo4i_Qxg7vub_63txQVB756j81-2hIhoRui4A1dI-ebY1Q_2ZCOrk3zuVrV5FoB06sxTD2TQLeGnjsRjrbQQZyJZREnjOHBAkOHCu5BfzgLioy6ZDVOfZNPeA32I1isVPtjA"

@app.api_route("/api/voip/token", methods=["GET", "POST"])
async def get_voip_token(current_user = Depends(deps.get_current_user)):
    refresh_token = os.getenv("FREJUN_REFRESH_TOKEN", "").strip()
    client_id = os.getenv("FREJUN_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("FREJUN_CLIENT_SECRET", "").strip()

    if not refresh_token or not client_id:
        return {"access_token": FALLBACK_TOKEN, "agent_email": "ashwin@growton.co"}

    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    try:
        r = http_requests.post(
            "https://api.frejun.com/api/v2/oauth/token/",
            headers={"Authorization": f"Bearer {creds}", "Content-Type": "application/json"},
            json={"refresh_token": refresh_token, "grant_type": "refresh_token"},
            timeout=10
        )
        if r.status_code == 200:
            return {
                "access_token": r.json().get("access_token"),
                "agent_email": os.getenv("FREJUN_USER_EMAIL", "ashwin@growton.co")
            }
    except: pass
    return {"access_token": FALLBACK_TOKEN, "agent_email": "ashwin@growton.co"}


from fastapi.responses import HTMLResponse as _HTMLResponse

@app.get("/api/auth/frejun-callback")
async def frejun_oauth_callback_main(code: str = None, error: str = None, email: str = None):
    """FreJun OAuth callback — set redirect URL to http://localhost:3002/api/auth/frejun-callback"""
    if error:
        return {"success": False, "error": error}
    if not code:
        return {"success": False, "error": "No authorization code received"}

    client_id = os.getenv("FREJUN_OAUTH_CLIENT_ID", os.getenv("FREJUN_CLIENT_ID", "")).strip()
    client_secret = os.getenv("FREJUN_CLIENT_SECRET", "").strip()
    redirect_uri = "http://localhost:3002/api/auth/frejun-callback"

    token_url = "https://api.frejun.com/api/v2/oauth/token/"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    import asyncio as _asyncio
    import httpx as _httpx
    import re as _re

    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(token_url, json=payload)

        body = resp.json() if resp.headers.get("content-type","").startswith("application/json") else {}
        access_token = body.get("access_token") or body.get("data", {}).get("access_token", "") if isinstance(body, dict) else ""
        refresh_token = body.get("refresh_token") or body.get("data", {}).get("refresh_token", "") if isinstance(body, dict) else ""

        if resp.status_code in (200, 201) and access_token:
            # Save to .env
            env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
            for key, val in [("FREJUN_ACCESS_TOKEN", access_token), ("FREJUN_REFRESH_TOKEN", refresh_token)]:
                if val:
                    try:
                        lines = open(env_path).readlines() if os.path.exists(env_path) else []
                        found = False
                        new_lines = []
                        for l in lines:
                            if l.startswith(f"{key}="):
                                new_lines.append(f"{key}={val}\n")
                                found = True
                            else:
                                new_lines.append(l)
                        if not found:
                            new_lines.append(f"{key}={val}\n")
                        open(env_path, "w").writelines(new_lines)
                    except: pass

            html = f"""<!DOCTYPE html>
<html><head><title>FreJun Token Ready</title>
<style>body{{font-family:monospace;padding:32px;background:#f0fdf4;}} pre{{background:#fff;padding:16px;border-radius:8px;word-break:break-all;border:1px solid #bbf7d0;max-width:900px;overflow-x:auto}} h2{{color:#166534;}}</style>
</head><body>
<h2>✅ FreJun OAuth Successful!</h2>
<p>Access token saved to <code>.env</code> automatically. <strong>Hard-refresh</strong> the Growton app (Cmd+Shift+R) to pick it up.</p>
<p><strong>Token:</strong></p><pre>{access_token}</pre>
<p><a href="http://localhost:3000/calls">→ Go back to Calls</a></p>
</body></html>"""
            return _HTMLResponse(content=html)

        return {"success": False, "status": resp.status_code, "body": resp.text[:500], "note": "Token exchange failed - code may be expired, please authorize again"}

    except Exception as e:
        return {"success": False, "error": str(e)}

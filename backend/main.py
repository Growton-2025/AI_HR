
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.api.routes import auth, roles, candidates, stats, outreach, admin, browse, calls, voip

from contextlib import asynccontextmanager
import asyncio
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


# Root routes
@app.get("/")
def read_root():
    return {"message": "Growton AI Backend is running"}

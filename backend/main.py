
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.api.routes import auth, roles, candidates, stats, outreach, admin, browse

from contextlib import asynccontextmanager
import asyncio
from backend.pipeline import query

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DONT let startup block! Let it yield immediately so Azure is happy.
    print("Backend is starting up...")
    asyncio.create_task(load_data_async())
    yield
    from backend.db.connection import close_all_connections
    close_all_connections()

async def load_data_async():
    await asyncio.sleep(2) # Wait for the app to be fully up
    print("Starting background data loading...")
    try:
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
def health_check():
    return {"status": "ok"}

@app.get("/api/ping")
def ping_check():
    return {"message": "pong"}

@app.get("/api/debug_db")
async def debug_db():
    from backend.db.connection import get_db_connection, return_db_connection
    from backend.pipeline.query import redis_client
    
    results = {"db": "testing...", "redis": "testing..."}
    
    # Test DB
    try:
        conn = get_db_connection(max_retries=1)
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                results["db"] = "ok"
            return_db_connection(conn)
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
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(browse.router, prefix="/api", tags=["browse"])
from backend.api.routes import enrichment
# Mount at /api for frontend calls to /api/enrich/{id}
app.include_router(enrichment.router, prefix="/api", tags=["enrichment"])
# Also mount at root for Clay callback to /results (Clay calls https://ngrok-url/results)
app.include_router(enrichment.router, tags=["enrichment-callback"])
 

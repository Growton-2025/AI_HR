
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
    # Load data in background to prevent startup blocking
    asyncio.create_task(load_data_async())
    yield
    # Cleanup: Close all database connections
    from backend.db.connection import close_all_connections
    close_all_connections()

async def load_data_async():
    print("Starting background data loading...")
    await asyncio.to_thread(query.initialize_cache)
    print("Background data loading complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="AI-powered candidate search and management",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

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
 


import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import auth, candidates, roles

# Initialize FastAPI
app = FastAPI(
    title="Growton AI - Talent Intelligence API",
    description="AI-powered candidate search and management",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(candidates.router)
app.include_router(roles.router)

@app.get("/api/health")
async def health_check():
    return {"message": "Growton AI API is running", "version": "1.0.0"}

# --- Serve Frontend (Combined Deployment) ---
# Mount static files if the build directory exists
# Go up from app/main.py -> backend -> web -> frontend -> dist
frontend_dist = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dist')
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    # Explicit root handler
    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    # Catch-all route for SPA (React Router)
    # This must be defined AFTER all API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Check if file exists in dist (e.g. favicon.ico, logo.png)
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Otherwise serve index.html
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"Frontend build not found at {frontend_dist}. Run 'npm run build' in frontend directory.")

# --- Run with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.api.routes import auth, roles, candidates, stats, outreach, admin, browse, calls, candidate_imports, ai_columns

from contextlib import asynccontextmanager
import asyncio
from backend.pipeline import query
from backend.db.connection import get_db_connection_context, get_connection_pool_state


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _apply_pool_migrations_blocking() -> None:
    """Run pool/ownership migrations before any request — avoids login 500s if `archived_at` is missing."""
    from backend.db.connection import get_db_connection, return_db_connection
    from backend.db.ai_column_migrate import ensure_ai_column_migrations
    from backend.db.candidate_pool_migrate import ensure_candidate_pool_migrations
    from backend.db.outreach_migrate import ensure_outreach_migrations
    from backend.db.resume_migrate import ensure_resume_migrations

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        print("WARNING: Could not connect to DB for startup migrations.")
        return
    try:
        ensure_candidate_pool_migrations(conn)
        ensure_ai_column_migrations(conn)
        ensure_outreach_migrations(conn)
        ensure_resume_migrations(conn)
    finally:
        return_db_connection(conn)


def _apply_outreach_migrations_blocking() -> None:
    """Ensure durable outreach columns exist before role dispatchers start."""
    from backend.db.connection import get_db_connection, return_db_connection
    from backend.db.outreach_migrate import ensure_outreach_migrations
    from backend.db.resume_migrate import ensure_resume_migrations

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Could not connect to DB for outreach migrations")
    try:
        ensure_outreach_migrations(conn)
        # Resume storage rides the unconditional migration path because
        # ENABLE_STARTUP_MIGRATIONS defaults to false in prod.
        ensure_resume_migrations(conn)
    finally:
        return_db_connection(conn)


async def warm_calls_backend():
    # Prime the DB-backed calls routes once so the first real page load
    # does not spend 10s+ initializing the pool and schema.
    try:
        # Runs the per-process schema check here (sentinel = 1 round trip)
        # so the first user request never pays for it.
        await asyncio.to_thread(calls.ensure_calls_schema_ready)
        await asyncio.to_thread(calls.warm_call_caches)
        print("Calls cache warmed successfully.")
    except Exception as e:
        print(f"CALLS WARMUP FAILED: {e}")

async def warm_profiles_backend():
    # Warm the in-memory candidate cache before serving requests so analytics,
    # dashboard cards, and Talent Pool do not come up empty after a cold start.
    try:
        await asyncio.to_thread(query.initialize_cache)
        print("Profiles cache warmed successfully.")
    except Exception as e:
        print(f"PROFILE WARMUP FAILED: {e}")

async def warm_backend_caches():
    # Sequence cold-start DB work so startup does not stampede the shared pool
    # with multiple large warmers at the same time.
    await warm_calls_backend()
    await warm_profiles_backend()

def _register_smartlead_webhooks(public_url: str) -> None:
    """Point every live campaign's EMAIL_REPLY event at our webhook.

    Blocking (requests + DB), so callers run it off the event loop. Campaign ids
    stored in candidate_outreach can be stale — Smartlead answers "Campaign not
    found" for several of them — so a failure on one campaign must never stop
    the rest from registering.
    """
    from backend.db.connection import get_db_connection_context
    from backend.integrations.smartlead import SmartleadBot

    api_key = os.getenv("SMARTLEAD_API_KEY")
    if not api_key:
        print("Smartlead webhook registration skipped: SMARTLEAD_API_KEY is not set")
        return

    campaign_ids = []
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT campaign_id
                    FROM candidate_outreach
                    WHERE campaign_id IS NOT NULL
                      AND updated_at > NOW() - INTERVAL '90 days'
                    """
                )
                campaign_ids = [row[0] for row in cur.fetchall() if row[0]]
    except Exception as e:
        print(f"Smartlead webhook registration could not list campaigns: {e}")
        return

    if not campaign_ids:
        print("Smartlead webhook registration skipped: no recent campaigns")
        return

    bot = SmartleadBot(api_key=api_key)
    registered = 0
    for campaign_id in campaign_ids:
        try:
            if bot.ensure_reply_webhook(campaign_id, public_url):
                registered += 1
        except Exception as e:
            print(f"Smartlead webhook registration failed for campaign {campaign_id}: {e}")
    print(f"Smartlead reply webhook registered on {registered}/{len(campaign_ids)} campaign(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Backend is starting up...")

    # Run migrations in the background so the gunicorn worker handshake
    # completes within the timeout window. The migrations are idempotent
    # (IF NOT EXISTS) so in-flight requests are safe even on a fresh schema.
    async def _run_migrations_bg():
        try:
            await asyncio.to_thread(_apply_pool_migrations_blocking)
            print("Startup migrations completed.")
        except Exception as e:
            print(f"STARTUP MIGRATION FAILED (non-fatal, will retry on next request): {e}")

    if _env_flag("ENABLE_STARTUP_MIGRATIONS", "false"):
        asyncio.create_task(_run_migrations_bg())
    else:
        print("Startup migrations skipped in worker lifespan.")

    # Initialize Plivo background endpoint logic
    if _env_flag("ENABLE_PLIVO_SETUP", "false"):
        try:
            from backend.integrations import plivo_service

            async def _plivo_setup():
                await plivo_service.setup_plivo()
                # Inbound needs its own application bound to PLIVO_NUMBER, and it
                # must be re-pointed whenever the public tunnel URL rotates —
                # otherwise incoming calls silently stop reaching us.
                await plivo_service.ensure_inbound_application()

            asyncio.create_task(_plivo_setup())
        except Exception as e:
            print(f"Failed to kick off Plivo setup: {e}")

    try:
        ai_columns.start_daily_ai_column_refresh_scheduler()
    except Exception as e:
        print(f"AI COLUMN DAILY REFRESH SCHEDULER FAILED TO START: {e}")

    try:
        await asyncio.to_thread(_apply_outreach_migrations_blocking)
        from backend.services import heyreach_role_campaigns, smartlead_role_dispatcher
        heyreach_role_campaigns.start_dispatcher()
        smartlead_role_dispatcher.start_dispatcher()
    except Exception as e:
        print(f"ROLE OUTREACH DISPATCHERS FAILED TO START: {e}")

    # LinkedIn replies must reach the lists WITHOUT anyone opening a modal or
    # pressing Sync — one workspace-wide watermark poll per cycle. The webhook
    # (HEYREACH_WEBHOOK_URL) remains the real-time path in hosted envs.
    try:
        from backend.services import heyreach_reply_sync
        heyreach_reply_sync.start_poller()
    except Exception as e:
        print(f"HEYREACH REPLY POLLER FAILED TO START: {e}")

    # Email replies had no background path at all: no webhook was ever
    # registered with Smartlead (the live campaign's only EMAIL_REPLY
    # subscriber points at Clay), so a reply reached the lists only when a
    # recruiter pressed "Sync Responses".
    try:
        from backend.services import smartlead_reply_sync
        smartlead_reply_sync.start_poller()
    except Exception as e:
        print(f"SMARTLEAD REPLY POLLER FAILED TO START: {e}")

    # Register the HeyReach reply webhook (EVERY_MESSAGE_REPLY_RECEIVED) so
    # candidate replies land in near real-time instead of waiting for a manual
    # sync. Requires the backend to be publicly reachable; set e.g.
    # HEYREACH_WEBHOOK_URL=https://<host>/api/outreach/heyreach/webhook
    heyreach_webhook_url = os.getenv("HEYREACH_WEBHOOK_URL")
    if heyreach_webhook_url:
        try:
            from backend.integrations.heyreach import HeyReachBot

            await asyncio.to_thread(
                HeyReachBot().ensure_reply_webhook, heyreach_webhook_url
            )
        except Exception as e:
            print(f"HEYREACH WEBHOOK REGISTRATION FAILED (non-fatal): {e}")

    # Register the Smartlead reply webhook (EMAIL_REPLY) on every campaign this
    # workspace is actually running, so email replies land in real time instead
    # of waiting for the poller or a manual sync. Requires the backend to be
    # publicly reachable; set e.g.
    # SMARTLEAD_WEBHOOK_URL=https://<host>/api/outreach/smartlead/webhook
    #
    # Unlike HeyReach, Smartlead webhooks are per campaign, so this registers
    # one per active campaign id found in candidate_outreach. Existing hooks
    # belonging to other integrations are left untouched.
    smartlead_webhook_url = os.getenv("SMARTLEAD_WEBHOOK_URL")
    if smartlead_webhook_url:
        try:
            await asyncio.to_thread(_register_smartlead_webhooks, smartlead_webhook_url)
        except Exception as e:
            print(f"SMARTLEAD WEBHOOK REGISTRATION FAILED (non-fatal): {e}")

    # Warm calls cache AFTER migrations so the DB pool is fully available.
    # This is awaited directly (takes ~2-3s) so it is ready before the first request.
    print("Warming calls cache...")
    await warm_calls_backend()

    # Optionally warm the full profile/candidate cache (heavier, off by default).
    # Awaited directly (~20-30s, mostly cross-region pool init) so it is ready
    # before the first request — previously fired via create_task() and left
    # unawaited, so the server accepted traffic immediately and the first
    # analytics/browse/dashboard request after every restart blocked on the
    # cold cache instead.
    if _env_flag("ENABLE_STARTUP_CACHE_WARMUP", "false"):
        print("Warming profiles cache...")
        await warm_profiles_backend()
    else:
        print("Profile cache warmup skipped; will load lazily on first search.")

    yield

    try:
        ai_columns.stop_daily_ai_column_refresh_scheduler()
    except Exception:
        pass

    try:
        from backend.services import heyreach_role_campaigns, smartlead_role_dispatcher
        heyreach_role_campaigns.stop_dispatcher()
        smartlead_role_dispatcher.stop_dispatcher()
    except Exception:
        pass

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

_DIAGNOSTIC_PATHS = {
    "/api/health",
    "/api/candidates/analytics",
    "/api/candidates/browse",
    "/api/candidates/browse/summary",
    "/api/calls",
    "/api/calls/lists",
    "/api/calls/stats",
}


@app.middleware("http")
async def diagnostic_timing_middleware(request, call_next):
    start = time.monotonic()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        if request.url.path in _DIAGNOSTIC_PATHS:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            status_code = getattr(response, "status_code", "error")
            print(f"DIAG request path={request.url.path} status={status_code} duration_ms={elapsed_ms} pool={get_connection_pool_state()}")

@app.get("/")
def read_root():
    return {"message": "Hayasa.ai Backend is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/ping")
async def ping_check():
    return {"message": "pong"}

@app.get("/api/debug_db")
async def debug_db():
    from backend.pipeline.query import (
        PROFILES_BY_ID,
        count_active_candidates_from_db,
        is_cache_initialized,
        redis_client,
    )

    results = {"db": "testing...", "redis": "testing..."}
    results["pool"] = get_connection_pool_state()
    results["cache_initialized"] = is_cache_initialized()
    results["profile_count"] = len(PROFILES_BY_ID)
    results["active_candidate_count"] = None

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

    results["active_candidate_count"] = count_active_candidates_from_db()
    results["pool"] = get_connection_pool_state()

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
app.include_router(candidate_imports.router, prefix="/api", tags=["imports"])
from backend.api.routes import resumes
app.include_router(resumes.router, prefix="/api", tags=["resumes"])
app.include_router(candidates.router, prefix="/api", tags=["candidates"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(outreach.router, prefix="/api/outreach", tags=["outreach"])
app.include_router(calls.router, prefix="/api/calls", tags=["calls"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(ai_columns.router, prefix="/api", tags=["ai-columns"])
from backend.api.routes import plivo
app.include_router(plivo.router, prefix="/api/plivo", tags=["plivo"])
from backend.api.routes import enrichment
# Mount at /api for frontend calls to /api/enrich/{id}
app.include_router(enrichment.router, prefix="/api", tags=["enrichment"])
# Also mount at root for Clay callback to /results (Clay calls https://ngrok-url/results)
app.include_router(enrichment.router, tags=["enrichment-callback"])


# Root routes
@app.get("/")
def read_root():
    return {"message": "Hayasa.ai Backend is running"}

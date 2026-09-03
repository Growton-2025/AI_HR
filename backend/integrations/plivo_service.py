import os
import hashlib
import logging
import re
from collections import Counter
import requests
import threading
import time
import plivo
from openai import AsyncOpenAI
import tempfile
import asyncio
import json
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")
PLIVO_NUMBER = os.getenv("PLIVO_NUMBER")

endpoint_username = ""
endpoint_password = ""
endpoint_public_url = ""
setup_error = ""
_setup_lock = asyncio.Lock()
_last_setup_ngrok_url = ""

# Store recordings in memory: CallUUID -> RecordingUrl
recordings = {}
last_calls = {}
last_call_states = {}
# dial_token -> dial state. Keyed per attempt, unlike last_call_states (keyed by
# SIP username), which is never cleared and so hands a redial the previous
# call's UUID.
dial_token_states = {}
call_insights = {}
latest_call_uuid = None

openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0)
    return openai_client

def get_ngrok_url():
    # Priority 1: Explicit environment override for hosted/production deployments
    env_url = os.getenv("PUBLIC_URL") or os.getenv("NGROK_URL")
    if env_url:
        return env_url.rstrip('/')

    # Priority 2: Fallback to local active ngrok tunnel API
    try:
        response = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=3)
        response.raise_for_status()
        data = response.json()
        for tunnel in data.get('tunnels', []):
            if tunnel.get('proto') == 'https':
                return tunnel.get('public_url')
        if data.get('tunnels'):
            return data['tunnels'][0]['public_url']
    except Exception:
        pass

    # Priority 3: Cloudflare quick tunnel (started by start_services.sh with
    # --metrics 127.0.0.1:20241; needs no account, URL changes per session).
    try:
        response = requests.get('http://127.0.0.1:20241/quicktunnel', timeout=3)
        response.raise_for_status()
        hostname = response.json().get('hostname')
        if hostname:
            return f"https://{hostname}"
    except Exception:
        pass

    logger.error("No public tunnel found: set PUBLIC_URL/NGROK_URL, or run ngrok / cloudflared.")
    return None

def normalize_number(number):
    if not number:
        return ""
    digits = "".join([c for c in number if c.isdigit()])
    if len(digits) == 10:
        return f"+91{digits}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    else:
        return f"+{digits}" if digits else ""


# Bound concurrent REST recording lookups so a burst of frontend sync polls
# cannot stack blocking Plivo calls (same pattern as resume_service._parse_semaphore).
_recording_lookup_semaphore = threading.BoundedSemaphore(
    int(os.getenv("PLIVO_MAX_CONCURRENT_RECORDING_LOOKUPS", "2"))
)


def lookup_recording_url(call_uuid: str):
    """Blocking Plivo REST lookup of a call's recording URL (run via asyncio.to_thread).

    Recovers recordings whose webhook callback was lost (e.g. backend restart
    wiped the in-memory `recordings` map, or ngrok was down when Plivo called).
    """
    if not call_uuid or not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
        return None
    with _recording_lookup_semaphore:
        try:
            client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
            result = client.recordings.list(call_uuid=call_uuid, limit=1)
            objects = getattr(result, "objects", None) or (result if isinstance(result, list) else [])
            for rec in objects:
                url = getattr(rec, "recording_url", None)
                if url:
                    return url
        except Exception as e:
            logger.warning(f"Plivo recording lookup failed for {call_uuid}: {e}")
    return None


def download_plivo_recording(record_url: str, timeout: int = 30):
    auth = (PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN) if PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN else None
    return requests.get(record_url, timeout=timeout, auth=auth)


def persist_recording_url(call_uuid: str, record_url: str) -> bool:
    """Store the recording on the call row as soon as its URL is known.

    Plivo hands us the URL in the recording callback, but it used to be written
    only at the END of process_call_insights — in the same UPDATE as the
    transcript and summary. So the player the recruiter is waiting on was gated
    behind a download, a transcription and two LLM passes: measured at 8s for a
    25-second call and over three minutes for a twenty-minute one, for a URL we
    had held since the moment the call ended.

    The row is the shared source of truth; the in-memory `recordings` map is
    per gunicorn worker, so a poll served by a sibling worker cannot see it.
    """
    if not call_uuid or not record_url:
        return False
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE calls
                   SET recording_url = %s, updated_at = NOW()
                 WHERE (plivo_call_uuid = %s OR plivo_transaction_id = %s)
                   AND COALESCE(recording_url, '') <> %s
                """,
                (record_url, call_uuid, call_uuid, record_url),
            )
            updated = cur.rowcount or 0
        conn.commit()
        if updated:
            logger.info("Stored recording_url for %s on %d call row(s)", call_uuid, updated)
        return updated > 0
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not store recording_url for %s: %s", call_uuid, exc)
        return False
    finally:
        return_db_connection(conn)


# How long a claimed insights run is assumed to still be running. Long enough
# to cover transcription of a long call, short enough that a worker killed
# mid-run does not block a retry for the rest of the day.
INSIGHTS_CLAIM_STALE_SECONDS = 15 * 60


def claim_insights_run(call_uuid: str) -> bool:
    """Take the right to transcribe this call, across workers. True if we won.

    The review modal polls every 5s while the client gives up on each request
    after 15s, so a single long call could stack half a dozen overlapping
    transcription runs — each one a paid OpenAI request for the same audio,
    all competing for the same worker pool. The in-process guard could not see
    any of them because it lives in a different process.
    """
    if not call_uuid:
        return False
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        # No database means no transcript could be stored even if we ran, and
        # the poll repeats every few seconds — granting the claim here would
        # start a fresh transcription on every one of them.
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE calls
                   SET recording_synced_at = NOW()
                 WHERE (plivo_call_uuid = %s OR plivo_transaction_id = %s)
                   AND (recording_synced_at IS NULL
                        OR recording_synced_at < NOW() - (%s * INTERVAL '1 second'))
                """,
                (call_uuid, call_uuid, INSIGHTS_CLAIM_STALE_SECONDS),
            )
            won = (cur.rowcount or 0) > 0
        conn.commit()
        return won
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not claim insights run for %s: %s", call_uuid, exc)
        # An unexpected query failure is rare and usually transient, so fall
        # back to the old behaviour — a possible duplicate run — rather than
        # dropping the transcript entirely.
        return True
    finally:
        return_db_connection(conn)


_PLIVO_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "plivo_softphone_state.json",
)


# Identifies which deployment owns a Plivo Application. The same Postgres is
# shared between a developer's laptop and the hosted backend, so without this
# they claim each other's app — which is exactly how hosted ended up handing out
# an endpoint bound to a laptop's dead ngrok tunnel.
#
# Derived rather than configured: hosted sets PUBLIC_URL (stable), local relies
# on tunnel auto-detection (rotating), so presence of the variable is itself the
# signal. No new deploy setting to forget.
def _env_key() -> str:
    explicit = os.getenv("PUBLIC_URL") or os.getenv("NGROK_URL")
    if not explicit:
        return "local"
    host = explicit.split("://")[-1].split("/")[0].strip().lower()
    return host or "local"


# Plivo rejects an endpoint alias that is longer than 50 characters or carries
# anything outside letters, digits, hyphen and underscore. _env_key() is a
# hostname on hosted deployments — 69 characters with three dots — so an alias
# built straight from it was refused by the API on every single call, and no
# recruiter on the hosted backend could ever be given a line. Keep the readable
# head of the host for the console, and a hash tail so two deployments that
# share a prefix can never collide on one endpoint.
_ALIAS_MAX = 50
# How deep to page when hunting for an endpoint to adopt. Bounded so a huge
# account cannot turn one login into an unbounded crawl of the Plivo API.
_ADOPT_SCAN_LIMIT = 400


def _env_alias_slug() -> str:
    env_key = _env_key()
    head = re.sub(r"[^a-z0-9-]+", "-", env_key.lower()).strip("-")[:20].strip("-")
    digest = hashlib.sha256(env_key.encode("utf-8")).hexdigest()[:8]
    return f"{head}_{digest}" if head else digest


def endpoint_alias_for_user(user_id: int) -> str:
    """The alias this user's endpoint carries in Plivo, for this environment."""
    return f"recruiter_{user_id}_{_env_alias_slug()}"[:_ALIAS_MAX]


# Arbitrary but fixed: the advisory-lock id every worker agrees on before
# creating an Application. asyncio.Lock cannot help here — gunicorn runs 4
# separate processes, which is how one restart could mint four Applications.
_PLIVO_PROVISION_LOCK_KEY = 918035312881 % 2147483647


def _load_app_state(kind: str):
    """Read this environment's Application record. None if absent or DB is down."""
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT app_id, answer_url, username, password FROM plivo_app_state "
                "WHERE kind = %s AND env_key = %s",
                (kind, _env_key()),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"app_id": row[0], "answer_url": row[1], "username": row[2], "password": row[3]}
    except Exception as exc:
        logger.warning("Could not read plivo_app_state(%s): %s", kind, exc)
        return None
    finally:
        return_db_connection(conn)


def _save_app_state(kind: str, app_id: str, answer_url: str,
                    username: str = None, password: str = None) -> bool:
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO plivo_app_state (kind, env_key, app_id, answer_url, username, password, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (kind, env_key) DO UPDATE
                SET app_id = EXCLUDED.app_id,
                    answer_url = EXCLUDED.answer_url,
                    username = COALESCE(EXCLUDED.username, plivo_app_state.username),
                    password = COALESCE(EXCLUDED.password, plivo_app_state.password),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (kind, _env_key(), str(app_id), answer_url, username, password),
            )
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not persist plivo_app_state(%s): %s", kind, exc)
        return False
    finally:
        return_db_connection(conn)


def _clear_app_state(kind: str) -> None:
    """Forget this environment's Application record.

    Called when the stored app turns out not to exist in Plivo any more (it was
    deleted, or the row was written by something that never created a real one).
    Without this the recovery path is a loop: reuse fails, we fall through to
    provisioning, and _provision_app_once reads the same broken row back and
    returns it again — so the environment can never heal itself.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM plivo_app_state WHERE kind = %s AND env_key = %s",
                (kind, _env_key()),
            )
        conn.commit()
        logger.warning("Cleared unusable Plivo %s app state for %s", kind, _env_key())
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not clear plivo_app_state(%s): %s", kind, exc)
    finally:
        return_db_connection(conn)


def _provision_app_once(kind: str, answer_url: str, create_fn):
    """Create this environment's Application exactly once across all workers.

    The advisory lock is held for the whole check-create-persist cycle,
    including the Plivo REST call. Holding a lock across a network call is
    normally worth avoiding, but it is the only thing that stops four gunicorn
    workers each creating an Application on the same restart, and it happens
    once per environment rather than per request.

    Returns the state dict, or None when the DB is unavailable so callers can
    fall back to the legacy on-disk path.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_PLIVO_PROVISION_LOCK_KEY,))
            cur.execute(
                "SELECT app_id, answer_url, username, password FROM plivo_app_state "
                "WHERE kind = %s AND env_key = %s",
                (kind, _env_key()),
            )
            row = cur.fetchone()
            if row:
                conn.commit()
                return {"app_id": row[0], "answer_url": row[1],
                        "username": row[2], "password": row[3]}

            created = create_fn()  # still inside the lock — one worker only
            cur.execute(
                """
                INSERT INTO plivo_app_state (kind, env_key, app_id, answer_url, username, password, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (kind, env_key) DO UPDATE
                SET app_id = EXCLUDED.app_id, answer_url = EXCLUDED.answer_url,
                    username = EXCLUDED.username, password = EXCLUDED.password,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (kind, _env_key(), str(created["app_id"]), answer_url,
                 created.get("username"), created.get("password")),
            )
        conn.commit()
        return created
    except Exception as exc:
        conn.rollback()
        logger.error("Could not provision Plivo application (%s): %s", kind, exc)
        return None
    finally:
        return_db_connection(conn)


def _load_persisted_softphone_state():
    try:
        with open(_PLIVO_STATE_FILE, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        if all(state.get(key) for key in ("app_id", "username", "password", "answer_url")):
            return state
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not read persisted Plivo softphone state: {e}")
    return None


def _persist_softphone_state(app_id: str, username: str, password: str, answer_url: str):
    try:
        os.makedirs(os.path.dirname(_PLIVO_STATE_FILE), exist_ok=True)
        with open(_PLIVO_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {"app_id": app_id, "username": username, "password": password, "answer_url": answer_url},
                fh,
            )
    except Exception as e:
        logger.warning(f"Could not persist Plivo softphone state: {e}")


_INBOUND_STATE_FILE = os.path.join(os.path.dirname(_PLIVO_STATE_FILE), "plivo_inbound_state.json")


def _inbound_number_owner_env() -> Optional[str]:
    """Which environment currently owns the inbound number, if any.

    There is one DID for the whole account, so inbound cannot be shared between
    a laptop and the hosted backend — the last writer wins and the other goes
    dark.

    Ownership is read from **Plivo**, not from our own most-recently-updated
    row. The latter is self-defeating: an environment provisions its inbound app
    (writing its own row) moments before this check, so it always looked like
    the owner and every local restart happily stole the number.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN or not PLIVO_NUMBER:
        return None

    try:
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        number = client.numbers.get(PLIVO_NUMBER)
        bound_app = str(getattr(number, "application", "") or "").rstrip("/").split("/")[-1]
    except Exception as exc:
        # Fail closed: if we cannot tell who owns it, do not take it.
        logger.warning("Could not read the inbound number's current app: %s", exc)
        return "unknown"

    if not bound_app:
        return None  # unbound — free to claim

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return "unknown"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT env_key FROM plivo_app_state WHERE kind = 'inbound' AND app_id = %s",
                (bound_app,),
            )
            row = cur.fetchone()
        # An app we have no record of belongs to something else (a hand-made
        # app, another deployment) — treat it as owned, not free.
        return row[0] if row else "unknown"
    except Exception as exc:
        logger.warning("Could not determine inbound number owner: %s", exc)
        return "unknown"
    finally:
        return_db_connection(conn)


async def ensure_inbound_application() -> Optional[str]:
    """Provision (or re-point) the Application that answers calls to PLIVO_NUMBER.

    This must be a SEPARATE application from the softphone one: that app answers
    at /api/plivo/dial, which expects a `To` field to place an outbound leg, so
    binding the number to it would push inbound calls into the outbound dialer.

    Like the softphone app, the answer URL has to be re-pointed whenever the
    public tunnel rotates, or inbound silently breaks.
    """
    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN or not PLIVO_NUMBER:
        return None
    ngrok_url = get_ngrok_url()
    if not ngrok_url:
        logger.warning("No public URL yet — skipping inbound Plivo application setup.")
        return None

    answer_url = f"{ngrok_url}/api/plivo/incoming"
    hangup_url = f"{ngrok_url}/api/plivo/incoming-hangup"
    client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)

    # Postgres first; the JSON file is a fallback for when the DB is down. The
    # file alone meant hosted lost the app on every deploy and left PLIVO_NUMBER
    # bound to a stale Application — inbound calls hit a dead tunnel silently.
    state = await asyncio.to_thread(_load_app_state, "inbound")
    if not state:
        try:
            with open(_INBOUND_STATE_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except FileNotFoundError:
            state = {}
        except Exception as exc:
            logger.warning("Could not read inbound Plivo state: %s", exc)
            state = {}

    app_id = (state or {}).get("app_id")
    try:
        if app_id:
            if (state or {}).get("answer_url") != answer_url:
                try:
                    await asyncio.to_thread(
                        client.applications.update,
                        app_id=app_id, answer_url=answer_url, answer_method="POST",
                        hangup_url=hangup_url, hangup_method="POST",
                    )
                    logger.info("Re-pointed inbound Plivo app %s to %s", app_id, answer_url)
                except Exception as reuse_err:
                    # The stored app no longer exists — forget it and create one,
                    # rather than failing inbound setup on every restart.
                    logger.warning("Inbound app %s unusable (%s); provisioning fresh.",
                                   app_id, reuse_err)
                    await asyncio.to_thread(_clear_app_state, "inbound")
                    app_id = None

        if not app_id:
            def _create_inbound():
                created = client.applications.create(
                    app_name=f"Inbound_App_{time.time_ns()}",
                    answer_url=answer_url, answer_method="POST",
                    hangup_url=hangup_url, hangup_method="POST",
                )
                logger.info("Created inbound Plivo app %s", created.app_id)
                return {"app_id": created.app_id}

            created = await asyncio.to_thread(
                _provision_app_once, "inbound", answer_url, _create_inbound,
            )
            if not created:
                created = await asyncio.to_thread(_create_inbound)
            app_id = created["app_id"]

        # Point the rented number at it. Idempotent — safe to repeat on restart.
        #
        # But there is only ONE number, so whoever calls this last owns inbound
        # for the whole account. A developer starting the stack locally would
        # silently take candidate callbacks away from the hosted deployment and
        # route them to a laptop tunnel that dies on sleep. Refuse unless this
        # environment already owns it, or the takeover is explicit.
        owner_env = await asyncio.to_thread(_inbound_number_owner_env)
        may_claim = (
            owner_env is None
            or owner_env == _env_key()
            or os.getenv("PLIVO_CLAIM_NUMBER", "").strip().lower() in ("1", "true", "yes")
        )
        if not may_claim:
            logger.warning(
                "Not binding %s to app %s: inbound is owned by environment %r. "
                "Set PLIVO_CLAIM_NUMBER=true to take it over deliberately.",
                PLIVO_NUMBER, app_id, owner_env,
            )
            return app_id
        await asyncio.to_thread(client.numbers.update, number=PLIVO_NUMBER, app_id=app_id)

        await asyncio.to_thread(_save_app_state, "inbound", app_id, answer_url)
        try:
            with open(_INBOUND_STATE_FILE, "w", encoding="utf-8") as fh:
                json.dump({"app_id": app_id, "answer_url": answer_url}, fh)
        except Exception:
            pass  # DB is the source of truth; the file is only a fallback
        return app_id
    except Exception as exc:
        logger.error("Failed to set up inbound Plivo application: %s", exc)
        return None


def record_browser_dial(username: str, call_uuid: str, to_number: str, dial_token: str = None):
    global latest_call_uuid
    if not username or not call_uuid:
        return

    normalized_to = normalize_number(to_number)
    state = {
        "call_uuid": call_uuid,
        "username": username,
        "to_number": normalized_to,
        "seen_at": time.time(),
        "dial_token": dial_token or "",
    }
    last_calls[username] = call_uuid
    last_call_states[username] = state
    latest_call_uuid = call_uuid
    if dial_token:
        # Keyed by attempt rather than by username, so the dial handshake cannot
        # be satisfied by a previous call's UUID (last_calls is never cleared).
        dial_token_states[dial_token] = state

    # Server-side busy signal. The browser also beacons busy/idle, but this
    # webhook is the one signal that cannot be lost to a wedged tab, and an
    # outbound dial is unambiguously "this recruiter is now on a call".
    try:
        mark_endpoint_busy(username=username)
    except Exception as exc:
        logger.warning("Could not mark %s busy on dial: %s", username, exc)

    try:
        from backend.api.routes.calls import get_calls_db_connection, return_db_connection
        conn = get_calls_db_connection()
        if not conn:
            return
        cur = conn.cursor()
        matched = 0
        if dial_token:
            # Exact match on the attempt that produced this call. No ordering,
            # no recency guess, so concurrent recruiters (and rapid redials by
            # one recruiter) can never claim each other's row.
            cur.execute(
                """
                UPDATE calls
                SET
                    plivo_call_uuid = %s,
                    plivo_status = 'dial_received',
                    updated_at = NOW()
                WHERE dial_token = %s
                """,
                (call_uuid, dial_token),
            )
            matched = cur.rowcount or 0

        if not matched:
            # Legacy path for clients running a bundle from before dial tokens
            # shipped. It is the exact bug this replaces — it picks the most
            # recently updated row for the SIP username, which is the wrong row
            # whenever two attempts share a username. Logged loudly so we can
            # confirm the token path covers everything, then delete this.
            # See docs/call-attribution-plan.md Phase 3.4.
            logger.warning(
                "[DialAttribution] Falling back to username matching for call %s "
                "(username=%s, dial_token=%s). This path can attribute the "
                "recording to the wrong candidate.",
                call_uuid, username, dial_token or "<none>",
            )
            cur.execute(
                """
                UPDATE calls
                SET
                    plivo_call_uuid = %s,
                    plivo_status = 'dial_received',
                    updated_at = NOW()
                WHERE id = (
                    SELECT id
                    FROM calls
                    WHERE plivo_endpoint_username = %s
                      AND status IN ('pending', 'in_progress', 'completed')
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT 1
                )
                """,
                (call_uuid, username),
            )
        else:
            logger.info("[DialAttribution] Matched call %s by dial token.", call_uuid)
        conn.commit()
        cur.close()
        return_db_connection(conn)
    except Exception as db_err:
        logger.error(f"Failed to persist Plivo dial state for {username}: {db_err}")

async def setup_plivo(force: bool = False):
    global endpoint_username, endpoint_password, endpoint_public_url, setup_error, _last_setup_ngrok_url

    # An explicit force (credentials/refresh) must re-provision from scratch;
    # the auto-force below only re-enters setup so the persisted app can be
    # reused with an updated answer_url.
    explicit_force = force

    # Auto-force re-setup when the ngrok/public URL has changed since last init
    current_ngrok_url = get_ngrok_url() or ""
    if endpoint_username and endpoint_password and not force:
        if current_ngrok_url and current_ngrok_url != _last_setup_ngrok_url:
            logger.warning(
                "Ngrok URL changed from %r to %r — forcing Plivo re-setup.",
                _last_setup_ngrok_url,
                current_ngrok_url,
            )
            force = True
        else:
            return {
                "success": True,
                "username": endpoint_username,
                "password": endpoint_password,
                "public_url": endpoint_public_url,
            }

    async with _setup_lock:
        # Re-check inside lock after acquiring it
        current_ngrok_url = get_ngrok_url() or ""
        if endpoint_username and endpoint_password and not force:
            if current_ngrok_url and current_ngrok_url != _last_setup_ngrok_url:
                force = True
            else:
                return {
                    "success": True,
                    "username": endpoint_username,
                    "password": endpoint_password,
                    "public_url": endpoint_public_url,
                }

        ngrok_url = current_ngrok_url
        endpoint_public_url = ngrok_url or ""
        if not ngrok_url:
            endpoint_username = ""
            endpoint_password = ""
            setup_error = (
                "Plivo softphone is not configured: set PUBLIC_URL or NGROK_URL, "
                "or run ngrok so Plivo can reach /api/plivo/dial."
            )
            logger.error(setup_error)
            return {"success": False, "error": setup_error, "code": "plivo_public_url_missing"}
        
        if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
            endpoint_username = ""
            endpoint_password = ""
            setup_error = "Plivo softphone is not configured: PLIVO_AUTH_ID or PLIVO_AUTH_TOKEN is missing."
            logger.error(setup_error)
            return {"success": False, "error": setup_error, "code": "plivo_credentials_missing"}
            
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        answer_url = f"{ngrok_url}/api/plivo/dial"

        # Reuse the previously provisioned app + endpoint when possible so a
        # backend restart (or URL change) doesn't pay two Plivo REST calls and
        # doesn't orphan apps/endpoints in the Plivo account.
        #
        # Postgres first, on-disk JSON only as a fallback: the file is gitignored
        # and Azure wipes the filesystem on every deploy, so on hosted it was
        # always absent and every deploy minted a fresh Application while
        # endpoints stayed bound to older ones with dead tunnel URLs.
        persisted = None
        if not explicit_force:
            persisted = await asyncio.to_thread(_load_app_state, "softphone")
            if persisted and not (persisted.get("username") and persisted.get("password")):
                # Application is known but its shared fallback endpoint is not
                # (an adopted row, or a failure between the two create calls).
                # Reuse the Application and create only the missing endpoint —
                # discarding it here would orphan a perfectly good app and mint
                # another, which is the leak this whole change exists to stop.
                try:
                    client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
                    new_username = f"user{uuid.uuid4().hex[:20]}"
                    new_password = f"Hy{uuid.uuid4().hex[:16]}!"
                    resp = await asyncio.to_thread(
                        client.endpoints.create,
                        username=new_username, password=new_password,
                        alias=f"shared_{_env_key()[:40]}", app_id=persisted["app_id"],
                    )
                    persisted["username"] = getattr(resp, "username", new_username)
                    persisted["password"] = new_password
                    await asyncio.to_thread(
                        _save_app_state, "softphone", persisted["app_id"],
                        answer_url, persisted["username"], persisted["password"],
                    )
                    logger.info("Created shared endpoint on existing app %s", persisted["app_id"])
                except Exception as exc:
                    logger.warning("Could not attach an endpoint to app %s: %s",
                                   persisted.get("app_id"), exc)
                    persisted = None
            if not persisted:
                # Adopt whatever the local file knows so a developer's existing
                # app is carried into the table instead of leaking another one.
                legacy = _load_persisted_softphone_state()
                if legacy:
                    await asyncio.to_thread(
                        _save_app_state, "softphone", legacy["app_id"],
                        legacy["answer_url"], legacy["username"], legacy["password"],
                    )
                    persisted = legacy
        if persisted:
            try:
                if persisted["answer_url"] != answer_url:
                    await asyncio.to_thread(
                        client.applications.update,
                        app_id=persisted["app_id"],
                        answer_url=answer_url,
                        answer_method="POST",
                    )
                    _persist_softphone_state(persisted["app_id"], persisted["username"], persisted["password"], answer_url)
                    await asyncio.to_thread(
                        _save_app_state, "softphone", persisted["app_id"], answer_url,
                        persisted["username"], persisted["password"],
                    )
                    logger.info(f"Updated answer_url on existing Plivo app {persisted['app_id']}")
                endpoint_username = persisted["username"]
                endpoint_password = persisted["password"]
                _last_setup_ngrok_url = ngrok_url
                setup_error = ""
                logger.info(f"Reusing persisted Plivo endpoint: {endpoint_username}")
                return {
                    "success": True,
                    "username": endpoint_username,
                    "password": endpoint_password,
                    "public_url": endpoint_public_url,
                }
            except Exception as reuse_err:
                logger.warning(f"Could not reuse persisted Plivo app, provisioning fresh: {reuse_err}")
                # Drop the unusable record, or the fresh-provision path below
                # reads it straight back and returns the same dead app.
                await asyncio.to_thread(_clear_app_state, "softphone")
                persisted = None

        setup_suffix = f"{time.time_ns()}{uuid.uuid4().hex[:8]}"
        app_name = f"Softphone_App_{setup_suffix}"

        try:
            def _create_softphone():
                app_response = client.applications.create(
                    app_name=app_name,
                    answer_url=answer_url,
                    answer_method="POST",
                )
                new_app_id = app_response.app_id
                logger.info("Created Plivo App: %s", new_app_id)

                new_username = f"user{uuid.uuid4().hex[:20]}"
                new_password = f"Hy{uuid.uuid4().hex[:16]}!"
                endpoint_response = client.endpoints.create(
                    username=new_username,
                    password=new_password,
                    alias=app_name,
                    app_id=new_app_id,
                )
                return {
                    "app_id": new_app_id,
                    # Plivo appends its own digits, so the endpoint that exists
                    # is not the name we asked for.
                    "username": getattr(endpoint_response, "username", new_username),
                    "password": new_password,
                }

            # Serialised across workers via a Postgres advisory lock, so a
            # 4-worker restart creates one Application, not four. Falls back to
            # creating directly when the DB is unreachable.
            created = await asyncio.to_thread(
                _provision_app_once, "softphone", answer_url, _create_softphone,
            )
            if not created:
                created = await asyncio.to_thread(_create_softphone)

            app_id = created["app_id"]
            endpoint_username = created["username"]
            endpoint_password = created["password"]
            _last_setup_ngrok_url = ngrok_url
            setup_error = ""
            _persist_softphone_state(app_id, endpoint_username, endpoint_password, answer_url)
            logger.info(f"Stored Credentials: {endpoint_username} / {endpoint_password}")
            return {
                "success": True,
                "username": endpoint_username,
                "password": endpoint_password,
                "public_url": endpoint_public_url,
            }
            
        except Exception as e:
            endpoint_username = ""
            endpoint_password = ""
            _last_setup_ngrok_url = ""
            setup_error = f"Failed to setup Plivo softphone components: {e}"
            logger.error(setup_error)
            return {"success": False, "error": setup_error, "code": "plivo_setup_failed"}

_endpoint_provision_lock = asyncio.Lock()

# Only ONE recruiter may hold the shared fallback endpoint at a time. Two
# browsers registering the same SIP username makes call attribution ambiguous
# (and the busy flag, which is keyed on username, marks them both at once), so
# the second user is refused rather than silently corrupting call records.
_shared_endpoint_lock = threading.Lock()
_shared_endpoint_holder = {"user_id": None, "at": 0.0}
# Claims are refreshed every time the browser fetches credentials (softphone
# init and every re-login), so an expiry this long only ever releases a holder
# who has genuinely stopped using the app — without it, one recruiter taking
# the fallback would lock everyone else out of it until the next restart.
SHARED_ENDPOINT_CLAIM_TTL_SECONDS = 30 * 60


def claim_shared_endpoint(user_id: int) -> bool:
    """Take (or renew) the shared-endpoint claim. False if someone else holds it."""
    if not user_id:
        return False
    with _shared_endpoint_lock:
        holder = _shared_endpoint_holder["user_id"]
        expired = (time.time() - _shared_endpoint_holder["at"]) > SHARED_ENDPOINT_CLAIM_TTL_SECONDS
        if holder is not None and holder != user_id and not expired:
            return False
        _shared_endpoint_holder["user_id"] = user_id
        _shared_endpoint_holder["at"] = time.time()
        return True


def release_shared_endpoint(user_id: int) -> None:
    """Give the claim back once this user has an endpoint of their own."""
    with _shared_endpoint_lock:
        if _shared_endpoint_holder["user_id"] == user_id:
            _shared_endpoint_holder["user_id"] = None
            _shared_endpoint_holder["at"] = 0.0


def get_shared_endpoint_holder():
    with _shared_endpoint_lock:
        return _shared_endpoint_holder["user_id"]


def _persist_endpoint_row(user_id: int, endpoint_id, username: str, password: str, app_id) -> bool:
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO plivo_endpoints (user_id, endpoint_id, username, password, app_id, env_key)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, env_key) DO UPDATE
                SET endpoint_id = EXCLUDED.endpoint_id,
                    username = EXCLUDED.username,
                    password = EXCLUDED.password,
                    app_id = EXCLUDED.app_id
                """,
                (user_id, str(endpoint_id) if endpoint_id else None, username, password,
                 app_id, _env_key()),
            )
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to persist Plivo endpoint row for user %s: %s", user_id, exc)
        return False
    finally:
        return_db_connection(conn)


async def _rebind_if_stale(user_id: int, row: dict) -> dict:
    """Re-point an endpoint at this environment's current Application.

    This is the fix for the outage: an endpoint carries an ``app_id``, and the
    Application is what holds the answer URL. A row created against an older
    Application keeps dialling through that Application's URL — which, for a row
    created on a developer's laptop, was a long-dead ngrok tunnel. Plivo then
    could not fetch call XML and dropped every call with
    ``Error Reaching Answer URL`` before the backend saw anything.

    Never fails the caller: if re-binding does not work, the existing (possibly
    stale) endpoint is still returned, because a stale endpoint is no worse than
    what we had and refusing would block dialling outright.
    """
    current = await asyncio.to_thread(_load_app_state, "softphone")
    current_app_id = (current or {}).get("app_id")
    if not current_app_id or not row.get("endpoint_id"):
        return row
    if str(row.get("app_id") or "") == str(current_app_id):
        return row

    logger.warning(
        "Endpoint %s for user %s is bound to app %s but this environment now "
        "uses app %s — re-binding.",
        row.get("username"), user_id, row.get("app_id"), current_app_id,
    )
    try:
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        await asyncio.to_thread(
            client.endpoints.update,
            endpoint_id=row["endpoint_id"], app_id=current_app_id,
        )
    except Exception as exc:
        logger.error("Could not re-bind endpoint for user %s: %s", user_id, exc)
        return row

    await asyncio.to_thread(
        _persist_endpoint_row, user_id, row["endpoint_id"],
        row["username"], row["password"], current_app_id,
    )
    row["app_id"] = current_app_id
    return row


async def _delete_endpoint_quietly(endpoint_id, user_id: int) -> None:
    """Undo a creation whose registry row could not be written.

    Best effort: if the delete also fails the endpoint is orphaned, which is
    what adoption exists to clean up on the next login.
    """
    if not endpoint_id or not (PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN):
        return
    try:
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        await asyncio.to_thread(client.endpoints.delete, endpoint_id=str(endpoint_id))
        logger.info("Rolled back unpersisted Plivo endpoint %s for user %s", endpoint_id, user_id)
    except Exception as exc:
        logger.warning(
            "Could not roll back unpersisted Plivo endpoint %s for user %s: %s",
            endpoint_id, user_id, exc,
        )


async def _adopt_orphaned_endpoint(user_id: int) -> Optional[dict]:
    """Reclaim an endpoint that exists in Plivo but not in our registry.

    Endpoints are created in Plivo first and persisted second, so any failure
    between the two leaves a live endpoint owned by this user that we have no
    record of. Without adoption every subsequent login mints another one.

    Plivo never returns an existing endpoint's password, so adoption resets it
    to a fresh value we can store — the browser re-registers with the new
    credentials on its next fetch either way.
    """
    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
        return None

    # Environment-scoped, and it must stay that way. A bare "recruiter_<id>"
    # alias made a second environment adopt the FIRST environment's live
    # endpoint and reset its password — which silently locked the original out
    # of SIP registration, so its softphone could not register, outbound
    # reported "Not Reachable" and every inbound call went straight to
    # voicemail with nothing to ring. Never match the unscoped legacy form.
    #
    # The pre-slug alias is accepted too, so a laptop that provisioned as
    # "recruiter_4_local" reclaims that endpoint instead of minting a second.
    # On hosted deployments the pre-slug form was too long for Plivo to ever
    # create, so there is nothing of that shape to find.
    aliases = {endpoint_alias_for_user(user_id), f"recruiter_{user_id}_{_env_key()}"}
    try:
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)

        def _find_by_alias():
            # Paginate: the account holds hundreds of endpoints, and a listing
            # capped at the first 20 found nothing, so adoption silently never
            # fired and every failed persist orphaned another endpoint.
            offset = 0
            while offset < _ADOPT_SCAN_LIMIT:
                page = client.endpoints.list(limit=20, offset=offset)
                items = list(getattr(page, "objects", None) or page or [])
                if not items:
                    return None
                for item in items:
                    if str(getattr(item, "alias", "") or "") in aliases:
                        return item
                offset += 20
            return None

        match = await asyncio.to_thread(_find_by_alias)
        if not match:
            return None

        endpoint_id = getattr(match, "endpoint_id", None) or getattr(match, "id", None)
        username = getattr(match, "username", None)
        if not endpoint_id or not username:
            return None

        password = f"Hy{uuid.uuid4().hex[:16]}!"
        await asyncio.to_thread(client.endpoints.update, endpoint_id=endpoint_id, password=password)
    except Exception as exc:
        logger.warning("Could not adopt an orphaned Plivo endpoint for user %s: %s", user_id, exc)
        return None

    app_id = (_load_persisted_softphone_state() or {}).get("app_id")
    if not await asyncio.to_thread(_persist_endpoint_row, user_id, endpoint_id, username, password, app_id):
        return None

    logger.info("Adopted orphaned Plivo endpoint %s for user %s", username, user_id)
    return {"username": username, "password": password,
            "endpoint_id": str(endpoint_id), "app_id": app_id}


async def ensure_endpoint_for_user(user_id: int) -> Optional[dict]:
    """Return this user's own SIP endpoint, creating it on first use.

    Inbound "ring everyone" dials one <User> per recruiter, so each recruiter
    needs a distinct endpoint: a single shared endpoint cannot be forked to, and
    gives no way to record who picked up. Endpoints reuse the softphone
    Application (it only supplies the outbound answer_url), so this adds an
    endpoint per user, not an app per user.

    Returns None on failure so callers can fall back to the shared endpoint
    rather than losing outbound dialling.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    if not user_id:
        return None

    def _read():
        """Returns (ok, row). `ok=False` means we could not look at all.

        The two must stay distinguishable: returning None for both "no row" and
        "database unreachable" made a DB outage look like a first-time user, so
        every login created a brand new Plivo endpoint, failed to persist it,
        and orphaned it in the account.
        """
        conn = get_db_connection(validate=False, register_pgvector=False)
        if not conn:
            return False, None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT username, password, endpoint_id, app_id FROM plivo_endpoints "
                    "WHERE user_id = %s AND env_key = %s",
                    (user_id, _env_key()),
                )
                return True, cur.fetchone()
        except Exception as exc:
            logger.error("Could not read plivo_endpoints for user %s: %s", user_id, exc)
            return False, None
        finally:
            return_db_connection(conn)

    ok, existing = await asyncio.to_thread(_read)
    if not ok:
        # Abort before touching the Plivo API: creating an endpoint we cannot
        # persist leaks a live endpoint into the account on every single login
        # for as long as the database is down.
        logger.error(
            "Skipping Plivo endpoint provisioning for user %s — the endpoint "
            "registry is unreadable, so a new endpoint could not be persisted.",
            user_id,
        )
        return None
    if existing:
        return await _rebind_if_stale(user_id, {
            "username": existing[0], "password": existing[1],
            "endpoint_id": existing[2], "app_id": existing[3],
        })

    # Provisioning touches the Plivo API; serialise so two concurrent logins by
    # the same user cannot create two endpoints.
    async with _endpoint_provision_lock:
        ok, existing = await asyncio.to_thread(_read)
        if not ok:
            return None
        if existing:
            return await _rebind_if_stale(user_id, {
                "username": existing[0], "password": existing[1],
                "endpoint_id": existing[2], "app_id": existing[3],
            })

        # Adopt an endpoint this user already owns in Plivo but which is missing
        # from the registry — otherwise a past failed write means we mint a
        # second one now, and another on the next login.
        adopted = await _adopt_orphaned_endpoint(user_id)
        if adopted:
            return adopted

        setup = await setup_plivo()
        if not setup.get("success"):
            return None
        app_id = _load_persisted_softphone_state().get("app_id") if _load_persisted_softphone_state() else None
        if not app_id:
            return None

        try:
            client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
            username = f"user{uuid.uuid4().hex[:20]}"
            password = f"Hy{uuid.uuid4().hex[:16]}!"
            resp = await asyncio.to_thread(
                client.endpoints.create,
                username=username,
                password=password,
                alias=endpoint_alias_for_user(user_id),
                app_id=app_id,
            )
            endpoint_id = getattr(resp, "endpoint_id", None) or getattr(resp, "id", None)
            # Plivo appends its own digits to the requested username, so the
            # endpoint that actually exists is NOT the name we asked for.
            # Storing the requested name makes SIP registration fail with
            # "Authentication Error" — always take the username Plivo returns.
            username = getattr(resp, "username", None) or username
        except Exception as exc:
            logger.error("Failed to create Plivo endpoint for user %s: %s", user_id, exc)
            return None

        # One writer for both the create and adopt paths. The inline INSERT this
        # replaces named ON CONFLICT (user_id) — a constraint the table does not
        # have, since endpoints are unique per (user_id, env_key) — so it raised
        # on every call and no endpoint created here was ever stored. It also
        # left env_key at its 'legacy' default, which the reader below and
        # mark_endpoint_registered would never match again.
        stored = await asyncio.to_thread(
            _persist_endpoint_row, user_id, endpoint_id, username, password, app_id,
        )
        if not stored:
            # An endpoint we cannot store is unreachable and permanent: nothing
            # points at it, and the next login mints another. Hand it back to
            # Plivo rather than leaving it in the account forever.
            await _delete_endpoint_quietly(endpoint_id, user_id)
            return None
        logger.info("Provisioned Plivo endpoint %s for user %s", username, user_id)
        return {"username": username, "password": password,
                "endpoint_id": str(endpoint_id) if endpoint_id else None, "app_id": app_id}


def mark_endpoint_registered(user_id: int) -> None:
    """Record that this user's softphone is live, so inbound only rings plausibly
    online endpoints. Stale rows simply do not answer — the <Dial timeout> covers it."""
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE plivo_endpoints SET last_registered_at = CURRENT_TIMESTAMP "
                "WHERE user_id = %s AND env_key = %s",
                (user_id, _env_key()),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not mark endpoint registered for user %s: %s", user_id, exc)
    finally:
        return_db_connection(conn)


# A busy flag is only ever cleared by a signal from the recruiter's browser (or
# the next dial), so a crashed tab could strand an endpoint as permanently
# busy — i.e. silently excluded from every inbound call, the worst failure mode
# here. Anything older than this is treated as not busy; the browser re-asserts
# busy on its own state changes, so a genuinely long call is not cut short from
# ringing (it just becomes eligible again, and Plivo/the banner guard handle it).
BUSY_STALE_SECONDS = 2 * 60 * 60


def _set_endpoint_busy(busy: bool, *, user_id: int = None, username: str = None) -> None:
    """Mark/clear 'this recruiter is on a call' so inbound ring-all can skip them."""
    from backend.db.connection import get_db_connection, return_db_connection

    if not user_id and not username:
        return

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        value = "CURRENT_TIMESTAMP" if busy else "NULL"
        column, key = ("user_id", user_id) if user_id else ("username", username)
        with conn.cursor() as cur:
            # Scoped to this environment: a user can hold an endpoint row per
            # environment, and marking a laptop's row busy must not remove the
            # hosted endpoint from the inbound ring-all.
            cur.execute(
                f"UPDATE plivo_endpoints SET in_call_since = {value} "
                f"WHERE {column} = %s AND env_key = %s",
                (key, _env_key()),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.warning("Could not set busy=%s for %s=%s: %s", busy, column, key, exc)
    finally:
        return_db_connection(conn)


def mark_endpoint_busy(user_id: int = None, username: str = None) -> None:
    _set_endpoint_busy(True, user_id=user_id, username=username)


def clear_endpoint_busy(user_id: int = None, username: str = None) -> None:
    _set_endpoint_busy(False, user_id=user_id, username=username)


def get_registered_endpoint_usernames(within_seconds: int = 900) -> list:
    """Usernames to ring on an inbound call: endpoints registered recently and
    not already on a call.

    Ringing a recruiter who is mid-conversation is worse than not ringing them:
    it rings over a live candidate call, and if they are the only one online the
    new caller burns the full <Dial timeout> before reaching voicemail instead
    of getting it straight away.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT username FROM plivo_endpoints
                WHERE last_registered_at IS NOT NULL
                  AND last_registered_at > CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                  AND (in_call_since IS NULL
                       OR in_call_since < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second'))
                  -- This deployment only. The registry is shared with every
                  -- developer laptop, and those endpoints belong to a different
                  -- Plivo Application: ringing one cannot connect, and it burns
                  -- the <Dial timeout> before the caller reaches voicemail.
                  AND env_key = %s
                ORDER BY last_registered_at DESC
                """,
                (within_seconds, BUSY_STALE_SECONDS, _env_key()),
            )
            usernames = [r[0] for r in cur.fetchall()]
            # An inbound call that rings nobody is otherwise indistinguishable
            # from one that rang everybody and was ignored.
            logger.info(
                "Inbound ring list for env %s: %d endpoint(s) %s",
                _env_key(), len(usernames), usernames,
            )
            return usernames
    except Exception as exc:
        logger.warning("Could not list registered endpoints: %s", exc)
        return []
    finally:
        return_db_connection(conn)


def initiate_call(candidate_phone: str, recruiter_email: str, candidate_name: str, candidate_id: str, transaction_id: str):
    logger.info(f"Initiating outbound Plivo call to {candidate_phone} for recruiter {recruiter_email}")
    
    if not PLIVO_AUTH_ID or not PLIVO_AUTH_TOKEN:
        logger.error("Plivo credentials missing, returning dummy success")
        return {"success": True, "call_uuid": "dummy-uuid"}

    client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
    ngrok_url = get_ngrok_url()
    
    try:
        # Plivo dials candidate via Answer URL
        answer_url = f"{ngrok_url}/api/plivo/dial"
        # We can trigger an outbound call directly using the REST API
        # To make it easy, we dial the candidate phone directly from the plivo number
        response = client.calls.create(
            from_=PLIVO_NUMBER,
            to_=normalize_number(candidate_phone),
            answer_url=answer_url,
            answer_method="POST"
        )
        logger.info(f"Plivo Call creation response: {response}")
        return {"success": True, "call_uuid": response.request_uuid}
    except Exception as e:
        logger.error(f"Failed to initiate Plivo call: {e}")
        return {"success": False, "error": str(e)}

# Plivo's real AMD only works on calls placed via the REST API (calls.create with
# machine_detection); this app's live flow bridges via XML <Dial> from a browser-SDK
# answer_url, which Plivo does not support AMD on. This heuristic is a best-effort
# suggestion from the post-call transcript, not a live detection signal — the
# recruiter still confirms/overrides the outcome.
VOICEMAIL_PHRASES = (
    "leave a message", "leave your message", "leave a detailed message",
    "after the tone", "after the beep", "record your message",
    "voice mailbox", "voicemail box", "reached the voicemail",
    "is not available", "cannot take your call", "unable to take your call",
    "mailbox is full", "no one is available to take your call",
)


def detect_likely_voicemail(transcript_text: Optional[str], duration_seconds: Optional[int]) -> bool:
    if transcript_text:
        lowered = transcript_text.lower()
        if any(phrase in lowered for phrase in VOICEMAIL_PHRASES):
            return True
        # A real conversation has "Lead:" turns; a voicemail greeting is one
        # uninterrupted block from whichever speaker label got assigned to it.
        has_two_sided_exchange = lowered.count("lead:") >= 1 and lowered.count("recruiter:") >= 1
        if duration_seconds is not None and duration_seconds < 25 and not has_two_sided_exchange:
            return True
    elif duration_seconds is not None and duration_seconds < 8:
        # No transcript at all plus a very short call is more consistent with an
        # unanswered/voicemail pickup than a genuine conversation.
        return True
    return False


# gpt-4o-transcribe has a hard 1500s (25 min) cap where it errors cleanly
# (caught below, falls back to whisper-1) — but multiple OpenAI developer-
# community reports document it silently truncating output well before that,
# commonly around the 10-11 minute mark, WITHOUT raising an exception. A
# silent truncation can't be caught by try/except, so for any call whose
# duration we already know exceeds this safe margin, skip gpt-4o-transcribe
# entirely and go straight to whisper-1 rather than gambling on it.
GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS = 480  # 8 minutes


# ── Keeping the transcript honest ────────────────────────────────────────────
# What the microphone captured is the record. An LLM may re-emit it with
# speaker labels, but it may not add to it: asked to "analyze a VoIP call" with
# four words of audio, gpt-4o-mini writes the recruiting conversation it
# expects to see, and that lands on a real candidate's file as fact.

# Below this many words there is nothing to interpret, so we never ask.
MIN_WORDS_FOR_ANALYSIS = 12
# How far the labelled rewrite may drift from the words actually spoken.
TRANSCRIPT_MIN_RETENTION = 0.7
TRANSCRIPT_MAX_EXPANSION = 1.3

# Unicode-aware, and wide enough for a real name: the label is now the
# candidate's own name, so an ASCII-only class left "Alice Rachael Mendonca:"
# (with a cedilla) unstripped, counted the name as spoken words, and could fail
# transcript_is_faithful — throwing away a correctly labelled transcript.
_SPEAKER_LABEL_RE = re.compile(
    r"^\s*[^\W\d_][^\n:]{0,40}:\s*", re.MULTILINE | re.UNICODE
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _transcript_words(text: str) -> list:
    """Spoken words only — speaker labels are ours, not the candidate's."""
    return _WORD_RE.findall(_SPEAKER_LABEL_RE.sub("", text or "").lower())


def transcript_is_faithful(raw_text: str, labelled_text: str) -> bool:
    """Does the speaker-labelled rewrite still say what the recording said?

    Guards both failure modes: dropping the second half of a long call, and
    inventing a conversation over near-silence. The comparison is on words
    rather than characters so the "Recruiter:" / "Lead:" prefixes the rewrite
    adds do not read as new content.
    """
    raw_words = _transcript_words(raw_text)
    labelled_words = _transcript_words(labelled_text)
    if not raw_words or not labelled_words:
        return False
    if len(labelled_words) > len(raw_words) * TRANSCRIPT_MAX_EXPANSION:
        return False          # invented material
    if len(labelled_words) < len(raw_words) * TRANSCRIPT_MIN_RETENTION:
        return False          # abridged away
    # Length alone is not enough: the words themselves have to be the spoken
    # ones, not a same-length paraphrase.
    raw_counts = Counter(raw_words)
    labelled_counts = Counter(labelled_words)
    kept = sum(min(count, labelled_counts[word]) for word, count in raw_counts.items())
    return kept >= len(raw_words) * TRANSCRIPT_MIN_RETENTION


def too_little_speech_to_analyse(raw_text: str) -> bool:
    return len(_transcript_words(raw_text)) < MIN_WORDS_FOR_ANALYSIS


def no_conversation_summary(duration_seconds) -> str:
    """Said plainly, so the recruiter knows the call yielded nothing.

    Deliberately avoids the "not enough information" phrasings the UI treats as
    a placeholder to keep polling for — this is a final answer, not a retry.
    """
    if duration_seconds:
        return (
            f"No conversation was captured. The {int(duration_seconds)}-second "
            "recording contains almost no speech."
        )
    return "No conversation was captured. The recording contains almost no speech."


# --- Dual-channel (per-leg) transcription -----------------------------------
#
# Plivo records each leg of a call on its own channel: channel 0 is the
# recruiter, channel 1 is the person they called. Downmixing that to mono threw
# away the only reliable evidence of who spoke, and left an LLM to reconstruct
# speaker turns from the words alone. It guessed, and it got it wrong in both
# directions — whole calls came back with the candidate's answers attributed to
# the recruiter.
#
# Transcribing each channel separately and interleaving by timestamp makes
# attribution exact rather than inferred. Inbound voicemail captures are
# genuinely mono (one caller, no second leg); those return None here and fall
# back to the single-file path, which is correct — there is no second speaker
# to find.

CHANNEL_SPEAKERS = ("Recruiter", "Candidate")


def _decode_channels(audio_path: str) -> Optional[list]:
    """De-interleaved int16 PCM per channel at 16 kHz, or None if not stereo."""
    try:
        import av
        import numpy as np
    except ImportError:
        logger.warning("PyAV/numpy unavailable — per-channel transcription disabled")
        return None

    try:
        with av.open(audio_path) as container:
            stream = container.streams.audio[0]
            if (stream.codec_context.channels or 1) < 2:
                return None
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="stereo", rate=16000,
            )
            chunks = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray())
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=1)
        # s16 stereo decodes to a single interleaved plane.
        if data.shape[0] == 1:
            data = data.reshape(-1, 2).T
        if data.shape[0] < 2:
            return None
        return [data[0], data[1]]
    except Exception as exc:
        logger.warning("Could not split channels for %s: %s", audio_path, exc)
        return None


def _write_mono_audio(samples, path: str, rate: int = 16000) -> None:
    """One call leg as a compact mono MP3.

    Deliberately not WAV: a single leg of a 15-minute call is 28.7 MB of raw
    PCM, and OpenAI rejects anything over 25 MB — the first run of this came
    back "413: Maximum content size limit exceeded" and silently fell back to
    the mono guesswork it was meant to replace. At 32 kbps the same leg is
    3.6 MB, which keeps roughly a hundred minutes of call inside the limit.
    """
    import av
    import numpy as np
    from fractions import Fraction

    with av.open(path, "w") as out:
        stream = out.add_stream("mp3", rate=rate)
        stream.bit_rate = 32000
        block = 1152 * 16          # whole MP3 frames, so nothing is padded
        pts = 0
        for start in range(0, len(samples), block):
            chunk = np.ascontiguousarray(samples[start:start + block]).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            frame.rate = rate
            frame.pts = pts
            frame.time_base = Fraction(1, rate)
            pts += chunk.shape[1]
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


async def _transcribe_per_channel(
    client, audio_path: str, candidate_label: str,
) -> Optional[str]:
    """Speaker-labelled transcript built from the two call legs, or None."""
    channels = await asyncio.to_thread(_decode_channels, audio_path)
    if not channels:
        return None

    speakers = ("Recruiter", candidate_label)
    segments = []
    tmp_paths = []
    try:
        for index, samples in enumerate(channels):
            path = f"{audio_path}.ch{index}.mp3"
            tmp_paths.append(path)
            await asyncio.to_thread(_write_mono_audio, samples, path)
            with open(path, "rb") as handle:
                # verbose_json is what carries the per-segment timing that the
                # interleave below depends on; the default response has none.
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=handle,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            for seg in (getattr(result, "segments", None) or []):
                text = (getattr(seg, "text", "") or "").strip()
                if text:
                    segments.append((float(getattr(seg, "start", 0.0)), index, text))
    except Exception as exc:
        logger.warning("Per-channel transcription failed for %s: %s", audio_path, exc)
        return None
    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    if not segments:
        return None

    segments.sort(key=lambda item: item[0])
    lines = []
    for _start, index, text in segments:
        speaker = speakers[index]
        if lines and lines[-1][0] == speaker:
            lines[-1][1] = f"{lines[-1][1]} {text}".strip()
        else:
            lines.append([speaker, text])
    return "\n".join(f"{speaker}: {text}" for speaker, text in lines)


def _candidate_name_for_call(call_uuid: str) -> Optional[str]:
    """The candidate on the other end of this call, for speaker labelling.

    Looked up here rather than passed in by each caller: insights are kicked off
    from the Plivo webhook, the manual sync endpoint and the insights route, and
    threading a name through all three would leave whichever one was missed
    silently emitting the generic "Lead" label again.
    """
    if not call_uuid:
        return None
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cand.name
                FROM calls c
                JOIN candidates cand ON cand.id = c.candidate_id
                WHERE c.plivo_call_uuid = %s OR c.plivo_transaction_id = %s
                ORDER BY c.created_at DESC
                LIMIT 1
                """,
                (call_uuid, call_uuid),
            )
            row = cur.fetchone()
            return (row[0] or "").strip() or None if row else None
    except Exception as exc:
        logger.warning("Could not resolve candidate name for %s: %s", call_uuid, exc)
        return None
    finally:
        return_db_connection(conn)


async def process_call_insights(call_uuid: str, record_url: str, initial_delay_seconds: int = 0, duration_seconds: int = None):
    try:
        existing_insights = call_insights.get(call_uuid)
        if existing_insights and not existing_insights.get("error"):
            logger.info(f"Skipping already generated insights for {call_uuid}")
            return

        if initial_delay_seconds > 0:
            logger.info(f"Waiting {initial_delay_seconds}s before downloading recording for insights: {call_uuid}")
            await asyncio.sleep(initial_delay_seconds)

        logger.info(f"Downloading recording for insights: {call_uuid}")

        max_retries = 6
        response = None
        for attempt in range(max_retries):
            # 90s (not 30s) — a longer call's recording is a bigger file, and a
            # timeout too tight for the transfer used to look identical to "not
            # ready yet" (both just retry), silently eating into the budget.
            response = await asyncio.to_thread(download_plivo_recording, record_url, 90)
            if response.status_code == 200:
                break
            logger.warning(f"Download failed with {response.status_code}. Retrying in 5s...")
            await asyncio.sleep(5)

        if not response or response.status_code != 200:
            logger.error(f"Failed to download recording after retries.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        logger.info("Transcribing audio...")
        client = get_openai_client()
        # Only a KNOWN-short call may use gpt-4o-transcribe. The manual
        # sync-recording path passes duration 0 for calls the recruiter hasn't
        # saved yet — a 23-minute call arriving as duration=0 used to slip past
        # this guard and get silently cut around the 10-minute mark.
        known_short = (
            duration_seconds is not None
            and 0 < duration_seconds <= GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS
        )
        if not known_short:
            logger.info(
                f"Call {call_uuid} duration is {duration_seconds}s (unknown or over the "
                f"{GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS}s safe margin) — using whisper-1 directly "
                "instead of risking gpt-4o-transcribe's silent truncation."
            )
            with open(temp_audio_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    timeout=300.0
                )
        else:
            try:
                # gpt-4o-transcribe eliminates whisper-1's well-documented repetition/
                # hallucination loops on quiet or long audio (e.g. transcripts that
                # devolve into the same sentence repeated dozens of times, or drift
                # into gibberish/other scripts) — worth using when duration is
                # unknown or safely short; see GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS
                # above for why known-longer calls skip it entirely instead.
                with open(temp_audio_path, "rb") as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file,
                        response_format="json",
                        timeout=300.0
                    )
            except Exception as e:
                logger.warning(f"gpt-4o-transcribe failed for {call_uuid} ({e}); falling back to whisper-1")
                with open(temp_audio_path, "rb") as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        timeout=300.0
                    )

        raw_text = transcript.text

        # Who spoke is decided by the audio, not by a model reading the words.
        candidate_name = await asyncio.to_thread(_candidate_name_for_call, call_uuid)
        channel_transcript = await _transcribe_per_channel(
            client, temp_audio_path, (candidate_name or "").strip() or "Candidate",
        )
        if channel_transcript:
            logger.info("Speaker labels for %s taken from the two call legs", call_uuid)

        os.remove(temp_audio_path)

        # Nothing was said, so there is nothing to analyse. Asking anyway is
        # how a 9-second "Thank you very much." became a five-turn discussion
        # of the candidate's open-source contributions, complete with a quote.
        if too_little_speech_to_analyse(raw_text):
            logger.info(
                f"Call {call_uuid} has {len(_transcript_words(raw_text))} spoken word(s) — "
                "storing the transcription as-is without asking for insights."
            )
            result_json = {
                "transcript": channel_transcript or raw_text,
                "summary": no_conversation_summary(duration_seconds),
                "insights": [],
                "sentiment": None,
                "sentiment_reason": None,
            }
        else:
            logger.info("Generating insights...")
            result_json = await _generate_call_insights(
                client, channel_transcript or raw_text, candidate_name=candidate_name,
            )

        call_insights[call_uuid] = result_json
        await _store_call_insights(
            call_uuid, record_url, raw_text, result_json, duration_seconds,
            channel_transcript=channel_transcript,
        )
        return
    except Exception as e:
        logger.error(f"Error processing call insights for {call_uuid}: {e}")
        call_insights[call_uuid] = {"error": str(e)}


async def _generate_call_insights(
    client, raw_text: str, candidate_name: Optional[str] = None,
) -> dict:
    """Split the raw transcription by speaker and analyse it.

    The model is told who the candidate is. Without that it could only emit the
    generic "Lead" label, which every downstream surface then had to rewrite,
    and it had nothing to anchor attribution on — so on a mono recording with no
    diarization it guessed turn boundaries and put the candidate's words in the
    recruiter's mouth. The names people use on the call ("hi Sanjay, this is
    Jaya") are the strongest evidence available, and the model can only use them
    if it knows which name belongs to whom.
    """
    speaker_label = (candidate_name or "").strip() or "Lead"
    identity = (
        f'The person the recruiter called is named "{speaker_label}".'
        if candidate_name
        else "The name of the person the recruiter called is not known."
    )

    prompt = f"""
        You are an expert AI recruiting assistant analyzing a VoIP call.

        {identity}
        There are exactly two speakers: the recruiter, and {speaker_label}.

        Transcript:
        {raw_text}

        Attributing turns correctly matters more than anything else here. Use
        the evidence in the words themselves:
          - The recruiter introduces themselves, explains why they are calling,
            describes roles, and asks the questions.
          - {speaker_label} answers about their own experience, notice period,
            salary and availability.
          - When someone is addressed by name, the NEXT speaker is usually the
            person just addressed.
        If a turn is genuinely ambiguous, keep it with the previous speaker
        rather than inventing an alternation.

        Use ONLY what appears in the transcript above. Do not add, infer or
        imagine anything that was not said. The "transcript" you return must be
        the same words, split by speaker — never a longer or richer version of
        the conversation. If the transcript is too sparse to work with, return
        it unchanged and say so in the summary.

        Also analyze {speaker_label}'s overall sentiment toward the opportunity
        discussed (not the recruiter's tone) — Positive (engaged/interested),
        Neutral (noncommittal/mixed), or Negative (uninterested/pushback/hostile)
        — with a one-sentence reason grounded in something they actually said.

        Write in plain professional text — no emoji, no markdown formatting.

        Output JSON object. The "speaker" field must be exactly "Recruiter" or
        "{speaker_label}" — no other value, and never a description or role:
        {{
            "transcript": [{{"speaker": "Recruiter" or "{speaker_label}", "text": "What they said"}}],
            "summary": "...",
            "insights": ["..."],
            "sentiment": "Positive" or "Neutral" or "Negative",
            "sentiment_reason": "..."
        }}
        """
        
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    clean = completion.choices[0].message.content.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        return json.loads(match.group(0)) if match else {}


async def _store_call_insights(
    call_uuid, record_url, raw_text, result_json, duration_seconds,
    channel_transcript: Optional[str] = None,
):
    logger.info(f"Successfully generated direct audio insights for {call_uuid}")

    sentiment_raw = str(result_json.get("sentiment") or "").strip().capitalize()
    sentiment = sentiment_raw if sentiment_raw in ("Positive", "Neutral", "Negative") else None
    sentiment_reason = (result_json.get("sentiment_reason") or "").strip() or None

    # Sync back to local SQL database
    try:
        from backend.api.routes.calls import get_calls_db_connection, return_db_connection, invalidate_calls_cache
        conn = get_calls_db_connection()
        if conn:
            cur = conn.cursor()
            t_items = result_json.get("transcript")
            t_str = ""
            if channel_transcript:
                # Built from the separated call legs: the speakers are known,
                # not inferred, so the model's own attribution is discarded.
                t_str = channel_transcript
            elif isinstance(t_items, list):
                t_str = "\n".join([f"{m.get('speaker', 'Unknown')}: {m.get('text', '')}" for m in t_items if isinstance(m, dict)])
            elif isinstance(t_items, str):
                t_str = t_items
            else:
                t_str = str(t_items) if t_items else raw_text

            # The speaker-labelled version comes from gpt-4o-mini RE-EMITTING
            # the conversation, and it can depart from the audio in BOTH
            # directions. On long calls it abridges (23-min calls came back
            # as ~1.2k-char digests full of "..." elisions). On short ones
            # it invents: a 9-second recording whose only words were "Thank
            # you very much." was stored as a five-turn discussion of the
            # candidate's open-source contributions, and the summary quoted
            # a sentence nobody said. The old check only caught shrinkage,
            # so a rewrite twenty times longer than the audio sailed past.
            #
            # Labels are a nicety; the words are the record.
            #
            # The guard is about the LLM re-emitting the conversation. A
            # per-channel transcript is not a rewrite — it is Whisper reading
            # the same audio one leg at a time, and it legitimately differs in
            # wording from the mono pass (each leg is clearer on its own, and
            # cross-talk is heard once per channel). Judging it against the mono
            # text would throw away the only exactly-attributed transcript we
            # have, so it is exempt.
            if not channel_transcript and not transcript_is_faithful(raw_text, t_str):
                logger.warning(
                    f"LLM transcript rewrite for {call_uuid} does not match the "
                    f"audio ({len(t_str)} chars vs {len(raw_text)} raw) — storing "
                    "the raw transcription instead."
                )
                t_str = raw_text

            likely_voicemail = detect_likely_voicemail(t_str, duration_seconds)

            # Authoritative provider duration wins when present; COALESCE
            # keeps any existing value (e.g. the client-side timer) when the
            # webhook didn't carry a usable duration.
            #
            # The transcript is written outright rather than "longest wins".
            # That rule existed so an early run over a not-yet-final recording
            # could not truncate a complete one — but it also made a fabricated
            # transcript permanent, since an invented conversation is always
            # longer than the handful of words actually spoken, and no correct
            # re-run could ever replace it. Only the final callback starts a
            # run now, and claim_insights_run stops concurrent ones racing, so
            # the newest result is the trustworthy one.
            cur.execute("""
                UPDATE calls
                SET
                    recording_url = %s,
                    transcript = %s,
                    summary = %s,
                    sentiment = %s,
                    sentiment_reason = %s,
                    duration = COALESCE(%s, duration),
                    likely_voicemail = %s,
                    status = 'completed',
                    updated_at = NOW()
                WHERE plivo_call_uuid = %s OR plivo_transaction_id = %s
            """, (record_url, t_str, result_json.get("summary"), sentiment, sentiment_reason, duration_seconds, likely_voicemail, call_uuid, call_uuid))
            conn.commit()

            if cur.rowcount == 0:
                logger.info("No matching UUID found in DB, falling back to updating latest call task record")
                cur.execute("""
                    UPDATE calls
                    SET
                        recording_url = %s,
                        transcript = %s,
                        summary = %s,
                        sentiment = %s,
                        sentiment_reason = %s,
                        duration = COALESCE(%s, duration),
                        likely_voicemail = %s,
                        status = 'completed',
                        updated_at = NOW()
                    WHERE id = (SELECT id FROM calls ORDER BY updated_at DESC LIMIT 1)
                """, (record_url, t_str, result_json.get("summary"), sentiment, sentiment_reason, duration_seconds, likely_voicemail))
                conn.commit()
            cur.close()
            return_db_connection(conn)
            # Refresh the in-memory caches so the newly stored duration,
            # transcript and summary surface immediately in the calls list
            # and stats instead of lagging behind by a cache cycle.
            try:
                invalidate_calls_cache()
            except Exception as cache_err:
                logger.warning(f"Cache invalidation after Plivo sync failed: {cache_err}")
            logger.info(f"Database synced for Plivo insights: {call_uuid}")
    except Exception as db_err:
        logger.error(f"DB update failed for Plivo insights: {db_err}")
    

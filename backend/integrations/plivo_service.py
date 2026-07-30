import os
import logging
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
    dark. This makes that ownership explicit instead of implicit in whoever
    restarted most recently.
    """
    from backend.db.connection import get_db_connection, return_db_connection

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT env_key FROM plivo_app_state WHERE kind = 'inbound' "
                "ORDER BY updated_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.warning("Could not determine inbound number owner: %s", exc)
        return None
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
                await asyncio.to_thread(
                    client.applications.update,
                    app_id=app_id, answer_url=answer_url, answer_method="POST",
                    hangup_url=hangup_url, hangup_method="POST",
                )
                logger.info("Re-pointed inbound Plivo app %s to %s", app_id, answer_url)
        else:
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

    alias = f"recruiter_{user_id}"
    try:
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        listing = await asyncio.to_thread(client.endpoints.list, limit=20)
        match = next(
            (e for e in (getattr(listing, "objects", None) or listing or [])
             if str(getattr(e, "alias", "") or "") == alias),
            None,
        )
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
                alias=f"recruiter_{user_id}",
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

        def _write():
            conn = get_db_connection(validate=False, register_pgvector=False)
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO plivo_endpoints (user_id, endpoint_id, username, password, app_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        (user_id, str(endpoint_id) if endpoint_id else None, username, password, app_id),
                    )
                conn.commit()
                return True
            except Exception as exc:
                conn.rollback()
                logger.error("Failed to persist Plivo endpoint for user %s: %s", user_id, exc)
                return False
            finally:
                return_db_connection(conn)

        if not await asyncio.to_thread(_write):
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
                "UPDATE plivo_endpoints SET last_registered_at = CURRENT_TIMESTAMP WHERE user_id = %s",
                (user_id,),
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
            cur.execute(
                f"UPDATE plivo_endpoints SET in_call_since = {value} WHERE {column} = %s",
                (key,),
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
                ORDER BY last_registered_at DESC
                """,
                (within_seconds, BUSY_STALE_SECONDS),
            )
            return [r[0] for r in cur.fetchall()]
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
        skip_gpt4o = duration_seconds is not None and duration_seconds > GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS
        if skip_gpt4o:
            logger.info(
                f"Call {call_uuid} is {duration_seconds}s (over the {GPT4O_TRANSCRIBE_SAFE_MAX_SECONDS}s safe margin) — "
                "using whisper-1 directly instead of risking gpt-4o-transcribe's silent truncation."
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
        os.remove(temp_audio_path)
        
        logger.info("Generating insights...")
        prompt = f"""
        You are an expert AI recruiting assistant analyzing a VoIP call.
        Transcript:
        {raw_text}
        
        Also analyze the Lead's overall sentiment toward the opportunity discussed
        (not the recruiter's tone) — Positive (engaged/interested), Neutral
        (noncommittal/mixed), or Negative (uninterested/pushback/hostile) — with a
        one-sentence reason grounded in something they actually said.

        Write in plain professional text — no emoji, no markdown formatting.

        Output JSON object:
        {{
            "transcript": [{{"speaker": "Recruiter" or "Lead", "text": "What they said"}}],
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
            result_json = json.loads(clean)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            result_json = json.loads(match.group(0)) if match else {}

        call_insights[call_uuid] = result_json
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
                if isinstance(t_items, list):
                    t_str = "\n".join([f"{m.get('speaker', 'Unknown')}: {m.get('text', '')}" for m in t_items if isinstance(m, dict)])
                elif isinstance(t_items, str):
                    t_str = t_items
                else:
                    t_str = str(t_items) if t_items else raw_text

                likely_voicemail = detect_likely_voicemail(t_str, duration_seconds)

                # Authoritative provider duration wins when present; COALESCE
                # keeps any existing value (e.g. the client-side timer) when the
                # webhook didn't carry a usable duration.
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
        
    except Exception as e:
        logger.error(f"Error processing insights for {call_uuid}: {e}")
        call_insights[call_uuid] = {"error": str(e)}

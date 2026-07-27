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


def record_browser_dial(username: str, call_uuid: str, to_number: str):
    global latest_call_uuid
    if not username or not call_uuid:
        return

    normalized_to = normalize_number(to_number)
    state = {
        "call_uuid": call_uuid,
        "username": username,
        "to_number": normalized_to,
        "seen_at": time.time(),
    }
    last_calls[username] = call_uuid
    last_call_states[username] = state
    latest_call_uuid = call_uuid

    try:
        from backend.api.routes.calls import get_calls_db_connection, return_db_connection
        conn = get_calls_db_connection()
        if not conn:
            return
        cur = conn.cursor()
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
        persisted = None if explicit_force else _load_persisted_softphone_state()
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
            app_response = await asyncio.to_thread(
                client.applications.create,
                app_name=app_name,
                answer_url=answer_url,
                answer_method="POST",
            )
            app_id = app_response.app_id
            logger.info(f"Created Plivo App: {app_id}")
            
            username = f"user{uuid.uuid4().hex[:20]}"
            password = "TestPassword123!"
            endpoint_response = await asyncio.to_thread(
                client.endpoints.create,
                username=username,
                password=password,
                alias=app_name,
                app_id=app_id,
            )
            logger.info(f"Created Plivo Endpoint: {endpoint_response}")
            
            endpoint_username = getattr(endpoint_response, 'username', username)
            endpoint_password = password
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

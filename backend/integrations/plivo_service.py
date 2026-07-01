import os
import logging
import requests
import time
import plivo
from openai import AsyncOpenAI
import tempfile
import asyncio
import json
import uuid

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
        response = requests.get('http://127.0.0.1:4040/api/tunnels')
        response.raise_for_status()
        data = response.json()
        for tunnel in data.get('tunnels', []):
            if tunnel.get('proto') == 'https':
                return tunnel.get('public_url')
        if data.get('tunnels'):
            return data['tunnels'][0]['public_url']
    except Exception as e:
        logger.error(f"Failed to get ngrok URL: {e}")
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
        setup_suffix = f"{time.time_ns()}{uuid.uuid4().hex[:8]}"
        app_name = f"Softphone_App_{setup_suffix}"
        answer_url = f"{ngrok_url}/api/plivo/dial"
        
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

async def process_call_insights(call_uuid: str, record_url: str, initial_delay_seconds: int = 0):
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
            response = await asyncio.to_thread(requests.get, record_url, timeout=30)
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
        
        Output JSON object:
        {{
            "transcript": [{{"speaker": "Recruiter" or "Lead", "text": "What they said"}}],
            "summary": "...",
            "insights": ["..."]
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

        # Sync back to local SQL database
        try:
            from backend.api.routes.calls import get_calls_db_connection, return_db_connection
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
                
                cur.execute("""
                    UPDATE calls
                    SET 
                        recording_url = %s,
                        transcript = %s,
                        summary = %s,
                        status = 'completed',
                        updated_at = NOW()
                    WHERE plivo_call_uuid = %s OR plivo_transaction_id = %s
                """, (record_url, t_str, result_json.get("summary"), call_uuid, call_uuid))
                conn.commit()

                if cur.rowcount == 0:
                    logger.info("No matching UUID found in DB, falling back to updating latest call task record")
                    cur.execute("""
                        UPDATE calls
                        SET 
                            recording_url = %s,
                            transcript = %s,
                            summary = %s,
                            status = 'completed',
                            updated_at = NOW()
                        WHERE id = (SELECT id FROM calls ORDER BY updated_at DESC LIMIT 1)
                    """, (record_url, t_str, result_json.get("summary")))
                    conn.commit()
                cur.close()
                return_db_connection(conn)
                logger.info(f"Database synced for Plivo insights: {call_uuid}")
        except Exception as db_err:
            logger.error(f"DB update failed for Plivo insights: {db_err}")
        
    except Exception as e:
        logger.error(f"Error processing insights for {call_uuid}: {e}")
        call_insights[call_uuid] = {"error": str(e)}

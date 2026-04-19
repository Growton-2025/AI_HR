import os
import requests
import tempfile
import threading
from typing import Optional
from openai import OpenAI
from backend.db.connection import get_db_connection_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_call_audio(call_id: int, recording_url: str):
    """
    Background worker to transcribe and summarize call audio.
    """
    if not recording_url:
        return

    # Run in a separate thread to avoid blocking the webhook response
    thread = threading.Thread(target=_process_audio_task, args=(call_id, recording_url))
    thread.daemon = True
    thread.start()

def _process_audio_task(call_id: int, recording_url: str):
    tmp_path = None
    try:
        # 1. Download the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            response = requests.get(recording_url, timeout=60)
            if response.status_code != 200:
                print(f"ERROR: Failed to download audio from {recording_url}")
                return
            tmp.write(response.content)

        # 2. Transcribe using Whisper
        print(f"DEBUG: [OpenAI Fallback] Transcribing call {call_id} using Whisper-1...")
        with open(tmp_path, "rb") as audio_file:
            transcript_res = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcript_text = transcript_res.text

        # 3. Summarize using GPT-4o
        print(f"DEBUG: [OpenAI Fallback] Summarizing call {call_id} using GPT-4o...")
        summary_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert HR assistant. Summarize the following recruitment call transcript concisely, highlighting key candidate points and the next steps."},
                {"role": "user", "content": transcript_text}
            ]
        )
        summary_text = summary_res.choices[0].message.content

        # 4. Update Database
        print(f"DEBUG: [OpenAI Fallback] Updating database for call {call_id}...")
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE calls
                        SET
                            transcript = COALESCE(NULLIF(transcript, ''), %s),
                            summary = COALESCE(NULLIF(summary, ''), %s)
                        WHERE id = %s
                        """,
                        (transcript_text, summary_text, call_id)
                    )
                    conn.commit()
                print(f"DEBUG: [OpenAI Fallback] Successfully processed AI content for call {call_id}")
                try:
                    from backend.api.routes.calls import invalidate_calls_cache, refresh_call_caches_async

                    invalidate_calls_cache()
                    refresh_call_caches_async()
                except Exception as cache_exc:
                    print(f"WARNING: [OpenAI Fallback] Failed to refresh calls cache: {cache_exc}")

    except Exception as e:
        print(f"ERROR: [OpenAI Fallback] AI Processing failed for call {call_id}: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

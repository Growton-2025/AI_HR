import os
import requests
import tempfile
import threading
from typing import Optional
from openai import OpenAI
from backend.db.connection import get_db_connection_context
from backend.services.frejun_calls import (
    SHORT_CALL_SUMMARY,
    extract_transcript_text,
    is_brief_transcript,
    normalize_summary_text,
    prefer_better_summary,
    prefer_richer_text,
)

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
        transcript_text = (transcript_res.text or "").strip()

        # 3. Summarize using GPT-4o
        if is_brief_transcript(transcript_text):
            summary_text = SHORT_CALL_SUMMARY
        else:
            print(f"DEBUG: [OpenAI Fallback] Summarizing call {call_id} using GPT-4o...")
            summary_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert HR assistant. Summarize the recruitment call concisely. "
                            "Use only the transcript provided. Never ask for more information, never say the transcript "
                            "is incomplete, and never mention missing context. If the call is too brief to extract hiring "
                            "signal, return exactly: Very brief exchange; no meaningful screening details captured."
                        ),
                    },
                    {"role": "user", "content": transcript_text},
                ]
            )
            summary_text = normalize_summary_text(
                summary_res.choices[0].message.content,
                transcript_text=transcript_text,
                short_call_fallback=True,
            ) or "Call completed; transcript captured but AI summary could not be generated reliably."

        # 4. Update Database
        print(f"DEBUG: [OpenAI Fallback] Updating database for call {call_id}...")
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.transcript, c.summary, cand.name
                        FROM calls c
                        JOIN candidates cand ON cand.id = c.candidate_id
                        WHERE c.id = %s
                        """,
                        (call_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return

                    formatted_transcript = extract_transcript_text(
                        transcript_text,
                        candidate_name=row[2],
                    ) or transcript_text

                    next_transcript = prefer_richer_text(row[0], formatted_transcript)
                    next_summary = prefer_better_summary(row[1], summary_text)
                    cur.execute(
                        """
                        UPDATE calls
                        SET
                            transcript = %s,
                            summary = %s
                        WHERE id = %s
                        """,
                        (next_transcript, next_summary, call_id)
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

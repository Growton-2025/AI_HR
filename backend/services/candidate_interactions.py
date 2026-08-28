"""First-party interaction history (calls, transcripts, recruiter notes) for Smart Columns.

The AI-column runner only ever saw the candidate's profile, enrichment and resume, so a
prompt about what actually happened on a call ("did they confirm their notice period?",
"what salary did they quote?") had nothing to read. Every call row already carries the
recording's transcript, the AI summary/sentiment, and the recruiter's own per-call notes;
inbound rows carry the same for calls the candidate placed to us.

This module loads that history on demand — the same lazy hydration used for resume text —
because transcripts are large and the warm profile cache must stay small.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from backend.db.connection import get_db_connection, return_db_connection
from backend.services.call_artifacts import extract_transcript_text, humanize_status

logger = logging.getLogger(__name__)

# How many calls (outbound + inbound, newest first) reach the model, and how much
# transcript text each one may contribute. Transcripts run to tens of KB; without a
# per-call budget one long call evicts every other data point from the context pack.
AI_COLUMN_CALL_LIMIT = int(os.getenv("AI_COLUMN_CALL_LIMIT", "10"))
AI_COLUMN_CALL_TRANSCRIPT_CHAR_BUDGET = int(
    os.getenv("AI_COLUMN_CALL_TRANSCRIPT_CHAR_BUDGET", "6000")
)

# Outbound rows that never connected (a queued task with no transcript, summary,
# note or disposition) are noise, not evidence — they would push real calls past
# the limit.
_OUTBOUND_SQL = """
    SELECT
        c.id,
        COALESCE(c.completed_at, c.updated_at, c.created_at) AS occurred_at,
        c.status,
        c.outcome,
        c.notes,
        c.duration,
        c.task_title,
        c.transcript,
        c.summary,
        c.sentiment,
        c.sentiment_reason,
        c.recording_url,
        cand.name
    FROM calls c
    JOIN candidates cand ON cand.id = c.candidate_id
    WHERE c.candidate_id = %s
      AND (
          NULLIF(TRIM(c.transcript), '') IS NOT NULL
          OR NULLIF(TRIM(c.summary), '') IS NOT NULL
          OR NULLIF(TRIM(c.notes), '') IS NOT NULL
          OR NULLIF(TRIM(c.outcome), '') IS NOT NULL
          OR c.completed_at IS NOT NULL
      )
    ORDER BY occurred_at DESC NULLS LAST
    LIMIT %s
"""

_INBOUND_SQL = """
    SELECT
        ic.id,
        COALESCE(ic.answered_at, ic.received_at) AS occurred_at,
        COALESCE(ic.call_status, ic.dial_status, ic.status),
        ic.resolution,
        ic.note,
        ic.duration,
        ic.transcript,
        ic.recording_url,
        cand.name
    FROM inbound_calls ic
    LEFT JOIN candidates cand ON cand.id = ic.candidate_id
    WHERE ic.candidate_id = %s
      AND (
          NULLIF(TRIM(ic.transcript), '') IS NOT NULL
          OR NULLIF(TRIM(ic.note), '') IS NOT NULL
          OR ic.answered_at IS NOT NULL
      )
    ORDER BY occurred_at DESC NULLS LAST
    LIMIT %s
"""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _iso(value: Any) -> str:
    if not value:
        return ""
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _clean_transcript(raw: Any, candidate_name: str) -> str:
    text = extract_transcript_text(raw, candidate_name=candidate_name) or ""
    text = text.strip()
    if len(text) > AI_COLUMN_CALL_TRANSCRIPT_CHAR_BUDGET:
        text = text[:AI_COLUMN_CALL_TRANSCRIPT_CHAR_BUDGET].rstrip() + " …[truncated]"
    return text


def _outbound_row_to_call(row) -> Dict[str, Any]:
    candidate_name = _text(row[12])
    return {
        "id": row[0],
        "direction": "outbound",
        "date": _iso(row[1]),
        "status": humanize_status(_text(row[2])) or _text(row[2]),
        "outcome": _text(row[3]),
        "notes": _text(row[4]),
        "duration_seconds": int(row[5] or 0),
        "task_title": _text(row[6]),
        "transcript": _clean_transcript(row[7], candidate_name),
        "summary": _text(row[8]),
        "sentiment": _text(row[9]),
        "sentiment_reason": _text(row[10]),
        "recording_url": _text(row[11]),
    }


def _inbound_row_to_call(row) -> Dict[str, Any]:
    candidate_name = _text(row[8])
    return {
        "id": row[0],
        "direction": "inbound",
        "date": _iso(row[1]),
        "status": humanize_status(_text(row[2])) or _text(row[2]),
        "outcome": _text(row[3]),
        "notes": _text(row[4]),
        "duration_seconds": int(row[5] or 0),
        "task_title": "",
        "transcript": _clean_transcript(row[6], candidate_name),
        "summary": "",
        "sentiment": "",
        "sentiment_reason": "",
        "recording_url": _text(row[7]),
    }


def fetch_candidate_calls(candidate_id: int, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Newest-first call history for one candidate: outbound and inbound merged.

    Returns [] rather than raising, so a missing inbound_calls table or a DB blip
    degrades the Smart Column to profile-only context instead of failing the run.
    """
    max_calls = limit if limit is not None else AI_COLUMN_CALL_LIMIT
    if max_calls <= 0:
        return []

    conn = get_db_connection()
    if not conn:
        return []

    calls: List[Dict[str, Any]] = []
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(_OUTBOUND_SQL, (candidate_id, max_calls))
                calls.extend(_outbound_row_to_call(row) for row in cur.fetchall())
            except Exception:
                conn.rollback()
                logger.exception("Failed to load outbound calls for candidate %s", candidate_id)

            try:
                cur.execute(_INBOUND_SQL, (candidate_id, max_calls))
                calls.extend(_inbound_row_to_call(row) for row in cur.fetchall())
            except Exception:
                conn.rollback()
                logger.exception("Failed to load inbound calls for candidate %s", candidate_id)
    finally:
        return_db_connection(conn)

    # Undated rows sort last; they are the least useful evidence either way.
    calls.sort(key=lambda call: (call["date"] or "", call["id"]), reverse=True)
    return calls[:max_calls]

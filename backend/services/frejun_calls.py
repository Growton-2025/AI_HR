import math
import re
from typing import Any, Dict, Iterable, Optional


TERMINAL_FREJUN_STATUSES = {
    "busy",
    "not-answered",
    "not_available",
    "user-busy",
    "user-not-available",
    "not-reachable",
    "user-not-answered",
    "blocked",
    "dnd",
    "dncr",
    "not-initiated",
    "completed",
}

RECORDING_READY_EVENTS = {
    "call.recording",
    "call.summary",
    "recording_ready",
}

SUMMARY_READY_EVENTS = {
    "call.summary",
}

SHORT_CALL_SUMMARY = "Very brief exchange; no meaningful screening details captured."
SUMMARY_PLACEHOLDER_PATTERNS = (
    "transcript isn't fully provided",
    "transcript is not fully provided",
    "please share the full",
    "please share the full or additional content",
    "please share additional content",
    "i can help summarize a typical recruitment call",
    "when provided with full details",
    "for me to assist you appropriately",
    "full or additional content",
    "cannot summarize",
    "can't summarize",
    "not enough information",
    "insufficient information",
)

RECRUITER_TRANSCRIPT_LABEL = "Recruiter"
RECRUITER_SPEAKER_HINTS = {
    "recruiter",
    "agent",
    "caller",
    "interviewer",
    "sales",
    "sales rep",
    "sales representative",
    "user",
    "speaker a",
    "speaker 1",
    "channel 0",
    "channel 1",
}
CANDIDATE_SPEAKER_HINTS = {
    "candidate",
    "callee",
    "customer",
    "client",
    "prospect",
    "lead",
    "speaker b",
    "speaker 2",
}


def normalize_phone(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    if raw.startswith("+"):
        return "+" + re.sub(r"\D", "", raw)

    digits = re.sub(r"\D", "", raw)
    if digits.startswith("00"):
        return "+" + digits[2:]
    return digits


def digits_only(value: Optional[str]) -> str:
    return re.sub(r"\D", "", value or "")


def normalize_status(value: Optional[str]) -> Optional[str]:
    raw = (value or "").strip()
    if not raw:
        return None
    return raw.lower().replace(" ", "-")


def humanize_status(value: Optional[str]) -> Optional[str]:
    normalized = normalize_status(value)
    if not normalized:
        return None
    return normalized.replace("_", " ").replace("-", " ").strip().title()


def normalize_duration_seconds(raw_duration: Any, event_name: Optional[str] = None) -> Optional[int]:
    if raw_duration in (None, ""):
        return None

    try:
        value = float(raw_duration)
    except (TypeError, ValueError):
        return None

    if value < 0:
        return None

    normalized_event = (event_name or "").strip().lower()
    if normalized_event in {"call.status", "call.summary"}:
        if value <= 0:
            return 0
        return max(1, int(round(value / 1000.0)))

    if value > 86400:
        return max(1, int(round(value / 1000.0)))

    return int(round(value))


def _normalize_speaker_token(value: Optional[str]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _candidate_transcript_label(candidate_name: Optional[str]) -> str:
    value = (candidate_name or "").strip()
    return value or "Candidate"


def _replace_string_transcript_labels(text: str, candidate_label: str) -> str:
    value = text.strip()
    if not value:
        return ""

    recruiter_pattern = r"^\s*(recruiter|agent|caller|interviewer|sales(?:\s+rep(?:resentative)?)?|user|speaker\s*a|speaker\s*1|channel\s*0|channel\s*1)\s*:"
    candidate_pattern = r"^\s*(candidate|callee|customer|client|prospect|lead|speaker\s*b|speaker\s*2)\s*:"
    value = re.sub(recruiter_pattern, f"{RECRUITER_TRANSCRIPT_LABEL}:", value, flags=re.IGNORECASE | re.MULTILINE)
    value = re.sub(candidate_pattern, f"{candidate_label}:", value, flags=re.IGNORECASE | re.MULTILINE)
    return value


def _has_explicit_speaker_labels(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*[^:\n]{1,40}:\s+\S", text or ""))


def _split_plain_transcript_turns(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    turns = [part.strip() for part in parts if part and part.strip()]
    return turns


def _guess_speaker_labeled_transcript(text: str, candidate_label: str) -> Optional[str]:
    value = (text or "").strip()
    if not value:
        return None

    if _has_explicit_speaker_labels(value):
        return value

    turns = _split_plain_transcript_turns(value)
    if len(turns) < 2:
        return None

    # Keep this heuristic narrow: only relabel short conversational turns,
    # where alternating recruiter/candidate is a reasonable fallback.
    if len(turns) > 20:
        return None
    if any(len(re.findall(r"\b\w+\b", turn)) > 18 for turn in turns):
        return None

    labeled_lines = []
    for idx, turn in enumerate(turns):
        label = RECRUITER_TRANSCRIPT_LABEL if idx % 2 == 0 else candidate_label
        labeled_lines.append(f"{label}: {turn}")
    return "\n".join(labeled_lines)


def _resolve_transcript_speaker_label(
    raw_speaker: Optional[str],
    *,
    candidate_label: str,
    speaker_map: Dict[str, str],
) -> str:
    normalized = _normalize_speaker_token(raw_speaker)
    normalized_candidate = _normalize_speaker_token(candidate_label)

    if normalized == normalized_candidate and normalized_candidate:
        return candidate_label
    if normalized in RECRUITER_SPEAKER_HINTS:
        return RECRUITER_TRANSCRIPT_LABEL
    if normalized in CANDIDATE_SPEAKER_HINTS:
        return candidate_label
    if normalized in speaker_map:
        return speaker_map[normalized]

    # FreJun transcript diarization is typically 2-party. For generic/unknown labels,
    # we pin the first distinct speaker to Recruiter and the second to the candidate.
    if len(speaker_map) == 0:
        speaker_map[normalized] = RECRUITER_TRANSCRIPT_LABEL
        return RECRUITER_TRANSCRIPT_LABEL
    if len(speaker_map) == 1:
        speaker_map[normalized] = candidate_label
        return candidate_label

    return candidate_label if normalized_candidate and normalized == normalized_candidate else (raw_speaker or candidate_label).strip() or candidate_label


def extract_transcript_text(transcript: Any, *, candidate_name: Optional[str] = None) -> Optional[str]:
    candidate_label = _candidate_transcript_label(candidate_name)
    if isinstance(transcript, str):
        value = _replace_string_transcript_labels(transcript, candidate_label).strip()
        guessed = _guess_speaker_labeled_transcript(value, candidate_label)
        if guessed:
            value = guessed
        return value or None

    if not isinstance(transcript, Iterable):
        return None

    lines = []
    speaker_map: Dict[str, str] = {}
    for item in transcript:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        speaker = _resolve_transcript_speaker_label(
            item.get("speaker"),
            candidate_label=candidate_label,
            speaker_map=speaker_map,
        )
        lines.append(f"{speaker}: {text}")

    return "\n".join(lines) if lines else None


def build_summary_text(ai_insights: Any, fallback: Any = None) -> Optional[str]:
    fallback_text = normalize_summary_text(fallback)

    if not isinstance(ai_insights, dict):
        return fallback_text

    summary_block = ai_insights.get("summary")
    if not isinstance(summary_block, dict):
        return fallback_text

    transcript_summary = (summary_block.get("transcript_summary") or "").strip()
    action_items = (summary_block.get("action_items") or "").strip()

    if transcript_summary and action_items:
        return normalize_summary_text(f"{transcript_summary}\n\nAction Items:\n{action_items}")
    if transcript_summary:
        return normalize_summary_text(transcript_summary)
    if action_items:
        return normalize_summary_text(f"Action Items:\n{action_items}")
    return fallback_text


def extract_outcome(payload: Dict[str, Any], details: Dict[str, Any], status: Optional[str], duration_seconds: Optional[int]) -> Optional[str]:
    explicit_outcome = (
        payload.get("call_outcome")
        or details.get("call_outcome")
        or payload.get("outcome")
        or details.get("outcome")
    )
    if isinstance(explicit_outcome, str) and explicit_outcome.strip():
        return explicit_outcome.strip()

    if duration_seconds and duration_seconds > 0:
        return "Answered"

    normalized_status = normalize_status(status)
    if normalized_status in TERMINAL_FREJUN_STATUSES:
        return humanize_status(normalized_status)

    return None


def is_terminal_event(
    *,
    event_name: Optional[str],
    status: Optional[str],
    end_time: Optional[str],
    duration: Any,
) -> bool:
    normalized_event = (event_name or "").strip().lower()
    normalized_status = normalize_status(status)

    if normalized_event in RECORDING_READY_EVENTS:
        return True

    if normalized_event in {"call_cut", "completed"}:
        return True

    if normalized_event == "call.status":
        if end_time:
            return True
        if duration not in (None, ""):
            return True
        if normalized_status in TERMINAL_FREJUN_STATUSES:
            return True

    return normalized_status in TERMINAL_FREJUN_STATUSES


def infer_recording_source(event_name: Optional[str], has_recording_url: bool, fallback_source: Optional[str] = None) -> Optional[str]:
    if fallback_source:
        return fallback_source
    if not has_recording_url:
        return None

    normalized_event = (event_name or "").strip().lower()
    if normalized_event == "call.recording":
        return "frejun_webhook_recording"
    if normalized_event == "call.summary":
        return "frejun_webhook_summary"
    if normalized_event == "call.status":
        return "frejun_webhook_status"
    if normalized_event in {"recording_ready", "completed"}:
        return "frejun_webhook_legacy"
    return "frejun_webhook"


def extract_payload_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}

    details = payload.get("call-details")
    if not isinstance(details, dict):
        details = payload.get("call_details")
    if not isinstance(details, dict):
        details = {}

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    event_name = payload.get("event") or payload.get("event_type") or data.get("event") or data.get("event_type")
    status = (
        payload.get("call_status")
        or data.get("call_status")
        or payload.get("status")
        or data.get("status")
    )
    duration = payload.get("duration")
    if duration in (None, ""):
        duration = data.get("duration")
    if duration in (None, ""):
        duration = payload.get("call_duration")
    if duration in (None, ""):
        duration = data.get("call_duration")

    transcript = payload.get("transcript")
    if transcript in (None, ""):
        transcript = data.get("transcript")
    if transcript in (None, ""):
        transcript = payload.get("call_transcript")
    if transcript in (None, ""):
        transcript = data.get("call_transcript")

    candidate_name = payload.get("candidate_name") or data.get("candidate_name")

    ai_insights = payload.get("ai_insights")
    if not isinstance(ai_insights, dict):
        ai_insights = data.get("ai_insights")
    if not isinstance(ai_insights, dict):
        ai_insights = {}

    normalized_status = normalize_status(status)
    duration_seconds = normalize_duration_seconds(duration, event_name=event_name)
    transcript_text = extract_transcript_text(transcript, candidate_name=candidate_name)
    summary_text = build_summary_text(
        ai_insights,
        fallback=payload.get("summary") or data.get("summary"),
    )

    recording_url = (
        payload.get("recording_url")
        or data.get("recording_url")
        or payload.get("recording")
        or data.get("recording")
        or payload.get("call_recording")
        or data.get("call_recording")
    )
    summary_url = payload.get("summary_url") or data.get("summary_url")
    candidate_number = payload.get("candidate_number") or data.get("candidate_number")
    virtual_number = payload.get("virtual_number") or data.get("virtual_number")
    call_id = payload.get("call_id") or data.get("call_id")
    event_id = payload.get("event_id") or data.get("event_id")
    link = payload.get("link") or data.get("link")
    recruiter_email = payload.get("call_creator") or data.get("call_creator") or payload.get("recruiter") or data.get("recruiter")
    candidate_id = metadata.get("candidate_id") or payload.get("candidate_id") or data.get("candidate_id")
    transaction_id = metadata.get("transaction_id") or payload.get("transaction_id") or data.get("transaction_id")
    job_id = metadata.get("job_id") or payload.get("job_id") or data.get("job_id")
    end_time = payload.get("end_time") or data.get("end_time")
    notes = (
        details.get("notes")
        or payload.get("call_notes")
        or data.get("call_notes")
        or payload.get("notes")
        or data.get("notes")
    )

    return {
        "event_name": event_name,
        "frejun_status": normalized_status,
        "duration_seconds": duration_seconds,
        "recording_url": (recording_url or "").strip() or None,
        "summary_url": (summary_url or "").strip() or None,
        "summary_text": summary_text,
        "transcript_text": transcript_text,
        "candidate_number": normalize_phone(candidate_number),
        "virtual_number": normalize_phone(virtual_number),
        "call_id": str(call_id).strip() if call_id not in (None, "") else None,
        "event_id": str(event_id).strip() if event_id not in (None, "") else None,
        "transaction_id": str(transaction_id).strip() if transaction_id not in (None, "") else None,
        "candidate_id": str(candidate_id).strip() if candidate_id not in (None, "") else None,
        "job_id": str(job_id).strip() if job_id not in (None, "") else None,
        "link": (link or "").strip() or None,
        "recruiter_email": (recruiter_email or "").strip() or None,
        "candidate_name": (candidate_name or "").strip() or None,
        "outcome": extract_outcome(payload, details, normalized_status, duration_seconds),
        "notes": (notes or "").strip() or None,
        "is_terminal": is_terminal_event(
            event_name=event_name,
            status=normalized_status,
            end_time=end_time,
            duration=duration,
        ),
        "recording_source": infer_recording_source(event_name, bool(recording_url)),
    }


def score_call_log_match(
    result: Dict[str, Any],
    *,
    frejun_call_id: Optional[str],
    frejun_event_id: Optional[str],
    frejun_transaction_id: Optional[str],
    candidate_number: Optional[str],
) -> int:
    score = 0
    result_call_id = str(result.get("call_id") or "").strip()
    result_event_id = str(result.get("event_id") or "").strip()
    result_transaction_id = str(result.get("transaction_id") or "").strip()
    result_number = normalize_phone(result.get("candidate_number"))

    if frejun_call_id and result_call_id and frejun_call_id == result_call_id:
        score += 1000
    if frejun_event_id and result_event_id and frejun_event_id == result_event_id:
        score += 700
    if frejun_transaction_id and result_transaction_id and frejun_transaction_id == result_transaction_id:
        score += 500

    normalized_candidate_number = normalize_phone(candidate_number)
    if normalized_candidate_number and result_number:
        if normalized_candidate_number == result_number:
            score += 300
        elif digits_only(normalized_candidate_number)[-10:] and digits_only(normalized_candidate_number)[-10:] == digits_only(result_number)[-10:]:
            score += 150

    return score


def select_best_call_log_result(
    results: list[Dict[str, Any]],
    *,
    frejun_call_id: Optional[str],
    frejun_event_id: Optional[str],
    frejun_transaction_id: Optional[str],
    candidate_number: Optional[str],
) -> Optional[Dict[str, Any]]:
    best_result = None
    best_score = -1

    for result in results or []:
        if not isinstance(result, dict):
            continue
        score = score_call_log_match(
            result,
            frejun_call_id=frejun_call_id,
            frejun_event_id=frejun_event_id,
            frejun_transaction_id=frejun_transaction_id,
            candidate_number=candidate_number,
        )
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is not None and best_score > 0:
        return best_result

    if len(results or []) == 1 and isinstance(results[0], dict):
        return results[0]

    return None


def transcript_preview(value: Optional[str], limit: int = 220) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def is_placeholder_summary(value: Optional[str]) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return False
    return any(pattern in text for pattern in SUMMARY_PLACEHOLDER_PATTERNS)


def is_brief_transcript(value: Optional[str]) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    words = re.findall(r"\b\w+\b", text)
    if len(words) <= 12:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) <= 2 and len(text) <= 120


def normalize_summary_text(
    value: Any,
    *,
    transcript_text: Optional[str] = None,
    short_call_fallback: bool = False,
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if is_placeholder_summary(text):
        if short_call_fallback and is_brief_transcript(transcript_text):
            return SHORT_CALL_SUMMARY
        return None
    return text


def prefer_richer_text(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    current = (existing or "").strip()
    candidate = (incoming or "").strip()

    if not current:
        return candidate or None
    if not candidate:
        return current
    if current == candidate:
        return current

    current_lines = [line for line in current.splitlines() if line.strip()]
    candidate_lines = [line for line in candidate.splitlines() if line.strip()]

    # Prefer the incoming text if it clearly contains more of the conversation.
    if len(candidate_lines) > len(current_lines):
        return candidate
    if len(candidate) > len(current):
        return candidate
    if current in candidate:
        return candidate

    return current


def prefer_better_summary(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    current = normalize_summary_text(existing)
    candidate = normalize_summary_text(incoming)

    if not current:
        return candidate or None
    if not candidate:
        return current
    if current == candidate:
        return current

    current_placeholder = is_placeholder_summary(existing)
    candidate_placeholder = is_placeholder_summary(incoming)

    if current_placeholder and not candidate_placeholder:
        return candidate
    if candidate_placeholder and not current_placeholder:
        return current

    return prefer_richer_text(current, candidate)

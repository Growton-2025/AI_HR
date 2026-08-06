"""
Capture a candidate's mobile number from an outreach reply (email/LinkedIn)
and enroll them into their roles' linked calling lists.

Asana: "Updating mobile number" + "Add the contact to calling campaign".
"""
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_NUMBER_HINTS = ("number", "mobile", "phone", "call me", "reach me", "whatsapp", "contact me")

# For bare numbers with no +country-code, try these numbering plans in order.
# India first (primary market), then common candidate geographies. A number is
# accepted only if it is VALID (not merely possible) in one of these plans —
# this is what keeps 1234567890-style junk out while staying global.
_PREFERRED_REGIONS = ("IN", "US", "GB", "AE", "SG", "CA", "AU", "DE", "FR", "PH")

# A digit run that could plausibly be a phone number (8-15 digits, common
# separators). Shorter runs are CTCs/years/dates far more often than phones.
_DIGIT_RUN_RE = re.compile(r"(?<![\d])\+?\d[\d\s().\-]{6,18}\d(?![\d])")


def _valid_e164(raw: str, region: Optional[str] = None, require_mobile: bool = False) -> Optional[str]:
    """Return the E.164 form iff `raw` is a VALID number (for `region` when
    it has no +country-code), else None. With require_mobile, the number must
    be mobile(-capable) — candidates share mobiles, and this rejects strings
    like 1234567890 that technically parse as fixed-line ranges."""
    import phonenumbers

    try:
        parsed = phonenumbers.parse(raw, region)
        if not phonenumbers.is_valid_number(parsed):
            return None
        if require_mobile:
            ntype = phonenumbers.number_type(parsed)
            if ntype not in (
                phonenumbers.PhoneNumberType.MOBILE,
                phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE,
            ):
                return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return None


def _validate_candidate_number(raw: str) -> Optional[str]:
    """Validate a candidate string against the international plan (+CC) or,
    for bare numbers, the preferred regional numbering plans."""
    raw = str(raw or "").strip()
    if not raw:
        return None
    if raw.startswith("+") or raw.startswith("00"):
        cleaned = "+" + re.sub(r"\D", "", raw.removeprefix("00"))
        return _valid_e164(cleaned)
    digits = re.sub(r"\D", "", raw)
    if not 8 <= len(digits) <= 15:
        return None
    # Bare local formats keep leading trunk zeros meaningful (07911… UK,
    # 050… UAE) — validate the separator-stripped ORIGINAL, not bare digits.
    compact = re.sub(r"[\s().\-]", "", raw)
    for region in _PREFERRED_REGIONS:
        e164 = _valid_e164(compact, region, require_mobile=True)
        if e164:
            return e164
    return None


def extract_phone(text: str) -> Optional[str]:
    """
    Pull one phone number out of free text, any country. Numbers with an
    explicit +country-code are validated against that plan; bare numbers are
    validated against preferred regional plans (India first). gpt-4o-mini
    fallback only when the text hints at a number the scan couldn't parse.
    Returns E.164 (e.g. +9186..., +1415...) or None.
    """
    if not text or not str(text).strip():
        return None
    text = str(text)

    for m in _DIGIT_RUN_RE.finditer(text):
        e164 = _validate_candidate_number(m.group(0))
        if e164:
            return e164

    # LLM fallback — only when the reply plausibly references a number the
    # regexes couldn't parse (odd spacing, digits-in-words, etc.).
    lowered = text.lower()
    has_digit_run = re.search(r"(?:\d[\s.\-]?){7,}", text) is not None
    if not has_digit_run and not any(h in lowered for h in _NUMBER_HINTS):
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Extract the sender's phone/mobile/WhatsApp number from this message, "
                    "if they explicitly shared one. Do not derive a phone number from a "
                    "name, company, CTC, salary, date, or year. If no phone number was "
                    "shared, return null.\n\n"
                    f"Message:\n{text[:4000]}\n\n"
                    'Reply with ONLY a JSON object: {"phone": "<digits with optional +>" or null}'
                ),
            }],
            timeout=20,
        )
        raw = (completion.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = (json.loads(match.group(0)) if match else {}).get("phone")
        if not candidate:
            return None
        # Never trust an implausible answer — must be a valid number somewhere.
        return _validate_candidate_number(candidate)
    except Exception as e:
        logger.warning(f"LLM phone extraction failed: {e}")
        return None


def add_to_role_call_lists(cur, candidate_id: int, restart_cadence: bool = False) -> None:
    """
    Enroll the candidate in the linked call list of every role they belong to.

    Default mode defers to sync_shortlisted_to_call_list, which self-guards on
    linked list existence, 'Shortlisted' status, phone presence and duplicates
    — but refuses candidates with ANY prior row in the list.

    restart_cadence mode (a candidate shared a NEW number): start a fresh
    "Call 1" even for previously-called candidates, as long as there is no
    pending task already and the candidate isn't rejected/not interested.
    """
    if restart_cadence:
        from backend.api.routes.calls import FIRST_ATTEMPT_TITLE

        cur.execute(
            """
            INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
            SELECT DISTINCT rrc.candidate_id, r.linked_call_list_id, 'pending', CURRENT_DATE, %s
            FROM recruitment_role_candidates rrc
            JOIN recruitment_roles r ON r.id = rrc.role_id
            JOIN candidates c ON c.id = rrc.candidate_id
            WHERE rrc.candidate_id = %s
              AND r.linked_call_list_id IS NOT NULL
              AND LOWER(COALESCE(c.status, '')) NOT LIKE '%%reject%%'
              AND LOWER(COALESCE(c.status, '')) NOT LIKE '%%not interested%%'
              AND NOT EXISTS (
                  SELECT 1 FROM calls p
                  WHERE p.candidate_id = rrc.candidate_id
                    AND p.list_id = r.linked_call_list_id
                    AND p.status = 'pending'
              )
            """,
            (FIRST_ATTEMPT_TITLE, candidate_id),
        )
        return

    from backend.services.auto_call_list import sync_shortlisted_to_call_list

    cur.execute(
        "SELECT DISTINCT role_id FROM recruitment_role_candidates WHERE candidate_id = %s",
        (candidate_id,),
    )
    for (role_id,) in cur.fetchall():
        sync_shortlisted_to_call_list(cur, role_id, [candidate_id])


def capture_phone_from_reply(candidate_id: int, reply_text: str) -> Optional[str]:
    """
    Capture a mobile number the candidate shared in their reply. A number the
    candidate sends themselves is authoritative: it fills an empty field AND
    replaces an outdated one — unless a recruiter locked the field. Also
    mirrors the Talent-Pool auto-shortlist rule and enrolls the candidate
    into their roles' calling lists. Failure-isolated: never raises.
    Returns the captured number, or None.
    """
    if not reply_text or not str(reply_text).strip():
        return None

    try:
        from backend.db.connection import get_db_connection_context

        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(NULLIF(TRIM(mobile_phone), ''), NULLIF(TRIM(phone), '')),
                           COALESCE(mobile_phone_locked_by_user, FALSE),
                           COALESCE(status, '')
                    FROM candidates WHERE id = %s
                    """,
                    (candidate_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                current_phone, locked, status = row
                if locked:
                    return None

                # Regex+LLM for phoneless candidates; regex-only when a number
                # already exists (an updated number arrives as digits — no need
                # to spend an LLM call on every reply from every candidate).
                if current_phone:
                    number = None
                    for m in _DIGIT_RUN_RE.finditer(str(reply_text)):
                        number = _validate_candidate_number(m.group(0))
                        if number:
                            break
                else:
                    number = extract_phone(reply_text)
                if not number:
                    return None

                # Same number we already have (in any formatting) — nothing to do.
                if current_phone and re.sub(r"\D", "", current_phone)[-10:] == re.sub(r"\D", "", number)[-10:]:
                    return None

                cur.execute(
                    """
                    UPDATE candidates
                    SET mobile_phone = %s,
                        mobile_phone_wrong = FALSE,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (number, candidate_id),
                )

                # Mirror the Talent-Pool UI rule: entering a phone for an
                # untouched candidate shortlists them (which is also what
                # makes them eligible for the auto call-list sync below).
                if status.strip() in ("", "To be started"):
                    cur.execute(
                        "UPDATE candidates SET status = 'Shortlisted' WHERE id = %s",
                        (candidate_id,),
                    )

                # A CHANGED number restarts the cadence even for previously
                # called candidates; a first number uses the standard sync.
                add_to_role_call_lists(cur, candidate_id, restart_cadence=bool(current_phone))
            conn.commit()

        logger.info(f"Captured phone {number} for candidate {candidate_id} from reply")

        # Cache invalidation — same set the enrichment callback refreshes.
        try:
            from backend.pipeline import query

            query.refresh_profiles_in_cache([candidate_id])
        except Exception:
            pass
        try:
            from backend.api.routes.calls import invalidate_calls_cache

            invalidate_calls_cache()
        except Exception:
            pass
        try:
            from backend.api.routes import browse as browse_mod

            browse_mod._invalidate_browse_cache()
        except Exception:
            pass

        return number
    except Exception as e:
        logger.warning(f"Phone capture failed for candidate {candidate_id}: {e}")
        return None

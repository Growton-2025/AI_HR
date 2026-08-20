
import os
import logging
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

# --- CONFIG (loaded from environment variables) ---
CLAY_URL = os.getenv("CLAY_URL", "https://api.clay.com/v3/sources/webhook/pull-in-data-from-a-webhook-5232297e-b5f6-4eed-a2f1-79fd3dbbc652")
CLAY_AUTH = os.getenv("CLAY_AUTH", "ed3167e26e52ac14c377")

# How many rows to hand Clay at once. Each call is a blocking HTTP round trip,
# so a bulk shortlist of a few hundred candidates must not be sent one after
# another on the request thread.
_MAX_PARALLEL_TRIGGERS = 8


def _callback_base() -> str:
    """Where Clay should post results back to.

    This used to be `os.getenv("NGROK_URL", "https://14af-....ngrok-free.app")`
    — an expired tunnel from somebody's laptop as the DEFAULT. Any deployment
    that did not happen to set NGROK_URL sent every candidate to Clay with a
    callback pointing at a dead host: Clay would run the waterfall, bill for
    it, post the result into the void, and the only symptom was contact
    coverage quietly sitting low.

    Must be configured explicitly. Auto-detecting "whatever tunnel is running
    on this machine" — as the Plivo integration does — is not safe here: the
    payload Clay posts back carries a candidate's email and phone number, and
    on a developer laptop the detected tunnel may well belong to an unrelated
    application. Better to refuse and say so.
    """
    explicit = os.getenv("PUBLIC_URL") or os.getenv("NGROK_URL")
    return explicit.rstrip("/") if explicit else ""


def trigger_clay(first_name: str, last_name: str, linkedin_url: str) -> bool:
    """Send one contact into the Clay waterfall."""
    base = _callback_base()
    if not base:
        # Sending anyway would spend Clay credits on a result nobody can
        # receive, and look identical to Clay finding nothing.
        logger.error(
            "Not sending %s %s to Clay: no public callback URL is configured "
            "(set PUBLIC_URL or NGROK_URL, or run a tunnel).",
            first_name, last_name,
        )
        return False

    payload = {
        "first_name": first_name,
        "last_name": last_name,
        "linkedin_url": linkedin_url,
        "callback_url": f"{base}/results",
    }
    headers = {
        "Content-Type": "application/json",
        "x-clay-webhook-auth": CLAY_AUTH,
    }

    logger.info(f"🚀 Sending {first_name} to Clay Waterfall...")
    try:
        resp = requests.post(CLAY_URL, json=payload, headers=headers, timeout=10)
        logger.info(f"Clay Webhook Status: {resp.text}")
        return resp.ok
    except Exception as e:
        logger.error(f"Clay trigger failed: {e}")
        return False


def trigger_clay_bulk(contacts) -> dict:
    """Send a whole selection into Clay, and report what happened.

    Queued as ONE background task rather than one per candidate: FastAPI runs
    background tasks sequentially, and trigger_clay blocks on the network, so
    a 46-candidate shortlist would otherwise tie up the worker for the length
    of 46 HTTP round trips.

    `contacts` is an iterable of (first_name, last_name, linkedin_url).
    """
    rows = [
        (str(f or "").strip(), str(l or "").strip(), str(li or "").strip())
        for f, l, li in (contacts or [])
    ]
    # A row with no LinkedIn cannot be enriched — Clay matches on the profile.
    sendable = [r for r in rows if r[2]]
    skipped = len(rows) - len(sendable)
    if not sendable:
        logger.info("Clay bulk: nothing to send (%d without a LinkedIn URL)", skipped)
        return {"sent": 0, "failed": 0, "skipped": skipped}

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_TRIGGERS) as pool:
        results = list(pool.map(lambda r: trigger_clay(*r), sendable))

    sent = sum(1 for ok in results if ok)
    failed = len(results) - sent
    logger.info(
        "Clay bulk: %d sent, %d failed, %d skipped without a LinkedIn URL",
        sent, failed, skipped,
    )
    return {"sent": sent, "failed": failed, "skipped": skipped}

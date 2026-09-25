"""Pull a person's conversation history from HeyReach and Smartlead when we
never stored it (a thread that predates our records, a lost webhook, a row
that was archived and re-added). PR 3 of docs/candidate-history-linking-plan.md.

Shape:
* ``person_history_jobs`` — a queue keyed (person_key, provider). Rows are
  enqueued when a candidate is added or looked up, and claimed with
  ``FOR UPDATE SKIP LOCKED`` so the four gunicorn workers never fetch the
  same person twice (the same lesson as the call-insights claim).
* ``person_provider_threads`` — what came back, keyed on the *person*, in the
  same message shape as ``candidate_outreach.*_chat_history_cache`` so the
  timeline merges and dedupes them with what we already hold.
* One daemon thread per process drains the queue slowly: HeyReach's 300
  req/min is shared by every integration, and production already brushes
  Smartlead's 200/min from the reply poller. A tick every 12s with at most
  one job per worker keeps this well under 20% of either budget.

Everything is behind ``ENABLE_PERSON_HISTORY_BACKFILL`` (default off).
"""
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

FLAG = "ENABLE_PERSON_HISTORY_BACKFILL"
PROVIDER_HEYREACH = "heyreach"
PROVIDER_SMARTLEAD = "smartlead"
MAX_ATTEMPTS = 5
BACKFILL_DAYS = 365          # decision 4 in the plan: last twelve months
TICK_SECONDS = int(os.getenv("PERSON_HISTORY_TICK_SECONDS", "12"))
# A claimed job that never completes (worker killed mid-fetch) becomes
# claimable again after this long.
RUNNING_STALE_MINUTES = 10

_schema_ready = False
_thread: Optional[threading.Thread] = None


def enabled() -> bool:
    return os.getenv(FLAG, "false").strip().lower() in {"1", "true", "yes", "on"}


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


# ── schema ─────────────────────────────────────────────────────────────────

def ensure_backfill_schema(cur) -> None:
    global _schema_ready
    if _schema_ready:
        return
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS person_provider_threads (
            person_key  TEXT NOT NULL,
            provider    VARCHAR(16) NOT NULL,
            thread_ref  TEXT NOT NULL,
            messages    JSONB NOT NULL,
            fetched_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (person_key, provider, thread_ref)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS person_history_jobs (
            id           SERIAL PRIMARY KEY,
            person_key   TEXT NOT NULL,
            provider     VARCHAR(16) NOT NULL,
            identity     TEXT NOT NULL,             -- profile URL or email to ask the provider about
            status       VARCHAR(16) NOT NULL DEFAULT 'queued',
            attempts     INTEGER NOT NULL DEFAULT 0,
            next_run_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_error   TEXT,
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at  TIMESTAMP,
            UNIQUE (person_key, provider)
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_person_history_jobs_due ON person_history_jobs (status, next_run_at);"
    )
    _schema_ready = True


# ── enqueue ────────────────────────────────────────────────────────────────

def _profile_url(person_key: str, linkedin_url: Optional[str]) -> Optional[str]:
    if linkedin_url and "linkedin.com" in str(linkedin_url).lower():
        return str(linkedin_url).strip()
    if person_key and person_key.startswith("/in/"):
        return f"https://www.linkedin.com{person_key}"
    return None


def enqueue_backfill(cur, person_key: Optional[str], *, linkedin_url: Optional[str] = None,
                     email: Optional[str] = None) -> List[str]:
    """Queue provider fetches for a person. Idempotent: an existing job is
    re-queued only when it failed or finished more than a week ago."""
    if not enabled() or not person_key:
        return []
    wanted: List[Tuple[str, str]] = []
    url = _profile_url(person_key, linkedin_url)
    if url and os.getenv("HEYREACH_API_KEY"):
        wanted.append((PROVIDER_HEYREACH, url))
    if email and os.getenv("SMARTLEAD_API_KEY"):
        wanted.append((PROVIDER_SMARTLEAD, str(email).strip().lower()))
    if not wanted:
        return []
    try:
        cur.execute("SAVEPOINT person_backfill")
        ensure_backfill_schema(cur)
        for provider, identity in wanted:
            cur.execute(
                """
                INSERT INTO person_history_jobs (person_key, provider, identity)
                VALUES (%s, %s, %s)
                ON CONFLICT (person_key, provider) DO UPDATE
                SET status = 'queued', attempts = 0, next_run_at = CURRENT_TIMESTAMP,
                    identity = EXCLUDED.identity, last_error = NULL
                WHERE person_history_jobs.status = 'failed'
                   OR (person_history_jobs.status = 'done'
                       AND person_history_jobs.finished_at < CURRENT_TIMESTAMP - INTERVAL '7 days')
                """,
                (person_key, provider, identity),
            )
        cur.execute("RELEASE SAVEPOINT person_backfill")
        return [p for p, _ in wanted]
    except Exception as exc:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT person_backfill")
        except Exception:
            pass
        logger.warning("Could not enqueue history backfill for %s: %s", person_key, exc)
        return []


# ── queue mechanics ─────────────────────────────────────────────────────────

def claim_job(cur) -> Optional[dict]:
    """Take one due job. SKIP LOCKED means four workers polling at once each
    take a different row; a job left 'running' by a dead worker is due again
    after RUNNING_STALE_MINUTES."""
    cur.execute(
        """
        UPDATE person_history_jobs
           SET status = 'running', attempts = attempts + 1,
               next_run_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 minute')
         WHERE id = (
            SELECT id FROM person_history_jobs
             WHERE (status = 'queued' OR status = 'running') AND next_run_at <= CURRENT_TIMESTAMP
             ORDER BY next_run_at, id
             FOR UPDATE SKIP LOCKED
             LIMIT 1
         )
        RETURNING id, person_key, provider, identity, attempts
        """,
        (RUNNING_STALE_MINUTES,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "person_key": row[1], "provider": row[2], "identity": row[3], "attempts": row[4]}


def complete_job(cur, job_id: int) -> None:
    cur.execute(
        "UPDATE person_history_jobs SET status = 'done', finished_at = CURRENT_TIMESTAMP, last_error = NULL WHERE id = %s",
        (job_id,),
    )


def backoff_minutes(attempts: int) -> int:
    return 2 ** max(1, attempts)          # 2, 4, 8, 16, 32


def fail_job(cur, job_id: int, attempts: int, error: str, *, retryable: bool = True) -> str:
    """Retry with exponential backoff up to MAX_ATTEMPTS, then give up."""
    if retryable and attempts < MAX_ATTEMPTS:
        cur.execute(
            """
            UPDATE person_history_jobs
               SET status = 'queued', last_error = %s,
                   next_run_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 minute')
             WHERE id = %s
            """,
            (error[:500], backoff_minutes(attempts), job_id),
        )
        return "queued"
    cur.execute(
        "UPDATE person_history_jobs SET status = 'failed', last_error = %s, finished_at = CURRENT_TIMESTAMP WHERE id = %s",
        (error[:500], job_id),
    )
    return "failed"


def store_threads(cur, person_key: str, provider: str, threads: List[Tuple[str, list]]) -> int:
    n = 0
    for thread_ref, messages in threads:
        if not messages:
            continue
        import json
        cur.execute(
            """
            INSERT INTO person_provider_threads (person_key, provider, thread_ref, messages)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (person_key, provider, thread_ref) DO UPDATE
            SET messages = EXCLUDED.messages, fetched_at = CURRENT_TIMESTAMP
            """,
            (person_key, provider, str(thread_ref), json.dumps(messages, default=str)),
        )
        n += 1
    return n


# ── provider fetchers ──────────────────────────────────────────────────────

def _since() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)


def _within_window(msg: dict) -> bool:
    raw = msg.get("time") or msg.get("created_at") or msg.get("sent_at")
    if not raw:
        return True
    try:
        from dateutil import parser as dateparser
        when = dateparser.parse(str(raw))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when >= _since()
    except Exception:
        return True


def fetch_heyreach(profile_url: str) -> List[Tuple[str, list]]:
    """The LinkedIn thread HeyReach holds for this profile (its search returns
    the matching conversation; the chatroom call returns every message)."""
    from backend.integrations.heyreach import HeyReachBot

    result = HeyReachBot().get_li_chat_history(profile_url)
    messages = [m for m in (result.get("messages") or []) if isinstance(m, dict) and _within_window(m)]
    conv = result.get("conversation_id")
    if not conv or not messages:
        return []
    return [(str(conv), messages)]


def fetch_smartlead(email: str) -> List[Tuple[str, list]]:
    """Every campaign the lead was in, and each campaign's message history —
    the existing bot only knows one campaign at a time."""
    import requests

    api_key = os.getenv("SMARTLEAD_API_KEY")
    base = os.getenv("SMARTLEAD_BASE_URL", "https://server.smartlead.ai")
    res = requests.get(f"{base}/api/v1/leads/", params={"api_key": api_key, "email": email}, timeout=15)
    if res.status_code == 429:
        raise ProviderError("Smartlead rate limit", retryable=True)
    if res.status_code != 200:
        raise ProviderError(f"Smartlead lead lookup {res.status_code}", retryable=res.status_code >= 500)
    try:
        data = res.json()
    except Exception:
        return []
    lead = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
    lead_id = lead.get("id") if isinstance(lead, dict) else None
    if not lead_id:
        return []
    campaigns = []
    for entry in (lead.get("lead_campaign_data") or []):
        cid = entry.get("campaign_id") if isinstance(entry, dict) else None
        if cid:
            campaigns.append(cid)
    threads: List[Tuple[str, list]] = []
    for cid in campaigns:
        hist = requests.get(
            f"{base}/api/v1/campaigns/{cid}/leads/{lead_id}/message-history",
            params={"api_key": api_key, "event_time_gt": _since().strftime("%Y-%m-%dT%H:%M:%SZ")},
            timeout=15,
        )
        if hist.status_code == 429:
            raise ProviderError("Smartlead rate limit", retryable=True)
        if hist.status_code != 200:
            continue
        try:
            raw = hist.json()
        except Exception:
            continue
        msgs = raw.get("history") or raw.get("data") or raw.get("messages") if isinstance(raw, dict) else raw
        msgs = [m for m in (msgs or []) if isinstance(m, dict)]
        if msgs:
            threads.append((f"campaign:{cid}", msgs))
    return threads


FETCHERS = {PROVIDER_HEYREACH: fetch_heyreach, PROVIDER_SMARTLEAD: fetch_smartlead}


# ── worker ─────────────────────────────────────────────────────────────────

def run_one(conn) -> Optional[str]:
    """Claim and run a single job on this connection. Returns the job status."""
    cur = conn.cursor()
    try:
        ensure_backfill_schema(cur)
        job = claim_job(cur)
        conn.commit()
        if not job:
            return None
        try:
            threads = FETCHERS[job["provider"]](job["identity"])
            stored = store_threads(cur, job["person_key"], job["provider"], threads)
            complete_job(cur, job["id"])
            conn.commit()
            logger.info("History backfill %s for %s: %s thread(s)", job["provider"], job["person_key"], stored)
            return "done"
        except Exception as exc:
            conn.rollback()
            retryable = getattr(exc, "retryable", True)
            status = fail_job(cur, job["id"], job["attempts"], str(exc), retryable=retryable)
            conn.commit()
            logger.warning("History backfill %s for %s -> %s: %s", job["provider"], job["person_key"], status, exc)
            return status
    finally:
        cur.close()


def _loop() -> None:
    from backend.db.connection import get_db_connection, return_db_connection

    while True:
        time.sleep(TICK_SECONDS)
        if not enabled():
            continue
        conn = None
        try:
            conn = get_db_connection(validate=False, register_pgvector=False)
            if conn:
                run_one(conn)
        except Exception:
            logger.exception("History backfill tick failed")
        finally:
            if conn:
                return_db_connection(conn)


def start_worker() -> None:
    global _thread
    if not enabled():
        logger.info("Person history backfill disabled (%s unset)", FLAG)
        return
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="person-history-backfill")
    _thread.start()
    logger.info("Person history backfill worker started (tick %ss)", TICK_SECONDS)


# ── readers for the timeline ───────────────────────────────────────────────

def pending_providers(cur, person_key: Optional[str]) -> List[str]:
    if not person_key:
        return []
    cur.execute(
        "SELECT provider FROM person_history_jobs WHERE person_key = %s AND status IN ('queued', 'running')",
        (person_key,),
    )
    return sorted({r[0] for r in cur.fetchall()})


def provider_threads(cur, person_key: Optional[str]) -> List[Tuple[str, str, list]]:
    if not person_key:
        return []
    cur.execute(
        "SELECT provider, thread_ref, messages FROM person_provider_threads WHERE person_key = %s",
        (person_key,),
    )
    return [(r[0], r[1], r[2] if isinstance(r[2], list) else []) for r in cur.fetchall()]

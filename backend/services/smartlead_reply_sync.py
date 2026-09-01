"""
Server-side email reply poller.

The Smartlead counterpart to heyreach_reply_sync. Until now email replies
reached the app ONLY when a recruiter pressed "Sync Responses" — there was no
webhook registered (the one EMAIL_REPLY subscriber on the live campaign points
at Clay) and no background sync, so a candidate could reply and go on reading as
"No response yet" indefinitely.

SMARTLEAD_WEBHOOK_URL is the real-time path once the backend is publicly
reachable; this poller needs nothing and covers every environment.

Smartlead has no "conversations changed since X" listing — the reply data lives
per lead, behind the campaign's message history. So rather than sweeping every
enrolled candidate (hundreds of calls a minute), each cycle checks a bounded
slice of the candidates that could still produce a NEW reply: enrolled, not yet
marked replied, and contacted recently. Anyone who has already replied is
skipped, so the work shrinks as a campaign matures.
"""
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Poll cadence; 0 disables the poller entirely.
POLL_SECONDS = int(os.getenv("SMARTLEAD_REPLY_POLL_SECONDS", "120"))
# Leads examined per cycle. Each costs one Smartlead message-history call, so
# this is the real rate limiter: 40 per 120s ≈ 20 req/min.
BATCH_SIZE = int(os.getenv("SMARTLEAD_REPLY_POLL_BATCH", "40"))
# Stop watching a lead this long after their last send. A reply after three
# weeks of silence is rare, and the manual sync and webhook both still catch it.
STALE_AFTER_DAYS = int(os.getenv("SMARTLEAD_REPLY_POLL_STALE_DAYS", "21"))

_thread: threading.Thread | None = None
# Round-robin cursor: without it the same first BATCH_SIZE rows are re-checked
# every cycle and a campaign larger than the batch never gets swept.
_offset = 0


# Candidates contacted within this window are checked EVERY cycle, not once per
# round-robin sweep. A reply almost always lands within days of the send, and a
# full sweep of a few hundred leads takes ~20 minutes at BATCH_SIZE per cycle —
# far too slow to feel like the real-time sync this is meant to replace.
HOT_WINDOW_HOURS = int(os.getenv("SMARTLEAD_REPLY_POLL_HOT_HOURS", "72"))
HOT_SLOTS = int(os.getenv("SMARTLEAD_REPLY_POLL_HOT_SLOTS", "15"))

# Enrolled with an email address. Deliberately NOT filtered on
# status <> 'replied': an earlier version excluded anyone who had replied once,
# so every FOLLOW-UP message in a thread was invisible to the sync — a candidate
# who answered, then sent three more mails, still showed only the first. That is
# the "missing a lot of messages" report. A conversation stays watched while it
# is active; the STALE_AFTER_DAYS window is what eventually retires it.
_ELIGIBLE_PREDICATE = """
          co.campaign_id IS NOT NULL
          AND NULLIF(TRIM(c.email), '') IS NOT NULL
"""


_HOT_COLUMNS = """
        co.candidate_id, c.email, co.campaign_id,
        COALESCE(jsonb_array_length(
            CASE WHEN jsonb_typeof(co.email_chat_history_cache) = 'array'
                 THEN co.email_chat_history_cache END), 0) AS cached_len,
        co.last_message_sent_at
"""

# Real conversation activity — NOT updated_at, which the poller bumps itself
# every time it writes. Ordering by updated_at made recently SYNCED leads crowd
# out recently CONTACTED ones, so a fresh send fell out of the watch list.
_ACTIVITY_EXPR = "GREATEST(co.last_message_sent_at, co.response_received_at)"


def _recently_active(cur, limit: int):
    """Threads with genuine recent traffic, newest first."""
    cur.execute(
        f"""
        SELECT {_HOT_COLUMNS}
        FROM candidate_outreach co
        JOIN candidates c ON c.id = co.candidate_id
        WHERE {_ELIGIBLE_PREDICATE}
          AND {_ACTIVITY_EXPR} > NOW() - (%s || ' hours')::interval
        ORDER BY {_ACTIVITY_EXPR} DESC
        LIMIT %s
        """,
        (str(HOT_WINDOW_HOURS), limit),
    )
    return cur.fetchall()


def _awaiting_first_reply(cur, limit: int):
    """Recently enrolled leads that have not answered yet.

    Their own send time is often missing (the send path does not always record
    last_message_sent_at), so they have no activity timestamp to sort on and
    would never surface in the slice above — which is exactly where a
    just-mailed candidate sits while you wait for the reply.
    """
    cur.execute(
        f"""
        SELECT {_HOT_COLUMNS}
        FROM candidate_outreach co
        JOIN candidates c ON c.id = co.candidate_id
        WHERE {_ELIGIBLE_PREDICATE}
          AND co.response_received_at IS NULL
          AND COALESCE(co.status, '') <> 'replied'
          AND COALESCE(co.last_message_sent_at, co.updated_at)
              > NOW() - (%s || ' hours')::interval
        ORDER BY COALESCE(co.last_message_sent_at, co.updated_at) DESC
        LIMIT %s
        """,
        (str(HOT_WINDOW_HOURS), limit),
    )
    return cur.fetchall()


def _hot_candidates(cur, limit: int):
    """Two tiers, so neither starves the other: live conversations AND leads
    still waiting on their first reply."""
    half = max(limit // 2, 1)
    active = _recently_active(cur, half)
    seen = {row[0] for row in active}
    waiting = [row for row in _awaiting_first_reply(cur, limit) if row[0] not in seen]
    return active + waiting[: max(limit - len(active), 0)]


def _candidates_to_check(cur, limit: int, offset: int):
    """Enrolled, not yet replied, contacted recently — oldest check first."""
    cur.execute(
        f"""
        SELECT co.candidate_id, c.email, co.campaign_id,
               COALESCE(jsonb_array_length(
                   CASE WHEN jsonb_typeof(co.email_chat_history_cache) = 'array'
                        THEN co.email_chat_history_cache END), 0) AS cached_len,
               co.last_message_sent_at
        FROM candidate_outreach co
        JOIN candidates c ON c.id = co.candidate_id
        WHERE {_ELIGIBLE_PREDICATE}
          AND COALESCE(co.last_message_sent_at, co.updated_at, NOW())
              > NOW() - (%s || ' days')::interval
        ORDER BY co.candidate_id
        OFFSET %s
        LIMIT %s
        """,
        (str(STALE_AFTER_DAYS), offset, limit),
    )
    return cur.fetchall()


def _batch_for_cycle(cur):
    """Hot leads first, then the round-robin sweep, de-duplicated.

    The hot slice keeps freshly contacted candidates at ~POLL_SECONDS latency
    while the sweep still guarantees every enrolled lead is eventually checked.
    """
    hot = _hot_candidates(cur, HOT_SLOTS)
    seen = {row[0] for row in hot}
    remaining = max(BATCH_SIZE - len(hot), 0)
    sweep = _candidates_to_check(cur, remaining, _offset) if remaining else []
    return hot + [row for row in sweep if row[0] not in seen], len(sweep)


def _promote_reply(cur, candidate_id: int, activity: dict) -> bool:
    """Write a discovered reply onto every outreach row for this candidate."""
    reply_text = (activity.get("reply_text") or "").strip()
    if not reply_text:
        return False

    cur.execute(
        """
        UPDATE candidate_outreach
           SET status = 'replied',
               response_text = %s,
               response_received_at = COALESCE(%s::timestamptz, response_received_at, NOW()),
               updated_at = NOW()
         WHERE candidate_id = %s
           AND COALESCE(response_text, '') IS DISTINCT FROM %s
        """,
        (reply_text, activity.get("reply_at"), candidate_id, reply_text),
    )
    return cur.rowcount > 0


def poll_once() -> int:
    """One polling cycle. Returns the number of candidates promoted to replied.

    Structured in three phases so that NO pooled DB connection is ever held
    across a Smartlead HTTP call. An earlier version wrapped the whole batch in
    one `with get_db_connection_context()`, which checked out a connection for
    the length of 40 sequential API round-trips: it starved a pool sized
    min=8/max=16 and the connection was dead by commit time
    ("SSL SYSCALL error: Operation timed out" after a 32-minute cycle). The same
    hazard is called out in ai_columns._process_ai_run.
    """
    global _offset

    from backend.db.connection import get_db_connection_context
    from backend.integrations.smartlead import SmartleadBot

    api_key = os.getenv("SMARTLEAD_API_KEY")
    if not api_key:
        return 0

    # Phase 1 — read the work list, then release the connection immediately.
    # validate=True: the pool hands out connections the server has already closed
    # (observed as "server closed the connection unexpectedly" mid-sweep), and a
    # cycle lost that way is a cycle in which no replies reach the app.
    with get_db_connection_context(validate=True, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            rows, sweep_len = _batch_for_cycle(cur)
        # Read-only, but end the transaction rather than leaving one idle.
        conn.rollback()

    # Wrap around once the sweep runs off the end of the set.
    sweep_size = max(BATCH_SIZE - HOT_SLOTS, 0)
    _offset = _offset + sweep_size if sweep_len == sweep_size and sweep_size else 0
    if not rows:
        return 0

    # Phase 2 — talk to Smartlead holding NO database connection.
    #
    # One HTTP call per lead and NOTHING written unless the thread actually
    # changed. Calling the route's full _sync_email_messages for every lead
    # instead meant a DB round-trip per lead across the cross-region link and
    # pushed a single cycle past eight minutes; the vast majority of leads have
    # nothing new, so detection has to be free.
    from backend.api.routes.outreach import _clean_email_body
    from backend.services.outreach_counts import count_inbound_messages

    bot = SmartleadBot(api_key=api_key)
    changed = []
    needs_stamp = []
    for candidate_id, email, campaign_id, cached_len, last_sent_at in rows:
        try:
            bot.campaign_id = campaign_id
            messages = bot.get_chat_history(email, campaign_id)
        except Exception:
            # A dead campaign id (several are stored that Smartlead no longer
            # knows about) must not kill the whole cycle.
            logger.debug("Smartlead thread fetch failed for %s", email, exc_info=True)
            continue

        if not isinstance(messages, list) or not messages:
            continue
        # Nothing new since the last sync — skip the expensive write, but still
        # heal a missing activity stamp. Without this an already-synced row keeps
        # last_message_sent_at NULL forever (the backfill below only runs when the
        # thread grows), so it never re-enters the priority tier and every future
        # reply on that thread waits for the slow round-robin sweep instead.
        if len(messages) <= cached_len:
            if last_sent_at is None:
                stamp = _latest_outbound_time(messages)
                if stamp:
                    needs_stamp.append((candidate_id, stamp))
            continue

        for message in messages:
            body = message.get("email_body")
            if body:
                try:
                    message["email_body"] = _clean_email_body(body)
                except Exception:
                    pass

        if count_inbound_messages(messages) == 0:
            continue
        latest = _latest_inbound(messages)
        if latest:
            changed.append((candidate_id, campaign_id, messages, latest))

    if needs_stamp:
        _backfill_send_stamps(needs_stamp)

    if not changed:
        return 0

    # Phase 3 — persist the threads, then promote, in short transactions.
    promoted = 0
    with get_db_connection_context(validate=True, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            for candidate_id, campaign_id, messages, latest in changed:
                try:
                    cur.execute(
                        """
                        UPDATE candidate_outreach
                           SET email_chat_history_cache = %s::jsonb,
                               email_chat_history_updated_at = NOW()
                         WHERE candidate_id = %s
                           AND (campaign_id = %s OR %s IS NULL)
                        """,
                        (json.dumps(messages, default=str), candidate_id, campaign_id, campaign_id),
                    )
                    # Self-healing: derive the send stamp from the thread, so a
                    # row enrolled by a path that forgot to set it stops looking
                    # stale to the next cycle's prioritisation.
                    newest_sent = _latest_outbound_time(messages)
                    if newest_sent:
                        cur.execute(
                            """
                            UPDATE candidate_outreach
                               SET last_message_sent_at = GREATEST(
                                       COALESCE(last_message_sent_at, %s::timestamptz),
                                       %s::timestamptz)
                             WHERE candidate_id = %s
                            """,
                            (newest_sent, newest_sent, candidate_id),
                        )
                    if _promote_reply(cur, candidate_id, latest):
                        promoted += 1
                except Exception:
                    logger.exception("Failed to store Smartlead thread for candidate %s", candidate_id)
        conn.commit()

    # Drop the in-memory thread cache so an open modal shows the new messages.
    try:
        from backend.api.routes.outreach import _email_chat_cache, _email_chat_lock
        with _email_chat_lock:
            for candidate_id, _campaign_id, _messages, _latest in changed:
                if candidate_id in _email_chat_cache:
                    _email_chat_cache[candidate_id]["ts"] = 0
    except Exception:
        pass

    for candidate_id, _campaign_id, _messages, latest in changed:
        _capture_phone(candidate_id, latest.get("reply_text") or "")

    if promoted:
        try:
            from backend.api.routes.browse import _invalidate_browse_cache
            _invalidate_browse_cache()
        except Exception:
            pass

    return promoted


def _backfill_send_stamps(pairs):
    """Repair rows whose send was never recorded, so they rank correctly again."""
    from backend.db.connection import get_db_connection_context
    try:
        with get_db_connection_context(validate=True, register_pgvector=False) as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                for candidate_id, stamp in pairs:
                    cur.execute(
                        """
                        UPDATE candidate_outreach
                           SET last_message_sent_at = %s::timestamptz
                         WHERE candidate_id = %s AND last_message_sent_at IS NULL
                        """,
                        (stamp, candidate_id),
                    )
            conn.commit()
        logger.info("Backfilled last_message_sent_at for %d row(s)", len(pairs))
    except Exception:
        logger.exception("Failed to backfill send stamps")


def _latest_outbound_time(messages):
    """Timestamp of the newest message WE sent, for the activity stamp."""
    best = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        direction = str(message.get("direction") or "").strip().lower()
        msg_type = str(message.get("type") or "").strip().upper()
        if direction == "outbound" or msg_type in {"SENT", "OUTBOX", "SEQUENCE", "INITIAL"}:
            stamp = str(message.get("time") or message.get("created_at") or "")
            if stamp > best:
                best = stamp
    return best or None


def _latest_inbound(messages):
    """Newest message that came FROM the candidate, as an activity dict."""
    inbound_types = {"INBOX", "REPLY", "REPLIED", "LEAD", "INCOMING"}
    best = None
    best_time = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        direction = str(message.get("direction") or "").strip().lower()
        msg_type = str(message.get("type") or "").strip().upper()
        if direction != "inbound" and not (not direction and msg_type in inbound_types):
            continue
        stamp = str(message.get("time") or message.get("created_at") or "")
        if best is None or stamp > best_time:
            best, best_time = message, stamp
    if best is None:
        return None
    body = str(best.get("email_body") or best.get("body") or "").strip()
    if not body:
        return None
    return {"reply_text": body, "reply_at": best_time or None}


def _capture_phone(candidate_id: int, reply_text: str) -> None:
    """A reply often carries the candidate's mobile number."""
    if not reply_text:
        return
    try:
        from backend.services.phone_capture import capture_phone_from_reply
        capture_phone_from_reply(candidate_id, reply_text)
    except Exception:
        logger.debug("Phone capture failed for candidate %s", candidate_id, exc_info=True)


def _loop():
    cycle = 0
    while True:
        cycle += 1
        try:
            started = time.monotonic()
            promoted = poll_once()
            elapsed = time.monotonic() - started
            # Heartbeat even when idle — silence must mean "thread dead", never
            # "nothing found". Duration included: the first version of this
            # poller ran a 32-minute cycle and the log gave no hint of it.
            if promoted or elapsed > 60 or cycle % 15 == 1:
                logger.info(
                    "Smartlead reply poller heartbeat: cycle %d, promoted %d, offset %d, took %.1fs",
                    cycle, promoted, _offset, elapsed,
                )
        except Exception:
            logger.exception("Smartlead reply poller cycle failed")
        time.sleep(POLL_SECONDS)


def start_poller():
    """Idempotently start the background reply poller."""
    global _thread
    if POLL_SECONDS <= 0:
        logger.info("Smartlead reply poller disabled (SMARTLEAD_REPLY_POLL_SECONDS=0)")
        return
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="smartlead-reply-poller")
    _thread.start()
    logger.info("Smartlead reply poller started (every %ss, batch %s)", POLL_SECONDS, BATCH_SIZE)

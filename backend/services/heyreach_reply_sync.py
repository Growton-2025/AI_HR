"""
Server-side LinkedIn reply poller.

Replies used to reach the app only when a human opened a conversation or
pressed "Sync Responses"/"Manual Sync" — a reply could sit in HeyReach
overnight (the Aug 7 "sync is still not working" report). The webhook covers
this in hosted environments once HEYREACH_WEBHOOK_URL is set, but needs a
publicly reachable URL; this poller needs nothing.

Each cycle asks GetConversationsV3 for conversations whose lastMessageAt is
after a watermark — normally ONE api call for the whole workspace — and
promotes any conversation whose last message came from the lead:
li_status/li_response_text on every matching candidate_outreach row, chat
cache invalidation, and phone capture. The conversation modal still fetches
the full thread on open; this keeps the LISTS honest without anyone clicking.
"""
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Poll cadence; 0 disables the poller entirely.
POLL_SECONDS = int(os.getenv("HEYREACH_REPLY_POLL_SECONDS", "180"))
# Re-read this much history behind the watermark: HeyReach backdates messages
# to their LinkedIn time when its scraper ingests them late, so a message can
# APPEAR with a lastMessageAt minutes in the past.
OVERLAP_MINUTES = 30

_thread: threading.Thread | None = None
_watermark: str | None = None


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def poll_once() -> int:
    """One polling cycle. Returns the number of candidates updated."""
    global _watermark

    from backend.db.connection import get_db_connection_context
    from backend.integrations.heyreach import HeyReachBot

    now = datetime.now(timezone.utc)
    since = _watermark or _utc_iso(now - timedelta(hours=24))

    bot = HeyReachBot()
    items = bot._list_conversations(
        {"seen": None}, limit=100, max_items=500, since=since
    )
    # Advance the watermark regardless of matches, keeping the safety overlap.
    _watermark = _utc_iso(now - timedelta(minutes=OVERLAP_MINUTES))
    if not items:
        return 0

    # EVERY conversation with new activity gets its thread persisted — not
    # just lead-last ones. A conversation where WE spoke last (Unibox sends,
    # our follow-ups) must stay current too, so the app opens with everything
    # already loaded instead of waiting on per-modal fetches.
    activity = []
    for conv in items:
        profile_url = (conv.get("correspondentProfile") or {}).get("profileUrl")
        norm = HeyReachBot._normalize_linkedin_url(profile_url)
        if not norm:
            continue
        last_sender = conv.get("lastMessageSender")
        activity.append(
            {
                "norm": norm,
                "profile_url": profile_url,
                "is_reply": bool(last_sender) and not bot._is_outbound_sender(last_sender),
                "text": (conv.get("lastMessageText") or "").strip(),
                "at": conv.get("lastMessageAt"),
                "conversation_id": conv.get("id"),
                "account_id": conv.get("linkedInAccountId")
                or (conv.get("linkedInAccount") or {}).get("id"),
                # Despite the docs calling it a preview, the listing's
                # messages[] carries the full thread — and FRESHER than the
                # chatroom endpoint, which can lag by hours. This is the only
                # reliable source for every message of a reply burst.
                "thread": bot.format_chat_messages(conv.get("messages") or []),
            }
        )
    if not activity:
        return 0

    import json as _json

    updated_candidates = []
    thread_updates = []
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            for item in activity:
                # candidates.normalized_linkedin exists in TWO formats in this
                # DB: the pipeline's bare slug ('john-doe-123') and HeyReach's
                # path ('/in/john-doe-123'). Match both, with a URL fallback —
                # matching only one format silently skipped real candidates.
                slug = item["norm"].split("/in/")[-1].strip("/")
                cur.execute(
                    """
                    SELECT id FROM candidates
                    WHERE (normalized_linkedin = %s OR normalized_linkedin = %s
                           OR linkedin ILIKE %s)
                      AND COALESCE(is_archived, FALSE) = FALSE
                    """,
                    (item["norm"], slug, f"%/in/{slug}%"),
                )
                rows = cur.fetchall()
                if not rows:
                    continue
                for (candidate_id,) in rows:
                    # Persist the listing's full fresh thread for ANY new
                    # activity — never shrink an existing one.
                    if item["thread"]:
                        cur.execute(
                            """
                            UPDATE candidate_outreach
                            SET li_chat_history_cache = CASE
                                    WHEN COALESCE(jsonb_array_length(li_chat_history_cache), 0) <= %s
                                    THEN %s::jsonb ELSE li_chat_history_cache END,
                                li_chat_history_updated_at = NOW(),
                                li_conversation_id = COALESCE(%s, li_conversation_id),
                                li_account_id = COALESCE(%s, li_account_id)
                            WHERE candidate_id = %s
                            """,
                            (
                                len(item["thread"]), _json.dumps(item["thread"]),
                                item["conversation_id"],
                                str(item["account_id"]) if item["account_id"] else None,
                                candidate_id,
                            ),
                        )
                        if cur.rowcount:
                            thread_updates.append((candidate_id, item["thread"]))

                    if not item["is_reply"]:
                        continue
                    # Promote only replies NEWER than what's stored — the
                    # 30-min overlap window would otherwise re-promote the
                    # same reply every cycle.
                    cur.execute(
                        """
                        UPDATE candidate_outreach
                        SET li_status = 'replied',
                            li_response_text = COALESCE(NULLIF(%s, ''), li_response_text),
                            li_response_received_at = COALESCE(%s::timestamptz, NOW()),
                            updated_at = NOW()
                        WHERE candidate_id = %s
                          AND (li_response_received_at IS NULL
                               OR li_response_received_at < COALESCE(%s::timestamptz, NOW()))
                        """,
                        (item["text"], item["at"], candidate_id, item["at"]),
                    )
                    if cur.rowcount:
                        updated_candidates.append((candidate_id, item, None))
        conn.commit()

    if not updated_candidates and not thread_updates:
        return 0

    # Freshness: replace this process's live thread cache with the listing's
    # full fresh thread (carrying forward local echoes it doesn't contain
    # yet), so an open modal shows every message on its next auto-poll and a
    # freshly opened one is served instantly from cache.
    try:
        from backend.api.routes.outreach import _li_chat_cache, _li_chat_lock

        with _li_chat_lock:
            for candidate_id, thread in thread_updates:
                entry = _li_chat_cache.get(candidate_id)
                if entry is None:
                    _li_chat_cache[candidate_id] = {
                        "messages": thread,
                        "ts": time.monotonic(),
                        "refreshing": False,
                    }
                    continue
                previous_messages = entry.get("messages", [])
                if len(previous_messages) > len(thread):
                    entry["ts"] = 0
                    continue
                thread_bodies = {str(m.get("email_body") or "").strip() for m in thread}
                echoes = [
                    m for m in previous_messages
                    if m.get("local_echo")
                    and str(m.get("email_body") or "").strip() not in thread_bodies
                ]
                entry["messages"] = thread + echoes
                entry["ts"] = time.monotonic()
    except Exception:
        pass
    try:
        from backend.api.routes import browse as browse_mod

        browse_mod._invalidate_browse_cache()
    except Exception:
        pass
    try:
        from backend.pipeline import query

        query.refresh_profiles_in_cache([cid for cid, _t, _e in updated_candidates])
    except Exception:
        pass
    try:
        from backend.api.routes.roles import invalidate_role_detail_cache_for_candidate

        for candidate_id, _t, _e in updated_candidates:
            invalidate_role_detail_cache_for_candidate(candidate_id)
    except Exception:
        pass

    # A reply may carry the candidate's number.
    try:
        from backend.services.phone_capture import capture_phone_from_reply

        for candidate_id, reply, _e in updated_candidates:
            capture_phone_from_reply(candidate_id, reply["text"])
    except Exception:
        logger.exception("Reply poller: phone capture failed")

    logger.info(
        "HeyReach reply poller: %d replies promoted, %d threads refreshed (%d active conversations)",
        len(updated_candidates), len(thread_updates), len(activity),
    )
    return len(updated_candidates)


def _loop():
    cycle = 0
    while True:
        cycle += 1
        try:
            promoted = poll_once()
            # Heartbeat every ~15 min even when idle — silence in the log must
            # mean "thread dead", never "nothing found".
            if promoted or cycle % 5 == 1:
                logger.info(
                    "HeyReach reply poller heartbeat: cycle %d, promoted %d, watermark %s",
                    cycle, promoted, _watermark,
                )
        except Exception:
            logger.exception("HeyReach reply poller cycle failed")
        time.sleep(POLL_SECONDS)


def start_poller():
    """Idempotently start the background reply poller."""
    global _thread
    if POLL_SECONDS <= 0:
        logger.info("HeyReach reply poller disabled (HEYREACH_REPLY_POLL_SECONDS=0)")
        return
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="heyreach-reply-poller")
    _thread.start()
    logger.info("HeyReach reply poller started (every %ss)", POLL_SECONDS)

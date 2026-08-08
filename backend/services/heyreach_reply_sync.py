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

    replied = []
    for conv in items:
        last_sender = conv.get("lastMessageSender")
        if not last_sender or bot._is_outbound_sender(last_sender):
            continue
        profile_url = (conv.get("correspondentProfile") or {}).get("profileUrl")
        norm = HeyReachBot._normalize_linkedin_url(profile_url)
        if not norm:
            continue
        replied.append(
            {
                "norm": norm,
                "profile_url": profile_url,
                "text": (conv.get("lastMessageText") or "").strip(),
                "at": conv.get("lastMessageAt"),
                "conversation_id": conv.get("id"),
                "account_id": conv.get("linkedInAccountId")
                or (conv.get("linkedInAccount") or {}).get("id"),
            }
        )
    if not replied:
        return 0

    updated_candidates = []
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            for reply in replied:
                # candidates.normalized_linkedin exists in TWO formats in this
                # DB: the pipeline's bare slug ('john-doe-123') and HeyReach's
                # path ('/in/john-doe-123'). Match both, with a URL fallback —
                # matching only one format silently skipped real candidates.
                slug = reply["norm"].split("/in/")[-1].strip("/")
                cur.execute(
                    """
                    SELECT id FROM candidates
                    WHERE (normalized_linkedin = %s OR normalized_linkedin = %s
                           OR linkedin ILIKE %s)
                      AND COALESCE(is_archived, FALSE) = FALSE
                    """,
                    (reply["norm"], slug, f"%/in/{slug}%"),
                )
                rows = cur.fetchall()
                if not rows:
                    continue
                for (candidate_id,) in rows:
                    # Promote only replies NEWER than what's stored — the
                    # 30-min overlap window would otherwise re-promote (and
                    # re-echo) the same reply every cycle.
                    cur.execute(
                        """
                        UPDATE candidate_outreach
                        SET li_status = 'replied',
                            li_response_text = COALESCE(NULLIF(%s, ''), li_response_text),
                            li_response_received_at = COALESCE(%s::timestamptz, NOW()),
                            li_conversation_id = COALESCE(%s, li_conversation_id),
                            li_account_id = COALESCE(%s, li_account_id),
                            updated_at = NOW()
                        WHERE candidate_id = %s
                          AND (li_response_received_at IS NULL
                               OR li_response_received_at < COALESCE(%s::timestamptz, NOW()))
                        """,
                        (
                            reply["text"], reply["at"], reply["conversation_id"],
                            str(reply["account_id"]) if reply["account_id"] else None,
                            candidate_id, reply["at"],
                        ),
                    )
                    if cur.rowcount:
                        # HeyReach's chatroom endpoint can lag its own listing
                        # by many minutes — echo the reply into the persisted
                        # thread AND the live memory cache so the conversation
                        # modal shows it NOW. The fetch merge in
                        # _sync_li_messages dedupes it (by body) once the
                        # chatroom catches up.
                        echo = None
                        if reply["text"]:
                            echo = {
                                "id": f"local-reply-{candidate_id}-{reply['at']}",
                                "type": "REPLY",
                                "direction": "inbound",
                                "email_body": reply["text"],
                                "time": reply["at"],
                                "sender_name": "Candidate",
                                "local_echo": True,
                            }
                            import json as _json

                            cur.execute(
                                """
                                UPDATE candidate_outreach
                                SET li_chat_history_cache = COALESCE(li_chat_history_cache, '[]'::jsonb) || %s::jsonb,
                                    li_chat_history_updated_at = NOW()
                                WHERE candidate_id = %s
                                """,
                                (_json.dumps([echo]), candidate_id),
                            )
                        updated_candidates.append((candidate_id, reply, echo))
        conn.commit()

    if not updated_candidates:
        return 0

    # Freshness: append the reply echo into this process's live thread cache
    # (the merge in _sync_li_messages preserves it across provider refetches)
    # and mark entries stale so open modals refetch on their next auto-poll.
    try:
        from backend.api.routes.outreach import _li_chat_cache, _li_chat_lock

        with _li_chat_lock:
            for candidate_id, _text, echo in updated_candidates:
                entry = _li_chat_cache.get(candidate_id)
                if entry is None:
                    continue
                messages = entry.setdefault("messages", [])
                if echo and not any(
                    str(m.get("email_body") or "").strip() == str(echo["email_body"]).strip()
                    for m in messages
                ):
                    messages.append(echo)
                entry["ts"] = 0
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

    # The conversation LISTING only carries the LAST message — a burst of
    # replies between poll cycles surfaces just its final message. Refetch the
    # full chatroom for each replied conversation, now and again after
    # HeyReach's chatroom endpoint has had time to catch up with its own
    # listing, so every message in the burst lands without anyone clicking.
    try:
        from backend.api.routes.outreach import _refresh_li_cache_task

        seen_convs = set()
        for candidate_id, reply, _e in updated_candidates:
            conv_id = reply.get("conversation_id")
            if not conv_id or conv_id in seen_convs:
                continue
            seen_convs.add(conv_id)
            args = (
                candidate_id,
                reply.get("profile_url") or "",
                None,
                str(conv_id),
                int(reply["account_id"]) if reply.get("account_id") else None,
            )
            threading.Thread(target=_refresh_li_cache_task, args=args, daemon=True).start()
            timer = threading.Timer(600, _refresh_li_cache_task, args=args)
            timer.daemon = True
            timer.start()
    except Exception:
        logger.exception("Reply poller: chatroom refetch scheduling failed")

    logger.info(
        "HeyReach reply poller: promoted replies for %d candidate rows (%d conversations)",
        len(updated_candidates), len(replied),
    )
    return len(updated_candidates)


def _loop():
    while True:
        try:
            poll_once()
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

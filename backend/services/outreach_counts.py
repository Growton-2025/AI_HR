"""Shared SQL for counting a candidate's replies (inbound messages only).

The conversation badge in the role table used to show the TOTAL size of a
thread, so a candidate who had never answered still showed "1" the moment the
campaign's first message went out. Recruiters read that badge the way every
messaging app trains them to — "they wrote back N times" — so count only the
messages that came FROM the candidate. No reply yet means no badge at all.
"""

# Mirrors CandidateConversationModal.isIncomingMessage on the frontend: an
# explicit direction wins, and only when it is missing do we fall back to the
# provider's message type (Smartlead/HeyReach both send these labels).
_INBOUND_TYPES = "('INBOX', 'REPLY', 'REPLIED', 'LEAD', 'INCOMING')"


def reply_count_sql(cache_column: str, response_text_column: str) -> str:
    """SQL expression counting inbound messages in a chat-history blob.

    `cache_column` is the jsonb thread cache (may be NULL or a non-array) and
    `response_text_column` the denormalized latest reply. The blob is the
    accurate source; the response text is a floor, because a reply promoted by
    the webhook/poller can land in the columns before the thread is fetched.
    """
    return f"""
        GREATEST(
            CASE
                WHEN jsonb_typeof({cache_column}) = 'array' THEN (
                    SELECT COUNT(*)
                    FROM jsonb_array_elements({cache_column}) AS msg
                    WHERE lower(COALESCE(msg->>'direction', '')) = 'inbound'
                       OR (
                            COALESCE(msg->>'direction', '') = ''
                            AND upper(COALESCE(msg->>'type', '')) IN {_INBOUND_TYPES}
                       )
                )
                ELSE 0
            END,
            CASE
                WHEN NULLIF(TRIM(COALESCE({response_text_column}, '')), '') IS NULL THEN 0
                ELSE 1
            END
        )
    """


def count_inbound_messages(messages) -> int:
    """Python twin of `reply_count_sql` for threads already held in memory."""
    if not isinstance(messages, list):
        return 0
    inbound_types = {"INBOX", "REPLY", "REPLIED", "LEAD", "INCOMING"}
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        direction = str(message.get("direction") or "").strip().lower()
        if direction == "inbound":
            total += 1
        elif not direction and str(message.get("type") or "").strip().upper() in inbound_types:
            total += 1
    return total

"""The conversation badge counts REPLIES, not the size of the thread.

Sending the campaign's first LinkedIn message put a "1" on the badge before the
candidate had said anything, and every recruiter reads that number the way a
chat app trains them to: "they wrote back". Same bug fed the Responded tab,
which listed anyone with an open conversation.

These tests pin the three places the rule now lives: the SQL builder, the two
endpoints that serve the counts, and the JSX that renders them.
"""

import asyncio
import pathlib
import re

from backend.api import schemas
from backend.api.routes import outreach, roles
from backend.services.outreach_counts import count_inbound_messages, reply_count_sql

ROLES_JSX = (
    pathlib.Path(__file__).resolve().parent.parent
    / "frontend" / "src" / "pages" / "Roles.jsx"
).read_text()


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append(sql)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Connection:
    def __init__(self, rows):
        self.rows = rows
        self.cursors = []

    def cursor(self):
        cur = _Cursor(self.rows)
        self.cursors.append(cur)
        return cur

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ── the classifier ───────────────────────────────────────────────────────────

def test_only_inbound_messages_count_as_replies():
    thread = [
        {"direction": "outbound", "email_body": "Hi, we're hiring"},   # ours
        {"type": "SENT", "email_body": "Following up"},                # ours
        {"direction": "inbound", "email_body": "Interested!"},         # theirs
        {"type": "REPLY", "email_body": "What's the CTC?"},            # theirs
    ]
    assert count_inbound_messages(thread) == 2


def test_direction_wins_over_message_type():
    # HeyReach labels a SENT echo with type SENT and direction outbound; a
    # provider type must never override an explicit direction.
    assert count_inbound_messages([{"direction": "outbound", "type": "REPLY"}]) == 0
    assert count_inbound_messages([{"direction": "INBOUND", "type": "SENT"}]) == 1


def test_campaign_send_alone_is_not_a_reply():
    assert count_inbound_messages([{"direction": "outbound"}]) == 0


def test_malformed_threads_never_raise():
    assert count_inbound_messages(None) == 0
    assert count_inbound_messages("[]") == 0
    assert count_inbound_messages([None, "junk", {}]) == 0


# ── the SQL ──────────────────────────────────────────────────────────────────

def test_reply_count_sql_filters_on_inbound_and_floors_at_stored_reply():
    sql = reply_count_sql("co.li_chat_history_cache", "co.li_response_text")
    # Counts elements of the thread blob, not its length.
    assert "jsonb_array_elements(co.li_chat_history_cache)" in sql
    assert "jsonb_array_length" not in sql
    assert "'inbound'" in sql
    # A reply promoted into the column by the webhook counts even when the
    # thread blob has not been fetched yet.
    assert "co.li_response_text" in sql
    assert sql.strip().startswith("GREATEST(")


# ── /outreach/status/{role_id} ───────────────────────────────────────────────

def _status_row(**overrides):
    row = {
        "candidate_id": 101,
        "status": "sent",
        "message_sent_count": 1,
        "last_message_sent_at": None,
        "response_received_at": None,
        "response_text": None,
        "li_status": "sent",
        "li_last_action_at": None,
        "li_response_text": None,
        "li_sent_count": 1,
        "li_response_received_at": None,
        "li_conversation_id": "conv-1",
        "email_cached_message_count": 0,
        "li_cached_message_count": 1,
        "response_read_at": None,
        "li_response_read_at": None,
        "email_reply_count": 0,
        "li_reply_count": 0,
    }
    row.update(overrides)
    return tuple(row.values())


def _statuses(monkeypatch, rows):
    connection = _Connection(rows)
    monkeypatch.setattr(
        outreach, "get_db_connection_context", lambda **kwargs: connection
    )
    result = asyncio.run(
        outreach.get_outreach_status(
            89, current_user=schemas.User(id=7, username="recruiter", role="recruiter")
        )
    )
    return result, connection


def test_status_reports_zero_replies_for_a_candidate_we_only_messaged(monkeypatch):
    statuses, _ = _statuses(monkeypatch, [_status_row()])

    assert statuses[101]["message_count"] == 2   # thread exists…
    assert statuses[101]["reply_count"] == 0     # …but they never answered


def test_status_sums_replies_across_both_channels(monkeypatch):
    statuses, _ = _statuses(
        monkeypatch, [_status_row(email_reply_count=1, li_reply_count=3)]
    )

    assert statuses[101]["email_reply_count"] == 1
    assert statuses[101]["li_reply_count"] == 3
    assert statuses[101]["reply_count"] == 4


def test_status_query_counts_inbound_messages(monkeypatch):
    _, connection = _statuses(monkeypatch, [_status_row()])
    sql = connection.cursors[0].executed[0]

    assert "AS email_reply_count" in sql
    assert "AS li_reply_count" in sql
    assert "'inbound'" in sql


# ── /roles/{role_name} ───────────────────────────────────────────────────────

def _role_row(email_replies=0, li_replies=0, message_count=2):
    # 42 columns; the reply counts are appended LAST so the existing positional
    # indexes in the row mapper keep pointing at the same fields.
    return (
        44, "Enterprise AE", "Role description", 7,
        "Recruiter", "recruiter@example.com",
        1, None, 1,
        101, "High", "Strong fit",
        "Aadarsh Goyal", "https://linkedin.com/in/aadarsh", "Bengaluru, India",
        "Account Executive", "Profile summary",
        "aadarsh@example.com", "+919999999999", "Followup / In conversation",
        "Aadarsh", "Goyal", "Bengaluru",
        8.4, 2.1, "Call next Tuesday", "",
        "Enterprise Account Executive", "Example Corp",
        "", "sent", 1, "sent", 1, "conv-1",
        0, message_count, message_count,
        None,
        email_replies, li_replies, email_replies + li_replies,
    )


def _role_details(monkeypatch, row):
    connection = _Connection([row])
    monkeypatch.setattr(roles, "get_db_connection", lambda: connection)
    monkeypatch.setattr(roles, "return_db_connection", lambda _connection: None)
    monkeypatch.setattr(roles, "fetch_role_activation", lambda _cursor, _role_id: {})
    monkeypatch.setattr(roles, "PROFILES_BY_ID", {})
    monkeypatch.setattr(
        "backend.services.resume_service.fetch_resume_metas", lambda ids: {}
    )
    roles.invalidate_role_detail_cache()
    result = asyncio.run(
        roles.get_role(
            "Enterprise AE",
            current_user=schemas.User(id=7, username="recruiter", role="recruiter"),
        )
    )
    roles.invalidate_role_detail_cache()
    return result, connection


def test_role_details_expose_reply_counts_separate_from_message_counts(monkeypatch):
    result, _ = _role_details(monkeypatch, _role_row(email_replies=1, li_replies=2))
    candidate = result["candidates"][0]

    assert candidate["message_count"] == 2
    assert candidate["email_reply_count"] == 1
    assert candidate["li_reply_count"] == 2
    assert candidate["reply_count"] == 3
    # The pre-existing positional fields must not have shifted.
    assert candidate["first_name"] == "Aadarsh"
    assert candidate["status"] == "Followup / In conversation"
    assert candidate["li_conversation_id"] == "conv-1"


def test_role_details_report_no_replies_when_only_we_have_written(monkeypatch):
    result, _ = _role_details(monkeypatch, _role_row())

    assert result["candidates"][0]["reply_count"] == 0


def test_role_detail_query_counts_inbound_messages(monkeypatch):
    _, connection = _role_details(monkeypatch, _role_row())
    sql = connection.cursors[0].executed[0]

    assert "AS email_reply_count" in sql
    assert "AS li_reply_count" in sql
    assert "'inbound'" in sql
    # The two blob scans are expensive enough that the total is summed from
    # their aliases rather than re-running both subqueries.
    assert sql.count("jsonb_array_elements") == 2


# ── the JSX that renders it ──────────────────────────────────────────────────
# Python is the only thing that runs automatically in this repo, so the badge's
# contract is asserted against the source, as in outreach_sync_responses_test.

def test_badge_renders_the_reply_count():
    badge = re.search(r"responseState\.replyCount == null(.|\n)*?: null\}", ROLES_JSX)
    assert badge, "conversation badge should key off replyCount"
    assert "responseState.messageCount" not in ROLES_JSX


def test_responded_tab_does_not_treat_an_open_thread_as_a_response():
    has_response = re.search(r"hasResponse: (?P<expr>.*),", ROLES_JSX).group("expr")

    assert "resolvedReplyCount" in has_response
    # Both of these are true the moment WE send, never mind the candidate.
    assert "li_conversation_id" not in has_response
    assert "messageCount" not in has_response

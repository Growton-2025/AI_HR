"""LinkedIn threads must converge on HeyReach's thread, not on whatever is longest.

A candidate's reply showed up fourteen times. Three things conspired: the
reply webhook was registered four times (every reply delivered 4x, then
retried for a day), the handler echoed each delivery into the stored thread
with no idempotency, and every sync path kept whichever thread was LONGER —
so a clean provider fetch could never replace the bloated one. These tests
pin the content merge that replaced "never shrink", the atomic idempotent
webhook echo, and the self-healing webhook registration.
"""

import inspect
import pathlib
import re

from backend.api.routes import outreach
from backend.integrations import heyreach
from backend.services import heyreach_reply_sync

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTREACH_SRC = (ROOT / "backend" / "api" / "routes" / "outreach.py").read_text()
POLLER_SRC = (ROOT / "backend" / "services" / "heyreach_reply_sync.py").read_text()


def _real(direction, body, t, mid="r1"):
    return {"id": mid, "type": "REPLY" if direction == "inbound" else "SENT",
            "direction": direction, "email_body": body, "time": t, "sender_name": "x"}


def _echo(direction, body, t):
    return {"id": f"local-{t}", "type": "REPLY" if direction == "inbound" else "SENT",
            "direction": direction, "email_body": body, "time": t,
            "sender_name": "x", "local_echo": True}


# ── the merge ────────────────────────────────────────────────────────────────

def test_shorter_clean_fetch_replaces_bloated_store():
    stored = [_real("outbound", "Hi", "2026-08-30T04:00:00.000Z")]
    stored += [_echo("inbound", "can you tell me", f"2026-08-30T07:0{i}:00.000000+00:00") for i in range(1, 6)]
    fetched = [_real("outbound", "Hi", "2026-08-30T04:00:00.000Z"),
               _real("inbound", "can you tell me", "2026-08-30T07:01:00.238Z")]
    merged = outreach._merge_li_thread(stored, fetched)
    assert [m["email_body"] for m in merged] == ["Hi", "can you tell me"]
    assert not any(outreach._is_echo(m) for m in merged)


def test_stored_real_missing_from_fetch_is_carried_forward_in_time_order():
    stored = [_real("outbound", "a", "2026-08-30T04:00:00.000Z"),
              _real("inbound", "b", "2026-08-30T05:00:00.000Z", "lagging")]
    fetched = [_real("outbound", "a", "2026-08-30T04:00:00.000Z"),
               _real("inbound", "c", "2026-08-30T06:00:00.000Z")]
    merged = outreach._merge_li_thread(stored, fetched)
    assert [m["email_body"] for m in merged] == ["a", "b", "c"]


def test_echo_dropped_when_provider_copy_differs_only_by_markup():
    stored = [_echo("outbound", "Hi Mayank,\n\nthanks", "2026-08-31T06:01:23.000000+00:00")]
    fetched = [_real("outbound", "Hi Mayank,<br><br>thanks", "2026-08-31T06:01:24.474Z")]
    merged = outreach._merge_li_thread(stored, fetched)
    assert len(merged) == 1 and not outreach._is_echo(merged[0])


def test_surviving_echo_is_placed_by_time_not_appended_last():
    stored = [_echo("outbound", "just sent", "2026-08-30T05:00:00.000000+00:00")]
    fetched = [_real("inbound", "earlier", "2026-08-30T04:00:00.000Z"),
               _real("inbound", "later", "2026-08-30T06:00:00.000Z")]
    merged = outreach._merge_li_thread(stored, fetched)
    assert [m["email_body"] for m in merged] == ["earlier", "just sent", "later"]


def test_legitimate_repeats_from_provider_are_preserved():
    fetched = [_real("outbound", "hello", "2026-03-27T10:30:00.000Z"),
               _real("inbound", "hi", "2026-03-27T10:31:00.000Z"),
               _real("outbound", "how are you", "2026-03-29T16:00:00.000Z"),
               _real("inbound", "hi", "2026-03-29T16:02:00.000Z"),
               _real("outbound", "hello", "2026-04-02T09:06:00.000Z")]
    merged = outreach._merge_li_thread(fetched, fetched)
    assert [m["email_body"] for m in merged] == ["hello", "hi", "how are you", "hi", "hello"]


def test_stored_only_rule_keeps_reals_and_drops_duplicate_echoes():
    stored = [_real("inbound", "Hi Ashwin", "2026-08-19T13:08:48.763Z"),
              _real("inbound", "call me", "2026-08-19T13:09:03.488Z"),
              _echo("inbound", "Hi Ashwin", "2026-08-23T12:36:50.287292+00:00"),
              _echo("inbound", "new echo", "2026-08-24T12:36:50.287292+00:00"),
              _echo("inbound", "new echo", "2026-08-25T12:36:50.287292+00:00")]
    merged = outreach._merge_li_thread(stored, [])
    assert [m["email_body"] for m in merged] == ["Hi Ashwin", "call me", "new echo"]


def test_mixed_timestamp_formats_sort_chronologically():
    msgs = [{"time": "2026-08-30T07:02:31.093916+00:00", "email_body": "b", "direction": "inbound"},
            {"time": "2026-08-30T07:01:00.238Z", "email_body": "a", "direction": "inbound"},
            {"time": "2026-08-31T06:01:24.474Z", "email_body": "c", "direction": "outbound"}]
    assert [m["email_body"] for m in sorted(msgs, key=outreach._msg_time_key)] == ["a", "b", "c"]


def test_utc_iso_z_renders_provider_format():
    assert outreach._utc_iso_z("2026-08-30T07:02:31.093916+00:00") == "2026-08-30T07:02:31.093Z"
    assert outreach._utc_iso_z("2026-08-30T12:32:31+05:30") == "2026-08-30T07:02:31.000Z"
    assert outreach._utc_iso_z(None) is None


# ── the writers ──────────────────────────────────────────────────────────────

def test_never_shrink_guard_is_gone():
    pattern = r"jsonb_array_length\(li_chat_history_cache\),\s*0\)\s*<="
    assert not re.search(pattern, OUTREACH_SRC)
    assert not re.search(pattern, POLLER_SRC)
    assert "_merge_li_thread" in POLLER_SRC


def test_sync_and_poller_lock_the_row_before_merging():
    assert OUTREACH_SRC.count("WHERE candidate_id = %s FOR UPDATE") >= 2
    assert "FOR UPDATE" in POLLER_SRC


def test_send_echo_is_guarded_against_double_submit():
    src = inspect.getsource(outreach._echo_sent_li_message)
    assert "NOT EXISTS" in src and "interval '2 minutes'" in src


# ── the webhook ──────────────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sql = sql
        self.conn.executed.append(sql)

    def fetchone(self):
        if "SELECT candidate_id FROM candidate_outreach" in self.sql:
            return (13918,)
        return None

    def fetchall(self):
        if "FOR UPDATE" in self.sql:
            return [(list(self.conn.blob),)]
        return []


class _Connection:
    def __init__(self, blob):
        self.blob = blob
        self.executed = []

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


PAYLOAD = {
    "event_type": "EVERY_MESSAGE_REPLY_RECEIVED",
    "correlation_id": "abc",
    "conversation_id": "conv-1",
    "timestamp": "2026-08-30T07:05:00Z",
    "recent_messages": [{"message": "can you tell me", "creation_time": "2026-08-30T07:01:00.238Z", "is_reply": True}],
}


def _deliver(monkeypatch, blob):
    conn = _Connection(blob)
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: conn)
    outreach._li_chat_cache.pop(13918, None)
    res = outreach.heyreach_webhook(dict(PAYLOAD))
    assert res["status"] == "success"
    return [s for s in conn.executed if "|| %s::jsonb" in s]


def test_webhook_is_sync_and_appends_once(monkeypatch):
    assert not inspect.iscoroutinefunction(outreach.heyreach_webhook)
    assert len(_deliver(monkeypatch, [])) == 1
    already = [{"direction": "inbound", "email_body": "can you tell me", "time": "2026-08-30T07:01:00.238Z"}]
    assert _deliver(monkeypatch, already) == []


def test_webhook_check_then_append_is_locked_and_survives_no_connection(monkeypatch):
    src = inspect.getsource(outreach.heyreach_webhook)
    assert "FOR UPDATE" in src and "already_present = False" in src
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: _Connection([]))

    class _NoConn(_Connection):
        def __enter__(self):
            return None
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: _NoConn([]))
    assert outreach.heyreach_webhook(dict(PAYLOAD))["status"] in ("error", "success")


def test_webhook_echo_uses_the_messages_own_time(monkeypatch):
    conn = _Connection([])
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: conn)
    outreach._li_chat_cache[13918] = {"messages": [], "ts": 0, "refreshing": False}
    outreach.heyreach_webhook(dict(PAYLOAD))
    echo = outreach._li_chat_cache[13918]["messages"][-1]
    assert echo["time"] == "2026-08-30T07:01:00.238Z"
    outreach._li_chat_cache.pop(13918, None)


# ── the registration ─────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _FakeRequests:
    def __init__(self):
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return _Resp()

    def delete(self, url, **kw):
        self.calls.append(("DELETE", url, kw))
        return _Resp()

    def patch(self, url, **kw):
        self.calls.append(("PATCH", url, kw))
        return _Resp()


class _LockConn:
    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return (True,)


def _bot(monkeypatch, hooks):
    import backend.db.connection as dbconn
    monkeypatch.setattr(dbconn, "get_db_connection_context", lambda **kw: _LockConn())
    bot = heyreach.HeyReachBot(api_key="k")
    fake = _FakeRequests()
    monkeypatch.setattr(heyreach, "requests", fake)
    monkeypatch.setattr(bot, "list_webhooks", lambda: hooks)
    return bot, fake


URL = "https://backend/api/outreach/heyreach/webhook"


def test_duplicate_registrations_are_deleted_keeping_the_oldest(monkeypatch):
    hooks = [
        {"id": 67737, "webhookUrl": URL, "eventType": "EVERY_MESSAGE_REPLY_RECEIVED", "isActive": True},
        {"id": 67734, "webhookUrl": URL, "eventType": "EVERY_MESSAGE_REPLY_RECEIVED", "isActive": True},
        {"id": 67735, "webhookUrl": URL, "eventType": "EVERY_MESSAGE_REPLY_RECEIVED", "isActive": True},
        {"id": 24358, "webhookUrl": URL, "eventType": "MESSAGE_REPLY_RECEIVED", "isActive": True},
        {"id": 12266, "webhookUrl": "https://api.clay.com/x", "eventType": "EVERY_MESSAGE_REPLY_RECEIVED", "isActive": True},
    ]
    bot, fake = _bot(monkeypatch, hooks)
    assert bot.ensure_reply_webhook(URL) is True
    deletes = sorted(c[2]["params"]["webhookId"] for c in fake.calls if c[0] == "DELETE")
    assert deletes == [67735, 67737]
    assert not any(c[0] == "POST" for c in fake.calls)


def test_missing_registration_is_created_once_outside_the_retrying_session(monkeypatch):
    bot, fake = _bot(monkeypatch, [])
    bot._session = None  # any use of the retrying session would blow up
    assert bot.ensure_reply_webhook(URL) is True
    posts = [c for c in fake.calls if c[0] == "POST"]
    assert len(posts) == 1 and posts[0][1].endswith("/CreateWebhook")
    assert posts[0][2]["json"]["webhookUrl"] == URL


def test_registration_skips_when_another_worker_holds_the_lock(monkeypatch):
    class _Busy(_LockConn):
        def fetchone(self):
            return (False,)
    import backend.db.connection as dbconn
    monkeypatch.setattr(dbconn, "get_db_connection_context", lambda **kw: _Busy())
    bot = heyreach.HeyReachBot(api_key="k")
    fake = _FakeRequests()
    monkeypatch.setattr(heyreach, "requests", fake)
    bot._session = None
    assert bot.ensure_reply_webhook(URL) is True
    assert fake.calls == []

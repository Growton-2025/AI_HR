"""The reply preview under "Open conversation" froze at the first reply.

Candidate 2519 on role 86 replied on Sep 1 and again on Sep 26. The Sep 26
reply was in the cached thread (the badge counted it) but response_text still
showed the Sep 1 text, because:

* response_text is written only by the Smartlead reply poller, and the poller
  skipped any thread whose cached length already matched Smartlead's;
* the conversation modal's own refresher persists the full thread every 60s
  without touching response_text — which is exactly what made the lengths
  match.

Now the poller promotes the newest inbound reply even when the thread did not
grow, and the modal refresher promotes it too, so whichever path fetched the
thread updates the list.
"""

import json

from backend.api.routes import outreach
from backend.services import smartlead_reply_sync as poller


class _Cursor:
    def __init__(self):
        self.executed = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def updates(self, column):
        return [(sql, params) for sql, params in self.executed if "UPDATE candidate_outreach" in sql and column in sql]


class _Connection:
    def __init__(self):
        self.cur = _Cursor()
        self.committed = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _thread():
    return [
        {"message_id": "m1", "type": "SENT", "time": "2026-09-01T10:00:00", "email_body": "Hi, are you open to roles?"},
        {"message_id": "m2", "type": "REPLY", "time": "2026-09-01T10:51:53", "email_body": "Yes, I got your email."},
        {"message_id": "m3", "type": "SENT", "time": "2026-09-26T06:23:58", "email_body": "Great, can we talk today?"},
        {"message_id": "m4", "type": "REPLY", "time": "2026-09-26T06:24:12", "email_body": "Sure, call me after 3pm."},
    ]


class _Bot:
    def __init__(self, api_key=None, messages=None):
        self.campaign_id = None
        self._messages = messages if messages is not None else _thread()

    def get_chat_history(self, email, campaign_id=None):
        return [dict(m) for m in self._messages]


def _run_poll(monkeypatch, rows, messages=None):
    connection = _Connection()
    monkeypatch.setenv("SMARTLEAD_API_KEY", "test")
    import backend.db.connection as dbconn
    monkeypatch.setattr(dbconn, "get_db_connection_context", lambda **kw: connection)
    import backend.integrations.smartlead as smartlead
    monkeypatch.setattr(smartlead, "SmartleadBot", lambda api_key=None: _Bot(api_key, messages))
    monkeypatch.setattr(poller, "_batch_for_cycle", lambda cur: (rows, 0))
    monkeypatch.setattr(poller, "_backfill_send_stamps", lambda pairs: None)
    monkeypatch.setattr(poller, "_capture_phone", lambda candidate_id, text: None)
    monkeypatch.setattr(outreach, "_clean_email_body", lambda body: body)
    promoted = poller.poll_once()
    return promoted, connection.cur


# (candidate_id, email, campaign_id, cached_len, last_sent_at)
ROW_CACHE_CURRENT = (2519, "n@example.com", "3874679", 4, "2026-09-26T06:23:58")
ROW_CACHE_STALE = (2519, "n@example.com", "3874679", 2, "2026-09-26T06:23:58")


def test_poller_promotes_newest_reply_when_cache_is_already_current(monkeypatch):
    promoted, cur = _run_poll(monkeypatch, [ROW_CACHE_CURRENT])
    assert promoted == 1
    assert cur.updates("email_chat_history_cache") == [], "thread did not grow, so the cache is not rewritten"
    (sql, params), = cur.updates("response_text")
    assert params[0] == "Sure, call me after 3pm."
    assert params[1] == "2026-09-26T06:24:12"
    assert params[2] == 2519


def test_poller_still_stores_the_thread_when_it_grew(monkeypatch):
    promoted, cur = _run_poll(monkeypatch, [ROW_CACHE_STALE])
    assert promoted == 1
    (sql, params), = cur.updates("email_chat_history_cache")
    assert [m["message_id"] for m in json.loads(params[0])] == ["m1", "m2", "m3", "m4"]
    (sql, params), = cur.updates("response_text")
    assert params[0] == "Sure, call me after 3pm."


def test_poller_does_nothing_for_a_thread_with_no_reply(monkeypatch):
    only_ours = [m for m in _thread() if m["type"] == "SENT"]
    promoted, cur = _run_poll(monkeypatch, [(2519, "n@example.com", "3874679", 2, None)], messages=only_ours)
    assert promoted == 0
    assert cur.executed == []


def test_promote_reply_is_a_no_op_when_text_is_unchanged():
    cur = _Cursor()
    cur.rowcount = 0
    assert poller._promote_reply(cur, 2519, {"reply_text": "Yes, I got your email.", "reply_at": None}) is False
    (sql, params), = cur.executed
    assert "IS DISTINCT FROM" in sql


def test_latest_inbound_picks_the_newest_candidate_message():
    latest = poller._latest_inbound(_thread())
    assert latest == {"reply_text": "Sure, call me after 3pm.", "reply_at": "2026-09-26T06:24:12"}


def test_modal_refresh_promotes_the_reply_into_the_list(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: connection)
    monkeypatch.setattr(outreach, "get_smartlead_bot", lambda: _Bot())
    monkeypatch.setattr(outreach, "_clean_email_body", lambda body: body)
    invalidated = []
    import backend.api.routes.browse as browse
    monkeypatch.setattr(browse, "_invalidate_browse_cache", lambda: invalidated.append(True))
    with outreach._email_chat_lock:
        outreach._email_chat_cache.pop(2519, None)

    messages = outreach._sync_email_messages(2519, "n@example.com", "3874679")

    assert [m["message_id"] for m in messages] == ["m1", "m2", "m3", "m4"]
    (sql, params), = connection.cur.updates("email_chat_history_cache")
    assert params[2] == 2519
    (sql, params), = connection.cur.updates("response_text")
    assert params[0] == "Sure, call me after 3pm."
    assert params[2] == 2519
    assert connection.committed == 1
    assert invalidated == [True]
    with outreach._email_chat_lock:
        outreach._email_chat_cache.pop(2519, None)


def test_modal_refresh_with_nothing_fetched_promotes_nothing(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(outreach, "get_db_connection_context", lambda **kw: connection)
    monkeypatch.setattr(outreach, "get_smartlead_bot", lambda: _Bot(messages=[]))
    with outreach._email_chat_lock:
        outreach._email_chat_cache.pop(2519, None)
    assert outreach._sync_email_messages(2519, "n@example.com", "3874679") == []
    assert connection.cur.executed == []

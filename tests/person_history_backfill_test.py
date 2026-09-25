"""Provider history backfill (docs/candidate-history-linking-plan.md PR 3):
queue semantics, rate-limit backoff, provider fetch shapes, timeline merge."""
import datetime as dt

import pytest

from backend.services import person_history_backfill as bf


class _Cur:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fresh_schema_flag(monkeypatch):
    monkeypatch.setattr(bf, "_schema_ready", True)   # no DDL noise in the SQL under test


# ── flag ───────────────────────────────────────────────────────────────────

def test_disabled_by_default_enqueues_nothing(monkeypatch):
    monkeypatch.delenv(bf.FLAG, raising=False)
    cur = _Cur()
    assert bf.enqueue_backfill(cur, "/in/jane", linkedin_url="https://linkedin.com/in/jane", email="j@x.com") == []
    assert cur.executed == []


def test_enqueues_one_job_per_provider_we_have_an_identity_and_a_key_for(monkeypatch):
    monkeypatch.setenv(bf.FLAG, "true")
    monkeypatch.setenv("HEYREACH_API_KEY", "k")
    monkeypatch.setenv("SMARTLEAD_API_KEY", "k")
    cur = _Cur()
    assert bf.enqueue_backfill(cur, "/in/jane", email="Jane@X.com") == ["heyreach", "smartlead"]
    inserts = [(sql, p) for sql, p in cur.executed if "INSERT INTO person_history_jobs" in sql]
    assert [p[1] for _, p in inserts] == ["heyreach", "smartlead"]
    assert inserts[0][1][2] == "https://www.linkedin.com/in/jane"     # URL rebuilt from the key
    assert inserts[1][1][2] == "jane@x.com"
    assert "ON CONFLICT (person_key, provider) DO UPDATE" in inserts[0][0]
    assert "status = 'failed'" in inserts[0][0]                          # only re-queues failed/stale jobs


def test_no_provider_key_means_no_job(monkeypatch):
    monkeypatch.setenv(bf.FLAG, "true")
    monkeypatch.delenv("HEYREACH_API_KEY", raising=False)
    monkeypatch.delenv("SMARTLEAD_API_KEY", raising=False)
    assert bf.enqueue_backfill(_Cur(), "/in/jane", email="j@x.com") == []


# ── queue ──────────────────────────────────────────────────────────────────

def test_claim_takes_one_due_job_with_skip_locked():
    cur = _Cur(rows=[(7, "/in/jane", "heyreach", "https://linkedin.com/in/jane", 1)])
    job = bf.claim_job(cur)
    sql, _ = cur.executed[0]
    assert "FOR UPDATE SKIP LOCKED" in sql and "LIMIT 1" in sql
    assert "status = 'running'" in sql                      # a dead worker's job becomes due again
    assert job == {"id": 7, "person_key": "/in/jane", "provider": "heyreach",
                   "identity": "https://linkedin.com/in/jane", "attempts": 1}
    assert bf.claim_job(_Cur(rows=[])) is None


def test_failures_back_off_exponentially_then_give_up():
    assert [bf.backoff_minutes(a) for a in (1, 2, 3, 4)] == [2, 4, 8, 16]
    cur = _Cur()
    assert bf.fail_job(cur, 7, attempts=2, error="429") == "queued"
    assert cur.executed[-1][1][1] == 4                       # minutes until the retry
    assert bf.fail_job(_Cur(), 7, attempts=bf.MAX_ATTEMPTS, error="429") == "failed"
    assert bf.fail_job(_Cur(), 7, attempts=1, error="404", retryable=False) == "failed"


# ── fetchers ───────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_smartlead_walks_every_campaign_of_the_lead_within_the_window(monkeypatch):
    monkeypatch.setenv("SMARTLEAD_API_KEY", "k")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        if url.endswith("/api/v1/leads/"):
            return _Resp(200, {"id": 99, "email": "j@x.com",
                               "lead_campaign_data": [{"campaign_id": 11}, {"campaign_id": 12}]})
        if "/campaigns/11/" in url:
            return _Resp(200, {"history": [{"type": "SENT", "time": "2026-09-01T10:00:00Z", "email_body": "hi"}]})
        return _Resp(200, {"history": []})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    threads = bf.fetch_smartlead("j@x.com")
    assert threads == [("campaign:11", [{"type": "SENT", "time": "2026-09-01T10:00:00Z", "email_body": "hi"}])]
    assert [u for u, _ in calls][1:] == [
        "https://server.smartlead.ai/api/v1/campaigns/11/leads/99/message-history",
        "https://server.smartlead.ai/api/v1/campaigns/12/leads/99/message-history",
    ]
    assert "event_time_gt" in calls[1][1]                    # twelve-month window, not all time


def test_smartlead_rate_limit_is_retryable(monkeypatch):
    monkeypatch.setenv("SMARTLEAD_API_KEY", "k")
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(429, {}))
    with pytest.raises(bf.ProviderError) as exc:
        bf.fetch_smartlead("j@x.com")
    assert exc.value.retryable is True


def test_heyreach_returns_the_conversation_thread_within_the_window(monkeypatch):
    class FakeBot:
        def get_li_chat_history(self, url):
            return {"conversation_id": "c1", "account_id": 5, "messages": [
                {"id": "old", "time": "2020-01-01T00:00:00Z", "type": "SENT", "email_body": "ancient"},
                {"id": "new", "time": "2026-09-01T00:00:00Z", "type": "REPLY", "email_body": "hello"},
            ]}
    import backend.integrations.heyreach as hr
    monkeypatch.setattr(hr, "HeyReachBot", FakeBot)
    assert bf.fetch_heyreach("https://linkedin.com/in/jane") == [("c1", [
        {"id": "new", "time": "2026-09-01T00:00:00Z", "type": "REPLY", "email_body": "hello"}])]


# ── worker run: claim -> fetch -> store -> done, or fail with backoff ──────

class _Conn:
    def __init__(self, cur):
        self._cur = cur
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_run_one_stores_threads_and_marks_done(monkeypatch):
    cur = _Cur(rows=[(7, "/in/jane", "heyreach", "u", 1)])
    monkeypatch.setitem(bf.FETCHERS, "heyreach", lambda ident: [("c1", [{"id": "m1"}])])
    conn = _Conn(cur)
    assert bf.run_one(conn) == "done"
    sqls = [s for s, _ in cur.executed]
    assert any("INSERT INTO person_provider_threads" in s for s in sqls)
    assert any("status = 'done'" in s for s in sqls)
    assert conn.commits >= 2


def test_run_one_backs_off_when_the_provider_fails(monkeypatch):
    cur = _Cur(rows=[(7, "/in/jane", "smartlead", "j@x.com", 1)])

    def boom(ident):
        raise bf.ProviderError("rate limit", retryable=True)
    monkeypatch.setitem(bf.FETCHERS, "smartlead", boom)
    conn = _Conn(cur)
    assert bf.run_one(conn) == "queued"
    assert conn.rollbacks == 1
    assert any("status = 'queued'" in s for s, _ in cur.executed)


# ── timeline merge ─────────────────────────────────────────────────────────

def test_timeline_merges_provider_threads_and_reports_pending(monkeypatch):
    from backend.services import person_identity as pi, person_timeline as pt
    pi._table_cache.clear()

    class Cur(_Cur):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            s = " ".join(sql.split())
            if "to_regclass" in s: self.rows = [(True,)]
            elif "WHERE c.id = %s" in s: self.rows = [(1, None, None, "Jane", "legacy_master", False, "/in/jane", None, None, None)]
            elif "WHERE (c.normalized_linkedin" in s: self.rows = [(1, None, None, "Jane", "legacy_master", False, "/in/jane", None, None, None)]
            elif "FROM candidate_outreach" in s:
                self.rows = [(1, None, None, [{"id": "m1", "time": "2026-09-10T10:00:00Z", "type": "REPLY", "email_body": "local copy"}], None)]
            elif "FROM person_provider_threads" in s:
                self.rows = [("heyreach", "c1", [
                    {"id": "m1", "time": "2026-09-10T10:00:00Z", "type": "REPLY", "email_body": "local copy"},   # duplicate of ours
                    {"id": "m0", "time": "2026-03-01T10:00:00Z", "type": "SENT", "email_body": "older, only at HeyReach"},
                ])]
            elif "FROM person_history_jobs" in s: self.rows = [("smartlead",)]
            else: self.rows = []

    out = pt.build_timeline(Cur(), 1, viewer_email="x@y", viewer_is_admin=True)
    ids = [i["id"] for i in out["items"] if i["type"] == "linkedin_message"]
    assert ids == ["linkedin:m1", "linkedin:m0"]              # deduped, newest first
    assert out["items"][-1]["source"] == "heyreach"
    assert out["pending_backfill"] == ["smartlead"]

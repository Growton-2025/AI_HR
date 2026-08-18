"""The recording player must not wait on transcription.

Plivo hands us the recording URL in its final callback, seconds after hangup.
That URL was only written to calls.recording_url at the very END of
process_call_insights — in the same UPDATE as the transcript, summary and
sentiment — so the audio player the recruiter is watching for was gated behind
a download, a transcription and two LLM passes. Measured on real calls: 8s for
a 25-second call, 101s for an 18-minute one, 202s for a 22-minute one, all
while we had been holding the link since the moment the call ended.

Two aggravators went with it. The URL lived in a per-process dict, so a poll
served by a sibling gunicorn worker could not see it; and the poll endpoint
awaited the whole pipeline inline while the browser abandoned each request
after 15s and fired another 5s later, stacking paid transcriptions of the same
audio.
"""

import asyncio

import pytest

from backend.api.routes import calls as calls_route
from backend.integrations import plivo_service as ps


class _Cursor:
    def __init__(self, rows=None, rowcount=1):
        self.rows = list(rows or [])
        self.executed = []
        self.rowcount = rowcount

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        return self.rows

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def _patch_db(monkeypatch, cursor):
    import backend.db.connection as dbc

    conn = _Conn(cursor)
    monkeypatch.setattr(dbc, "get_db_connection", lambda **kw: conn)
    monkeypatch.setattr(dbc, "return_db_connection", lambda _c: None)
    return conn


# ── the URL reaches the row, not just one worker's memory ───────────────────

def test_recording_url_is_written_to_the_row(monkeypatch):
    cursor = _Cursor(rowcount=1)
    conn = _patch_db(monkeypatch, cursor)

    assert ps.persist_recording_url("uuid-1", "https://plivo/rec.mp3") is True
    sql, params = cursor.executed[0]

    assert "UPDATE calls" in sql
    assert "SET recording_url = %s" in sql
    # Matched on either UUID column, the same way the insights writer does.
    assert "plivo_call_uuid = %s OR plivo_transaction_id = %s" in sql
    assert params[0] == "https://plivo/rec.mp3"
    assert conn.committed
    # Crucially it does NOT touch the artifacts — those arrive much later.
    assert "transcript" not in sql
    assert "summary" not in sql


def test_storing_the_same_url_twice_does_not_churn_the_row(monkeypatch):
    cursor = _Cursor(rowcount=0)
    _patch_db(monkeypatch, cursor)

    assert ps.persist_recording_url("uuid-1", "https://plivo/rec.mp3") is False
    sql, _ = cursor.executed[0]

    assert "COALESCE(recording_url, '') <> %s" in sql


@pytest.mark.parametrize("call_uuid,url", [(None, "u"), ("uuid", None), ("", "")])
def test_nothing_to_store_costs_no_query(monkeypatch, call_uuid, url):
    cursor = _Cursor()
    _patch_db(monkeypatch, cursor)

    assert ps.persist_recording_url(call_uuid, url) is False
    assert cursor.executed == []


# ── one transcription run, across workers ───────────────────────────────────

def test_first_caller_wins_the_insights_claim(monkeypatch):
    cursor = _Cursor(rowcount=1)
    _patch_db(monkeypatch, cursor)

    assert ps.claim_insights_run("uuid-1") is True
    sql, params = cursor.executed[0]

    assert "SET recording_synced_at = NOW()" in sql
    assert "recording_synced_at IS NULL" in sql
    assert ps.INSIGHTS_CLAIM_STALE_SECONDS in params


def test_a_second_caller_does_not_start_another_run(monkeypatch):
    cursor = _Cursor(rowcount=0)
    _patch_db(monkeypatch, cursor)

    assert ps.claim_insights_run("uuid-1") is False


def test_a_stale_claim_can_be_retried(monkeypatch):
    # A worker killed mid-transcription must not block the call forever.
    cursor = _Cursor(rowcount=1)
    _patch_db(monkeypatch, cursor)
    ps.claim_insights_run("uuid-1")
    sql, _ = cursor.executed[0]

    assert "recording_synced_at < NOW() - (%s * INTERVAL '1 second')" in sql
    assert 0 < ps.INSIGHTS_CLAIM_STALE_SECONDS <= 30 * 60


def test_no_database_means_no_claim(monkeypatch):
    import backend.db.connection as dbc

    monkeypatch.setattr(dbc, "get_db_connection", lambda **kw: None)

    # Nothing could be stored anyway, and the poll repeats every couple of
    # seconds — granting it here would start a transcription on every one.
    assert ps.claim_insights_run("uuid-1") is False


def test_an_unexpected_query_failure_falls_open(monkeypatch):
    class _Boom(_Cursor):
        def execute(self, sql, params=None):
            raise RuntimeError("connection reset")

    _patch_db(monkeypatch, _Boom())

    # Rare and usually transient: a duplicate run beats a lost transcript.
    assert ps.claim_insights_run("uuid-1") is True


# ── the poll returns without waiting for transcription ──────────────────────

def _sync(monkeypatch, call_row, *, claimed=True, lookup=None):
    """Run the poll endpoint and report what it did, and when.

    `finished_on_return` is the crux: the transcription must NOT have completed
    by the time the endpoint answers, or we are back to holding a worker for
    the whole pipeline.
    """
    started, finished, stored = [], [], []

    async def _fake_insights(call_uuid, record_url, **kwargs):
        started.append((call_uuid, record_url))
        await asyncio.sleep(0.05)          # stands in for download + ASR + LLM
        finished.append(call_uuid)

    monkeypatch.setattr(calls_route, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls_route, "get_calls_db_connection", lambda: _Conn(_Cursor()))
    monkeypatch.setattr(calls_route, "return_db_connection", lambda _c: None)
    monkeypatch.setattr(calls_route, "fetch_call_by_id", lambda cur, cid, owner=None: call_row)
    monkeypatch.setattr(calls_route, "invalidate_calls_cache", lambda: None)
    monkeypatch.setattr(ps, "process_call_insights", _fake_insights)
    monkeypatch.setattr(ps, "claim_insights_run", lambda uuid: claimed)
    monkeypatch.setattr(ps, "persist_recording_url",
                        lambda uuid, url: stored.append((uuid, url)) or True)
    monkeypatch.setattr(ps, "lookup_recording_url", lambda uuid: lookup)
    monkeypatch.setattr(ps, "recordings", {})

    from backend.api import schemas

    user = schemas.User(id=7, username="rec", email="rec@example.com", role="recruiter")

    async def scenario():
        result = await calls_route.sync_call_recording(1, current_user=user)
        finished_on_return = list(finished)
        await asyncio.sleep(0.2)           # let the queued work run to the end
        return result, finished_on_return

    result, finished_on_return = asyncio.run(scenario())
    return {
        "result": result,
        "started": started,
        "finished_on_return": finished_on_return,
        "finished": finished,
        "stored": stored,
    }


def _row(**overrides):
    row = {
        "id": 1, "plivo_call_uuid": "uuid-1", "plivo_transaction_id": None,
        "recording_url": "https://plivo/rec.mp3", "transcript": None,
        "summary": None, "completed_at": "2026-08-18T10:00:00", "duration": 300,
    }
    row.update(overrides)
    return row


def test_the_poll_returns_the_row_without_awaiting_transcription(monkeypatch):
    run = _sync(monkeypatch, _row())

    # It returns the call — so the player can render — while the expensive work
    # is merely queued. Awaiting it held a worker for the whole pipeline, long
    # after the browser had given up on the request at 15s.
    assert run["result"] is not None
    assert run["finished_on_return"] == []   # answered before transcription ended
    assert run["started"]                    # but the work really was queued
    assert run["finished"]                   # and it ran to completion


def test_the_poll_does_not_start_a_second_transcription(monkeypatch):
    run = _sync(monkeypatch, _row(), claimed=False)

    assert run["result"] is not None
    assert run["started"] == []


def test_a_recovered_url_is_stored_before_any_slow_work(monkeypatch):
    run = _sync(monkeypatch, _row(recording_url=None), lookup="https://plivo/found.mp3")

    assert run["stored"] == [("uuid-1", "https://plivo/found.mp3")]
    assert run["finished_on_return"] == []


def test_finished_calls_short_circuit(monkeypatch):
    run = _sync(monkeypatch, _row(transcript="hello there"))

    assert run["result"]["transcript"] == "hello there"
    assert run["started"] == []

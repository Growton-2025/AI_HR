from contextlib import AbstractContextManager
from datetime import date, datetime

import psycopg2
import pytest
from fastapi import HTTPException

from backend.api import schemas
from backend.api.routes import calls


class _FakeCursor:
    def __init__(self, fetchone_results=None, fetchall_results=None, execute_side_effect=None):
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])
        self._execute_side_effect = execute_side_effect
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self._execute_side_effect:
            raise self._execute_side_effect

    def fetchone(self):
        if self._fetchone_results:
            return self._fetchone_results.pop(0)
        return None

    def fetchall(self):
        if self._fetchall_results:
            return self._fetchall_results.pop(0)
        return []

    def close(self):
        return None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakeConnectionContext(AbstractContextManager):
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_user():
    return schemas.User(
        username="owner@example.com",
        email="owner@example.com",
        full_name="Owner",
    )


def test_initiate_call_returns_structured_error_on_db_failure(monkeypatch):
    db_error = psycopg2.errors.OutOfMemory("oom")
    first_cursor = _FakeCursor(execute_side_effect=db_error)
    first_conn = _FakeConnection(first_cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection_context", lambda **kwargs: _FakeConnectionContext(first_conn))

    with pytest.raises(HTTPException) as exc_info:
        calls.initiate_call(
            calls.CallInitiateRequest(call_id=42, dial_mode="voip"),
            current_user=_build_user(),
        )

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.detail["code"] == "call_lookup_failed"
    assert exc.detail["message"] == "Unable to load this call task right now. Please retry."
    assert exc.detail["action_label"] == "Retry VoIP"
    assert exc.detail["metadata"]["call_id"] == 42
    assert exc.detail["metadata"]["db_error"] == "OutOfMemory"


def test_initiate_call_uses_split_lookup_queries_and_updates_call(monkeypatch):
    first_cursor = _FakeCursor(
        fetchone_results=[
            (2519, 77, "", ""),
            ("owner@example.com",),
            ("Candidate Name", "+918088116167"),
        ]
    )
    first_conn = _FakeConnection(first_cursor)
    second_cursor = _FakeCursor()
    second_conn = _FakeConnection(second_cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection_context", lambda **kwargs: _FakeConnectionContext(first_conn))
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: second_conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: None)
    monkeypatch.setattr(
        calls,
        "fetch_call_by_id",
        lambda cur, call_id, owner=None: {"id": call_id, "plivo_status": "pending", "candidate_id": 2519},
    )

    result = calls.initiate_call(
        calls.CallInitiateRequest(call_id=42, dial_mode="voip", plivo_username="endpoint-user"),
        current_user=_build_user(),
    )

    assert result["success"] is True
    assert result["dial_mode"] == "voip"
    assert ("fre" + "jun_data") not in result
    assert result["plivo_data"]["endpoint_username"] == "endpoint-user"
    assert result["call"]["id"] == 42
    assert first_conn.commits == 1
    assert second_conn.commits == 1

    first_queries = [query for query, _params in first_cursor.executed[:3]]
    assert all("JOIN" not in query.upper() for query in first_queries)
    assert "FROM calls" in first_queries[0]
    assert "FROM call_lists" in first_queries[1]
    assert "FROM candidates" in first_queries[2]


def test_add_candidates_duplicate_returns_400_without_wrapping(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(1,), (1,)])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.add_candidates_to_list(
            calls.AddCandidatesRequest(candidate_ids=[101], list_id=7),
            current_user=_build_user(),
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.detail == "Candidate is already in this call list"
    assert conn.rollbacks == 1


def test_add_candidates_success_invalidates_cache(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(1,), (0,)], fetchall_results=[[(10,), (11,)]])
    conn = _FakeConnection(cursor)
    invalidated = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda conn: invalidated.append(conn))

    result = calls.add_candidates_to_list(
        calls.AddCandidatesRequest(candidate_ids=[101, 102], list_id=7),
        current_user=_build_user(),
    )

    assert result == {"success": True, "added_count": 2}
    assert conn.commits == 1
    assert invalidated == [conn]


def test_delete_missing_call_returns_404(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[None])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.delete_call(42, current_user=_build_user())

    exc = exc_info.value
    assert exc.status_code == 404
    assert exc.detail == "Call task not found"
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_delete_missing_call_list_returns_404(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(3,), None])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.delete_call_list(7, current_user=_build_user())

    exc = exc_info.value
    assert exc.status_code == 404
    assert exc.detail == "Call list not found"
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_delete_call_list_returns_deleted_call_count_and_invalidates(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(3,), (7,)])
    conn = _FakeConnection(cursor)
    invalidated = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda conn: invalidated.append(conn))

    result = calls.delete_call_list(7, current_user=_build_user())

    assert result == {"success": True, "deleted_call_count": 3}
    assert conn.commits == 1
    assert invalidated == [conn]


def test_warm_call_caches_publishes_one_consistent_snapshot(monkeypatch):
    cursor = _FakeCursor(
        fetchall_results=[
            [(7, "Customer Marketing - Locad", datetime(2026, 7, 1, 10, 0, 0), "owner@example.com")],
            [(
                42,
                101,
                7,
                "pending",
                None,
                None,
                0,
                date(2026, 7, 1),
                datetime(2026, 7, 1, 10, 1, 0),
                "Call 1 - Day 1",
                "Latha Ramakrishnan",
                "Candidate",
                "+919008999139",
                None,
                None,
                None,
                None,
                "owner@example.com",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )],
        ]
    )
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    calls._stats_cache = {"owner@example.com": {"due_today": 99}}
    calls._stats_cache_ts = {"owner@example.com": 1}
    calls._call_lists_cache = [{"id": 99, "name": "Old", "created_by": "owner@example.com"}]
    calls._calls_cache = []

    calls.warm_call_caches(conn)

    with calls._calls_lock:
        assert calls._call_lists_cache == [
            {
                "id": 7,
                "name": "Customer Marketing - Locad",
                "created_at": datetime(2026, 7, 1, 10, 0, 0),
                "created_by": "owner@example.com",
            }
        ]
        assert [call["id"] for call in calls._calls_cache] == [42]
        assert calls._calls_cache[0]["list_id"] == 7
        assert calls._stats_cache == {}
        assert calls._stats_cache_ts == {}

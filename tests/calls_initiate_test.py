from contextlib import AbstractContextManager

import psycopg2
import pytest
from fastapi import HTTPException

from backend.api import schemas
from backend.api.routes import calls


class _FakeCursor:
    def __init__(self, fetchone_results=None, execute_side_effect=None):
        self._fetchone_results = list(fetchone_results or [])
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

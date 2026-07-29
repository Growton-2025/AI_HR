import asyncio
from contextlib import AbstractContextManager
from datetime import date, datetime, time

import httpx
import psycopg2
import pytest
from fastapi import FastAPI, HTTPException

from backend.api import deps, schemas
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


class _ConnectionQueue:
    def __init__(self, connections):
        self.connections = list(connections)

    def __call__(self, *args, **kwargs):
        if not self.connections:
            raise AssertionError("No fake DB connection queued")
        return self.connections.pop(0)


def _build_user():
    return schemas.User(
        username="owner@example.com",
        email="owner@example.com",
        full_name="Owner",
    )


def _build_calls_app(monkeypatch):
    app = FastAPI()
    app.dependency_overrides[deps.get_current_user] = lambda: _build_user()
    app.include_router(calls.router, prefix="/api/calls")
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    return app


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
    # Single combined statement returns
    # (list_found, duplicate_count, inserted_count, requested_count, callable_count)
    cursor = _FakeCursor(fetchone_results=[(1, 1, 0, 1, 1)])
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
    cursor = _FakeCursor(fetchone_results=[(1, 0, 2, 2, 2)])
    conn = _FakeConnection(cursor)
    invalidated = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: invalidated.append(True))

    result = calls.add_candidates_to_list(
        calls.AddCandidatesRequest(candidate_ids=[101, 102], list_id=7),
        current_user=_build_user(),
    )

    assert result == {"success": True, "added_count": 2, "skipped_no_phone": 0}
    assert conn.commits == 1
    assert invalidated == [True]


def test_add_candidates_skips_contactless_and_reports_count(monkeypatch):
    """Candidates with no phone number are undiallable — they must not be
    inserted, and the caller must be told how many were left out so the shorter
    list is explainable rather than looking like data loss."""
    # 5 requested, only 3 have a number, so 3 inserted and 2 reported as skipped.
    cursor = _FakeCursor(fetchone_results=[(1, 0, 3, 5, 3)])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.add_candidates_to_list(
        calls.AddCandidatesRequest(candidate_ids=[1, 2, 3, 4, 5], list_id=7),
        current_user=_build_user(),
    )

    assert result == {"success": True, "added_count": 3, "skipped_no_phone": 2}
    assert conn.commits == 1


def test_add_candidates_all_contactless_returns_400(monkeypatch):
    """Nothing insertable and nothing duplicated — the recruiter needs the real
    reason (no numbers), not a silent success reporting zero added."""
    cursor = _FakeCursor(fetchone_results=[(1, 0, 0, 2, 0)])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.add_candidates_to_list(
            calls.AddCandidatesRequest(candidate_ids=[101, 102], list_id=7),
            current_user=_build_user(),
        )

    assert exc_info.value.status_code == 400
    assert "phone number" in str(exc_info.value.detail)
    assert conn.commits == 0


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
    evicted = []
    refreshed = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "evict_call_list_from_cache", lambda list_id: evicted.append(list_id))
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: refreshed.append(True))

    result = calls.delete_call_list(7, current_user=_build_user())

    assert result == {"success": True, "deleted_call_count": 3}
    assert conn.commits == 1
    assert evicted == [7]
    assert refreshed == [True]


def test_invalidate_calls_cache_marks_call_caches_dirty(monkeypatch):
    refreshed = []
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: refreshed.append(True))

    with calls._calls_lock:
        calls._cache_generation = 10
        calls._calls_cache = [{"id": 42}]
        calls._call_lists_cache = [{"id": 7}]
        calls._stats_cache = {"owner@example.com": {"due_today": 99}}
        calls._stats_cache_ts = {"owner@example.com": 1}

    calls.invalidate_calls_cache()

    with calls._calls_lock:
        assert calls._cache_generation == 11
        assert calls._calls_cache is None
        assert calls._call_lists_cache is None
        assert calls._stats_cache == {}
        assert calls._stats_cache_ts == {}
    assert refreshed == [True]


def test_create_call_list_success_invalidates_cache(monkeypatch):
    # Combined dup-check + insert returns the new row in one statement
    cursor = _FakeCursor(
        fetchone_results=[
            (7, "New List", datetime(2026, 7, 1, 10, 0, 0)),
        ]
    )
    conn = _FakeConnection(cursor)
    invalidated = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: invalidated.append(True))

    result = calls.create_call_list(
        calls.CallListCreate(name="New List"),
        current_user=_build_user(),
    )

    assert result["id"] == 7
    assert result["name"] == "New List"
    assert result["candidate_count"] == 0
    assert conn.commits == 1
    assert invalidated == [True]


def test_update_call_success_invalidates_cache(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(101, 7, "pending", None, "Call 1 - Day 1")])
    conn = _FakeConnection(cursor)
    invalidated = []

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: invalidated.append(True))

    result = calls.update_call(
        42,
        calls.CallUpdate(notes="Updated notes"),
        current_user=_build_user(),
    )

    assert result["success"] is True
    assert result["scheduled_next_title"] is None
    assert result["auto_unreachable"] is False
    assert result["wrong_number_tagged"] is False
    assert conn.commits == 1
    assert invalidated == [True]


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
                None,   # due_time
                False,  # candidate mobile_phone_wrong
                None,   # candidate notes
                "To be started",  # candidate status
                None,   # candidate linkedin
                None,   # sentiment
                None,   # sentiment_reason
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


def test_calls_mutation_routes_return_contracts_and_invalidate(monkeypatch):
    app = _build_calls_app(monkeypatch)

    create_conn = _FakeConnection(_FakeCursor(
        fetchone_results=[(7, "Route List", datetime(2026, 7, 1, 10, 0, 0))]
    ))
    add_conn = _FakeConnection(_FakeCursor(
        fetchone_results=[(1, 0, 2)],
    ))
    update_conn = _FakeConnection(_FakeCursor(
        fetchone_results=[
            (101, 7, "completed", "No Answer", "Call 1 - Day 1 - Second Half"),
            (False,),  # phone not tagged wrong → cadence continues
        ]
    ))
    delete_call_conn = _FakeConnection(_FakeCursor(fetchone_results=[(7,)]))
    delete_list_conn = _FakeConnection(_FakeCursor(fetchone_results=[(2,), (7,)]))

    calls_conns = _ConnectionQueue([create_conn, add_conn, update_conn])
    delete_conns = _ConnectionQueue([delete_call_conn, delete_list_conn])
    invalidated = []
    evicted_calls = []
    evicted_lists = []
    refreshed = []

    monkeypatch.setattr(calls, "get_calls_db_connection", calls_conns)
    monkeypatch.setattr(calls, "get_db_connection", delete_conns)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: invalidated.append(True))
    monkeypatch.setattr(calls, "evict_call_from_cache", lambda call_id: evicted_calls.append(call_id))
    monkeypatch.setattr(calls, "evict_call_list_from_cache", lambda list_id: evicted_lists.append(list_id))
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: refreshed.append(True))

    async def run_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            create_res = await client.post("/api/calls/lists", json={"name": "Route List"})
            add_res = await client.post(
                "/api/calls/add-candidates",
                json={"candidate_ids": [101, 102], "list_id": 7},
            )
            update_res = await client.patch(
                "/api/calls/41",
                json={"status": "completed", "outcome": "No Answer", "duration": 12},
            )
            delete_call_res = await client.delete("/api/calls/41")
            delete_list_res = await client.delete("/api/calls/lists/7")
        return create_res, add_res, update_res, delete_call_res, delete_list_res

    create_res, add_res, update_res, delete_call_res, delete_list_res = asyncio.run(run_requests())

    assert create_res.status_code == 200
    assert create_res.json()["id"] == 7
    assert create_res.json()["candidate_count"] == 0
    assert add_res.status_code == 200
    assert add_res.json() == {"success": True, "added_count": 2}
    assert update_res.status_code == 200
    assert update_res.json() == {
        "success": True,
        "scheduled_next_title": "Call 2 - Day 2 - First Half",
        "auto_unreachable": False,
        "wrong_number_tagged": False,
    }
    assert delete_call_res.status_code == 200
    assert delete_call_res.json() == {"success": True, "list_id": 7}
    assert delete_list_res.status_code == 200
    assert delete_list_res.json() == {"success": True, "deleted_call_count": 2}

    assert create_conn.commits == 1
    assert add_conn.commits == 1
    assert update_conn.commits == 1
    assert delete_call_conn.commits == 1
    assert delete_list_conn.commits == 1
    assert invalidated == [True, True, True]
    assert evicted_calls == [41]
    assert evicted_lists == [7]
    assert refreshed == [True, True]


def test_add_candidates_partial_duplicates_report_counts(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(1, 1, 0, 2, 2)])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.add_candidates_to_list(
            calls.AddCandidatesRequest(candidate_ids=[101, 102], list_id=7),
            current_user=_build_user(),
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.detail == "1 of 2 selected candidates are already in this call list"


def test_add_candidates_all_duplicates_report_clear_message(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(1, 2, 0, 2, 2)])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.add_candidates_to_list(
            calls.AddCandidatesRequest(candidate_ids=[101, 102], list_id=7),
            current_user=_build_user(),
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.detail == "All selected candidates are already in this call list"


def test_update_call_no_answer_schedules_next_sequence_step(monkeypatch):
    # Legacy "No Answer" outcome still advances the cadence.
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "No Answer", "Call 1 - Day 1 - Second Half"),
        (False,),  # candidate phone is not tagged wrong
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="No Answer", duration=12),
        current_user=_build_user(),
    )

    assert result["success"] is True
    assert result["scheduled_next_title"] == "Call 2 - Day 2 - First Half"
    assert conn.commits == 1

    insert_queries = [
        (query, params)
        for query, params in cursor.executed
        if "INSERT INTO calls" in query
    ]
    assert len(insert_queries) == 1
    _, insert_params = insert_queries[0]
    assert insert_params == (101, 7, 1, "Call 2 - Day 2 - First Half")


def test_update_call_not_connected_schedules_next_cadence_step(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "Not Connected", "Call 2 - Day 2 - First Half"),
        (False,),
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="Not Connected", duration=0),
        current_user=_build_user(),
    )

    assert result["scheduled_next_title"] == "Call 3 - Day 4 - Second Half"
    insert_queries = [
        params for query, params in cursor.executed if "INSERT INTO calls" in query
    ]
    assert insert_queries == [(101, 7, 2, "Call 3 - Day 4 - Second Half")]


def test_update_call_fifth_failed_attempt_marks_unreachable(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "Not Connected - Not Reachable", "Call 5 - Day 10 - Second Half"),
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="Not Connected - Not Reachable", duration=0),
        current_user=_build_user(),
    )

    assert result["auto_unreachable"] is True
    assert result["scheduled_next_title"] is None
    assert not any("INSERT INTO calls" in query for query, _params in cursor.executed)
    unreachable_updates = [
        params for query, params in cursor.executed
        if "SET outcome = %s" in query
    ]
    assert unreachable_updates == [("Unreachable", 41)]


def test_update_call_cadence_pauses_while_phone_tagged_wrong(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "Not Connected", "Call 2 - Day 2 - First Half"),
        (True,),  # wrong-number tag set → cadence paused
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="Not Connected", duration=0),
        current_user=_build_user(),
    )

    assert result["scheduled_next_title"] is None
    assert not any("INSERT INTO calls" in query for query, _params in cursor.executed)


def test_update_call_followup_outcome_requires_slot(monkeypatch):
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        calls.update_call(
            41,
            calls.CallUpdate(status="completed", outcome="Connected - Follow-up", duration=30),
            current_user=_build_user(),
        )

    assert exc_info.value.status_code == 400
    assert "Follow-up date and time are required" in exc_info.value.detail


def test_update_call_followup_outcome_schedules_slot(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "Connected - Follow-up", "Call 2 - Day 2 - First Half"),
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(
            status="completed",
            outcome="Connected - Follow-up",
            duration=30,
            followup_due_date=date(2026, 7, 10),
            followup_due_time=time(15, 30),
        ),
        current_user=_build_user(),
    )

    assert result["scheduled_next_title"] == "Follow-up Call"
    insert_queries = [
        params for query, params in cursor.executed if "INSERT INTO calls" in query
    ]
    assert insert_queries == [(101, 7, date(2026, 7, 10), time(15, 30), "Follow-up Call")]


def test_update_call_wrong_number_tags_candidate_and_pauses(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[
        (101, 7, "completed", "Wrong Number", "Call 1 - Day 1 - Second Half"),
    ])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="Wrong Number", duration=0),
        current_user=_build_user(),
    )

    assert result["wrong_number_tagged"] is True
    assert result["scheduled_next_title"] is None
    assert not any("INSERT INTO calls" in query for query, _params in cursor.executed)
    tag_updates = [
        (query, params) for query, params in cursor.executed
        if "mobile_phone_wrong = TRUE" in query
    ]
    assert len(tag_updates) == 1
    assert tag_updates[0][1] == (101,)


def test_update_call_interested_outcome_does_not_schedule_followup(monkeypatch):
    cursor = _FakeCursor(fetchone_results=[(101, 7, "completed", "Connected - Interested", "Call 1 - Day 1")])
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        41,
        calls.CallUpdate(status="completed", outcome="Connected - Interested", duration=45),
        current_user=_build_user(),
    )

    assert result["success"] is True
    assert result["scheduled_next_title"] is None
    assert not any("INSERT INTO calls" in query for query, _params in cursor.executed)


def test_stats_reflect_call_eviction_immediately(monkeypatch):
    app = _build_calls_app(monkeypatch)
    owner = "owner@example.com"
    today = date.today()

    with calls._calls_lock:
        calls._cache_warmed_at = calls.time.time()
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = [{"id": 7, "created_by": owner}]
        calls._calls_cache = [
            {"id": 1, "list_id": 7, "created_by": owner, "status": "pending", "due_date": today},
            {"id": 2, "list_id": 7, "created_by": owner, "status": "pending", "due_date": today},
        ]

    async def fetch_stats():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/calls/stats")

    before = asyncio.run(fetch_stats())
    assert before.json()["due_today"] == 2

    # Synchronous eviction (used by DELETE /calls/{id}) must be visible on the
    # very next stats read — no waiting for the async cache warm.
    calls.evict_call_from_cache(1)

    after = asyncio.run(fetch_stats())
    assert after.json()["due_today"] == 1


def test_stats_reflect_list_eviction_immediately(monkeypatch):
    app = _build_calls_app(monkeypatch)
    owner = "owner@example.com"
    today = date.today()

    with calls._calls_lock:
        calls._cache_warmed_at = calls.time.time()
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = [
            {"id": 7, "created_by": owner},
            {"id": 8, "created_by": owner},
        ]
        calls._calls_cache = [
            {"id": 1, "list_id": 7, "created_by": owner, "status": "pending", "due_date": today},
            {"id": 2, "list_id": 8, "created_by": owner, "status": "pending", "due_date": today},
        ]

    async def fetch_stats():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/calls/stats")

    before = asyncio.run(fetch_stats())
    assert before.json() == {"due_today": 2, "upcoming": 0, "completed": 0, "active_lists": 2, "inbound_pending": 0}

    calls.evict_call_list_from_cache(7)

    after = asyncio.run(fetch_stats())
    assert after.json() == {"due_today": 1, "upcoming": 0, "completed": 0, "active_lists": 1, "inbound_pending": 0}


def test_call_stats_route_matches_db_and_memory_cache_paths(monkeypatch):
    app = _build_calls_app(monkeypatch)
    owner = "owner@example.com"
    today = date.today()

    with calls._calls_lock:
        calls._cache_warmed_at = calls.time.time()
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = [{"id": 7, "created_by": owner}]
        calls._calls_cache = [
            {"id": 1, "list_id": 7, "created_by": owner, "status": "pending", "due_date": today},
            {"id": 2, "list_id": 7, "created_by": owner, "status": "pending", "due_date": date(today.year + 1, 1, 1)},
            {"id": 3, "list_id": 7, "created_by": owner, "status": "completed", "due_date": today},
            {"id": 4, "list_id": 7, "created_by": owner, "status": "cancelled", "due_date": today},
            {"id": 5, "list_id": 8, "created_by": "someone@example.com", "status": "pending", "due_date": today},
        ]

    async def fetch_stats():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/calls/stats")

    cache_res = asyncio.run(fetch_stats())
    assert cache_res.status_code == 200
    assert cache_res.json() == {
        "due_today": 1,
        "upcoming": 1,
        "completed": 1,
        "active_lists": 1,
        "inbound_pending": 0,
    }

    db_cursor = _FakeCursor(fetchone_results=[(1, 1, 1, 1)])
    db_conn = _FakeConnection(db_cursor)
    with calls._calls_lock:
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = None
        calls._calls_cache = None

    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: db_conn)
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: None)

    db_res = asyncio.run(fetch_stats())
    assert db_res.status_code == 200
    assert db_res.json() == cache_res.json()

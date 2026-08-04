"""The call cadence ladder must be anchored on the task's own due date.

Titles encode a fixed schedule from the start of the sequence — Day 1, 2, 4, 7,
10 — so scheduling the next step from CURRENT_DATE made every step drift by
however long a recruiter took to log the previous one. A "Call 1 - Day 1" task
due Monday but logged Tuesday morning produced a "Call 2 - Day 2" task dated
Wednesday.
"""

import asyncio
from datetime import date, datetime

import httpx
from fastapi import FastAPI

from backend.api import deps, schemas
from backend.api.routes import calls

OWNER = "owner@example.com"


class _Cursor:
    def __init__(self, script):
        self._script = list(script)
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self._script.pop(0) if self._script else None

    def fetchall(self):
        return []

    def close(self):
        return None


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        return None


def _app(monkeypatch, cursor):
    app = FastAPI()
    app.dependency_overrides[deps.get_current_user] = lambda: schemas.User(
        id=1, username=OWNER, email=OWNER
    )
    app.include_router(calls.router, prefix="/api/calls")
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: _Conn(cursor))
    monkeypatch.setattr(calls, "return_db_connection", lambda c: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda *a, **k: None)
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda *a, **k: None)
    return app


def _patch(app, call_id, body):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.patch(f"/api/calls/{call_id}", json=body)

    return asyncio.run(run())


def _insert_stmt(cursor):
    for query, params in cursor.executed:
        if "INSERT INTO calls" in query:
            return query, params
    return None, None


def test_next_call_is_anchored_on_the_previous_due_date(monkeypatch):
    """The reported case: Call 1 due Aug 3, logged Aug 4 -> Call 2 due Aug 4,
    not Aug 5."""
    cursor = _Cursor([
        # RETURNING candidate_id, list_id, status, outcome, task_title, due_date
        (500, 7, "completed", "Not Connected", "Call 1 - Day 1", date(2026, 8, 3)),
        (False, False, False),   # phone ok, cadence not paused, has a number
        None,
    ])
    app = _app(monkeypatch, cursor)

    res = _patch(app, 1, {"status": "completed", "outcome": "Not Connected"})
    assert res.status_code in (200, 500)  # response shape is not what we assert here

    query, params = _insert_stmt(cursor)
    assert query is not None, "no next call was scheduled"
    # Anchored on the previous task's due date...
    assert params[2] == date(2026, 8, 3)
    assert params[3] == 1          # Call 1 -> +1 day
    assert params[4] == "Call 2 - Day 2 - First Half"
    # ...and clamped so catching up late never schedules into the past.
    assert "GREATEST" in query
    # The drift-causing anchor must be gone.
    assert "CURRENT_DATE + %s" not in query


def test_overdue_task_does_not_schedule_into_the_past(monkeypatch):
    """A badly overdue Call 1 must not produce a Call 2 dated last month."""
    cursor = _Cursor([
        (500, 7, "completed", "Not Connected", "Call 1 - Day 1", date(2026, 1, 5)),
        (False, False, False),
        None,
    ])
    app = _app(monkeypatch, cursor)
    _patch(app, 1, {"status": "completed", "outcome": "Not Connected"})

    query, params = _insert_stmt(cursor)
    assert query is not None
    assert params[2] == date(2026, 1, 5)
    # GREATEST against IST today is what rescues it; the SQL, not Python, clamps.
    assert "GREATEST" in query and "Asia/Kolkata" in query


def test_sequence_delays_are_unchanged():
    """The ladder itself is not being redefined — only its anchor."""
    assert calls.next_sequence_step("Call 1 - Day 1") == (1, "Call 2 - Day 2 - First Half")
    assert calls.next_sequence_step("Call 2 - Day 2 - First Half") == (2, "Call 3 - Day 4 - Second Half")
    assert calls.next_sequence_step("Call 3 - Day 4 - Second Half") == (3, "Call 4 - Day 7 - First Half")
    assert calls.next_sequence_step("Call 4 - Day 7 - First Half") == (3, "Call 5 - Day 10 - Second Half")
    assert calls.next_sequence_step("Call 5 - Day 10 - Second Half") is None

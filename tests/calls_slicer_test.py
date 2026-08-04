"""Slicer (date-range / outcome-group) filters on GET /api/calls and
GET /api/calls/stats.

The endpoints each have an in-memory cache path and a SQL fallback that must
stay behaviorally identical — tests below exercise both (cache path by seeding
calls._calls_cache while fresh; SQL path by making the cache stale and
asserting on the captured query text/params)."""

import asyncio
from datetime import date, datetime, timedelta

import httpx
from fastapi import FastAPI

from backend.api import deps, schemas
from backend.api.routes import calls

OWNER = "owner@example.com"


class _FakeCursor:
    def __init__(self, fetchone_results=None, fetchall_results=None):
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

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


def _build_user():
    return schemas.User(username=OWNER, email=OWNER, full_name="Owner")


def _build_calls_app(monkeypatch):
    app = FastAPI()
    app.dependency_overrides[deps.get_current_user] = lambda: _build_user()
    app.include_router(calls.router, prefix="/api/calls")
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda: None)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    return app


def _get(app, path):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path)

    return asyncio.run(run())


def _cache_call(call_id, *, status, due_date, outcome=None, completed_at=None, list_id=7):
    """A calls-cache entry with every field CallResponse requires."""
    return {
        "id": call_id,
        "candidate_id": call_id,
        "list_id": list_id,
        "status": status,
        "outcome": outcome,
        "notes": None,
        "duration": 0,
        "due_date": due_date,
        "created_at": datetime(2026, 1, 1, 9, 0),
        "task_title": None,
        "candidate_name": f"Candidate {call_id}",
        "candidate_title": None,
        "candidate_company": None,
        "candidate_phone": None,
        "completed_at": completed_at,
        "created_by": OWNER,
    }


def _db_row(call_id, *, status, due_date, outcome=None, completed_at=None, list_id=7):
    """A row shaped like CALLS_SELECT_QUERY for call_row_to_dict."""
    return (
        call_id, call_id, list_id, status, outcome, None, 0, due_date,
        datetime(2026, 1, 1, 9, 0), None, f"Candidate {call_id}", None, None,
        completed_at, None, None, None, OWNER, None, None, None, None, None,
        None, None, None, None, False, None, "To be started", None, None, None,
    )


def _seed_cache(calls_list, call_lists=None):
    with calls._calls_lock:
        calls._cache_warmed_at = calls.time.time()
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = call_lists if call_lists is not None else [{"id": 7, "created_by": OWNER}]
        calls._calls_cache = calls_list


def _stale_cache():
    with calls._calls_lock:
        calls._cache_warmed_at = 0.0
        calls._stats_cache = {}
        calls._stats_cache_ts = {}
        calls._call_lists_cache = None
        calls._calls_cache = None


def _use_db(monkeypatch, cursor):
    _stale_cache()
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: None)
    return conn


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_resolve_range_bounds_presets():
    today = date(2026, 7, 23)
    assert calls.resolve_range_bounds(None, None, None, today) is None
    assert calls.resolve_range_bounds("today", None, None, today) == (today, today)
    assert calls.resolve_range_bounds("yesterday", None, None, today) == (date(2026, 7, 22), date(2026, 7, 22))
    assert calls.resolve_range_bounds("last7", None, None, today) == (date(2026, 7, 17), today)
    assert calls.resolve_range_bounds("last30", None, None, today) == (date(2026, 6, 24), today)
    assert calls.resolve_range_bounds("custom", date(2026, 7, 1), date(2026, 7, 5), today) == (
        date(2026, 7, 1), date(2026, 7, 5),
    )


def test_outcome_groups_cover_known_outcomes():
    assert calls.OUTCOME_GROUPS["connected"] == frozenset(calls.TERMINAL_CALL_OUTCOMES)
    assert calls.OUTCOME_GROUPS["followup"] == frozenset(calls.FOLLOWUP_CALL_OUTCOMES)
    assert calls.UNREACHABLE_OUTCOME in calls.OUTCOME_GROUPS["not_connected"]
    assert "Left Voicemail" in calls.OUTCOME_GROUPS["not_connected"]
    # Wrong Number deliberately belongs to no group (All Outcomes only).
    assert all(calls.WRONG_NUMBER_OUTCOME not in group for group in calls.OUTCOME_GROUPS.values())


def test_custom_range_inclusive_of_end_date():
    today = date(2026, 7, 23)
    bounds = calls.resolve_range_bounds("custom", date(2026, 7, 18), date(2026, 7, 20), today)
    # 23:59 IST on the end date. completed_at holds UTC, so that is 18:29 UTC
    # the same day — buckets are IST days, matching what the recruiter picked
    # in the date box.
    late_on_end_date = _cache_call(
        1, status="completed", due_date=date(2026, 7, 1),
        completed_at=datetime(2026, 7, 20, 18, 29),
    )
    assert calls.call_matches_slicer(late_on_end_date, bounds, None, use_completed_at=True)

    # 23:59 UTC is 05:29 IST the NEXT day, so it belongs to the 21st and falls
    # outside an 18th-20th range. Bucketing on the raw UTC date used to pull it
    # in, which is the mismatch that made calls appear to move between days.
    just_past_midnight_ist = _cache_call(
        2, status="completed", due_date=date(2026, 7, 1),
        completed_at=datetime(2026, 7, 20, 23, 59),
    )
    assert not calls.call_matches_slicer(
        just_past_midnight_ist, bounds, None, use_completed_at=True
    )

    sql, params = calls.build_range_sql(
        "custom", date(2026, 7, 18), date(2026, 7, 20), use_completed_at=True
    )
    # The picked dates are IST days; the bounds are converted to UTC instants so
    # completed_at (a naive UTC timestamp) can be compared without wrapping the
    # column, keeping any index on it usable.
    assert "c.completed_at >= ((%s::date)::timestamp AT TIME ZONE 'Asia/Kolkata')" in sql
    assert "c.completed_at < ((((%s::date) + 1))::timestamp AT TIME ZONE 'Asia/Kolkata')" in sql
    assert params == ["2026-07-18", "2026-07-20"]

    due_sql, due_params = calls.build_range_sql(
        "custom", date(2026, 7, 18), date(2026, 7, 20), use_completed_at=False
    )
    assert "c.due_date >= %s::date" in due_sql
    assert "c.due_date <= %s::date" in due_sql
    assert due_params == ["2026-07-18", "2026-07-20"]


# ── Validation ───────────────────────────────────────────────────────────────

def test_slicer_param_validation(monkeypatch):
    app = _build_calls_app(monkeypatch)
    _seed_cache([])

    assert _get(app, "/api/calls?range=custom&date_from=2026-07-01").status_code == 400
    assert _get(app, "/api/calls/stats?range=custom&date_to=2026-07-01").status_code == 400
    assert _get(app, "/api/calls?range=fortnight").status_code == 400
    assert _get(app, "/api/calls?outcome_group=bogus").status_code == 400
    assert _get(app, "/api/calls/stats?outcome_group=bogus").status_code == 400
    assert _get(
        app, "/api/calls?range=custom&date_from=2026-07-05&date_to=2026-07-01"
    ).status_code == 400
    assert _get(app, "/api/calls?range=custom&date_from=notadate&date_to=2026-07-01").status_code == 422


# ── GET /api/calls ───────────────────────────────────────────────────────────

def test_get_calls_cache_path_slices_range_and_outcome(monkeypatch):
    app = _build_calls_app(monkeypatch)
    today = date.today()
    now = datetime.now()
    _seed_cache([
        _cache_call(1, status="completed", due_date=today - timedelta(days=20), outcome="Connected - Interested", completed_at=now),
        _cache_call(2, status="completed", due_date=today, outcome="Connected - Interested", completed_at=now - timedelta(days=10)),
        _cache_call(3, status="completed", due_date=today, outcome="Not Connected", completed_at=now),
    ])

    res = _get(app, "/api/calls?status=completed&range=today&outcome_group=connected")
    assert res.status_code == 200
    assert [c["id"] for c in res.json()] == [1]

    # Pending views slice on due_date, not completed_at.
    _seed_cache([
        _cache_call(4, status="pending", due_date=today),
        _cache_call(5, status="pending", due_date=today - timedelta(days=3)),
    ])
    res = _get(app, "/api/calls?status=pending&range=today")
    assert res.status_code == 200
    assert [c["id"] for c in res.json()] == [4]


def test_get_calls_sql_path_builds_expected_predicates(monkeypatch):
    app = _build_calls_app(monkeypatch)
    cursor = _FakeCursor(fetchall_results=[[]])
    _use_db(monkeypatch, cursor)

    res = _get(app, "/api/calls?status=completed&range=last7&outcome_group=not_connected")
    assert res.status_code == 200
    query, params = cursor.executed[0]
    # Day boundaries are IST, converted to UTC for comparison so the column
    # stays bare and any index on completed_at remains usable.
    assert "Asia/Kolkata" in query
    assert "CURRENT_DATE" not in query
    assert "c.completed_at >=" in query and "c.completed_at <" in query
    assert "c.outcome = ANY(%s)" in query
    assert params == [OWNER, "completed", sorted(calls.OUTCOME_GROUPS["not_connected"])]

    cursor = _FakeCursor(fetchall_results=[[]])
    _use_db(monkeypatch, cursor)
    res = _get(app, "/api/calls?status=pending&due_filter=today&range=yesterday")
    assert res.status_code == 200
    query, params = cursor.executed[0]
    # The existing due_filter predicate and the new range predicate coexist.
    ist_today = "((NOW() AT TIME ZONE 'Asia/Kolkata')::date)"
    assert f"c.due_date <= {ist_today}" in query
    assert f"c.due_date >= ({ist_today} - 1)" in query
    assert f"c.due_date <= ({ist_today} - 1)" in query
    assert params == [OWNER, "pending"]


def test_get_calls_cache_and_sql_parity(monkeypatch):
    app = _build_calls_app(monkeypatch)
    now = datetime.now()
    matching = dict(
        status="completed", due_date=date.today(),
        outcome="Connected - Interested", completed_at=now,
    )
    _seed_cache([
        _cache_call(1, **matching),
        _cache_call(2, status="completed", due_date=date.today(), outcome="Not Connected", completed_at=now),
        _cache_call(3, status="completed", due_date=date.today(), outcome="Connected - Interested", completed_at=now - timedelta(days=30)),
    ])
    path = "/api/calls?status=completed&range=last7&outcome_group=connected"
    cache_res = _get(app, path)
    assert cache_res.status_code == 200
    assert [c["id"] for c in cache_res.json()] == [1]

    cursor = _FakeCursor(fetchall_results=[[_db_row(1, **matching)]])
    _use_db(monkeypatch, cursor)
    db_res = _get(app, path)
    assert db_res.status_code == 200
    assert [c["id"] for c in db_res.json()] == [c["id"] for c in cache_res.json()]


# ── GET /api/calls/stats ─────────────────────────────────────────────────────

def test_call_stats_slicer_cache_and_sql_parity(monkeypatch):
    app = _build_calls_app(monkeypatch)
    today = date.today()
    now = datetime.now()
    _seed_cache([
        _cache_call(1, status="pending", due_date=today),
        _cache_call(2, status="pending", due_date=date(today.year + 1, 1, 1)),
        _cache_call(3, status="completed", due_date=today, outcome="Connected - Interested", completed_at=now),
        _cache_call(4, status="completed", due_date=today, outcome="Connected - Interested", completed_at=now - timedelta(days=40)),
        _cache_call(5, status="completed", due_date=today, outcome="Not Connected", completed_at=now),
    ])

    path = "/api/calls/stats?range=last7&outcome_group=connected"
    cache_res = _get(app, path)
    assert cache_res.status_code == 200
    # The outcome filter applies ONLY to the completed bucket — pending rows
    # have no outcome yet, so due_today/upcoming slice on the date range only.
    assert cache_res.json() == {"due_today": 1, "upcoming": 0, "completed": 1, "active_lists": 1, "inbound_pending": 0}

    cursor = _FakeCursor(fetchone_results=[(1, 0, 1, 1)])
    _use_db(monkeypatch, cursor)
    db_res = _get(app, path)
    assert db_res.status_code == 200
    assert db_res.json() == cache_res.json()

    query, params = cursor.executed[0]
    assert "due_date >= CURRENT_DATE - 6 AND due_date <= CURRENT_DATE" in query
    assert "completed_at >= CURRENT_DATE - 6 AND completed_at < CURRENT_DATE + INTERVAL '1 day'" in query
    assert query.count("outcome = ANY(%s)") == 1
    assert params == [OWNER, sorted(calls.OUTCOME_GROUPS["connected"])]


def test_call_stats_outcome_filter_never_zeroes_pending_buckets(monkeypatch):
    """Regression for the "everything vanished" glitch: an outcome filter with
    no date range must leave due_today/upcoming untouched."""
    app = _build_calls_app(monkeypatch)
    today = date.today()
    _seed_cache([
        _cache_call(1, status="pending", due_date=today),
        _cache_call(2, status="pending", due_date=date(today.year + 1, 1, 1)),
        _cache_call(3, status="completed", due_date=today, outcome="Not Connected", completed_at=datetime.now()),
    ])

    res = _get(app, "/api/calls/stats?outcome_group=connected")
    assert res.status_code == 200
    assert res.json() == {"due_today": 1, "upcoming": 1, "completed": 0, "active_lists": 1, "inbound_pending": 0}


def test_call_stats_range_without_outcome_keeps_pending_buckets(monkeypatch):
    app = _build_calls_app(monkeypatch)
    today = date.today()
    _seed_cache([
        _cache_call(1, status="pending", due_date=today),
        _cache_call(2, status="pending", due_date=today - timedelta(days=10)),
        _cache_call(3, status="pending", due_date=date(today.year + 1, 1, 1)),
    ])

    res = _get(app, "/api/calls/stats?range=today")
    assert res.status_code == 200
    # Overdue call (id=2) falls outside the "today" range window.
    assert res.json() == {"due_today": 1, "upcoming": 0, "completed": 0, "active_lists": 1, "inbound_pending": 0}


def test_call_stats_cache_key_includes_slicer_params(monkeypatch):
    app = _build_calls_app(monkeypatch)
    today = date.today()
    _seed_cache([
        _cache_call(1, status="completed", due_date=today, outcome="Connected - Interested", completed_at=datetime.now()),
        _cache_call(2, status="completed", due_date=today, outcome="Connected - Interested", completed_at=datetime.now() - timedelta(days=40)),
    ])

    default_res = _get(app, "/api/calls/stats")
    assert default_res.json()["completed"] == 2
    # A sliced request must not be served from the default-params cache entry.
    sliced_res = _get(app, "/api/calls/stats?range=last7")
    assert sliced_res.json()["completed"] == 1
    # And the default entry is still intact afterwards.
    assert _get(app, "/api/calls/stats").json()["completed"] == 2


# ── Regression: default params unchanged ─────────────────────────────────────

def test_default_params_regression_unchanged(monkeypatch):
    app = _build_calls_app(monkeypatch)
    cursor = _FakeCursor(fetchall_results=[[]])
    _use_db(monkeypatch, cursor)

    res = _get(app, "/api/calls?due_filter=today&status=pending")
    assert res.status_code == 200
    query, params = cursor.executed[0]
    assert params == [OWNER, "pending"]
    assert "INTERVAL" not in query
    assert "ANY(" not in query
    assert "CURRENT_DATE -" not in query

    cursor = _FakeCursor(fetchone_results=[(1, 2, 3, 4)])
    _use_db(monkeypatch, cursor)
    res = _get(app, "/api/calls/stats")
    assert res.status_code == 200
    assert res.json() == {"due_today": 1, "upcoming": 2, "completed": 3, "active_lists": 4, "inbound_pending": 0}
    query, params = cursor.executed[0]
    assert params == [OWNER]
    assert "INTERVAL" not in query
    assert "ANY(" not in query

"""Hosted runs WEB_CONCURRENCY=4: whatever one worker keeps in a module dict is
invisible to the other three. These pin the two places that bit recruiters —
the dial handshake and the candidate notes the call modal pre-fills."""
import asyncio
import contextlib
import datetime as dt

from backend.api.routes import candidates, plivo as plivo_routes
from backend.integrations import plivo_service


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self, rows):
        self._cursor = _Cursor(rows)

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


def _fake_calls_db(monkeypatch, rows):
    """Route plivo_service's lazy `from backend.api.routes.calls import ...` to a fake."""
    from backend.api.routes import calls as calls_routes
    conn = _Conn(rows)
    monkeypatch.setattr(calls_routes, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls_routes, "return_db_connection", lambda c: None)
    return conn


def test_token_handshake_is_answered_from_the_calls_row_when_memory_is_empty(monkeypatch):
    # The webhook was served by another worker: this process never saw the token.
    plivo_service.dial_token_states.clear()
    seen = dt.datetime(2026, 9, 25, 5, 0, 0)
    conn = _fake_calls_db(monkeypatch, [("call-uuid-1", "endpointuser", "9876543210", seen)])

    state = asyncio.run(plivo_routes.get_call_state_by_token("tok-1"))

    assert state["call_uuid"] == "call-uuid-1"
    assert state["username"] == "endpointuser"
    assert state["to_number"] == "+919876543210"
    assert state["dial_token"] == "tok-1"
    assert conn._cursor.executed[0][1] == ("tok-1",)


def test_token_handshake_prefers_this_workers_memory(monkeypatch):
    plivo_service.dial_token_states["tok-2"] = {
        "call_uuid": "mem-uuid", "username": "u", "to_number": "+911111111111",
        "seen_at": 1.0, "dial_token": "tok-2",
    }
    conn = _fake_calls_db(monkeypatch, [("db-uuid", "u", "1111111111", None)])
    try:
        state = asyncio.run(plivo_routes.get_call_state_by_token("tok-2"))
    finally:
        plivo_service.dial_token_states.clear()
    assert state["call_uuid"] == "mem-uuid"
    assert conn._cursor.executed == []


def test_token_handshake_still_reports_nothing_before_the_webhook_lands(monkeypatch):
    plivo_service.dial_token_states.clear()
    _fake_calls_db(monkeypatch, [])  # row exists but plivo_call_uuid is NULL -> query returns nothing
    state = asyncio.run(plivo_routes.get_call_state_by_token("tok-3"))
    assert state["call_uuid"] is None


def test_username_handshake_falls_back_to_the_latest_calls_row(monkeypatch):
    plivo_service.last_call_states.clear()
    _fake_calls_db(monkeypatch, [("call-uuid-9", "endpointuser", "+91 98765 43210", None)])
    state = asyncio.run(plivo_routes.get_call_state("endpointuser"))
    assert state["call_uuid"] == "call-uuid-9"
    assert state["to_number"] == "+919876543210"


def test_handshake_survives_a_db_outage(monkeypatch):
    plivo_service.dial_token_states.clear()
    from backend.api.routes import calls as calls_routes
    monkeypatch.setattr(calls_routes, "get_calls_db_connection", lambda: None)
    state = asyncio.run(plivo_routes.get_call_state_by_token("tok-4"))
    assert state["call_uuid"] is None


class _User:
    id = 4
    role = "admin"
    email = "admin@example.com"


def _fake_candidates_db(monkeypatch, row):
    @contextlib.contextmanager
    def ctx(**kwargs):
        yield _Conn([row] if row else [])
    monkeypatch.setattr(candidates, "get_db_connection_context", ctx)


def test_get_candidate_serves_notes_saved_through_another_worker(monkeypatch):
    # This worker's snapshot predates the recruiter's edit.
    stale = {"id": 14427, "name": "Cand", "notes": "", "email": "", "mobile_phone": "",
             "phone": "", "linkedin": "", "status": "To be started", "owner_user_id": None}
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", {14427: stale})
    monkeypatch.setattr(candidates, "profile_passes_scope", lambda *a, **k: True)
    _fake_candidates_db(monkeypatch, ("Not reachable - Incoming freeze", "a@b.c", "9999999999",
                                      "https://linkedin.com/in/x", "Called"))

    prof = asyncio.run(candidates.get_candidate(14427, current_user=_User()))

    assert prof["notes"] == "Not reachable - Incoming freeze"
    assert prof["mobile_phone"] == "9999999999" and prof["phone"] == "9999999999"
    assert prof["status"] == "Called"
    # The snapshot itself is healed, so this worker's list views agree too.
    assert stale["notes"] == "Not reachable - Incoming freeze"


def test_get_candidate_keeps_cached_profile_when_db_is_unavailable(monkeypatch):
    cached = {"id": 1, "name": "Cand", "notes": "kept", "owner_user_id": None}
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", {1: cached})
    monkeypatch.setattr(candidates, "profile_passes_scope", lambda *a, **k: True)

    @contextlib.contextmanager
    def ctx(**kwargs):
        yield None
    monkeypatch.setattr(candidates, "get_db_connection_context", ctx)

    prof = asyncio.run(candidates.get_candidate(1, current_user=_User()))
    assert prof["notes"] == "kept"


def _fake_refresh(cache, profile):
    def refresh(ids):
        assert ids == [profile["id"]]
        cache[profile["id"]] = dict(profile)
        return 1
    return refresh


def test_get_candidate_loads_the_profile_on_a_cold_worker(monkeypatch):
    from backend.pipeline import query as query_mod
    cache = {}  # this worker has not run a search since the deploy
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", cache)
    monkeypatch.setattr(query_mod, "refresh_profiles_in_cache",
                        _fake_refresh(cache, {"id": 2519, "name": "Nethranand", "notes": "", "owner_user_id": None}))
    monkeypatch.setattr(candidates, "profile_passes_scope", lambda *a, **k: True)
    _fake_candidates_db(monkeypatch, ("n", "", "+918618884276", "", "To be started"))

    prof = asyncio.run(candidates.get_candidate(2519, current_user=_User()))
    assert prof["name"] == "Nethranand"
    assert prof["mobile_phone"] == "+918618884276"


def test_update_candidate_no_longer_404s_on_a_cold_worker(monkeypatch):
    from backend.pipeline import query as query_mod
    cache = {}
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", cache)
    monkeypatch.setattr(query_mod, "refresh_profiles_in_cache",
                        _fake_refresh(cache, {"id": 2519, "name": "Nethranand", "notes": "", "owner_user_id": None}))
    monkeypatch.setattr(candidates, "invalidate_candidate_count_caches", lambda *a, **k: None)
    _fake_candidates_db(monkeypatch, None)

    result = asyncio.run(candidates.update_candidate(2519, {"notes": "reachable now"}, current_user=_User()))
    assert result.get("success") is True
    assert cache[2519]["notes"] == "reachable now"


def test_unknown_candidate_is_still_404(monkeypatch):
    from fastapi import HTTPException
    from backend.pipeline import query as query_mod
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", {})
    monkeypatch.setattr(query_mod, "refresh_profiles_in_cache", lambda ids: 0)
    try:
        asyncio.run(candidates.get_candidate(999999, current_user=_User()))
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("expected 404")

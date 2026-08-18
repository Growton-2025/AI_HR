"""A retired candidate leaves the calling cadence — Hayasa Calling Framework.

"Post the call, when the candidate status is updated to any of the below,
remove the candidate from the future cadence (next calls). Retain the record in
the completed log (the task should be shown as completed)."

The statuses were already listed in TERMINAL_CANDIDATE_STATUSES and the sweep
that closes pending tasks already existed — but the cadence itself never
consulted either. Logging "Not Connected" minted the next attempt no matter
what the candidate's status said, so a candidate marked "Shared with customer"
was handed a fresh call task every time a recruiter logged the previous one,
and no amount of sweeping could keep them out of the loop.

These tests pin every gate the fix added: schedule, close, read, and add.
"""

import pytest

from backend.api import schemas
from backend.api.routes import calls
from backend.api.routes.calls import (
    TERMINAL_CANDIDATE_STATUSES,
    close_pending_calls_for_candidates,
    is_terminal_candidate_status,
    terminal_candidate_status_sql,
)

# Verbatim from "Removing the candidate from future calls based on candidate
# Status" in the framework document.
FRAMEWORK_STATUSES = [
    "Not Interested",
    "High CTC",
    "Shared with Customer",
    "For Future",
    "Shortlist - Rejected",
    "Duplicate",
    "Rejected",
]

USER = schemas.User(id=7, username="recruiter", email="rec@example.com", role="recruiter")


class _ScriptedCursor:
    """Returns a queued result per executed statement, keyed by a SQL fragment."""

    def __init__(self, script):
        self.script = script
        self.executed = []
        self._next = None
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        self._next = None
        for fragment, result in self.script.items():
            if fragment in " ".join(sql.split()):
                self._next = result
                break

    def fetchone(self):
        return self._next

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ran(self, fragment):
        return [sql for sql, _ in self.executed if fragment in sql]


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


# ── the status list itself ───────────────────────────────────────────────────

def test_every_framework_status_retires_the_candidate():
    assert {s.lower() for s in FRAMEWORK_STATUSES} == TERMINAL_CANDIDATE_STATUSES


@pytest.mark.parametrize("status", FRAMEWORK_STATUSES)
def test_status_matching_ignores_case_and_padding(status):
    assert is_terminal_candidate_status(status)
    assert is_terminal_candidate_status(f"  {status.upper()} ")
    assert is_terminal_candidate_status(status.lower())


@pytest.mark.parametrize(
    "status",
    ["Shortlisted", "Followup / In conversation", "Reached out - Phone", "", None],
)
def test_active_statuses_stay_in_the_cadence(status):
    # "Shortlisted" shares a prefix with "Shortlist - Rejected" — matching must
    # be exact or the whole shortlist would stop being called.
    assert not is_terminal_candidate_status(status)


def test_sql_predicate_matches_the_python_rule():
    sql = terminal_candidate_status_sql("cand.status")
    assert "LOWER(TRIM(COALESCE(cand.status, '')))" in sql
    for status in FRAMEWORK_STATUSES:
        assert f"'{status.lower()}'" in sql
    assert "'shortlisted'" not in sql


# ── 1. the cadence must not schedule the next attempt ────────────────────────

def _update_call_to_not_connected(monkeypatch, gate_row):
    """Log "Not Connected" on Call 1 and report what the cadence did."""
    cursor = _ScriptedCursor({
        "UPDATE calls SET": (101, 55, "completed", "Not Connected", "Call 1 - Day 1", None),
        "FROM candidates WHERE id": gate_row,
    })
    connection = _Connection(cursor)
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: connection)
    monkeypatch.setattr(calls, "return_db_connection", lambda _c: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)
    result = calls.update_call(
        1, calls.CallUpdate(status="completed", outcome="Not Connected"), current_user=USER
    )
    return result, cursor


def test_terminal_candidate_gets_no_next_attempt(monkeypatch):
    # (wrong number, cadence paused, no phone, TERMINAL STATUS)
    result, cursor = _update_call_to_not_connected(
        monkeypatch, (False, False, False, True)
    )

    assert cursor.ran("INSERT INTO calls") == []
    assert result["scheduled_next_title"] is None


def test_active_candidate_still_advances_through_the_cadence(monkeypatch):
    result, cursor = _update_call_to_not_connected(
        monkeypatch, (False, False, False, False)
    )

    assert cursor.ran("INSERT INTO calls")
    assert result["scheduled_next_title"] == "Call 2 - Day 2 - First Half"


def test_the_cadence_gate_reads_candidate_status(monkeypatch):
    _, cursor = _update_call_to_not_connected(monkeypatch, (False, False, False, True))
    gate = cursor.ran("FROM candidates WHERE id")

    assert gate, "the cadence must check the candidate before scheduling"
    assert "cadence_paused" in gate[0]      # pre-existing guards intact
    assert "mobile_phone_wrong" in gate[0]
    assert "'for future'" in gate[0]        # …and the status rule alongside them


def test_followup_is_not_scheduled_for_a_retired_candidate(monkeypatch):
    cursor = _ScriptedCursor({
        "UPDATE calls SET": (101, 55, "completed", "Connected - Follow-up", "Call 1 - Day 1", None),
        "FROM candidates WHERE id": (True,),
    })
    connection = _Connection(cursor)
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: connection)
    monkeypatch.setattr(calls, "return_db_connection", lambda _c: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)

    result = calls.update_call(
        1,
        calls.CallUpdate(
            status="completed",
            outcome="Connected - Follow-up",
            due_date="2026-08-20",
            due_time="10:00",
        ),
        current_user=USER,
    )

    assert cursor.ran("INSERT INTO calls") == []
    assert result["scheduled_next_title"] is None


# ── 2. closing pending tasks is not scoped to one recruiter's lists ──────────

def test_closing_covers_every_list_the_candidate_sits_in():
    cursor = _ScriptedCursor({})
    close_pending_calls_for_candidates(cursor, [101], "Shared with customer")
    sql, params = cursor.executed[0]

    # The owner-scoped version left the candidate being dialled from a list
    # someone else created.
    assert "call_lists" not in sql
    assert "created_by" not in sql
    assert params[0] == "Closed - Shared with customer"
    assert params[1] == [101]


def test_closing_completes_the_task_and_never_deletes_it():
    cursor = _ScriptedCursor({})
    close_pending_calls_for_candidates(cursor, [101, 102], "For Future")
    sql, _ = cursor.executed[0]

    assert "DELETE" not in sql.upper()
    assert "SET status = 'completed'" in sql
    assert "completed_at = NOW()" in sql
    assert "AND status = 'pending'" in sql   # completed history is untouched


def test_closing_nothing_costs_no_query():
    cursor = _ScriptedCursor({})
    assert close_pending_calls_for_candidates(cursor, [], "Rejected") == 0
    assert cursor.executed == []


def test_sweep_targets_only_pending_rows_of_retired_candidates():
    class _Cur(_ScriptedCursor):
        pass

    cursor = _Cur({})
    conn = _Connection(cursor)
    calls.sweep_terminal_candidate_call_tasks(conn)
    sql, _ = cursor.executed[0]

    assert "c.status = 'pending'" in sql
    assert "'shared with customer'" in sql
    assert "Closed - ' || TRIM(cand.status)" in sql


# ── 3. a retired candidate never appears in the active queue ────────────────

def _cached_calls(monkeypatch, rows):
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls, "refresh_call_caches_async", lambda: None)
    monkeypatch.setattr(calls, "_calls_cache", rows, raising=False)
    monkeypatch.setattr(calls, "_cache_warmed_at", float("inf"), raising=False)
    return lambda status: calls.get_calls(
        status=status, list_id=None, due_filter=None, range_=None,
        date_from=None, date_to=None, outcome_group=None, current_user=USER,
    )


def _call_row(**overrides):
    row = {
        "id": 1, "candidate_id": 101, "created_by": "rec@example.com",
        "status": "pending", "candidate_status": "Reached out - Phone",
        "due_date": None, "created_at": None, "completed_at": None, "outcome": None,
    }
    row.update(overrides)
    return row


def test_pending_task_for_a_retired_candidate_is_hidden(monkeypatch):
    fetch = _cached_calls(monkeypatch, [
        _call_row(id=1, candidate_status="For Future"),
        _call_row(id=2, candidate_status="Shared with customer"),
        _call_row(id=3),
    ])

    assert [c["id"] for c in fetch("pending")] == [3]


def test_completed_tasks_stay_in_the_log(monkeypatch):
    fetch = _cached_calls(monkeypatch, [
        _call_row(id=1, status="completed", outcome="Closed - For Future",
                  candidate_status="For Future"),
    ])

    # The framework asks for the record to be retained, shown as completed.
    assert [c["id"] for c in fetch("completed")] == [1]


# ── 4. and cannot be added back to a list ───────────────────────────────────

def _add_candidates(monkeypatch, counts, candidate_ids=(101,)):
    """counts = (list_found, duplicates, inserted, requested, callable, retired)"""
    cursor = _ScriptedCursor({"WITH target_list AS": counts})
    connection = _Connection(cursor)
    monkeypatch.setattr(calls, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: connection)
    monkeypatch.setattr(calls, "return_db_connection", lambda _c: None)
    monkeypatch.setattr(calls, "invalidate_calls_cache", lambda: None)
    monkeypatch.setattr(calls, "evict_call_list_from_cache", lambda _id: None)
    request = calls.AddCandidatesRequest(list_id=55, candidate_ids=list(candidate_ids))
    return calls.add_candidates_to_list(request, current_user=USER), cursor


def test_adding_a_retired_candidate_to_a_list_is_refused(monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _add_candidates(monkeypatch, (1, 0, 0, 1, 0, 1))

    assert excinfo.value.status_code == 400
    assert "removed them from calling" in excinfo.value.detail


def test_retired_candidates_are_reported_separately_from_phoneless_ones(monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _add_candidates(monkeypatch, (1, 0, 0, 2, 0, 0), candidate_ids=(101, 102))

    # Nobody was retired here, so the reason must still be the phone one.
    assert "phone number" in excinfo.value.detail


def test_the_insert_filters_on_status_not_just_on_having_a_phone(monkeypatch):
    _, cursor = _add_candidates(monkeypatch, (1, 0, 1, 1, 1, 0))
    sql, _ = cursor.executed[0]

    assert "mobile_phone" in sql            # pre-existing filter intact
    assert "'shared with customer'" in sql  # …and the status rule alongside it

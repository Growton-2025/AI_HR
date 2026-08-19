"""One call must not read as two completed entries.

A recruiter reported that Shreya Shroff showed two entries in Completed for a
single call. She was right. Shreya is a candidate for two roles, both call
lists were built the same morning (04:20:24 and 04:29:46), and each build gave
her its own "Call 1 - Day 1 - Second Half". She was dialled once from the EMEA
list; marking her "High CTC" retired the untouched Hevo task, which landed in
Completed looking exactly like a second call.

Four faults behind that one report:

1. De-duplication was per list only, so one person could hold a live calling
   thread in several lists at once — and be dialled twice for two roles.
2. A retired task rendered in the same pill as a real outcome, and counted
   toward Completed.
3. Logging "Connected - Not Interested" overwrote the "High CTC" she had set
   51 seconds earlier. Five candidates had lost their recorded reason this way.
4. Nothing recorded who changed a status, so none of the above could be
   reconstructed without inferring from synthetic outcome strings.
"""

import pathlib

import pytest

from backend.api import schemas
from backend.api.routes import calls as calls_route
from backend.services import candidate_status_log as status_log

CALLS_JSX = (
    pathlib.Path(__file__).resolve().parent.parent
    / "frontend" / "src" / "components" / "CandidateActivityPanel.jsx"
).read_text()

USER = schemas.User(id=7, username="rec", email="rec@example.com", role="recruiter")


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

    def ran(self, fragment):
        return [sql for sql, _ in self.executed if fragment in sql]


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


def _add_candidates(monkeypatch, counts, candidate_ids=(101,)):
    """counts = (list_found, duplicates, inserted, requested, callable,
                 retired, open_elsewhere)"""
    cursor = _Cursor([counts])
    monkeypatch.setattr(calls_route, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls_route, "get_calls_db_connection", lambda: _Conn(cursor))
    monkeypatch.setattr(calls_route, "return_db_connection", lambda _c: None)
    monkeypatch.setattr(calls_route, "invalidate_calls_cache", lambda: None)
    monkeypatch.setattr(calls_route, "evict_call_list_from_cache", lambda _id: None)
    request = calls_route.AddCandidatesRequest(list_id=55, candidate_ids=list(candidate_ids))
    return calls_route.add_candidates_to_list(request, current_user=USER), cursor


# ── 1. one person, one live calling thread ──────────────────────────────────

def test_a_candidate_already_being_called_elsewhere_is_refused(monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _add_candidates(monkeypatch, (1, 0, 0, 1, 0, 0, 1))

    assert excinfo.value.status_code == 400
    assert "already in another call list" in excinfo.value.detail


def test_the_insert_excludes_candidates_open_in_another_list(monkeypatch):
    _, cursor = _add_candidates(monkeypatch, (1, 0, 1, 1, 1, 0, 0))
    sql = cursor.executed[0][0]

    assert "open_elsewhere_ids" in sql
    assert "other.list_id <> %s" in sql
    assert "other.status IN ('pending', 'in_progress')" in sql
    assert "c_id NOT IN (SELECT c_id FROM open_elsewhere_ids)" in sql


def test_the_skipped_reasons_do_not_double_count(monkeypatch):
    from fastapi import HTTPException

    # 3 requested, 0 callable: 1 retired, 1 open elsewhere — so exactly 1 is
    # phoneless. Reporting it as 3 would be nonsense to the recruiter.
    with pytest.raises(HTTPException) as excinfo:
        _add_candidates(monkeypatch, (1, 0, 0, 3, 0, 1, 1), candidate_ids=(101, 102, 103))

    # The retired reason is reported first, being the more decisive one, and
    # it names one candidate rather than lumping all three skips together.
    assert "from calling" in excinfo.value.detail
    assert "3" not in excinfo.value.detail


def test_the_auto_list_builder_applies_the_same_rule():
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backend" / "services" / "auto_call_list.py"
    ).read_text()

    # Both roles' lists were built by this path on the morning of the report.
    assert "other.list_id <> %s" in source
    assert "other.status IN ('pending', 'in_progress')" in source


# ── 2. a retired task is not a call ─────────────────────────────────────────

def test_retired_tasks_are_left_out_of_the_completed_count():
    import inspect

    source = inspect.getsource(calls_route.get_call_stats)

    assert "NOT LIKE 'Closed - %%'" in source


def _completed_via_cache(monkeypatch, rows):
    monkeypatch.setattr(calls_route, "ensure_calls_schema_ready", lambda *a, **k: None)
    monkeypatch.setattr(calls_route, "refresh_call_caches_async", lambda: None)
    monkeypatch.setattr(calls_route, "_calls_cache", rows, raising=False)
    monkeypatch.setattr(calls_route, "_cache_warmed_at", float("inf"), raising=False)
    return calls_route.get_calls(
        status="completed", list_id=None, due_filter=None, range_=None,
        date_from=None, date_to=None, outcome_group=None, current_user=USER,
    )


def _completed_row(**overrides):
    row = {
        "id": 1, "candidate_id": 13624, "created_by": "rec@example.com",
        "status": "completed", "candidate_status": "High CTC",
        "outcome": "Connected - Not Interested", "duration": 227,
        "due_date": None, "created_at": None, "completed_at": None,
    }
    row.update(overrides)
    return row


def test_the_retired_task_is_not_listed_as_a_call(monkeypatch):
    # Exactly Shreya's two rows: one dialled from EMEA, one retired in Hevo.
    rows = _completed_via_cache(monkeypatch, [
        _completed_row(id=961, outcome="Connected - Not Interested", duration=227),
        _completed_row(id=923, outcome="Closed - High CTC", duration=0),
    ])

    assert [r["id"] for r in rows] == [961]


def test_a_real_outcome_that_merely_starts_with_closed_is_not_hidden(monkeypatch):
    # The prefix has to be the synthetic one, not any outcome mentioning it.
    rows = _completed_via_cache(monkeypatch, [
        _completed_row(id=1, outcome="Connected - Not Interested"),
        _completed_row(id=2, outcome="Not Connected"),
        _completed_row(id=3, outcome="Unreachable"),
    ])

    assert [r["id"] for r in rows] == [1, 2, 3]


def test_the_record_is_retained_on_the_candidate():
    import inspect
    from backend.api.routes import candidates as candidates_route

    source = inspect.getsource(candidates_route.get_candidate_activity)

    # The framework asks for the task to be retained in the log. It is — the
    # candidate's Activity History shows every completed row, including the
    # retired one, which is why hiding it from the call list is safe.
    assert "c.status = 'completed'" in source
    assert "Closed" not in source


def test_the_badge_says_it_was_never_called():
    assert "Not called — " in CALLS_JSX
    assert "RETIRED_OUTCOME_PREFIX = 'Closed - '" in CALLS_JSX
    # And explains itself on hover rather than leaving the recruiter guessing.
    assert "Not a call. This task was removed from calling" in CALLS_JSX


# ── 3. a hand-set reason survives the call outcome ──────────────────────────

def test_the_outcome_will_not_overwrite_a_terminal_status():
    import inspect

    source = inspect.getsource(calls_route.update_call)

    # Shreya set High CTC; 51 seconds later the outcome replaced it.
    assert "UPDATE candidates" in source
    assert "AND NOT {terminal_candidate_status_sql('status')}" in source
    # …and that predicate really does cover the status she chose.
    assert "'high ctc'" in calls_route.terminal_candidate_status_sql("status")


# ── 4. every status change leaves a trace ───────────────────────────────────

def test_a_status_change_is_recorded_with_who_and_from_what():
    cursor = _Cursor()
    status_log.record_status_change(
        cursor, 13624, "High CTC", "Not Interested", "Jaya@Growton.co", "call_outcome",
    )
    insert = [sql for sql in cursor.ran("INSERT INTO candidate_status_history")]

    assert insert
    params = cursor.executed[-1][1]
    assert params[0] == 13624
    assert params[1] == "High CTC"          # what it was
    assert params[2] == "Not Interested"    # what it became
    assert params[3] == "jaya@growton.co"   # who did it, normalised
    assert params[4] == "call_outcome"      # and by which route


def test_the_bulk_variant_reads_the_old_values_before_they_are_lost():
    cursor = _Cursor(rows=[(1, "Shortlisted"), (2, "Reached out - Phone")])
    status_log.record_status_changes(cursor, [1, 2], "Rejected", "rec@example.com", "bulk")

    selects = cursor.ran("SELECT id, status FROM candidates")
    assert selects, "previous statuses must be read before the UPDATE"
    logged = [params for sql, params in cursor.executed if "INSERT" in sql]
    assert {p[1] for p in logged} == {"Shortlisted", "Reached out - Phone"}


def test_logging_never_blocks_the_status_change():
    class _Broken(_Cursor):
        def execute(self, sql, params=None):
            raise RuntimeError("table is missing")

    # No exception escapes: the recruiter's status change matters more than
    # our record of it.
    status_log.record_status_change(_Broken(), 1, "a", "b", "who", "src")


def test_history_reads_newest_first():
    cursor = _Cursor(rows=[("High CTC", "Not Interested", "jaya", "call_outcome", None)])
    history = status_log.fetch_status_history(cursor, 13624)

    assert history[0]["old_status"] == "High CTC"
    assert "ORDER BY changed_at DESC" in cursor.executed[0][0]

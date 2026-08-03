import asyncio

import pytest
from fastapi import HTTPException

from backend.api import schemas
from backend.api.routes import browse


class _Cursor:
    def __init__(self, rows=()):
        self.executed = []
        self._rows = list(rows)
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        self.rowcount = len(self._rows)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        return None


class _Conn:
    def __init__(self, rows=()):
        self.cursor_obj = _Cursor(rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _user(role="recruiter", uid=19):
    return schemas.User(id=uid, username="r@x.com", email="r@x.com", role=role)


def _run(update, user):
    return asyncio.run(browse.bulk_update_status(update, user))


def _patch(monkeypatch, conn, allowed_ids):
    conn.cursor_obj._rows = [(i,) for i in allowed_ids]
    monkeypatch.setattr(browse, "get_db_connection", lambda *a, **k: conn)
    monkeypatch.setattr(browse, "return_db_connection", lambda c: None)
    monkeypatch.setattr(browse, "PROFILES_BY_ID", {})
    import backend.api.routes.candidates as candidates_mod
    import backend.api.routes.roles as roles_mod
    monkeypatch.setattr(candidates_mod, "invalidate_candidate_count_caches",
                        lambda **kw: None, raising=False)
    monkeypatch.setattr(roles_mod, "invalidate_role_detail_cache_for_candidate",
                        lambda cid: None, raising=False)


def test_unknown_status_is_rejected_before_any_write(monkeypatch):
    """The dropdown is a fixed vocabulary. A typo would create a stage nothing
    can filter by — on an unbounded number of rows."""
    conn = _Conn()
    _patch(monkeypatch, conn, [])

    with pytest.raises(HTTPException) as exc:
        _run(browse.BulkStatusUpdate(candidate_ids=[1, 2], status="Shortlistd"), _user())

    assert exc.value.status_code == 400
    assert conn.cursor_obj.executed == []


def test_every_recruitment_stage_is_accepted(monkeypatch):
    """Guards against the UI list and the backend validator drifting apart."""
    for stage in browse.RECRUITMENT_STAGES:
        conn = _Conn()
        _patch(monkeypatch, conn, [1])
        res = _run(browse.BulkStatusUpdate(candidate_ids=[1], status=stage), _user("admin"))
        assert res["updated"] == 1, stage


def test_batch_size_is_capped(monkeypatch):
    conn = _Conn()
    _patch(monkeypatch, conn, [])
    too_many = list(range(1, browse.BULK_STATUS_MAX + 2))

    with pytest.raises(HTTPException) as exc:
        _run(browse.BulkStatusUpdate(candidate_ids=too_many, status="Shortlisted"), _user())

    assert exc.value.status_code == 400
    assert conn.cursor_obj.executed == []


def test_empty_selection_is_a_no_op(monkeypatch):
    conn = _Conn()
    _patch(monkeypatch, conn, [])
    res = _run(browse.BulkStatusUpdate(candidate_ids=[], status="Shortlisted"), _user())
    assert res == {"success": True, "updated": 0, "skipped": 0, "status": "Shortlisted"}
    assert conn.cursor_obj.executed == []


def test_unauthorized_ids_are_skipped_not_fatal(monkeypatch):
    """One foreign id in a 77-row selection must not fail the batch, and rows
    the user cannot edit must not be written."""
    conn = _Conn()
    _patch(monkeypatch, conn, [10, 11])  # only 2 of the 4 requested are permitted

    res = _run(browse.BulkStatusUpdate(candidate_ids=[10, 11, 12, 13], status="Shortlisted"), _user())

    assert res["updated"] == 2
    assert res["skipped"] == 2
    assert res["updated_ids"] == [10, 11]
    # The UPDATE must target only the permitted ids.
    update_sql = [e for e in conn.cursor_obj.executed if "UPDATE candidates" in e[0]]
    assert len(update_sql) == 1
    assert update_sql[0][1] == ("Shortlisted", [10, 11])


def test_admins_skip_the_ownership_join(monkeypatch):
    conn = _Conn()
    _patch(monkeypatch, conn, [1, 2])
    _run(browse.BulkStatusUpdate(candidate_ids=[1, 2], status="Shortlisted"), _user("admin"))

    select_sql = conn.cursor_obj.executed[0][0]
    assert "recruitment_role_candidates" not in select_sql
    assert "is_archived" in select_sql


def test_non_admins_are_scoped_by_ownership_or_role_access(monkeypatch):
    """Mirrors _authorize_candidate_update(allow_role_access=True) as a set."""
    conn = _Conn()
    _patch(monkeypatch, conn, [1])
    _run(browse.BulkStatusUpdate(candidate_ids=[1], status="Shortlisted"), _user())

    select_sql = conn.cursor_obj.executed[0][0]
    assert "owner_user_id = %s" in select_sql
    assert "recruitment_role_candidates" in select_sql
    assert "is_archived" in select_sql


def test_the_whole_batch_updates_in_one_statement(monkeypatch):
    """Looping the single endpoint would be one round trip per candidate against
    a remote DB costing ~0.6s per statement."""
    conn = _Conn()
    ids = list(range(1, 78))
    _patch(monkeypatch, conn, ids)

    _run(browse.BulkStatusUpdate(candidate_ids=ids, status="Shortlisted"), _user("admin"))

    update_sql = [e for e in conn.cursor_obj.executed if "UPDATE candidates" in e[0]]
    assert len(update_sql) == 1
    assert "ANY(%s::int[])" in update_sql[0][0]

import asyncio

from backend.api import schemas
from backend.api.routes import roles


class _Cursor:
    def __init__(self):
        self.calls = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return (42, "Founding AE", "Close enterprise deals.")


class _Connection:
    def __init__(self, cursor):
        self.cursor_value = cursor
        self.committed = False

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.committed = True


def test_update_role_saves_trimmed_job_description(monkeypatch):
    cursor = _Cursor()
    conn = _Connection(cursor)
    monkeypatch.setattr(roles, "get_db_connection", lambda: conn)
    monkeypatch.setattr(roles, "return_db_connection", lambda _conn: None)

    result = asyncio.run(
        roles.update_role(
            "Founding AE",
            schemas.RoleUpdate(job_description="  Close enterprise deals.  "),
            current_user=schemas.User(id=7, username="recruiter@example.com"),
        )
    )

    assert result["job_description"] == "Close enterprise deals."
    assert conn.committed is True
    assert cursor.calls[0][1] == ("Close enterprise deals.", 7, "Founding AE")

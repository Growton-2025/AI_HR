import asyncio
from contextlib import contextmanager

from backend.api.routes import enrichment


class _Request:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.cursor_instance = _Cursor(rows)
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


def test_clean_val_removes_clay_invisible_blank_values():
    assert enrichment.clean_val(" \ufeff\u200b ") is None
    assert enrichment.clean_val("  candidate@example.com  ") == "candidate@example.com"
    assert enrichment.clean_val("Not Found") is None


def test_clay_callback_matches_legacy_raw_linkedin_and_refreshes_subset(monkeypatch):
    conn = _Connection([
        (577, "https://www.linkedin.com/in/example-person/?trk=public_profile", None),
    ])

    @contextmanager
    def fake_connection(**_kwargs):
        yield conn

    cache_updates = []
    subset_refreshes = []
    monkeypatch.setattr(enrichment, "get_db_connection_context", fake_connection)
    monkeypatch.setattr(
        enrichment.query,
        "update_candidate_contact",
        lambda *args, **kwargs: cache_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        enrichment.query,
        "refresh_profiles_in_cache",
        lambda ids: subset_refreshes.append(ids) or len(ids),
    )

    response = asyncio.run(enrichment.receive_results(_Request({
        "first_name": "Example",
        "last_name": "Person",
        "linkedin_url": "linkedin.com/in/example-person",
        "result_email": "person@example.com",
        "mobile_phone": "+91 99999 00000",
    })))

    assert response["status"] == "updated"
    assert response["updated_candidates"] == 1
    assert conn.committed is True
    assert subset_refreshes == [[577]]
    assert len(cache_updates) == 1
    update_sql, update_params = conn.cursor_instance.executed[1]
    assert "WHERE id = ANY(%s)" in update_sql
    assert update_params[-1] == [577]


def test_clay_callback_reports_no_contact_without_database_reload(monkeypatch):
    cache_updates = []
    monkeypatch.setattr(
        enrichment.query,
        "update_candidate_contact",
        lambda *args, **kwargs: cache_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        enrichment.query,
        "initialize_cache",
        lambda: (_ for _ in ()).throw(AssertionError("full cache reload must not run")),
    )

    response = asyncio.run(enrichment.receive_results(_Request({
        "linkedin_url": "https://linkedin.com/in/example-person",
        "result_email": "\ufeff",
        "mobile_phone": "Not Found",
    })))

    assert response == {
        "status": "no_contact",
        "matched_candidates": 0,
        "updated_candidates": 0,
        "has_email": False,
        "has_phone": False,
    }
    assert len(cache_updates) == 1

import asyncio

import pytest
from fastapi import HTTPException

from backend.api import schemas
from backend.api.routes import candidates
from backend.pipeline import query


@pytest.fixture
def isolated_profile_cache(monkeypatch):
    profiles = {}
    companies = []
    profile_list = []
    monkeypatch.setattr(query, "PROFILES_BY_ID", profiles)
    monkeypatch.setattr(query, "ALL_COMPANY_NAMES", companies)
    monkeypatch.setattr(query, "_PROFILES_CACHE", profile_list)
    monkeypatch.setattr(query, "CACHE_INITIALIZED", False)
    return profiles, companies


def test_initialize_cache_preserves_existing_cache_when_refresh_returns_false_empty(
    monkeypatch,
    isolated_profile_cache,
):
    profiles, companies = isolated_profile_cache
    profiles[1] = {"id": 1, "name": "Existing"}
    companies.append("Existing Co")
    monkeypatch.setattr(query, "CACHE_INITIALIZED", True)
    monkeypatch.setattr(query, "load_all_profiles_from_db", lambda: [])
    monkeypatch.setattr(query, "count_active_candidates_from_db", lambda: 10)
    monkeypatch.setattr(query, "load_all_company_names_from_db", lambda: ["New Co"])

    assert query.initialize_cache() is False
    assert query.CACHE_INITIALIZED is True
    assert query.PROFILES_BY_ID == {1: {"id": 1, "name": "Existing"}}
    assert query.ALL_COMPANY_NAMES == ["Existing Co"]


def test_initialize_cache_does_not_mark_empty_cache_initialized_when_db_has_rows(
    monkeypatch,
    isolated_profile_cache,
):
    monkeypatch.setattr(query, "load_all_profiles_from_db", lambda: [])
    monkeypatch.setattr(query, "count_active_candidates_from_db", lambda: 10)
    monkeypatch.setattr(query, "load_all_company_names_from_db", lambda: ["New Co"])

    assert query.initialize_cache() is False
    assert query.CACHE_INITIALIZED is False
    assert query.PROFILES_BY_ID == {}
    assert query.ALL_COMPANY_NAMES == []


def test_initialize_cache_allows_empty_cache_when_db_is_empty(
    monkeypatch,
    isolated_profile_cache,
):
    monkeypatch.setattr(query, "load_all_profiles_from_db", lambda: [])
    monkeypatch.setattr(query, "count_active_candidates_from_db", lambda: 0)
    monkeypatch.setattr(query, "load_all_company_names_from_db", lambda: [])

    assert query.initialize_cache() is True
    assert query.CACHE_INITIALIZED is True
    assert query.PROFILES_BY_ID == {}


def test_initialize_cache_preserves_existing_cache_when_loader_raises(
    monkeypatch,
    isolated_profile_cache,
):
    profiles, _companies = isolated_profile_cache
    profiles[1] = {"id": 1, "name": "Existing"}
    monkeypatch.setattr(query, "CACHE_INITIALIZED", True)

    def fail_load():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(query, "load_all_profiles_from_db", fail_load)

    assert query.initialize_cache() is False
    assert query.CACHE_INITIALIZED is True
    assert query.PROFILES_BY_ID == {1: {"id": 1, "name": "Existing"}}


def test_candidate_analytics_returns_503_when_active_db_cache_stays_empty(monkeypatch):
    calls = []

    def fake_initialize_cache():
        calls.append("init")

    user = schemas.User(
        id=29,
        username="ashwin@example.com",
        email="ashwin@example.com",
        full_name="Ashwin",
        role="recruiter",
        permissions={},
    )
    candidates._analytics_cache.clear()
    monkeypatch.setattr(candidates, "PROFILES_BY_ID", {})
    monkeypatch.setattr(candidates, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(candidates, "initialize_cache", fake_initialize_cache)
    monkeypatch.setattr(candidates, "count_active_candidates_from_db", lambda: 4174)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(candidates.get_candidate_analytics(current_user=user))

    assert calls == ["init"]
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "profile_cache_unavailable"

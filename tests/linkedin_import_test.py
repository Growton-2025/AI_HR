"""Orchestration tests for the LinkedIn URL import (no network, no database)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
from fastapi import HTTPException

from backend.api import schemas
from backend.api.routes import candidate_imports
from backend.services import linkedin_import as li


ADA = "https://www.linkedin.com/in/ada-lovelace/"
ALAN = "https://linkedin.com/in/alan-turing"


def _row(url: str, name: str, company: str) -> Dict[str, Any]:
    return {"linkedinUrl": url, "fullName": name, "experiences/0/companyName": company}


@pytest.fixture
def fake_db(monkeypatch):
    """Replace every DB helper with in-memory stand-ins and record what happened."""
    state: Dict[str, Any] = {"progress": [], "existing": {}, "deleted": [], "persisted": [], "roles": [],
                             "mapped": [], "companies_asked": []}

    monkeypatch.setattr(li, "write_progress", lambda upload_id, **u: state["progress"].append(u))
    monkeypatch.setattr(li, "existing_candidates", lambda key, owner: state["existing"].get(key, []))
    monkeypatch.setattr(li, "delete_candidates", lambda ids: state["deleted"].extend(ids) or len(ids))
    monkeypatch.setattr(li, "known_companies", lambda names: state["companies_asked"].append(list(names)) or {})
    monkeypatch.setattr(li, "assign_role", lambda role_id, cid: state["roles"].append((role_id, cid)) or bool(role_id))
    monkeypatch.setattr(li, "refresh_caches", lambda ids: None)

    def fake_map(raw, **kwargs):
        state["mapped"].append(raw["fullName"])
        return {"candidates": {"name": raw["fullName"]}, "coverage": {"tables_empty": []}}

    def fake_persist(tables, **kwargs):
        state["persisted"].append({"tables": tables, **kwargs})
        return {"candidate_id": 100 + len(state["persisted"]), "counts": {"roles": 1}}

    monkeypatch.setattr(li, "map_scraped_profile", fake_map)
    monkeypatch.setattr(li, "persist_mapped_profile", fake_persist)
    return state


def test_validate_urls_dedupes_and_rejects_non_profiles():
    checked = li.validate_linkedin_urls([ADA, "https://www.linkedin.com/in/ada-lovelace", "",
                                         "https://linkedin.com/company/acme", "not a url"])
    assert checked["valid"] == [ADA]
    assert checked["invalid"] == ["https://linkedin.com/company/acme", "not a url"]


def test_run_inserts_each_scraped_profile_and_reports_missing(fake_db, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: {
        "rows": [_row(ADA, "Ada Lovelace", "Analytical Engines")], "failed": [ALAN], "requested": 2})

    result = li.run_linkedin_import(upload_id=7, urls=[ADA, ALAN], owner_user_id=28, role_id=5)

    assert result["status"] == "completed_with_errors"
    assert result["inserted"] == 1 and result["skipped"] == 1 and result["updated"] == 0
    assert result["role_assigned_count"] == 1
    assert fake_db["mapped"] == ["Ada Lovelace"]
    saved = fake_db["persisted"][0]
    assert saved["dry_run"] is False
    assert saved["owner_user_id"] == 28
    assert saved["created_by"] == li.CREATED_BY
    assert saved["extra_candidate_fields"] == {"source_upload_ids": [7]}
    assert fake_db["roles"] == [(5, 101)]
    assert fake_db["companies_asked"] == [["Analytical Engines"]]
    assert any("no profile returned by Apify" in e for e in result["errors"])
    statuses = [u["status"] for u in fake_db["progress"] if "status" in u]
    assert statuses == ["scraping", "enriching", "completed_with_errors"]
    final_counts = [u for u in fake_db["progress"] if "processed_count" in u][-1]
    assert final_counts["processed_count"] == 2 and final_counts["inserted_count"] == 1


def test_duplicate_skip_by_default_and_replace_deletes_owned_rows(fake_db, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: {
        "rows": [_row(ADA, "Ada Lovelace", "Analytical Engines")], "failed": [], "requested": 1})
    fake_db["existing"]["/in/ada-lovelace"] = [{"id": 42, "pool_source": "recruiter_upload"}]

    scrape_calls: List[List[str]] = []

    def counting_scrape(urls):
        scrape_calls.append(list(urls))
        return {"rows": [_row(ADA, "Ada Lovelace", "Analytical Engines")], "failed": [], "requested": len(urls)}

    monkeypatch.setattr(li, "scrape_profiles", counting_scrape)

    skipped = li.run_linkedin_import(upload_id=1, urls=[ADA], owner_user_id=28)
    assert skipped["status"] == "completed_with_errors"
    assert skipped["skipped"] == 1 and skipped["inserted"] == 0
    assert fake_db["mapped"] == [] and fake_db["deleted"] == []
    assert skipped["results"][0]["candidate_id"] == 42
    assert scrape_calls == []  # skipped duplicates are never sent to Apify (it bills per profile)

    replaced = li.run_linkedin_import(upload_id=2, urls=[ADA], owner_user_id=28, duplicate_policy="replace")
    assert replaced["status"] == "completed"
    assert replaced["updated"] == 1 and replaced["inserted"] == 0
    assert fake_db["deleted"] == [42]
    assert scrape_calls == [[ADA]]
    assert len(fake_db["persisted"]) == 1


def test_apify_failure_marks_upload_failed(fake_db, monkeypatch):
    def boom(urls):
        raise li.ApifyScrapeError("APIFY_API_TOKEN is not set")

    monkeypatch.setattr(li, "scrape_profiles", boom)
    result = li.run_linkedin_import(upload_id=3, urls=[ADA], owner_user_id=28)
    assert result["status"] == "failed"
    assert fake_db["progress"][-1]["status"] == "failed"
    assert "APIFY_API_TOKEN" in fake_db["progress"][-1]["error_message"]


def test_one_bad_profile_does_not_sink_the_batch(fake_db, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: {
        "rows": [_row(ADA, "Ada Lovelace", "A"), _row(ALAN, "Alan Turing", "B")], "failed": [], "requested": 2})
    real_map = li.map_scraped_profile

    def flaky(raw, **kw):
        if raw["fullName"] == "Ada Lovelace":
            raise RuntimeError("model exploded")
        return real_map(raw, **kw)

    monkeypatch.setattr(li, "map_scraped_profile", flaky)
    result = li.run_linkedin_import(upload_id=4, urls=[ADA, ALAN], owner_user_id=28)
    assert result["inserted"] == 1 and result["skipped"] == 1
    assert result["results"][0]["status"] == "error" and result["results"][1]["status"] == "inserted"


def _user(role: str = "recruiter") -> schemas.User:
    return schemas.User(id=28, username="r@x.co", email="r@x.co", role=role)


def test_route_rejects_bad_urls_admins_and_over_cap(monkeypatch):
    body = candidate_imports.LinkedInImportBody(linkedin_urls=["https://linkedin.com/company/acme"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(candidate_imports.import_linkedin_profiles(body, current_user=_user()))
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        asyncio.run(candidate_imports.import_linkedin_profiles(
            candidate_imports.LinkedInImportBody(linkedin_urls=[ADA]), current_user=_user("admin")))
    assert exc.value.status_code == 403

    monkeypatch.setenv("APIFY_MAX_PROFILES_PER_REQUEST", "1")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(candidate_imports.import_linkedin_profiles(
            candidate_imports.LinkedInImportBody(linkedin_urls=[ADA, ALAN]), current_user=_user()))
    assert exc.value.status_code == 400 and "At most 1" in str(exc.value.detail)


def test_route_starts_background_import(monkeypatch):
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(li, "start_linkedin_import", lambda **kw: calls.append(kw) or 99)
    body = candidate_imports.LinkedInImportBody(linkedin_urls=[ADA, ADA.rstrip("/")], duplicate_policy="REPLACE")
    out = asyncio.run(candidate_imports.import_linkedin_profiles(body, current_user=_user()))
    assert out["upload_id"] == 99 and out["row_count"] == 1 and out["status"] == "processing"
    assert calls == [{"urls": [ADA], "owner_user_id": 28, "role_id": None, "duplicate_policy": "replace"}]

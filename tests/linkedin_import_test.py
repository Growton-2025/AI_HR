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
    monkeypatch.setattr(li, "existing_candidates_bulk",
                        lambda keys, owner: {k: state["existing"].get(k, []) for k in keys})
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


# ------------------------------------------------------------- pipeline: speed + order
def _urls(n: int) -> List[str]:
    return [f"https://www.linkedin.com/in/person-{i}" for i in range(n)]


def test_scrape_runs_overlap_with_mapping_and_results_stay_in_sheet_order(fake_db, monkeypatch):
    """Later actor runs go out side by side, profiles map while runs are still out,
    and the returned results are in sheet order however the work finished."""
    import threading
    import time

    monkeypatch.setenv("APIFY_MAX_PROFILES_PER_REQUEST", "10")
    monkeypatch.setenv("APIFY_PARALLEL_RUNS", "2")
    monkeypatch.setenv("LINKEDIN_IMPORT_MAP_WORKERS", "4")
    lock = threading.Lock()
    live = {"scrapes": 0, "max_scrapes": 0, "maps": 0, "max_maps": 0, "mapped_while_scraping": False}

    def scrape(urls):
        with lock:
            live["scrapes"] += 1
            live["max_scrapes"] = max(live["max_scrapes"], live["scrapes"])
        time.sleep(0.15)
        with lock:
            live["scrapes"] -= 1
        return {"rows": [_row(u, u, "Acme") for u in urls], "failed": [], "requested": len(urls)}

    def slow_map(raw, **kw):
        with lock:
            live["maps"] += 1
            live["max_maps"] = max(live["max_maps"], live["maps"])
            if live["scrapes"]:
                live["mapped_while_scraping"] = True
        time.sleep(0.02)
        with lock:
            live["maps"] -= 1
        return {"candidates": {"name": raw["fullName"]}, "coverage": {}}

    monkeypatch.setattr(li, "scrape_profiles", scrape)
    monkeypatch.setattr(li, "map_scraped_profile", slow_map)
    urls = _urls(40)                       # 4 actor runs of 10
    started = time.monotonic()
    result = li.run_linkedin_import(upload_id=9, urls=urls, owner_user_id=28)
    elapsed = time.monotonic() - started

    assert result["inserted"] == 40 and result["skipped"] == 0 and result["status"] == "completed"
    assert [r["url"] for r in result["results"]] == urls
    assert sorted(p["tables"]["candidates"]["name"] for p in fake_db["persisted"]) == sorted(urls)
    assert live["max_scrapes"] == 2                      # the first run alone, then two at a time
    assert live["max_maps"] > 1 and live["mapped_while_scraping"]
    assert elapsed < 0.15 * 4 + 0.02 * 40                # faster than doing it all one after another
    assert result["apify_requested"] == 40


def test_a_failed_later_run_only_loses_its_own_profiles(fake_db, monkeypatch):
    monkeypatch.setenv("APIFY_MAX_PROFILES_PER_REQUEST", "2")
    urls = _urls(6)                        # runs: [0,1] [2,3] [4,5]

    def scrape(batch):
        if batch == urls[2:4]:
            raise li.ApifyScrapeError("quota exceeded")
        return {"rows": [_row(u, u, "Acme") for u in batch], "failed": [], "requested": len(batch)}

    monkeypatch.setattr(li, "scrape_profiles", scrape)
    result = li.run_linkedin_import(upload_id=10, urls=urls, owner_user_id=28, defer_missing=True)

    assert [r["status"] for r in result["results"]] == ["inserted", "inserted", "not_scraped", "not_scraped",
                                                        "inserted", "inserted"]
    assert result["inserted"] == 4 and result["skipped"] == 2 and result["processed"] == 6
    assert result["deferred"] == []


def test_first_run_failing_stops_before_any_other_run_is_paid_for(fake_db, monkeypatch):
    monkeypatch.setenv("APIFY_MAX_PROFILES_PER_REQUEST", "2")
    calls: List[List[str]] = []

    def scrape(batch):
        calls.append(list(batch))
        raise li.ApifyScrapeError("APIFY_API_TOKEN is not set")

    monkeypatch.setattr(li, "scrape_profiles", scrape)
    result = li.run_linkedin_import(upload_id=11, urls=_urls(6), owner_user_id=28)
    assert result["status"] == "failed" and len(calls) == 1


def test_each_employer_is_researched_once_across_the_run(monkeypatch):
    """Real mapper, stubbed network: two profiles sharing an employer cost one research call."""
    import threading

    from backend.services import profile_mapper as pm

    lock = threading.Lock()
    researched: List[str] = []

    def fake_research(name, hint=""):
        with lock:
            researched.append(name)
        return {"funding_stage": "Seed", "product_service": "Widgets"}

    monkeypatch.setattr(pm, "research_company", fake_research)
    monkeypatch.setattr(pm.ie, "classify_company", lambda name, **kw: {"product_service": "Widgets", "industry": "Tools"})
    cache = pm.CompanyCache()

    def raw(name: str) -> Dict[str, Any]:
        return {"fullName": name, "linkedinUrl": f"https://www.linkedin.com/in/{name}",
                "experiences/0/companyName": "Acme", "experiences/0/title": "Engineer",
                "experiences/1/companyName": f"{name} Labs", "experiences/1/title": "Intern"}

    first = pm.map_scraped_profile(raw("ada"), use_llm=False, company_cache=cache)
    second = pm.map_scraped_profile(raw("alan"), use_llm=False, company_cache=cache)

    assert sorted(researched) == ["Acme", "ada Labs", "alan Labs"]
    assert {c["name"] for c in first["companies"]} == {"Acme", "ada Labs"}
    assert {c["name"] for c in second["companies"]} == {"Acme", "alan Labs"}
    # without a shared cache every profile pays again
    pm.map_scraped_profile(raw("ada"), use_llm=False)
    assert researched.count("Acme") == 2


def test_saves_run_a_few_at_a_time_and_progress_only_moves_forward(fake_db, monkeypatch):
    import threading
    import time

    monkeypatch.setenv("LINKEDIN_IMPORT_MAP_WORKERS", "8")
    monkeypatch.setenv("LINKEDIN_IMPORT_SAVE_WORKERS", "3")
    urls = _urls(24)
    monkeypatch.setattr(li, "scrape_profiles", lambda batch: {
        "rows": [_row(u, u, "Acme") for u in batch], "failed": [], "requested": len(batch)})
    lock = threading.Lock()
    live = {"saves": 0, "max": 0}

    def slow_persist(tables, **kwargs):
        with lock:
            live["saves"] += 1
            live["max"] = max(live["max"], live["saves"])
            fake_db["persisted"].append({"tables": tables, **kwargs})
            cid = 100 + len(fake_db["persisted"])
        time.sleep(0.03)
        with lock:
            live["saves"] -= 1
        if tables["candidates"]["name"] == urls[5]:
            raise RuntimeError("database said no")
        return {"candidate_id": cid, "counts": {"roles": 1}}

    monkeypatch.setattr(li, "persist_mapped_profile", slow_persist)
    result = li.run_linkedin_import(upload_id=12, urls=urls, owner_user_id=28, role_id=7)

    assert 1 < live["max"] <= 3                                   # parallel, but never above the limit
    assert result["inserted"] == 23 and result["skipped"] == 1 and result["processed"] == 24
    assert result["role_assigned_count"] == 23
    assert [r["url"] for r in result["results"]] == urls          # sheet order, whatever finished first
    assert [r["status"] for r in result["results"]].count("error") == 1 and result["results"][5]["status"] == "error"
    assert result["errors"] == [f"{urls[5]}: database said no"]
    seen = [u["processed_count"] for u in fake_db["progress"] if "processed_count" in u]
    assert seen == sorted(seen) and seen[-1] == 24                # never backwards, ends at the row count
    assert fake_db["progress"][-1]["status"] == "completed_with_errors"


def test_report_callable_receives_progress_instead_of_the_upload_row(fake_db, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda batch: {
        "rows": [_row(u, u, "Acme") for u in batch], "failed": [], "requested": len(batch)})
    got: List[Dict[str, Any]] = []
    result = li.run_linkedin_import(upload_id=13, urls=_urls(3), owner_user_id=28, finalize=False,
                                    report=lambda **u: got.append(u))
    assert result["inserted"] == 3 and fake_db["progress"] == []
    assert got[0] == {"status": "scraping"} and got[-1]["processed_count"] == 3
    assert "completed_at" not in got[-1]            # left open for the caller's next phase

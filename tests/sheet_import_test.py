"""Google Sheet import orchestration and the LinkedIn-import extensions it relies on
(no network, no database)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from backend.api.routes import candidate_imports as ci
from backend.services import google_oauth, linkedin_import as li, sheet_import as si


ADA = "https://www.linkedin.com/in/ada-lovelace/"
ALAN = "https://linkedin.com/in/alan-turing"
GRACE = "https://www.linkedin.com/in/grace-hopper"

MAPPING = {"Profile": "linkedin", "Mail": "email", "Mobile": "phone", "Remarks": "notes",
           "Expected CTC": "custom", "First": "first_name"}


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"Profile": ADA, "Mail": "Ada@Example.com", "Mobile": "111", "Remarks": "met at expo",
         "Expected CTC": "30 LPA", "First": "Ada"},
        {"Profile": None, "Mail": "noli@example.com", "Mobile": None, "Remarks": None,
         "Expected CTC": None, "First": "Noli"},
        {"Profile": ALAN, "Mail": None, "Mobile": None, "Remarks": None, "Expected CTC": "40 LPA", "First": "Alan"},
        {"Profile": "https://www.linkedin.com/in/ada-lovelace", "Mail": None, "Mobile": None, "Remarks": None,
         "Expected CTC": None, "First": "Ada again"},
        {"Profile": "https://linkedin.com/company/acme", "Mail": "co@example.com", "Mobile": None,
         "Remarks": None, "Expected CTC": None, "First": "Company row"},
    ], dtype=object)


def _row(url: str, name: str) -> Dict[str, Any]:
    return {"linkedinUrl": url, "fullName": name, "experiences/0/companyName": "Acme", "email": "li@scraped.com"}


@pytest.fixture
def fake_li(monkeypatch):
    """In-memory stand-ins for every LinkedIn-import DB helper."""
    state: Dict[str, Any] = {"progress": [], "existing": {}, "persisted": [], "roles": [], "scrapes": []}
    monkeypatch.setattr(li, "write_progress", lambda upload_id, **u: state["progress"].append(u))
    monkeypatch.setattr(li, "existing_candidates", lambda key, owner: state["existing"].get(key, []))
    monkeypatch.setattr(li, "delete_candidates", lambda ids: len(ids))
    monkeypatch.setattr(li, "known_companies", lambda names: {})
    monkeypatch.setattr(li, "assign_role", lambda role_id, cid: state["roles"].append((role_id, cid)) or bool(role_id))
    monkeypatch.setattr(li, "refresh_caches", lambda ids: None)
    monkeypatch.setattr(li, "map_scraped_profile", lambda raw, **kw: {
        "candidates": {"name": raw["fullName"], "email": raw.get("email"), "phone": None,
                       "raw_fields": dict(raw)}, "coverage": {}})

    def fake_persist(tables, **kwargs):
        state["persisted"].append(tables["candidates"])
        return {"candidate_id": 100 + len(state["persisted"]), "counts": {}}

    monkeypatch.setattr(li, "persist_mapped_profile", fake_persist)
    return state


# ------------------------------------------------------------------ mapping + split
def test_mapping_needs_only_linkedin_when_enriching():
    assert si.validate_sheet_mapping({"Profile": "linkedin", "X": "nonsense"}, enrich_linkedin=True) == {
        "Profile": "linkedin", "X": "ignore"}
    with pytest.raises(si.SheetImportError):
        si.validate_sheet_mapping({"Mail": "email"}, enrich_linkedin=True)
    with pytest.raises(si.SheetImportError) as exc:
        si.validate_sheet_mapping({"Profile": "linkedin"}, enrich_linkedin=False)
    assert "first_name" in str(exc.value)


def test_linkedin_url_counts_finds_the_profile_column():
    assert si.linkedin_url_counts(_frame()) == {"Profile": 3}


def test_split_rows_routes_profiles_sheet_only_and_repeats():
    split = si.split_rows(_frame(), MAPPING)
    assert split["urls"] == [ADA, ALAN]
    assert split["sheet_only"] == [1, 4]          # no URL, and a company page (not a profile)
    assert split["repeated"] == [3]               # same profile as row 0, different spelling
    ada = split["extras"]["/in/ada-lovelace"]
    assert ada["email"] == "ada@example.com" and ada["phone"] == "111" and ada["notes"] == "met at expo"
    assert ada["raw"]["Expected CTC"] == "30 LPA"
    assert ada["raw"]["imported_extra_fields"]["expected_ctc"]["value"] == "30 LPA"


# ------------------------------------------------------- linkedin import extensions
def test_apply_row_extras_sheet_wins_for_contact_and_keeps_scraped_profile():
    tables = {"candidates": {"name": "Ada", "email": "li@scraped.com", "phone": None, "headline": "Engineer",
                             "raw_fields": {"fullName": "Ada", "experiences/0/title": "Engineer",
                                            "imported_extra_fields": {"old": {"value": "x"}}}}}
    li.apply_row_extras(tables, {"email": "sheet@example.com", "phone": None, "notes": "warm lead",
                                 "raw": {"Expected CTC": "30 LPA", "experiences/0/title": "Sheet title",
                                         "imported_extra_fields": {"expected_ctc": {"value": "30 LPA"}}}})
    cand = tables["candidates"]
    assert cand["email"] == "sheet@example.com"       # sheet wins
    assert cand["phone"] is None                      # blank sheet value does not erase
    assert cand["notes"] == "warm lead"
    assert cand["headline"] == "Engineer"             # profile field untouched
    assert cand["raw_fields"]["experiences/0/title"] == "Engineer"   # scraped key not replaced
    assert cand["raw_fields"]["Expected CTC"] == "30 LPA"
    assert set(cand["raw_fields"]["imported_extra_fields"]) == {"old", "expected_ctc"}


def test_scrapes_in_batches_and_keeps_earlier_batches_when_a_later_one_fails(fake_li, monkeypatch):
    monkeypatch.setenv("APIFY_MAX_PROFILES_PER_REQUEST", "2")
    calls: List[List[str]] = []

    def scrape(urls):
        calls.append(list(urls))
        if len(calls) == 2:
            raise li.ApifyScrapeError("quota exceeded")
        return {"rows": [_row(u, u) for u in urls], "failed": [], "requested": len(urls)}

    monkeypatch.setattr(li, "scrape_profiles", scrape)
    result = li.run_linkedin_import(upload_id=1, urls=[ADA, ALAN, GRACE], owner_user_id=28, defer_missing=True)
    assert calls == [[ADA, ALAN], [GRACE]]
    assert result["inserted"] == 2 and result["skipped"] == 1 and result["processed"] == 3
    assert result["results"][2]["status"] == "not_scraped"
    assert result["deferred"] == []          # an Apify outage must not turn into un-enriched rows
    assert any("quota exceeded" in e for e in result["errors"])


def test_duplicate_is_attached_to_the_role_and_can_be_quiet(fake_li, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: pytest.fail("duplicates must not be scraped"))
    fake_li["existing"]["/in/ada-lovelace"] = [{"id": 42, "pool_source": "recruiter_upload"}]
    result = li.run_linkedin_import(upload_id=1, urls=[ADA], owner_user_id=28, role_id=5, quiet_duplicates=True)
    assert result["skipped"] == 1 and result["role_assigned_count"] == 1
    assert fake_li["roles"] == [(5, 42)]
    assert result["errors"] == [] and result["status"] == "completed"


# -------------------------------------------------------------------- orchestration
def test_run_sheet_import_enriches_profiles_and_loads_the_rest_from_the_sheet(fake_li, monkeypatch):
    # Apify returns Ada but not Alan -> Alan falls back to the sheet values.
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: fake_li["scrapes"].append(list(urls)) or {
        "rows": [_row(ADA, "Ada Lovelace")], "failed": [ALAN], "requested": len(urls)})
    phase2: Dict[str, Any] = {}
    monkeypatch.setattr(ci, "_process_upload_rows", lambda **kw: phase2.update(kw))

    result = si.run_sheet_import(upload_id=9, df=_frame(), mapping=MAPPING, owner_user_id=28,
                                 user_role="recruiter", role_id=5, enrich_linkedin=True,
                                 enrichment_mode="verified_profile")

    assert fake_li["scrapes"] == [[ADA, ALAN]]
    saved = fake_li["persisted"][0]
    assert saved["name"] == "Ada Lovelace"
    assert saved["email"] == "ada@example.com" and saved["phone"] == "111" and saved["notes"] == "met at expo"
    assert saved["raw_fields"]["Expected CTC"] == "30 LPA"
    assert fake_li["roles"] == [(5, 101)]

    # phase 2 gets the no-URL row, the company-page row, and Alan (deferred), in sheet order
    assert list(phase2["df"]["First"]) == ["Noli", "Alan", "Company row"]
    assert phase2["enrichment_mode"] == "verified_profile" and phase2["role_id"] == 5
    assert phase2["base_counts"] == {"inserted": 1, "updated": 0, "skipped": 1, "role_assigned": 1, "processed": 2}
    assert any("loaded from the sheet values" in e for e in phase2["base_errors"])
    assert any("repeat a LinkedIn profile" in e for e in phase2["base_errors"])
    # phase 1 must leave the upload open for phase 2
    assert not any(u.get("status") in {"completed", "completed_with_errors"} for u in fake_li["progress"])
    assert result["linkedin_rows"] == 2 and result["sheet_rows"] == 3 and result["repeated_rows"] == 1


def test_run_sheet_import_finalizes_itself_when_every_row_was_enriched(fake_li, monkeypatch):
    monkeypatch.setattr(li, "scrape_profiles", lambda urls: {
        "rows": [_row(u, u) for u in urls], "failed": [], "requested": len(urls)})
    monkeypatch.setattr(ci, "_process_upload_rows", lambda **kw: pytest.fail("no sheet-only rows expected"))
    frame = _frame().iloc[[0, 2]]
    si.run_sheet_import(upload_id=9, df=frame, mapping=MAPPING, owner_user_id=28, user_role="recruiter",
                        role_id=None)
    final = fake_li["progress"][-1]
    assert final["status"] == "completed" and final["processed_count"] == 2 and "completed_at" in final


def test_run_sheet_import_stops_when_apify_is_down(fake_li, monkeypatch):
    def boom(urls):
        raise li.ApifyScrapeError("APIFY_API_TOKEN is not set")

    monkeypatch.setattr(li, "scrape_profiles", boom)
    monkeypatch.setattr(ci, "_process_upload_rows", lambda **kw: pytest.fail("must not load un-enriched rows"))
    result = si.run_sheet_import(upload_id=9, df=_frame(), mapping=MAPPING, owner_user_id=28,
                                 user_role="recruiter", role_id=None)
    assert result["status"] == "failed"
    assert fake_li["progress"][-1]["status"] == "failed"


def test_without_linkedin_enrichment_it_is_a_plain_spreadsheet_import(monkeypatch):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(ci, "_process_upload_rows", lambda **kw: seen.update(kw))
    monkeypatch.setattr(li, "run_linkedin_import", lambda **kw: pytest.fail("Apify must not be used"))
    si.run_sheet_import(upload_id=9, df=_frame(), mapping=MAPPING, owner_user_id=28, user_role="recruiter",
                        role_id=None, enrich_linkedin=False)
    assert len(seen["df"].index) == 5 and "base_counts" not in seen


# ------------------------------------------------------------------- token storage
def test_refresh_token_is_stored_encrypted(monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("GOOGLE_TOKEN_ENC_KEY", Fernet.generate_key().decode())
    blob = google_oauth.encrypt_token("1//refresh-token")
    assert "refresh-token" not in blob
    assert google_oauth.decrypt_token(blob) == "1//refresh-token"

    monkeypatch.setenv("GOOGLE_TOKEN_ENC_KEY", Fernet.generate_key().decode())   # key rotated/lost
    with pytest.raises(google_oauth.GoogleAuthError) as exc:
        google_oauth.decrypt_token(blob)
    assert exc.value.code == "reconnect"


def test_connect_requires_the_sheets_scope_and_server_setup(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    with pytest.raises(google_oauth.GoogleAuthError) as exc:
        google_oauth.connect(1, "code")
    assert exc.value.code == "not_configured"

    from cryptography.fernet import Fernet

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_TOKEN_ENC_KEY", Fernet.generate_key().decode())

    class Resp:
        status_code = 200
        headers = {"content-type": "application/json; charset=utf-8"}

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    sent: Dict[str, Any] = {}
    monkeypatch.setattr(google_oauth, "_post_token",
                        lambda data: sent.update(data) or Resp({"scope": "openid email", "refresh_token": "r"}))
    with pytest.raises(google_oauth.GoogleAuthError) as exc:
        google_oauth.connect(1, "code")
    assert exc.value.code == "reconnect"
    assert sent["redirect_uri"] == "postmessage" and sent["grant_type"] == "authorization_code"

    saved: Dict[str, Any] = {}
    monkeypatch.setattr(google_oauth, "save_tokens",
                        lambda uid, tok, scope, email: saved.update(uid=uid, tok=tok, email=email))
    monkeypatch.setattr(google_oauth, "_post_token", lambda data: Resp(
        {"scope": f"openid email {google_oauth.SHEETS_SCOPE}", "refresh_token": "r-123"}))
    assert google_oauth.connect(7, "code") == {"connected": True, "google_email": None}
    assert saved == {"uid": 7, "tok": "r-123", "email": None}

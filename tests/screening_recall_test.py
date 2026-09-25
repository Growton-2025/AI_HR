"""The strict screener must see the employer and tenure a recruiter's sheet
records, wherever the sheet put them. On the "Clear – AE (ME & SEA)" role
(680 candidates, 2026-09-25) a "Testsigma competitors" screen matched 1
person when 69 qualified: 5 had roles rows, 594 carried the employer in
raw_fields.import_company, and competitor names were validated against the
companies table alone."""
from backend.pipeline import query as q
from backend.services import profile_experience as pe


# ── profile_experience (pure) ──────────────────────────────────────────────

def test_import_company_becomes_the_current_role():
    profile = {"headline": "Senior Account Executive", "raw_fields": {"import_company": "BrowserStack"}}
    roles = pe.synthesize_roles(profile, [])
    assert roles[0]["company"] == "BrowserStack" and roles[0]["title"] == "Senior Account Executive"
    assert roles[0]["_source"] == "raw_fields.import_company"


def test_wide_columns_pair_company_n_with_the_suffixed_title_and_dates():
    raw = {
        "Company 1 Name": "Databricks", "Title": "Named AE", "Start date": "2024-05-01 00:00:00", "End Date": "",
        "Company 2 Name": "Oracle", "Title.1": "Field Sales Rep", "Start date.1": "2018-02-01 00:00:00", "End Date.1": "2019-04-01 00:00:00",
        "Company 3 Name": "", "Title.2": "",
    }
    roles = pe.wide_column_roles(raw)
    assert [(r["company"], r["title"], r["start_date"], r["end_date"]) for r in roles] == [
        ("Databricks", "Named AE", "2024-05-01", ""), ("Oracle", "Field Sales Rep", "2018-02-01", "2019-04-01"),
    ]


def test_headline_employer_parsing():
    assert pe.headline_employer("Account Executive- Mid Market @Salesforce") == ("Account Executive- Mid Market", "Salesforce")
    assert pe.headline_employer("Senior AE at LambdaTest") == ("Senior AE", "LambdaTest")
    assert pe.headline_employer("Account Executive - EMEA") is None          # a region, not an employer
    assert pe.headline_employer("Helping customers take their Data and AI to production") is None


def test_synthesis_never_duplicates_an_employer_already_in_roles():
    profile = {"headline": "AE at BrowserStack", "raw_fields": {"import_company": "BrowserStack"}}
    assert pe.synthesize_roles(profile, [{"company": "browserstack technologies"}]) == []


def test_uploaded_tenure_fields():
    assert pe.raw_total_experience_years({"Overall Exp (yrs)": "10.0"}) == 10.0
    assert pe.raw_total_experience_years({"Total work experience ": "9"}) == 9.0
    assert pe.raw_total_experience_years({"Work Ex": "8+ Years ( AE - 5 Yrs, B2B SaaS - 8 Years)"}) == 8.0
    assert pe.raw_total_experience_years({"Notice Period": "30 Days"}) is None
    fn = pe.raw_function_years({"AE Exp (yrs)": "2.3", "Work Ex": "8+ Years ( AE - 5.5 Yrs, B2B SaaS - 8 Years)"})
    assert fn["account executive"] == 5.5 and fn["b2b saas"] == 8.0
    assert pe.function_years_for(fn, "Account Executive", aliases=["AE"]) == 5.5
    assert pe.function_years_for(fn, "BDR") is None


def test_employer_universe_covers_every_source():
    names = pe.employer_names_from_profiles([
        {"roles": [{"company": "LambdaTest"}]},
        {"raw_fields": {"import_company": "Katalon"}},
        {"raw_fields": {"Company 2 Name": "Sauce Labs", "experiences/0/companyName": "testRigor"}},
        {"headline": "AE at Tricentis"},
    ])
    assert names == {"LambdaTest", "Katalon", "Sauce Labs", "testRigor", "Tricentis"}


# ── wired into the pipeline ────────────────────────────────────────────────

def test_competitor_validation_accepts_employers_seen_only_in_uploads(monkeypatch):
    monkeypatch.setattr(q, "ALL_COMPANY_NAMES", ["BrowserStack"])
    monkeypatch.setattr(q, "PROFILES_BY_ID", {
        1: {"raw_fields": {"import_company": "LambdaTest"}},
        2: {"raw_fields": {"import_company": "Katalon"}},
        3: {"headline": "AE at Labs"},
    })
    validated = q._validate_company_names_against_db(
        ["BrowserStack", "LambdaTest", "Katalon", "Sauce Labs", "Tricentis"], exclude="Testsigma")
    assert validated == ["BrowserStack", "LambdaTest", "Katalon"]     # Sauce Labs must not match "Labs"


def test_strict_scorer_matches_a_competitor_recorded_only_as_import_company():
    profile = {"id": 13319, "name": "Aseem", "headline": "Account Executive - EMEA",
               "raw_fields": {"import_company": "LambdaTest", "Overall Exp (yrs)": "7"},
               "roles": [], "total_experience_years": 0}
    criteria = {"required_companies": {"operator": "OR", "employment_scope": "any_employer",
                "values": [{"company": "LambdaTest", "employment_scope": "any_employer", "source": "competitor_of:Testsigma"},
                           {"company": "BrowserStack", "employment_scope": "any_employer"}]}}
    reasons = []
    result = q._strict_shortlist_score_candidate(profile, criteria, reasons)
    assert result is not None, reasons
    assert result["matched_criteria"] and result["total_experience_years"] == 7.0
    assert any(e["source"] == "role company" for e in result["evidence_log"])


def test_current_employer_scope_uses_the_uploaded_employer():
    profile = {"id": 1, "headline": "AE", "raw_fields": {"import_company": "BrowserStack"}, "roles": []}
    criteria = {"required_companies": {"operator": "OR", "employment_scope": "current_employer",
                "values": [{"company": "BrowserStack", "employment_scope": "current_employer"}]}}
    assert q._strict_shortlist_score_candidate(profile, criteria, []) is not None
    # …and a mere mention elsewhere is NOT evidence of the current employer.
    profile = {"id": 2, "headline": "AE", "raw_fields": {"import_company": "Freshworks", "Reasoning": "previously at BrowserStack"}, "roles": []}
    assert q._strict_shortlist_score_candidate(profile, criteria, []) is None


def test_any_employer_scope_finds_a_competitor_mentioned_in_profile_text():
    profile = {"id": 3, "headline": "AE", "raw_fields": {"import_company": "Freshworks", "Summary": "5 years at testRigor selling test automation"}, "roles": []}
    criteria = {"required_companies": {"operator": "OR", "employment_scope": "any_employer",
                "values": [{"company": "testRigor", "employment_scope": "any_employer"}]}}
    result = q._strict_shortlist_score_candidate(profile, criteria, [])
    assert result is not None and result["evidence_log"][0]["source"] == "profile text"


def test_uploaded_function_years_satisfy_min_function_years():
    profile = {"id": 4, "headline": "Account Executive", "raw_fields": {"import_company": "Rippling", "AE Exp (yrs)": "4.0", "Overall Exp (yrs)": "8"}, "roles": []}
    criteria = {"min_function_years": [{"function": "Account Executive", "aliases": ["AE"], "min_years": 3}], "min_total_experience": 5}
    reasons = []
    result = q._strict_shortlist_score_candidate(profile, criteria, reasons)
    assert result is not None, reasons
    assert any(e.get("source") == "uploaded field" for e in result["evidence_log"])
    assert q._strict_shortlist_score_candidate(dict(profile, raw_fields={"import_company": "Rippling", "AE Exp (yrs)": "2"}), criteria, []) is None

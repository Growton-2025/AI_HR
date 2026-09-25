"""Why "experience at Testsigma competitors" returned 1 of ~60 (role 92,
2026-09-25), and the guards that keep it fixed:

* the plan model scoped "experience at" to the *current* employer;
* one matched value of an OR list scored 1/6 ("16.7% evidence fit");
* the audit model rejected 46 of 51 deterministic matches for a "missing
  tenure" the query never asked for;
* the web competitor list carried product names and no rebrands, so
  LambdaTest AEs never matched "TestMu AI".
"""
import asyncio
from datetime import datetime

import pytest

from backend.pipeline import query as q


# ── employment scope comes from the query, not the plan model ──────────────

def test_experience_at_is_career_history_not_current_employer():
    assert q._query_company_scope("Candidates with experience at Testsigma competitors") == "any_employer"
    assert q._query_company_scope("ex-Freshworks account executives") == "any_employer"
    assert q._query_company_scope("Candidates currently working at a BrowserStack competitor") == "current_employer"
    assert q._query_company_scope("People whose present company is Datadog") == "current_employer"


def test_plan_models_current_employer_is_overridden_unless_the_query_says_so():
    query = "Candidates with experience at Testsigma competitors"
    assert q._reconcile_company_scope("current_employer", query) == "any_employer"
    assert q._reconcile_company_scope("current_employer", "people currently at Datadog") == "current_employer"
    normalized = q._normalize_companies_with_scope(
        {"operator": "OR", "employment_scope": "current_employer",
         "values": [{"company": "BrowserStack", "employment_scope": "current_employer"}]}, query)
    assert normalized["employment_scope"] == "any_employer"
    assert normalized["values"][0]["employment_scope"] == "any_employer"


# ── OR criterion score ─────────────────────────────────────────────────────

def test_one_matched_value_of_an_or_list_is_a_full_match():
    profile = {"id": 1, "headline": "AE", "raw_fields": {"import_company": "BrowserStack"}, "roles": []}
    criterion = {"operator": "OR", "employment_scope": "any_employer",
                 "values": [{"company": c, "employment_scope": "any_employer"} for c in ("Katalon", "BrowserStack", "SmartBear", "UiPath", "OpenText", "TestMu AI")]}
    result = q._strict_presence_result(profile, "required_companies", criterion)
    assert result["met"] and result["score"] == 1.0
    and_criterion = dict(criterion, operator="AND")
    assert q._strict_presence_result(profile, "required_companies", and_criterion)["score"] == pytest.approx(1 / 6)


# ── audit rejections must name a real criterion ────────────────────────────

def test_rejection_for_a_requirement_the_query_never_had_is_ungrounded():
    criteria = {"required_companies": {"operator": "OR", "values": []}, "_screening_query": "x"}
    for missing in (["Duration of current role"], ["current_employer with valid tenure"], [], ["employment_scope"]):
        review = {"final_status": "not_verified", "missing_criteria": missing}
        assert q._audit_rejection_is_grounded(review, criteria, {}) is False, missing


def test_rejection_that_names_a_criterion_but_argues_about_tenure_is_ungrounded():
    criteria = {"required_segments": {}, "required_functions": {}}
    review = {"final_status": "not_verified", "missing_criteria": ["required_segments", "required_functions"],
              "reasoning": "While the headline indicates enterprise segment (ev1), there is no duration or specific roles confirming it."}
    assert q._audit_rejection_is_grounded(review, criteria, {}) is False
    # …but with a tenure criterion in the query the same wording is a real rejection
    assert q._audit_rejection_is_grounded(review, {**criteria, "min_function_years": [{"function": "AE", "min_years": 3}]}, {}) is True


def test_rejection_naming_a_real_criterion_stands():
    criteria = {"required_companies": {}, "required_geographies": {}, "min_tenure_in_latest_role": 2}
    assert q._audit_rejection_is_grounded({"missing_criteria": ["EMEA geography"]}, criteria, {})
    assert q._audit_rejection_is_grounded({"missing_criteria": ["tenure in latest role"]}, criteria, {})
    assert q._audit_rejection_is_grounded({"missing_criteria": ["Companies: no competitor employer"]}, criteria, {})


# ── competitor list: aliases, rebrands from the data, judge pass ───────────

@pytest.fixture
def employers(monkeypatch):
    monkeypatch.setattr(q, "ALL_COMPANY_NAMES", [])
    monkeypatch.setattr(q, "PROFILES_BY_ID", {
        1: {"raw_fields": {"import_company": "LambdaTest"}},
        2: {"raw_fields": {"import_company": "LambdaTest is now TestMu AI"}},
        3: {"raw_fields": {"import_company": "Testmu Ai"}},
        4: {"raw_fields": {"import_company": "Micro Focus (formerly Hewlett Packard)"}},
        5: {"raw_fields": {"import_company": "Salesforce"}},
        6: {"raw_fields": {"import_company": "testRigor"}},
        7: {"raw_fields": {"import_company": "Scrut Automation"}},
    })
    q._rebrand_alias_cache.clear()


def test_rebrands_are_read_from_the_employer_strings_candidates_typed(employers):
    graph = q._rebrand_aliases_from_employers()
    assert graph["lambdatest"] == {"TestMu AI"} and graph["testmu ai"] == {"LambdaTest"}
    assert graph["micro focus"] == {"Hewlett Packard"} and graph["hewlett packard"] == {"Micro Focus"}


def test_competitor_named_by_its_new_brand_matches_the_old_one_and_drops_product_aliases(employers):
    entries = q._validate_competitor_entries(
        [{"name": "TestMu AI", "aliases": ["TestMu AI Web Automation"]},
         {"name": "SmartBear", "aliases": ["SmartBear ReadyAPI"]},          # nobody here works there
         {"name": "Salesforce", "aliases": ["Test Automation"]}],
        exclude="Testsigma")
    by_name = {e["company"]: e["aliases"] for e in entries}
    assert set(by_name) == {"Testmu Ai", "Salesforce"}
    assert "LambdaTest" in by_name["Testmu Ai"]
    assert "TestMu AI Web Automation" not in by_name["Testmu Ai"]
    assert by_name["Salesforce"] == []                                        # generic alias dropped


def test_judge_adds_confident_role_employers_and_never_drops_a_web_entry(employers, monkeypatch):
    prompts = []

    def fake_call(system_prompt, user_prompt, **kwargs):
        prompts.append(user_prompt)
        return {"competitors": [                                          # scope-wide judge (additive)
            {"name": "Testmu Ai", "confidence": "high"},
            {"name": "testRigor", "confidence": "high"},                  # role employer, confidently added
            {"name": "Scrut Automation", "confidence": "low"},            # role employer, not confident enough
        ]}
    monkeypatch.setattr(q, "call_openai_json", fake_call)
    # BrowserStack is absent from the model's answer — web + data still keep it.
    web = [{"company": "Testmu Ai", "aliases": ["LambdaTest"]}, {"company": "BrowserStack", "aliases": []}]
    kept = asyncio.run(q._judge_competitors("Testsigma", "test automation platform", web, ["testRigor", "Scrut Automation", "BrowserStack"], q.TokenCostTracker()))
    assert [e["company"] for e in kept] == ["Testmu Ai", "BrowserStack", "testRigor"]
    assert kept[0]["aliases"] == ["LambdaTest"] and kept[2].get("source") == "scope"
    assert len(prompts) == 1 and "- Scrut Automation" in prompts[0] and "- testRigor" in prompts[0]


def test_a_company_is_never_its_own_competitor_even_with_a_suffix(employers, monkeypatch):
    monkeypatch.setattr(q, "PROFILES_BY_ID", {1: {"raw_fields": {"import_company": "Hevo Data"}}, 2: {"raw_fields": {"import_company": "Fivetran"}}})
    entries = q._validate_competitor_entries([{"name": "Hevo Data", "aliases": []}, {"name": "Fivetran", "aliases": []}], exclude="Hevo")
    assert [e["company"] for e in entries] == ["Fivetran"]
    monkeypatch.setattr(q, "call_openai_json", lambda *a, **k: {"competitors": [{"name": "Hevo Data", "confidence": "high"}, {"name": "Fivetran", "confidence": "high"}]})
    kept = asyncio.run(q._judge_competitors("Hevo", None, entries, ["Hevo Data"], q.TokenCostTracker()))
    assert [e["company"] for e in kept] == ["Fivetran"]


def test_judge_failure_leaves_the_web_list_untouched(employers, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("openai down")
    monkeypatch.setattr(q, "call_openai_json", boom)
    web = [{"company": "Testmu Ai", "aliases": ["LambdaTest"]}]
    assert asyncio.run(q._judge_competitors("Testsigma", None, web, ["testRigor"], q.TokenCostTracker())) == web


def test_cache_refresh_unions_competitor_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "SHORTLIST_COMPANY_FACT_CACHE_PATH", tmp_path / "facts.json")
    first = {"competitors": [{"target": "Testsigma", "companies": [{"name": "Katalon", "aliases": []}, "BrowserStack"]}]}
    q._cache_company_facts_from_structured({}, first)
    second = {"competitors": [{"target": "Testsigma", "companies": [{"name": "Sauce Labs", "aliases": []}]}]}
    q._cache_company_facts_from_structured({}, second)
    cached = q._load_shortlist_company_fact_cache()["testsigma"]
    assert [e["name"] for e in cached["competitors"]] == ["Sauce Labs", "Katalon", "BrowserStack"]
    assert cached["prompt_version"] == q.SHORTLIST_COMPANY_FACT_PROMPT_VERSION


# ── query hints the plan model gets wrong ──────────────────────────────────

def test_duration_minimums_need_a_number_in_the_query():
    criteria = {"required_segments": {"operator": "OR", "values": ["mid-market"]},
                "min_function_years": [{"function": "Sales Development", "min_years": 1}]}
    q._prune_unstated_duration_requirements(criteria, "Candidates with mid-market sales experience")
    assert "min_function_years" not in criteria and "required_segments" in criteria
    kept = {"min_total_experience": 5}
    q._prune_unstated_duration_requirements(kept, "Account executives with 5+ years of overall experience")
    assert kept == {"min_total_experience": 5}


def test_senior_means_the_title_says_senior_not_five_years_in_four_functions():
    criteria = {"required_functions": {"operator": "OR", "values": ["Account Executive", "Business Development"]},
                "min_function_years": [{"function": "Account Executive", "min_years": 5}]}
    q._apply_seniority_title_hint(criteria, "Senior account executives")
    assert "min_function_years" not in criteria
    assert "Senior Account Executive" in criteria["required_functions"]["values"]
    assert "Sr. AE" in criteria["required_functions"]["values"]
    explicit = {"required_functions": {"operator": "OR", "values": ["Account Executive"]}, "min_function_years": [{"function": "Account Executive", "min_years": 8}]}
    q._apply_seniority_title_hint(explicit, "Senior AEs with 8+ years")
    assert explicit["min_function_years"]                                   # a number in the query wins


def test_notice_period_parsing_and_hint():
    assert q._notice_period_days("30 Days") == 30 and q._notice_period_days("1 month") == 30
    assert q._notice_period_days("2 months (negotiable)") == 60 and q._notice_period_days("Immediate joiner") == 0
    assert q._notice_period_days("BLR") is None and q._notice_period_days("None") is None
    criteria = {"required_keywords": {"operator": "OR", "values": ["immediate joiners", "notice period of 30 days or less"]}}
    q._apply_notice_period_hint(criteria, "Immediate joiners or notice period of 30 days or less")
    assert criteria == {"max_notice_period_days": 30.0}


def test_notice_period_is_scored_from_the_uploaded_column():
    criteria = {"max_notice_period_days": 30}
    ok = {"id": 1, "headline": "AE", "raw_fields": {"import_company": "X", "Notice Period": "15 days"}, "roles": []}
    result = q._strict_shortlist_score_candidate(ok, criteria, [])
    assert result is not None and result["evidence_log"][0]["source"] == "uploaded field"
    for value in ("60 days", "2 months", "BLR", ""):
        reasons = []
        assert q._strict_shortlist_score_candidate(dict(ok, raw_fields={"Notice Period": value}), criteria, reasons) is None
        assert reasons == ["max_notice_period_days"]


def test_based_in_city_is_a_location_not_a_market_geography():
    criteria = {"required_geographies": {"operator": "OR", "values": ["US", "Bangalore"]}}
    q._split_location_from_geography(criteria, "Enterprise account executives based in Bangalore with US market experience")
    assert criteria["required_locations"]["values"] == ["Bangalore"]
    assert criteria["required_geographies"]["values"] == ["US"]


def test_segment_evidence_comes_from_the_candidates_own_selling_record():
    criteria = {"required_segments": {"operator": "OR", "values": ["enterprise"], "employment_scope": "any_employer"}}
    titled = {"id": 1, "headline": "Enterprise Account Executive", "raw_fields": {"import_company": "Rippling"}, "roles": []}
    result = q._strict_shortlist_score_candidate(titled, criteria, [])
    assert result is not None and result["evidence_log"][0]["source"] == "candidate selling record"
    # a recruiter's note saying "not enterprise" must not count as enterprise evidence
    noted = {"id": 2, "headline": "Account Executive", "raw_fields": {"import_company": "Rippling", "Reasoning": "not enterprise, SMB only"}, "roles": []}
    assert q._strict_shortlist_score_candidate(noted, criteria, []) is None


def test_generic_competitor_target_is_repaired_from_the_query():
    assert q._competitor_target_from_query("Candidates currently working at a BrowserStack competitor") == "BrowserStack"
    assert q._competitor_target_from_query("People with experience at Hevo competitors") == "Hevo"
    assert q._competitor_target_from_query("Candidates with experience at in testsigma competators") == "testsigma"
    assert q._competitor_target_from_query("Account executives at Testsigma competitors with EMEA experience") == "Testsigma"
    assert q._competitor_target_from_query("find competitors of Sauce Labs") == "Sauce Labs"
    criteria = {"competitors_of": [{"target": "company", "employment_scope": "current_employer"}]}
    q._repair_competitor_target(criteria, "Candidates currently working at a BrowserStack competitor")
    assert criteria["competitors_of"][0]["target"] == "BrowserStack"
    fine = {"competitors_of": [{"target": "browserstack"}]}
    q._repair_competitor_target(fine, "Candidates currently working at a BrowserStack competitor")
    assert fine["competitors_of"][0]["target"] == "browserstack"


def test_cache_ttl_reads_the_iso_timestamp_the_cache_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "SHORTLIST_COMPANY_FACT_CACHE_PATH", tmp_path / "facts.json")
    q._cache_company_facts_from_structured({}, {"competitors": [{"target": "Testsigma", "companies": ["Katalon"]}]})
    stamp = q._load_shortlist_company_fact_cache()["testsigma"]["last_verified_at"]
    parsed = q._parse_cache_timestamp(stamp)
    assert parsed is not None and abs((datetime.utcnow() - parsed).total_seconds()) < 120
    facts = q._cached_company_facts_for_criteria({"competitor_of": [{"target": "Testsigma"}]})
    assert [c["name"] for c in facts["competitors"][0]["companies"]] == ["Katalon"]    # fresh: served from cache


def test_us_market_reaches_north_america_and_amer_territories_but_uae_never_reaches_emea():
    allowed = {"operator": "OR", "values": ["US"], "allow_region_reverse_match": True}
    terms = q._geography_match_terms("US", allowed)
    assert {"north america", "amer", "namer", "americas"} <= set(terms)
    assert "americas" not in q._geography_match_terms("US", {"operator": "OR", "values": ["US"]})   # policy off
    assert "emea" not in q._geography_match_terms("UAE", {"operator": "OR", "values": ["UAE"], "allow_region_reverse_match": True})
    profile = {"id": 1, "headline": "Account Executive", "raw_fields": {"import_company": "X", "Focused Geo": "North America (1 yr), PAN India"}, "roles": []}
    assert q._strict_shortlist_score_candidate(profile, {"required_geographies": allowed}, []) is not None


def test_hyphenated_terms_match_spaced_and_joined_spellings():
    for text in ("account executive- mid market @salesforce", "midmarket ae", "mid-market sales"):
        assert q._term_matches_text("mid-market", text), text
    assert q._term_matches_text("mid market", "mid-market sales")
    assert not q._term_matches_text("mid-market", "enterprise sales")


def test_per_company_research_entries_are_folded_under_the_requested_target():
    raw = [
        {"target": "Airbyte", "companies": ["Airbyte, Inc."], "sources": ["s1"]},
        {"target": "Fivetran", "companies": ["Fivetran, Inc."], "sources": ["s2"]},
        {"target": "Hevo Data", "companies": [{"name": "Matillion", "aliases": []}], "sources": ["s3"]},
    ]
    grouped = q._regroup_competitor_entries(raw, ["Hevo"])
    assert len(grouped) == 1 and grouped[0]["target"] == "Hevo"
    assert [c["name"] for c in grouped[0]["companies"]] == ["Matillion", "Airbyte", "Fivetran"]
    assert q._regroup_competitor_entries(raw, ["Hevo", "MongoDB"])[0]["companies"][0]["name"] == "Matillion"   # ambiguous strays are not guessed


def test_verified_verdict_with_a_slipped_citation_is_repaired_not_dropped(monkeypatch):
    profile = {"id": 1, "evidence_log": [{"id": "ev1", "criterion": "Customer segments", "snippet": "Enterprise Account Executive"}]}
    monkeypatch.setattr(q, "call_openai_json", lambda *a, **k: {"final_status": "verified_match", "answer": "Enterprise AE.", "reasoning": "Title shows enterprise (ev2).", "evidence_ids": ["ev2"]})
    review = asyncio.run(q.generate_reasoning_for_profile(profile, {"required_segments": {}}, q.TokenCostTracker()))
    assert review["final_status"] == "verified_match" and review["evidence_ids"] == ["ev1"] and review["auditor_status"] == "citations_repaired"
    monkeypatch.setattr(q, "call_openai_json", lambda *a, **k: {"final_status": "not_verified", "reasoning": "no", "evidence_ids": ["ev9"]})
    review = asyncio.run(q.generate_reasoning_for_profile(profile, {"required_segments": {}}, q.TokenCostTracker()))
    assert review.get("audit_unavailable") is True                    # a rejection with bad citations stays unverified


def test_function_requirements_must_be_named_in_the_query():
    criteria = {"required_segments": {"operator": "OR", "values": ["mid-market"]},
                "required_functions": {"operator": "OR", "values": ["Sales Development", "Business Development", "SDR", "BDR"]},
                "min_function_years": [{"function": "Sales Development", "min_years": 1}]}
    q._prune_functions_not_in_query(criteria, "Candidates with mid-market sales experience")
    assert "required_functions" not in criteria and "min_function_years" not in criteria
    named = {"required_functions": {"operator": "OR", "values": ["Account Executive", "Sales Development"]}}
    q._prune_functions_not_in_query(named, "Account executives with enterprise segment experience")
    assert named["required_functions"]["values"] == ["Account Executive"]
    abbreviated = {"required_functions": {"operator": "OR", "values": ["Account Executive"]}}
    q._prune_functions_not_in_query(abbreviated, "Senior AEs in Bangalore")
    assert abbreviated["required_functions"]["values"] == ["Account Executive"]


def test_sub_regions_satisfy_their_super_region():
    terms = q._geography_match_terms("EMEA", {"operator": "OR", "values": ["EMEA"]})
    assert {"europe", "middle east"} <= set(terms)
    profile = {"id": 1, "headline": "Account Executive - Europe", "raw_fields": {"import_company": "X"}, "roles": []}
    assert q._strict_shortlist_score_candidate(profile, {"required_geographies": {"operator": "OR", "values": ["EMEA"]}}, []) is not None


def test_based_in_city_survives_the_market_geography_merge():
    plan = {"filter_plan": {"hard_filters": {
        "required_locations": {"operator": "AND", "values": ["Bangalore"]},
        "required_segments": {"operator": "AND", "values": ["enterprise"]},
        "required_functions": {"operator": "AND", "values": ["Account Executive"]},
        "required_geographies": {"operator": "AND", "values": ["US"]}},
        "geography_policy": {"use_current_location": False, "expand_regions": True, "allow_country_region_reverse_match": True}}}
    criteria = q._coerce_filter_plan_to_criteria(plan, "Enterprise account executives based in Bangalore with US market experience")
    assert q.get_values_from_criteria(criteria["required_locations"]) == ["Bangalore"]
    assert q.get_values_from_criteria(criteria["required_geographies"]) == ["US"]
    assert criteria["required_geographies"].get("allow_region_reverse_match") is True


def test_competition_is_symmetric_across_cached_research(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "SHORTLIST_COMPANY_FACT_CACHE_PATH", tmp_path / "facts.json")
    q._cache_company_facts_from_structured({}, {"competitors": [
        {"target": "Testsigma", "companies": [{"name": "BrowserStack", "aliases": []}, {"name": "Katalon", "aliases": []}]},
        {"target": "Hevo", "companies": [{"name": "Fivetran", "aliases": []}]},
    ]})
    assert [e["name"] for e in q._reverse_competitors_from_cache("browserstack")] == ["Testsigma"]
    assert q._reverse_competitors_from_cache("Testsigma") == []


def test_generic_employer_strings_cannot_validate_a_web_name(monkeypatch):
    monkeypatch.setattr(q, "ALL_COMPANY_NAMES", [])
    monkeypatch.setattr(q, "PROFILES_BY_ID", {1: {"raw_fields": {"import_company": "Healthcare"}}, 2: {"raw_fields": {"import_company": "Tricentis"}}})
    assert q._validate_company_names_against_db(["Cross Country Healthcare", "Tricentis Inc."]) == ["Tricentis"]


def test_target_hints_come_from_candidate_data(monkeypatch):
    monkeypatch.setattr(q, "PROFILES_BY_ID", {
        1: {"headline": "Account Executive at Hevo - data pipelines", "raw_fields": {"import_company": "Hevo Data"}},
        2: {"headline": "AE", "raw_fields": {"import_company": "Fivetran"}},
    })
    hints = q._target_disambiguation_hints(["Hevo"])
    assert "Hevo Data" in hints and "data pipelines" in hints and "Fivetran" not in hints


def test_role_experience_rejection_is_not_grounded_by_a_total_experience_criterion():
    review = {"final_status": "not_verified", "missing_criteria": ["Role experience as an account executive"],
              "reasoning": "Total experience of 12 years is confirmed (ev1) but there is no evidence of the account executive role."}
    assert q._audit_rejection_is_grounded(review, {"min_total_experience": 5}, {}) is False
    assert q._audit_rejection_is_grounded({"missing_criteria": ["total experience below 5 years"]}, {"min_total_experience": 5}, {}) is True


def test_auditor_cannot_overrule_structured_evidence_but_can_reject_loose_text():
    criteria = {"required_segments": {}, "required_functions": {}}
    review = {"final_status": "not_verified", "missing_criteria": ["required_segments", "required_functions"],
              "reasoning": "The headline indicates enterprise (ev1) but this does not sufficiently demonstrate the requirement."}
    structured = {"evidence_log": [{"id": "ev1", "criterion": "Customer segments", "source": "headline"},
                                   {"id": "ev2", "criterion": "Functions", "source": "role 1 title"}]}
    assert q._audit_rejection_is_grounded(review, criteria, structured) is False
    loose = {"evidence_log": [{"id": "ev1", "criterion": "Customer segments", "source": "about"},
                              {"id": "ev2", "criterion": "Functions", "source": "role 1 title"}]}
    assert q._audit_rejection_is_grounded(review, criteria, loose) is True


def test_competitor_list_is_found_under_whatever_key_the_model_used():
    assert q._pick_competitor_list({"key_competitors": [{"name": "Airbyte"}]}) == [{"name": "Airbyte"}]
    assert q._pick_competitor_list({"sources": [], "results": [{"name": "Fivetran", "aliases": []}]}) == [{"name": "Fivetran", "aliases": []}]
    assert q._pick_competitor_list({"sources": ["x"]}) == []


def test_a_product_name_does_not_validate_as_its_parent_company(monkeypatch):
    monkeypatch.setattr(q, "ALL_COMPANY_NAMES", [])
    monkeypatch.setattr(q, "PROFILES_BY_ID", {1: {"raw_fields": {"import_company": "Amazon"}}, 2: {"raw_fields": {"import_company": "SmartBear Software"}},
                                              3: {"raw_fields": {"import_company": "Couchbase"}}})
    assert q._validate_company_names_against_db(["Amazon DynamoDB", "SmartBear", "Couchbase Inc.", "Google Firestore"]) == ["SmartBear Software", "Couchbase"]

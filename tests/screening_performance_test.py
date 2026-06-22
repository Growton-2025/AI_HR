import asyncio
import re

from backend.api import schemas
from backend.api.routes import browse
from backend.api.routes import roles
from backend.pipeline import query


def _user(user_id=7, role="recruiter"):
    return schemas.User(
        id=user_id,
        username=f"user{user_id}@example.com",
        email=f"user{user_id}@example.com",
        full_name="User",
        role=role,
        permissions={},
    )


def test_screening_scores_uploaded_fields_about_roles_and_company_metadata():
    profile = {
        "id": 101,
        "name": "Sanjeev Naik",
        "headline": "Inside Sales Manager - Enterprise AI",
        "about": "BDR leader creating outbound pipeline for US and UK SaaS accounts.",
        "location": "Bengaluru, India",
        "total_experience_years": 7.1,
        "raw_fields": {
            "Focused Geography": "US, UK, EMEA",
            "Outbound Exp": "70% outbound prospecting",
            "Shift timings": "12:00 PM to 2:00 AM",
            "Recruiter Summary": "Sold SaaS to enterprise customers.",
        },
        "roles": [
            {
                "company": "Highradius",
                "title": "Financial Technology Advisor",
                "details": "Outbound prospecting, SQL creation, enterprise account development.",
                "duration_years": 1.8,
                "company_details": {
                    "product_service": "SaaS fintech platform",
                    "customer_segment": ["Enterprise"],
                    "customer_presence": ["United States", "United Kingdom"],
                    "business_model": "B2B SaaS",
                },
            }
        ],
    }
    criteria = {
        "required_functions": {"operator": "OR", "values": ["Sales Development"]},
        "required_geographies": {"operator": "OR", "values": ["EMEA"]},
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "required_segments": {"operator": "OR", "values": ["enterprise"]},
        "required_keywords": {"operator": "OR", "values": ["12:00 PM"]},
        "min_total_experience": 5,
    }

    scored = query.score_candidate_against_criteria(profile, criteria)

    assert scored is not None
    assert scored["match_score"] >= 80
    assert scored["id"] == 101
    assert any("uploaded fields" in item["source"] for item in scored["evidence_log"])
    assert any(item["criterion"] == "Customer segments" for item in scored["matched_criteria"])
    assert "evidence_log" not in profile


def test_screening_rejects_missing_hard_experience_requirement():
    profile = {
        "id": 102,
        "name": "Junior Candidate",
        "headline": "BDR in SaaS",
        "total_experience_years": 1.5,
        "raw_fields": {"Focused Geography": "US", "Outbound Exp": "Outbound SaaS"},
        "roles": [],
    }
    criteria = {
        "required_functions": {"operator": "OR", "values": ["Sales Development"]},
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "min_total_experience": 5,
    }

    assert query.score_candidate_against_criteria(profile, criteria) is None


def test_filter_candidates_returns_ranked_non_mutating_results():
    strong = {
        "id": 1,
        "name": "Strong Match",
        "headline": "Enterprise SaaS BDR",
        "about": "Outbound US pipeline generation.",
        "total_experience_years": 6,
        "raw_fields": {"Focused Geography": "US", "Outbound Exp": "Outbound"},
        "roles": [{"title": "BDR", "company": "Acme", "details": "SaaS enterprise outbound", "duration_years": 2}],
    }
    weak = {
        "id": 2,
        "name": "Weak Match",
        "headline": "Operations Associate",
        "about": "Internal operations.",
        "total_experience_years": 8,
        "raw_fields": {},
        "roles": [],
    }
    criteria = {
        "required_functions": {"operator": "OR", "values": ["Sales Development"]},
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "required_geographies": {"operator": "OR", "values": ["US"]},
    }

    results = asyncio.run(query.filter_candidates_by_criteria([weak, strong], criteria))

    assert [item["id"] for item in results] == [1]
    assert "match_score" in results[0]
    assert "evidence_log" not in strong
    assert "evidence_log" not in weak


def test_shortlist_intelligence_reads_raw_profile_fields_and_gap_education_overlap():
    profile = {
        "id": 301,
        "name": "Raw Field Seller",
        "headline": "Revenue leader",
        "about": "Built net-new enterprise SaaS pipeline across APAC.",
        "location": "Bengaluru, Karnataka, India",
        "raw_fields": {
            "Focused Geography": "India, ASEAN and APAC markets",
            "Recruiter Summary": "Managed a team of 18 sellers.",
            "imported_extra_fields": {
                "segment": {"label": "Segment", "value": "Enterprise customers"},
            },
            "enrichment": {
                "education": [
                    {
                        "college": "Indian School of Business",
                        "degree": "MBA",
                        "start_date": "2020-01-01",
                        "end_date": "2021-06-01",
                    }
                ],
                "profile_claims": {
                    "max_people_managed": 18,
                    "years_team_management": 3,
                },
            },
        },
        "roles": [
            {
                "company": "Alpha SaaS",
                "title": "Account Executive",
                "start_date": "2018-01-01",
                "end_date": "2019-12-01",
                "duration_years": 1.9,
            },
            {
                "company": "Beta Cloud",
                "title": "Enterprise Sales Manager",
                "start_date": "2021-07-01",
                "end_date": "2024-01-01",
                "duration_years": 2.5,
            },
        ],
    }

    pack = query.build_shortlist_intelligence_pack(profile, {})

    assert pack["current_location"] == {
        "raw": "Bengaluru, Karnataka, India",
        "city": "Bengaluru",
        "state": "Karnataka",
        "country": "India",
    }
    assert pack["gap_analysis"]["has_gap_years"] is True
    assert pack["gap_analysis"]["gaps"][0]["education_overlap"] is True
    assert pack["team_management"]["max_people_managed"] == 18
    assert pack["team_management"]["years_team_management"] == 3
    assert "Focused Geography" in pack["full_profile_evidence"]["raw_fields"]


def test_evidence_catalog_fallback_samples_raw_field_keys(monkeypatch):
    monkeypatch.setattr(query, "_EVIDENCE_CATALOG_CACHE", None)
    monkeypatch.setattr(query, "get_db_connection", lambda **_kwargs: None)

    catalog = query.build_db_evidence_catalog(
        force_refresh=True,
        profiles=[
            {
                "id": 1,
                "raw_fields": {
                    "Focused Geography": "APAC",
                    "LinkedIn Enrichment": {
                        "profile_claims": {
                            "alliance_years": "5+ years",
                        }
                    },
                },
            }
        ],
    )

    raw_paths = {item["path"] for item in catalog["raw_field_keys"]}
    assert catalog["source"] == "fallback"
    assert "candidates" in catalog["tables"]
    assert "roles" in catalog["tables"]
    assert "Focused Geography" in raw_paths
    assert "LinkedIn Enrichment.profile_claims.alliance_years" in raw_paths


def test_strict_shortlist_uses_enriched_fields_and_requires_role_duration_for_years():
    criteria = {
        "min_function_years": [{"function": "Partner Sales", "min_years": 5, "aliases": ["alliance management"]}],
        "required_geographies": {"operator": "OR", "values": ["APAC"], "min_years": 5},
    }
    explicit = {
        "id": 501,
        "name": "Explicit Alliance Candidate",
        "headline": "Partner ecosystem leader",
        "location": "Bengaluru, India",
        "raw_fields": {
            "Recruiter Notes": "5+ years in alliance management with APAC market ownership.",
        },
        "roles": [{"title": "Alliance Manager", "company": "Acme", "details": "Alliance management for APAC market.", "duration_years": 6}],
    }
    location_only = {
        "id": 502,
        "name": "Location Only Candidate",
        "headline": "Sales leader",
        "location": "Bengaluru, India",
        "raw_fields": {},
        "roles": [],
    }

    scored = query._strict_shortlist_score_candidate(explicit, criteria)

    assert scored["shortlist_status"] == "shortlisted"
    assert any(item["criterion"] == "Function-specific tenure" for item in scored["matched_criteria"])
    assert query._strict_shortlist_score_candidate(location_only, criteria) is None


def test_geography_uses_enriched_profile_and_market_evidence():
    criteria = {"required_geographies": {"operator": "OR", "values": ["APAC"]}}
    company_geo = {
        "id": 601,
        "name": "Company Geo",
        "location": "Gurgaon, India",
        "raw_fields": {"addressWithCountry": "Gurgaon, Haryana India"},
        "roles": [{
            "title": "Account Manager",
            "company": "Capillary",
            "details": "Enterprise account planning.",
            "company_details": {"headquarters": "Bengaluru, India", "offices": ["Singapore"]},
        }],
    }
    address_only = {
        "id": 603,
        "name": "Address Only",
        "location": "Gurgaon, India",
        "raw_fields": {"addressWithCountry": "Gurgaon, Haryana India"},
        "roles": [{"title": "Account Manager", "company": "Acme", "details": "Enterprise account planning."}],
    }
    explicit_market = {
        "id": 602,
        "name": "Market Owner",
        "raw_fields": {"Summary": "Owned APAC partner market coverage and generated channel pipeline across India and Singapore."},
        "roles": [],
    }

    company_geo_scored = query._strict_shortlist_score_candidate(company_geo, criteria)
    address_scored = query._strict_shortlist_score_candidate(address_only, criteria)
    explicit_scored = query._strict_shortlist_score_candidate(explicit_market, criteria)

    assert company_geo_scored["shortlist_status"] == "shortlisted"
    assert address_scored is None
    assert explicit_scored["shortlist_status"] == "shortlisted"
    assert any(item["criterion"] == "Geographies" for item in explicit_scored["matched_criteria"])


def test_country_query_matches_explicit_region_profile_evidence():
    criteria = {"required_geographies": {"operator": "OR", "values": ["Singapore"]}}
    profile = {
        "id": 605,
        "name": "Region Mention",
        "about": "Owned APAC expansion and partner revenue.",
        "roles": [{"title": "Channel Manager", "company": "Acme", "details": "APAC partner ecosystem.", "duration_years": 4}],
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)

    assert scored["shortlist_status"] == "shortlisted"


def test_current_company_scope_requires_latest_employer():
    criteria = {
        "required_companies": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": [{"company": "Google", "employment_scope": "current_employer"}],
        }
    }
    current = {
        "id": 606,
        "roles": [
            {"title": "BDR", "company": "Google", "details": "Outbound", "duration_years": 2},
            {"title": "SDR", "company": "Acme", "details": "Outbound", "duration_years": 2},
        ],
    }
    past = {
        "id": 607,
        "roles": [
            {"title": "BDR", "company": "Acme", "details": "Outbound", "duration_years": 2},
            {"title": "SDR", "company": "Google", "details": "Outbound", "duration_years": 2},
        ],
    }

    assert query._strict_shortlist_score_candidate(current, criteria)["shortlist_status"] == "shortlisted"
    assert query._strict_shortlist_score_candidate(past, criteria) is None


def test_current_company_scope_uses_dates_not_role_insertion_order():
    criteria = {
        "required_companies": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": [{"company": "NewCo", "employment_scope": "current_employer"}],
        }
    }
    profile = {
        "id": 610,
        "roles": [
            {
                "title": "SDR",
                "company": "OldCo",
                "start_date": "2019-01-01",
                "end_date": "2021-01-01",
            },
            {
                "title": "AE",
                "company": "NewCo",
                "start_date": "2024-01-01",
                "end_date": "",
            },
        ],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria)["shortlist_status"] == "shortlisted"


def test_duplicate_title_column_cannot_replace_current_role_company_metadata():
    profile = {
        "id": 613,
        "raw_fields": {
            "experiences/0/companyName": "Razorpay",
            "experiences/0/title": "Senior Sales Associate",
            "experiences/0/title.1": "Keka HR",
            "experiences/0/jobStartedOn": "2026-04-01 00:00:00",
            "experiences/0/companyIndustry": "Computer Software",
        },
        "roles": [
            {
                "company": "Razorpay",
                "title": "Senior Sales Associate",
                "start_date": "2026-04-01T00:00:00+00:00",
                "end_date": "2026-06-02T19:03:01+00:00",
                "company_details": {
                    "business_model": "Fintech",
                    "product_service": "Payment gateway",
                },
            },
            {
                "company": "Keka HR",
                "title": "BDR",
                "start_date": "2025-08-01",
                "end_date": "2026-04-01",
                "company_details": {"business_model": "B2B"},
            },
        ],
    }
    criteria = {
        "required_industries": {
            "operator": "OR",
            "values": ["fintech"],
            "employment_scope": "current_employer",
        }
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)

    assert scored["shortlist_status"] == "shortlisted"
    assert scored["contributing_roles_details"]["roles"][0]["company"] == "Razorpay"


def test_same_company_and_start_date_merge_preserves_normalized_current_title():
    profile = {
        "id": 614,
        "raw_fields": {
            "experiences/0/companyName": "Highradius",
            "experiences/0/title": "HighRadius",
            "experiences/0/title.1": "Student Volunteer",
            "experiences/0/jobStartedOn": "2026-01-01 00:00:00",
            "experiences/0/jobDescription": "HighRadius FinTech solutions and outbound pipeline.",
        },
        "roles": [
            {
                "company": "Highradius",
                "title": "Inside Sales (ABM)",
                "start_date": "2026-01-01T00:00:00+00:00",
                "end_date": "2026-06-02T16:32:50+00:00",
                "company_details": {"business_model": "Fintech SaaS"},
            }
        ],
    }

    roles = query._profile_roles_with_raw_experience(profile)
    current = query._current_roles({"roles": roles})

    assert len(roles) == 1
    assert current[0]["title"] == "Inside Sales (ABM)"
    assert "FinTech solutions" in current[0]["details"]


def test_current_company_attributes_cannot_match_a_past_employer():
    criteria = {
        "required_company_details": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": ["SaaS"],
        }
    }
    profile = {
        "id": 611,
        "roles": [
            {
                "title": "SDR",
                "company": "OldSaaSCo",
                "start_date": "2019-01-01",
                "end_date": "2021-01-01",
                "company_details": {"business_model": "B2B SaaS"},
            },
            {
                "title": "Consultant",
                "company": "CurrentServicesCo",
                "start_date": "2024-01-01",
                "end_date": "",
                "company_details": {"business_model": "IT services"},
            },
        ],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria) is None


def test_company_industry_cannot_match_customer_or_role_description_text():
    criteria = {
        "required_industries": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": ["fintech", "financial services"],
        }
    }
    profile = {
        "id": 615,
        "roles": [{
            "title": "Account Executive",
            "company": "Engagement SaaS Co",
            "start_date": "2025-01-01",
            "end_date": "",
            "details": "Help fintech companies improve customer engagement.",
            "company_details": {
                "industry": "Customer engagement software",
                "product_service": "Marketing automation platform",
                "business_model": "SaaS",
                "customer_segment": ["Financial Services", "Retail"],
            },
        }],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria) is None


def test_company_product_descriptors_match_schema_grounded_exact_aliases():
    criteria = {
        "required_industries": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": ["Cross-border payment solutions"],
        }
    }
    profile = {
        "id": 616,
        "roles": [{
            "title": "Account Executive",
            "company": "Airwallex",
            "start_date": "2025-01-01",
            "end_date": "",
            "company_details": {
                "industry": "Cross-border payment solutions",
                "product_service": "Cross-border payment solutions",
                "business_model": "B2B",
            },
        }],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria)["shortlist_status"] == "shortlisted"


def test_current_company_attribute_tenure_only_counts_current_role():
    criteria = {
        "required_company_details": {
            "operator": "OR",
            "employment_scope": "current_employer",
            "values": ["SaaS"],
            "min_years": 2,
        }
    }
    profile = {
        "id": 612,
        "roles": [
            {
                "title": "SDR",
                "company": "OldSaaSCo",
                "start_date": "2018-01-01",
                "end_date": "2023-01-01",
                "duration_years": 5,
                "company_details": {"business_model": "B2B SaaS"},
            },
            {
                "title": "AE",
                "company": "CurrentSaaSCo",
                "start_date": "2025-01-01",
                "end_date": "",
                "duration_years": 1,
                "company_details": {"business_model": "B2B SaaS"},
            },
        ],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria) is None


def test_current_company_language_and_global_scope_apply_to_company_attributes():
    assert query._query_company_scope("Candidates whose current company is Google") == "current_employer"
    assert query._query_company_scope("Candidates at their present employer in SaaS") == "current_employer"

    plan = {
        "filter_plan": {
            "hard_filters": {
                "required_company_details": {"operator": "OR", "values": ["SaaS"]},
                "required_segments": {"operator": "OR", "values": ["Enterprise"]},
            },
            "company_scope": {"employment_scope": "current_employer"},
        }
    }
    criteria = query._coerce_filter_plan_to_criteria(plan, "currently working at an enterprise SaaS company")

    assert criteria["required_company_details"]["employment_scope"] == "current_employer"
    assert criteria["required_segments"]["employment_scope"] == "current_employer"


def test_filter_plan_sanitizes_echoed_schema_contract_metadata():
    plan = {
        "filter_plan": {
            "hard_filters": {
                "required_industries": {
                    "shape": {"operator": "AND", "values": ["fintech"]},
                    "evidence": "strict evidence match; missing evidence does not pass",
                    "supports_min_years": False,
                    "supports_employment_scope": True,
                }
            },
            "company_scope": {"employment_scope": "current_employer"},
        }
    }

    criteria = query._coerce_filter_plan_to_criteria(
        plan,
        "candidates who are currently working in fintech companies",
    )

    assert criteria["required_industries"] == {
        "operator": "OR",
        "values": ["fintech"],
        "employment_scope": "current_employer",
    }


def test_industry_expansion_receives_observed_company_product_vocabulary(monkeypatch):
    monkeypatch.setattr(query, "PROFILES_BY_ID", {
        1: {
            "roles": [{
                "company": "Airwallex",
                "company_details": {
                    "industry": "Cross-border payment solutions",
                    "product_service": "Cross-border payment solutions",
                    "business_model": "B2B",
                },
            }]
        }
    })

    terms = query._observed_company_terms_for_expansion("Industry")

    assert "Cross-border payment solutions" in terms


def test_fintech_base_domain_taxonomy_covers_characteristic_products():
    terms = query.INDUSTRY_DOMAIN_TAXONOMY["fintech"]

    assert "payment gateway" in terms
    assert "cross-border payment" in terms
    assert "digital banking" in terms
    assert "regtech" in terms


def test_prompt_catalog_keeps_complete_discovered_schema():
    catalog = {
        "version": "test",
        "source": "database",
        "tables": {
            f"table_{table_idx}": {
                "candidate_related": True,
                "columns": [
                    {"name": f"column_{column_idx}", "type": "text", "category": "candidate_fact"}
                    for column_idx in range(45)
                ],
            }
            for table_idx in range(26)
        },
        "raw_field_keys": [
            {"path": f"custom.field_{idx}", "category": "candidate_fact"}
            for idx in range(125)
        ],
    }

    compact = query.compact_evidence_catalog_for_prompt(catalog)

    assert len(compact["tables"]) == 26
    assert len(compact["tables"][0]["columns"]) == 45
    assert len(compact["raw_field_keys"]) == 125


def test_series_c_and_above_uses_ordered_funding_stage():
    criteria = {"funding_stage_min": {"stage": "Series C", "employment_scope": "current_employer"}}
    series_d = {
        "id": 608,
        "roles": [{"title": "BDR", "company": "ScaleCo", "company_details": {"funding_stage": "Series D"}}],
    }
    series_b = {
        "id": 609,
        "roles": [{"title": "BDR", "company": "EarlyCo", "company_details": {"funding_stage": "Series B"}}],
    }

    assert query._strict_shortlist_score_candidate(series_d, criteria)["shortlist_status"] == "shortlisted"
    assert query._strict_shortlist_score_candidate(series_b, criteria) is None


def test_filter_plan_moves_market_region_out_of_current_location():
    plan = {
        "filter_plan": {
            "hard_filters": {
                "required_locations": {"operator": "OR", "values": ["APAC"]},
                "required_functions": {"operator": "OR", "values": ["Sales Development"], "min_years": 5},
            },
            "geography_policy": {"use_current_location": False},
        }
    }

    criteria = query._coerce_filter_plan_to_criteria(
        plan,
        "Candidates with 5 years of sales development experience and have worked in APAC market",
    )

    assert "required_locations" not in criteria
    assert criteria["required_geographies"]["values"] == ["APAC"]


def test_filter_plan_enforces_every_explicit_requirement_in_bdr_us_saas_query():
    criteria = query._coerce_filter_plan_to_criteria(
        {
            "filter_plan": {
                "hard_filters": {
                    "required_functions": ["Sales Development"],
                    "required_industries": ["SaaS"],
                },
                "duration_rules": [
                    {"field": "function", "function": "Sales Development", "min_years": 15}
                ],
            }
        },
        "BDRs with US experience and 15+ years in SaaS",
    )

    assert "BDR" in criteria["required_functions"]["values"]
    assert "US" in criteria["required_geographies"]["values"]
    assert criteria["required_industries"]["min_years"] == 15
    assert "min_function_years" not in criteria


def test_outbound_lead_qualification_query_rejects_inbound_only_bdrs():
    criteria = query._coerce_filter_plan_to_criteria(
        {
            "filter_plan": {
                "hard_filters": {
                    "required_functions": ["BDR"],
                    "min_function_years": [
                        {"function": "Sales Development", "min_years": 5, "aliases": ["BDR"]}
                    ],
                }
            }
        },
        "BDRs with 5+ years of exp in outbound lead qualification",
    )

    assert criteria["min_function_years"] == [
        {
            "function": "Outbound lead qualification",
            "min_years": 5.0,
            "aliases": [
                "outbound lead qualification",
                "outbound prospecting",
                "outbound lead generation",
                "cold outreach",
                "cold calling",
            ],
        }
    ]

    inbound_only = {
        "id": 9101,
        "roles": [
            {
                "title": "BDR",
                "company": "InboundCo",
                "details": "Inbound lead qualification and demo request follow-up.",
                "duration_years": 6,
            }
        ],
    }
    outbound = {
        "id": 9102,
        "roles": [
            {
                "title": "BDR",
                "company": "OutboundCo",
                "details": "Outbound lead qualification and cold outreach.",
                "duration_years": 6,
            }
        ],
    }

    assert query._strict_shortlist_score_candidate(inbound_only, criteria) is None
    assert query._strict_shortlist_score_candidate(outbound, criteria)["shortlist_status"] == "shortlisted"


def test_filter_plan_coerces_field_value_duration_rules_and_embedded_function_years():
    plan = {
        "filter_plan": {
            "hard_filters": {
                "required_functions": {
                    "operator": "OR",
                    "values": [
                        {"function": "Sales Development", "min_function_years": 5},
                        {"function": "BDR"},
                    ],
                },
                "required_geographies": {"operator": "OR", "values": ["North America"]},
            },
            "duration_rules": [
                {"field": "min_total_experience", "value": 5},
                {"field": "min_tenure_in_latest_role", "value": 2, "context": "selling into North America"},
                {"geography": "north america", "min_years": 2},
                {"geography": "north america", "min_function_years": 2},
            ],
        }
    }

    criteria = query._coerce_filter_plan_to_criteria(
        plan,
        "top 5 BDR candidates with 5+ years and 2 years selling into North America",
    )

    assert criteria["min_total_experience"] == 5
    assert criteria["required_geographies"]["min_years"] == 2
    assert any(
        item["function"] == "Sales Development" and float(item["min_years"]) == 5
        for item in criteria["min_function_years"]
    )


def test_filter_plan_preserves_scalar_min_function_years_with_required_functions():
    plan = {
        "filter_plan": {
            "hard_filters": {
                "required_functions": {"operator": "OR", "values": ["Sales Development", "BDR"]},
                "min_function_years": 5,
            },
        }
    }

    criteria = query._coerce_filter_plan_to_criteria(
        plan,
        "top 5 BDR candidates with at least 5 years specifically in BDR roles",
    )

    assert len(criteria["min_function_years"]) == 1
    assert criteria["min_function_years"][0]["function"] == "Sales Development"
    assert float(criteria["min_function_years"][0]["min_years"]) == 5
    assert "BDR" in criteria["min_function_years"][0]["aliases"]


def test_filter_plan_top_n_not_disabled_by_specifically():
    criteria = query._coerce_filter_plan_to_criteria(
        {"filter_plan": {"top_n": 5, "hard_filters": {"required_functions": ["BDR"]}}},
        "top 5 BDR candidates specifically with 5 years in sales development",
    )

    assert criteria["top_n"] == 5


def test_filter_plan_applies_query_years_to_company_detail_tenure():
    criteria = query._coerce_filter_plan_to_criteria(
        {
            "filter_plan": {
                "top_n": 5,
                "hard_filters": {
                    "required_functions": {"operator": "OR", "values": ["BDR"]},
                    "min_function_years": 5,
                    "required_company_details": {"operator": "OR", "values": ["B2B SaaS", "software"]},
                },
            }
        },
        "top 5 BDR candidates with at least 5 years specifically in B2B SaaS or software industry experience",
    )

    assert criteria["required_company_details"]["min_years"] == 5
    assert "min_function_years" not in criteria


def test_filter_plan_applies_query_years_to_segment_tenure():
    criteria = query._coerce_filter_plan_to_criteria(
        {
            "filter_plan": {
                "top_n": 5,
                "hard_filters": {
                    "required_functions": {"operator": "OR", "values": ["BDR"]},
                    "min_function_years": 3,
                    "required_segments": {"enterprise": 3},
                },
            }
        },
        "top 5 BDR candidates with at least 3 years specifically selling to enterprise customers",
    )

    assert criteria["required_segments"]["min_years"] == 3
    assert criteria["required_segments"]["values"] == ["enterprise"]
    assert "min_function_years" not in criteria


def test_filter_plan_applies_query_years_to_market_tenure():
    criteria = query._coerce_filter_plan_to_criteria(
        {
            "filter_plan": {
                "top_n": 5,
                "hard_filters": {
                    "required_functions": {"operator": "OR", "values": ["BDR"]},
                    "min_function_years": 2,
                    "required_geographies": {"operator": "OR", "values": ["APAC"]},
                },
            }
        },
        "top 5 BDR candidates with at least 2 years specifically selling into or covering APAC market; current location in APAC alone should not count",
    )

    assert criteria["required_geographies"]["min_years"] == 2
    assert "min_function_years" not in criteria


def test_skills_only_alliances_does_not_satisfy_function_years():
    criteria = {"min_function_years": [{"function": "Partner Sales", "min_years": 5, "aliases": ["alliance management", "alliances"]}]}
    skills_only = {
        "id": 603,
        "name": "Skills Only",
        "raw_fields": {"Skills": "Business Alliances, Sales, CRM"},
        "roles": [{"title": "Business Development", "company": "Acme", "details": "General sales role.", "duration_years": 8}],
    }
    explicit_claim = {
        "id": 604,
        "name": "Explicit Partner Seller",
        "raw_fields": {"Summary": "6+ years in alliance management and partner sales."},
        "roles": [{"title": "Partner Sales Manager", "company": "Beta", "details": "Alliance management and partner sales.", "duration_years": 6}],
    }

    skills_scored = query._strict_shortlist_score_candidate(skills_only, criteria)
    explicit_scored = query._strict_shortlist_score_candidate(explicit_claim, criteria)

    assert skills_scored is None
    assert explicit_scored["shortlist_status"] == "shortlisted"
    assert any(item["criterion"] == "Function-specific tenure" for item in explicit_scored["matched_criteria"])


def test_low_tenure_bdr_is_rejected_even_with_keyword_matches():
    criteria = {
        "min_total_experience": 5,
        "min_function_years": [{"function": "Sales Development", "min_years": 5, "aliases": ["BDR", "SDR", "outbound"]}],
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "required_keywords": {"operator": "OR", "values": ["outbound", "salesforce", "hubspot"]},
    }
    profile = {
        "id": 610,
        "name": "Keyword Rich Junior",
        "headline": "BDR with Salesforce and HubSpot",
        "total_experience_years": 6,
        "raw_fields": {"Skills": "Salesforce, HubSpot, outbound prospecting"},
        "roles": [{
            "title": "BDR",
            "company": "Acme",
            "details": "SaaS outbound prospecting using Salesforce and HubSpot.",
            "duration_years": 0.75,
            "company_details": {"business_model": "SaaS", "product_service": "SaaS"},
        }],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria) is None


def test_raw_dated_experience_rows_can_satisfy_function_years_when_roles_are_incomplete():
    criteria = {
        "min_function_years": [{"function": "Sales Development", "min_years": 3, "aliases": ["business development", "outbound"]}],
    }
    profile = {
        "id": 611,
        "name": "Raw Dated Seller",
        "total_experience_years": 5,
        "raw_fields": {
            "experiences/0/title": "Business Development Representative",
            "experiences/0/companyName": "Acme",
            "experiences/0/jobStartedOn": "2020-01-01 00:00:00",
            "experiences/0/jobEndedOn": "2023-01-01 00:00:00",
            "experiences/0/jobDescription": "Business development and outbound pipeline generation.",
        },
        "roles": [],
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)

    assert scored["shortlist_status"] == "shortlisted"
    assert scored["calculated_experience"]["min_function_years"][0]["duration"] >= 3


def test_skills_only_bdr_crm_does_not_satisfy_function_years():
    criteria = {
        "min_function_years": [{"function": "Sales Development", "min_years": 5, "aliases": ["BDR", "SDR"]}],
    }
    profile = {
        "id": 612,
        "name": "Skills Only BDR",
        "total_experience_years": 8,
        "raw_fields": {"Skills": "BDR, SDR, Salesforce, HubSpot, CRM"},
        "roles": [{"title": "Business Analyst", "company": "Acme", "details": "Operations role.", "duration_years": 8}],
    }

    assert query._strict_shortlist_score_candidate(profile, criteria) is None


def test_bdr_hard_constraint_regression_rejects_juniors_and_accepts_prasang_like_profile():
    criteria = {
        "min_total_experience": 5,
        "min_function_years": [{
            "function": "Sales Development",
            "min_years": 5,
            "aliases": ["BDR", "SDR", "business development", "account development", "inside sales", "outbound", "lead generation"],
        }],
        "required_geographies": {"operator": "OR", "values": ["North America"], "min_years": 2},
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "required_segments": {"operator": "OR", "values": ["B2B"]},
        "required_keywords": {"operator": "OR", "values": ["Salesforce", "outbound", "lead generation"]},
    }
    abhidip_like = {
        "id": 613,
        "name": "Junior SDR",
        "total_experience_years": 0.92,
        "raw_fields": {"Skills": "Salesforce, HubSpot, outbound"},
        "roles": [{
            "title": "Senior Associate - SDR",
            "company": "GEP Worldwide",
            "details": "North America enterprise outreach.",
            "duration_years": 0.42,
            "company_details": {"business_model": "B2B", "product_service": "SaaS", "customer_presence": ["North America"]},
        }],
    }
    divyansh_like = {
        "id": 614,
        "name": "Short Tenure Inside Seller",
        "total_experience_years": 6.8,
        "raw_fields": {"Skills": "Salesforce, outbound, lead generation"},
        "roles": [{
            "title": "Inside Sales",
            "company": "HighRadius",
            "details": "North America outbound lead generation.",
            "duration_years": 0.5,
            "company_details": {"business_model": "SaaS", "product_service": "SaaS", "customer_segment": ["Enterprise"], "customer_presence": ["North America"]},
        }],
    }
    prasang_like = {
        "id": 615,
        "name": "Senior Account Development",
        "total_experience_years": 9.6,
        "raw_fields": {
            "Skills": "Salesforce.com, SaaS Sales, Sales Development, B2B Commerce, Lead Generation",
        },
        "roles": [
            {
                "title": "Account Development",
                "company": "Avalara",
                "details": "Tax automation solutions for MM/Enterprise companies (B2B SaaS Sales) with North America account coverage.",
                "duration_years": 2.42,
                "company_details": {
                    "business_model": "SaaS",
                    "product_service": "Tax compliance software",
                    "customer_segment": ["Businesses", "Mid-Market"],
                    "customer_presence": ["North America", "Europe"],
                    "headquarters": "Seattle, WA, USA",
                },
            },
            {
                "title": "Assistant Manager Business Development",
                "company": "Configurator Solutions",
                "details": "Business development, appointment setting, and lead generation for B2B SaaS.",
                "duration_years": 3.5,
                "company_details": {
                    "business_model": "SaaS",
                    "product_service": "SaaS",
                    "customer_segment": ["Enterprise", "Mid-Market"],
                },
            },
        ],
    }

    assert query._strict_shortlist_score_candidate(abhidip_like, criteria) is None
    assert query._strict_shortlist_score_candidate(divyansh_like, criteria) is None
    assert query._strict_shortlist_score_candidate(prasang_like, criteria)["shortlist_status"] == "shortlisted"


def test_scoped_duration_counts_only_matching_function_roles():
    profile = {
        "id": 616,
        "total_experience_years": 8,
        "roles": [
            {"title": "BDR", "company": "Acme", "details": "Outbound sales development.", "duration_years": 2},
            {"title": "Operations Manager", "company": "Beta", "details": "Internal operations.", "duration_years": 6},
        ],
    }

    result = query.evaluate_scoped_duration(
        profile,
        dimension="function",
        criterion={"operator": "OR", "values": ["Sales Development", "BDR"]},
        min_years=5,
        label="Sales Development",
    )

    assert result["qualified"] is False
    assert result["duration"] == 2


def test_scoped_duration_counts_industry_segment_and_company_detail_roles():
    profile = {
        "id": 617,
        "roles": [
            {
                "title": "BDR",
                "company": "Acme",
                "details": "Sold enterprise SaaS platform.",
                "duration_years": 3,
                "company_details": {
                    "industry": "Software",
                    "product_service": "B2B SaaS platform",
                    "customer_segment": ["Enterprise"],
                },
            },
            {
                "title": "BDR",
                "company": "RetailCo",
                "details": "Retail field sales.",
                "duration_years": 4,
                "company_details": {
                    "industry": "Retail",
                    "product_service": "Consumer goods",
                    "customer_segment": ["Consumer"],
                },
            },
        ],
    }

    industry = query.evaluate_scoped_duration(
        profile,
        dimension="company_detail",
        criterion={"operator": "OR", "values": ["SaaS"]},
        min_years=3,
        label="SaaS",
    )
    segment = query.evaluate_scoped_duration(
        profile,
        dimension="segment",
        criterion={"operator": "OR", "values": ["Enterprise"]},
        min_years=3,
        label="Enterprise",
    )

    assert industry["qualified"] is True
    assert industry["duration"] == 3
    assert segment["qualified"] is True
    assert segment["duration"] == 3


def test_geography_scoped_duration_requires_explicit_market_action_not_customer_presence():
    criteria = {"required_geographies": {"operator": "OR", "values": ["North America"], "min_years": 2}}
    customer_presence_only = {
        "id": 618,
        "roles": [{
            "title": "BDR",
            "company": "Acme",
            "details": "Outbound SaaS sales development.",
            "duration_years": 3,
            "company_details": {"customer_presence": ["North America"]},
        }],
    }
    accounts_title_only = {
        "id": 621,
        "roles": [{
            "title": "Accounts & Administration",
            "company": "Acme India",
            "details": "Back-office accounting and administration.",
            "source_location": "India",
            "duration_years": 7,
        }],
    }
    explicit_market = {
        "id": 619,
        "roles": [{
            "title": "BDR",
            "company": "Acme",
            "details": "Owned North America outbound pipeline and quota.",
            "duration_years": 3,
            "company_details": {"customer_presence": ["North America"]},
        }],
    }

    assert query._strict_shortlist_score_candidate(customer_presence_only, criteria) is None
    assert query.evaluate_scoped_duration(
        accounts_title_only,
        dimension="geography",
        criterion={"operator": "OR", "values": ["APAC"]},
        min_years=2,
        label="APAC",
    )["qualified"] is False
    scored = query._strict_shortlist_score_candidate(explicit_market, criteria)
    assert scored["shortlist_status"] == "shortlisted"
    assert scored["scoped_tenure"][0]["evidence_ids"]


def test_overlapping_scoped_roles_are_not_double_counted():
    profile = {
        "id": 620,
        "roles": [
            {
                "title": "BDR",
                "company": "Acme",
                "details": "Sales development.",
                "start_date": "2020-01-01",
                "end_date": "2022-01-01",
                "duration_years": 2,
            },
            {
                "title": "Sales Development Representative",
                "company": "Acme",
                "details": "Outbound sales development.",
                "start_date": "2021-01-01",
                "end_date": "2023-01-01",
                "duration_years": 2,
            },
        ],
    }

    result = query.evaluate_scoped_duration(
        profile,
        dimension="function",
        criterion={"operator": "OR", "values": ["Sales Development", "BDR"]},
        min_years=4,
        label="Sales Development",
    )

    assert result["qualified"] is False
    assert 2.9 <= result["duration"] <= 3.1
    assert result["roles"][0]["company"] == "Acme"
    assert result["roles"][0]["title"] == "BDR"
    assert result["roles"][0]["start_date"] == "2020-01-01"
    assert result["roles"][0]["end_date"] == "2022-01-01"
    assert result["roles"][0]["duration_years"] == 2
    assert result["roles"][0]["matched_value"] in {"Sales Development", "BDR"}
    assert "why_counted" in result["roles"][0]
    assert result["evidence"][0]["role"]["source_label"].startswith("LinkedIn/imported Experience:")
    assert result["evidence"][0]["role"]["source_type"] == "internal_db"


def test_auditor_output_requires_valid_evidence_citations():
    profile = {
        "id": 621,
        "evidence_log": [
            {"id": "ev1", "criterion": "Function-specific tenure", "snippet": "BDR for 5 years"},
        ],
    }

    valid = {"final_status": "verified_match", "answer": "Matches based on role history.", "reasoning": "BDR tenure is supported by role duration.", "evidence_ids": ["ev1"]}
    invalid_id = {"final_status": "verified_match", "answer": "Matches based on ev9.", "reasoning": "Supported by ev9.", "evidence_ids": ["ev9"]}
    clean_prose = {"final_status": "verified_match", "answer": "Matches.", "reasoning": "BDR tenure is supported.", "evidence_ids": ["ev1"]}

    assert query._audit_output_is_evidence_valid(profile, valid) is True
    assert query._audit_output_is_evidence_valid(profile, invalid_id) is False
    assert query._audit_output_is_evidence_valid(profile, clean_prose) is True


def test_fallback_reasoning_hides_evidence_ids_but_keeps_citations():
    profile = {
        "id": 622,
        "match_score": 91,
        "evidence_log": [
            {"id": "ev1", "criterion": "Function-specific tenure", "source": "profile", "snippet": "BDR for 5 years"},
            {"id": "ev2", "criterion": "Company details", "source": "web company profile", "snippet": "SaaS platform"},
        ],
    }

    reasoning = query._fallback_reasoning_from_evidence(profile)
    payload = query._fallback_audit_payload_from_evidence(profile)

    assert "ev1" not in reasoning
    assert "ev2" not in reasoning
    assert "Function-specific tenure from profile" in reasoning
    assert payload["evidence_ids"] == ["ev1", "ev2"]


def test_visible_shortlist_reasoning_strips_model_evidence_ids():
    text = (
        "The candidate has over 15 years of experience in Sales Development (ev1), "
        "worked in SaaS (ev2), and matches business development (ev3, ev5)."
    )

    cleaned = query._clean_visible_evidence_ids(text)

    assert "ev1" not in cleaned
    assert "ev2" not in cleaned
    assert "ev3" not in cleaned
    assert "ev5" not in cleaned
    assert "Sales Development" in cleaned
    assert "worked in SaaS" in cleaned


def test_evidence_log_dedupes_repeated_uploaded_field_snippets_before_ids():
    evidence = [
        {"criterion": "Functions", "source": "uploaded fields.Skills", "snippet": "Business Development, SaaS"},
        {"criterion": "Functions", "source": "uploaded fields.Skills", "snippet": "Business Development, SaaS"},
        {"criterion": "Industries", "source": "uploaded fields.Skills", "snippet": "SaaS, Cloud Computing"},
    ]

    deduped = query._assign_evidence_ids(query._dedupe_evidence_log(evidence))

    assert len(deduped) == 2
    assert [item["id"] for item in deduped] == ["ev1", "ev2"]
    assert deduped[0]["criterion"] == "Functions"
    assert deduped[1]["criterion"] == "Industries"


def test_friendly_evidence_text_explains_where_signal_was_mentioned():
    uploaded = query._friendly_evidence_text({
        "criterion": "Industries",
        "value": "SaaS",
        "source": "uploaded fields.Skills",
        "snippet": "CRM, Enterprise Software, SaaS, Lead Generation",
    })
    role = query._friendly_evidence_text({
        "criterion": "Sales Development tenure",
        "value": "Sales Development",
        "source": "role history",
        "snippet": "Senior Business Development Consultant at Oracle: title: senior business development consultant",
        "role": {"title": "Senior Business Development Consultant", "company": "Oracle"},
    })

    assert uploaded == (
        'He has mentioned SaaS; the matching text includes '
        '"CRM, Enterprise Software, SaaS, Lead Generation".'
    )
    assert role == (
        "At Oracle, his role was Senior Business Development Consultant, "
        "so this role was counted toward Sales Development experience."
    )


def test_strict_shortlist_sort_prefers_relevant_duration_then_score():
    lower_duration = {
        "id": 701,
        "shortlist_status": "shortlisted",
        "match_score": 100,
        "total_experience_years": 20,
        "calculated_experience": {"required_functions": {"duration": 2}},
    }
    higher_score = {
        "id": 702,
        "shortlist_status": "shortlisted",
        "match_score": 80,
        "total_experience_years": 5,
        "calculated_experience": {"required_functions": {"duration": 5}},
    }
    higher_duration = {
        "id": 703,
        "shortlist_status": "shortlisted",
        "match_score": 70,
        "total_experience_years": 4,
        "calculated_experience": {"required_functions": {"duration": 6}},
    }

    criteria = {"required_functions": {"operator": "OR", "values": ["BDR"]}}
    assert [item["id"] for item in query._sort_strict_shortlist_candidates([lower_duration, higher_score, higher_duration], criteria)] == [703, 702, 701]


def test_strict_numeric_queries_scan_full_scope_not_semantic_sample():
    assert query._strict_search_should_scan_full_scope({"required_functions": {"operator": "OR", "values": ["BDR"]}}) is False
    assert query._strict_search_should_scan_full_scope({"min_total_experience": 5}) is True
    assert query._strict_search_should_scan_full_scope({
        "min_function_years": [{"function": "Sales Development", "min_years": 5}]
    }) is True
    assert query._strict_search_should_scan_full_scope({
        "required_geographies": {"operator": "OR", "values": ["North America"], "min_years": 2}
    }) is True


def test_strict_filter_returns_only_shortlisted_matches():
    profiles = [
        {"id": 1, "name": "Match", "roles": [{"title": "BDR", "company": "Acme", "details": "Outbound BDR", "duration_years": 2}]},
        {"id": 2, "name": "Miss", "roles": [{"title": "Engineer", "company": "Acme", "details": "Backend", "duration_years": 2}]},
    ]
    criteria = {"required_functions": {"operator": "OR", "values": ["BDR"]}}

    results = asyncio.run(query.filter_candidates_by_criteria(profiles, criteria))

    assert [item["id"] for item in results] == [1]
    assert results[0]["shortlist_status"] == "shortlisted"


def test_cost_tracker_reports_known_and_unknown_pricing():
    known = query.TokenCostTracker()
    known.add_usage("gpt-4.1-mini", "hello", "world", "test")
    known_summary = known.get_summary()
    assert "Estimated Cost" in known_summary
    assert "pricing_missing" not in known_summary
    assert known.total_cost > 0

    unknown = query.TokenCostTracker()
    unknown.add_usage("unknown-shortlist-model", "hello", "world", "test")
    assert "pricing_missing for unknown-shortlist-model" in unknown.get_summary()


def test_shortlist_dynamic_matching_routes_sales_queries_to_ai_review():
    criteria = {
        "hiring_company": {"company": "Mixpanel", "prioritize_competitors": True},
        "required_functions": {"operator": "OR", "values": ["Hunting"]},
        "required_segments": {"operator": "OR", "values": ["enterprise"]},
    }

    assert query._uses_dynamic_ai_matching(criteria) is True


def test_hiring_company_relevance_scores_competitor_as_priority_not_hard_requirement():
    profile = {
        "id": 302,
        "name": "Relevant Competitor Seller",
        "total_experience_years": 8,
        "raw_fields": {},
        "roles": [
            {
                "company": "Amplitude",
                "title": "Enterprise Account Executive",
                "details": "Closed new enterprise analytics logos.",
                "duration_years": 4,
            }
        ],
    }
    criteria = {
        "hiring_company": {"company": "Mixpanel", "prioritize_competitors": True},
        "_web_company_facts": {
            "competitors": [
                {
                    "target": "Mixpanel",
                    "companies": ["Amplitude", "Heap"],
                    "sources": [{"url": "https://example.com", "title": "Analytics competitors"}],
                }
            ]
        },
    }

    scored = query.score_candidate_against_criteria(profile, criteria)

    assert scored is not None
    assert any(item["criterion"] == "Hiring company relevance" for item in scored["matched_criteria"])
    assert scored["shortlist_intelligence"]["query_relevant_company_facts"]["competitors"][0]["target"] == "Mixpanel"


def test_dynamic_review_order_keeps_complete_profile_list(monkeypatch):
    profiles = [
        {
            "id": 1,
            "name": "Relevant Partner Seller",
            "headline": "Alliance manager",
            "about": "Partner sales and APAC market coverage.",
            "total_experience_years": 7,
            "roles": [{"title": "Alliance Manager", "company": "Acme", "details": "Partner sales across APAC"}],
        },
        {"id": 2, "name": "High Tenure", "total_experience_years": 12, "roles": []},
        {"id": 3, "name": "Other 1", "total_experience_years": 1, "roles": []},
        {"id": 4, "name": "Other 2", "total_experience_years": 2, "roles": []},
        {"id": 5, "name": "Other 3", "total_experience_years": 3, "roles": []},
    ]
    criteria = {
        "min_function_years": [{"function": "Partner Sales", "min_years": 5, "aliases": ["alliance management"]}],
        "required_geographies": {"operator": "OR", "values": ["APAC"]},
    }

    ordered = query._dynamic_candidate_candidates(profiles, "alliance management APAC", criteria)

    assert len(ordered) == len(profiles)
    assert {profile["id"] for profile in ordered} == {1, 2, 3, 4, 5}
    assert ordered[0]["id"] in {1, 2}


def test_process_query_shortlists_complete_source_with_reasoning(monkeypatch):
    profiles = [
        {
            "id": idx,
            "name": f"Candidate {idx}",
            "headline": "BDR",
            "total_experience_years": idx,
            "roles": [{"title": "BDR", "company": "Acme", "details": "Outbound", "duration_years": 2}],
        }
        for idx in range(1, 8)
    ]

    class _FakeCriteriaResponse:
        content = '{"min_function_years": [{"function": "Sales Development", "min_years": 1, "aliases": ["BDR"]}]}'

    async def fake_ainvoke(_prompt):
        return _FakeCriteriaResponse()

    class _FakeLLM:
        model_name = "fake-model"
        ainvoke = staticmethod(fake_ainvoke)

    async def fake_reasoning(profile, *_args, **_kwargs):
        return f"{profile['name']} matches the BDR requirement."

    monkeypatch.setattr(query, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(query, "normalize_query_with_llm", lambda value: value)
    monkeypatch.setattr(query, "llm", _FakeLLM())
    monkeypatch.setattr(query, "generate_reasoning_for_profile", fake_reasoning)
    monkeypatch.setattr(query, "PROFILES_BY_ID", {profile["id"]: profile for profile in profiles})

    async def collect_events():
        events = []
        async for item in query.process_query_main("BDR partner sales", "session", query.TokenCostTracker()):
            events.append(item)
        return events

    events = asyncio.run(collect_events())
    progress = next(item for item in events if isinstance(item, dict) and item.get("type") == "progress_start")
    complete = next(item for item in events if isinstance(item, dict) and item.get("type") == "complete")
    chunks = [item for item in events if isinstance(item, dict) and item.get("type") == "profile_chunk"]

    assert progress["total"] == 7
    assert len(chunks) == 7
    assert complete["total_reviewed"] == 7
    assert complete["selected_candidate_count"] == 7
    assert complete["evidence_scored"] == 7
    assert complete["llm_reviewed"] == 0
    assert complete["audited_count"] == 0
    assert complete["verified_count"] == 7
    assert {item["data"]["shortlist_status"] for item in chunks} == {"verified_match"}
    assert all(item["data"]["answer"] for item in chunks)


def test_process_query_scores_complete_scope_without_semantic_prefilter(monkeypatch):
    profiles = [
        {
            "id": 1,
            "name": "Semantic Near Miss",
            "headline": "Account executive",
            "total_experience_years": 5,
            "roles": [{"title": "Account Executive", "company": "Acme", "details": "Renewals"}],
        },
        {
            "id": 2,
            "name": "Actual Match",
            "headline": "BDR",
            "total_experience_years": 4,
            "roles": [{"title": "BDR", "company": "Beta", "details": "Outbound prospecting"}],
        },
    ]

    class _FakeCriteriaResponse:
        content = '{"required_keywords": {"operator": "OR", "values": ["outbound"]}}'

    async def fake_ainvoke(_prompt):
        return _FakeCriteriaResponse()

    class _FakeLLM:
        model_name = "fake-model"
        ainvoke = staticmethod(fake_ainvoke)

    async def fake_reasoning(profile, *_args, **_kwargs):
        return f"{profile['name']} has explicit outbound evidence."

    monkeypatch.setattr(query, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(query, "normalize_query_with_llm", lambda value: value)
    monkeypatch.setattr(query, "llm", _FakeLLM())
    monkeypatch.setattr(query, "generate_reasoning_for_profile", fake_reasoning)
    monkeypatch.setattr(query, "PROFILES_BY_ID", {profile["id"]: profile for profile in profiles})
    monkeypatch.setattr(query, "build_db_evidence_catalog", lambda **_kwargs: {})

    class _FailIfUsedEmbeddings:
        @staticmethod
        def embed_query(_text):
            raise AssertionError("shortlist must not use a semantic prefilter")

    monkeypatch.setattr(query, "embeddings", _FailIfUsedEmbeddings())

    async def collect_events():
        events = []
        async for item in query.process_query_main("people with outbound experience", "session", query.TokenCostTracker()):
            events.append(item)
        return events

    events = asyncio.run(collect_events())
    complete = next(item for item in events if isinstance(item, dict) and item.get("type") == "complete")

    assert "Loading the complete selected candidate scope..." in events
    assert [item["id"] for item in complete["data"]] == [2]
    assert complete["total_reviewed"] == 2
    assert complete["verified_count"] == 1


def test_strict_shortlist_emits_enriched_geography_match():
    profile = {
        "id": 401,
        "name": "APAC Seller",
        "headline": "Sales leader",
        "location": "Gurgaon, India",
        "total_experience_years": 10,
        "roles": [{"title": "Business Development", "company": "Acme", "details": "Owned APAC market pipeline.", "duration_years": 8}],
        "raw_fields": {"Notes": "Covered India and Singapore customers."},
    }
    criteria = {"required_geographies": {"operator": "OR", "values": ["APAC"]}}

    reviewed = query._strict_shortlist_score_candidate(profile, criteria)

    assert reviewed["shortlist_status"] == "shortlisted"
    assert reviewed["is_verified_match"] is True
    assert reviewed["missing_criteria"] == []
    assert any(item["criterion"] == "Geographies" for item in reviewed["matched_criteria"])


def test_strict_shortlist_geography_rejects_company_hq_without_market_evidence():
    profile = {
        "id": 402,
        "name": "HQ Only Seller",
        "headline": "Sales leader",
        "location": "Bengaluru, India",
        "total_experience_years": 18,
        "roles": [
            {
                "title": "Senior Business Development Consultant",
                "company": "Oracle",
                "details": "Business development and SaaS consulting.",
                "duration_years": 15,
                "company_details": {"headquarters": "Redwood City, California, USA"},
            }
        ],
        "raw_fields": {"Skills": "SaaS, Business Development, Lead Generation"},
    }
    criteria = {
        "required_geographies": {"operator": "OR", "values": ["US", "USA", "United States"]},
        "_screening_query": "BDRs with US experience and 15+ years in SaaS",
    }

    reviewed = query._strict_shortlist_score_candidate(profile, criteria)

    assert reviewed is None


def test_strict_shortlist_returns_dynamic_requirement_verification():
    profile = {
        "id": 403,
        "name": "Verified BDR",
        "headline": "Senior Business Development Consultant",
        "total_experience_years": 18.4,
        "roles": [
            {
                "title": "Senior Business Development Consultant",
                "company": "Oracle",
                "details": "Owned US outbound pipeline and account development for B2B SaaS and enterprise software customers.",
                "duration_years": 15.3,
                "company_details": {"product_service": "B2B SaaS enterprise software"},
            }
        ],
        "raw_fields": {"Skills": "SaaS, Enterprise Software, Lead Generation, Business Development"},
    }
    criteria = {
        "min_function_years": [{
            "function": "Sales Development",
            "min_years": 15,
            "aliases": ["BDR", "business development", "account development", "outbound", "lead generation"],
        }],
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "required_geographies": {"operator": "OR", "values": ["US"]},
        "_screening_query": "BDRs with US experience and 15+ years in SaaS",
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)

    assert scored["shortlist_status"] == "shortlisted"
    assert scored["decision_narrative"].startswith("Qualified match: Yes.")
    assert "BDRs with US experience and 15+ years in SaaS" in scored["decision_narrative"]
    assert "company headquarters" in scored["decision_narrative"].lower()
    assert re.search(r"\bev\d+\b", scored["decision_narrative"], re.IGNORECASE) is None
    assert len(scored["requirement_breakdown"]) == 3
    assert {item["status"] for item in scored["requirement_breakdown"]} == {"qualified"}
    assert {item["category"] for item in scored["requirement_breakdown"]} == {"Tenure", "Company details", "Geography"}
    assert any(item["requirement"] == "At least 15 years in Sales Development" for item in scored["requirement_breakdown"])
    assert any(item["requirement"] == "Company details: SaaS" for item in scored["requirement_breakdown"])
    assert any(item["requirement"] == "Geographies: US" for item in scored["requirement_breakdown"])
    assert any(item["evidence_ids"] for item in scored["requirement_breakdown"])
    tenure_item = next(item for item in scored["requirement_breakdown"] if item["category"] == "Tenure")
    role_evidence = tenure_item["profile_evidence"][0]
    assert role_evidence["display_title"] == "Oracle"
    assert "Senior Business Development Consultant" in role_evidence["display_subtitle"]
    assert "15.3 yrs" in role_evidence["display_subtitle"]

    visible_text = " ".join(
        " ".join(
            [
                item.get("requirement", ""),
                item.get("why_it_supports", ""),
                " ".join(item.get("exact_evidence_text") or []),
                " ".join(item.get("cross_check") or []),
                " ".join(item.get("not_counted") or []),
                " ".join(evidence.get("summary", "") for evidence in item.get("evidence_found") or []),
            ]
        )
        for item in scored["requirement_breakdown"]
    )
    assert re.search(r"\bev\d+\b", visible_text, re.IGNORECASE) is None


def test_requirement_breakdown_groups_duplicate_evidence_text_under_one_category():
    criteria = {"required_company_details": {"operator": "OR", "values": ["SaaS"]}}
    repeated_text = "CIOs, Software as a Service (SaaS), Business Development, Cloud Software"
    evidence_log = query._assign_evidence_ids(query._add_friendly_evidence_text([
        {
            "criterion": "Company details",
            "value": value,
            "source": "uploaded fields.Skills",
            "snippet": repeated_text,
            "source_text": repeated_text,
        }
        for value in ["SaaS", "Software as a Service", "Cloud Software"]
    ]))

    breakdown = query._build_requirement_breakdown(
        criteria,
        matched_criteria=[{"criterion": "Company details", "value": "SaaS"}],
        missing_criteria=[],
        evidence_log=evidence_log,
    )

    assert len(breakdown) == 1
    assert breakdown[0]["category"] == "Company details"
    assert breakdown[0]["requirement"] == "Company details: SaaS"
    assert breakdown[0]["status"] == "qualified"
    assert len(breakdown[0]["evidence_found"]) == 3
    assert len(breakdown[0]["profile_evidence"]) == 1
    assert set(breakdown[0]["profile_evidence"][0]["matched_terms"]) == {"Cloud Software", "SaaS", "Software as a Service"}
    assert len(breakdown[0]["exact_evidence_text"]) == 1


def test_requirement_breakdown_missing_market_requirement_explains_not_counted():
    criteria = {
        "required_geographies": {"operator": "OR", "values": ["US"]},
        "_screening_query": "BDRs with US experience",
    }
    evidence_log = query._assign_evidence_ids(query._add_friendly_evidence_text([
        {
            "criterion": "Company details",
            "value": "Oracle",
            "source": "role company details",
            "snippet": "Oracle headquarters: Redwood City, California, USA",
            "source_text": "Oracle headquarters: Redwood City, California, USA",
        }
    ]))

    breakdown = query._build_requirement_breakdown(
        criteria,
        matched_criteria=[],
        missing_criteria=["Geographies: US"],
        evidence_log=evidence_log,
    )
    narrative = query._build_decision_narrative(criteria, breakdown)

    assert len(breakdown) == 1
    assert breakdown[0]["status"] == "missing"
    assert "company headquarters" in " ".join(breakdown[0]["not_counted"]).lower()
    assert "directly verify" in breakdown[0]["why_it_supports"]
    assert narrative.startswith("Qualified match: No.")
    assert re.search(r"\bev\d+\b", narrative, re.IGNORECASE) is None


def test_requirement_exact_evidence_keeps_full_bulleted_text_readable():
    profile = {
        "id": 404,
        "name": "Market Evidence Seller",
        "headline": "SDR/BDR Architect Enabler",
        "raw_fields": {
            "Profile Summary": (
                "Consistently delivering measurable results across APAC, EMEA, and US markets."
                "Throughout my career, I have:• built and led high-performing SDR teams"
                "• generated US outbound pipeline• covered North America accounts"
            )
        },
        "roles": [],
    }
    criteria = {
        "required_geographies": {"operator": "OR", "values": ["US"]},
        "_screening_query": "BDRs with US experience",
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)
    geography = scored["requirement_breakdown"][0]
    exact = geography["exact_evidence_text"][0]

    assert scored["shortlist_status"] == "shortlisted"
    assert geography["category"] == "Geography"
    assert "..." not in exact
    assert "markets. throughout" in exact
    assert "\n- built and led high-performing sdr teams" in exact
    assert "\n- generated us outbound pipeline" in exact
    assert "\n- covered north america accounts" in exact
    guide_text = " ".join(geography["cross_check"])
    assert "Verify" not in guide_text
    assert "Check" not in guide_text
    assert "Confirm" not in guide_text
    assert "he has mentioned" in guide_text.lower()


def test_requirement_breakdown_includes_web_source_links_for_funding_and_company_facts():
    profile = {
        "id": 405,
        "name": "Web Backed Seller",
        "roles": [{"title": "BDR", "company": "Acme", "details": "Outbound prospecting.", "duration_years": 3}],
    }
    criteria = {
        "funding_stage_min": "Series C",
        "required_company_details": {"operator": "OR", "values": ["SaaS"]},
        "_web_company_facts": {
            "funding": [
                {
                    "company": "Acme",
                    "stage": "Series C",
                    "sources": [{"url": "https://example.com/acme-funding", "title": "Acme funding"}],
                }
            ],
            "company_profiles": [
                {
                    "company": "Acme",
                    "product_service": "B2B SaaS platform",
                    "sources": [{"url": "https://example.com/acme-profile", "title": "Acme profile"}],
                }
            ],
        },
    }

    scored = query._strict_shortlist_score_candidate(profile, criteria)
    by_key = {item["key"]: item for item in scored["requirement_breakdown"]}

    assert scored["shortlist_status"] == "shortlisted"
    assert by_key["funding_stage_min"]["category"] == "Funding"
    assert by_key["required_company_details"]["category"] == "Company details"
    assert by_key["funding_stage_min"]["sources"][0]["url"] == "https://example.com/acme-funding"
    assert by_key["required_company_details"]["sources"][0]["url"] == "https://example.com/acme-profile"
    assert by_key["funding_stage_min"]["evidence_found"][0]["sources"][0]["title"] == "Acme funding"
    assert by_key["required_company_details"]["evidence_found"][0]["sources"][0]["title"] == "Acme profile"
    assert by_key["funding_stage_min"]["profile_evidence"][0]["sources"][0]["title"] == "Acme funding"
    assert by_key["required_company_details"]["profile_evidence"][0]["sources"][0]["title"] == "Acme profile"


def test_requirement_breakdown_maps_competitor_requirement_to_competitor_category():
    criteria = {
        "competitor_of": [{"company": "Mixpanel"}],
        "_competitor_resolution": {"target": "Mixpanel", "validated_companies": ["Amplitude"]},
    }
    evidence_log = query._assign_evidence_ids(query._add_friendly_evidence_text([
        {
            "criterion": "Companies",
            "value": "Amplitude",
            "source": "role company",
            "snippet": "Enterprise Account Executive at Amplitude",
            "source_text": "Enterprise Account Executive at Amplitude",
        }
    ]))

    breakdown = query._build_requirement_breakdown(
        criteria,
        matched_criteria=[{"criterion": "Competitor", "value": "Amplitude"}],
        missing_criteria=[],
        evidence_log=evidence_log,
    )

    assert len(breakdown) == 1
    assert breakdown[0]["category"] == "Competitor"
    assert breakdown[0]["status"] == "qualified"
    assert set(breakdown[0]["profile_evidence"][0]["matched_terms"]) == {"Amplitude", "Mixpanel"}


def test_process_query_returns_only_shortlisted_matches(monkeypatch):
    profiles = [
        {
            "id": 1,
            "name": "Verified Candidate",
            "headline": "Enterprise SaaS BDR",
            "total_experience_years": 6,
            "match_score": 91,
            "roles": [{"title": "BDR", "company": "Acme", "details": "SaaS outbound", "duration_years": 2}],
            "evidence_log": [{"source": "profile", "snippet": "SaaS outbound"}],
        },
        {
            "id": 2,
            "name": "Potential Candidate",
            "headline": "BDR",
            "total_experience_years": 5,
            "match_score": 82,
            "roles": [{"title": "BDR", "company": "Beta", "details": "Outbound", "duration_years": 2}],
            "evidence_log": [{"source": "profile", "snippet": "Outbound"}],
        },
        {
            "id": 3,
            "name": "Review Error Candidate",
            "headline": "SDR",
            "total_experience_years": 4,
            "match_score": 75,
            "roles": [{"title": "SDR", "company": "Gamma", "details": "Outbound prospecting", "duration_years": 2}],
            "evidence_log": [{"source": "profile", "snippet": "Prospecting"}],
        },
    ]

    class _FakeCriteriaResponse:
        content = '{"required_keywords": {"operator": "OR", "values": ["outbound"]}}'

    async def fake_ainvoke(_prompt):
        return _FakeCriteriaResponse()

    class _FakeLLM:
        model_name = "fake-model"
        ainvoke = staticmethod(fake_ainvoke)

    async def fake_reasoning(profile, *_args, **_kwargs):
        return f"{profile['name']} has outbound evidence."

    monkeypatch.setattr(query, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(query, "normalize_query_with_llm", lambda value: value)
    monkeypatch.setattr(query, "llm", _FakeLLM())
    monkeypatch.setattr(query, "generate_reasoning_for_profile", fake_reasoning)
    monkeypatch.setattr(query, "PROFILES_BY_ID", {profile["id"]: profile for profile in profiles})

    async def collect_events():
        events = []
        async for item in query.process_query_main("BDR", "session", query.TokenCostTracker()):
            events.append(item)
        return events

    events = asyncio.run(collect_events())
    complete = next(item for item in events if isinstance(item, dict) and item.get("type") == "complete")
    chunks = [item for item in events if isinstance(item, dict) and item.get("type") == "profile_chunk"]

    assert len(chunks) == 3
    assert len(complete["data"]) == 3
    assert complete["verified_count"] == 3
    assert complete["total_reviewed"] == 3
    assert complete["llm_reviewed"] == 0
    assert complete["audited_count"] == 0
    assert [item["id"] for item in complete["data"]] == [1, 2, 3]
    assert [item["shortlist_status"] for item in complete["data"]] == ["verified_match", "verified_match", "verified_match"]
    assert all(item["is_verified_match"] for item in complete["data"])


def test_process_query_web_toggle_gates_company_fact_enrichment(monkeypatch):
    profile = {
        "id": 44,
        "name": "Series C Seller",
        "headline": "Sales development lead",
        "total_experience_years": 7,
        "roles": [
            {
                "title": "BDR",
                "company": "Acme",
                "details": "SaaS outbound sales development",
                "duration_years": 5,
                "company_details": {
                    "product_service": "B2B SaaS platform",
                    "funding_stage": "Series C",
                },
            }
        ],
    }

    class _FakeCriteriaResponse:
        content = (
            '{"required_company_details": {"operator": "OR", "values": ["SaaS"]}, '
            '"funding_stage_min": "Series C"}'
        )

    async def fake_ainvoke(_prompt):
        return _FakeCriteriaResponse()

    class _FakeLLM:
        model_name = "fake-model"
        ainvoke = staticmethod(fake_ainvoke)

    class _FakeEmbeddings:
        model = "fake-embedding-model"

        @staticmethod
        def embed_query(_text):
            return [0.1, 0.2, 0.3]

    async def fake_reasoning(profile, *_args, **_kwargs):
        return f"{profile['name']} matches."

    calls = {"web_enrichment": 0}

    async def fake_company_web_enrichment(_query, criteria, _pool, _tracker):
        calls["web_enrichment"] += 1
        enriched = dict(criteria)
        enriched["_web_company_facts"] = {
            "company_profiles": [
                {"company": "Acme", "funding_stage": "Series C", "sources": [{"url": "https://example.com/acme"}]}
            ]
        }
        return enriched

    monkeypatch.setattr(query, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(query, "normalize_query_with_llm", lambda value: value)
    monkeypatch.setattr(query, "llm", _FakeLLM())
    monkeypatch.setattr(query, "generate_reasoning_for_profile", fake_reasoning)
    monkeypatch.setattr(query, "_expand_keywords_with_llm", lambda *_args, **_kwargs: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(query, "get_db_connection", lambda *args, **kwargs: None)
    monkeypatch.setattr(query, "embeddings", _FakeEmbeddings())
    monkeypatch.setattr(query, "PROFILES_BY_ID", {profile["id"]: profile})
    monkeypatch.setattr(query, "SCREENING_WEB_SEARCH_DEFAULT", True)
    monkeypatch.setattr(query, "enrich_criteria_with_candidate_company_web_facts", fake_company_web_enrichment)

    async def collect_events(use_web_search):
        events = []
        async for item in query.process_query_main(
            "Series C SaaS sales development candidates",
            "session",
            query.TokenCostTracker(),
            use_web_search=use_web_search,
        ):
            events.append(item)
        return events

    off_events = asyncio.run(collect_events(False))
    assert calls["web_enrichment"] == 0
    assert "Scoring profiles against enriched candidate data..." in off_events

    on_events = asyncio.run(collect_events(True))
    assert calls["web_enrichment"] == 1
    assert "Researching company facts..." in on_events
    assert "Scoring profiles with web-backed company data..." in on_events


class _SummaryCursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params or []))

    def fetchall(self):
        return [("Shortlisted", 3), ("To be started", 2)]


class _SummaryConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj


def test_browse_summary_counts_use_direct_sql_for_recruiter_scope(monkeypatch):
    cursor = _SummaryCursor()
    conn = _SummaryConnection(cursor)
    returned = []
    monkeypatch.setattr(browse, "get_db_connection", lambda **kwargs: conn)
    monkeypatch.setattr(browse, "return_db_connection", lambda c: returned.append(c))

    result = asyncio.run(
        browse.fetch_browse_summary_counts(
            current_user=_user(user_id=7, role="recruiter"),
            role_id=44,
        )
    )

    sql, params = cursor.executed[0]
    assert result["total"] == 5
    assert result["status_counts"] == {"Shortlisted": 3, "To be started": 2}
    assert "FROM candidates c" in sql
    assert "recruitment_role_candidates" in sql
    assert "c.owner_user_id = %s" in sql
    assert params == [7, 44, 7]
    assert returned == [conn]


def test_browse_summary_counts_scope_admin_recruiter_pool(monkeypatch):
    cursor = _SummaryCursor()
    conn = _SummaryConnection(cursor)
    monkeypatch.setattr(browse, "get_db_connection", lambda **kwargs: conn)
    monkeypatch.setattr(browse, "return_db_connection", lambda c: None)

    result = asyncio.run(
        browse.fetch_browse_summary_counts(
            current_user=_user(user_id=1, role="admin"),
            view_scope="recruiter_pools",
            recruiter_filter_id=9,
        )
    )

    sql, params = cursor.executed[0]
    assert result["effective_scope"] == "recruiter_pools"
    assert result["effective_recruiter"] == 9
    assert "c.owner_user_id IS NOT NULL" in sql
    assert "c.owner_user_id = %s" in sql
    assert params == [9]


class _RolesCursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params or []))

    def fetchall(self):
        return [
            (11, "BDR Role", "", 4, 1, None, 7, "Recruiter", "recruiter@example.com"),
        ]


def test_role_counts_join_active_candidates(monkeypatch):
    cursor = _RolesCursor()
    conn = _SummaryConnection(cursor)
    monkeypatch.setattr(roles, "get_db_connection", lambda *args, **kwargs: conn)
    monkeypatch.setattr(roles, "return_db_connection", lambda c: None)

    result = asyncio.run(roles.get_roles(current_user=_user(user_id=7, role="recruiter")))

    sql, params = cursor.executed[0]
    assert result["roles"][0]["candidate_count"] == 4
    assert "COUNT(DISTINCT c.id)" in sql
    assert "LEFT JOIN candidates c ON c.id = rc.candidate_id" in sql
    assert "COALESCE(c.is_archived, FALSE) = FALSE" in sql
    assert tuple(params) == (7,)

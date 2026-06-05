import asyncio

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from backend.api import schemas
from backend.api.routes import ai_columns
from backend.api.routes import browse
from backend.db.ai_column_migrate import ensure_ai_column_migrations
from backend.services import ai_columns as ai_columns_service
from backend.services.ai_columns import (
    build_candidate_context_pack,
    build_field_catalog,
    build_candidate_context,
    build_query_plan,
    classify_ai_column_prompt,
    compute_career_facts,
    evaluate_required_fields,
    flatten_profile_context,
    fill_prompt_template,
    map_raw_outputs_to_schema_keys,
    map_career_facts_to_outputs,
    run_candidate_query_tools,
)
from backend.services.ai_column_presets import list_ai_column_presets


class _FakeCursor:
    def __init__(self):
        self.executed = []
        self.fetchone_results = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        if self.fetchone_results:
            return self.fetchone_results.pop(0)
        return None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.autocommit = False

    def cursor(self):
        return self._cursor


def _user(user_id=7, role="recruiter"):
    return schemas.User(
        id=user_id,
        username=f"user{user_id}@example.com",
        email=f"user{user_id}@example.com",
        full_name="User",
        role=role,
        permissions={},
    )


def test_fetch_profile_refreshes_stale_cache_without_roles(monkeypatch):
    stale_profile = {"id": 123, "name": "Stale Candidate", "roles": []}
    refreshed_profile = {
        "id": 123,
        "name": "Fresh Candidate",
        "roles": [{"company": "Acme", "title": "Account Executive"}],
    }
    profiles_by_id = {123: stale_profile}

    def fake_refresh(candidate_ids):
        assert candidate_ids == [123]
        profiles_by_id[123] = refreshed_profile
        return 1

    monkeypatch.setattr(ai_columns.query, "PROFILES_BY_ID", profiles_by_id)
    monkeypatch.setattr(ai_columns.query, "refresh_profiles_in_cache", fake_refresh)

    assert ai_columns._fetch_profile(123) is refreshed_profile


def test_ai_column_migration_resets_legacy_raw_fields():
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)
    ensure_ai_column_migrations(conn)

    executed_sql = "\n".join(query for query, _ in cursor.executed)
    assert "CREATE TABLE IF NOT EXISTS ai_column_definitions" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS ai_column_cells" in executed_sql
    assert "context_inputs JSONB" in executed_sql
    assert "raw_fields = raw_fields - 'ai_columns'" in executed_sql


def test_field_catalog_includes_default_imported_and_ai_columns():
    profiles = [
        {
            "id": 1,
            "name": "Deepak Basavaraj",
            "first_name": "Deepak",
            "last_name": "Basavaraj",
            "linkedin": "https://linkedin.com/in/deepak",
            "raw_fields": {
                "import_company": "Exotel",
                "linkedin_url": "https://linkedin.com/in/deepak",
                "Current CTC": "15 LPA",
                "imported_extra_fields": {
                    "current_ctc": {
                        "label": "Current CTC",
                        "source_header": "Current CTC",
                        "value": "15 LPA",
                    }
                },
            },
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        }
    ]
    catalog = build_field_catalog(
        profiles,
        ai_columns=[
            {
                "id": 9,
                "name": "Company Location",
                "slug": "company_location",
                "output_schema": [{"key": "result", "label": "Result", "primary": True}],
            }
        ],
    )
    groups = {group["group"]: group["items"] for group in catalog}
    default_keys = {item["key"] for item in groups["Default Fields"]}
    imported_keys = {item["key"] for item in groups["Imported Fields"]}
    extra_keys = {item["key"] for item in groups["Imported Extra Fields"]}
    ai_keys = {item["key"] for item in groups["AI Columns"]}

    assert "candidate.linkedin" in default_keys
    assert "candidate.full_name" in default_keys
    assert "raw.import_company" in imported_keys
    assert "ai.company_location.result" in ai_keys
    assert "extra.current_ctc" in extra_keys
    assert groups["Imported Extra Fields"][0]["token"].startswith("{extra.")


def test_required_fields_and_prompt_fill_use_ai_context():
    profile = {
        "id": 1,
        "name": "Deepak Basavaraj",
        "first_name": "Deepak",
        "last_name": "Basavaraj",
        "linkedin": "https://linkedin.com/in/deepak",
        "roles": [{"title": "Account Director", "company": "Exotel"}],
        "raw_fields": {
            "import_company": "Exotel",
            "Current CTC": "15 LPA",
            "imported_extra_fields": {
                "current_ctc": {
                    "label": "Current CTC",
                    "source_header": "Current CTC",
                    "value": "15 LPA",
                }
            },
        },
    }
    context = build_candidate_context(profile, ai_values={"ai.company_location.result": "Bengaluru"})
    rendered = fill_prompt_template(
        "Check {candidate.linkedin}, {extra.current_ctc}, and compare with {ai.company_location.result}",
        context,
    )
    ok, missing = evaluate_required_fields(
        ["candidate.linkedin", "ai.company_location.result"],
        context,
    )

    assert ok is True
    assert missing == []
    assert "https://linkedin.com/in/deepak" in rendered
    assert "15 LPA" in rendered
    assert "Bengaluru" in rendered
    assert context["candidate.full_name"] == "Deepak Basavaraj"
    assert context["candidate.name"] == "Deepak Basavaraj"
    assert context["Linkedin Profile"] == "https://linkedin.com/in/deepak"
    assert context["row.raw_fields.import_company"] == "Exotel"
    assert context["raw.Current CTC"] == "15 LPA"
    assert context["extra.current_ctc"] == "15 LPA"
    assert build_candidate_context_pack(context)["imported_extra_fields"]["current_ctc"] == "15 LPA"


def test_candidate_context_exposes_role_and_column_context_inputs():
    context = build_candidate_context(
        {
            "id": 1,
            "name": "Ada Lovelace",
            "first_name": "Ada",
            "last_name": "Lovelace",
            "linkedin": "https://linkedin.com/in/ada",
        },
        role_context={"name": "Founding AE", "job_description": "Close enterprise deals."},
        context_inputs={"our_product": "AI recruiting", "pitch_context": "Founder-led outreach."},
    )

    assert context["role.name"] == "Founding AE"
    assert context["role.job_description"] == "Close enterprise deals."
    assert context["context.our_product"] == "AI recruiting"
    assert context["context.pitch_context"] == "Founder-led outreach."
    assert context["context.our_product_or_pitch_context"] == "AI recruiting"


def test_field_catalog_includes_role_and_column_context_tokens():
    catalog = build_field_catalog([])
    groups = {group["group"]: group["items"] for group in catalog}

    assert {item["key"] for item in groups["Role Context"]} == {
        "role.name",
        "role.job_description",
    }
    assert {item["key"] for item in groups["Column Context"]} == {
        "context.our_product",
        "context.pitch_context",
    }


def test_preset_catalog_covers_research_company_fit_and_outreach_packs():
    presets = {preset["id"]: preset for preset in list_ai_column_presets()}

    assert presets["current_role_verification"]["category"] == "Person"
    assert presets["pricing_strategy"]["category"] == "Company"
    assert "role.job_description" in presets["jd_fit_score"]["required_inputs"]
    assert presets["personalized_email_opener"]["context_fields"] == ["pitch_context"]
    assert presets["average_current_tenure"]["mode"] == "auto"
    assert presets["linkedin_recent_activity"]["mode"] == "web_research"


def test_prompt_classifier_routes_row_web_and_hybrid_queries():
    assert classify_ai_column_prompt("Calculate the average tenure and current job tenure")["data_source"] == "row"
    assert classify_ai_column_prompt("Have they posted content on LinkedIn in the last 30 days?") == {
        "data_source": "web",
        "web_required_reason": "public_linkedin_recent_activity",
        "routing_mode": "web_research",
    }
    assert classify_ai_column_prompt("Score each candidate 1-10 against this JD: https://example.com/job")["data_source"] == "hybrid"
    assert classify_ai_column_prompt("Any recent layoffs at the current company?")["data_source"] == "web"


def test_career_facts_collapse_company_and_exclude_memberships(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 1, tzinfo=tz)

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    profile = {
        "id": 1,
        "name": "Sales Candidate",
        "city": "Bengaluru",
        "total_experience_years": 12,
        "roles": [
            {"title": "Enterprise Account Executive", "company": "Acme SaaS", "start_date": "2020-01", "end_date": "Present", "city": "Bengaluru"},
            {"title": "Account Executive", "company": "Acme SaaS", "start_date": "2018-01", "end_date": "2019-12", "city": "Bengaluru"},
            {"title": "Sales Development Representative", "company": "Beta Co", "start_date": "2015", "end_date": "2017", "city": "Mumbai"},
            {"title": "Member", "company": "RevGenius", "start_date": "2022", "end_date": "Present", "city": "Remote"},
        ],
    }
    context = build_candidate_context(profile)
    facts = compute_career_facts(context)
    outputs = map_career_facts_to_outputs(
        "Calculate average tenure, current job tenure, and number of cities.",
        [
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "career_city_count", "label": "Career City Count"},
        ],
        facts,
    )

    assert facts["unique_company_count"] == 2
    assert facts["completed_company_count"] == 1
    assert "RevGenius" not in facts["companies"]
    assert facts["ae_experience_months"] >= 60
    assert outputs["average_tenure_months"] == "25"
    assert outputs["current_job_months"].isdigit()
    assert outputs["career_city_count"] == "2"


def test_career_facts_average_tenure_excludes_current_company(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 1, tzinfo=tz)

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    context = build_candidate_context(
        {
            "id": 101,
            "name": "Completed Tenure Candidate",
            "roles": [
                {"title": "Current Role", "company": "C", "start_date": "2025-08-01", "end_date": "Present"},
                {"title": "Role A", "company": "A", "start_date": "2020-01-01", "end_date": "2021-12-01"},
                {"title": "Role B", "company": "B", "start_date": "2022-01-01", "end_date": "2024-12-01"},
                {"title": "Member", "company": "RevGenius", "start_date": "2020-01-01", "end_date": "Present"},
            ],
        }
    )

    facts = compute_career_facts(context)
    outputs = map_career_facts_to_outputs(
        "Calculate average tenure and current job tenure in months.",
        [
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "tenure_reasoning", "label": "Tenure Reasoning"},
        ],
        facts,
    )

    assert facts["completed_company_count"] == 2
    assert facts["completed_company_months"] == 60
    assert facts["average_tenure_months"] == 30
    assert facts["current_job_months"] == 11
    assert outputs["average_tenure_months"] == "30"
    assert outputs["current_job_months"] == "11"
    assert "Average tenure (completed roles): 30 months" in outputs["tenure_reasoning"]


def test_career_facts_parse_month_year_and_use_duration_fallback(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 2, tzinfo=tz)

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    context = build_candidate_context(
        {
            "id": 44,
            "name": "Legacy Candidate",
            "roles": [
                {
                    "title": "Sales Development Representative",
                    "company": "Wayground",
                    "start_date": "03-2026",
                    "end_date": "",
                },
                {
                    "title": "Financial Technology Advisor",
                    "company": "Highradius",
                    "duration_years": 1.92,
                },
            ],
        }
    )

    facts = compute_career_facts(context)
    tenures = {item["company"]: item["months"] for item in facts["company_tenures"]}

    assert facts["unique_company_count"] == 2
    assert tenures["Wayground"] == 4
    assert tenures["Highradius"] == 23
    assert facts["total_experience_months"] == 27
    assert "Highradius" in facts["companies"]


def test_career_facts_merge_same_company_without_counting_gap_months():
    context = build_candidate_context(
        {
            "id": 45,
            "name": "Repeat Company Candidate",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Highradius",
                            "title": "HighRadius",
                            "start_date": "2025-07-01",
                            "end_date": "2025-11-01",
                        },
                        {
                            "company": "Highradius",
                            "title": "Business Development Intern",
                            "start_date": "2023-08-01",
                            "end_date": "2023-09-01",
                        },
                    ]
                }
            },
        }
    )

    facts = compute_career_facts(context)

    assert facts["total_experience_months"] == 7
    assert facts["company_tenures"] == [
        {
            "company": "Highradius",
            "months": 7,
            "years": 0.58,
            "titles": ["HighRadius", "Business Development Intern"],
            "is_current_company": True,
        }
    ]


def test_career_facts_marks_enterprise_saas_ae_qualification():
    profile = {
        "id": 2,
        "name": "Qualified AE",
        "roles": [
            {
                "title": "Enterprise Account Executive",
                "company": "Acme SaaS",
                "start_date": "2021-01",
                "end_date": "Present",
                "company_details": {
                    "product_service": "SaaS platform",
                    "customer_segment": ["Enterprise"],
                    "business_model": "B2B subscription",
                },
            },
            {"title": "Senior AE", "company": "Beta SaaS", "start_date": "2017-01", "end_date": "2020-12"},
            {"title": "Account Executive", "company": "Gamma Cloud", "start_date": "2013-01", "end_date": "2016-12"},
            {"title": "Member", "company": "RevGenius", "start_date": "2022", "end_date": "Present"},
        ],
    }
    context = build_candidate_context(profile)
    facts = compute_career_facts(context)
    outputs = map_career_facts_to_outputs(
        (
            "Calculate average tenure in months, current job tenure in months, and mark Yes if "
            "10+ years overall experience, minimum 5+ years account executive experience, and "
            "currently working in an enterprise segment focused SaaS company."
        ),
        [
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "overall_experience_months", "label": "Total Experience Months"},
            {"key": "ae_experience_months", "label": "Account Executive Experience Months"},
            {"key": "current_enterprise_saas", "label": "Current Enterprise SaaS"},
            {"key": "qualified", "label": "Qualified"},
        ],
        facts,
    )

    assert "RevGenius" not in facts["companies"]
    assert facts["total_experience_months"] >= 120
    assert facts["ae_experience_months"] >= 60
    assert facts["current_company_enterprise_saas"] == "Yes"
    assert outputs["qualified"] == "Yes"
    assert outputs["current_enterprise_saas"] == "Yes"


def test_career_facts_detect_job_hopping_from_company_windows():
    profile = {
        "id": 5,
        "name": "Frequent Switcher",
        "roles": [
            {"title": "Account Executive", "company": "Alpha SaaS", "start_date": "2023-01", "end_date": "2023-10"},
            {"title": "Senior Account Executive", "company": "Alpha SaaS", "start_date": "2023-11", "end_date": "2024-04"},
            {"title": "Account Executive", "company": "Beta Cloud", "start_date": "2022-01", "end_date": "2022-10"},
            {"title": "BDR", "company": "Gamma Tech", "start_date": "2021-01", "end_date": "2021-09"},
            {"title": "Member", "company": "RevGenius", "start_date": "2021-01", "end_date": "Present"},
        ],
    }
    context = build_candidate_context(profile)
    facts = compute_career_facts(context)
    plan = build_query_plan(
        "Is this candidate a job hopper with too many short stints?",
        context,
        [{"key": "job_hopping", "label": "Job Hopping", "primary": True}],
        classify_ai_column_prompt("Is this candidate a job hopper with too many short stints?"),
    )
    tools = run_candidate_query_tools("Is this candidate a job hopper?", context, facts, plan)
    outputs = map_career_facts_to_outputs(
        "Is this candidate a job hopper with too many short stints?",
        [{"key": "job_hopping", "label": "Job Hopping", "primary": True}],
        facts,
    )

    assert facts["unique_company_count"] == 3
    assert facts["short_company_stints_count"] == 3
    assert facts["job_hopping_status"] == "Yes"
    assert "RevGenius" not in facts["companies"]
    assert "job_hopping" in plan["tool_calls"]
    assert tools["job_hopping"]["status"] == "Yes"
    assert outputs["job_hopping"] == "Yes"


def test_universal_query_plan_and_tools_cover_enriched_profile_dimensions():
    profile = {
        "id": 3,
        "name": "Segment Seller",
        "headline": "Enterprise SaaS AE across EMEA",
        "raw_fields": {
            "enrichment": {
                "verification_status": "passed",
                "roles": [
                    {
                        "company": "Acme SaaS",
                        "title": "Enterprise Account Executive",
                        "start_date": "2020-01-01",
                        "end_date": "2022-12-01",
                        "duration_months": 36,
                        "function": "Hunting",
                        "product_service": "SaaS",
                        "industry": "SaaS",
                        "customer_segment": ["Enterprise", "SMB"],
                        "business_model": "B2B subscription",
                        "details": "Closed new logo SMB and enterprise deals across EMEA.",
                    }
                ],
                "profile_claims": {
                    "segments": ["Enterprise", "SMB"],
                    "geographies": ["EMEA"],
                    "functions": [{"function": "Hunting", "reason": "about"}],
                },
            }
        },
    }
    context = build_candidate_context(profile)
    facts = compute_career_facts(context)
    plan = build_query_plan(
        "Has 3+ years selling to SMB in EMEA with hunting experience?",
        context,
        [{"key": "result", "label": "Result", "primary": True}],
        classify_ai_column_prompt("Has 3+ years selling to SMB in EMEA with hunting experience?"),
    )
    tools = run_candidate_query_tools(
        "Has 3+ years selling to SMB in EMEA with hunting experience?",
        context,
        facts,
        plan,
    )
    pack = build_candidate_context_pack(context, facts)

    assert plan["web_needed"] is False
    assert {"segment_experience", "geography_experience", "functional_experience"} <= set(plan["tool_calls"])
    assert tools["segment_experience"]["SMB"]["months"] == 36
    assert tools["geography_experience"]["EMEA"]["months"] == 36
    assert tools["functional_experience"]["Hunting"]["months"] == 36
    assert pack["profile_claims"]


def test_career_facts_prefer_verified_enrichment_roles_over_db_roles():
    context = build_candidate_context(
        {
            "id": 88,
            "name": "Verified Over Db",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Acme SaaS",
                            "title": "Account Executive",
                            "start_date": "2020-01-01",
                            "end_date": "2021-12-01",
                            "duration_months": 24,
                        }
                    ]
                }
            },
            "roles": [
                {
                    "company": "Wrong DB Co",
                    "title": "Old Role",
                    "duration_years": 10,
                }
            ],
        }
    )

    facts = compute_career_facts(context)

    assert facts["companies"] == ["Acme SaaS"]
    assert facts["total_experience_months"] == 24
    assert facts["average_tenure_months"] == 0
    assert facts["completed_company_count"] == 0


def test_query_tools_cover_geography_function_industry_and_competitors(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 2, tzinfo=tz)

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    profile = {
        "id": 99,
        "name": "Query Candidate",
        "city": "Bengaluru",
        "raw_fields": {
            "enrichment": {
                "roles": [
                    {
                        "company": "MoEngage",
                        "title": "Channel Sales Manager APAC",
                        "start_date": "2020-01-01",
                        "end_date": "",
                        "duration_months": 78,
                        "function": "Channel Sales",
                        "industry": "Customer Engagement",
                        "product_service": "Customer engagement platform",
                        "location": "Bengaluru, India",
                        "details": "Built partner and reseller channel sales across APAC.",
                    },
                    {
                        "company": "WebEngage",
                        "title": "Sales Development Representative",
                        "start_date": "2014-01-01",
                        "end_date": "2020-01-01",
                        "duration_months": 73,
                        "function": "Sales Development",
                        "industry": "Customer Engagement",
                        "product_service": "Lifecycle engagement CRM",
                        "location": "Singapore",
                    },
                    {
                        "company": "Braze",
                        "title": "Inside Sales Representative and Account Development",
                        "duration_months": 72,
                        "function": "Inside Sales Account Development",
                        "industry": "Customer Engagement",
                        "product_service": "Marketing automation",
                        "location": "APAC",
                    },
                ],
                "profile_claims": {
                    "geographies": ["APAC"],
                    "segments": ["Enterprise"],
                    "functions": [{"function": "Channel Sales", "reason": "profile"}],
                },
            }
        },
    }
    context = build_candidate_context(profile)
    facts = compute_career_facts(context)

    assert facts["career_city_count"] == 2
    assert facts["average_tenure_months"] > 0
    assert facts["current_job_months"] >= 77

    channel_prompt = "Candidates who are working for Clevertap competitors and has 5 years in channel sales"
    channel_tools = run_candidate_query_tools(
        channel_prompt,
        context,
        facts,
        build_query_plan(
            channel_prompt,
            context,
            [{"key": "result", "label": "Result", "primary": True}],
            classify_ai_column_prompt(channel_prompt),
        ),
    )
    assert channel_tools["competitor_match"]["CleverTap"]["status"] == "Yes"
    assert channel_tools["functional_experience"]["Channel Sales"]["months"] >= 60

    apac_prompt = "Candidates who are working for Clevertap competitors and has 5 years in APAC market"
    apac_tools = run_candidate_query_tools(apac_prompt, context, facts)
    assert apac_tools["geography_experience"]["APAC"]["months"] >= 60

    singapore_tools = run_candidate_query_tools("Candidates who have Singapore experience", context, facts)
    assert singapore_tools["geography_experience"]["Singapore"]["profile_claim_match"] is True

    india_tools = run_candidate_query_tools("Candidates who have India experience", context, facts)
    assert india_tools["geography_experience"]["India"]["months"] >= 60

    sales_dev_tools = run_candidate_query_tools(
        "Candidates with 6 years of sales development experience and have worked in customer engagement industry",
        context,
        facts,
    )
    assert sales_dev_tools["functional_experience"]["Sales Development"]["months"] >= 72
    assert sales_dev_tools["industry_experience"]["Customer Engagement"]["months"] >= 72

    inside_tools = run_candidate_query_tools(
        "Candidates with 6 years of inside sales experience and have worked in customer engagement industry",
        context,
        facts,
    )
    assert inside_tools["functional_experience"]["Inside Sales"]["months"] >= 72

    account_dev_tools = run_candidate_query_tools(
        "Candidates with 5 years of account development experience and have worked in customer engagement industry",
        context,
        facts,
    )
    assert account_dev_tools["functional_experience"]["Account Development"]["months"] >= 60


def test_run_ai_task_sends_full_row_context_even_without_tokens(monkeypatch):
    captured = {}

    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        captured["user_prompt"] = user_prompt
        captured["use_web"] = use_web
        return {
            "outputs": {"result": "Works from the second role"},
            "reasoning": "Used full row context.",
            "confidence": "high",
            "steps": ["Read all row fields"],
            "sources": [],
        }

    profile = {
        "id": 1,
        "name": "Deepak Basavaraj",
        "roles": [
            {"title": "Account Director", "company": "Exotel"},
            {"title": "Regional Director", "company": "LaterCo"},
        ],
        "raw_fields": {"Spreadsheet Note": "Imported context"},
    }
    context = build_candidate_context(profile)

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Answer like a Clay chatbot using whatever row data is useful.",
        mode="content",
        output_schema=[{"key": "result", "label": "Result", "primary": True}],
        context=context,
    )

    assert result["primary_output"] == "Works from the second role"
    assert captured["use_web"] is False
    assert "Full row context JSON" in captured["user_prompt"]
    assert "row.roles.1.title" in captured["user_prompt"]
    assert "Regional Director" in captured["user_prompt"]
    assert "row.raw_fields.Spreadsheet Note" in captured["user_prompt"]


def test_full_row_context_handles_array_like_values():
    class ArrayLike:
        size = 2

        def tolist(self):
            return ["sales", "ops"]

        def __eq__(self, _other):
            raise ValueError("array truth value is ambiguous")

    profile = {
        "id": 42,
        "name": "Array Row",
        "embedding": ArrayLike(),
        "raw_fields": {"tags": ArrayLike()},
    }

    flattened = flatten_profile_context(profile)
    context = build_candidate_context(profile)

    assert flattened["row.embedding"] == "sales, ops"
    assert context["row.raw_fields.tags"] == "sales, ops"


def test_build_browse_candidate_rows_matches_filtered_scope(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Deepak Basavaraj",
            "first_name": "Deepak",
            "last_name": "Basavaraj",
            "linkedin": "https://linkedin.com/in/deepak",
            "city": "Bengaluru",
            "location": "Bengaluru, Karnataka, India",
            "headline": "Account Director",
            "status": "Shortlisted",
            "created_by": "Recruiter A",
            "owner_user_id": 7,
            "roles": [{"title": "Account Director", "company": "Exotel", "company_details": {"product_service": "CCaaS"}}],
            "raw_fields": {},
        },
        2: {
            "id": 2,
            "name": "Aadarsh Goyal",
            "first_name": "Aadarsh",
            "last_name": "Goyal",
            "linkedin": "https://linkedin.com/in/aadarsh",
            "city": "Delhi",
            "location": "Delhi, India",
            "headline": "Sales Manager",
            "status": "To be started",
            "created_by": "Recruiter A",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme", "company_details": {"product_service": "Payments"}}],
            "raw_fields": {},
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)

    result = asyncio.run(
        browse.build_browse_candidate_rows(
            current_user=_user(),
            city="bengaluru",
            status="Shortlisted",
            sort_by="name",
            sort_dir="asc",
        )
    )

    assert [row["id"] for row in result["candidates"]] == [1]
    assert result["status_counts"]["Shortlisted"] == 1


def test_browse_summary_uses_unfiltered_scope_counts(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Deepak Basavaraj",
            "city": "Bengaluru",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Aadarsh Goyal",
            "city": "Delhi",
            "status": "To be started",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
        3: {
            "id": 3,
            "name": "Other Recruiter",
            "city": "Mumbai",
            "status": "Shortlisted",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)

    filtered = asyncio.run(
        browse.build_browse_candidate_rows(
            current_user=_user(),
            city="bengaluru",
            status="Shortlisted",
        )
    )
    summary = asyncio.run(browse.browse_summary(current_user=_user()))

    assert len(filtered["candidates"]) == 1
    assert summary["total"] == 2
    assert summary["status_counts"]["Shortlisted"] == 1
    assert summary["status_counts"]["To be started"] == 1


def test_browse_summary_initializes_empty_profile_cache(monkeypatch):
    profiles = {}
    calls = []

    def fake_initialize_cache():
        calls.append("init")
        profiles[1] = {
            "id": 1,
            "name": "Cold Worker Candidate",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        }

    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    monkeypatch.setattr(browse, "initialize_cache", fake_initialize_cache)
    browse._browse_cache.clear()

    summary = asyncio.run(
        browse.browse_summary(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
        )
    )

    assert calls == ["init"]
    assert summary["total"] == 1
    assert summary["status_counts"] == {"Shortlisted": 1}


def test_browse_summary_returns_503_when_active_db_cache_stays_empty(monkeypatch):
    calls = []

    def fake_initialize_cache():
        calls.append("init")

    monkeypatch.setattr(browse, "PROFILES_BY_ID", {})
    monkeypatch.setattr(browse, "is_cache_initialized", lambda: True)
    monkeypatch.setattr(browse, "initialize_cache", fake_initialize_cache)
    monkeypatch.setattr(browse, "count_active_candidates_from_db", lambda: 4174)
    browse._browse_cache.clear()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            browse.browse_summary(
                current_user=_user(role="admin"),
                view_scope="master",
                recruiter_filter_id=None,
            )
        )

    assert calls == ["init"]
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "profile_cache_unavailable"


def test_browse_meta_initializes_empty_profile_cache(monkeypatch):
    profiles = {}
    calls = []

    def fake_initialize_cache():
        calls.append("init")
        profiles[1] = {
            "id": 1,
            "name": "Cold Meta Candidate",
            "city": "Bengaluru",
            "status": "To be started",
            "owner_user_id": None,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        }

    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    monkeypatch.setattr(browse, "initialize_cache", fake_initialize_cache)
    browse._browse_cache.clear()

    meta = asyncio.run(
        browse.browse_metadata(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            role_id=None,
        )
    )

    assert calls == ["init"]
    assert "Exotel" in meta["companies"]
    assert "Bengaluru" in meta["cities"]


def test_browse_candidate_ids_preserves_scope_and_order(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Deepak Basavaraj",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Aadarsh Goyal",
            "status": "To be started",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
        3: {
            "id": 3,
            "name": "Hidden Recruiter",
            "status": "Shortlisted",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(),
            candidate_ids="2,1,3",
            page=1,
            page_size=25,
        )
    )

    assert [row["id"] for row in result["candidates"]] == [2, 1]
    assert result["total"] == 2
    assert result["status_counts"]["Shortlisted"] == 1


def test_admin_master_browse_includes_master_and_recruiter_rows(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Master Row",
            "status": "To be started",
            "owner_user_id": None,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Recruiter A Row",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
        3: {
            "id": 3,
            "name": "Recruiter B Row",
            "status": "Rejected",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            page=1,
            page_size=25,
        )
    )

    assert result["total"] == 3
    assert {row["id"] for row in result["candidates"]} == {1, 2, 3}
    assert result["status_counts"]["Shortlisted"] == 1
    assert result["status_counts"]["Rejected"] == 1


def test_admin_recruiter_scope_is_strict_to_selected_recruiter(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Master Row",
            "status": "Shortlisted",
            "owner_user_id": None,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Recruiter A Row",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
        3: {
            "id": 3,
            "name": "Recruiter B Row",
            "status": "To be started",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="recruiter_pools",
            recruiter_filter_id=7,
            page=1,
            page_size=25,
        )
    )

    assert result["total"] == 1
    assert [row["id"] for row in result["candidates"]] == [2]
    assert result["status_counts"] == {"Shortlisted": 1}


def test_admin_recruiter_scope_without_recruiter_id_is_400(monkeypatch):
    calls = []

    def fake_initialize_cache():
        calls.append("init")

    monkeypatch.setattr(browse, "PROFILES_BY_ID", {})
    monkeypatch.setattr(browse, "initialize_cache", fake_initialize_cache)
    browse._browse_cache.clear()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            browse.browse_candidates(
                current_user=_user(role="admin"),
                view_scope="recruiter_pools",
                recruiter_filter_id=None,
                page=1,
                page_size=25,
            )
        )

    assert exc.value.status_code == 400
    assert calls == []


def test_admin_master_role_filter_accepts_recruiter_owned_role(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Master Row",
            "status": "To be started",
            "owner_user_id": None,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Recruiter Role Row",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
    }

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            pass

        def fetchone(self):
            return (7,)

        def fetchall(self):
            return [(2,)]

    class Conn:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    monkeypatch.setattr(browse, "get_db_connection", lambda **_kwargs: Conn())
    monkeypatch.setattr(browse, "return_db_connection", lambda _conn: None)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            role_id=58,
            page=1,
            page_size=25,
        )
    )

    assert result["total"] == 1
    assert [row["id"] for row in result["candidates"]] == [2]


def test_stale_role_filter_returns_empty_result_not_http_error(monkeypatch):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            pass

        def fetchone(self):
            return None

    class Conn:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(browse, "PROFILES_BY_ID", {
        1: {
            "id": 1,
            "name": "Visible Candidate",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
    })
    monkeypatch.setattr(browse, "get_db_connection", lambda **_kwargs: Conn())
    monkeypatch.setattr(browse, "return_db_connection", lambda _conn: None)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            role_id=999,
            page=1,
            page_size=25,
        )
    )

    assert result["total"] == 0
    assert result["candidates"] == []


def test_browse_status_counts_are_before_status_filter_and_total_is_paginated_scope(monkeypatch):
    profiles = {
        1: {
            "id": 1,
            "name": "Bengaluru Shortlisted A",
            "city": "Bengaluru",
            "status": "Shortlisted",
            "owner_user_id": None,
            "roles": [{"title": "Account Director", "company": "Exotel"}],
        },
        2: {
            "id": 2,
            "name": "Bengaluru Shortlisted B",
            "city": "Bengaluru",
            "status": "Shortlisted",
            "owner_user_id": 7,
            "roles": [{"title": "Sales Manager", "company": "Acme"}],
        },
        3: {
            "id": 3,
            "name": "Bengaluru Rejected",
            "city": "Bengaluru",
            "status": "Rejected",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
        4: {
            "id": 4,
            "name": "Delhi Shortlisted",
            "city": "Delhi",
            "status": "Shortlisted",
            "owner_user_id": 9,
            "roles": [{"title": "AE", "company": "Other"}],
        },
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            city="bengaluru",
            status="Shortlisted",
            page=1,
            page_size=1,
            sort_by="name",
            sort_dir="asc",
        )
    )

    assert result["total"] == 2
    assert result["total_pages"] == 2
    assert len(result["candidates"]) == 1
    assert result["status_counts"] == {"Shortlisted": 2, "Rejected": 1}


def test_admin_master_totals_are_not_capped_at_5000(monkeypatch):
    profiles = {
        i: {
            "id": i,
            "name": f"Candidate {i:05d}",
            "status": "Shortlisted" if i % 2 == 0 else "To be started",
            "owner_user_id": None if i % 3 == 0 else 7,
            "roles": [{"title": "Account Executive", "company": "Acme"}],
        }
        for i in range(1, 5006)
    }
    monkeypatch.setattr(browse, "PROFILES_BY_ID", profiles)
    browse._browse_cache.clear()

    result = asyncio.run(
        browse.browse_candidates(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
            page=2,
            page_size=25,
            sort_by="name",
            sort_dir="asc",
        )
    )
    summary = asyncio.run(
        browse.browse_summary(
            current_user=_user(role="admin"),
            view_scope="master",
            recruiter_filter_id=None,
        )
    )

    assert result["total"] == 5005
    assert result["total_pages"] == 201
    assert len(result["candidates"]) == 25
    assert summary["total"] == 5005


def test_generate_config_defaults_to_auto_without_explicit_web(monkeypatch):
    def fake_openai(*args, **kwargs):
        return {
            "name": "Extract Current Location from LinkedIn",
            "prompt_template": "Analyze {candidate.full_name} at {candidate.linkedin} and find the current location.",
            "mode": "web_research",
            "output_schema": [{"key": "current_location", "label": "Current Location", "primary": True}],
            "required_fields": ["candidate.full_name", "candidate.linkedin"],
        }

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._generate_config(
        "Go through {candidate.linkedin} and find current location.",
        field_catalog=[],
    )

    assert result["mode"] == "auto"
    assert "candidate.full_name" in result["required_fields"]


def test_generate_config_strips_required_fields_not_in_prompt_template(monkeypatch):
    def fake_openai(*args, **kwargs):
        return {
            "name": "Extract Current Location from LinkedIn",
            "prompt_template": "Check {candidate.linkedin} for {candidate.full_name}.",
            "mode": "auto",
            "output_schema": [{"key": "current_location", "label": "Current Location", "primary": True}],
            "required_fields": [
                "candidate.linkedin",
                "candidate.full_name",
                "candidate.email",
                "candidate.phone",
            ],
        }

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._generate_config("Find location from LinkedIn", field_catalog=[])
    assert set(result["required_fields"]) == {"candidate.linkedin", "candidate.full_name"}


def test_generate_config_prefer_web_search_keeps_web_research(monkeypatch):
    def fake_openai(*args, **kwargs):
        return {
            "name": "Competitors",
            "prompt_template": "List competitors for {role.current_company}.",
            "mode": "web_research",
            "output_schema": [{"key": "competitor", "label": "Competitor", "primary": True}],
            "required_fields": ["role.current_company"],
        }

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._generate_config(
        "Find competitors of {role.current_company}.",
        field_catalog=[],
        prefer_web_search=True,
    )

    assert result["mode"] == "web_research"


def test_web_json_call_uses_ga_search_and_stamps_freshness(monkeypatch):
    captured = {}

    class FakeUsage:
        input_tokens = 2000
        output_tokens = 1000
        total_tokens = 3000

    class FakeResponse:
        output_text = '{"outputs":{"result":"fresh"},"confidence":"high"}'
        usage = FakeUsage()

        def model_dump(self):
            return {
                "output": [
                    {
                        "content": [
                            {
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/news",
                                        "title": "Example News",
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_columns, "_openai_client", FakeClient())
    monkeypatch.setattr(ai_columns, "_utc_now_iso", lambda: "2026-05-23T10:30:00Z")

    result = ai_columns._call_openai_for_json("system", "user", use_web=True)

    assert captured["model"] == "gpt-4o-mini"
    assert captured["tools"][0]["type"] == "web_search"
    assert captured["tools"][0]["search_context_size"] == "high"
    assert "Today is 2026-05-23" in captured["input"]
    assert result["searched_at"] == "2026-05-23T10:30:00Z"
    assert result["freshness_date"] == "2026-05-23"
    assert result["web_search_tool"] == "web_search"
    assert result["sources"][0]["url"] == "https://example.com/news"
    assert result["openai_usage"]["payload_type"] == "responses_api_usage"
    assert result["ai_credits"]["input_tokens"] == 2000
    assert result["ai_credits"]["output_tokens"] == 1000
    assert result["ai_credits"]["usd"] == pytest.approx(0.0009)


def test_chat_json_call_records_dynamic_usage_payload(monkeypatch):
    class FakeUsage:
        prompt_tokens = 1000
        completion_tokens = 500
        total_tokens = 1500

    class FakeMessage:
        content = '{"outputs":{"result":"ok"},"confidence":"high"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(ai_columns, "_AI_COLUMN_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(ai_columns, "_openai_client", FakeClient())

    result = ai_columns._call_openai_for_json("system", "user", use_web=False)

    assert result["outputs"]["result"] == "ok"
    assert result["openai_usage"]["payload_type"] == "chat_completions_usage"
    assert result["ai_credits"]["input_tokens"] == 1000
    assert result["ai_credits"]["output_tokens"] == 500
    assert result["ai_credits"]["usd"] == pytest.approx(0.00045)


def test_map_raw_outputs_to_schema_keys_aligns_label_style_keys():
    raw = {"Competitor Name": "Acme Corp", "competitor_industry": "Software"}
    got = map_raw_outputs_to_schema_keys(raw, ["competitor_name", "competitor_industry"])
    assert got["competitor_name"] == "Acme Corp"
    assert got["competitor_industry"] == "Software"


def test_map_raw_outputs_to_schema_keys_serializes_nested_values():
    raw = {"result": {"answer": "Yes", "months": 9}}
    got = map_raw_outputs_to_schema_keys(raw, ["result"])
    assert got["result"] == '{"answer": "Yes", "months": 9}'


def test_model_to_dict_supports_pydantic_v1_style_objects():
    class V1Style:
        def dict(self):
            return {"selection_mode": "selected_ids"}

    assert ai_columns._model_to_dict(V1Style()) == {"selection_mode": "selected_ids"}


def test_run_ai_task_fills_all_outputs_when_model_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", object())

    def fake_openai(*args, **kwargs):
        return {"outputs": {}, "reasoning": "nothing parsed"}

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Test {candidate.full_name}",
        mode="content",
        output_schema=[
            {"key": "a", "label": "A", "primary": True},
            {"key": "b", "label": "B", "primary": False},
        ],
        context={"candidate.full_name": "Jane"},
    )
    assert result["outputs"]["a"] == result["outputs"]["b"]
    assert result["outputs"]["a"] == "nothing parsed"


def test_run_ai_task_uses_top_level_model_answer_when_outputs_missing(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", object())

    def fake_openai(*args, **kwargs):
        return {"answer": "Stable: current role is 9 months.", "confidence": "medium"}

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Summarize stability for {candidate.full_name}",
        mode="content",
        output_schema=[{"key": "result", "label": "Result", "primary": True}],
        context={"candidate.full_name": "Aarthi Nambiar"},
    )

    assert result["primary_output"] == "Stable: current role is 9 months."
    assert "No structured response" not in result["primary_output"]


def test_run_ai_task_empty_model_output_returns_no_not_generic_string(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", object())

    def fake_openai(*args, **kwargs):
        return {}

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Find unsupported info for {candidate.full_name}",
        mode="content",
        output_schema=[{"key": "result", "label": "Result", "primary": True}],
        context={"candidate.full_name": "Empty Candidate"},
    )

    assert result["primary_output"] == "No"
    assert "No structured response" not in result["primary_output"]


def test_run_ai_task_promotes_first_non_empty_output_when_primary_is_blank(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", object())

    def fake_openai(*args, **kwargs):
        return {
            "outputs": {"status": "", "reason": "Clear: no contradiction found in the row."},
            "reasoning": "The status field was omitted but the reason contains the answer.",
            "confidence": "medium",
        }

    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Find contradictions for {candidate.full_name}.",
        mode="content",
        output_schema=[
            {"key": "status", "label": "Status", "type": "text", "primary": True},
            {"key": "reason", "label": "Reason", "type": "text"},
        ],
        context={"candidate.full_name": "Blank Primary Candidate"},
    )

    assert result["primary_output"] == "Clear: no contradiction found in the row."
    assert result["details"]["response"] == "Clear: no contradiction found in the row."


def test_career_facts_datadog_october_2025_current_role_is_nine_months(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 5, tzinfo=tz)

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    context = build_candidate_context(
        {
            "id": 104,
            "name": "Aarthi Nambiar",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Datadog",
                            "title": "Enterprise Sales Development Representative",
                            "start_date": "10-2025",
                            "end_date": "",
                            "duration_months": 13,
                        }
                    ]
                }
            },
        }
    )

    facts = compute_career_facts(context)

    assert facts["current_company"] == "Datadog"
    assert facts["current_job_months"] == 9
    assert facts["current_company_tenure_months"] == 9
    assert facts["company_tenures"] == [
        {
            "company": "Datadog",
            "months": 9,
            "years": 0.75,
            "titles": ["Enterprise Sales Development Representative"],
            "is_current_company": True,
        }
    ]


def test_career_context_uses_raw_current_company_details_when_structured_unknown():
    context = build_candidate_context(
        {
            "id": 105,
            "name": "Raw Details Candidate",
            "headline": "Sales Development Manager",
            "raw_fields": {
                "experiences/0/companyName": "Parkar",
                "experiences/0/title": "Manager - Data Solutions",
                "experiences/0/companyIndustry": "Information Technology And Services",
                "experiences/0/companyWebsite": "parkardigital.com",
                "experiences/0/companySize": "201-500",
            },
            "roles": [
                {
                    "company": "Parkar",
                    "title": "Manager - Data Solutions",
                    "start_date": "2025-01-01",
                    "end_date": "Present",
                    "company_details": {
                        "business_model": "Unknown",
                        "product_service": "Unknown",
                        "customer_segment": "Mid-Market",
                        "industry": "Unknown",
                    },
                }
            ],
        }
    )

    facts = compute_career_facts(context)
    pack = build_candidate_context_pack(context, facts)
    current_role = pack["current_role"]

    assert current_role["source_industry"] == "Information Technology And Services"
    assert current_role["source_website"] == "parkardigital.com"
    assert current_role["source_company_size"] == "201-500"
    assert "current role industry: Information Technology And Services" in facts["current_company_enterprise_saas_evidence"]
    assert "current role company website: parkardigital.com" in facts["current_company_enterprise_saas_evidence"]


def test_data_quality_with_unknown_current_company_details_routes_to_web():
    context = build_candidate_context(
        {
            "id": 106,
            "name": "Web Detail Candidate",
            "headline": "Sales Development Manager",
            "raw_fields": {
                "experiences/0/companyName": "Parkar",
                "experiences/0/title": "Manager - Data Solutions",
                "experiences/0/companyIndustry": "Information Technology And Services",
                "experiences/0/companyWebsite": "parkardigital.com",
            },
            "roles": [
                {
                    "company": "Parkar",
                    "title": "Manager - Data Solutions",
                    "start_date": "2025-01-01",
                    "end_date": "Present",
                    "company_details": {
                        "business_model": "Unknown",
                        "product_service": "Unknown",
                        "customer_segment": "Mid-Market",
                        "industry": "Unknown",
                    },
                }
            ],
        }
    )
    prompt = (
        "Score this candidate row data quality for recruiting decisions from 1 to 5. "
        "Penalize missing current company details, product/service, or business model."
    )

    routing = classify_ai_column_prompt(prompt)
    plan = build_query_plan(
        prompt,
        context,
        [{"key": "quality_score", "label": "Quality Score", "primary": True}],
        routing,
    )

    assert routing["data_source"] == "hybrid"
    assert plan["web_needed"] is True
    assert "company_verification" in plan["tool_calls"]


def test_run_ai_task_uses_deterministic_tenure_when_model_outputs_are_empty(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 1, tzinfo=tz)

    def fake_openai(*args, **kwargs):
        return {
            "outputs": {},
            "reasoning": (
                "The completed companies total 60 months across two unique companies. "
                "The current role is 11 months."
            ),
            "confidence": "high",
            "sources": [],
        }

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    context = build_candidate_context(
        {
            "id": 102,
            "name": "Structured Fallback Candidate",
            "linkedin": "https://linkedin.com/in/structured-fallback",
            "roles": [
                {"title": "Current Role", "company": "C", "start_date": "2025-08-01", "end_date": "Present"},
                {"title": "Role A", "company": "A", "start_date": "2020-01-01", "end_date": "2021-12-01"},
                {"title": "Role B", "company": "B", "start_date": "2022-01-01", "end_date": "2024-12-01"},
            ],
        }
    )

    result = ai_columns._run_ai_task(
        prompt_template=(
            "Calculate the average tenure of the candidate ({Linkedin Profile}). "
            "Also give the time spent by the candidate in the current job."
        ),
        mode="web_research",
        output_schema=[
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "tenure_reasoning", "label": "Tenure Reasoning"},
        ],
        context=context,
    )

    assert result["primary_output"] == "30"
    assert result["outputs"]["average_tenure_months"] == "30"
    assert result["outputs"]["current_job_months"] == "11"
    assert "No structured response" not in result["outputs"]["tenure_reasoning"]
    assert result["details"]["data_source"] == "row"
    assert result["details"]["source_verification_status"] == "row_context"
    assert result["details"]["verification_status"] == "passed"
    assert result["details"]["ai_credits_display"] == "$0.000000"
    assert result["details"]["ai_credits"]["usd"] == 0.0


def test_run_ai_task_old_stability_prompt_stays_deterministic(monkeypatch):
    class FrozenDateTime(ai_columns_service.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 1, tzinfo=tz)

    def fail_openai(*args, **kwargs):
        raise AssertionError("stability tenure prompt should be answered deterministically")

    monkeypatch.setattr(ai_columns_service, "datetime", FrozenDateTime)
    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fail_openai)
    context = build_candidate_context(
        {
            "id": 103,
            "name": "Old Stability Prompt Candidate",
            "linkedin": "https://linkedin.com/in/old-stability",
            "roles": [
                {"title": "Current Role", "company": "C", "start_date": "2025-08-01", "end_date": "Present"},
                {"title": "Role A", "company": "A", "start_date": "2020-01-01", "end_date": "2021-12-01"},
                {"title": "Role B", "company": "B", "start_date": "2022-01-01", "end_date": "2024-12-01"},
            ],
        }
    )

    result = ai_columns._run_ai_task(
        prompt_template=(
            "Calculate the average tenure of the candidate ({Linkedin Profile}). "
            "Average Tenure = total years of work experience / number of unique companies. "
            "Count different roles in the same company as one job. "
            "Do not count community memberships, such as RevGenius. "
            "Give the output in months. Also give the time spent by the candidate in the current job."
        ),
        mode="auto",
        output_schema=[
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "tenure_reasoning", "label": "Tenure Reasoning"},
        ],
        context=context,
    )

    assert result["outputs"]["average_tenure_months"] == "30"
    assert result["outputs"]["current_job_months"] == "11"
    assert result["details"]["query_plan"]["web_needed"] is False
    assert result["details"]["data_source"] == "row"
    assert result["details"]["verification_status"] == "passed"
    assert result["details"]["ai_credits_display"] == "$0.000000"

    shorthand_result = ai_columns._run_ai_task(
        prompt_template="Stability: compute avg tenure completed roles and current role tenure for {candidate.full_name}.",
        mode="auto",
        output_schema=[
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
            {"key": "tenure_reasoning", "label": "Tenure Reasoning"},
        ],
        context=context,
    )

    assert shorthand_result["outputs"]["average_tenure_months"] == "30"
    assert shorthand_result["outputs"]["current_job_months"] == "11"
    assert shorthand_result["details"]["query_plan"]["web_needed"] is False


def test_run_ai_task_auto_uses_row_context_before_web(monkeypatch):
    calls = []

    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        calls.append(use_web)
        if use_web:
            raise AssertionError("web fallback should not be used when row context is enough")
        return {
            "outputs": {"current_location": "Bengaluru"},
            "reasoning": "The row already contains the current location.",
            "confidence": "high",
            "steps": ["Read the row context"],
            "sources": [],
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Find the current location for {candidate.full_name}. Current row location: {candidate.location}",
        mode="auto",
        output_schema=[{"key": "current_location", "label": "Current Location", "primary": True}],
        context={"candidate.full_name": "Deepak Basavaraj", "candidate.location": "Bengaluru"},
    )

    assert result["primary_output"] == "Bengaluru"
    assert calls == [False]


def test_run_ai_task_records_ai_credits_from_openai_usage_payload(monkeypatch):
    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        return {
            "outputs": {"result": "Yes"},
            "reasoning": "The row supports the answer.",
            "confidence": "high",
            "openai_usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            "model": "gpt-4o-mini",
        }

    monkeypatch.setattr(ai_columns, "_AI_COLUMN_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Does this person have location data? {candidate.location}",
        mode="content",
        output_schema=[{"key": "result", "label": "Result", "primary": True}],
        context={"candidate.location": "Bengaluru"},
    )

    assert result["primary_output"] == "Yes"
    assert result["details"]["ai_credits"]["usage_payload_type"] == "chat_completions_usage"
    assert result["details"]["ai_credits"]["input_tokens"] == 1000
    assert result["details"]["ai_credits"]["output_tokens"] == 500
    assert result["details"]["ai_credits"]["usd"] == pytest.approx(0.00045)
    assert result["details"]["ai_credits_display"] == "$0.000450"


def test_run_ai_task_auto_computes_tenure_without_openai(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", None)
    context = build_candidate_context(
        {
            "id": 1,
            "name": "Tenure Candidate",
            "roles": [
                {"title": "Account Executive", "company": "Acme", "start_date": "2021-01", "end_date": "Present"},
                {"title": "Senior Account Executive", "company": "Acme", "start_date": "2019-01", "end_date": "2020-12"},
                {"title": "Sales Development Representative", "company": "Beta", "start_date": "2017-01", "end_date": "2018-12"},
                {"title": "Member", "company": "RevGenius", "start_date": "2022", "end_date": "Present"},
            ],
        }
    )

    result = ai_columns._run_ai_task(
        prompt_template="Calculate the average tenure and current job tenure in months.",
        mode="auto",
        output_schema=[
            {"key": "average_tenure_months", "label": "Average Tenure Months", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months"},
        ],
        context=context,
    )

    assert result["details"]["data_source"] == "row"
    assert result["outputs"]["average_tenure_months"].isdigit()
    assert result["outputs"]["current_job_months"].isdigit()
    assert result["details"]["query_plan"]["tool_calls"]
    assert result["details"]["tool_results"]["career_metrics"]["average_tenure_months"]
    assert result["details"]["verification_status"] == "passed"


def test_run_ai_task_uses_enriched_tools_without_static_sql_for_complex_query(monkeypatch):
    calls = []

    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        calls.append(use_web)
        assert "Candidate context pack JSON" in user_prompt
        assert "Deterministic tool results JSON" in user_prompt
        return {
            "outputs": {
                "result": "Yes",
                "reasoning": "Tool results show SMB, EMEA and Hunting experience.",
            },
            "query_plan": {
                "intent": "Check segment, geography and function experience.",
                "tool_calls": ["segment_experience", "geography_experience", "functional_experience"],
                "web_needed": False,
            },
            "reasoning": "Used enriched profile tool results.",
            "confidence": "high",
            "steps": ["Read enriched roles", "Used deterministic tools"],
            "sources": [],
        }

    context = build_candidate_context(
        {
            "id": 4,
            "name": "Tool Candidate",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Acme SaaS",
                            "title": "Enterprise Account Executive",
                            "start_date": "2020-01-01",
                            "end_date": "2022-12-01",
                            "duration_months": 36,
                            "function": "Hunting",
                            "product_service": "SaaS",
                            "industry": "SaaS",
                            "customer_segment": ["SMB"],
                            "details": "New business in EMEA.",
                        }
                    ]
                }
            },
        }
    )
    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)

    result = ai_columns._run_ai_task(
        prompt_template="Has 3+ years selling to SMB in EMEA with hunting experience?",
        mode="auto",
        output_schema=[
            {"key": "result", "label": "Result", "primary": True},
            {"key": "reasoning", "label": "Reasoning"},
        ],
        context=context,
    )

    assert calls == [False]
    assert result["primary_output"] == "Yes"
    assert result["details"]["query_plan"]["web_needed"] is False
    assert result["details"]["tool_results"]["segment_experience"]["SMB"]["months"] == 36
    assert result["details"]["verification_status"] == "passed"


def test_run_ai_task_auto_forces_web_for_linkedin_activity(monkeypatch):
    calls = []

    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        calls.append(use_web)
        return {
            "outputs": {"posted_last_30_days": "Yes", "activity_reasoning": "Public post found."},
            "reasoning": "Used public LinkedIn evidence.",
            "confidence": "high",
            "steps": ["Searched web"],
            "sources": [{"title": "LinkedIn post", "url": "https://www.linkedin.com/in/example/recent-activity/all/"}],
            "searched_at": "2026-05-23T10:00:00Z",
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Has the candidate posted content on LinkedIn in the last 30 days?",
        mode="auto",
        output_schema=[{"key": "posted_last_30_days", "label": "Posted Last 30 Days", "primary": True}],
        context={"candidate.linkedin": "https://linkedin.com/in/example"},
    )

    assert calls == [True]
    assert result["details"]["data_source"] == "web"
    assert result["details"]["web_required_reason"] == "public_linkedin_recent_activity"
    assert result["details"]["source_verification_status"] == "verified"


def test_run_ai_task_linkedin_activity_without_sources_is_unverified(monkeypatch):
    def fake_openai(*args, **kwargs):
        return {
            "outputs": {"posted_last_30_days": "Yes", "activity_reasoning": "No public source."},
            "confidence": "medium",
            "sources": [],
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Has the candidate posted content on LinkedIn in the last 30 days?",
        mode="auto",
        output_schema=[{"key": "posted_last_30_days", "label": "Posted Last 30 Days", "primary": True}],
        context={"candidate.linkedin": "https://linkedin.com/in/example"},
    )

    assert result["primary_output"] == "Not publicly verifiable"
    assert result["details"]["source_verification_status"] == "not_publicly_verifiable"


def test_run_ai_task_linkedin_activity_rejects_same_name_profile_sources(monkeypatch):
    def fake_openai(*args, **kwargs):
        return {
            "outputs": {"posted_last_30_days": "Yes", "activity_reasoning": "Same-name source."},
            "confidence": "medium",
            "sources": [{"url": "https://www.linkedin.com/in/different-person"}],
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Has the candidate posted content on LinkedIn in the last 30 days? Candidate: {Linkedin Profile}",
        mode="auto",
        output_schema=[{"key": "posted_last_30_days", "label": "Posted Last 30 Days", "primary": True}],
        context={"Linkedin Profile": "https://www.linkedin.com/in/example"},
    )

    assert result["primary_output"] == "Not publicly verifiable"
    assert result["details"]["sources"] == []
    assert result["details"]["source_verification_status"] == "not_publicly_verifiable"


def test_daily_refresh_group_query_targets_existing_stale_web_cells(monkeypatch):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_cur
    cm.__exit__.return_value = False
    mock_conn.cursor.return_value = cm
    mock_cur.fetchall.return_value = [
        (9, 7, "recruiter@example.com", "Recruiter", "recruiter", [101, 102])
    ]

    class ConnCM:
        def __enter__(self):
            return mock_conn

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ai_columns, "get_db_connection_context", lambda **kwargs: ConnCM())

    groups = ai_columns._fetch_daily_refresh_groups(max_cells=10)
    sql = mock_cur.execute.call_args[0][0]

    assert "c.status = 'completed'" in sql
    assert "d.mode = 'web_research'" in sql
    assert "details->>'data_source'" in sql
    assert "COALESCE(d.is_archived, FALSE) = FALSE" in sql
    assert groups[0]["candidate_ids"] == [101, 102]
    assert groups[0]["user"].id == 7


def test_fetch_visible_definitions_includes_compact_ai_credits(monkeypatch):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_cur
    cm.__exit__.return_value = False
    mock_conn.cursor.return_value = cm
    mock_cur.fetchall.side_effect = [
        [],
        [
            (
                5,
                "Fit",
                "fit",
                7,
                "master",
                None,
                "Prompt",
                "content",
                [{"key": "result", "label": "Result", "primary": True}],
                [],
                {},
                {},
                None,
                None,
                None,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
            )
        ],
        [
            (
                5,
                123,
                "Yes",
                {"result": "Yes"},
                "completed",
                "",
                None,
                None,
                {"ai_credits": {"usd": 0.00045, "display": "$0.000450"}},
            )
        ],
    ]

    class ConnCM:
        def __enter__(self):
            return mock_conn

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ai_columns, "get_db_connection_context", lambda **kwargs: ConnCM())

    definitions = ai_columns._fetch_visible_definitions(
        _user(role="admin"),
        view_scope="master",
        recruiter_filter_id=None,
        candidate_ids=[123],
    )

    cell = definitions[0]["cells_by_candidate"][123]
    assert cell["ai_credits_usd"] == 0.00045
    assert cell["ai_credits_display"] == "$0.000450"


def test_run_ai_task_auto_falls_back_to_web_when_row_context_is_insufficient(monkeypatch):
    calls = []

    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        calls.append(use_web)
        if not use_web:
            return {
                "outputs": {"competitors": ""},
                "reasoning": "The row does not contain competitor data.",
                "confidence": "low",
                "steps": ["Checked the row context"],
                "sources": [],
            }
        return {
            "outputs": {"competitors": "Acme, BetaCo, GammaInc"},
            "reasoning": "Used the web to find competitors.",
            "confidence": "medium",
            "steps": ["Checked the row context", "Searched the web"],
            "sources": [{"title": "Search result", "url": "https://example.com/competitors"}],
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Find competitors of {role.current_company}.",
        mode="auto",
        output_schema=[{"key": "competitors", "label": "Competitors", "primary": True}],
        context={"role.current_company": "Exotel"},
    )

    assert result["primary_output"] == "Acme, BetaCo, GammaInc"
    assert calls == [True]


def test_run_ai_task_web_without_source_url_keeps_answer_with_verification_warning(monkeypatch):
    def fake_openai(system_prompt, user_prompt, *, use_web=False):
        assert use_web is True
        return {
            "outputs": {"result": "Yes"},
            "reasoning": "No direct source URL returned.",
            "confidence": "medium",
            "steps": ["Searched the web"],
            "sources": [{"title": "Unlinked source"}],
        }

    monkeypatch.setattr(ai_columns, "_openai_client", object())
    monkeypatch.setattr(ai_columns, "_call_openai_for_json", fake_openai)
    result = ai_columns._run_ai_task(
        prompt_template="Any recent layoffs at the current company?",
        mode="auto",
        output_schema=[{"key": "result", "label": "Result", "primary": True}],
        context={"role.current_company": "ExampleCo"},
    )

    assert result["primary_output"] == "Yes"
    assert result["details"]["verification_status"] == "passed_with_unknowns"
    assert "Web-derived answer has no source URL." in result["details"]["unknown_reasons"]


def test_limit_profiles_for_field_catalog_caps_size(monkeypatch):
    from backend.api.routes import ai_columns as ac

    monkeypatch.setattr(ac, "_MAX_PROFILES_FOR_FIELD_CATALOG", 12)
    big = [{"id": i, "raw_fields": {}} for i in range(100)]
    out = ac._limit_profiles_for_field_catalog(big)
    assert len(out) == 12
    assert out[0]["id"] == 0
    assert out[-1]["id"] == 99

    small = [{"id": 1}, {"id": 2}]
    assert ac._limit_profiles_for_field_catalog(small) == small


def _mock_db_conn_for_delete(fetchone_sequence):
    """fetchone_sequence: list of return values for each fetchone call."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_cur
    cm.__exit__.return_value = False
    mock_conn.cursor.return_value = cm
    fq = list(fetchone_sequence)

    def fetchone():
        return fq.pop(0) if fq else None

    mock_cur.fetchone.side_effect = fetchone
    return mock_conn, mock_cur


def test_delete_ai_column_returns_404_when_not_found(monkeypatch):
    mock_conn, _ = _mock_db_conn_for_delete([None])

    class CM:
        def __enter__(self):
            return mock_conn

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ai_columns, "get_db_connection_context", lambda **kw: CM())
    with pytest.raises(HTTPException) as ei:
        ai_columns.delete_ai_column(999, current_user=_user())
    assert ei.value.status_code == 404


def test_delete_ai_column_archives_and_cancels_when_found(monkeypatch):
    mock_conn, mock_cur = _mock_db_conn_for_delete([(42,), (42,)])

    class CM:
        def __enter__(self):
            return mock_conn

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ai_columns, "get_db_connection_context", lambda **kw: CM())
    out = ai_columns.delete_ai_column(42, current_user=_user())
    assert out == {"deleted": True}
    assert mock_conn.commit.called
    assert mock_cur.execute.call_count == 3
    sqls = [call.args[0] for call in mock_cur.execute.call_args_list]
    assert "SELECT id FROM ai_column_definitions" in sqls[0]
    assert "UPDATE ai_column_runs" in sqls[1] and "status = 'canceled'" in sqls[1]
    assert "UPDATE ai_column_definitions" in sqls[2] and "is_archived = TRUE" in sqls[2]

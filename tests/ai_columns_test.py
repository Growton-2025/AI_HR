import asyncio

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from backend.api import schemas
from backend.api.routes import ai_columns
from backend.api.routes import browse
from backend.db.ai_column_migrate import ensure_ai_column_migrations
from backend.services.ai_columns import (
    build_field_catalog,
    build_candidate_context,
    classify_ai_column_prompt,
    compute_career_facts,
    evaluate_required_fields,
    flatten_profile_context,
    fill_prompt_template,
    map_raw_outputs_to_schema_keys,
    map_career_facts_to_outputs,
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
    ai_keys = {item["key"] for item in groups["AI Columns"]}

    assert "candidate.linkedin" in default_keys
    assert "candidate.full_name" in default_keys
    assert "raw.import_company" in imported_keys
    assert "ai.company_location.result" in ai_keys


def test_required_fields_and_prompt_fill_use_ai_context():
    profile = {
        "id": 1,
        "name": "Deepak Basavaraj",
        "first_name": "Deepak",
        "last_name": "Basavaraj",
        "linkedin": "https://linkedin.com/in/deepak",
        "roles": [{"title": "Account Director", "company": "Exotel"}],
        "raw_fields": {"import_company": "Exotel"},
    }
    context = build_candidate_context(profile, ai_values={"ai.company_location.result": "Bengaluru"})
    rendered = fill_prompt_template(
        "Check {candidate.linkedin} and compare with {ai.company_location.result}",
        context,
    )
    ok, missing = evaluate_required_fields(
        ["candidate.linkedin", "ai.company_location.result"],
        context,
    )

    assert ok is True
    assert missing == []
    assert "https://linkedin.com/in/deepak" in rendered
    assert "Bengaluru" in rendered
    assert context["candidate.full_name"] == "Deepak Basavaraj"
    assert context["candidate.name"] == "Deepak Basavaraj"
    assert context["Linkedin Profile"] == "https://linkedin.com/in/deepak"
    assert context["row.raw_fields.import_company"] == "Exotel"


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


def test_career_facts_collapse_company_and_exclude_memberships():
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
    assert "RevGenius" not in facts["companies"]
    assert facts["ae_experience_months"] >= 60
    assert outputs["average_tenure_months"].isdigit()
    assert outputs["current_job_months"].isdigit()
    assert outputs["career_city_count"] == "2"


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

    class FakeResponse:
        output_text = '{"outputs":{"result":"fresh"},"confidence":"high"}'

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


def test_map_raw_outputs_to_schema_keys_aligns_label_style_keys():
    raw = {"Competitor Name": "Acme Corp", "competitor_industry": "Software"}
    got = map_raw_outputs_to_schema_keys(raw, ["competitor_name", "competitor_industry"])
    assert got["competitor_name"] == "Acme Corp"
    assert got["competitor_industry"] == "Software"


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
    assert "No structured response" in result["outputs"]["a"] or "No structured response" in result["outputs"]["b"]


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


def test_run_ai_task_auto_computes_tenure_without_openai(monkeypatch):
    monkeypatch.setattr(ai_columns, "_openai_client", None)
    context = build_candidate_context(
        {
            "id": 1,
            "name": "Tenure Candidate",
            "roles": [
                {"title": "Account Executive", "company": "Acme", "start_date": "2021-01", "end_date": "Present"},
                {"title": "Senior Account Executive", "company": "Acme", "start_date": "2019-01", "end_date": "2020-12"},
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
            "sources": [{"title": "Search result"}],
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
    assert calls == [False, True]


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

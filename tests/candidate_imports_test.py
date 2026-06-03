import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from backend.api.routes import candidate_imports
from backend.services.imported_fields import normalize_imported_field_key


def _upload_file(csv_text: str, filename: str = "candidates.csv") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(csv_text.encode("utf-8")))


def test_preview_falls_back_to_alias_mapping_when_model_unavailable(monkeypatch):
    async def no_model_mapping(_headers, _sample_rows):
        return {}

    monkeypatch.setattr(candidate_imports, "_model_mapping", no_model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "First Name,Last Name,LinkedIn URL,City,Job Title,Company\n"
                "Ada,Lovelace,https://linkedin.com/in/ada,London,Founder,Analytical Engines\n"
            ),
            use_llm=True,
        )
    )

    assert result["headers"] == [
        "First Name",
        "Last Name",
        "LinkedIn URL",
        "City",
        "Job Title",
        "Company",
    ]
    assert result["row_count"] == 1
    assert result["suggested_mapping"]["First Name"] == "first_name"
    assert result["suggested_mapping"]["LinkedIn URL"] == "linkedin"
    assert result["suggested_mapping"]["Job Title"] == "title"
    assert result["mapping_details"]["Company"]["source"] == "alias"
    assert result["mapping_details"]["Company"]["sample_values"] == ["Analytical Engines"]
    assert result["missing_required"] == []


def test_preview_prefers_model_mapping_with_details(monkeypatch):
    async def model_mapping(_headers, _sample_rows):
        return {
            "Given": {
                "target": "first_name",
                "confidence": 0.88,
                "reason": "Given names are first names.",
            },
            "Surname": {"target": "last_name", "confidence": 0.91, "reason": "Surname."},
            "Profile": {"target": "linkedin", "confidence": 0.83, "reason": "LinkedIn URL."},
            "Metro": {"target": "city", "confidence": 0.79, "reason": "Metro area."},
            "Role": {"target": "title", "confidence": 0.81, "reason": "Job role."},
        }

    monkeypatch.setattr(candidate_imports, "_model_mapping", model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "Given,Surname,Profile,Metro,Role\n"
                "Grace,Hopper,https://linkedin.com/in/gracehopper,New York,Admiral\n"
            ),
            use_llm=True,
        )
    )

    assert result["suggested_mapping"]["Given"] == "first_name"
    assert result["mapping_details"]["Given"]["source"] == "model"
    assert result["mapping_details"]["Given"]["confidence"] == pytest.approx(0.88)
    assert result["mapping_details"]["Given"]["reason"] == "Given names are first names."
    assert result["missing_required"] == []


def test_preview_forces_work_history_columns_to_custom_even_if_model_maps_them(monkeypatch):
    async def model_mapping(_headers, _sample_rows):
        return {
            "Current Role": {"target": "title", "confidence": 0.9, "reason": "Current title."},
            "Company 2 Name": {"target": "company_name", "confidence": 0.9, "reason": "Company."},
            "Title.1": {"target": "title", "confidence": 0.9, "reason": "Title."},
            "Start date.1": {"target": "notes", "confidence": 0.9, "reason": "Date."},
            "End Date.1": {"target": "notes", "confidence": 0.9, "reason": "Date."},
            "Details .1": {"target": "notes", "confidence": 0.9, "reason": "Details."},
        }

    monkeypatch.setattr(candidate_imports, "_model_mapping", model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "First Name,Last Name,LinkedIn URL,City,Current Role,Company 2 Name,Title.1,Start date.1,End Date.1,Details .1\n"
                "Ada,Lovelace,https://linkedin.com/in/ada,London,Founder,Beta,AE,Jan 2020,Present,Sold SaaS\n"
            ),
            use_llm=True,
        )
    )

    assert result["suggested_mapping"]["Current Role"] == "title"
    assert result["suggested_mapping"]["Company 2 Name"] == "custom"
    assert result["suggested_mapping"]["Title.1"] == "custom"
    assert result["suggested_mapping"]["Start date.1"] == "custom"
    assert result["suggested_mapping"]["End Date.1"] == "custom"
    assert result["suggested_mapping"]["Details .1"] == "custom"
    assert result["mapping_details"]["Start date.1"]["source"] == "history"


def test_preview_defaults_non_empty_unknown_columns_to_custom(monkeypatch):
    async def no_model_mapping(_headers, _sample_rows):
        return {}

    monkeypatch.setattr(candidate_imports, "_model_mapping", no_model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "First Name,Last Name,LinkedIn URL,City,Job Title,Current CTC,Expected CTC.,Notice Period,Preferred Loc,Focused Geog,Shift timings,CV\n"
                "Ada,Lovelace,https://linkedin.com/in/ada,London,Founder,15 LPA,20 LPA,30 days,Bangalore,US,6 PM,https://docs.example/cv\n"
            ),
            use_llm=True,
        )
    )

    for header in [
        "Current CTC",
        "Expected CTC.",
        "Notice Period",
        "Preferred Loc",
        "Focused Geog",
        "Shift timings",
        "CV",
    ]:
        assert result["suggested_mapping"][header] == "custom"
        assert result["mapping_details"][header]["source"] == "custom"


def test_row_values_store_original_headers_and_normalized_extra_metadata():
    import pandas as pd

    row = pd.Series(
        {
            "First Name": "Ada",
            "Current CTC": "15 LPA",
            "Expected CTC.": "20 LPA",
            "Notice Period": "30 days",
            "Preferred Loc": "Bangalore",
            "Ignored": "drop me",
        }
    )

    vals, raw = candidate_imports._row_values(
        row,
        {
            "First Name": "first_name",
            "Current CTC": "custom",
            "Expected CTC.": "custom",
            "Notice Period": "custom",
            "Preferred Loc": "custom",
            "Ignored": "ignore",
        },
    )

    assert vals["first_name"] == "Ada"
    assert raw["Current CTC"] == "15 LPA"
    assert raw["Expected CTC."] == "20 LPA"
    assert "Ignored" not in raw
    assert raw["imported_extra_fields"]["current_ctc"]["value"] == "15 LPA"
    assert raw["imported_extra_fields"]["expected_ctc"]["source_header"] == "Expected CTC."
    assert raw["imported_extra_fields"]["notice_period"]["value"] == "30 days"
    assert raw["imported_extra_fields"]["preferred_location"]["value"] == "Bangalore"


def test_hayasa_apify_preview_keeps_education_custom_and_headline_as_title(monkeypatch):
    async def no_model_mapping(_headers, _sample_rows):
        return {}

    monkeypatch.setattr(candidate_imports, "_model_mapping", no_model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "firstName,lastName,linkedinPublicUrl,headline,addressWithCountry,fullName,Skills,experiences/0/companyName,experiences/0/title,experiences/0/jobStartedOn,experiences/0/jobEndedOn,experiences/0/title.1,educations/0/subtitle,educations/0/title,Recruiter Summary,Current CTC,CV\n"
                "Anuj,Anand,https://www.linkedin.com/in/anuj-anand-9351311b8/,Sales Development Representative @Wayground,Bengaluru Karnataka India,Anuj Anand,Cold Calling,Wayground,SDR,2026-03-01,,Consultant,BTech Computer Science,Amity University,Great candidate,15 LPA,https://docs.example/cv\n"
            ),
            use_llm=True,
        )
    )

    assert result["suggested_mapping"]["headline"] == "title"
    assert result["suggested_mapping"]["addressWithCountry"] == "city"
    assert result["suggested_mapping"]["educations/0/subtitle"] == "custom"
    assert result["suggested_mapping"]["educations/0/title"] == "custom"
    assert result["mapping_details"]["educations/0/subtitle"]["category"] == "Education"
    assert result["mapping_details"]["educations/0/subtitle"]["detected_field_kind"] == "details"
    assert result["mapping_details"]["experiences/0/title.1"]["category"] == "Work History"
    assert result["mapping_details"]["experiences/0/title.1"]["detected_role_index"] == 2
    assert result["mapping_details"]["Recruiter Summary"]["target"] == "custom"
    assert result["missing_required"] == []


def test_hayasa_row_values_preserve_raw_location_education_and_extra_fields():
    import pandas as pd

    row = pd.Series(
        {
            "firstName": "Anuj",
            "lastName": "Anand",
            "linkedinPublicUrl": "https://www.linkedin.com/in/anuj-anand-9351311b8/",
            "headline": "Sales Development Representative @Wayground",
            "addressWithCountry": "Bengaluru, Karnataka India",
            "educations/0/subtitle": "Bachelor of Technology, Computer Science",
            "Recruiter Summary": "Outbound sales summary",
            "Current CTC": "15 LPA",
        }
    )

    vals, raw = candidate_imports._row_values(
        row,
        {
            "firstName": "first_name",
            "lastName": "last_name",
            "linkedinPublicUrl": "linkedin",
            "headline": "title",
            "addressWithCountry": "city",
            "educations/0/subtitle": "custom",
            "Recruiter Summary": "custom",
            "Current CTC": "custom",
        },
    )

    assert vals["title"] == "Sales Development Representative @Wayground"
    assert vals["headline"] == "Sales Development Representative @Wayground"
    assert vals["city"] == "Bengaluru, Karnataka India"
    assert vals["location"] == "Bengaluru, Karnataka India"
    assert raw["addressWithCountry"] == "Bengaluru, Karnataka India"
    assert raw["educations/0/subtitle"] == "Bachelor of Technology, Computer Science"
    assert raw["Recruiter Summary"] == "Outbound sales summary"
    assert raw["imported_extra_fields"]["current_ctc"]["value"] == "15 LPA"


def test_imported_extra_header_normalization_handles_common_variants():
    assert normalize_imported_field_key("Current CTC") == "current_ctc"
    assert normalize_imported_field_key("curr ctc") == "current_ctc"
    assert normalize_imported_field_key("Expected CTC.") == "expected_ctc"
    assert normalize_imported_field_key("Preferred Loc") == "preferred_location"
    assert normalize_imported_field_key("Focused Geog") == "focused_geography"
    assert normalize_imported_field_key("Shift timings") == "shift_timings"
    assert normalize_imported_field_key("Outbound Exp") == "outbound_experience"


def test_preview_returns_grouped_metadata_for_large_dynamic_history_upload(monkeypatch):
    async def no_model_mapping(_headers, _sample_rows):
        return {}

    monkeypatch.setattr(candidate_imports, "_model_mapping", no_model_mapping)

    base_headers = ["First Name", "Last Name", "LinkedIn URL", "City", "Headline"]
    history_headers = []
    for idx in range(40):
        history_headers.extend(
            [
                f"experiences/{idx}/companyName",
                f"experiences/{idx}/title",
                f"experiences/{idx}/jobStartedOn",
                f"experiences/{idx}/jobEndedOn",
                f"experiences/{idx}/jobDescription",
            ]
        )
    extra_headers = ["Mystery Revenue Signal", "Current CTC"]
    headers = base_headers + history_headers + extra_headers
    values = [
        "Ada",
        "Lovelace",
        "https://linkedin.com/in/ada",
        "London",
        "Founder",
    ]
    for idx in range(40):
        values.extend([f"Company {idx}", f"Role {idx}", "2020-01-01", "2021-01-01", "Sold software"])
    values.extend(["Important custom note", "15 LPA"])

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(",".join(headers) + "\n" + ",".join(values) + "\n"),
            use_llm=True,
        )
    )

    assert result["headers"] == headers
    assert len(result["headers"]) > 200
    assert set(result["mapping_details"]) == set(headers)
    assert len(result["field_groups"]["Work History"]) == len(history_headers)
    assert result["mapping_details"]["experiences/7/jobStartedOn"]["category"] == "Work History"
    assert result["mapping_details"]["experiences/7/jobStartedOn"]["friendly_label"] == "Experience 8 Start Date"
    assert result["mapping_details"]["experiences/7/jobStartedOn"]["detected_role_index"] == 8
    assert result["mapping_details"]["experiences/7/jobStartedOn"]["detected_field_kind"] == "start_date"
    assert result["mapping_details"]["Mystery Revenue Signal"]["target"] == "custom"
    assert result["mapping_details"]["Mystery Revenue Signal"]["category"] == "Other Fields"
    assert result["mapping_details"]["Mystery Revenue Signal"]["preserve_reason"] == "Preserved as imported extra data"
    assert result["mapping_details"]["Current CTC"]["category"] == "Contact/Compensation"


def test_preview_marks_incompatible_model_mapping_as_needs_review(monkeypatch):
    async def model_mapping(_headers, _sample_rows):
        return {
            "Profile Maybe": {
                "target": "linkedin",
                "confidence": 0.9,
                "reason": "Model guessed LinkedIn.",
            }
        }

    monkeypatch.setattr(candidate_imports, "_model_mapping", model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "First Name,Last Name,LinkedIn URL,City,Headline,Profile Maybe\n"
                "Ada,Lovelace,https://linkedin.com/in/ada,London,Founder,not a url\n"
            ),
            use_llm=True,
        )
    )

    detail = result["mapping_details"]["Profile Maybe"]
    assert detail["category"] == "Needs Review"
    assert detail["confidence"] <= 0.45
    assert "do not look like LinkedIn URLs" in detail["reason"]


def test_preview_preserves_dynamic_exp_company_start_end_headers(monkeypatch):
    async def no_model_mapping(_headers, _sample_rows):
        return {}

    monkeypatch.setattr(candidate_imports, "_model_mapping", no_model_mapping)

    result = asyncio.run(
        candidate_imports.build_upload_preview_response(
            _upload_file(
                "First Name,Last Name,LinkedIn URL,City,Headline,exp 0 company,exp 0 start,exp 0 end,company 0 name,company 0 start date,experience_2_duration\n"
                "Ada,Lovelace,https://linkedin.com/in/ada,London,Founder,Acme,2020,2021,Beta,2022,18 months\n"
            ),
            use_llm=True,
        )
    )

    assert result["mapping_details"]["exp 0 company"]["category"] == "Work History"
    assert result["mapping_details"]["exp 0 company"]["friendly_label"] == "Experience 1 Company"
    assert result["mapping_details"]["company 0 start date"]["detected_field_kind"] == "start_date"
    assert result["mapping_details"]["experience_2_duration"]["detected_field_kind"] == "duration_raw"
    assert result["suggested_mapping"]["experience_2_duration"] == "custom"


def test_validate_mapping_requires_core_fields_and_ignores_unknown_targets():
    with pytest.raises(HTTPException) as exc_info:
        candidate_imports._validate_mapping(
            {
                "First": "first_name",
                "Last": "last_name",
                "LI": "linkedin",
                "City": "city",
            }
        )

    assert exc_info.value.status_code == 400
    assert "title" in exc_info.value.detail

    mapping = candidate_imports._validate_mapping(
        {
            "First": "first_name",
            "Last": "last_name",
            "LI": "linkedin",
            "City": "city",
            "Title": "title",
            "Mystery": "not_a_target",
        }
    )

    assert mapping["Mystery"] == "ignore"


def test_assign_imported_candidate_to_role_is_append_only_dedupe():
    class Cursor:
        rowcount = 1

        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

    cur = Cursor()

    assigned = candidate_imports._assign_imported_candidate_to_role(
        cur,
        role_id=42,
        candidate_id=9001,
    )

    assert assigned is True
    assert "ON CONFLICT (role_id, candidate_id) DO NOTHING" in cur.calls[0][0]
    assert cur.calls[0][1] == (42, 9001)

    cur.rowcount = 0
    assigned_again = candidate_imports._assign_imported_candidate_to_role(
        cur,
        role_id=42,
        candidate_id=9001,
    )
    assert assigned_again is False


def test_no_mutate_duplicate_policy_counts_existing_as_updated_without_writes(monkeypatch):
    import pandas as pd

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _sql, _params=None):
            return None

    class Conn:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0

        def cursor(self):
            return Cursor()

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    updates = []
    upsert_calls = []
    refreshed = []

    def prefetch(_cur, *, owner_user_id, prepared_rows):
        if owner_user_id is None:
            return {"linkedin": {"/in/existing": 123}, "email": {}, "identity": {}}
        return {"linkedin": {}, "email": {}, "identity": {}}

    def forbidden_upsert(*_args, **_kwargs):
        upsert_calls.append(_kwargs)
        raise AssertionError("existing rows must not be upserted")

    monkeypatch.setattr(candidate_imports, "get_db_connection", lambda **_kwargs: Conn())
    monkeypatch.setattr(candidate_imports, "return_db_connection", lambda _conn: None)
    monkeypatch.setattr(candidate_imports, "fetch_best_contacts_for_normalized_lis", lambda _cur, _lis: {})
    monkeypatch.setattr(candidate_imports, "_prefetch_candidate_matches", prefetch)
    monkeypatch.setattr(candidate_imports, "_fast_new_import_rows", lambda prepared_rows, **_kwargs: ([], prepared_rows))
    monkeypatch.setattr(candidate_imports, "upsert_master_catalog_row", forbidden_upsert)
    monkeypatch.setattr(candidate_imports, "upsert_recruiter_pool_row", forbidden_upsert)
    monkeypatch.setattr(candidate_imports, "_assign_imported_candidate_to_role", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(candidate_imports, "_write_upload_progress", lambda _cur, _upload_id, **kwargs: updates.append(kwargs))
    monkeypatch.setattr(candidate_imports, "_refresh_import_caches", lambda ids: refreshed.append(list(ids)))

    df = pd.DataFrame(
        [
            {
                "First Name": "Existing",
                "Last Name": "Person",
                "LinkedIn URL": "https://www.linkedin.com/in/existing/",
                "City": "Bengaluru",
                "Headline": "Sales Development Representative",
            }
        ]
    )

    candidate_imports._process_upload_rows(
        upload_id=99,
        df=df,
        mapping={
            "First Name": "first_name",
            "Last Name": "last_name",
            "LinkedIn URL": "linkedin",
            "City": "city",
            "Headline": "title",
        },
        owner_user_id=77,
        user_role="recruiter",
        role_id=None,
        enrichment_mode="none",
        duplicate_policy=candidate_imports.NO_MUTATE_EXISTING_POLICY,
    )

    assert upsert_calls == []
    assert refreshed == [[]]
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["processed_count"] == 1
    assert updates[-1]["inserted_count"] == 0
    assert updates[-1]["updated_count"] == 1
    assert updates[-1]["skipped_count"] == 0


def test_upload_status_payload_includes_progress_and_errors():
    class Stamp:
        def isoformat(self):
            return "2026-05-22T10:00:00"

    payload = candidate_imports._upload_status_payload(
        (
            7,
            22,
            "talent.xlsx",
            83,
            37,
            12,
            24,
            1,
            9,
            "processing",
            44,
            Stamp(),
            "row 8: bad url\nrow 9: missing name",
            None,
        )
    )

    assert payload["upload_id"] == 7
    assert payload["row_count"] == 83
    assert payload["processed_count"] == 37
    assert payload["role_assigned_count"] == 9
    assert payload["errors"] == ["row 8: bad url", "row 9: missing name"]

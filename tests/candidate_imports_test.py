import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile

from backend.api.routes import candidate_imports


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

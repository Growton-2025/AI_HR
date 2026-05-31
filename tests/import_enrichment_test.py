from datetime import datetime, timezone

from backend.services import import_enrichment as enrich
from backend.services.ai_columns import build_candidate_context, compute_career_facts
from backend.services.candidate_pool import suggest_header_mapping


def test_parse_linkedin_style_duplicate_role_headers():
    raw = {
        "import_company": "Acme SaaS",
        "Title": "Enterprise Account Executive",
        "Start date": "Jan 2020",
        "End Date": "Dec 2021",
        "Details ": "Sold enterprise platform",
        "Company 2 Name": "Beta Co",
        "Title.1": "SDR",
        "Start date.1": "2018",
        "End Date.1": "2019",
        "Details .1": "Generated pipeline for SMB customers",
        "Education 1 - College Name": "State University",
        "Degree Name": "MBA",
        "Start date.10": "2016",
        "End Date.10": "2018",
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": "Enterprise Account Executive"})
    education = enrich.parse_education_from_raw(raw)

    assert [role.company for role in roles] == ["Acme SaaS", "Beta Co"]
    assert roles[0].title == "Enterprise Account Executive"
    assert roles[1].details == "Generated pipeline for SMB customers"
    assert education[0].college == "State University"
    assert education[0].degree == "MBA"


def test_tenure_merges_same_company_and_overlapping_windows():
    roles = [
        enrich.ParsedRole(
            index=1,
            company="Acme Inc",
            title="Enterprise Account Executive",
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2021, 12, 1, tzinfo=timezone.utc),
            duration_months=24,
        ),
        enrich.ParsedRole(
            index=2,
            company="Acme",
            title="Senior Account Executive",
            start=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end=datetime(2022, 12, 1, tzinfo=timezone.utc),
            duration_months=24,
        ),
        enrich.ParsedRole(
            index=3,
            company="Beta Co",
            title="SDR",
            start=datetime(2018, 1, 1, tzinfo=timezone.utc),
            end=datetime(2019, 12, 1, tzinfo=timezone.utc),
            duration_months=24,
        ),
    ]

    metrics = enrich.calculate_tenure_metrics(roles)

    assert metrics["total_experience_months"] == 60
    assert metrics["total_experience_years"] == 5.0
    assert metrics["avg_tenure_months"] == 30
    assert {row["company"]: row["months"] for row in metrics["company_years"]} == {
        "Acme Inc": 36,
        "Beta Co": 24,
    }


def test_function_and_company_classification_do_not_guess_unknowns():
    role = enrich.ParsedRole(index=1, company="UnknownCo", title="Account Executive", details="")
    label, confidence, reason = enrich.classify_function(role)

    assert label == "Hunting"
    assert confidence == "high"
    assert "taxonomy" in reason.lower()

    company = enrich.classify_company(
        "No Such Company 987",
        role_texts=["Sold to SMB and SME accounts"],
        allow_web=False,
    )
    assert company["customer_segment"] == ["SMB"]
    assert company["verification_status"] == "row_context"

    unknown = enrich.classify_company("No Such Company 987", role_texts=[], allow_web=False)
    assert unknown["product_service"] == "Unknown"
    assert unknown["verification_status"] == "not_verified"


def test_company_segments_are_canonicalized_from_cache_or_db_labels():
    company = enrich.classify_company(
        "Existing Co",
        db_details={
            "product_service": "Cloud platform",
            "customer_segment": ["Enterprises", "Government"],
            "business_model": "B2B",
        },
        allow_web=False,
    )

    assert "Enterprise" in company["customer_segment"]
    assert "Government" in company["customer_segment"]


def test_header_mapping_saves_mobile_directly_as_phone_and_keeps_history_raw():
    mapping = suggest_header_mapping(
        [
            "Profile Link",
            "Person Linkedin Url",
            "addressWithCountry",
            "Mobile Number",
            "Bio",
            "Company 1 Name",
            "Title.1",
        ]
    )

    assert mapping["Profile Link"] == "linkedin"
    assert mapping["Person Linkedin Url"] == "linkedin"
    assert mapping["addressWithCountry"] == "location"
    assert mapping["Mobile Number"] == "phone"
    assert mapping["Bio"] == "about"
    assert "Company 1 Name" not in mapping
    assert "Title.1" not in mapping


def test_smart_column_career_facts_read_verified_enrichment_roles():
    context = build_candidate_context(
        {
            "id": 12,
            "name": "Verified Candidate",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Acme SaaS",
                            "title": "Account Executive",
                            "start_date": "2020-01-01",
                            "end_date": "2021-12-01",
                        },
                        {
                            "company": "Beta Co",
                            "title": "SDR",
                            "start_date": "2018-01-01",
                            "end_date": "2019-12-01",
                        },
                    ]
                }
            },
        }
    )

    facts = compute_career_facts(context)

    assert facts["unique_company_count"] == 2
    assert facts["total_experience_months"] == 48
    assert facts["average_tenure_months"] == 24


def test_role_specific_tenure_is_preserved_inside_same_company():
    context = build_candidate_context(
        {
            "id": 13,
            "name": "Multi Role Candidate",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": "Acme SaaS",
                            "title": "Enterprise Account Executive",
                            "start_date": "2020-01-01",
                            "end_date": "2021-12-01",
                        },
                        {
                            "company": "Acme SaaS",
                            "title": "Sales Development Representative",
                            "start_date": "2018-01-01",
                            "end_date": "2019-12-01",
                        },
                    ]
                }
            },
        }
    )

    facts = compute_career_facts(context)
    role_months = {
        (role["company"], role["title"]): role["months"]
        for role in facts["role_tenures"]
    }

    assert facts["unique_company_count"] == 1
    assert facts["total_experience_months"] == 48
    assert role_months[("Acme SaaS", "Enterprise Account Executive")] == 24
    assert role_months[("Acme SaaS", "Sales Development Representative")] == 24


def test_about_text_adds_profile_claims_without_overriding_tenure():
    candidate = {
        "headline": "Enterprise SaaS sales leader",
        "about": "8 years selling to enterprise, SMB and SME customers across APAC and India. Strong hunting and customer success exposure.",
        "city": "Bengaluru",
        "raw_fields": {
            "services": "Cloud software and SaaS",
            "Skills": "Hunting, customer success, renewals",
        },
    }

    claims = enrich.extract_profile_claims(candidate)

    assert "Enterprise" in claims["segments"]
    assert "SMB" in claims["segments"]
    assert "APAC" in claims["geographies"]
    assert {item["function"] for item in claims["functions"]} >= {"Hunting", "Customer Success"}
    assert claims["product_service"] == "SaaS"
    assert "do not override" in claims["note"]
    assert claims["claimed_experience"][0]["claimed_years"] == 8
    assert "Enterprise" in claims["claimed_experience"][0]["segments"]
    assert "APAC" in claims["claimed_experience"][0]["geographies"]

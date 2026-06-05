from datetime import datetime, timezone

from backend.services import import_enrichment as enrich
from backend.services.ai_columns import build_candidate_context, compute_career_facts
from backend.services.candidate_pool import suggest_header_mapping


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 2, tzinfo=tz or timezone.utc)


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


def test_apify_current_role_month_year_uses_dates_over_stale_duration(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, 5, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(enrich, "datetime", FrozenDateTime)
    raw = {
        "experiences/0/companyName": "Datadog",
        "experiences/0/title": "Enterprise Sales Development Representative",
        "experiences/0/jobStartedOn": "10-2025",
        "experiences/0/jobEndedOn": "",
        "experiences/0/duration": "13 months",
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": "Enterprise Sales Development Representative"})
    metrics = enrich.calculate_tenure_metrics(roles)
    context = build_candidate_context(
        {
            "name": "Aarthi Nambiar",
            "raw_fields": {
                "enrichment": {
                    "roles": [
                        {
                            "company": role.company,
                            "title": role.title,
                            "start_date": role.start.date().isoformat() if role.start else "",
                            "end_date": "",
                            "duration_months": role.duration_months,
                        }
                        for role in roles
                    ]
                }
            },
        }
    )
    facts = compute_career_facts(context)

    assert roles[0].start.date().isoformat() == "2025-10-01"
    assert roles[0].duration_months == 9
    assert roles[0].duration_source == "date_range"
    assert metrics["current_job_months"] == 9
    assert metrics["company_tenures"][0]["months"] == 9
    assert facts["current_job_months"] == 9
    assert facts["current_company_tenure_months"] == 9


def test_parse_apify_experience_columns_and_profile_claim_context():
    raw = {
        "headline": "SaaS account executive for US and EMEA markets",
        "Experience": "Total 6+ yrs; SaaS 4 yrs; US 3 yrs",
        "Focused Geography": "US, EMEA",
        "experiences/0/companyName": "Acme Cloud",
        "experiences/0/title": "Enterprise Account Executive",
        "experiences/0/jobStartedOn": "2022-01-01 00:00:00",
        "experiences/0/jobEndedOn": "",
        "experiences/0/companyIndustry": "Software Development",
        "experiences/0/companySize": "501-1000",
        "experiences/0/jobLocation": "Bengaluru, India",
        "experiences/1/companyName": "Beta HR",
        "experiences/1/title": "Sales Development Representative",
        "experiences/1/jobStartedOn": "2020-01-01 00:00:00",
        "experiences/1/jobEndedOn": "2021-12-01 00:00:00",
        "experiences/1/jobDescription": "Generated outbound pipeline for SMB accounts",
        "imported_extra_fields": {
            "focused_geography": {
                "label": "Focused Geography",
                "value": "US, EMEA",
                "source_header": "Focused Geography",
            }
        },
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": raw["headline"]})
    claims = enrich.extract_profile_claims({"headline": raw["headline"], "raw_fields": raw})

    assert [role.company for role in roles] == ["Acme Cloud", "Beta HR"]
    assert roles[0].source_industry == "Software Development"
    assert roles[0].source_company_size == "501-1000"
    assert roles[1].details == "Generated outbound pipeline for SMB accounts"
    assert {"Americas", "EMEA"} <= set(claims["geographies"])
    assert claims["claimed_experience"][0]["claimed_years"] == 6


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
    assert metrics["avg_tenure_months"] == 24
    assert metrics["completed_company_count"] == 1
    assert metrics["completed_company_months"] == 24
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
    assert mapping["addressWithCountry"] == "city"
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


def test_month_year_dates_parse_without_falling_back_to_january(monkeypatch):
    monkeypatch.setattr(enrich, "datetime", _FrozenDateTime)

    assert enrich.parse_profile_date("03-2026").date().isoformat() == "2026-03-01"
    assert enrich.parse_profile_date("12/2025").date().isoformat() == "2025-12-01"
    assert enrich.parse_profile_date("07-2025").date().isoformat() == "2025-07-01"
    assert enrich.parse_profile_date("11-2025").date().isoformat() == "2025-11-01"
    assert enrich.parse_profile_date("2025-07-15 00:00:00").date().isoformat() == "2025-07-15"

    current = enrich.parse_profile_date("", default_current=True)
    assert current.date().isoformat() == "2026-06-02"


def test_anuj_dynamic_apify_headers_map_without_hallucinating(monkeypatch):
    monkeypatch.setattr(enrich, "datetime", _FrozenDateTime)
    raw = {
        "firstName": "Anuj",
        "lastName": "Anand",
        "fullName": "Anuj Anand",
        "linkedinPublicUrl": "https://www.linkedin.com/in/anuj-anand-9351311b8/",
        "headline": "Sales Development Representative @Wayground | Building Outbound Sales Pipelines",
        "addressWithCountry": "Bengaluru, Karnataka India",
        "Skills": "Cold Calling, Revenue Generation, Lead Generation, CRM",
        "experiences/0/companyName": "Wayground (Formerly Quizizz)",
        "experiences/0/companyWebsite": "wayground.com",
        "experiences/0/companyIndustry": "E-Learning",
        "experiences/0/companySize": "201-500",
        "experiences/0/title": "Sales Development Representative - ROW",
        "experiences/0/jobStartedOn": "03-2026",
        "experiences/0/jobEndedOn": "",
        "experiences/0/jobLocation": "Bengaluru",
        "experiences/1/companyName": "Nvelop Technologies",
        "experiences/1/companyWebsite": "nvelop.ai",
        "experiences/1/companyIndustry": "Internet",
        "experiences/1/companySize": "1-10",
        "experiences/0/title.1": "Sourcing Transformation Consultant | EMEA",
        "experiences/1/jobStartedOn": "12-2025",
        "experiences/1/jobEndedOn": "03-2026",
        "experiences/1/jobLocation": "Bengaluru",
        "experiences/2/companyName": "Highradius",
        "experiences/2/companyWebsite": "highradius.com",
        "experiences/2/companyIndustry": "Computer Software",
        "experiences/2/companySize": "1001-5000",
        "experiences/2/title": "HighRadius",
        "experiences/2/jobStartedOn": "07-2025",
        "experiences/2/jobEndedOn": "11-2025",
        "experiences/2/jobDescription": (
            "Booked SQLs with CFOs, Controllers, and VPs of Finance across "
            "Mid-Market & Enterprise segments in North America. Generated pipeline."
        ),
        "experiences/3/companyName": "Highradius",
        "experiences/3/companyWebsite": "highradius.com",
        "experiences/3/companyIndustry": "Computer Software",
        "experiences/3/companySize": "1001-5000",
        "experiences/3/title": "Business Development Intern | International Market",
        "experiences/3/jobStartedOn": "08-2023",
        "experiences/3/jobEndedOn": "09-2023",
        "experiences/3/jobLocation": "Bhubaneswar, Odisha, India",
        "Recruiter Summary": (
            "Total yrs of experience -2 yrs. No. of years of experience in SAAS(2 yrs). "
            "Geographies targeted: US (2 years), Europe, India. Enterprise and Mid Market; "
            "Nvelop Technologies-SMB, MM, Enterprise"
        ),
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": raw["headline"]})
    for role in roles:
        role.function, role.function_confidence, role.function_reason = enrich.classify_function(role)
        details = enrich._details_from_source_industry(role.source_industry)
        role.product_service = details.get("product_service") or "Unknown"
        role.industry = details.get("industry") or "Unknown"
        role.verification_status = details.get("verification_status") or "not_verified"

    metrics = enrich.calculate_tenure_metrics(roles)
    claims = enrich.extract_profile_claims({"headline": raw["headline"], "raw_fields": raw})
    payload = enrich.build_enrichment_payload(
        roles=roles,
        education=[],
        metrics=metrics,
        profile_claims=claims,
        errors=[],
        contact_from_excel=True,
    )

    assert [role.company for role in roles] == [
        "Wayground (Formerly Quizizz)",
        "Nvelop Technologies",
        "Highradius",
        "Highradius",
    ]
    assert roles[1].title == "Sourcing Transformation Consultant | EMEA"
    assert roles[0].start.date().isoformat() == "2026-03-01"
    assert roles[1].start.date().isoformat() == "2025-12-01"
    assert roles[2].start.date().isoformat() == "2025-07-01"
    assert roles[2].end.date().isoformat() == "2025-11-01"
    assert roles[0].duration_months == 4
    assert roles[1].duration_months == 4
    assert roles[2].duration_months == 5
    assert roles[3].duration_months == 2
    assert {row["company"]: row["months"] for row in metrics["company_years"]} == {
        "Wayground (Formerly Quizizz)": 4,
        "Nvelop Technologies": 4,
        "Highradius": 7,
    }
    assert metrics["total_experience_months"] == 14
    assert roles[2].function == "Sales Development"
    assert roles[2].title == ""
    assert roles[0].industry == "E-Learning"
    assert roles[1].industry == "Internet"
    assert roles[2].industry == "Computer Software"
    assert {"Enterprise", "Mid-Market", "SMB"} <= set(claims["segments"])
    assert {"Americas", "APAC"} <= set(claims["geographies"])
    assert payload["claimed_vs_dated_experience"]["mismatch"] is True


def test_current_role_title_equal_to_company_uses_headline_title(monkeypatch):
    monkeypatch.setattr(enrich, "datetime", _FrozenDateTime)
    raw = {
        "headline": "Account Executive @ BrowserStack | Driving Sales Growth",
        "experiences/0/companyName": "Browserstack",
        "experiences/0/title": "BrowserStack",
        "experiences/0/jobStartedOn": "01-2026",
        "experiences/0/jobEndedOn": "",
        "experiences/1/companyName": "Highradius",
        "experiences/0/title.1": "HighRadius",
        "experiences/1/jobStartedOn": "07-2022",
        "experiences/1/jobEndedOn": "06-2023",
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": raw["headline"]})

    assert roles[0].company == "Browserstack"
    assert roles[0].title == "Account Executive"
    assert roles[1].company == "Highradius"
    assert roles[1].title == ""


def test_financial_technology_advisor_is_not_filtered_as_community_role():
    roles = [
        enrich.ParsedRole(
            index=1,
            company="Highradius",
            title="Financial Technology Advisor",
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            duration_months=23,
        )
    ]

    metrics = enrich.calculate_tenure_metrics(roles)

    assert enrich.is_community_role(roles[0]) is False
    assert metrics["company_years"] == [{"company": "Highradius", "months": 23, "years": 1.92}]


def test_dynamic_zero_based_work_history_headers_parse_and_store_duration_details(monkeypatch):
    monkeypatch.setattr(enrich, "datetime", _FrozenDateTime)
    raw = {
        "exp 0 company": "Acme Engage",
        "exp 0 title": "Channel Sales Manager",
        "exp 0 start": "2020-01-01",
        "exp 0 end": "2021-12-01",
        "exp 0 industry": "Customer Engagement",
        "exp 0 location": "Bengaluru, India",
        "company 2 name": "Beta Engage",
        "company 2 start date": "2022-01-01",
        "company 2 end date": "2023-12-01",
        "company 2 title": "Inside Sales Representative",
        "experience_3_company": "Gamma Engage",
        "experience_3_started_on": "",
        "experience_3_duration": "18 months",
        "experience_3_title": "Account Development Representative",
        "experience_3_location": "Singapore",
    }

    roles = enrich.parse_roles_from_raw(raw, {"headline": "Sales"})
    for role in roles:
        role.function, role.function_confidence, role.function_reason = enrich.classify_function(role)
        details = enrich._details_from_source_industry(role.source_industry)
        role.product_service = details.get("product_service") or "Unknown"
        role.industry = details.get("industry") or role.source_industry or "Unknown"
        role.customer_segment = ["Enterprise"] if role.company == "Acme Engage" else []
        role.verification_status = details.get("verification_status") or "row_source"

    metrics = enrich.calculate_tenure_metrics(roles)
    payload = enrich.build_enrichment_payload(
        roles=roles,
        education=[],
        metrics=metrics,
        profile_claims={},
        errors=[],
        contact_from_excel=False,
    )

    assert [role.company for role in roles] == ["Acme Engage", "Beta Engage", "Gamma Engage"]
    assert roles[0].source_headers["company"] == "exp 0 company"
    assert roles[2].duration_months == 18
    assert roles[2].duration_source == "duration_field"
    assert metrics["unique_company_count"] == 3
    assert metrics["total_experience_months"] == 66
    assert metrics["avg_tenure_months"] == 21
    tenures = {item["company"]: item for item in metrics["company_tenures"]}
    assert tenures["Acme Engage"]["months"] == 24
    assert tenures["Acme Engage"]["industries"] == ["Customer Engagement"]
    assert tenures["Acme Engage"]["segments"] == ["Enterprise"]
    assert "APAC" in tenures["Acme Engage"]["geographies"]
    assert tenures["Gamma Engage"]["undated_duration_months"] == 18
    role_payload = {item["company"]: item for item in payload["roles"]}
    assert role_payload["Gamma Engage"]["duration_source"] == "duration_field"
    assert role_payload["Gamma Engage"]["source_headers"]["duration_raw"] == "experience_3_duration"
    assert payload["metrics"]["company_tenures"][0]["roles"]


def test_enrichment_payload_matches_db_projection_duration_contract():
    roles = [
        enrich.ParsedRole(
            index=1,
            company="Highradius",
            title="Sales Development Representative",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 1, tzinfo=timezone.utc),
            duration_months=12,
            function="Sales Development",
            industry="Computer Software",
            product_service="Computer Software",
            customer_segment=["Enterprise"],
            source_location="Bengaluru, India",
            verification_status="row_source",
            source_headers={
                "company": "experiences/0/companyName",
                "title": "experiences/0/title",
                "start_raw": "experiences/0/jobStartedOn",
                "end_raw": "experiences/0/jobEndedOn",
            },
        ),
        enrich.ParsedRole(
            index=2,
            company="Highradius",
            title="Business Development Intern",
            start=datetime(2023, 8, 1, tzinfo=timezone.utc),
            end=datetime(2023, 9, 1, tzinfo=timezone.utc),
            duration_months=2,
            function="Sales Development",
            industry="Computer Software",
            product_service="Computer Software",
            source_location="Bhubaneswar, Odisha, India",
            verification_status="row_source",
        ),
    ]

    metrics = enrich.calculate_tenure_metrics(roles)
    payload = enrich.build_enrichment_payload(
        roles=roles,
        education=[],
        metrics=metrics,
        profile_claims={},
        errors=[],
        contact_from_excel=True,
    )

    assert metrics["company_years"] == [{"company": "Highradius", "months": 14, "years": 1.17}]
    assert payload["metrics"]["unique_company_count"] == 1
    assert payload["metrics"]["avg_tenure_months"] == 0
    assert payload["metrics"]["company_tenures"][0]["months"] == 14
    assert payload["metrics"]["company_tenures"][0]["titles"] == [
        "Business Development Intern",
        "Sales Development Representative",
    ]
    assert payload["metrics"]["role_tenures"][0]["source_headers"]["company"] == "experiences/0/companyName"
    assert payload["roles"][0]["duration_years"] == 1.0

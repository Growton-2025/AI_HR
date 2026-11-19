from backend.services.apify_scrape import flatten_item


def test_flatten_uses_slash_keys_and_strips_literal_null_segments():
    item = {
        "fullName": "Ada",
        "educations": [{"title": "KS Institute", "subtitle": "Bachelor of Engineering - BE, null",
                        "period": {"startedOn": None}}],
        "skills": [{"title": "null"}, {"title": "Python"}],
        "experiences": [{"companyName": "Nullsoft, null, Inc"}],
    }
    flat = flatten_item(item)
    assert flat["educations/0/title"] == "KS Institute"
    assert flat["educations/0/subtitle"] == "Bachelor of Engineering - BE"
    assert flat["educations/0/period/startedOn"] is None
    assert flat["skills/0/title"] is None and flat["skills/1/title"] == "Python"
    assert flat["experiences/0/companyName"] == "Nullsoft, Inc"
    assert flat["fullName"] == "Ada"


import json

import pytest

from backend.services import apify_scrape
from backend.services import import_enrichment as ie


CC_ITEM = {
    "firstName": "Raul", "lastName": "Quinones", "publicIdentifier": "raulnquinones",
    "headline": "Sr. Scrum Master at Tandem Diabetes", "summary": "IT Scrum Master with 10 years...",
    "geoLocationName": "San Diego, California", "geoCountryName": "United States", "countryCode": "us",
    "jobTitle": "Sr. Scrum Master I", "companyName": "Tandem Diabetes Care", "industryName": "Medical Device",
    "positions": [
        {"title": "Sr. Scrum Master I", "companyName": "Tandem Diabetes Care",
         "locationName": "San Diego, California, United States",
         "timePeriod": {"startDate": {"month": 10, "year": 2022}},
         "company": {"employeeCountRange": {"start": 1001, "end": 5000},
                     "industries": [{"name": "Medical Equipment Manufacturing"}]}},
        {"title": "Scrum Master", "locationName": "Remote", "companyUrn": "urn:li:fs_miniCompany:1234",
         "timePeriod": {"endDate": {"month": 9, "year": 2022}, "startDate": {"month": 11, "year": 2019}}},
        "more positions...",
    ],
    "educations": [
        {"degreeName": "Bachelor's Degree", "fieldOfStudy": "Electrical Engineering",
         "schoolName": "University of Puerto Rico-Mayaguez",
         "timePeriod": {"endDate": {"year": 2011}, "startDate": {"year": 2005}}},
    ],
    "certifications": [{"name": "Security+ Certification", "authority": "CompTIA",
                        "timePeriod": {"startDate": {"month": 12, "year": 2016}}}],
    "skills": ["Jira", "Scrum"],
    "languages": [{"name": "English", "proficiency": "NATIVE_OR_BILINGUAL"}],
}


def test_curious_coder_item_converts_to_dev_fusion_shape():
    row = apify_scrape.convert_curious_coder_item(CC_ITEM)
    assert row["linkedinUrl"] == "https://www.linkedin.com/in/raulnquinones"
    assert row["fullName"] == "Raul Quinones"
    assert row["about"].startswith("IT Scrum Master")
    assert row["addressWithCountry"] == "San Diego, California, United States"
    assert row["addressWithoutCountry"] == "San Diego, California"
    assert row["experiences/0/companyName"] == "Tandem Diabetes Care"
    assert row["experiences/0/jobStartedOn"] == "10-2022"
    assert row["experiences/0/jobStillWorking"] is True
    assert row["experiences/0/companyIndustry"] == "Medical Equipment Manufacturing"
    assert row["experiences/0/companySize"] == "1001-5000"
    assert row["experiences/1/jobStartedOn"] == "11-2019" and row["experiences/1/jobEndedOn"] == "09-2022"
    assert row["experiences/1/jobStillWorking"] is False
    assert row["experiences/1/companyName"] == "1234"  # urn fallback, better than nothing
    assert "experiences/2/title" not in row  # the "more positions..." string is ignored
    assert row["educations/0/title"] == "University of Puerto Rico-Mayaguez"
    assert row["educations/0/subtitle"] == "Bachelor's Degree, Electrical Engineering"
    assert row["educations/0/period/startedOn"] == "2005" and row["educations/0/period/endedOn"] == "2011"
    assert row["skills/0/title"] == "Jira" and row["skills/1/title"] == "Scrum"
    assert row["licenseAndCertificates/0/title"] == "Security+ Certification"
    assert row["licenseAndCertificates/0/subtitle"] == "CompTIA"


def test_converted_row_parses_through_existing_enrichment():
    row = apify_scrape.convert_curious_coder_item(CC_ITEM)
    roles = ie.parse_roles_from_raw(row, {"id": 0, "raw_fields": row, "name": row["fullName"],
                                          "headline": row["headline"], "about": row["about"]})
    assert [(r.title, r.company) for r in roles] == [("Sr. Scrum Master I", "Tandem Diabetes Care"),
                                                     ("Scrum Master", "1234")]
    assert roles[0].start.year == 2022 and roles[0].start.month == 10
    assert roles[1].end.year == 2022 and roles[1].end.month == 9
    edu = ie.parse_education_from_raw(row)
    assert edu[0].college == "University of Puerto Rico-Mayaguez" and edu[0].start.year == 2005 and edu[0].end.year == 2011


def test_curious_coder_input_requires_cookie(monkeypatch):
    monkeypatch.delenv("APIFY_LINKEDIN_COOKIE", raising=False)
    monkeypatch.delenv("APIFY_LINKEDIN_COOKIE_FILE", raising=False)
    with pytest.raises(apify_scrape.ApifyScrapeError, match="LinkedIn session"):
        apify_scrape._curious_coder_input(["https://linkedin.com/in/x"])

    monkeypatch.setenv("APIFY_LINKEDIN_COOKIE", json.dumps([{"name": "li_at", "value": "test", "domain": ".linkedin.com"}]))
    monkeypatch.setenv("APIFY_PROXY_COUNTRY", "in")
    payload = apify_scrape._curious_coder_input(["https://linkedin.com/in/x"])
    assert payload["urls"] == ["https://linkedin.com/in/x"]
    assert payload["cookie"][0]["name"] == "li_at"
    assert payload["proxy"] == {"useApifyProxy": True, "apifyProxyCountry": "IN"}
    assert payload["userAgent"] and payload["scrapeCompany"] is False


def test_unknown_actor_is_rejected(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "t")
    monkeypatch.setenv("APIFY_LINKEDIN_ACTOR", "someone/other-scraper")
    with pytest.raises(apify_scrape.ApifyScrapeError, match="No adapter"):
        apify_scrape.scrape_profiles(["https://linkedin.com/in/x"])


def test_scrape_stamps_requested_url_when_actor_returns_other_slug(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "t")
    monkeypatch.setenv("APIFY_LINKEDIN_ACTOR", "curious_coder/linkedin-profile-scraper")
    monkeypatch.setenv("APIFY_LINKEDIN_COOKIE", json.dumps([{"name": "li_at", "value": "x"}]))

    class Resp:
        status_code = 200

        def json(self):
            return [dict(CC_ITEM, publicIdentifier="raul-quinones-7a1b2c3d")]

    calls = []
    monkeypatch.setattr(apify_scrape.requests, "post", lambda url, **kw: calls.append(kw["json"]) or Resp())
    out = apify_scrape.scrape_profiles(["https://www.linkedin.com/in/raulnquinones"])
    assert out["failed"] == [] and len(out["rows"]) == 1 and len(calls) == 1
    assert out["rows"][0]["linkedinUrl"] == "https://www.linkedin.com/in/raulnquinones"
    assert out["rows"][0]["requestedUrl"] == "https://www.linkedin.com/in/raulnquinones"


def test_supreme_coder_input_uses_request_list_objects():
    build_input, convert = apify_scrape.ACTOR_ADAPTERS["supreme_coder/linkedin-profile-scraper"]
    assert build_input(["https://linkedin.com/in/x"]) == {"urls": [{"url": "https://linkedin.com/in/x"}], "scrapeCompany": False}
    assert convert is apify_scrape.convert_auto
    assert apify_scrape.convert_auto({"fullName": "Flat", "experiences": [{"title": "T"}]})["experiences/0/title"] == "T"


def test_converter_handles_real_supreme_coder_quirks():
    item = {
        "firstName": "N", "lastName": "P S", "inputUrl": "https://www.linkedin.com/in/n-p-s-1",
        "publicIdentifier": "n-p-s-1", "geoLocationName": "Bengaluru, Karnataka, India", "geoCountryName": "India",
        "positions": [{"title": "AIML Engineer ", "locationName": "Bengaluru, Karnataka, India · On-site",
                       "employmentType": "Full-time", "timePeriod": {"startDate": {"month": 1, "year": 2026}, "endDate": None},
                       "company": {"url": "https://www.linkedin.com/company/14404611/", "name": "Rootent Technologies"}}],
        "educations": [{"schoolName": "KS Institute Of Technology", "degreeName": "Bachelor of Engineering - BE",
                        "fieldOfStudy": "Bachelor of Engineering - BE",
                        "timePeriod": {"startDate": {"year": 2020}, "endDate": {"year": 2024}}}],
        "skills": [{"name": "SIP Trunking", "endorsements": 0}],
        "certifications": [{"name": "AWS technical Essentials", "authority": "Udemy", "timePeriod": {}, "issuer": "Udemy"}],
        "projects": [{"title": "Automated Text Analysis", "description": "NLP tool"}],
    }
    row = apify_scrape.convert_auto(item)
    assert row["linkedinUrl"] == "https://www.linkedin.com/in/n-p-s-1"  # inputUrl wins over the slug
    assert row["addressWithCountry"] == "Bengaluru, Karnataka, India"
    assert row["addressWithoutCountry"] == "Bengaluru, Karnataka"
    assert row["experiences/0/companyName"] == "Rootent Technologies"
    assert row["experiences/0/jobLocation"] == "Bengaluru, Karnataka, India"
    assert row["experiences/0/workplaceType"] == "On-site"
    assert row["experiences/0/employmentType"] == "Full-time"
    assert row["experiences/0/jobStillWorking"] is True
    assert row["educations/0/subtitle"] == "Bachelor of Engineering - BE"
    assert row["skills/0/title"] == "SIP Trunking"
    assert row["licenseAndCertificates/0/subtitle"] == "Udemy"
    assert row["projects/0/title"] == "Automated Text Analysis"

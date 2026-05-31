"""Verified profile enrichment for LinkedIn-style recruiter uploads.

This module is intentionally additive to the existing upload path. It parses
wide Apify/LinkedIn-like rows already preserved in candidates.raw_fields, writes
structured role/education/company tenure records, and marks uncertain company
classification as Unknown instead of guessing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.db.connection import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = ROOT / "data" / "taxonomies"
COMPANY_CACHE_PATH = ROOT / "data" / "cache" / "company_cache.json"

COMMUNITY_TERMS = (
    "revgenius",
    "revgenuis",
    "community",
    "member",
    "membership",
    "mentor",
    "advisor",
    "volunteer",
)


def _load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.warning("Could not load %s", path, exc_info=True)
        return default


SALES_TAXONOMY: Dict[str, List[str]] = _load_json(TAXONOMY_DIR / "sales_taxonomy.json", {})
SEGMENT_SYNONYMS: Dict[str, List[str]] = _load_json(TAXONOMY_DIR / "segment_synonyms.json", {})
COMPANY_DETAILS_TAXONOMY: Dict[str, List[str]] = _load_json(
    TAXONOMY_DIR / "company_details_taxonomy.json", {}
)
GEOGRAPHY_COUNTRY_TO_REGION_MAP: Dict[str, str] = _load_json(TAXONOMY_DIR / "geography_map.json", {})
COMPANY_CACHE: Dict[str, Dict[str, Any]] = _load_json(COMPANY_CACHE_PATH, {})


@dataclass
class ParsedRole:
    index: int
    company: str
    title: str = ""
    start_raw: str = ""
    end_raw: str = ""
    details: str = ""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    duration_months: int = 0
    duration_unknown: bool = False
    function: str = "Unknown"
    function_confidence: str = "low"
    function_reason: str = ""
    product_service: str = "Unknown"
    industry: str = "Unknown"
    customer_segment: List[str] = field(default_factory=list)
    business_model: str = "Unknown"
    verification_status: str = "not_verified"
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ParsedEducation:
    index: int
    college: str
    degree: str = ""
    start_raw: str = ""
    end_raw: str = ""
    details: str = ""
    start: Optional[datetime] = None
    end: Optional[datetime] = None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_company_name(company: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", clean_text(company).lower()).strip()
    text = re.sub(
        r"\b(private|pvt|ltd|limited|inc|corp|corporation|llc|plc|company|co)\b",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _raw_get(raw: Dict[str, Any], *keys: str) -> str:
    lowered = {str(k).strip().lower(): k for k in raw.keys()}
    for key in keys:
        if key in raw:
            return clean_text(raw.get(key))
        actual = lowered.get(str(key).strip().lower())
        if actual is not None:
            return clean_text(raw.get(actual))
    return ""


def parse_profile_date(value: Any, *, default_current: bool = False) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return datetime.now(timezone.utc) if default_current else None
    lower = text.lower()
    if lower in {"present", "current", "now", "till date", "ongoing"}:
        return datetime.now(timezone.utc)

    normalized = re.sub(r"\s+", " ", text.replace(",", " ")).strip()
    for pattern in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%Y-%m",
        "%Y",
        "%b %Y",
        "%B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ):
        try:
            return datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        return datetime(int(year_match.group(0)), 1, 1, tzinfo=timezone.utc)
    return None


def months_between(start: Optional[datetime], end: Optional[datetime]) -> int:
    if not start:
        return 0
    end = end or datetime.now(timezone.utc)
    if end < start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day >= start.day:
        months += 1
    return max(months, 0)


def years_from_months(months: int) -> float:
    return round((months or 0) / 12.0, 2)


def _role_field_variants(base: str, idx: int) -> List[str]:
    if idx == 1:
        return [base, f"{base} ", f"{base}.0", f"{base} 1"]
    suffix = idx - 1
    return [
        f"{base}.{suffix}",
        f"{base} .{suffix}",
        f"{base} {idx}",
        f"{base}_{idx}",
        f"{base}{idx}",
    ]


def parse_roles_from_raw(raw_fields: Dict[str, Any], candidate: Optional[Dict[str, Any]] = None) -> List[ParsedRole]:
    candidate = candidate or {}
    roles: List[ParsedRole] = []
    for idx in range(1, 11):
        company = _raw_get(raw_fields, f"Company {idx} Name", f"Company {idx}", f"Company Name.{idx - 1}")
        title = _raw_get(raw_fields, *_role_field_variants("Title", idx))
        start_raw = _raw_get(raw_fields, *_role_field_variants("Start date", idx), *_role_field_variants("Start Date", idx))
        end_raw = _raw_get(raw_fields, *_role_field_variants("End Date", idx), *_role_field_variants("End date", idx))
        details = _raw_get(raw_fields, *_role_field_variants("Details", idx), *_role_field_variants("Details ", idx))

        if idx == 1:
            company = company or clean_text(raw_fields.get("import_company")) or clean_text(candidate.get("company_name"))
            title = title or clean_text(candidate.get("headline")) or clean_text(candidate.get("title"))

        if not company and not title and not start_raw and not end_raw and not details:
            continue
        if not company:
            continue

        start = parse_profile_date(start_raw)
        end = parse_profile_date(end_raw, default_current=idx == 1)
        duration = months_between(start, end)
        roles.append(
            ParsedRole(
                index=idx,
                company=company,
                title=title,
                start_raw=start_raw,
                end_raw=end_raw,
                details=details,
                start=start,
                end=end,
                duration_months=duration,
                duration_unknown=start is None,
            )
        )
    return roles


def parse_education_from_raw(raw_fields: Dict[str, Any]) -> List[ParsedEducation]:
    out: List[ParsedEducation] = []
    for idx in range(1, 4):
        college = _raw_get(raw_fields, f"Education {idx} - College Name", f"Education {idx} College Name")
        degree = _raw_get(raw_fields, *_role_field_variants("Degree Name", idx))
        edu_date_idx = 9 + idx
        start_raw = _raw_get(raw_fields, f"Start date.{edu_date_idx}", f"Start Date.{edu_date_idx}", f"Education {idx} Start date")
        end_raw = _raw_get(raw_fields, f"End Date.{edu_date_idx}", f"End date.{edu_date_idx}", f"Education {idx} End Date")
        details = _raw_get(raw_fields, f"Education {idx} Details", f"Details.{edu_date_idx}", f"Details .{edu_date_idx}")
        if not college and not degree:
            continue
        out.append(
            ParsedEducation(
                index=idx,
                college=college,
                degree=degree,
                start_raw=start_raw,
                end_raw=end_raw,
                details=details,
                start=parse_profile_date(start_raw),
                end=parse_profile_date(end_raw),
            )
        )
    return out


def is_community_role(role: ParsedRole) -> bool:
    combined = f"{role.title} {role.company} {role.details}".lower().replace("revgenuis", "revgenius")
    return any(term in combined for term in COMMUNITY_TERMS)


def classify_function(role: ParsedRole, *, use_llm: bool = False) -> Tuple[str, str, str]:
    text = f"{role.title} {role.details}".lower()
    matches: List[Tuple[str, int, str]] = []
    for label, terms in SALES_TAXONOMY.items():
        for term in terms:
            term_l = str(term).lower()
            if term_l and term_l in text:
                matches.append((label, len(term_l), term))
    if matches:
        label, _score, term = sorted(matches, key=lambda item: item[1], reverse=True)[0]
        return label, "high", f"Matched sales taxonomy term: {term}"
    if use_llm:
        llm_label = _llm_classify_function(role)
        if llm_label:
            return llm_label, "medium", "Classified from title/details by row-only LLM."
    return "Unknown", "low", "No taxonomy match; not guessed."


def _llm_classify_function(role: ParsedRole) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        from openai import OpenAI

        labels = list(SALES_TAXONOMY.keys())
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("IMPORT_ENRICHMENT_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify a sales role into one of these labels, or Unknown. "
                        f"Labels: {', '.join(labels)}. Return only the label."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Title: {role.title}\nCompany: {role.company}\nDetails: {role.details}",
                },
            ],
            timeout=20,
        )
        label = clean_text(resp.choices[0].message.content)
        return label if label in labels else ""
    except Exception:
        logger.warning("Function LLM classification failed", exc_info=True)
        return ""


def _cache_lookup_company(company: str) -> Dict[str, Any]:
    if company in COMPANY_CACHE:
        return COMPANY_CACHE[company]
    norm = normalize_company_name(company)
    for key, value in COMPANY_CACHE.items():
        if normalize_company_name(key) == norm:
            return value
    return {}


def _segments_from_text(text: str) -> List[str]:
    lower = (text or "").lower()
    found: List[str] = []
    for segment, terms in SEGMENT_SYNONYMS.items():
        if segment in lower or any(str(term).lower() in lower for term in terms):
            found.append(segment.upper() if segment == "smb" else segment.title())
    return sorted(set(found))


def _canonical_customer_segments(values: Any) -> List[str]:
    raw_values = values if isinstance(values, list) else [values] if values else []
    cleaned = [clean_text(value) for value in raw_values if clean_text(value)]
    canonical = _segments_from_text(" ".join(cleaned))
    return sorted(set(cleaned + canonical))


def _canonical_region(region: str) -> str:
    value = clean_text(region).lower()
    if value == "apac":
        return "APAC"
    if value == "emea":
        return "EMEA"
    if value == "latam":
        return "LATAM"
    if value == "americas":
        return "Americas"
    return clean_text(region).title()


def _geographies_from_text(text: str) -> List[str]:
    lower = (text or "").lower()
    found = set()
    for direct in ("apac", "emea", "latam", "americas"):
        if re.search(rf"\b{re.escape(direct)}\b", lower):
            found.add(_canonical_region(direct))
    for country, region in GEOGRAPHY_COUNTRY_TO_REGION_MAP.items():
        if re.search(rf"\b{re.escape(str(country).lower())}\b", lower):
            found.add(_canonical_region(region))
    return sorted(found)


def _functions_from_text(text: str) -> List[Dict[str, str]]:
    lower = (text or "").lower()
    found: Dict[str, str] = {}
    for label, terms in SALES_TAXONOMY.items():
        for term in terms:
            term_l = str(term).lower()
            if term_l and term_l in lower:
                found.setdefault(label, f"Matched sales taxonomy term in profile text: {term}")
    return [{"function": label, "reason": reason} for label, reason in sorted(found.items())]


def _claimed_experience_from_text(text: str) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|yr)\b", text or "", re.IGNORECASE):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        snippet = clean_text(text[start:end])
        if not snippet:
            continue
        claims.append(
            {
                "claimed_years": float(match.group(1)),
                "evidence_text": snippet,
                "segments": _segments_from_text(snippet),
                "geographies": _geographies_from_text(snippet),
                "functions": _functions_from_text(snippet),
                "verification_status": "profile_claim",
                "note": "Claimed duration from profile/about text; not used as verified tenure unless supported by dated roles.",
            }
        )
    return claims[:10]


def _profile_context_text(candidate: Dict[str, Any]) -> str:
    raw = candidate.get("raw_fields") or {}
    parts = [
        candidate.get("headline"),
        candidate.get("about"),
        candidate.get("city"),
        raw.get("headline"),
        raw.get("about"),
        raw.get("Bio"),
        raw.get("addressWithCountry"),
        raw.get("Skills"),
        raw.get("services"),
    ]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def extract_profile_claims(candidate: Dict[str, Any]) -> Dict[str, Any]:
    text = _profile_context_text(candidate)
    company_details = _company_details_from_taxonomy(text)
    return {
        "segments": _segments_from_text(text),
        "geographies": _geographies_from_text(text),
        "functions": _functions_from_text(text),
        "claimed_experience": _claimed_experience_from_text(text),
        "product_service": company_details.get("product_service") or "",
        "business_model": company_details.get("business_model") or "",
        "evidence_text": text[:1200],
        "verification_status": "row_context" if text else "not_available",
        "note": (
            "Profile/about claims are used as supporting evidence for segment, geography, "
            "function, and product-service. They do not override date-based tenure calculations."
        ),
    }


def _company_details_from_taxonomy(text: str) -> Dict[str, Any]:
    lower = (text or "").lower()
    out: Dict[str, Any] = {}
    if any(term in lower for term in COMPANY_DETAILS_TAXONOMY.get("saas", [])):
        out["product_service"] = "SaaS"
        out["business_model"] = "SaaS"
    elif any(term in lower for term in COMPANY_DETAILS_TAXONOMY.get("b2b", [])):
        out["business_model"] = "B2B"
    elif any(term in lower for term in COMPANY_DETAILS_TAXONOMY.get("b2c", [])):
        out["business_model"] = "B2C"
    return out


def classify_company(
    company: str,
    *,
    role_texts: Sequence[str] = (),
    db_details: Optional[Dict[str, Any]] = None,
    allow_web: bool = True,
) -> Dict[str, Any]:
    db_details = db_details or {}
    if db_details.get("product_service") or db_details.get("customer_segment"):
        return {
            "product_service": db_details.get("product_service") or "Unknown",
            "industry": db_details.get("industry") or db_details.get("product_service") or "Unknown",
            "customer_segment": _canonical_customer_segments(db_details.get("customer_segment") or []),
            "business_model": db_details.get("business_model") or "Unknown",
            "verification_status": "verified",
            "confidence": "high",
            "source": "db",
            "sources": [],
        }

    cached = _cache_lookup_company(company)
    if cached:
        return {
            "product_service": cached.get("product_service") or "Unknown",
            "industry": cached.get("industry") or cached.get("product_service") or "Unknown",
            "customer_segment": _canonical_customer_segments(cached.get("customer_segment") or []),
            "business_model": cached.get("business_model") or "Unknown",
            "verification_status": "verified",
            "confidence": "high",
            "source": "cache",
            "sources": [],
            "funding_stage": cached.get("funding_stage"),
            "revenue": cached.get("revenue"),
            "customer_presence": cached.get("customer_presence"),
            "culture_type": cached.get("culture_type"),
            "headquarters": cached.get("headquarters"),
        }

    combined = f"{company} {' '.join(role_texts)}"
    taxonomy_details = _company_details_from_taxonomy(combined)
    segments = _segments_from_text(combined)
    if taxonomy_details or segments:
        return {
            "product_service": taxonomy_details.get("product_service") or "Unknown",
            "industry": taxonomy_details.get("product_service") or "Unknown",
            "customer_segment": segments,
            "business_model": taxonomy_details.get("business_model") or "Unknown",
            "verification_status": "row_context",
            "confidence": "medium",
            "source": "taxonomy",
            "sources": [],
        }

    if allow_web:
        researched = _web_research_company(company)
        if researched:
            sources = researched.get("sources") or researched.get("source_urls") or []
            if sources:
                return {
                    "product_service": clean_text(researched.get("product_service")) or "Unknown",
                    "industry": clean_text(researched.get("industry")) or clean_text(researched.get("product_service")) or "Unknown",
                    "customer_segment": _canonical_customer_segments(researched.get("customer_segment") or []),
                    "business_model": clean_text(researched.get("business_model")) or "Unknown",
                    "verification_status": "verified",
                    "confidence": clean_text(researched.get("confidence")) or "medium",
                    "source": "web",
                    "sources": sources,
                }

    return {
        "product_service": "Unknown",
        "industry": "Unknown",
        "customer_segment": [],
        "business_model": "Unknown",
        "verification_status": "not_verified",
        "confidence": "low",
        "source": "unknown",
        "sources": [],
        "unknown_reason": "No cache, taxonomy, or directly sourced web result.",
    }


def _web_research_company(company: str) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            "Research this company and return JSON only with keys product_service, industry, "
            "customer_segment, business_model, confidence, sources. sources must be an array "
            "of objects with url/title/note and must directly support this exact company. "
            f"Company: {company}"
        )
        response = client.responses.create(
            model=os.getenv("IMPORT_ENRICHMENT_OPENAI_MODEL", "gpt-4o-mini"),
            tools=[{"type": os.getenv("IMPORT_ENRICHMENT_WEB_TOOL", "web_search_preview")}],
            input=prompt,
            timeout=75,
        )
        text = response.output_text or ""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.warning("Company web research failed for %s", company, exc_info=True)
        return {}


def merge_intervals(intervals: Iterable[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    ordered = sorted(intervals, key=lambda item: item[0])
    merged: List[Tuple[datetime, datetime]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def calculate_tenure_metrics(roles: Sequence[ParsedRole]) -> Dict[str, Any]:
    work_roles = [role for role in roles if not role.duration_unknown and role.start and role.end and not is_community_role(role)]
    company_windows: Dict[str, Dict[str, Any]] = {}
    for role in work_roles:
        norm = normalize_company_name(role.company)
        bucket = company_windows.setdefault(norm, {"company": role.company, "intervals": []})
        bucket["intervals"].append((role.start, role.end))

    company_years: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[datetime, datetime]] = []
    for item in company_windows.values():
        merged = merge_intervals(item["intervals"])
        months = sum(months_between(start, end) for start, end in merged)
        all_intervals.extend(merged)
        company_years.append({"company": item["company"], "months": months, "years": years_from_months(months)})

    merged_all = merge_intervals(all_intervals)
    total_months = sum(months_between(start, end) for start, end in merged_all)
    company_count = len(company_years)
    avg_months = int(round(total_months / company_count)) if company_count else 0
    current_job_months = 0
    if roles and roles[0].start:
        current_job_months = months_between(roles[0].start, roles[0].end or datetime.now(timezone.utc))

    return {
        "company_years": company_years,
        "total_experience_months": total_months,
        "total_experience_years": years_from_months(total_months),
        "avg_tenure_months": avg_months,
        "avg_tenure_years": years_from_months(avg_months),
        "current_job_months": current_job_months,
    }


def _jsonable_dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.date().isoformat() if dt else None


def build_enrichment_payload(
    *,
    roles: Sequence[ParsedRole],
    education: Sequence[ParsedEducation],
    metrics: Dict[str, Any],
    profile_claims: Optional[Dict[str, Any]] = None,
    errors: Sequence[str],
    contact_from_excel: bool,
) -> Dict[str, Any]:
    profile_claims = profile_claims or {}
    unknowns = [
        role.company
        for role in roles
        if role.verification_status in {"not_verified", "failed"} or role.product_service == "Unknown"
    ]
    verification_errors = list(errors)
    if not roles:
        verification_errors.append("No structured work history found in upload columns.")
    if metrics.get("total_experience_months", 0) < 0:
        verification_errors.append("Computed total experience was negative.")

    status = "failed" if verification_errors else ("passed_with_unknowns" if unknowns else "passed")
    return {
        "status": "completed" if status != "failed" else "completed_with_errors",
        "verification_status": status,
        "verification_errors": verification_errors,
        "contact_from_excel": contact_from_excel,
        "roles": [
            {
                "index": role.index,
                "company": role.company,
                "title": role.title,
                "start_date": _jsonable_dt(role.start),
                "end_date": _jsonable_dt(role.end),
                "start_raw": role.start_raw,
                "end_raw": role.end_raw,
                "duration_months": role.duration_months,
                "duration_years": years_from_months(role.duration_months),
                "duration_unknown": role.duration_unknown,
                "function": role.function,
                "function_confidence": role.function_confidence,
                "function_reason": role.function_reason,
                "product_service": role.product_service,
                "industry": role.industry,
                "customer_segment": role.customer_segment,
                "business_model": role.business_model,
                "verification_status": role.verification_status,
                "sources": role.sources,
            }
            for role in roles
        ],
        "education": [
            {
                "index": item.index,
                "college": item.college,
                "degree": item.degree,
                "start_date": _jsonable_dt(item.start),
                "end_date": _jsonable_dt(item.end),
                "start_raw": item.start_raw,
                "end_raw": item.end_raw,
                "details": item.details,
            }
            for item in education
        ],
        "metrics": metrics,
        "profile_claims": profile_claims,
        "sources": {
            role.company: role.sources
            for role in roles
            if role.sources
        },
        "unknown_companies": sorted(set(unknowns)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_candidates(candidate_ids: Sequence[int]) -> List[Dict[str, Any]]:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, first_name, last_name, headline, about, email,
                       COALESCE(mobile_phone, phone), raw_fields
                FROM candidates
                WHERE id = ANY(%s) AND COALESCE(is_archived, FALSE) = FALSE
                ORDER BY id
                """,
                (list(candidate_ids),),
            )
            rows = []
            for row in cur.fetchall():
                raw = row[8] or {}
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {}
                rows.append(
                    {
                        "id": row[0],
                        "name": row[1] or "",
                        "first_name": row[2] or "",
                        "last_name": row[3] or "",
                        "headline": row[4] or "",
                        "about": row[5] or "",
                        "email": row[6] or "",
                        "phone": row[7] or "",
                        "raw_fields": raw if isinstance(raw, dict) else {},
                    }
                )
            return rows
    finally:
        return_db_connection(conn)


def _load_db_company_details(company_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    names = sorted({clean_text(name) for name in company_names if clean_text(name)})
    if not names:
        return {}
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, funding_stage, revenue, business_model, product_service,
                       customer_segment, customer_presence, culture_type, headquarters
                FROM companies
                WHERE name = ANY(%s)
                """,
                (names,),
            )
            out: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                out[normalize_company_name(row[0])] = {
                    "name": row[0],
                    "funding_stage": row[1],
                    "revenue": row[2],
                    "business_model": row[3],
                    "product_service": row[4],
                    "customer_segment": row[5] or [],
                    "customer_presence": row[6] or [],
                    "culture_type": row[7],
                    "headquarters": row[8],
                }
            return out
    finally:
        return_db_connection(conn)


def _company_contexts_by_norm(candidates: Sequence[Dict[str, Any]], roles_by_candidate: Dict[int, Sequence[ParsedRole]]) -> Dict[str, List[str]]:
    contexts: Dict[str, List[str]] = {}
    for candidate in candidates:
        profile_text = _profile_context_text(candidate)
        for role in roles_by_candidate.get(candidate["id"], []):
            norm = normalize_company_name(role.company)
            contexts.setdefault(norm, [])
            contexts[norm].append(f"{role.title} {role.details}".strip())
            if profile_text:
                contexts[norm].append(profile_text)
    return contexts


def _persist_candidate_enrichment(
    candidate: Dict[str, Any],
    roles: Sequence[ParsedRole],
    education: Sequence[ParsedEducation],
    metrics: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Database connection failed while saving enrichment")
    try:
        with conn.cursor() as cur:
            candidate_id = int(candidate["id"])
            if roles:
                cur.execute("DELETE FROM roles WHERE candidate_id = %s", (candidate_id,))
                cur.execute("DELETE FROM company_years WHERE candidate_id = %s", (candidate_id,))
                cur.execute("DELETE FROM functional_experiences WHERE candidate_id = %s", (candidate_id,))
                cur.execute("DELETE FROM industry_experiences WHERE candidate_id = %s", (candidate_id,))
                cur.execute("DELETE FROM segment_experiences WHERE candidate_id = %s", (candidate_id,))
                cur.execute("DELETE FROM geography_experiences WHERE candidate_id = %s", (candidate_id,))
            if education:
                cur.execute("DELETE FROM education WHERE candidate_id = %s", (candidate_id,))

            company_ids: Dict[str, int] = {}
            for role in roles:
                norm = normalize_company_name(role.company)
                if norm in company_ids:
                    continue
                cur.execute(
                    """
                    INSERT INTO companies (
                        name, business_model, product_service, customer_segment, created_by, updated_at
                    ) VALUES (%s, %s, %s, %s, 'import_enrichment', NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        business_model = COALESCE(NULLIF(EXCLUDED.business_model, 'Unknown'), companies.business_model),
                        product_service = COALESCE(NULLIF(EXCLUDED.product_service, 'Unknown'), companies.product_service),
                        customer_segment = CASE
                            WHEN EXCLUDED.customer_segment IS NOT NULL AND array_length(EXCLUDED.customer_segment, 1) IS NOT NULL
                            THEN EXCLUDED.customer_segment ELSE companies.customer_segment END,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        role.company,
                        role.business_model,
                        role.product_service,
                        role.customer_segment or None,
                    ),
                )
                company_ids[norm] = int(cur.fetchone()[0])

            for role in roles:
                company_id = company_ids.get(normalize_company_name(role.company))
                if not company_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO roles (candidate_id, company_id, title, details, duration_years)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id,
                        company_id,
                        role.title,
                        role.details,
                        None if role.duration_unknown else years_from_months(role.duration_months),
                    ),
                )

            for item in education:
                cur.execute(
                    """
                    INSERT INTO education (candidate_id, college, degree, start_date, end_date, details)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id,
                        item.college,
                        item.degree,
                        item.start.date() if item.start else None,
                        item.end.date() if item.end else None,
                        item.details,
                    ),
                )

            for company_year in metrics.get("company_years") or []:
                cur.execute(
                    """
                    INSERT INTO company_years (candidate_id, company, years)
                    VALUES (%s, %s, %s)
                    """,
                    (candidate_id, company_year["company"], company_year["years"]),
                )

            func_roles = [role for role in roles if role.function != "Unknown"]
            profile_claims = payload.get("profile_claims") if isinstance(payload.get("profile_claims"), dict) else {}
            profile_functions = profile_claims.get("functions") if isinstance(profile_claims.get("functions"), list) else []
            if func_roles or profile_functions:
                cur.execute(
                    """
                    INSERT INTO functional_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id
                    """,
                    (candidate_id, 1, "Derived from upload title/details taxonomy."),
                )
                fx_id = int(cur.fetchone()[0])
                for role in func_roles:
                    cur.execute(
                        """
                        INSERT INTO functional_experience_roles
                          (functional_experience_id, company, activity_type, reason, duration_years)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            fx_id,
                            role.company,
                            role.function,
                            role.function_reason,
                            None if role.duration_unknown else years_from_months(role.duration_months),
                        ),
                    )
                for claim in profile_functions:
                    if not isinstance(claim, dict) or not claim.get("function"):
                        continue
                    cur.execute(
                        """
                        INSERT INTO functional_experience_roles
                          (functional_experience_id, company, activity_type, reason, duration_years)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            fx_id,
                            "Profile/About",
                            str(claim.get("function"))[:100],
                            str(claim.get("reason") or "Matched in profile/about text."),
                            None,
                        ),
                    )

            ind_roles = [role for role in roles if role.industry != "Unknown"]
            if ind_roles:
                cur.execute(
                    """
                    INSERT INTO industry_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id
                    """,
                    (candidate_id, 1, "Derived from verified company enrichment."),
                )
                ix_id = int(cur.fetchone()[0])
                for role in ind_roles:
                    cur.execute(
                        """
                        INSERT INTO industry_experience_roles
                          (industry_experience_id, company, industry, reason, duration_years)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            ix_id,
                            role.company,
                            role.industry[:100],
                            role.verification_status,
                            None if role.duration_unknown else years_from_months(role.duration_months),
                        ),
                    )

            seg_roles = [role for role in roles if role.customer_segment]
            profile_segments = profile_claims.get("segments") if isinstance(profile_claims.get("segments"), list) else []
            if seg_roles or profile_segments:
                cur.execute(
                    """
                    INSERT INTO segment_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id
                    """,
                    (candidate_id, 1, "Derived from verified company/customer segment enrichment."),
                )
                sx_id = int(cur.fetchone()[0])
                for role in seg_roles:
                    for segment in role.customer_segment:
                        cur.execute(
                            """
                            INSERT INTO segment_experience_roles
                              (segment_experience_id, company, segment, reason, duration_years)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                sx_id,
                                role.company,
                                str(segment)[:100],
                                role.verification_status,
                                None if role.duration_unknown else years_from_months(role.duration_months),
                            ),
                        )
                for segment in profile_segments:
                    cur.execute(
                        """
                        INSERT INTO segment_experience_roles
                          (segment_experience_id, company, segment, reason, duration_years)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            sx_id,
                            "Profile/About",
                            str(segment)[:100],
                            "Matched in profile/about/services text.",
                            None,
                        ),
                    )

            profile_geographies = profile_claims.get("geographies") if isinstance(profile_claims.get("geographies"), list) else []
            if profile_geographies:
                cur.execute(
                    """
                    INSERT INTO geography_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id
                    """,
                    (candidate_id, 1, "Derived from profile/about/location text."),
                )
                gx_id = int(cur.fetchone()[0])
                for region in profile_geographies:
                    cur.execute(
                        """
                        INSERT INTO geography_experience_regions (geography_experience_id, region)
                        VALUES (%s, %s)
                        """,
                        (gx_id, str(region)[:100]),
                    )

            raw = candidate.get("raw_fields") or {}
            raw["enrichment"] = payload
            current_product = roles[0].product_service if roles else ""
            if current_product and current_product != "Unknown":
                raw.setdefault("extracted_industry", current_product)
            if _raw_get(raw, "Skills"):
                cur.execute("UPDATE candidates SET skills = COALESCE(skills, %s) WHERE id = %s", (_raw_get(raw, "Skills"), candidate_id))
            if _raw_get(raw, "Licenses and certifications"):
                cur.execute(
                    "UPDATE candidates SET licenses_and_certifications = COALESCE(licenses_and_certifications, %s) WHERE id = %s",
                    (_raw_get(raw, "Licenses and certifications"), candidate_id),
                )
            cur.execute(
                """
                UPDATE candidates
                SET total_experience_years = COALESCE(%s, total_experience_years),
                    avg_years_in_company = COALESCE(%s, avg_years_in_company),
                    raw_fields = %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    metrics.get("total_experience_years") or None,
                    metrics.get("avg_tenure_years") or None,
                    json.dumps(raw, default=str),
                    candidate_id,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_connection(conn)


def enrich_candidate_profiles(candidate_ids: Sequence[int], *, allow_web: bool = True) -> Dict[str, Any]:
    ids = sorted({int(item) for item in candidate_ids if item is not None})
    if not ids:
        return {"processed": 0, "failed": 0, "errors": []}
    candidates = _fetch_candidates(ids)
    all_roles_by_candidate = {
        candidate["id"]: parse_roles_from_raw(candidate.get("raw_fields") or {}, candidate)
        for candidate in candidates
    }
    company_contexts = _company_contexts_by_norm(candidates, all_roles_by_candidate)
    unique_companies = {
        role.company
        for roles in all_roles_by_candidate.values()
        for role in roles
        if clean_text(role.company)
    }
    db_company_details = _load_db_company_details(unique_companies)
    company_classification: Dict[str, Dict[str, Any]] = {}
    for company in sorted(unique_companies):
        norm = normalize_company_name(company)
        role_texts = [
            text for text in company_contexts.get(norm, []) if text
        ]
        company_classification[norm] = classify_company(
            company,
            role_texts=role_texts,
            db_details=db_company_details.get(norm),
            allow_web=allow_web,
        )
        time.sleep(float(os.getenv("IMPORT_ENRICHMENT_COMPANY_DELAY_SECONDS", "0")))

    processed = failed = 0
    errors: List[str] = []
    for candidate in candidates:
        try:
            raw = candidate.get("raw_fields") or {}
            roles = all_roles_by_candidate.get(candidate["id"], [])
            for role in roles:
                role.function, role.function_confidence, role.function_reason = classify_function(role)
                details = company_classification.get(normalize_company_name(role.company), {})
                role.product_service = details.get("product_service") or "Unknown"
                role.industry = details.get("industry") or "Unknown"
                role.customer_segment = list(details.get("customer_segment") or [])
                role.business_model = details.get("business_model") or "Unknown"
                role.verification_status = details.get("verification_status") or "not_verified"
                role.sources = list(details.get("sources") or [])
            education = parse_education_from_raw(raw)
            metrics = calculate_tenure_metrics(roles)
            profile_claims = extract_profile_claims(candidate)
            contact_from_excel = bool(candidate.get("email") or candidate.get("phone"))
            payload = build_enrichment_payload(
                roles=roles,
                education=education,
                metrics=metrics,
                profile_claims=profile_claims,
                errors=[],
                contact_from_excel=contact_from_excel,
            )
            _persist_candidate_enrichment(candidate, roles, education, metrics, payload)
            processed += 1
        except Exception as exc:
            failed += 1
            msg = f"candidate {candidate.get('id')}: {exc}"
            errors.append(msg)
            logger.exception("Import enrichment failed for candidate %s", candidate.get("id"))
    return {"processed": processed, "failed": failed, "errors": errors}

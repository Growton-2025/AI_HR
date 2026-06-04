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

from psycopg2.extras import execute_values

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
    source_industry: str = ""
    source_company_size: str = ""
    source_location: str = ""
    source_website: str = ""
    duration_raw: str = ""
    duration_source: str = ""
    source_headers: Dict[str, str] = field(default_factory=dict)


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
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y",
        "%m-%Y",
        "%m/%Y",
        "%Y-%m",
        "%Y/%m",
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


def parse_duration_months(value: Any) -> int:
    text = clean_text(value).lower()
    if not text:
        return 0
    number_match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not number_match:
        return 0
    amount = float(number_match.group(1))
    if any(unit in text for unit in ("month", "months", "mo", "mos")):
        return int(round(amount))
    if any(unit in text for unit in ("year", "years", "yr", "yrs")):
        return int(round(amount * 12))
    # Bare numeric duration columns in these imports are more commonly months
    # when attached to a role; totalExperienceYears is handled separately.
    return int(round(amount))


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


ROLE_FIELD_KIND_ALIASES: Dict[str, Tuple[str, ...]] = {
    "company": (
        "company",
        "company name",
        "companyname",
        "organization",
        "organisation",
        "employer",
    ),
    "title": ("title", "role", "designation", "job title", "jobtitle"),
    "start_raw": ("start", "start date", "started on", "startedon", "job started on", "jobstartedon", "from"),
    "end_raw": ("end", "end date", "ended on", "endedon", "job ended on", "jobendedon", "to"),
    "details": ("details", "description", "job description", "jobdescription"),
    "duration_raw": ("duration", "duration months", "duration_months", "tenure", "time spent"),
    "source_industry": ("industry", "company industry", "companyindustry"),
    "source_company_size": ("size", "company size", "companysize"),
    "source_location": ("location", "job location", "joblocation", "job location country", "joblocationcountry"),
    "source_website": ("website", "company website", "companywebsite"),
}


def _normalized_header_token(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _role_field_kind(label: str) -> str:
    token = _normalized_header_token(label)
    if not token:
        return ""
    token = re.sub(r"^(exp|experience|experiences|job|role)\s+", "", token).strip()
    for kind, aliases in ROLE_FIELD_KIND_ALIASES.items():
        if token in aliases:
            return kind
    return ""


def _coerce_role_index(raw_idx: str) -> int:
    idx = int(raw_idx)
    # Most indexed export paths are zero-based; user-facing numbered columns are often one-based.
    return idx + 1 if idx == 0 else idx


def _compact_role_token(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _split_pandas_suffix(header: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*)\.(\d+)$", str(header or "").strip())
    if not match:
        return str(header or ""), None
    return match.group(1), int(match.group(2))


def _role_field_from_header(header: str) -> Tuple[Optional[int], str]:
    text = str(header or "").strip()
    if not text:
        return None, ""

    # Guard against education, skill, and other non-role fields
    token_lower = text.lower()
    if any(term in token_lower for term in (
        "education", "school", "college", "university", "degree", "academic", "course", 
        "certif", "skill", "language", "project", "award", "interest", "patent", "publication"
    )):
        return None, ""

    apify_match = re.match(r"^experiences/(\d+)/(.+)$", text, flags=re.IGNORECASE)
    if apify_match:
        apify_idx = int(apify_match.group(1))
        tail, suffix = _split_pandas_suffix(apify_match.group(2))
        kind = _role_field_kind(tail)
        if not kind:
            return None, ""
        # Some flattened exports accidentally repeat experiences/0/title for
        # later roles. Pandas turns those into experiences/0/title.1, .2, ...
        # Treat those suffixes as the role index only when the base path is 0.
        idx = suffix + 1 if apify_idx == 0 and suffix and suffix > 0 else apify_idx + 1
        return idx, kind

    base, suffix = _split_pandas_suffix(text)
    if suffix is not None:
        kind = _role_field_kind(base)
        if kind:
            return suffix + 1, kind

    token = _normalized_header_token(text)
    indexed_prefix = re.match(r"^(exp|experience|experiences|job|role)\s+(\d+)\s+(.+)$", token)
    if indexed_prefix:
        kind = _role_field_kind(indexed_prefix.group(3))
        if kind:
            return _coerce_role_index(indexed_prefix.group(2)), kind

    indexed_infix = re.match(r"^(.+?)\s+(\d+)\s+(.+)$", token)
    if indexed_infix:
        left_kind = _role_field_kind(indexed_infix.group(1))
        right_kind = _role_field_kind(indexed_infix.group(3))
        kind = right_kind or left_kind
        if kind:
            return _coerce_role_index(indexed_infix.group(2)), kind

    company_match = re.match(r"^(company|organization|organisation|employer)\s+(\d+)(?:\s+name)?$", token)
    if company_match:
        return _coerce_role_index(company_match.group(2)), "company"
    company_name_match = re.match(r"^company\s+name\s+(\d+)$", token)
    if company_name_match:
        return _coerce_role_index(company_name_match.group(1)), "company"

    trailing_idx_match = re.match(r"^(.+?)\s+(\d+)$", token)
    if trailing_idx_match:
        kind = _role_field_kind(trailing_idx_match.group(1))
        if kind:
            return _coerce_role_index(trailing_idx_match.group(2)), kind

    compact_match = re.match(
        r"^(exp|experience|experiences|job|role)(\d+)(company|companyname|title|role|start|startdate|startedon|end|enddate|endedon|duration|details|description|industry|location|website|size)$",
        re.sub(r"[^a-z0-9]+", "", text.lower()),
    )
    if compact_match:
        kind = _role_field_kind(_compact_role_token(compact_match.group(3)))
        if kind:
            return _coerce_role_index(compact_match.group(2)), kind

    kind = _role_field_kind(token)
    if kind:
        return 1, kind
    return None, ""


def _dynamic_role_groups(raw_fields: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    groups: Dict[int, Dict[str, str]] = {}
    for header, value in (raw_fields or {}).items():
        if header == "imported_extra_fields":
            continue
        clean_value = clean_text(value)
        if not clean_value:
            continue
        idx, kind = _role_field_from_header(str(header))
        if not idx or not kind:
            continue
        group = groups.setdefault(idx, {})
        group.setdefault(kind, clean_value)
        source_headers = group.setdefault("__source_headers", {})
        if isinstance(source_headers, dict):
            source_headers.setdefault(kind, str(header))
    return groups


def parse_roles_from_raw(raw_fields: Dict[str, Any], candidate: Optional[Dict[str, Any]] = None) -> List[ParsedRole]:
    candidate = candidate or {}
    roles: List[ParsedRole] = []
    groups = _dynamic_role_groups(raw_fields)
    for idx in sorted(groups):
        group = groups[idx]
        company = group.get("company", "")
        title = group.get("title", "")
        start_raw = group.get("start_raw", "")
        end_raw = group.get("end_raw", "")
        details = group.get("details", "")
        duration_raw = group.get("duration_raw", "")
        source_industry = group.get("source_industry", "")
        source_company_size = group.get("source_company_size", "")
        source_location = group.get("source_location", "")
        source_website = group.get("source_website", "")
        source_headers = group.get("__source_headers", {})

        if idx == 1:
            company = company or clean_text(raw_fields.get("import_company")) or clean_text(candidate.get("company_name"))
            # Bug 5 fix: prefer the structured job title from experiences/0/title over
            # the LinkedIn tagline (headline). The headline is a marketing blurb, not a
            # job title. We fall back to headline/title only when no structured title exists.
            title = title or clean_text(candidate.get("title")) or clean_text(candidate.get("headline"))

        if not company and not title and not start_raw and not end_raw and not details:
            continue
        if not company:
            continue

        start = parse_profile_date(start_raw)
        end = parse_profile_date(end_raw, default_current=idx == 1)
        duration_from_dates = months_between(start, end)
        duration_from_field = parse_duration_months(duration_raw)
        duration = duration_from_dates or duration_from_field
        duration_source = "date_range" if duration_from_dates else ("duration_field" if duration_from_field else "")
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
                duration_unknown=duration == 0,
                source_industry=source_industry,
                source_company_size=source_company_size,
                source_location=source_location,
                source_website=source_website,
                duration_raw=duration_raw,
                duration_source=duration_source,
                source_headers=source_headers if isinstance(source_headers, dict) else {},
            )
        )
    return roles


def parse_education_from_raw(raw_fields: Dict[str, Any]) -> List[ParsedEducation]:
    out: List[ParsedEducation] = []

    # 1. Parse apify-style (educations/0/title etc.)
    apify_groups = {}
    for header, value in (raw_fields or {}).items():
        clean_value = clean_text(value)
        if not clean_value:
            continue
        match = re.match(r"^educations/(\d+)/(.+)$", str(header), flags=re.IGNORECASE)
        if match:
            idx = int(match.group(1)) + 1
            field = match.group(2).lower()
            group = apify_groups.setdefault(idx, {})
            if "title" in field:
                group["college"] = clean_value
            elif "subtitle" in field:
                group["degree"] = clean_value
                group["details"] = clean_value
            elif "start" in field:
                group["start_raw"] = clean_value
            elif "end" in field:
                group["end_raw"] = clean_value
            elif "description" in field or "details" in field:
                group["details"] = clean_value

    if apify_groups:
        for idx in sorted(apify_groups.keys()):
            g = apify_groups[idx]
            college = g.get("college", "")
            degree = g.get("degree", "")
            start_raw = g.get("start_raw", "")
            end_raw = g.get("end_raw", "")
            details = g.get("details", "")
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

    # 2. Fall back to legacy Excel parser
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
    if re.search(r"\b(member|mentor|volunteer)\b", combined):
        return True
    return any(
        re.search(rf"\b{re.escape(term)}\b", combined)
        for term in COMMUNITY_TERMS
        if term not in {"member", "mentor", "volunteer"}
    )


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
    sales_development_terms = (
        "outbound",
        "cold call",
        "cold calling",
        "sql",
        "mql",
        "qualified meeting",
        "discovery call",
        "prospecting",
        "pipeline",
        "lead generation",
        "sales navigator",
        "apollo",
        "zoominfo",
    )
    for term in sales_development_terms:
        if term in text:
            return "Sales Development", "medium", f"Matched sales development context term: {term}"
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
    imported = raw.get("imported_extra_fields") if isinstance(raw.get("imported_extra_fields"), dict) else {}
    imported_values = [
        item.get("value")
        for item in imported.values()
        if isinstance(item, dict) and clean_text(item.get("value"))
    ]
    parts = [
        candidate.get("headline"),
        candidate.get("about"),
        candidate.get("city"),
        raw.get("headline"),
        raw.get("about"),
        raw.get("Bio"),
        raw.get("Recruiter Summary"),
        raw.get("Summary"),
        raw.get("Summary(Double Tap)"),
        raw.get("Experience"),
        raw.get("Focused Geography"),
        raw.get("Focused Geog"),
        raw.get("Outbound Exp"),
        raw.get("Targets"),
        raw.get("addressWithCountry"),
        raw.get("addressCountryOnly"),
        raw.get("Skills"),
        raw.get("services"),
        *imported_values,
    ]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def _team_management_from_text(text: str) -> Dict[str, int]:
    if not text or not text.strip():
        return {"max_people_managed": 0, "years_team_management": 0}
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = (
            "Analyze the following resume/profile text and extract team management metrics. "
            "Return ONLY a JSON object with exactly these two integer keys:\n"
            "1. 'max_people_managed': The maximum number of direct or indirect reports the person has managed. If not mentioned, return 0.\n"
            "2. 'years_team_management': The total number of years they have spent in a management role leading people. If not mentioned, return 0.\n"
            "Be highly accurate. Do not hallucinate numbers. If the text says 'managed a team of 15', max_people_managed is 15. "
            "If it says 'led a team for 3 years', years_team_management is 3.\n\n"
            f"Profile Text:\n{text[:12000]}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content or "{}"
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(content[start:end+1])
            return {
                "max_people_managed": int(parsed.get("max_people_managed", 0)),
                "years_team_management": int(parsed.get("years_team_management", 0))
            }
    except Exception as e:
        logger.warning(f"LLM team management extraction failed: {e}")
        
    return {"max_people_managed": 0, "years_team_management": 0}

def extract_profile_claims(candidate: Dict[str, Any]) -> Dict[str, Any]:
    text = _profile_context_text(candidate)
    company_details = _company_details_from_taxonomy(text)
    
    # Dump the ENTIRE raw dictionary so not a single word is missed
    raw_dict = candidate.get("raw_fields") or {}
    full_raw_text = json.dumps(raw_dict, default=str)
    team_metrics = _team_management_from_text(full_raw_text)
    return {
        "segments": _segments_from_text(text),
        "geographies": _geographies_from_text(text),
        "functions": _functions_from_text(text),
        "claimed_experience": _claimed_experience_from_text(text),
        "product_service": company_details.get("product_service") or "",
        "business_model": company_details.get("business_model") or "",
        "max_people_managed": team_metrics["max_people_managed"],
        "years_team_management": team_metrics["years_team_management"],
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


def _details_from_source_industry(industry: str) -> Dict[str, Any]:
    value = clean_text(industry)
    if not value:
        return {}
    taxonomy = _company_details_from_taxonomy(value)
    return {
        "product_service": taxonomy.get("product_service") or value,
        "industry": value,
        "business_model": taxonomy.get("business_model") or "Unknown",
        "verification_status": "row_source",
        "confidence": "medium",
        "source": "apify_company_industry",
        "sources": [],
    }


def _source_industry_details_for_company(
    norm: str,
    roles_by_candidate: Dict[int, Sequence[ParsedRole]],
) -> Dict[str, Any]:
    for roles in roles_by_candidate.values():
        for role in roles:
            if normalize_company_name(role.company) == norm and clean_text(role.source_industry):
                return _details_from_source_industry(role.source_industry)
    return {}


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


def calculate_tenure_metrics(
    roles: Sequence[ParsedRole],
    *,
    raw_total_exp_years: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute date-verified tenure metrics from parsed roles.

    ``raw_total_exp_years`` may be supplied from a CSV column such as
    ``totalExperienceYears``.  It is used as a fallback when the date-derived
    total is lower (e.g. because only a subset of roles had parseable dates).

    Bug-2 fix: total experience is computed by merging ALL role intervals across
    ALL companies before summing, which correctly handles the common
    vendor/contractor pattern where two roles (e.g. "Google" via "Marketstar")
    share identical start dates and both have no end date.  Without this merge
    step, each company's open-ended span was counted independently, inflating
    the total.
    """
    work_roles = [role for role in roles if not role.duration_unknown and not is_community_role(role)]
    current_work_role = next((role for role in roles if not is_community_role(role)), None)
    current_company_norm = normalize_company_name(current_work_role.company) if current_work_role else ""
    company_windows: Dict[str, Dict[str, Any]] = {}
    for role in work_roles:
        norm = normalize_company_name(role.company)
        bucket = company_windows.setdefault(
            norm,
            {
                "company": role.company,
                "normalized_company": norm,
                "intervals": [],
                "undated_months": 0,
                "titles": [],
                "functions": [],
                "industries": [],
                "product_services": [],
                "segments": [],
                "geographies": [],
                "roles": [],
                "verification_statuses": [],
            },
        )
        if role.start and role.end:
            bucket["intervals"].append((role.start, role.end))
        elif role.duration_months:
            bucket["undated_months"] += role.duration_months
        if role.title:
            bucket["titles"].append(role.title)
        if role.function and role.function != "Unknown":
            bucket["functions"].append(role.function)
        if role.industry and role.industry != "Unknown":
            bucket["industries"].append(role.industry)
        if role.product_service and role.product_service != "Unknown":
            bucket["product_services"].append(role.product_service)
        bucket["segments"].extend(role.customer_segment or [])
        bucket["geographies"].extend(_geographies_from_text(role.source_location))
        if role.verification_status:
            bucket["verification_statuses"].append(role.verification_status)
        bucket["roles"].append(
            {
                "index": role.index,
                "title": role.title,
                "start_date": _jsonable_dt(role.start),
                "end_date": _jsonable_dt(role.end),
                "duration_months": role.duration_months,
                "duration_years": years_from_months(role.duration_months),
                "duration_source": role.duration_source,
            }
        )

    company_years: List[Dict[str, Any]] = []
    company_tenures: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[datetime, datetime]] = []
    for item in company_windows.values():
        merged = merge_intervals(item["intervals"])
        dated_months = sum(months_between(start, end) for start, end in merged)
        months = dated_months + int(item.get("undated_months") or 0)
        all_intervals.extend(merged)
        company_years.append({"company": item["company"], "months": months, "years": years_from_months(months)})
        company_tenures.append(
            {
                "company": item["company"],
                "normalized_company": item["normalized_company"],
                "months": months,
                "years": years_from_months(months),
                "is_current_company": item["normalized_company"] == current_company_norm,
                "dated_months": dated_months,
                "undated_duration_months": int(item.get("undated_months") or 0),
                "date_windows": [
                    {
                        "start_date": _jsonable_dt(start),
                        "end_date": _jsonable_dt(end),
                        "months": months_between(start, end),
                    }
                    for start, end in merged
                ],
                "titles": sorted(set(item.get("titles") or [])),
                "functions": sorted(set(item.get("functions") or [])),
                "industries": sorted(set(item.get("industries") or [])),
                "product_services": sorted(set(item.get("product_services") or [])),
                "segments": sorted(set(item.get("segments") or [])),
                "geographies": sorted(set(item.get("geographies") or [])),
                "roles": item.get("roles") or [],
                "verification_status": (
                    "verified"
                    if "verified" in (item.get("verification_statuses") or [])
                    else (item.get("verification_statuses") or ["not_verified"])[0]
                ),
            }
        )

    # Bug-2 fix: merge ALL intervals (across companies) before summing so that
    # concurrent / vendor-overlap spans are counted only once in the total.
    merged_all = merge_intervals(all_intervals)
    undated_total_months = sum(int(item.get("undated_months") or 0) for item in company_windows.values())
    total_months = sum(months_between(start, end) for start, end in merged_all) + undated_total_months
    date_total_years = years_from_months(total_months)

    # Bug-3 fix: if the date-derived total is less than the raw CSV value
    # (e.g. because early roles had no parseable dates), prefer the raw value.
    total_experience_years = date_total_years
    if raw_total_exp_years is not None and raw_total_exp_years > (date_total_years or 0):
        total_experience_years = round(raw_total_exp_years, 2)
        total_months = int(round(raw_total_exp_years * 12))

    company_count = len(company_years)
    completed_company_tenures = [
        item for item in company_tenures
        if not item.get("is_current_company")
    ]
    completed_company_count = len(completed_company_tenures)
    completed_company_months = sum(int(item.get("months") or 0) for item in completed_company_tenures)
    avg_months = (
        int(round(completed_company_months / completed_company_count))
        if completed_company_count
        else 0
    )
    current_job_months = 0
    if current_work_role:
        if current_work_role.start:
            current_job_months = months_between(current_work_role.start, current_work_role.end or datetime.now(timezone.utc))
        else:
            current_job_months = current_work_role.duration_months

    role_tenures = [
        {
            "index": role.index,
            "company": role.company,
            "normalized_company": normalize_company_name(role.company),
            "title": role.title,
            "start_date": _jsonable_dt(role.start),
            "end_date": _jsonable_dt(role.end),
            "start_raw": role.start_raw,
            "end_raw": role.end_raw,
            "duration_raw": role.duration_raw,
            "duration_months": role.duration_months,
            "duration_years": years_from_months(role.duration_months),
            "duration_source": role.duration_source,
            "duration_unknown": role.duration_unknown,
            "function": role.function,
            "industry": role.industry,
            "product_service": role.product_service,
            "customer_segment": role.customer_segment,
            "business_model": role.business_model,
            "geographies": _geographies_from_text(role.source_location),
            "source_industry": role.source_industry,
            "source_company_size": role.source_company_size,
            "source_location": role.source_location,
            "source_website": role.source_website,
            "verification_status": role.verification_status,
            "source_headers": role.source_headers,
        }
        for role in roles
    ]

    return {
        "company_years": company_years,
        "company_tenures": company_tenures,
        "role_tenures": role_tenures,
        "total_experience_months": total_months,
        "total_experience_years": total_experience_years,
        "unique_company_count": company_count,
        "avg_tenure_months": avg_months,
        "avg_tenure_years": years_from_months(avg_months),
        "completed_company_count": completed_company_count,
        "completed_company_months": completed_company_months,
        "completed_company_tenures": completed_company_tenures,
        "current_company": current_work_role.company if current_work_role else "",
        "current_job_months": current_job_months,
        "date_derived_total_years": date_total_years,
        "undated_duration_months": undated_total_months,
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
    claimed_years = [
        float(item.get("claimed_years"))
        for item in profile_claims.get("claimed_experience", [])
        if isinstance(item, dict) and item.get("claimed_years") is not None
    ]
    claimed_total_years = max(claimed_years) if claimed_years else None
    dated_total_years = metrics.get("total_experience_years")
    experience_mismatch = (
        claimed_total_years is not None
        and dated_total_years is not None
        and abs(float(claimed_total_years) - float(dated_total_years)) >= 0.5
    )

    status = "failed" if verification_errors else ("passed_with_unknowns" if unknowns else "passed")
    return {
        "status": "completed" if status != "failed" else "completed_with_errors",
        "verification_status": status,
        "verification_errors": verification_errors,
        "contact_from_excel": contact_from_excel,
        "profile_claims": profile_claims,
        "roles": [
            {
                "index": role.index,
                "company": role.company,
                "title": role.title,
                "start_date": _jsonable_dt(role.start),
                "end_date": _jsonable_dt(role.end),
                "start_raw": role.start_raw,
                "end_raw": role.end_raw,
                "duration_raw": role.duration_raw,
                "duration_months": role.duration_months,
                "duration_years": years_from_months(role.duration_months),
                "duration_source": role.duration_source,
                "duration_unknown": role.duration_unknown,
                "function": role.function,
                "function_confidence": role.function_confidence,
                "function_reason": role.function_reason,
                "product_service": role.product_service,
                "industry": role.industry,
                "customer_segment": role.customer_segment,
                "business_model": role.business_model,
                "source_industry": role.source_industry,
                "source_company_size": role.source_company_size,
                "source_location": role.source_location,
                "source_website": role.source_website,
                "verification_status": role.verification_status,
                "sources": role.sources,
                "source_headers": role.source_headers,
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
        "claimed_vs_dated_experience": {
            "claimed_total_years": claimed_total_years,
            "dated_total_years": dated_total_years,
            "mismatch": bool(experience_mismatch),
            "note": "Claimed experience is kept separate from date-verified tenure.",
        },
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
            contexts[norm].append(
                f"{role.title} {role.details} {role.source_industry} {role.source_location}".strip()
            )
            if profile_text:
                contexts[norm].append(profile_text)
    return contexts


def _persist_candidate_enrichment(
    candidate: Dict[str, Any],
    roles: Sequence[ParsedRole],
    education: Sequence[ParsedEducation],
    metrics: Dict[str, Any],
    payload: Dict[str, Any],
    company_ids: Optional[Dict[str, int]] = None,
) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Database connection failed while saving enrichment")
    try:
        with conn.cursor() as cur:
            candidate_id = int(candidate["id"])
            cur.execute("DELETE FROM roles WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM company_years WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM functional_experiences WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM industry_experiences WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM segment_experiences WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM geography_experiences WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM education WHERE candidate_id = %s", (candidate_id,))
            cur.execute("DELETE FROM titles_held WHERE candidate_id = %s", (candidate_id,))

            if company_ids is None:
                company_ids = {}
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
                        json.dumps({
                            "company": role.company,
                            "start_date": role.start.isoformat() if role.start else None,
                            "end_date": role.end.isoformat() if role.end else None,
                            "details": role.details,
                            "duration_months": role.duration_months
                        }, default=str),
                        None if role.duration_unknown else years_from_months(role.duration_months),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO titles_held (candidate_id, title, company, start_date, end_date)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id,
                        role.title,
                        role.company,
                        role.start.date() if role.start else None,
                        role.end.date() if role.end else None,
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
            role_geographies = sorted(
                {
                    region
                    for role in roles
                    for region in _geographies_from_text(role.source_location)
                }
            )
            all_geographies = sorted({*profile_geographies, *role_geographies})
            if all_geographies:
                cur.execute(
                    """
                    INSERT INTO geography_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id
                    """,
                    (candidate_id, 1, "Derived from profile/about/location and role-location text."),
                )
                gx_id = int(cur.fetchone()[0])
                for region in all_geographies:
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

            # Bug-3 fix: prefer date-derived total but use the CSV's
            # totalExperienceYears as an authoritative fallback when the
            # date-derived figure is 0 or None (e.g. all roles lacked dates).
            best_total_years = metrics.get("total_experience_years") or None
            best_avg_years = metrics.get("avg_tenure_years") or None

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
                    best_total_years,
                    best_avg_years,
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
    started = time.perf_counter()
    ids = sorted({int(item) for item in candidate_ids if item is not None})
    if not ids:
        return {"processed": 0, "failed": 0, "errors": []}
    candidates = _fetch_candidates(ids)
    parse_started = time.perf_counter()
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

    def classify_single_company(company: str) -> Tuple[str, Dict[str, Any]]:
        norm = normalize_company_name(company)
        role_texts = [
            text for text in company_contexts.get(norm, []) if text
        ]
        local_details = classify_company(
            company,
            role_texts=role_texts,
            db_details=db_company_details.get(norm),
            allow_web=False,
        )
        if local_details.get("product_service") == "Unknown" and not local_details.get("customer_segment"):
            source_details = _source_industry_details_for_company(norm, all_roles_by_candidate)
            if source_details:
                local_details = {**local_details, **source_details}
        if local_details.get("product_service") != "Unknown" or local_details.get("customer_segment"):
            return norm, local_details

        web_details = classify_company(
            company,
            role_texts=role_texts,
            db_details=db_company_details.get(norm),
            allow_web=allow_web,
        )
        delay = float(os.getenv("IMPORT_ENRICHMENT_COMPANY_DELAY_SECONDS", "0"))
        if delay > 0:
            time.sleep(delay)
        return norm, web_details

    from concurrent.futures import ThreadPoolExecutor, as_completed
    company_workers = int(os.getenv("IMPORT_ENRICHMENT_COMPANY_WORKERS", "10"))
    if unique_companies:
        classify_started = time.perf_counter()
        logger.info(
            "Starting concurrent company classification for %d unique companies with %d workers allow_web=%s parse_ms=%s",
            len(unique_companies),
            company_workers,
            allow_web,
            round((classify_started - parse_started) * 1000),
        )
        with ThreadPoolExecutor(max_workers=company_workers, thread_name_prefix="company-classifier") as executor:
            futures = {
                executor.submit(classify_single_company, company): company
                for company in unique_companies
            }
            for future in as_completed(futures):
                company = futures[future]
                try:
                    norm, details = future.result()
                    company_classification[norm] = details
                except Exception as e:
                    logger.exception("Failed to classify company: %s", company)
                    norm = normalize_company_name(company)
                    company_classification[norm] = {
                        "product_service": "Unknown",
                        "industry": "Unknown",
                        "customer_segment": [],
                        "business_model": "Unknown",
                        "verification_status": "not_verified",
                        "confidence": "low",
                        "source": "unknown",
                        "sources": [],
                    }
        logger.info(
            "Completed company classification for %d unique companies allow_web=%s duration_ms=%s",
            len(unique_companies),
            allow_web,
            round((time.perf_counter() - classify_started) * 1000),
        )

    # Pre-populate/Pre-insert companies sequentially into the DB to avoid deadlocks/conflicts
    company_ids: Dict[str, int] = {}
    company_prepopulate_limit = int(os.getenv("IMPORT_ENRICHMENT_COMPANY_PREPOPULATE_LIMIT", "5000"))
    skip_company_prepopulate = len(unique_companies) > company_prepopulate_limit
    if skip_company_prepopulate:
        logger.info(
            "Skipping global company pre-population for %d unique companies over limit=%d",
            len(unique_companies),
            company_prepopulate_limit,
        )
        company_ids = None
    conn = None if skip_company_prepopulate else get_db_connection(validate=False, register_pgvector=False)
    if conn:
        try:
            with conn.cursor() as cur:
                company_batch_size = max(1, int(os.getenv("IMPORT_ENRICHMENT_COMPANY_DB_BATCH_SIZE", "250")))
                company_list = sorted(unique_companies)
                for offset in range(0, len(company_list), company_batch_size):
                    batch = company_list[offset : offset + company_batch_size]
                    values = []
                    for company in batch:
                        norm = normalize_company_name(company)
                        details = company_classification.get(norm) or {}
                        values.append(
                            (
                                company,
                                details.get("business_model") or "Unknown",
                                details.get("product_service") or "Unknown",
                                details.get("customer_segment") or None,
                            )
                        )
                    rows = execute_values(
                        cur,
                        """
                        INSERT INTO companies (
                            name, business_model, product_service, customer_segment, created_by, updated_at
                        ) VALUES %s
                        ON CONFLICT (name) DO UPDATE SET
                            business_model = COALESCE(NULLIF(EXCLUDED.business_model, 'Unknown'), companies.business_model),
                            product_service = COALESCE(NULLIF(EXCLUDED.product_service, 'Unknown'), companies.product_service),
                            customer_segment = CASE
                                WHEN EXCLUDED.customer_segment IS NOT NULL AND array_length(EXCLUDED.customer_segment, 1) IS NOT NULL
                                THEN EXCLUDED.customer_segment ELSE companies.customer_segment END,
                            updated_at = NOW()
                        RETURNING name, id
                        """,
                        values,
                        template="(%s, %s, %s, %s, 'import_enrichment', NOW())",
                        fetch=True,
                    )
                    for name, company_id in rows:
                        company_ids[normalize_company_name(name)] = int(company_id)
                    conn.commit()
                    logger.info(
                        "Pre-populated %d/%d companies in DB",
                        min(offset + len(batch), len(company_list)),
                        len(company_list),
                    )
            conn.commit()
            logger.info("Successfully pre-populated %d companies in DB", len(company_ids))
        except Exception as e:
            conn.rollback()
            logger.exception("Failed to pre-populate companies in DB. Fallback to per-candidate DB inserts.")
            company_ids = None
        finally:
            return_db_connection(conn)

    processed = failed = 0
    errors: List[str] = []

    def enrich_single_candidate(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            raw = candidate.get("raw_fields") or {}
            roles = all_roles_by_candidate.get(candidate["id"], [])
            for role in roles:
                role.function, role.function_confidence, role.function_reason = classify_function(role)
                details = company_classification.get(normalize_company_name(role.company), {})
                if (
                    role.source_industry
                    and (not details or details.get("product_service") == "Unknown")
                    and details.get("source") not in {"db", "cache", "web"}
                ):
                    details = {**details, **_details_from_source_industry(role.source_industry)}
                role.product_service = details.get("product_service") or "Unknown"
                role.industry = details.get("industry") or "Unknown"
                role.customer_segment = list(details.get("customer_segment") or [])
                role.business_model = details.get("business_model") or "Unknown"
                role.verification_status = details.get("verification_status") or "not_verified"
                role.sources = list(details.get("sources") or [])
            education = parse_education_from_raw(raw)

            raw_total_exp_years: Optional[float] = None
            raw_total_str = _raw_get(raw, "totalExperienceYears", "Total Experience Years", "total experience years")
            if raw_total_str:
                try:
                    raw_total_exp_years = float(raw_total_str)
                except (TypeError, ValueError):
                    pass

            metrics = calculate_tenure_metrics(roles, raw_total_exp_years=raw_total_exp_years)
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
            _persist_candidate_enrichment(candidate, roles, education, metrics, payload, company_ids=company_ids)
            return True, None
        except Exception as exc:
            msg = f"candidate {candidate.get('id')}: {exc}"
            logger.exception("Import enrichment failed for candidate %s", candidate.get("id"))
            return False, msg

    candidate_workers = int(os.getenv("IMPORT_ENRICHMENT_CANDIDATE_WORKERS", "4"))
    if candidates:
        candidate_started = time.perf_counter()
        logger.info("Starting concurrent candidate enrichment for %d candidates with %d workers", len(candidates), candidate_workers)
        with ThreadPoolExecutor(max_workers=candidate_workers, thread_name_prefix="candidate-enricher") as executor:
            futures = {executor.submit(enrich_single_candidate, candidate): candidate for candidate in candidates}
            for future in as_completed(futures):
                success, error_msg = future.result()
                if success:
                    processed += 1
                else:
                    failed += 1
                    if error_msg:
                        errors.append(error_msg)
        logger.info(
            "Completed candidate enrichment candidates=%s processed=%s failed=%s duration_ms=%s total_ms=%s",
            len(candidates),
            processed,
            failed,
            round((time.perf_counter() - candidate_started) * 1000),
            round((time.perf_counter() - started) * 1000),
        )

    return {"processed": processed, "failed": failed, "errors": errors}

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.services.candidate_pool import profile_passes_scope


DEFAULT_FIELD_DEFINITIONS: Sequence[Dict[str, Any]] = (
    # ── Candidate / DB columns ────────────────────────────────────────────────
    {"key": "candidate.id", "label": "candidate.id", "group": "Default Fields", "path": ("id",)},
    {"key": "candidate.full_name", "label": "candidate.name (Full Name)", "group": "Default Fields", "path": ("name",)},
    {"key": "candidate.first_name", "label": "candidate.first_name", "group": "Default Fields", "path": ("first_name",)},
    {"key": "candidate.last_name", "label": "candidate.last_name", "group": "Default Fields", "path": ("last_name",)},
    {"key": "candidate.linkedin", "label": "candidate.linkedin", "group": "Default Fields", "path": ("linkedin",)},
    {"key": "candidate.normalized_linkedin", "label": "candidate.normalized_linkedin", "group": "Default Fields", "path": ("normalized_linkedin",)},
    {"key": "candidate.location", "label": "candidate.location", "group": "Default Fields", "path": ("location",)},
    {"key": "candidate.city", "label": "candidate.city", "group": "Default Fields", "path": ("city",)},
    {"key": "candidate.headline", "label": "candidate.headline", "group": "Default Fields", "path": ("headline",)},
    {"key": "candidate.about", "label": "candidate.about", "group": "Default Fields", "path": ("about",)},
    {"key": "candidate.email", "label": "candidate.email", "group": "Default Fields", "path": ("email",)},
    {"key": "candidate.phone", "label": "candidate.phone", "group": "Default Fields", "path": ("phone",)},
    {"key": "candidate.mobile_phone", "label": "candidate.mobile_phone", "group": "Default Fields", "path": ("mobile_phone",)},
    {"key": "candidate.status", "label": "candidate.status", "group": "Default Fields", "path": ("status",)},
    {"key": "candidate.notes", "label": "candidate.notes", "group": "Default Fields", "path": ("notes",)},
    {"key": "candidate.work_preference", "label": "candidate.work_preference", "group": "Default Fields", "path": ("work_preference",)},
    {"key": "candidate.extracted_industry", "label": "candidate.extracted_industry", "group": "Default Fields", "path": ("extracted_industry",)},
    {
        "key": "candidate.total_experience_years",
        "label": "candidate.total_experience_years",
        "group": "Default Fields",
        "path": ("total_experience_years",),
    },
    {
        "key": "candidate.avg_tenure_years",
        "label": "candidate.avg_years_in_company",
        "group": "Default Fields",
        "path": ("avg_years_in_company",),
    },
    {
        "key": "candidate.product_service",
        "label": "candidate.candidate_services",
        "group": "Default Fields",
        "path": ("candidate_services",),
    },
    # ── Role fields ───────────────────────────────────────────────────────────
    {"key": "role.current_title", "label": "roles[0].title", "group": "Role Fields", "path": ("roles", 0, "title")},
    {"key": "role.current_company", "label": "roles[0].company", "group": "Role Fields", "path": ("roles", 0, "company")},
    {"key": "role.current_industry", "label": "roles[0].industry", "group": "Role Fields", "path": ("roles", 0, "industry")},
    {"key": "role.start_date", "label": "roles[0].start_date", "group": "Role Fields", "path": ("roles", 0, "start_date")},
    {"key": "role.end_date", "label": "roles[0].end_date", "group": "Role Fields", "path": ("roles", 0, "end_date")},
    {
        "key": "role.company_product_service",
        "label": "roles[0].company_details.product_service",
        "group": "Role Fields",
        "path": ("roles", 0, "company_details", "product_service"),
    },
    {
        "key": "role.company_headquarters",
        "label": "roles[0].company_details.headquarters",
        "group": "Role Fields",
        "path": ("roles", 0, "company_details", "headquarters"),
    },
    {
        "key": "role.company_size",
        "label": "roles[0].company_details.size",
        "group": "Role Fields",
        "path": ("roles", 0, "company_details", "size"),
    },
)

ROLE_CONTEXT_FIELD_DEFINITIONS: Sequence[Dict[str, Any]] = (
    {"key": "role.name", "label": "role.name", "group": "Role Context"},
    {"key": "role.job_description", "label": "role.job_description", "group": "Role Context"},
)

COLUMN_CONTEXT_FIELD_DEFINITIONS: Sequence[Dict[str, Any]] = (
    {"key": "context.our_product", "label": "context.our_product", "group": "Column Context"},
    {"key": "context.pitch_context", "label": "context.pitch_context", "group": "Column Context"},
)

TOKEN_PATTERN = re.compile(r"\{([^{}]+)\}")
URL_PATTERN = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)
MAX_IMPORTED_FIELDS = 120
COMMUNITY_MEMBERSHIP_TERMS = (
    "revgenuis",
    "revgenius",
    "community",
    "member",
    "membership",
    "mentor",
    "advisor",
    "volunteer",
)
WEB_FRESHNESS_TERMS = (
    "recent",
    "latest",
    "last 30 days",
    "last thirty days",
    "today",
    "current news",
    "layoff",
    "layoffs",
    "restructuring",
    "downsizing",
    "funding",
    "acquisition",
    "posted content",
    "posted on linkedin",
)


def _context_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size == 0
    return False


def _normalize_context_container(value: Any) -> Any:
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return value


def normalize_output_key(name: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    return text or "output"


def labelize_key(name: str) -> str:
    parts = re.split(r"[_\.\-]+", str(name or ""))
    return " ".join(p.capitalize() for p in parts if p)


def safe_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def flatten_raw_fields(raw_fields: Any, prefix: str = "raw") -> List[Dict[str, Any]]:
    raw = safe_json(raw_fields)
    if not raw:
        return []
    out: List[Dict[str, Any]] = []

    def walk(value: Any, parts: List[str]) -> None:
        if len(out) >= MAX_IMPORTED_FIELDS:
            return
        value = _normalize_context_container(value)
        if isinstance(value, dict):
            for k, child in value.items():
                walk(child, parts + [str(k)])
            return
        if isinstance(value, list):
            if value and all(not isinstance(item, (dict, list)) for item in value):
                joined = ", ".join(str(item) for item in value if item not in (None, ""))
                if joined:
                    key = ".".join(parts)
                    out.append(
                        {
                            "key": key,
                            "label": labelize_key(parts[-1]),
                            "group": "Imported Fields",
                            "path": tuple(parts),
                            "sample": joined[:180],
                        }
                    )
            return
        if _context_value_is_empty(value):
            return
        key = ".".join(parts)
        out.append(
            {
                "key": key,
                "label": labelize_key(parts[-1]),
                "group": "Imported Fields",
                "path": tuple(parts),
                "sample": str(value)[:180],
            }
        )

    walk(raw, [prefix])
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in out:
        dedup[item["key"]] = item
    return list(dedup.values())


def _get_nested_value(obj: Any, path: Sequence[Any]) -> Any:
    cur = obj
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or len(cur) <= part:
                return None
            cur = cur[part]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
    return cur


def build_candidate_context(
    profile: Dict[str, Any],
    ai_values: Optional[Dict[str, Any]] = None,
    role_context: Optional[Dict[str, Any]] = None,
    context_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    for item in DEFAULT_FIELD_DEFINITIONS:
        value = _get_nested_value(profile, item["path"])
        if item["key"] == "candidate.product_service" and not value:
            value = profile.get("extracted_industry") or profile.get("candidate_services")
        ctx[item["key"]] = stringify_context_value(value)
    # Keep a legacy alias so older prompts using {candidate.name} still work.
    if "candidate.full_name" in ctx:
        ctx["candidate.name"] = ctx["candidate.full_name"]
    # Clay examples often paste the visible spreadsheet header directly.
    linkedin_value = ctx.get("candidate.linkedin", "")
    for alias in ("Linkedin Profile", "LinkedIn Profile", "linkedin profile"):
        ctx[alias] = linkedin_value
    for imported in flatten_raw_fields(profile.get("raw_fields")):
        raw_value = _get_nested_value(profile.get("raw_fields") or {}, imported["path"][1:])
        ctx[imported["key"]] = stringify_context_value(raw_value)
    for key, value in flatten_profile_context(profile).items():
        ctx.setdefault(key, value)
    for ai_key, ai_val in (ai_values or {}).items():
        ctx[ai_key] = stringify_context_value(ai_val)
    role_data = role_context or {}
    ctx["role.name"] = stringify_context_value(role_data.get("name"))
    ctx["role.job_description"] = stringify_context_value(role_data.get("job_description"))
    input_data = context_inputs or {}
    ctx["context.our_product"] = stringify_context_value(input_data.get("our_product"))
    ctx["context.pitch_context"] = stringify_context_value(input_data.get("pitch_context"))
    ctx["context.our_product_or_pitch_context"] = (
        ctx["context.our_product"] or ctx["context.pitch_context"]
    )
    return ctx


def flatten_profile_context(profile: Dict[str, Any], *, max_fields: int = 220) -> Dict[str, str]:
    """Expose the full row as row.* keys so AI columns can behave like Clay chatbot prompts."""
    out: Dict[str, str] = {}

    def walk(value: Any, parts: List[str]) -> None:
        if len(out) >= max_fields:
            return
        value = _normalize_context_container(value)
        if _context_value_is_empty(value):
            return
        if isinstance(value, dict):
            for k, child in value.items():
                walk(child, parts + [str(k)])
                if len(out) >= max_fields:
                    return
            return
        if isinstance(value, list):
            if all(not isinstance(item, (dict, list)) for item in value):
                text = stringify_context_value(value)
                if text:
                    out["row." + ".".join(parts)] = text
                return
            for idx, child in enumerate(value[:12]):
                walk(child, parts + [str(idx)])
                if len(out) >= max_fields:
                    return
            return
        text = stringify_context_value(value)
        if text:
            out["row." + ".".join(parts)] = text

    walk(profile or {}, [])
    return out


def stringify_context_value(value: Any) -> str:
    if value is None:
        return ""
    value = _normalize_context_container(value)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                parts.append(text)
        return ", ".join(parts)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""
    return str(value).strip()


def extract_urls(text: str) -> List[str]:
    return [match.group(0).rstrip(".,;") for match in URL_PATTERN.finditer(text or "")]


def classify_ai_column_prompt(prompt_template: str) -> Dict[str, str]:
    """Decide whether an AI-column prompt should use row data, web, or both."""
    prompt = (prompt_template or "").strip()
    lower = prompt.lower()
    urls = extract_urls(prompt)

    linkedin_activity = (
        "linkedin" in lower
        and (
            "posted" in lower
            or "post " in lower
            or "posts" in lower
            or "content" in lower
            or "activity" in lower
        )
        and ("last 30" in lower or "30 days" in lower or "recent" in lower)
    )
    if linkedin_activity:
        return {
            "data_source": "web",
            "web_required_reason": "public_linkedin_recent_activity",
            "routing_mode": "web_research",
        }

    if urls and ("jd" in lower or "job description" in lower or "score" in lower or "fit" in lower):
        return {
            "data_source": "hybrid",
            "web_required_reason": "jd_url_or_public_fit_evidence",
            "routing_mode": "web_research",
        }

    if any(term in lower for term in ("score", "fit score", "against this jd", "job description", "paste jd")):
        return {
            "data_source": "hybrid",
            "web_required_reason": "jd_fit_requires_public_verification",
            "routing_mode": "web_research",
        }

    if "account executive" in lower and (
        "enterprise" in lower or "saas" in lower or "10+" in lower or "overall experience" in lower
    ):
        return {
            "data_source": "hybrid",
            "web_required_reason": "ae_experience_and_employer_segment_verification",
            "routing_mode": "web_research",
        }

    if "enterprise" in lower and "saas" in lower and ("currently working" in lower or "current" in lower):
        return {
            "data_source": "hybrid",
            "web_required_reason": "current_employer_enterprise_saas_verification",
            "routing_mode": "web_research",
        }

    if any(term in lower for term in WEB_FRESHNESS_TERMS) or any(
        term in lower for term in ("public evidence", "publicly", "web", "website", "news")
    ):
        return {
            "data_source": "web",
            "web_required_reason": "fresh_or_public_web_evidence_requested",
            "routing_mode": "web_research",
        }

    if any(
        term in lower
        for term in (
            "average tenure",
            "current job",
            "time spent",
            "overall experience",
            "total years",
            "number of cities",
            "cities the person has worked",
            "unique companies",
        )
    ):
        return {
            "data_source": "row",
            "web_required_reason": "",
            "routing_mode": "content",
        }

    return {
        "data_source": "row",
        "web_required_reason": "",
        "routing_mode": "content",
    }


def _parse_role_context_entries(context: Dict[str, str]) -> List[Dict[str, str]]:
    grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
    for key, value in (context or {}).items():
        match = re.match(r"row\.roles\.(\d+)\.(.+)", str(key))
        if not match:
            match = re.match(r"row\.raw_fields\.enrichment\.roles\.(\d+)\.(.+)", str(key))
        if not match:
            continue
        text = stringify_context_value(value)
        if text:
            grouped[int(match.group(1))][match.group(2)] = text
    entries = [grouped[idx] for idx in sorted(grouped)]
    if not entries and (context.get("role.current_company") or context.get("role.current_title")):
        entries.append(
            {
                "company": context.get("role.current_company", ""),
                "title": context.get("role.current_title", ""),
                "start_date": context.get("role.start_date", ""),
                "end_date": context.get("role.end_date", ""),
                "industry": context.get("role.current_industry", ""),
            }
        )
    return entries


def _normalize_company_name(company: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", (company or "").lower()).strip()
    text = re.sub(r"\b(private|pvt|ltd|limited|inc|corp|corporation|llc|plc|company|co)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _role_is_community_membership(role: Dict[str, str]) -> bool:
    combined = " ".join(
        stringify_context_value(role.get(key))
        for key in ("title", "company", "headline", "description", "summary")
    ).lower()
    if "revgenuis" in combined:
        combined = combined.replace("revgenuis", "revgenius")
    return any(term in combined for term in COMMUNITY_MEMBERSHIP_TERMS)


def _parse_role_date(value: str, *, default_current: bool = False) -> Optional[datetime]:
    text = stringify_context_value(value).strip()
    if not text:
        return datetime.now(timezone.utc) if default_current else None
    lower = text.lower()
    if lower in {"present", "current", "now", "till date", "ongoing"}:
        return datetime.now(timezone.utc)
    patterns = (
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%b %Y",
        "%B %Y",
        "%b %d %Y",
        "%B %d %Y",
    )
    normalized = text.replace(",", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for pattern in patterns:
        try:
            parsed = datetime.strptime(normalized, pattern)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        return datetime(int(year_match.group(0)), 1, 1, tzinfo=timezone.utc)
    return None


def _months_between(start: Optional[datetime], end: Optional[datetime]) -> int:
    if not start:
        return 0
    end = end or datetime.now(timezone.utc)
    if end < start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day >= start.day:
        months += 1
    return max(0, months)


def _merge_intervals(intervals: Sequence[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    valid = sorted(
        [(start, end) for start, end in intervals if start and end and end >= start],
        key=lambda item: item[0],
    )
    merged: List[Tuple[datetime, datetime]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def _float_context(context: Dict[str, str], key: str) -> float:
    try:
        return float(stringify_context_value(context.get(key)).replace(",", ""))
    except Exception:
        return 0.0


def _role_is_account_executive(role: Dict[str, str]) -> bool:
    title = stringify_context_value(role.get("title")).lower()
    return "account executive" in title or re.search(r"\bae\b", title) is not None


def _current_company_enterprise_saas_status(context: Dict[str, str], current_role: Dict[str, str]) -> Tuple[str, List[str]]:
    fields = {
        "current role product/service": current_role.get("product_service") or context.get("role.company_product_service"),
        "current role customer segment": current_role.get("customer_segment")
        or context.get("role.company_customer_segment")
        or context.get("row.roles.0.company_details.customer_segment"),
        "current role business model": current_role.get("business_model")
        or context.get("row.roles.0.company_details.business_model"),
        "candidate product/service": context.get("candidate.product_service"),
        "headline": context.get("candidate.headline"),
    }
    evidence = [
        f"{label}: {stringify_context_value(value)}"
        for label, value in fields.items()
        if stringify_context_value(value)
    ]
    combined = " ".join(evidence).lower()
    has_enterprise = bool(re.search(r"\benterprise|enterprises|large accounts?|strategic accounts?\b", combined))
    has_saas = bool(re.search(r"\bsaas|software|cloud platform|cloud software|subscription\b", combined))
    if has_enterprise and has_saas:
        return "Yes", evidence
    if evidence and ("unknown" not in combined):
        return "No", evidence
    return "Needs web verification", evidence


def compute_career_facts(context: Dict[str, str]) -> Dict[str, Any]:
    roles = [role for role in _parse_role_context_entries(context) if not _role_is_community_membership(role)]
    companies: Dict[str, Dict[str, Any]] = {}
    cities = set()
    ae_months = 0
    role_tenures: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[datetime, datetime]] = []

    for idx, role in enumerate(roles):
        company = stringify_context_value(role.get("company"))
        normalized_company = _normalize_company_name(company)
        if not normalized_company:
            continue
        start = _parse_role_date(role.get("start_date") or role.get("starts_at") or role.get("start"))
        end = _parse_role_date(role.get("end_date") or role.get("ends_at") or role.get("end"), default_current=idx == 0)
        months = _months_between(start, end)
        title = stringify_context_value(role.get("title"))
        location = stringify_context_value(role.get("city") or role.get("location"))
        role_tenures.append(
            {
                "company": company,
                "title": title,
                "start_date": start.date().isoformat() if start else "",
                "end_date": end.date().isoformat() if end else "",
                "months": months,
                "years": round(months / 12, 1) if months else 0,
            }
        )
        if location:
            cities.add(location.lower())
        bucket = companies.setdefault(
            normalized_company,
            {"company": company, "months": 0, "starts": [], "ends": [], "titles": []},
        )
        bucket["months"] += months
        if start:
            bucket["starts"].append(start)
        if end:
            bucket["ends"].append(end)
        if start and end:
            all_intervals.append((start, end))
        if title:
            bucket["titles"].append(title)
        if _role_is_account_executive(role):
            ae_months += months

    for company in companies.values():
        starts = company.get("starts") or []
        ends = company.get("ends") or []
        if starts:
            company["months"] = _months_between(min(starts), max(ends) if ends else None)

    company_tenures = [
        {
            "company": company.get("company") or "",
            "months": int(company.get("months") or 0),
            "years": round(int(company.get("months") or 0) / 12, 1) if company.get("months") else 0,
            "titles": company.get("titles") or [],
        }
        for company in companies.values()
    ]
    short_company_stints = [
        tenure for tenure in company_tenures
        if 0 < int(tenure.get("months") or 0) < 24
    ]
    very_short_company_stints = [
        tenure for tenure in company_tenures
        if 0 < int(tenure.get("months") or 0) < 12
    ]
    merged_all_intervals = _merge_intervals(all_intervals)
    total_months_from_roles = sum(_months_between(start, end) for start, end in merged_all_intervals)
    total_exp_years = _float_context(context, "candidate.total_experience_years")
    total_months = total_months_from_roles or int(round(total_exp_years * 12))
    unique_company_count = len(companies)
    average_tenure_months = int(round(total_months / unique_company_count)) if unique_company_count else 0
    job_hopping_flag = (
        len(very_short_company_stints) >= 2
        or len(short_company_stints) >= 3
        or (unique_company_count >= 3 and 0 < average_tenure_months < 18)
    )
    if not unique_company_count:
        job_hopping_status = "Unknown"
        job_hopping_reason = "No dated company tenure available."
    elif job_hopping_flag:
        job_hopping_status = "Yes"
        job_hopping_reason = (
            f"{len(short_company_stints)} company stint(s) under 24 months; "
            f"average company tenure is {average_tenure_months} months."
        )
    else:
        job_hopping_status = "No"
        job_hopping_reason = (
            f"Average company tenure is {average_tenure_months} months with "
            f"{len(short_company_stints)} stint(s) under 24 months."
        )

    current_role = roles[0] if roles else {}
    current_start = _parse_role_date(
        current_role.get("start_date") or current_role.get("starts_at") or current_role.get("start")
    )
    current_end = _parse_role_date(
        current_role.get("end_date") or current_role.get("ends_at") or current_role.get("end"),
        default_current=True,
    )
    current_job_months = _months_between(current_start, current_end)
    if context.get("candidate.city"):
        cities.add(context["candidate.city"].lower())
    current_company_enterprise_saas, current_company_evidence = _current_company_enterprise_saas_status(context, current_role)

    return {
        "total_experience_months": total_months,
        "total_experience_years": round(total_months / 12, 1) if total_months else total_exp_years,
        "unique_company_count": unique_company_count,
        "average_tenure_months": average_tenure_months,
        "current_job_months": current_job_months,
        "ae_experience_months": ae_months,
        "ae_experience_years": round(ae_months / 12, 1),
        "career_city_count": len(cities),
        "career_cities": sorted(cities),
        "companies": [company["company"] for company in companies.values()],
        "company_tenures": company_tenures,
        "short_company_stints_count": len(short_company_stints),
        "very_short_company_stints_count": len(very_short_company_stints),
        "job_hopping_status": job_hopping_status,
        "job_hopping_reason": job_hopping_reason,
        "role_tenures": role_tenures,
        "current_company_enterprise_saas": current_company_enterprise_saas,
        "current_company_enterprise_saas_evidence": current_company_evidence,
    }


def career_facts_to_text(facts: Dict[str, Any]) -> str:
    if not facts:
        return ""
    role_lines = []
    for role in (facts.get("role_tenures") or [])[:12]:
        role_lines.append(
            f"{role.get('title') or 'Unknown role'} at {role.get('company') or 'Unknown company'}: "
            f"{role.get('months') or 0} months"
            f" ({role.get('start_date') or '?'} to {role.get('end_date') or '?'})"
        )
    return (
        "Deterministic row-derived career facts:\n"
        f"- total_experience_months: {facts.get('total_experience_months') or 0}\n"
        f"- total_experience_years: {facts.get('total_experience_years') or 0}\n"
        f"- unique_company_count: {facts.get('unique_company_count') or 0}\n"
        f"- average_tenure_months: {facts.get('average_tenure_months') or 0}\n"
        f"- current_job_months: {facts.get('current_job_months') or 0}\n"
        f"- ae_experience_months: {facts.get('ae_experience_months') or 0}\n"
        f"- current_company_enterprise_saas: {facts.get('current_company_enterprise_saas') or 'Needs web verification'}\n"
        f"- job_hopping_status: {facts.get('job_hopping_status') or 'Unknown'}\n"
        f"- job_hopping_reason: {facts.get('job_hopping_reason') or ''}\n"
        f"- career_city_count: {facts.get('career_city_count') or 0}\n"
        f"- companies_counted: {', '.join(facts.get('companies') or []) or 'unknown'}\n"
        f"- role_tenures: {'; '.join(role_lines) if role_lines else 'unknown'}"
    )


def map_career_facts_to_outputs(
    prompt_template: str,
    output_schema: Sequence[Dict[str, Any]],
    facts: Dict[str, Any],
) -> Dict[str, str]:
    prompt_l = (prompt_template or "").lower()
    wants_career = any(
        term in prompt_l
        for term in (
            "average tenure",
            "current job",
            "time spent",
            "number of cities",
            "cities the person has worked",
            "total years",
            "overall experience",
            "unique companies",
            "job hopping",
            "job hopper",
            "hopper",
            "short stint",
            "frequent switch",
        )
    )
    if not wants_career or not facts:
        return {}

    outputs: Dict[str, str] = {}
    yes_no = ""
    if "10+" in prompt_l or "10 plus" in prompt_l or "minimum 5+" in prompt_l:
        current_company_ok = facts.get("current_company_enterprise_saas") == "Yes"
        qualified = (
            float(facts.get("total_experience_months") or 0) >= 120
            and float(facts.get("ae_experience_months") or 0) >= 60
            and (current_company_ok if "enterprise" in prompt_l and "saas" in prompt_l else True)
        )
        yes_no = "Yes" if qualified else "No"

    summary = (
        f"Average tenure: {facts.get('average_tenure_months') or 0} months. "
        f"Current job tenure: {facts.get('current_job_months') or 0} months. "
        f"Overall experience: {facts.get('total_experience_months') or 0} months. "
        f"Account executive experience: {facts.get('ae_experience_months') or 0} months. "
        f"Current enterprise SaaS company: {facts.get('current_company_enterprise_saas') or 'Needs web verification'}. "
        f"Job hopping: {facts.get('job_hopping_status') or 'Unknown'} "
        f"({facts.get('job_hopping_reason') or 'No reason available'}). "
        f"Unique companies counted: {facts.get('unique_company_count') or 0}. "
        f"Career cities counted: {facts.get('career_city_count') or 0}."
    )
    if yes_no:
        summary += f" Qualification: {yes_no}."

    for item in normalize_output_schema(output_schema):
        key = item["key"]
        label = f"{key} {item.get('label', '')}".lower()
        if "average" in label and "tenure" in label:
            outputs[key] = str(facts.get("average_tenure_months") or 0)
        elif "current" in label and ("job" in label or "tenure" in label or "role" in label):
            outputs[key] = str(facts.get("current_job_months") or 0)
        elif "city" in label and "count" in label:
            outputs[key] = str(facts.get("career_city_count") or 0)
        elif "cities" in label:
            outputs[key] = ", ".join(facts.get("career_cities") or [])
        elif "company" in label and "count" in label:
            outputs[key] = str(facts.get("unique_company_count") or 0)
        elif "total" in label and ("experience" in label or "month" in label):
            outputs[key] = str(facts.get("total_experience_months") or 0)
        elif "ae" in label or "account_executive" in label or "account executive" in label:
            outputs[key] = str(facts.get("ae_experience_months") or 0)
        elif "enterprise" in label and "saas" in label:
            outputs[key] = str(facts.get("current_company_enterprise_saas") or "Needs web verification")
        elif "job" in label and ("hop" in label or "switch" in label):
            outputs[key] = str(facts.get("job_hopping_status") or "Unknown")
        elif "short" in label and "stint" in label:
            outputs[key] = str(facts.get("short_company_stints_count") or 0)
        elif "qualified" in label or "qualification" in label or "eligible" in label:
            outputs[key] = yes_no or "Needs web verification"
        elif key in {"result", "summary", "reasoning"}:
            outputs[key] = summary
        else:
            outputs[key] = summary
    return outputs


def _context_json_value(context: Dict[str, str], key: str, default: Any = None) -> Any:
    value = stringify_context_value((context or {}).get(key))
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _flat_context_section(context: Dict[str, str], prefix: str) -> Dict[str, str]:
    section: Dict[str, str] = {}
    needle = f"{prefix}."
    for key, value in (context or {}).items():
        if key.startswith(needle):
            section[key[len(needle) :]] = stringify_context_value(value)
    return section


def build_candidate_context_pack(context: Dict[str, str], facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact, structured row/enrichment context for free-form Smart Column reasoning."""
    facts = facts or {}
    roles = _parse_role_context_entries(context)
    work_roles = [role for role in roles if not _role_is_community_membership(role)]
    enrichment = _context_json_value(context, "raw.enrichment", {}) or _context_json_value(
        context, "row.raw_fields.enrichment", {}
    ) or {}
    profile_claims = {}
    if isinstance(enrichment, dict):
        profile_claims = enrichment.get("profile_claims") if isinstance(enrichment.get("profile_claims"), dict) else {}
    if not profile_claims:
        profile_claims = _flat_context_section(context, "row.raw_fields.enrichment.profile_claims")
    enrichment_flat = _flat_context_section(context, "row.raw_fields.enrichment")

    return {
        "candidate": {
            "id": stringify_context_value(context.get("candidate.id")),
            "name": stringify_context_value(context.get("candidate.full_name") or context.get("candidate.name")),
            "linkedin": stringify_context_value(
                context.get("candidate.linkedin")
                or context.get("Linkedin Profile")
                or context.get("LinkedIn Profile")
            ),
            "headline": stringify_context_value(context.get("candidate.headline")),
            "location": stringify_context_value(context.get("candidate.location") or context.get("candidate.city")),
            "about": stringify_context_value(context.get("candidate.about")),
            "email": stringify_context_value(context.get("candidate.email")),
            "phone": stringify_context_value(context.get("candidate.mobile_phone") or context.get("candidate.phone")),
        },
        "current_role": work_roles[0] if work_roles else {},
        "roles": work_roles[:12],
        "community_roles_excluded": [role for role in roles if _role_is_community_membership(role)][:8],
        "education": enrichment.get("education", [])[:8] if isinstance(enrichment, dict) else [],
        "profile_claims": profile_claims,
        "career_facts": facts,
        "enrichment_status": {
            "verification_status": (
                enrichment.get("verification_status")
                if isinstance(enrichment, dict)
                else enrichment_flat.get("verification_status", "")
            ),
            "verification_errors": (
                enrichment.get("verification_errors", [])
                if isinstance(enrichment, dict)
                else enrichment_flat.get("verification_errors", "")
            ),
            "sources": enrichment.get("sources", {}) if isinstance(enrichment, dict) else _flat_context_section(context, "row.raw_fields.enrichment.sources"),
        },
    }


def _prompt_has_any(prompt: str, terms: Sequence[str]) -> bool:
    lower = (prompt or "").lower()
    return any(term.lower() in lower for term in terms)


def _prompt_terms(prompt: str, taxonomy: Dict[str, Sequence[str]]) -> List[str]:
    lower = (prompt or "").lower()
    found: List[str] = []
    for canonical, variants in taxonomy.items():
        values = [canonical, *variants]
        if any(re.search(rf"\b{re.escape(value.lower())}\b", lower) for value in values if value):
            found.append(canonical)
    return found


SEGMENT_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "Enterprise": ("enterprise", "large account", "strategic account"),
    "SMB": ("smb", "small business", "small businesses"),
    "SME": ("sme", "small and medium", "small medium"),
    "Mid-Market": ("mid market", "mid-market", "midmarket"),
    "Government": ("government", "public sector"),
}
FUNCTION_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "Account Executive": ("account executive", "enterprise account executive", "ae"),
    "Hunting": ("hunting", "new business", "new logo"),
    "Farming": ("farming", "account management", "expansion", "renewal"),
    "SDR/BDR": ("sdr", "bdr", "sales development", "business development representative"),
    "Customer Success": ("customer success", "csm", "customer success manager"),
    "Partnerships": ("partner", "channel", "alliances"),
}
GEOGRAPHY_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "EMEA": ("emea", "europe", "middle east", "africa"),
    "APAC": ("apac", "asia pacific", "india", "singapore", "australia"),
    "LATAM": ("latam", "latin america", "brazil", "mexico"),
    "Americas": ("americas", "north america", "usa", "united states", "canada"),
    "India": ("india", "mumbai", "bengaluru", "bangalore", "delhi"),
}
INDUSTRY_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "SaaS": ("saas", "software", "cloud platform", "subscription"),
    "Fintech": ("fintech", "payments", "banking", "lending"),
    "HRTech": ("hrtech", "hr tech", "recruiting", "talent"),
    "BPO": ("bpo", "business process outsourcing"),
    "Analytics": ("analytics", "data platform", "business intelligence"),
}


def _role_text(role: Dict[str, Any]) -> str:
    return " ".join(
        stringify_context_value(role.get(key))
        for key in (
            "title",
            "company",
            "details",
            "function",
            "industry",
            "product_service",
            "business_model",
            "customer_segment",
            "location",
            "city",
        )
    ).lower()


def _matched_duration(roles: Sequence[Dict[str, Any]], terms: Sequence[str]) -> Dict[str, Any]:
    matches = []
    total_months = 0
    for role in roles:
        role_text = _role_text(role)
        if not any(term.lower() in role_text for term in terms):
            continue
        months = int(float(role.get("duration_months") or role.get("months") or 0))
        if not months:
            start = _parse_role_date(role.get("start_date") or role.get("start"))
            end = _parse_role_date(role.get("end_date") or role.get("end"), default_current=False)
            months = _months_between(start, end)
        total_months += months
        matches.append(
            {
                "company": stringify_context_value(role.get("company")),
                "title": stringify_context_value(role.get("title")),
                "months": months,
                "evidence": role_text[:240],
            }
        )
    return {"months": total_months, "years": round(total_months / 12, 1), "matches": matches}


def build_query_plan(
    prompt_template: str,
    context: Dict[str, str],
    output_schema: Sequence[Dict[str, Any]],
    routing: Optional[Dict[str, str]] = None,
    planner_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bounded, no-SQL plan for an arbitrary per-candidate Smart Column query."""
    prompt = prompt_template or ""
    routing = routing or classify_ai_column_prompt(prompt)
    tool_calls: List[str] = []
    if _prompt_has_any(
        prompt,
        (
            "tenure",
            "experience",
            "current job",
            "current role",
            "years",
            "months",
            "unique compan",
            "how long",
            "worked at",
            "work at",
            "time at",
        ),
    ):
        tool_calls.append("career_metrics")
    if _prompt_has_any(
        prompt,
        (
            "job hopping",
            "job hopper",
            "hopper",
            "short stint",
            "frequent switch",
            "switching jobs",
            "job switching",
            "switch jobs",
            "too many jobs",
        ),
    ):
        tool_calls.extend(["career_metrics", "job_hopping"])
    if _prompt_terms(prompt, FUNCTION_QUERY_TERMS):
        tool_calls.append("functional_experience")
    if _prompt_terms(prompt, SEGMENT_QUERY_TERMS):
        tool_calls.append("segment_experience")
    if _prompt_terms(prompt, GEOGRAPHY_QUERY_TERMS):
        tool_calls.append("geography_experience")
    if _prompt_terms(prompt, INDUSTRY_QUERY_TERMS) or _prompt_has_any(prompt, ("industry", "product", "service")):
        tool_calls.append("industry_experience")
    if _prompt_has_any(prompt, ("company", "currently working", "current employer", "public", "recent", "latest", "layoff")):
        tool_calls.append("company_verification")
    if not tool_calls:
        tool_calls.append("profile_lookup")

    current_status = ""
    facts = compute_career_facts(context)
    if facts:
        current_status = stringify_context_value(facts.get("current_company_enterprise_saas"))
    urls = extract_urls(prompt)
    web_needed = routing.get("data_source") == "web"
    if routing.get("data_source") == "hybrid":
        web_needed = bool(urls) or _prompt_has_any(prompt, ("public", "latest", "recent", "news", "layoff", "posted"))
    if current_status == "Needs web verification" and _prompt_has_any(
        prompt, ("enterprise", "saas", "company", "employer")
    ):
        web_needed = True

    plan = {
        "intent": "Answer a per-candidate Smart Column query from enriched profile context.",
        "needed_data": sorted(set(tool_calls + ["candidate_context_pack"])),
        "tool_calls": sorted(set(tool_calls)),
        "web_needed": bool(web_needed),
        "web_policy": "auto_fallback",
        "strictness": "unknown_instead_of_guess",
        "missing_info": [],
        "output_fields": [item["key"] for item in normalize_output_schema(output_schema)],
        "routing": routing,
    }
    if isinstance(planner_json, dict) and planner_json:
        for key in ("intent", "needed_data", "tool_calls", "web_needed", "missing_info", "output_fields"):
            value = planner_json.get(key)
            if value not in (None, "", []):
                plan[key] = value
    return plan


def run_candidate_query_tools(
    prompt_template: str,
    context: Dict[str, str],
    facts: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    facts = facts or compute_career_facts(context)
    plan = plan or build_query_plan(prompt_template, context, [], classify_ai_column_prompt(prompt_template))
    context_pack = build_candidate_context_pack(context, facts)
    roles = context_pack.get("roles") or []
    prompt = prompt_template or ""
    segment_terms = _prompt_terms(prompt, SEGMENT_QUERY_TERMS)
    function_terms = _prompt_terms(prompt, FUNCTION_QUERY_TERMS)
    geography_terms = _prompt_terms(prompt, GEOGRAPHY_QUERY_TERMS)
    industry_terms = _prompt_terms(prompt, INDUSTRY_QUERY_TERMS)

    return {
        "context_pack": context_pack,
        "career_metrics": facts,
        "current_role": context_pack.get("current_role") or {},
        "role_specific_tenures": facts.get("role_tenures") or [],
        "company_specific_tenures": {
            _normalize_company_name(item.get("company") or ""): item
            for item in (facts.get("company_tenures") or [])
            if item.get("company")
        },
        "job_hopping": {
            "status": facts.get("job_hopping_status") or "Unknown",
            "reason": facts.get("job_hopping_reason") or "",
            "short_company_stints_count": facts.get("short_company_stints_count") or 0,
            "very_short_company_stints_count": facts.get("very_short_company_stints_count") or 0,
            "company_tenures": facts.get("company_tenures") or [],
        },
        "segment_experience": {
            term: _matched_duration(roles, [term, *SEGMENT_QUERY_TERMS.get(term, ())])
            for term in segment_terms
        },
        "functional_experience": {
            term: _matched_duration(roles, [term, *FUNCTION_QUERY_TERMS.get(term, ())])
            for term in function_terms
        },
        "geography_experience": {
            term: _matched_duration(roles, [term, *GEOGRAPHY_QUERY_TERMS.get(term, ())])
            for term in geography_terms
        },
        "industry_experience": {
            term: _matched_duration(roles, [term, *INDUSTRY_QUERY_TERMS.get(term, ())])
            for term in industry_terms
        },
        "company_verification": {
            "current_company_enterprise_saas": facts.get("current_company_enterprise_saas"),
            "evidence": facts.get("current_company_enterprise_saas_evidence") or [],
            "web_needed": bool(plan.get("web_needed")),
        },
    }


def verify_smart_column_outputs(
    prompt_template: str,
    outputs: Dict[str, Any],
    *,
    data_source: str,
    sources: Sequence[Dict[str, Any]],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    unknown_reasons: List[str] = []
    facts = (tool_results or {}).get("career_metrics") or {}
    numeric_expectations = {
        "average_tenure": facts.get("average_tenure_months"),
        "current_job": facts.get("current_job_months"),
        "overall": facts.get("total_experience_months"),
        "total_experience": facts.get("total_experience_months"),
        "ae_experience": facts.get("ae_experience_months"),
        "account_executive": facts.get("ae_experience_months"),
    }
    for key, value in (outputs or {}).items():
        label = str(key).lower()
        expected = next((exp for marker, exp in numeric_expectations.items() if marker in label), None)
        if expected is None:
            continue
        text = stringify_context_value(value)
        if not text or not re.search(r"\d", text):
            continue
        actual_match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if not actual_match:
            continue
        actual = float(actual_match.group(0))
        if abs(actual - float(expected or 0)) > 1:
            errors.append(f"{key}={actual:g} does not match deterministic tool value {expected}.")

    has_source_url = any(isinstance(source, dict) and stringify_context_value(source.get("url")) for source in sources or [])
    if data_source in {"web", "hybrid"} and not has_source_url:
        unknown_reasons.append("Web-derived answer has no source URL.")

    output_text = " ".join(stringify_context_value(v) for v in (outputs or {}).values()).lower()
    if "unknown" in output_text or "needs verification" in output_text or "not publicly verifiable" in output_text:
        unknown_reasons.append("Answer contains unknown or unverifiable fields.")

    if errors:
        status = "failed"
    elif unknown_reasons:
        status = "passed_with_unknowns"
    else:
        status = "passed"
    return {
        "verification_status": status,
        "verification_errors": errors,
        "unknown_reasons": list(dict.fromkeys(unknown_reasons)),
    }


def extract_prompt_tokens(prompt_template: str) -> List[str]:
    found = []
    seen = set()
    for match in TOKEN_PATTERN.findall(prompt_template or ""):
        token = match.strip()
        if token and token not in seen:
            found.append(token)
            seen.add(token)
    return found


def fill_prompt_template(prompt_template: str, context: Dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        return context.get(token, "")

    return TOKEN_PATTERN.sub(replace, prompt_template or "")


def evaluate_required_fields(required_fields: Sequence[str], context: Dict[str, str]) -> Tuple[bool, List[str]]:
    missing = [field for field in required_fields if not stringify_context_value(context.get(field))]
    return (len(missing) == 0, missing)


def summarize_only_run_if(required_fields: Sequence[str]) -> str:
    if not required_fields:
        return ""
    if len(required_fields) == 1:
        return f"Only run if {required_fields[0]} has a value"
    return f"Only run if all of these fields have values: {', '.join(required_fields)}"


def default_output_schema(goal: str) -> List[Dict[str, Any]]:
    goal_l = (goal or "").lower()
    if "competitor" in goal_l:
        return [
            {"key": "competitors", "label": "Competitors", "type": "text", "primary": True},
            {"key": "description", "label": "Description", "type": "text", "primary": False},
            {"key": "priority", "label": "Priority", "type": "text", "primary": False},
        ]
    if "priority" in goal_l or "priorit" in goal_l:
        return [
            {"key": "priority", "label": "Priority", "type": "text", "primary": True},
            {"key": "reason", "label": "Reason", "type": "text", "primary": False},
        ]
    return [
        {"key": "result", "label": "Result", "type": "text", "primary": True},
        {"key": "summary", "label": "Summary", "type": "text", "primary": False},
    ]


def normalize_output_schema(schema: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    raw = list(schema or [])
    if not raw:
        raw = [{"key": "result", "label": "Result", "type": "text", "primary": True}]

    normalized: List[Dict[str, Any]] = []
    seen = set()
    primary_assigned = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = normalize_output_key(item.get("key") or item.get("label") or "result")
        if key in seen:
            continue
        seen.add(key)
        label = str(item.get("label") or labelize_key(key)).strip() or labelize_key(key)
        is_primary = bool(item.get("primary")) and not primary_assigned
        if is_primary:
            primary_assigned = True
        normalized.append(
            {
                "key": key,
                "label": label,
                "type": str(item.get("type") or "text"),
                "primary": is_primary,
            }
        )
    if not normalized:
        return [{"key": "result", "label": "Result", "type": "text", "primary": True}]
    if normalized and not primary_assigned:
        normalized[0]["primary"] = True
    return normalized


def map_raw_outputs_to_schema_keys(raw: Any, schema_keys: List[str]) -> Dict[str, str]:
    """
    Map model JSON `outputs` into schema keys. Models often use label-like keys
    ("Competitor Name") while the UI stores snake_case ("competitor_name"); normalize both sides.
    """
    if not isinstance(raw, dict):
        raw = {}
    by_norm: Dict[str, str] = {}
    for k, v in raw.items():
        nk = normalize_output_key(k)
        if not nk:
            continue
        s = str(v).strip() if v is not None else ""
        if not s:
            continue
        prev = by_norm.get(nk)
        if prev is None or len(s) > len(prev):
            by_norm[nk] = s
    out: Dict[str, str] = {}
    for sk in schema_keys:
        direct = str(raw.get(sk) or "").strip()
        if direct:
            out[sk] = direct
            continue
        nk = normalize_output_key(sk)
        out[sk] = by_norm.get(nk, "")
    return out


def build_field_catalog(
    profiles: Iterable[Dict[str, Any]],
    ai_columns: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for field in DEFAULT_FIELD_DEFINITIONS:
        grouped[field["group"]][field["key"]] = {
            "key": field["key"],
            "label": field["label"],
            "group": field["group"],
            "token": f"{{{field['key']}}}",
        }
    for field in ROLE_CONTEXT_FIELD_DEFINITIONS:
        grouped[field["group"]][field["key"]] = {
            "key": field["key"],
            "label": field["label"],
            "group": field["group"],
            "token": f"{{{field['key']}}}",
        }
    for field in COLUMN_CONTEXT_FIELD_DEFINITIONS:
        grouped[field["group"]][field["key"]] = {
            "key": field["key"],
            "label": field["label"],
            "group": field["group"],
            "token": f"{{{field['key']}}}",
        }
    for profile in profiles:
        for imported in flatten_raw_fields(profile.get("raw_fields")):
            grouped["Imported Fields"][imported["key"]] = {
                "key": imported["key"],
                "label": imported["label"],
                "group": "Imported Fields",
                "token": f"{{{imported['key']}}}",
                "sample": imported.get("sample", ""),
            }
    for column in ai_columns or []:
        slug = column.get("slug") or normalize_output_key(column.get("name"))
        outputs = normalize_output_schema(column.get("output_schema"))
        for output in outputs:
            key = f"ai.{slug}.{output['key']}"
            grouped["AI Columns"][key] = {
                "key": key,
                "label": f"{column.get('name')} · {output['label']}",
                "group": "AI Columns",
                "token": f"{{{key}}}",
                "ai_column_id": column.get("id"),
                "output_key": output["key"],
            }
    ordered_groups = [
        "Default Fields",
        "Role Fields",
        "Role Context",
        "Column Context",
        "Imported Fields",
        "AI Columns",
    ]
    return [
        {
            "group": group,
            "items": sorted(items.values(), key=lambda item: item["label"].lower()),
        }
        for group in ordered_groups
        if grouped.get(group)
        for items in [grouped[group]]
    ]


def get_profiles_for_scope(
    profiles_by_id: Dict[int, Dict[str, Any]],
    *,
    user_role: str,
    user_id: int,
    view_scope: Optional[str],
    recruiter_filter_id: Optional[int],
) -> List[Dict[str, Any]]:
    return [
        profile
        for profile in profiles_by_id.values()
        if profile_passes_scope(
            profile,
            user_role=user_role,
            user_id=user_id,
            view_scope=view_scope,
            recruiter_filter_id=recruiter_filter_id,
        )
    ]

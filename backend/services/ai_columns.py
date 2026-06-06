from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.services.candidate_pool import profile_passes_scope
from backend.services.imported_fields import (
    IMPORTED_EXTRA_FIELDS_KEY,
    iter_imported_extra_fields,
)


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

SMART_COLUMN_TOOL_INTENTS: Sequence[str] = (
    "career_locations",
    "tenure_metrics",
    "company_history",
    "role_history",
    "job_hopping",
    "current_role",
    "experience_summary",
    "function_experience",
    "industry_experience",
    "segment_experience",
    "geography_experience",
    "company_verification",
)
MAX_IMPORTED_FIELDS = 120
COMMUNITY_MEMBERSHIP_TERMS = (
    "revgenuis",
    "revgenius",
    "community",
    "member",
    "membership",
    "mentor",
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
                if prefix == "raw" and len(parts) == 1 and str(k) == IMPORTED_EXTRA_FIELDS_KEY:
                    continue
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


def flatten_imported_extra_fields(raw_fields: Any) -> List[Dict[str, Any]]:
    raw = safe_json(raw_fields)
    out: List[Dict[str, Any]] = []
    for key, meta in iter_imported_extra_fields(raw):
        label = str(meta.get("label") or labelize_key(key)).strip() or labelize_key(key)
        value = stringify_context_value(meta.get("value"))
        source_header = str(meta.get("source_header") or "").strip()
        out.append(
            {
                "key": f"extra.{key}",
                "label": label,
                "group": "Imported Extra Fields",
                "path": (IMPORTED_EXTRA_FIELDS_KEY, key, "value"),
                "sample": value[:180],
                "source_header": source_header,
            }
        )
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
    for imported in flatten_imported_extra_fields(profile.get("raw_fields")):
        raw_value = _get_nested_value(profile.get("raw_fields") or {}, imported["path"])
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


def flatten_profile_context(profile: Dict[str, Any], *, max_fields: int = 650) -> Dict[str, str]:
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


def _intent_text(prompt_template: str, output_schema: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    parts = [prompt_template or ""]
    for item in output_schema or []:
        if isinstance(item, dict):
            parts.append(str(item.get("key") or ""))
            parts.append(str(item.get("label") or ""))
    return " ".join(parts).lower()


def _text_has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def infer_smart_column_intents(
    prompt_template: str,
    output_schema: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[str]:
    """Infer the row/tool capabilities needed for a free-form Smart Column query.

    This is intentionally capability-oriented rather than answer-oriented: it
    maps arbitrary user wording to stable profile tools, then downstream code
    formats the answer from the selected tools.
    """
    text = _intent_text(prompt_template, output_schema)
    intents: List[str] = []

    def add(intent: str) -> None:
        if intent in SMART_COLUMN_TOOL_INTENTS and intent not in intents:
            intents.append(intent)

    if _text_has_any(text, ("city", "cities", "location", "locations", "worked from", "work from")):
        add("career_locations")
    if _text_has_any(text, ("geograph", "region", "regions", "market", "markets", "emea", "apac", "americas", "india", "singapore", "us market")):
        add("geography_experience")
    if _text_has_any(text, ("tenure", "stability", "current job", "current role tenure", "time spent", "average stay", "avg stay")):
        add("tenure_metrics")
    if _text_has_any(text, ("job hop", "hopper", "hopping", "short stint", "frequent switch", "switching jobs", "stability")):
        add("job_hopping")
    if _text_has_any(text, ("unique compan", "company history", "companies worked", "employers", "worked at", "career order", "company count", "list companies")):
        add("company_history")
    if _text_has_any(text, ("role history", "roles", "titles", "career path", "career order", "list role", "designation")):
        add("role_history")
    if _text_has_any(text, ("current company", "current title", "current employer", "current role", "present company", "present role")):
        add("current_role")
    if _text_has_any(text, ("account executive", " ae ", "sales", "sdr", "bdr", "hunting", "farming", "function experience")):
        add("function_experience")
    if _text_has_any(text, ("saas", "industry", "industries", "product", "service", "software", "fintech", "it services")):
        add("industry_experience")
    if _text_has_any(text, ("enterprise", "smb", "mid market", "mid-market", "segment", "customer segment")):
        add("segment_experience")
    if _text_has_any(text, ("company details", "company verification", "business model", "company size", "company website", "funding", "revenue", "competitor", "competitors")):
        add("company_verification")
    if _text_has_any(text, ("summarize", "summary", "profile", "candidate", "experience", "background")):
        add("experience_summary")

    if not intents and _text_has_any(text, ("work", "career", "job")):
        add("experience_summary")
    return intents


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

    if any(term in lower for term in WEB_FRESHNESS_TERMS) or any(
        term in lower for term in ("public evidence", "publicly", "web", "website", "news")
    ):
        return {
            "data_source": "web",
            "web_required_reason": "fresh_or_public_web_evidence_requested",
            "routing_mode": "web_research",
        }

    if urls and ("jd" in lower or "job description" in lower or "score" in lower or "fit" in lower):
        return {
            "data_source": "hybrid",
            "web_required_reason": "jd_url_or_public_fit_evidence",
            "routing_mode": "web_research",
        }

    if any(term in lower for term in ("fit score", "against this jd", "job description", "paste jd")) or (
        "score" in lower and any(term in lower for term in (" jd", "job description", "role fit", "candidate fit"))
    ):
        return {
            "data_source": "hybrid",
            "web_required_reason": "jd_fit_requires_public_verification",
            "routing_mode": "web_research",
        }

    if any(
        term in lower
        for term in COMPANY_ENRICHMENT_PROMPT_TERMS
    ):
        return {
            "data_source": "hybrid",
            "web_required_reason": "company_details_or_data_quality_verification",
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
    enriched_grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
    db_grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
    raw_grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
    for key, value in (context or {}).items():
        enriched_match = re.match(r"row\.raw_fields\.enrichment\.roles\.(\d+)\.(.+)", str(key))
        db_match = re.match(r"row\.roles\.(\d+)\.(.+)", str(key))
        raw_match = re.match(r"row\.raw_fields\.experiences/(\d+)/(.+)", str(key))
        if raw_match:
            text = stringify_context_value(value)
            if not text:
                continue
            idx = int(raw_match.group(1))
            field = raw_match.group(2)
            mapped = {
                "companyName": "company",
                "title": "title",
                "jobStartedOn": "start_date",
                "jobEndedOn": "end_date",
                "companyIndustry": "source_industry",
                "companyWebsite": "source_website",
                "companySize": "source_company_size",
                "jobLocation": "location",
                "jobLocationCountry": "location",
            }.get(field)
            if mapped:
                raw_grouped[idx][mapped] = text
                if mapped == "source_industry":
                    raw_grouped[idx].setdefault("industry", text)
            continue
        match = enriched_match or db_match
        if not match:
            continue
        text = stringify_context_value(value)
        if not text:
            continue
        target = enriched_grouped if enriched_match else db_grouped
        target[int(match.group(1))][match.group(2)] = text

    # Verified import enrichment keeps original role dates/details in raw_fields.
    # Prefer that source for tenure so DB roles without dates cannot erase it.
    grouped = enriched_grouped if enriched_grouped else db_grouped
    for idx, raw_role in raw_grouped.items():
        if idx not in grouped:
            grouped[idx] = dict(raw_role)
            continue
        role = grouped[idx]
        for key, raw_value in raw_role.items():
            current = stringify_context_value(role.get(key))
            if not current or current.lower() == "unknown":
                role[key] = raw_value
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
    if re.search(r"\b(member|mentor|volunteer)\b", combined):
        return True
    return any(
        re.search(rf"\b{re.escape(term)}\b", combined)
        for term in COMMUNITY_MEMBERSHIP_TERMS
        if term not in {"member", "mentor", "volunteer"}
    )


def _parse_role_date(value: str, *, default_current: bool = False) -> Optional[datetime]:
    text = stringify_context_value(value).strip()
    if not text:
        return datetime.now(timezone.utc) if default_current else None
    lower = text.lower()
    if lower in {"present", "current", "now", "till date", "ongoing"}:
        return datetime.now(timezone.utc)

    try:
        clean_str = text.replace('Z', '+00:00')
        parsed = datetime.fromisoformat(clean_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        pass
    patterns = (
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


def _role_duration_months(role: Dict[str, str]) -> int:
    for key in ("duration_months", "months"):
        try:
            value = int(float(stringify_context_value(role.get(key)).replace(",", "")))
            if value > 0:
                return value
        except Exception:
            continue
    for key in ("duration_years", "years"):
        try:
            value = float(stringify_context_value(role.get(key)).replace(",", ""))
            if value > 0:
                return int(round(value * 12))
        except Exception:
            continue
    return 0


def _role_date_duration_months(role: Dict[str, Any], *, default_current: bool = False) -> int:
    start = _parse_role_date(role.get("start_date") or role.get("starts_at") or role.get("start"))
    end_raw = role.get("end_date") or role.get("ends_at") or role.get("end")
    end = _parse_role_date(end_raw, default_current=default_current or bool(start and not stringify_context_value(end_raw)))
    return _months_between(start, end)


def _current_company_enterprise_saas_status(context: Dict[str, str], current_role: Dict[str, str]) -> Tuple[str, List[str]]:
    fields = {
        "current role product/service": current_role.get("product_service") or context.get("role.company_product_service"),
        "current role industry": current_role.get("industry")
        or current_role.get("source_industry")
        or context.get("row.raw_fields.experiences/0/companyIndustry"),
        "current role customer segment": current_role.get("customer_segment")
        or context.get("role.company_customer_segment")
        or context.get("row.roles.0.company_details.customer_segment"),
        "current role business model": current_role.get("business_model")
        or context.get("row.roles.0.company_details.business_model"),
        "current role company size": current_role.get("source_company_size")
        or context.get("row.raw_fields.experiences/0/companySize"),
        "current role company website": current_role.get("source_website")
        or context.get("row.raw_fields.experiences/0/companyWebsite"),
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


def _city_from_location(value: Any) -> str:
    text = stringify_context_value(value)
    if not text:
        return ""
    first = re.split(r"[,|/]", text, maxsplit=1)[0].strip().lower()
    if not first:
        return ""
    non_city_terms = {
        "remote",
        "hybrid",
        "onsite",
        "apac",
        "emea",
        "latam",
        "americas",
        "india",
        "australia",
        "usa",
        "united states",
        "canada",
    }
    return "" if first in non_city_terms else first


def _place_from_location(value: Any) -> str:
    text = stringify_context_value(value)
    if not text:
        return ""
    first = re.split(r"[,|/]", text, maxsplit=1)[0].strip().lower()
    if not first or first in {"remote", "hybrid", "onsite"}:
        return ""
    return first


def compute_career_facts(context: Dict[str, str]) -> Dict[str, Any]:
    roles = [role for role in _parse_role_context_entries(context) if not _role_is_community_membership(role)]
    companies: Dict[str, Dict[str, Any]] = {}
    cities = set()
    places = set()
    ae_months = 0
    undated_duration_months = 0
    role_tenures: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[datetime, datetime]] = []
    # Bug-5 fix: ensure the most-recent role is always at index 0 so that
    # `idx == 0` correctly identifies the current employer when setting
    # default_current=True on a missing end date.  Sort by end_date descending
    # with None (still-current) roles first.
    def _role_sort_key(r: Dict[str, str]) -> tuple:
        end_raw = r.get("end_date") or r.get("ends_at") or r.get("end") or ""
        end_text = stringify_context_value(end_raw).lower()
        if not end_text or end_text in {"present", "current", "now", "till date", "ongoing"}:
            return (1, datetime.max.replace(tzinfo=timezone.utc))  # current roles sort first
        parsed = _parse_role_date(end_raw)
        return (0, parsed) if parsed else (0, datetime.min.replace(tzinfo=timezone.utc))

    roles = sorted(roles, key=_role_sort_key, reverse=True)

    current_role = roles[0] if roles else {}
    current_company = stringify_context_value(current_role.get("company"))
    current_company_norm = _normalize_company_name(current_company)

    for idx, role in enumerate(roles):
        company = stringify_context_value(role.get("company"))
        normalized_company = _normalize_company_name(company)
        if not normalized_company:
            continue
        start = _parse_role_date(role.get("start_date") or role.get("starts_at") or role.get("start"))
        end_raw = role.get("end_date") or role.get("ends_at") or role.get("end")
        end = _parse_role_date(end_raw, default_current=idx == 0)
        months = _months_between(start, end) or _role_duration_months(role)
        title = stringify_context_value(role.get("title"))
        location = stringify_context_value(role.get("city") or role.get("location"))
        role_tenures.append(
            {
                "company": company,
                "title": title,
                "start_date": start.date().isoformat() if start else "",
                "end_date": end.date().isoformat() if end else "",
                "months": months,
                "years": round(months / 12, 2) if months else 0,
            }
        )
        city = _city_from_location(location)
        if city:
            cities.add(city)
        place = _place_from_location(location)
        if place:
            places.add(place)
        bucket = companies.setdefault(
            normalized_company,
            {"company": company, "months": 0, "intervals": [], "undated_months": 0, "titles": []},
        )
        bucket["months"] += months
        if start and end:
            all_intervals.append((start, end))
            bucket["intervals"].append((start, end))
        elif months:
            undated_duration_months += months
            bucket["undated_months"] += months
        if title:
            bucket["titles"].append(title)
        if _role_is_account_executive(role):
            ae_months += months

    for company in companies.values():
        intervals = company.get("intervals") or []
        if intervals:
            company["months"] = sum(
                _months_between(start, end)
                for start, end in _merge_intervals(intervals)
            ) + int(company.get("undated_months") or 0)

    company_tenures = [
        {
            "company": company.get("company") or "",
            "months": int(company.get("months") or 0),
            "years": round(int(company.get("months") or 0) / 12, 2) if company.get("months") else 0,
            "titles": company.get("titles") or [],
            "is_current_company": normalized_company == current_company_norm,
        }
        for normalized_company, company in companies.items()
    ]
    completed_company_tenures = [
        tenure for tenure in company_tenures
        if not tenure.get("is_current_company")
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
    total_months_from_durations = sum(int(role.get("months") or 0) for role in role_tenures)
    total_exp_years = _float_context(context, "candidate.total_experience_years")
    total_months = (
        (total_months_from_roles + undated_duration_months)
        if total_months_from_roles
        else total_months_from_durations
        or int(round(total_exp_years * 12))
    )
    unique_company_count = len(companies)
    completed_company_count = len(completed_company_tenures)
    completed_company_months = sum(int(company.get("months") or 0) for company in completed_company_tenures)
    average_tenure_months = (
        int(round(completed_company_months / completed_company_count))
        if completed_company_count
        else 0
    )
    job_hopping_flag = (
        len(very_short_company_stints) >= 2
        or len(short_company_stints) >= 3
        or (completed_company_count >= 2 and 0 < average_tenure_months < 18)
    )
    if not unique_company_count:
        job_hopping_status = "Unknown"
        job_hopping_reason = "No dated company tenure available."
    elif not completed_company_count:
        job_hopping_status = "Unknown"
        job_hopping_reason = "No completed company tenure available after excluding the current company."
    elif job_hopping_flag:
        job_hopping_status = "Yes"
        job_hopping_reason = (
            f"{len(short_company_stints)} company stint(s) under 24 months; "
            f"average completed-company tenure is {average_tenure_months} months."
        )
    else:
        job_hopping_status = "No"
        job_hopping_reason = (
            f"Average completed-company tenure is {average_tenure_months} months with "
            f"{len(short_company_stints)} stint(s) under 24 months."
        )

    current_job_months = _role_date_duration_months(current_role, default_current=True) or _role_duration_months(current_role)
    candidate_city = _city_from_location(context.get("candidate.city"))
    if candidate_city:
        cities.add(candidate_city)
    candidate_place = _place_from_location(context.get("candidate.city") or context.get("candidate.location"))
    if candidate_place:
        places.add(candidate_place)
    current_company_enterprise_saas, current_company_evidence = _current_company_enterprise_saas_status(context, current_role)

    return {
        "total_experience_months": total_months,
        "total_experience_years": round(total_months / 12, 1) if total_months else total_exp_years,
        "unique_company_count": unique_company_count,
        "average_tenure_months": average_tenure_months,
        "completed_company_count": completed_company_count,
        "completed_company_months": completed_company_months,
        "completed_company_tenures": completed_company_tenures,
        "current_company": current_company,
        "current_company_tenure_months": next(
            (
                int(company.get("months") or 0)
                for company in company_tenures
                if company.get("is_current_company")
            ),
            0,
        ),
        "current_job_months": current_job_months,
        "ae_experience_months": ae_months,
        "ae_experience_years": round(ae_months / 12, 1),
        "career_city_count": len(cities),
        "career_cities": sorted(cities),
        "career_location_count": len(places),
        "career_locations": sorted(places),
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
    avg_months = facts.get('average_tenure_months') or 0
    avg_years = round(avg_months / 12, 1) if avg_months else 0
    return (
        "Deterministic row-derived career facts (DO NOT RECALCULATE — use these values exactly):\n"
        "NOTE: These were computed by an overlap-aware algorithm with correct month arithmetic.\n"
        "Recalculating from raw dates will always be wrong (LinkedIn dates are month-precision, day=1).\n"
        f"- total_experience_months: {facts.get('total_experience_months') or 0}\n"
        f"- total_experience_years: {facts.get('total_experience_years') or 0}\n"
        f"- unique_company_count: {facts.get('unique_company_count') or 0}\n"
        f"- completed_company_count: {facts.get('completed_company_count') or 0} (current employer excluded)\n"
        f"- completed_company_months: {facts.get('completed_company_months') or 0}\n"
        f"- average_tenure_months_completed_roles: {avg_months} ({avg_years} yrs) — current employer excluded\n"
        f"- current_company: {facts.get('current_company') or 'unknown'}\n"
        f"- current_job_months: {facts.get('current_job_months') or 0}\n"
        f"- ae_experience_months: {facts.get('ae_experience_months') or 0}\n"
        f"- ae_experience_years: {facts.get('ae_experience_years') or 0}\n"
        f"- current_company_enterprise_saas: {facts.get('current_company_enterprise_saas') or 'Needs web verification'}\n"
        f"- job_hopping_status: {facts.get('job_hopping_status') or 'Unknown'}\n"
        f"- job_hopping_reason: {facts.get('job_hopping_reason') or ''}\n"
        f"- career_city_count: {facts.get('career_city_count') or 0}\n"
        f"- companies_counted: {', '.join(facts.get('companies') or []) or 'unknown'}\n"
        f"- role_tenures: {'; '.join(role_lines) if role_lines else 'unknown'}"
    )


def _display_list(values: Sequence[Any]) -> str:
    return ", ".join(
        stringify_context_value(value).strip().title()
        for value in values
        if stringify_context_value(value).strip()
    )


def _ordered_companies(facts: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for role in facts.get("role_tenures") or []:
        company = stringify_context_value(role.get("company"))
        norm = _normalize_company_name(company)
        if company and norm and norm not in seen:
            out.append(company)
            seen.add(norm)
    for company in facts.get("companies") or []:
        norm = _normalize_company_name(company)
        if company and norm and norm not in seen:
            out.append(company)
            seen.add(norm)
    return out


def _format_role_history(facts: Dict[str, Any], *, limit: int = 6) -> str:
    roles = facts.get("role_tenures") or []
    if not roles:
        return "No role history found in profile."
    lines = []
    for role in roles[:limit]:
        title = stringify_context_value(role.get("title")) or "Unknown title"
        company = stringify_context_value(role.get("company")) or "Unknown company"
        start = stringify_context_value(role.get("start_date")) or "?"
        end = stringify_context_value(role.get("end_date")) or "?"
        lines.append(f"{title} at {company} ({start} to {end})")
    return "Career order, current to past: " + "; ".join(lines)


def _format_current_role(facts: Dict[str, Any]) -> str:
    role = (facts.get("role_tenures") or [{}])[0] if facts.get("role_tenures") else {}
    company = stringify_context_value(facts.get("current_company") or role.get("company"))
    title = stringify_context_value(role.get("title"))
    if company and title:
        return f"Current role: {title} at {company}."
    if company:
        return f"Current company: {company}. Current title not found in profile."
    return "No current role data found in profile."


def _format_tenure_answer(facts: Dict[str, Any]) -> str:
    if not facts.get("role_tenures") and not facts.get("company_tenures"):
        return "No tenure data found in profile."
    return (
        f"Average tenure across completed companies: {facts.get('average_tenure_months') or 0} months. "
        f"Current job tenure: {facts.get('current_job_months') or 0} months. "
        f"Overall experience: {facts.get('total_experience_months') or 0} months."
    )


def _wants_places(prompt: str) -> bool:
    prompt_l = (prompt or "").lower()
    return any(term in prompt_l for term in ("place", "places", "location", "locations")) and "city" not in prompt_l


def _format_locations_answer(prompt: str, facts: Dict[str, Any]) -> str:
    if _wants_places(prompt):
        locations = facts.get("career_locations") or []
        count = int(facts.get("career_location_count") or 0)
        if not count:
            return "No location data found in profile."
        label = "place/location" if count == 1 else "places/locations"
        return f"{count} {label}: {_display_list(locations)}."
    cities = facts.get("career_cities") or []
    count = int(facts.get("career_city_count") or 0)
    if not count:
        return "No city data found in profile."
    label = "city" if count == 1 else "cities"
    return f"{count} {label}: {_display_list(cities)}."


def _format_company_history_answer(facts: Dict[str, Any]) -> str:
    companies = _ordered_companies(facts)
    if not companies:
        return "No company history found in profile."
    label = "company" if len(companies) == 1 else "companies"
    return f"{len(companies)} unique {label}, current to past: {', '.join(companies)}."


def _format_job_hopping_answer(facts: Dict[str, Any]) -> str:
    status = stringify_context_value(facts.get("job_hopping_status") or "Unknown")
    reason = stringify_context_value(facts.get("job_hopping_reason"))
    return f"Job hopping: {status}. {reason}".strip()


def _format_stability_answer(facts: Dict[str, Any]) -> str:
    status = stringify_context_value(facts.get("job_hopping_status") or "Unknown")
    reason = stringify_context_value(facts.get("job_hopping_reason"))
    return (
        f"Stability: {status}. "
        f"Average completed-company tenure: {facts.get('average_tenure_months') or 0} months. "
        f"Current job tenure: {facts.get('current_job_months') or 0} months. "
        f"{reason}"
    ).strip()


def _format_function_answer(prompt: str, facts: Dict[str, Any]) -> str:
    prompt_l = (prompt or "").lower()
    if "account executive" in prompt_l or " ae " in f" {prompt_l} ":
        return f"Account Executive experience: {facts.get('ae_experience_months') or 0} months."
    roles = [
        role for role in (facts.get("role_tenures") or [])
        if re.search(r"\b(sales|account|business development|sdr|bdr|executive)\b", stringify_context_value(role.get("title")).lower())
    ]
    if roles:
        return "Sales experience found in roles: " + "; ".join(
            f"{role.get('title') or 'Unknown title'} at {role.get('company') or 'Unknown company'}"
            for role in roles[:5]
        )
    return "No matching sales/function experience found in profile."


def _format_industry_answer(prompt: str, facts: Dict[str, Any]) -> str:
    if "saas" in (prompt or "").lower():
        status = stringify_context_value(facts.get("current_company_enterprise_saas") or "Needs web verification")
        evidence = facts.get("current_company_enterprise_saas_evidence") or []
        suffix = f" Evidence: {'; '.join(evidence[:3])}." if evidence else ""
        return f"SaaS company signal: {status}.{suffix}"
    return "Industry-specific answer needs the profile industry fields or company enrichment."


def _format_geography_answer(facts: Dict[str, Any]) -> str:
    cities = facts.get("career_cities") or []
    if cities:
        return f"Geography/location signals from profile: {_display_list(cities)}."
    return "No geography or market data found in profile."


def _format_experience_summary(facts: Dict[str, Any]) -> str:
    companies = _ordered_companies(facts)
    current = _format_current_role(facts)
    return (
        f"{current} Total experience: {facts.get('total_experience_months') or 0} months "
        f"across {facts.get('unique_company_count') or len(companies)} companies."
    )


def _intent_specific_answer(prompt: str, facts: Dict[str, Any], intents: Sequence[str]) -> str:
    if not facts:
        return "No profile data found for this query."
    ordered = list(intents or infer_smart_column_intents(prompt))
    if "stability" in (prompt or "").lower() and ("job_hopping" in ordered or "tenure_metrics" in ordered):
        return _format_stability_answer(facts)
    if "career_locations" in ordered:
        return _format_locations_answer(prompt, facts)
    if "tenure_metrics" in ordered:
        return _format_tenure_answer(facts)
    if "company_history" in ordered:
        return _format_company_history_answer(facts)
    if "role_history" in ordered:
        return _format_role_history(facts)
    if "job_hopping" in ordered:
        return _format_job_hopping_answer(facts)
    if "current_role" in ordered:
        return _format_current_role(facts)
    if "function_experience" in ordered:
        return _format_function_answer(prompt, facts)
    if "industry_experience" in ordered:
        return _format_industry_answer(prompt, facts)
    if "geography_experience" in ordered:
        return _format_geography_answer(facts)
    if "experience_summary" in ordered:
        return _format_experience_summary(facts)
    return "No profile data found for this query."


def _intent_specific_reason(prompt: str, intents: Sequence[str]) -> str:
    names = ", ".join(intents or infer_smart_column_intents(prompt) or ["profile_lookup"])
    return f"Answered from selected profile tool intent(s): {names}."


def map_career_facts_to_outputs(
    prompt_template: str,
    output_schema: Sequence[Dict[str, Any]],
    facts: Dict[str, Any],
) -> Dict[str, str]:
    prompt_l = (prompt_template or "").lower()
    intents = infer_smart_column_intents(prompt_template, output_schema)
    if not intents or not facts:
        return {}
    asks_thresholded_fit = bool(re.search(r"\b\d+\+?\s*(?:years?|yrs?|months?|mos?)\b", prompt_l))
    if asks_thresholded_fit and {"function_experience", "segment_experience", "geography_experience"}.issubset(set(intents)):
        return {}

    # Check if all keys in output_schema can be deterministically satisfied.
    # If any key falls to the 'else' block (i.e. is not a known deterministic field),
    # we return {} to let the LLM handle it instead of hijacking it with a flat summary.
    for item in normalize_output_schema(output_schema):
        key = item["key"]
        label = f"{key} {item.get('label', '')}".lower()
        
        is_supported = (
            ("average" in label and "tenure" in label)
            or ("current" in label and ("tenure" in label or "month" in label or "current_job" in key))
            or ("city" in label and "count" in label)
            or ("cities" in label)
            or ("location" in label and "count" in label)
            or ("place" in label and "count" in label)
            or ("locations" in label or "places" in label)
            or ("company" in label and "count" in label)
            or ("company" in label and ("history" in label or "list" in label))
            or ("role" in label and "history" in label)
            or key == "current_role"
            or ("current" in label and ("company" in label or "title" in label))
            or ("total" in label and ("experience" in label or "month" in label))
            or ("ae" in label or "account_executive" in label or "account executive" in label)
            or ("sales" in label and "experience" in label)
            or ("saas" in label)
            or ("geograph" in label or "market" in label or "location" in label)
            or ("enterprise" in label and "saas" in label)
            or ("job" in label and ("hop" in label or "switch" in label))
            or ("short" in label and "stint" in label)
            or ("qualified" in label or "qualification" in label or "eligible" in label)
            or ("reasoning" in label)
            or (key in {"result", "summary", "reasoning"} or key.endswith("_reasoning"))
        )
        if not is_supported:
            return {}  # Abort deterministic mapping; LLM is required!

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
        f"Average tenure (completed roles): {facts.get('average_tenure_months') or 0} months. "
        f"Current job tenure: {facts.get('current_job_months') or 0} months. "
        f"Overall experience: {facts.get('total_experience_months') or 0} months. "
        f"Account executive experience: {facts.get('ae_experience_months') or 0} months. "
        f"Current enterprise SaaS company: {facts.get('current_company_enterprise_saas') or 'Needs web verification'}. "
        f"Job hopping: {facts.get('job_hopping_status') or 'Unknown'} "
        f"({facts.get('job_hopping_reason') or 'No reason available'}). "
        f"Unique companies counted: {facts.get('unique_company_count') or 0}. "
        f"Completed companies counted for average tenure: {facts.get('completed_company_count') or 0}. "
        f"Career cities counted: {facts.get('career_city_count') or 0}."
    )
    if yes_no:
        summary += f" Qualification: {yes_no}."
    intent_answer = _intent_specific_answer(prompt_template, facts, intents)
    intent_reason = _intent_specific_reason(prompt_template, intents)

    for item in normalize_output_schema(output_schema):
        key = item["key"]
        label = f"{key} {item.get('label', '')}".lower()
        if "average" in label and "tenure" in label:
            outputs[key] = str(facts.get("average_tenure_months") or 0)
        elif "current" in label and ("tenure" in label or "month" in label or "current_job" in key):
            outputs[key] = str(facts.get("current_job_months") or 0)
        elif "city" in label and "count" in label:
            outputs[key] = str(facts.get("career_city_count") or 0)
        elif "cities" in label:
            outputs[key] = ", ".join(facts.get("career_cities") or [])
        elif ("location" in label or "place" in label) and "count" in label:
            outputs[key] = str(facts.get("career_location_count") or 0)
        elif "locations" in label or "places" in label:
            outputs[key] = ", ".join(facts.get("career_locations") or [])
        elif "company" in label and "count" in label:
            outputs[key] = str(facts.get("unique_company_count") or 0)
        elif "company" in label and ("history" in label or "list" in label):
            outputs[key] = ", ".join(_ordered_companies(facts))
        elif "role" in label and "history" in label:
            outputs[key] = _format_role_history(facts)
        elif key == "current_role" or ("current" in label and ("company" in label or "title" in label)):
            outputs[key] = _format_current_role(facts)
        elif "total" in label and ("experience" in label or "month" in label):
            outputs[key] = str(facts.get("total_experience_months") or 0)
        elif "ae" in label or "account_executive" in label or "account executive" in label:
            outputs[key] = str(facts.get("ae_experience_months") or 0)
        elif "sales" in label and "experience" in label:
            outputs[key] = _format_function_answer(prompt_template, facts)
        elif "enterprise" in label and "saas" in label:
            outputs[key] = str(facts.get("current_company_enterprise_saas") or "Needs web verification")
        elif "saas" in label:
            outputs[key] = _format_industry_answer(prompt_template, facts)
        elif "geograph" in label or "market" in label or "location" in label:
            outputs[key] = _format_geography_answer(facts)
        elif "job" in label and ("hop" in label or "switch" in label):
            outputs[key] = str(facts.get("job_hopping_status") or "Unknown")
        elif "short" in label and "stint" in label:
            outputs[key] = str(facts.get("short_company_stints_count") or 0)
        elif "qualified" in label or "qualification" in label or "eligible" in label:
            outputs[key] = yes_no or "Needs web verification"
        elif "reasoning" in label or key.endswith("_reasoning"):
            outputs[key] = summary if "tenure" in label else intent_reason
        elif key == "result":
            outputs[key] = intent_answer
        elif key == "summary":
            outputs[key] = summary if set(intents).issubset({"experience_summary"}) else intent_answer
        elif key == "reasoning":
            outputs[key] = intent_reason
        else:
            outputs[key] = intent_answer
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
        "imported_extra_fields": _flat_context_section(context, "extra"),
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


COMPANY_ENRICHMENT_PROMPT_TERMS = (
    "all company",
    "all companies",
    "current company",
    "current employer",
    "company details",
    "current company details",
    "company classification",
    "company enrichment",
    "company size",
    "company website",
    "employer",
    "industry",
    "product/service",
    "product service",
    "business model",
    "customer segment",
    "segment",
    "customer presence",
    "enterprise",
    "saas",
    "funding",
    "revenue",
    "headquarters",
    "competitor",
    "competitors",
    "data quality",
)


def _prompt_needs_company_enrichment(prompt: str) -> bool:
    return _prompt_has_any(prompt, COMPANY_ENRICHMENT_PROMPT_TERMS)


def _role_value_unknown(value: Any) -> bool:
    text = stringify_context_value(value).strip()
    return not text or text.lower() in {"unknown", "n/a", "na", "none", "null", "[]", "{}"}


def _role_company_detail_gaps(role: Dict[str, Any]) -> List[str]:
    gaps: List[str] = []
    checks = {
        "industry": role.get("industry") or role.get("source_industry") or role.get("company_details.industry"),
        "product_service": role.get("product_service") or role.get("company_details.product_service"),
        "business_model": role.get("business_model") or role.get("company_details.business_model"),
        "customer_segment": role.get("customer_segment") or role.get("company_details.customer_segment"),
        "company_website": role.get("source_website") or role.get("website") or role.get("companyWebsite"),
        "company_size": role.get("source_company_size") or role.get("company_size") or role.get("companySize"),
    }
    for key, value in checks.items():
        if _role_value_unknown(value):
            gaps.append(key)
    return gaps


def _company_detail_missing_info(roles: Sequence[Dict[str, Any]]) -> List[str]:
    missing: List[str] = []
    for role in roles:
        company = stringify_context_value(role.get("company")) or "Unknown company"
        gaps = _role_company_detail_gaps(role)
        if gaps:
            missing.append(f"{company}: missing {', '.join(gaps)}")
    return missing


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
    "Account Development": ("account development", "account development representative", "adr"),
    "Channel Sales": ("channel sales", "channel partner", "partner sales", "reseller", "alliances"),
    "Hunting": ("hunting", "new business", "new logo"),
    "Inside Sales": ("inside sales", "inbound sales", "remote sales"),
    "Farming": ("farming", "account management", "expansion", "renewal"),
    "Sales Development": ("sdr", "bdr", "sales development", "business development representative"),
    "Customer Success": ("customer success", "csm", "customer success manager"),
    "Partnerships": ("partner", "channel", "alliances"),
}
GEOGRAPHY_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "EMEA": ("emea", "europe", "middle east", "africa"),
    "APAC": ("apac", "asia pacific", "asia-pacific"),
    "LATAM": ("latam", "latin america", "brazil", "mexico"),
    "Americas": ("americas", "north america", "usa", "united states", "canada"),
    "India": ("india", "mumbai", "bengaluru", "bangalore", "delhi"),
    "Singapore": ("singapore",),
    "Australia": ("australia", "sydney", "melbourne"),
}
INDUSTRY_QUERY_TERMS: Dict[str, Sequence[str]] = {
    "SaaS": ("saas", "software", "cloud platform", "subscription"),
    "Customer Engagement": (
        "customer engagement",
        "clevertap",
        "moengage",
        "webengage",
        "braze",
        "customer.io",
        "customer io",
        "retention platform",
        "lifecycle engagement",
        "marketing automation",
        "crm",
    ),
    "Fintech": ("fintech", "payments", "banking", "lending"),
    "HRTech": ("hrtech", "hr tech", "recruiting", "talent"),
    "BPO": ("bpo", "business process outsourcing"),
    "Analytics": ("analytics", "data platform", "business intelligence"),
}
REGION_COUNTRIES: Dict[str, Sequence[str]] = {
    "APAC": (
        "india",
        "singapore",
        "australia",
        "japan",
        "indonesia",
        "malaysia",
        "philippines",
        "thailand",
        "vietnam",
        "new zealand",
    ),
    "EMEA": ("united kingdom", "uk", "germany", "france", "uae", "dubai", "south africa"),
    "LATAM": ("brazil", "mexico", "argentina", "chile", "colombia"),
    "Americas": ("usa", "united states", "canada", "mexico", "brazil"),
}
COUNTRY_TO_REGIONS = {
    country: region
    for region, countries in REGION_COUNTRIES.items()
    for country in countries
}
COMPETITOR_QUERY_TARGETS: Dict[str, Sequence[str]] = {
    "CleverTap": ("clevertap", "clever tap"),
}
COMPETITOR_COMPANIES: Dict[str, Sequence[str]] = {
    "CleverTap": (
        "moengage",
        "webengage",
        "netcore",
        "netcore cloud",
        "braze",
        "iterable",
        "customer.io",
        "customer io",
        "onesignal",
        "insider",
    ),
}


def _role_text(role: Dict[str, Any]) -> str:
    selected = " ".join(
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
            "source_location",
            "company_location",
            "headquarters",
            "company_details.headquarters",
        )
    )
    all_values = " ".join(stringify_context_value(value) for value in (role or {}).values())
    return f"{selected} {all_values}".lower()


def _matched_duration(roles: Sequence[Dict[str, Any]], terms: Sequence[str]) -> Dict[str, Any]:
    matches = []
    total_months = 0
    for role in roles:
        role_text = _role_text(role)
        if not any(term.lower() in role_text for term in terms):
            continue
        months = _role_date_duration_months(role) or _role_duration_months(role)
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


def _role_months(role: Dict[str, Any]) -> int:
    return _role_date_duration_months(role) or _role_duration_months(role)


def _expanded_geography_terms(term: str) -> List[str]:
    canonical = stringify_context_value(term)
    values = [canonical, *GEOGRAPHY_QUERY_TERMS.get(canonical, ())]
    if canonical in REGION_COUNTRIES:
        values.extend(REGION_COUNTRIES[canonical])
    lower_values = {value.lower() for value in values if value}
    for value in list(lower_values):
        region = COUNTRY_TO_REGIONS.get(value)
        if region:
            lower_values.add(region.lower())
            lower_values.update(v.lower() for v in GEOGRAPHY_QUERY_TERMS.get(region, ()))
    return sorted(lower_values)


def _matched_geography_duration(
    roles: Sequence[Dict[str, Any]],
    term: str,
    profile_claims: Dict[str, Any],
) -> Dict[str, Any]:
    terms = _expanded_geography_terms(term)
    matches = []
    total_months = 0
    for role in roles:
        role_text = _role_text(role)
        if not any(re.search(rf"\b{re.escape(value)}\b", role_text) for value in terms if value):
            continue
        months = _role_months(role)
        total_months += months
        matches.append(
            {
                "company": stringify_context_value(role.get("company")),
                "title": stringify_context_value(role.get("title")),
                "months": months,
                "evidence": role_text[:240],
            }
        )
    raw_claim_values = (profile_claims or {}).get("geographies", [])
    if isinstance(raw_claim_values, str):
        raw_claim_values = re.split(r"[,;|]", raw_claim_values)
    claim_values = [stringify_context_value(value) for value in raw_claim_values]
    claim_text = " ".join(claim_values).lower()
    profile_claim_match = any(
        re.search(rf"\b{re.escape(value)}\b", claim_text)
        for value in terms
        if value
    )
    return {
        "months": total_months,
        "years": round(total_months / 12, 1),
        "matches": matches,
        "profile_claim_match": bool(profile_claim_match),
        "profile_claims": claim_values,
        "expanded_terms": terms,
    }


def _prompt_competitor_targets(prompt: str) -> List[str]:
    lower = (prompt or "").lower()
    if "competitor" not in lower and "competitors" not in lower:
        return []
    found: List[str] = []
    for target, aliases in COMPETITOR_QUERY_TARGETS.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases):
            found.append(target)
    return found


def _competitor_match(roles: Sequence[Dict[str, Any]], target: str) -> Dict[str, Any]:
    competitors = [_normalize_company_name(value) for value in COMPETITOR_COMPANIES.get(target, ())]
    if not competitors:
        return {"status": "Needs competitor taxonomy", "target": target, "matches": []}
    current_role = roles[0] if roles else {}
    current_company_norm = _normalize_company_name(stringify_context_value(current_role.get("company")))
    is_current_match = current_company_norm in competitors
    matches = []
    for idx, role in enumerate(roles):
        company = stringify_context_value(role.get("company"))
        if _normalize_company_name(company) in competitors:
            matches.append(
                {
                    "company": company,
                    "title": stringify_context_value(role.get("title")),
                    "is_current": idx == 0,
                    "months": _role_months(role),
                }
            )
    return {
        "status": "Yes" if is_current_match else "No",
        "target": target,
        "current_company": stringify_context_value(current_role.get("company")),
        "competitors": list(COMPETITOR_COMPANIES.get(target, ())),
        "matches": matches,
    }


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
    tool_intents = infer_smart_column_intents(prompt, output_schema)
    wants_career_metrics = _prompt_has_any(
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
    )
    wants_company_verification = _prompt_has_any(
        prompt,
        (
            "currently working",
            "public",
            "publicly",
            "recent",
            "latest",
            "layoff",
            "website",
            "news",
            *COMPANY_ENRICHMENT_PROMPT_TERMS,
        ),
    )
    if wants_career_metrics:
        tool_calls.append("career_metrics")
    if any(intent in tool_intents for intent in ("career_locations", "tenure_metrics", "company_history", "role_history", "experience_summary", "current_role")):
        tool_calls.append("career_metrics")
    if "career_locations" in tool_intents:
        tool_calls.append("career_locations")
    if "company_history" in tool_intents:
        tool_calls.append("company_history")
    if "role_history" in tool_intents:
        tool_calls.append("role_history")
    if "current_role" in tool_intents:
        tool_calls.append("profile_lookup")
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
    if "job_hopping" in tool_intents:
        tool_calls.extend(["career_metrics", "job_hopping"])
    if _prompt_terms(prompt, FUNCTION_QUERY_TERMS):
        tool_calls.append("functional_experience")
    if "function_experience" in tool_intents:
        tool_calls.append("functional_experience")
    if _prompt_terms(prompt, SEGMENT_QUERY_TERMS):
        tool_calls.append("segment_experience")
    if "segment_experience" in tool_intents:
        tool_calls.append("segment_experience")
    if _prompt_terms(prompt, GEOGRAPHY_QUERY_TERMS):
        tool_calls.append("geography_experience")
    if "geography_experience" in tool_intents:
        tool_calls.append("geography_experience")
    if _prompt_terms(prompt, INDUSTRY_QUERY_TERMS) or _prompt_has_any(prompt, ("industry", "product", "service")):
        tool_calls.append("industry_experience")
    if "industry_experience" in tool_intents:
        tool_calls.append("industry_experience")
    if _prompt_competitor_targets(prompt):
        tool_calls.append("competitor_match")
    if wants_company_verification or "company_verification" in tool_intents:
        tool_calls.append("company_verification")
    if not tool_calls:
        tool_calls.append("profile_lookup")

    current_status = ""
    facts = compute_career_facts(context)
    context_pack = build_candidate_context_pack(context, facts)
    roles = context_pack.get("roles") or []
    company_missing_info = (
        _company_detail_missing_info(roles)
        if wants_company_verification or _prompt_needs_company_enrichment(prompt)
        else []
    )
    if facts:
        current_status = stringify_context_value(facts.get("current_company_enterprise_saas"))
    urls = extract_urls(prompt)
    web_needed = routing.get("data_source") == "web"
    if routing.get("data_source") == "hybrid":
        web_needed = bool(urls) or _prompt_has_any(
            prompt,
            (
                "public",
                "latest",
                "recent",
                "news",
                "layoff",
                "posted",
                "data quality",
                "company details",
                "current company details",
                "company classification",
                "product/service",
                "product service",
                "business model",
                "competitor",
                "competitors",
            ),
        )
    if (
        "company_verification" in tool_intents
        and (company_missing_info or current_status == "Needs web verification")
    ):
        web_needed = True

    plan = {
        "intent": "Answer a per-candidate Smart Column query from enriched profile context.",
        "needed_data": sorted(set(tool_calls + ["candidate_context_pack"])),
        "tool_calls": sorted(set(tool_calls)),
        "web_needed": bool(web_needed),
        "web_policy": "auto_fallback",
        "strictness": "unknown_instead_of_guess",
        "missing_info": company_missing_info,
        "output_fields": [item["key"] for item in normalize_output_schema(output_schema)],
        "tool_intents": tool_intents,
        "routing": routing,
    }
    if isinstance(planner_json, dict) and planner_json:
        for key in ("intent", "needed_data", "tool_calls", "tool_intents", "web_needed", "missing_info", "output_fields"):
            value = planner_json.get(key)
            if value not in (None, "", []):
                if key == "tool_intents" and isinstance(value, list):
                    plan[key] = [intent for intent in value if intent in SMART_COLUMN_TOOL_INTENTS]
                else:
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
    competitor_targets = _prompt_competitor_targets(prompt)
    profile_claims = context_pack.get("profile_claims") if isinstance(context_pack.get("profile_claims"), dict) else {}

    return {
        "context_pack": context_pack,
        "career_metrics": facts,
        "current_role": context_pack.get("current_role") or {},
        "career_locations": {
            "count": int(facts.get("career_city_count") or 0),
            "cities": facts.get("career_cities") or [],
            "location_count": int(facts.get("career_location_count") or 0),
            "locations": facts.get("career_locations") or [],
            "source": "structured_role_history",
        },
        "company_history": {
            "count": int(facts.get("unique_company_count") or 0),
            "companies": facts.get("companies") or [],
            "company_tenures": facts.get("company_tenures") or [],
        },
        "role_history": {
            "roles": facts.get("role_tenures") or [],
        },
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
            term: _matched_geography_duration(roles, term, profile_claims)
            for term in geography_terms
        },
        "industry_experience": {
            term: _matched_duration(roles, [term, *INDUSTRY_QUERY_TERMS.get(term, ())])
            for term in industry_terms
        },
        "competitor_match": {
            target: _competitor_match(roles, target)
            for target in competitor_targets
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
    if "[object object]" in output_text:
        errors.append("Output contains an unserialized object.")
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
    intents = infer_smart_column_intents(goal)
    if "stability" in goal_l and "summary" in goal_l:
        return [
            {"key": "result", "label": "Result", "type": "text", "primary": True},
            {"key": "reasoning", "label": "Reasoning", "type": "text", "primary": False},
        ]
    if "career_locations" in intents:
        list_first = "list" in goal_l or "which" in goal_l
        if _wants_places(goal):
            return [
                {"key": "career_location_count", "label": "Career Location Count", "type": "text", "primary": not list_first},
                {"key": "career_locations", "label": "Career Locations", "type": "text", "primary": list_first},
                {"key": "reasoning", "label": "Reasoning", "type": "text", "primary": False},
            ]
        return [
            {"key": "career_city_count", "label": "Career City Count", "type": "text", "primary": not list_first},
            {"key": "career_cities", "label": "Career Cities", "type": "text", "primary": list_first},
            {"key": "reasoning", "label": "Reasoning", "type": "text", "primary": False},
        ]
    if "tenure_metrics" in intents:
        return [
            {"key": "average_tenure_months", "label": "Average Tenure Months", "type": "text", "primary": True},
            {"key": "current_job_months", "label": "Current Job Months", "type": "text", "primary": False},
            {"key": "tenure_reasoning", "label": "Tenure Reasoning", "type": "text", "primary": False},
        ]
    if "company_history" in intents:
        list_first = "list" in goal_l or "career order" in goal_l or "which" in goal_l
        return [
            {"key": "company_count", "label": "Company Count", "type": "text", "primary": not list_first},
            {"key": "company_history", "label": "Company History", "type": "text", "primary": list_first},
        ]
    if "role_history" in intents:
        return [
            {"key": "role_history", "label": "Role History", "type": "text", "primary": True},
        ]
    if "current_role" in intents:
        return [
            {"key": "current_role", "label": "Current Role", "type": "text", "primary": True},
        ]
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
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False, default=str).strip()
        else:
            s = str(v).strip() if v is not None else ""
        if not s:
            continue
        prev = by_norm.get(nk)
        if prev is None or len(s) > len(prev):
            by_norm[nk] = s
    out: Dict[str, str] = {}
    for sk in schema_keys:
        direct_value = raw.get(sk)
        if isinstance(direct_value, (dict, list)):
            direct = json.dumps(direct_value, ensure_ascii=False, default=str).strip()
        else:
            direct = str(direct_value or "").strip()
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
        for imported in flatten_imported_extra_fields(profile.get("raw_fields")):
            grouped["Imported Extra Fields"][imported["key"]] = {
                "key": imported["key"],
                "label": imported["label"],
                "group": "Imported Extra Fields",
                "token": f"{{{imported['key']}}}",
                "sample": imported.get("sample", ""),
                "source_header": imported.get("source_header", ""),
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
        "Imported Extra Fields",
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

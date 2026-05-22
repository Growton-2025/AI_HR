from __future__ import annotations

import json
import re
from collections import defaultdict
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
MAX_IMPORTED_FIELDS = 120


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

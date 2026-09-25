"""Make every imported profile shape look like a role history.

The strict screener reasons over ``profile["roles"]`` (employer, title, dates)
and ``total_experience_years``. Those are populated from the ``roles`` table
and from ``raw_fields["experiences/N/…"]`` — but most uploaded sheets carry
the employer somewhere else entirely. For the "Clear – AE (ME & SEA)" role on
2026-09-25: 5 of 680 candidates had ``roles`` rows, 594 had
``raw_fields.import_company``, 2 used wide ``Company N Name`` columns, 279 had
tenure in ``Overall Exp (yrs)`` / ``Work Ex`` / ``Total work experience``, so
a "Testsigma competitors" screen matched 1 person when 69 qualified.

Pure functions; query.py wires them into the strict scorer.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_COMPANY_NOISE_RE = re.compile(
    r"\b(inc|inc\.|llc|ltd|limited|pvt|private|corp|corporation|technologies|technology|software)\b"
)
_WIDE_MAX = 15
# "Title at Company", "Title @Company", "Title | Company", "Title - Company".
_HEADLINE_EMPLOYER_RE = re.compile(
    r"^(?P<title>.+?)\s*(?:\bat\b|@|\|)\s*(?P<company>[A-Z][\w&.'\- ]{1,60})$"
)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|y\b)", re.I)
_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+?")
# "(AE - 1.5 Years)", "AE - 5 Yrs", "B2B SaaS - 8 Years"
_FUNCTION_YEARS_RE = re.compile(r"([A-Za-z][A-Za-z0-9 &/]{1,30}?)\s*[-–:]\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.I)
_FUNCTION_ALIASES = {
    "ae": "account executive",
    "account executive": "account executive",
    "account exec": "account executive",
    "bdr": "business development representative",
    "sdr": "sales development representative",
    "csm": "customer success manager",
    "am": "account manager",
    "account manager": "account manager",
    "sales": "sales",
    "b2b saas": "b2b saas",
    "saas": "saas",
    "enterprise sales": "enterprise sales",
}


def company_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = _COMPANY_NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .,-")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "n/a", "-"} else text


def _raw(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw = profile.get("raw_fields")
    return raw if isinstance(raw, dict) else {}


# ── employer sources ───────────────────────────────────────────────────────

def wide_column_roles(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``Company 1 Name`` pairs with ``Title`` / ``Start date`` / ``End Date`` /
    ``Details ``; ``Company N Name`` (N≥2) with ``Title.N-1`` etc. — pandas'
    duplicate-header suffixes from the recruiter's spreadsheet."""
    roles: List[Dict[str, Any]] = []
    for n in range(1, _WIDE_MAX + 1):
        company = _clean(raw.get(f"Company {n} Name"))
        suffix = "" if n == 1 else f".{n - 1}"
        title = _clean(raw.get(f"Title{suffix}"))
        if not company and not title:
            continue
        start = _clean(raw.get(f"Start date{suffix}") or raw.get(f"Start Date{suffix}"))
        end = _clean(raw.get(f"End Date{suffix}") or raw.get(f"End date{suffix}"))
        details = _clean(raw.get(f"Details {suffix}") or raw.get(f"Details{suffix}"))
        roles.append({
            "company": company,
            "title": title,
            "details": details,
            "start_date": start[:10] if start else "",
            "end_date": end[:10] if end else "",
            "duration_years": 0.0,
            "_source": f"raw_fields.Company {n} Name",
        })
    return roles


def headline_employer(headline: Any) -> Optional[Tuple[str, str]]:
    text = _clean(headline)
    if not text or len(text) > 140:
        return None
    m = _HEADLINE_EMPLOYER_RE.match(text)
    if not m:
        return None
    title, company = m.group("title").strip(" -|"), m.group("company").strip(" .")
    if not company or company_key(company) in {"present", "remote", "home"}:
        return None
    return title, company


def import_company_role(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The employer column the recruiter uploaded (``import_company``) is the
    person's *current* employer in every sheet we have seen."""
    raw = _raw(profile)
    company = _clean(raw.get("import_company") or raw.get("Company") or raw.get("Current Company") or raw.get("company"))
    if not company:
        return None
    title = _clean(raw.get("Title") or raw.get("import_title") or raw.get("Current Title"))
    if not title:
        parsed = headline_employer(profile.get("headline"))
        title = parsed[0] if parsed else _clean(profile.get("headline"))
    return {
        "company": company,
        "title": title,
        "details": "",
        "start_date": "",
        "end_date": "",
        "duration_years": 0.0,
        "_source": "raw_fields.import_company",
    }


def synthesize_roles(profile: Dict[str, Any], existing_roles: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Roles the strict scorer would otherwise never see, current employer
    first so the current-employer scope resolves to it when nothing is dated."""
    known = {company_key(r.get("company")) for r in existing_roles if isinstance(r, dict) and company_key(r.get("company"))}
    out: List[Dict[str, Any]] = []

    def add(role: Optional[Dict[str, Any]]) -> None:
        if not role:
            return
        key = company_key(role.get("company"))
        if key and key in known:
            return
        if key:
            known.add(key)
        out.append(role)

    add(import_company_role(profile))
    for role in wide_column_roles(_raw(profile)):
        add(role)
    parsed = headline_employer(profile.get("headline"))
    if parsed:
        add({"company": parsed[1], "title": parsed[0], "details": "", "start_date": "", "end_date": "",
             "duration_years": 0.0, "_source": "headline"})
    return out


# ── tenure sources ─────────────────────────────────────────────────────────

def _first_years(text: Any) -> Optional[float]:
    s = _clean(text)
    if not s:
        return None
    m = _YEARS_RE.search(s) or _LEADING_NUMBER_RE.match(s)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if 0 < value < 60 else None


def raw_total_experience_years(raw: Dict[str, Any]) -> Optional[float]:
    """``Overall Exp (yrs)``: 10.0 · ``Total work experience``: 9 ·
    ``Work Ex``: '8+ Years ( AE - 5 Yrs, B2B SaaS - 8 Years)' → 8."""
    for key, value in raw.items():
        k = str(key).strip().lower()
        if re.search(r"(overall|total)\s*(work\s*)?exp", k) or k in {"work ex", "work experience", "experience (yrs)", "experience"}:
            years = _first_years(value)
            if years is not None:
                return years
    return None


def raw_function_years(raw: Dict[str, Any]) -> Dict[str, float]:
    """``AE Exp (yrs)``: 2.3 → {'account executive': 2.3}; ``Work Ex``
    '(AE - 1.5 Years, B2B SaaS - 8 Years)' → both."""
    out: Dict[str, float] = {}
    for key, value in raw.items():
        k = str(key).strip().lower()
        m = re.match(r"^([a-z0-9 &/]+?)\s*exp(?:erience)?\s*\(?\s*(?:yrs?|years?)?\)?$", k)
        if m and m.group(1).strip() not in {"overall", "total", "total work", "work"}:
            years = _first_years(value)
            fn = _FUNCTION_ALIASES.get(m.group(1).strip(), m.group(1).strip())
            if years is not None:
                out[fn] = max(out.get(fn, 0.0), years)
        if k in {"work ex", "work experience"}:
            for label, years in _FUNCTION_YEARS_RE.findall(_clean(value)):
                fn = _FUNCTION_ALIASES.get(label.strip().lower(), label.strip().lower())
                try:
                    out[fn] = max(out.get(fn, 0.0), float(years))
                except ValueError:
                    pass
    return out


def function_years_for(raw_years: Dict[str, float], function: str, aliases: Iterable[str] = ()) -> Optional[float]:
    wanted = {function.strip().lower(), *[str(a).strip().lower() for a in aliases if str(a or "").strip()]}
    wanted |= {_FUNCTION_ALIASES.get(w, w) for w in list(wanted)}
    best = None
    for fn, years in raw_years.items():
        if fn in wanted or any(w and (w in fn or fn in w) for w in wanted):
            best = years if best is None else max(best, years)
    return best


# ── employer universe (for validating competitor lists) ────────────────────

def employer_names_from_profiles(profiles: Iterable[Dict[str, Any]]) -> Set[str]:
    """Every employer name that appears anywhere in candidate data: the roles
    table, ``experiences/N/companyName``, ``import_company``, wide columns and
    headlines. Competitor lists are validated against this, not just the
    ``companies`` table (which only knows employers of candidates with a
    ``roles`` row — 5 of 680 on the role that surfaced this)."""
    names: Set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for role in profile.get("roles") or []:
            if isinstance(role, dict) and _clean(role.get("company")):
                names.add(_clean(role.get("company")))
        raw = _raw(profile)
        for key, value in raw.items():
            k = str(key)
            if k == "import_company" or re.match(r"^Company \d+ Name$", k) or re.match(r"^experiences/\d+/companyName$", k):
                if _clean(value):
                    names.add(_clean(value))
        parsed = headline_employer(profile.get("headline"))
        if parsed:
            names.add(parsed[1])
    return names

"""Map one scraped LinkedIn profile onto every candidate-related table.

Pure transformation: nothing here touches the database. ``map_scraped_profile``
returns one dict per table (rows already shaped like the column lists in
``backend/db/connection.py``) plus a ``coverage`` report of columns it could not
fill, so a caller can persist it and an operator can see the holes.

Where each value comes from:
  * deterministic  - copied or computed from the scraped fields (tenure, gaps,
                     titles, education, skills ...), same rules as the original
                     ``individual.py`` enrichment (gaps are > 180 days).
  * LLM            - the five experience scores with their per-role detail and
                     team management, one gpt-4o call per profile.
  * web            - company funding stage, revenue, culture, headquarters ...,
                     one web-search call per unseen company.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.services import import_enrichment as ie
from backend.services.linkedin_normalize import normalize_linkedin

logger = logging.getLogger(__name__)

GAP_DAYS = 180
SCORE_MODEL = os.getenv("PROFILE_MAPPER_MODEL", "gpt-4o")
COMPANY_MODEL = os.getenv("PROFILE_MAPPER_COMPANY_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"  # candidates.embedding is VECTOR(1536)

# Columns each table has, minus ids/foreign keys. Used by the coverage report.
CANDIDATE_COLUMNS = [
    "name", "first_name", "last_name", "linkedin", "normalized_linkedin", "location",
    "city", "headline", "about", "email", "phone", "mobile_phone", "skills",
    "licenses_and_certifications", "total_experience_years", "avg_years_in_company",
    "has_gap_years", "has_education_gaps", "has_industry_gaps",
    "functional_experience_score", "functional_experience_rationale",
    "industry_experience_score", "industry_experience_rationale",
    "segment_experience_score", "segment_experience_rationale",
    "geography_experience_score", "geography_experience_rationale",
    "team_management_score", "team_management_rationale", "max_people_managed",
    "years_team_management", "raw_fields", "embedding",
]
COMPANY_COLUMNS = [
    "funding_stage", "revenue", "business_model", "product_service", "customer_segment",
    "customer_presence", "culture_type", "headquarters",
]


# --------------------------------------------------------------------------- helpers
def _txt(value: Any) -> str:
    return ie.clean_text(value)


def _d(value: Optional[datetime]) -> Optional[date]:
    return value.date() if isinstance(value, datetime) else value


def _join(raw: Dict[str, Any], prefix: str, field: str) -> List[str]:
    out, i = [], 0
    while f"{prefix}/{i}/{field}" in raw or f"{prefix}/{i + 1}/{field}" in raw:
        v = _txt(raw.get(f"{prefix}/{i}/{field}"))
        if v:
            out.append(v)
        i += 1
    return out


def _json_object(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model response")
    snippet = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
    return json.loads(snippet)


_openai_client = None
_openai_client_lock = threading.Lock()


def _client():
    """One shared client: it is thread-safe, keeps connections open between calls, and
    backs off and retries when several profiles hit a rate limit at the same time."""
    global _openai_client
    if _openai_client is None:
        with _openai_client_lock:
            if _openai_client is None:
                from openai import OpenAI

                _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=5)
    return _openai_client


# ------------------------------------------------------------ deterministic: gaps
def _merge(periods: Sequence[Tuple[date, date]]) -> List[Tuple[date, date]]:
    merged: List[Tuple[date, date]] = []
    for start, end in sorted(periods):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _gap_row(a_end: date, b_start: date, reason: str, **extra: Any) -> Optional[Dict[str, Any]]:
    days = (b_start - a_end).days
    if days <= GAP_DAYS:
        return None
    return {"from_date": a_end, "to_date": b_start, "duration_months": round(days / 30),
            "reason": reason, **extra}


def experience_gaps(roles: Sequence[ie.ParsedRole]) -> List[Dict[str, Any]]:
    periods = [(_d(r.start), _d(r.end) or date.today()) for r in roles if r.start]
    periods = [(s, e) for s, e in periods if e > s]
    merged = _merge(periods)
    return [g for a, b in zip(merged, merged[1:]) if (g := _gap_row(a[1], b[0], "unknown"))]


def education_gaps(education: Sequence[ie.ParsedEducation]) -> List[Dict[str, Any]]:
    periods = [(_d(e.start), _d(e.end) or date.today()) for e in education if e.start and e.college]
    periods = [(s, e) for s, e in periods if e > s]
    merged = _merge(periods)
    return [g for a, b in zip(merged, merged[1:]) if (g := _gap_row(a[1], b[0], "education"))]


def industry_gaps(roles: Sequence[ie.ParsedRole]) -> List[Dict[str, Any]]:
    stints = sorted(
        (_d(r.start), _d(r.end) or date.today(), r.industry if r.industry != "Unknown" else r.source_industry)
        for r in roles if r.start
    )
    stints = [s for s in stints if s[2] and s[1] > s[0]]
    out = []
    for (_, e1, i1), (s2, _, i2) in zip(stints, stints[1:]):
        if i1 != i2 and (g := _gap_row(e1, s2, "industry")):
            out.append(g)
    return out


# ---------------------------------------------------------------- web: companies
# Model and web calls made while mapping one profile wait on the network, not the CPU,
# so they share this pool and run side by side (employer research, the embedding).
_IO_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("PROFILE_MAPPER_IO_WORKERS", "24") or 24),
                              thread_name_prefix="profile-map-io")


class CompanyCache:
    """Employers already researched during one import run, shared by every profile in it.

    The first profile that needs an employer researches it; others asking for the same
    employer meanwhile wait for that one answer instead of paying for the web search again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Future] = {}

    def get(self, name: str, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        key = name.strip().lower()
        with self._lock:
            job = self._jobs.get(key)
            owner = job is None
            if owner:
                job = self._jobs[key] = Future()
        if owner:
            try:
                job.set_result(build())
            except BaseException as exc:
                job.set_exception(exc)
        return dict(job.result())


def research_company(name: str, hint: str = "") -> Dict[str, Any]:
    """Web-search one employer for the companies-table columns. Values the model
    cannot support from sources are returned as null, never guessed."""
    prompt = (
        f"Research the company '{name}'. {hint}\n"
        "Use web search. Return ONLY a JSON object with keys: funding_stage "
        "(one of Bootstrapped, Seed, Series A, Series B, Series C+, Public, Private, Acquired, Non-profit), "
        "revenue (short text such as 'Approx. $10 billion' or 'Undisclosed'), "
        "business_model (the single best fit, one of: B2B, B2C, B2B SaaS, B2B and B2C, Service-based), "
        "product_service (one short sentence), "
        "customer_segment (list of up to 4 short labels such as Enterprise, SMB, Government), "
        "customer_presence (list of regions, e.g. India, North America, Global), "
        "culture_type (1-3 words), headquarters ('City, Country'). "
        "Use null for anything you cannot support from sources."
    )
    resp = _client().responses.create(
        model=COMPANY_MODEL, tools=[{"type": os.getenv("IMPORT_ENRICHMENT_WEB_TOOL", "web_search_preview")}],
        input=prompt,
    )
    return _json_object(resp.output_text)


def _short_list(value: Any, limit: int = 40) -> List[str]:
    """Keep short label-like entries; drop sentences the model leaked into lists."""
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in (value or []) if isinstance(v, (str, int)) and 0 < len(str(v).strip()) <= limit]


def _product_text(value: Any) -> Optional[str]:
    if isinstance(value, list):
        parts = [(f"{p.get('name')}: {p.get('description')}" if isinstance(p, dict) else str(p)) for p in value]
        return "; ".join(parts)[:500] or None
    return _txt(value)[:500] or None


def build_company_row(name: str, classified: Dict[str, Any], researched: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce classifier + web output into the exact shapes the companies table holds."""
    def pick(key: str) -> Any:
        v = researched.get(key)
        return v if v not in (None, "", "Unknown") else classified.get(key)

    product = _product_text(pick("product_service"))
    segments = _short_list(pick("customer_segment"))
    return {
        "name": name,
        "funding_stage": _txt(pick("funding_stage")) or None,
        "revenue": _txt(pick("revenue")) or None,
        "business_model": (_txt(pick("business_model")) or None) if _txt(pick("business_model")) != "Unknown" else None,
        "product_service": product if product != "Unknown" else None,
        "customer_segment": segments,
        "customer_presence": _short_list(pick("customer_presence")),
        "culture_type": _txt(pick("culture_type")) or None,
        "headquarters": _txt(pick("headquarters")) or None,
    }


# ------------------------------------------------------------------- LLM: scoring
SCORE_PROMPT = """You are scoring one sales candidate from their LinkedIn profile.
Return ONLY a JSON object with exactly these keys:
{
 "functional": {"score": int 0-10, "rationale": str,
   "roles": [{"company": str, "activity_type": str, "reason": str, "duration_years": number}]},
 "industry":   {"score": int 0-10, "rationale": str,
   "roles": [{"company": str, "industry": str, "reason": str, "duration_years": number}]},
 "segment":    {"score": int 0-10, "rationale": str,
   "roles": [{"company": str, "segment": str, "reason": str, "duration_years": number}]},
 "geography":  {"score": int 0-10, "rationale": str, "regions": [str]},
 "team_management": {"score": int 0-10, "rationale": str,
   "max_people_managed": int, "years_team_management": number}
}
Rules: judge only from the data given; use 0 and say so when there is no evidence;
one roles[] entry per employer; duration_years must come from the tenure given.
"""


def score_candidate(context: Dict[str, Any]) -> Dict[str, Any]:
    resp = _client().chat.completions.create(
        model=SCORE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SCORE_PROMPT},
                  {"role": "user", "content": json.dumps(context, default=str)}],
    )
    return json.loads(resp.choices[0].message.content)


def embedding_text(c: Dict[str, Any], roles: Sequence[ie.ParsedRole]) -> str:
    bits = [c.get("name"), c.get("headline"), c.get("about"), c.get("location"), c.get("skills")]
    bits += [f"{r.title} at {r.company}" for r in roles]
    return " | ".join(str(b) for b in bits if b)


def embed(text: str) -> List[float]:
    return _client().embeddings.create(model=EMBED_MODEL, input=text[:24000]).data[0].embedding


# ---------------------------------------------------------------------- the mapper
def map_scraped_profile(raw: Dict[str, Any], *, use_llm: bool = True, use_web: bool = True,
                        known_companies: Optional[Dict[str, Dict[str, Any]]] = None,
                        company_cache: Optional[CompanyCache] = None) -> Dict[str, Any]:
    """``known_companies`` maps lower-case name -> existing companies row; those are
    reused instead of re-researched (saves a web call and never overwrites data).
    ``company_cache`` shares employer research between the profiles of one import run."""
    known_companies = known_companies or {}
    cand_in = {"id": 0, "raw_fields": raw, "name": raw.get("fullName"),
               "headline": raw.get("headline"), "about": raw.get("about")}
    roles = ie.parse_roles_from_raw(raw, cand_in)
    education = ie.parse_education_from_raw(raw)
    metrics = ie.calculate_tenure_metrics(roles, raw_total_exp_years=None)

    # companies: classify locally, then web-research anything not already stored
    contexts = ie._company_contexts_by_norm([cand_in], {0: roles})

    def company_row(name: str) -> Dict[str, Any]:
        existing = known_companies.get(name.lower())
        if existing and existing.get("funding_stage") and existing.get("product_service"):
            return {**existing, "name": name, "_source": "db"}

        def build() -> Dict[str, Any]:
            classified = ie.classify_company(
                name, role_texts=contexts.get(ie.normalize_company_name(name), []),
                db_details=existing, allow_web=use_web)
            researched: Dict[str, Any] = {}
            if use_web:
                try:
                    researched = research_company(name, f"Website: {next((r.source_website for r in roles if r.company == name and r.source_website), '')}")
                except Exception as exc:  # one bad company must not sink the profile
                    logger.warning("company research failed for %s: %s", name, exc)
            return {**build_company_row(name, classified, researched),
                    "_source": "web" if researched else "local", "_classified": classified}

        row = company_cache.get(name, build) if company_cache is not None else build()
        return {**row, "name": name}

    names = sorted({r.company for r in roles if _txt(r.company)})
    # Employers are independent lookups, so one profile's unknown employers run together.
    company_jobs = [(name, _IO_POOL.submit(company_row, name)) for name in names[1:]]
    companies: Dict[str, Dict[str, Any]] = {names[0]: company_row(names[0])} if names else {}
    for name, job in company_jobs:
        companies[name] = job.result()
    for r in roles:
        c = companies.get(r.company, {}).get("_classified", {})
        r.product_service = c.get("product_service") if isinstance(c.get("product_service"), str) else (companies.get(r.company, {}).get("product_service") or "Unknown")
        r.industry = c.get("industry") if isinstance(c.get("industry"), str) else r.source_industry or "Unknown"
        r.customer_segment = list(companies.get(r.company, {}).get("customer_segment") or [])
        r.business_model = companies.get(r.company, {}).get("business_model") or "Unknown"

    skills = ", ".join(_join(raw, "skills", "title")) or _txt(raw.get("topSkillsByEndorsements"))
    certs = []
    for i in range(0, 50):
        t = _txt(raw.get(f"licenseAndCertificates/{i}/title"))
        if not t:
            if i > 0 and f"licenseAndCertificates/{i + 1}/title" not in raw:
                break
            continue
        issuer = _txt(raw.get(f"licenseAndCertificates/{i}/subtitle"))
        certs.append(f"{t}, {issuer}" if issuer else t)

    exp_gaps, edu_gaps, ind_gaps = experience_gaps(roles), education_gaps(education), industry_gaps(roles)
    loc = _txt(raw.get("addressWithCountry"))
    cand: Dict[str, Any] = {
        "name": _txt(raw.get("fullName")), "first_name": _txt(raw.get("firstName")) or None,
        "last_name": _txt(raw.get("lastName")) or None,
        "linkedin": _txt(raw.get("linkedinPublicUrl") or raw.get("linkedinUrl")),
        "normalized_linkedin": normalize_linkedin(raw.get("linkedinPublicUrl") or raw.get("linkedinUrl")),
        "location": loc or None,
        "city": (_txt(raw.get("addressWithoutCountry")).split(",")[0] or loc.split(",")[0]).strip() or None,
        "headline": _txt(raw.get("headline")) or None, "about": _txt(raw.get("about")) or None,
        "email": _txt(raw.get("email")) or None, "phone": None,
        "mobile_phone": _txt(raw.get("mobileNumber")) or None,
        "skills": skills or None, "licenses_and_certifications": "; ".join(certs) or None,
        "total_experience_years": metrics.get("total_experience_years") or None,
        "avg_years_in_company": metrics.get("avg_tenure_years") or None,
        "has_gap_years": bool(exp_gaps), "has_education_gaps": bool(edu_gaps), "has_industry_gaps": bool(ind_gaps),
        "raw_fields": raw,
    }

    scores: Dict[str, Any] = {}
    if use_llm:
        tenure = {c["company"]: c["years"] for c in metrics.get("company_years", [])}
        # The embedding text does not depend on the scores, so it is fetched while the model scores.
        embedding_job = _IO_POOL.submit(embed, embedding_text(cand, roles))
        scores = score_candidate({
            "headline": cand["headline"], "about": cand["about"], "location": loc,
            "roles": [{"title": r.title, "company": r.company, "details": r.details,
                       "start": str(_d(r.start)), "end": str(_d(r.end)),
                       "company_industry": r.industry, "company_product": r.product_service,
                       "company_segments": r.customer_segment} for r in roles],
            "tenure_years_by_company": tenure})
        for key, col in (("functional", "functional_experience"), ("industry", "industry_experience"),
                         ("segment", "segment_experience"), ("geography", "geography_experience"),
                         ("team_management", "team_management")):
            block = scores.get(key) or {}
            cand[f"{col}_score"] = block.get("score")
            cand[f"{col}_rationale"] = block.get("rationale")
        tm = scores.get("team_management") or {}
        cand["max_people_managed"] = tm.get("max_people_managed")
        cand["years_team_management"] = tm.get("years_team_management")
        cand["embedding"] = embedding_job.result()

    def exp_table(key: str, child: str) -> Optional[Dict[str, Any]]:
        block = scores.get(key)
        if not block:
            return None
        rows = block.get("regions") if key == "geography" else block.get("roles")
        return {"score": block.get("score"), "rationale": block.get("rationale"), child: rows or []}

    tables: Dict[str, Any] = {
        "candidates": cand,
        "companies": [{k: v for k, v in c.items() if not k.startswith("_")} for c in companies.values()],
        "roles": [{"company": r.company, "title": r.title,
                   "details": json.dumps({"company": r.company, "start_date": str(_d(r.start)) if r.start else None,
                                          "end_date": str(_d(r.end)) if r.end else None, "details": r.details,
                                          "duration_months": r.duration_months}),
                   "duration_years": None if r.duration_unknown else round((r.duration_months or 0) / 12, 2)}
                  for r in roles],
        "titles_held": [{"title": r.title, "company": r.company, "start_date": _d(r.start), "end_date": _d(r.end)}
                        for r in roles],
        "education": [{"college": e.college, "degree": e.degree or None, "start_date": _d(e.start),
                       "end_date": _d(e.end), "details": e.details or None} for e in education],
        "company_years": [{"company": c["company"], "years": c["years"]} for c in metrics.get("company_years", [])],
        "experience_gaps": exp_gaps, "education_gaps": edu_gaps, "industry_gaps": ind_gaps,
        "functional_experiences": exp_table("functional", "roles"),
        "industry_experiences": exp_table("industry", "roles"),
        "segment_experiences": exp_table("segment", "roles"),
        "geography_experiences": exp_table("geography", "regions"),
    }
    tables["coverage"] = coverage_report(tables)
    return tables


def coverage_report(tables: Dict[str, Any]) -> Dict[str, Any]:
    """List what is still empty so holes are visible instead of silent."""
    cand = tables["candidates"]
    empty_cand = [c for c in CANDIDATE_COLUMNS if cand.get(c) in (None, "", [])]
    empty_co = {c["name"]: [k for k in COMPANY_COLUMNS if c.get(k) in (None, "", [])] for c in tables["companies"]}
    empty_tables = [t for t in ("roles", "titles_held", "education", "company_years", "functional_experiences",
                                "industry_experiences", "segment_experiences", "geography_experiences")
                    if not tables.get(t)]
    return {"candidate_columns_empty": empty_cand,
            "company_columns_empty": {k: v for k, v in empty_co.items() if v},
            "tables_empty": empty_tables,
            "tables_legitimately_empty_if_no_gaps": [t for t in ("experience_gaps", "education_gaps", "industry_gaps")
                                                     if not tables.get(t)]}

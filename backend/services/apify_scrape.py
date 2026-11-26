"""Scrape LinkedIn profiles through Apify and return wide, flattened rows.

Every actor's output is normalised to the same slash-delimited keys the
spreadsheet upload path already parses (``educations/0/title``,
``experiences/1/companyName`` ...), so the result can be fed to the existing
verified-profile enrichment and the profile mapper unchanged.

Supported actors (``APIFY_LINKEDIN_ACTOR``):

* ``curious_coder/linkedin-profile-scraper`` - logs in with your LinkedIn
  session cookie (``APIFY_LINKEDIN_COOKIE`` JSON, or ``APIFY_LINKEDIN_COOKIE_FILE``),
  ``APIFY_LINKEDIN_USER_AGENT`` and an Apify proxy in ``APIFY_PROXY_COUNTRY``.
  The actor warns not to exceed ~500 profiles a day per LinkedIn account.
* ``supreme_coder/linkedin-profile-scraper`` - the same author's cookie-free
  variant with the same output shape; takes ``urls`` only.
* ``dev_fusion/Linkedin-Profile-Scraper`` - cookie-free, takes ``profileUrls``
  and already returns the flat key shape.

``APIFY_LINKEDIN_FALLBACK_ACTOR`` lists, comma-separated and in order, the actors asked
for the profiles the main actor did not return, so one actor being out of service does
not stop an import. Default: dev_fusion, supreme_coder, curious_coder (the main actor is
left out; curious_coder only when a LinkedIn cookie is configured). Empty turns it off.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from backend.services.linkedin_normalize import normalize_linkedin

logger = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
# The actor bills per result, so a stuck run must not hang a request thread.
RUN_TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

DEFAULT_ACTOR = "curious_coder/linkedin-profile-scraper"
DEFAULT_FALLBACK_ACTORS = ("dev_fusion/Linkedin-Profile-Scraper,supreme_coder/linkedin-profile-scraper,"
                           "curious_coder/linkedin-profile-scraper")
FALLBACK_ATTEMPTS = 2
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


class ApifyScrapeError(RuntimeError):
    pass


def _config() -> tuple[str, str, int]:
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not token:
        raise ApifyScrapeError("APIFY_API_TOKEN is not set")
    actor = os.getenv("APIFY_LINKEDIN_ACTOR", DEFAULT_ACTOR).strip()
    cap = int(os.getenv("APIFY_MAX_PROFILES_PER_REQUEST", "50") or 50)
    return token, actor, cap


# ----------------------------------------------------------------------- flattening
def flatten_item(value: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten nested dict/list JSON into ``a/0/b`` keys (the Apify export shape)."""
    out = {} if out is None else out
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_item(child, f"{prefix}/{key}" if prefix else str(key), out)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            flatten_item(child, f"{prefix}/{idx}", out)
    elif prefix:
        out[prefix] = _clean_scalar(value)
    return out


_NULL_SEGMENT = re.compile(r"(^|,\s*)null(?=\s*(,|$))", flags=re.IGNORECASE)


def _clean_scalar(value: Any) -> Any:
    """Some actors join sub-fields with a literal ``null`` ("BE, null"); drop those."""
    if not isinstance(value, str):
        return value
    cleaned = _NULL_SEGMENT.sub("", value).strip().strip(",").strip()
    return cleaned or None


# ------------------------------------------------------- curious_coder conversion
def _month_year(period: Any) -> Optional[str]:
    """``{"month": 10, "year": 2022}`` -> ``"10-2022"``; year-only -> ``"2022"``."""
    if not isinstance(period, dict) or not period.get("year"):
        return None
    month = period.get("month")
    return f"{int(month):02d}-{period['year']}" if month else str(period["year"])


def _first(*values: Any) -> Optional[str]:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _position_company(pos: Dict[str, Any]) -> Dict[str, Any]:
    """The actor nests employer details differently across versions; read them all."""
    company = pos.get("company") if isinstance(pos.get("company"), dict) else {}
    name = _first(pos.get("companyName"), company.get("name"), company.get("universalName"),
                  (pos.get("companyUrn") or "").split(":")[-1] if pos.get("companyUrn") else None)
    industries = company.get("industries") or pos.get("industries") or []
    industry = None
    if isinstance(industries, list) and industries:
        first = industries[0]
        industry = first.get("name") if isinstance(first, dict) else _first(first)
    size = None
    rng = company.get("employeeCountRange")
    if isinstance(rng, dict) and rng.get("start"):
        size = f"{rng['start']}-{rng['end']}" if rng.get("end") else f"{rng['start']}+"
    return {"name": name, "industry": industry, "size": size,
            "website": _first(company.get("websiteUrl"), pos.get("companyWebsite")),
            "linkedin": _first(pos.get("companyLinkedinUrl"), company.get("url"))}


def convert_curious_coder_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map one ``curious_coder/linkedin-profile-scraper`` record onto the dev_fusion
    key shape, then flatten it. Unknown or missing sections just stay absent."""
    first, last = _first(item.get("firstName")), _first(item.get("lastName"))
    public_id = _first(item.get("publicIdentifier"))
    url = _first(item.get("linkedinUrl"), item.get("url"), item.get("inputUrl"),
                 f"https://www.linkedin.com/in/{public_id}" if public_id else None)
    loc, country = _first(item.get("geoLocationName"), item.get("locationName")), _first(item.get("geoCountryName"))
    # geoLocationName usually already ends with the country ("Bengaluru, Karnataka, India").
    with_country = loc if (loc and country and loc.lower().endswith(country.lower())) else ", ".join(p for p in (loc, country) if p)

    out: Dict[str, Any] = {
        "linkedinUrl": url,
        "linkedinPublicUrl": url,
        "publicIdentifier": public_id,
        "firstName": first,
        "lastName": last,
        "fullName": " ".join(p for p in (first, last) if p) or None,
        "headline": _first(item.get("headline"), item.get("occupation")),
        "about": _first(item.get("summary"), item.get("about")),
        "addressWithCountry": with_country or None,
        "addressWithoutCountry": (loc[: -len(country)].rstrip(" ,") if loc and country and loc.lower().endswith(country.lower()) else loc) or None,
        "countryCode": _first(item.get("countryCode")),
        "email": _first(item.get("email")),
        "mobileNumber": _first(item.get("phone"), item.get("mobileNumber")),
        "jobTitle": _first(item.get("jobTitle")),
        "companyName": _first(item.get("companyName")),
        "industryName": _first(item.get("industryName")),
        "profilePic": _first(item.get("pictureUrl")),
        "followers": item.get("followerCount"),
        "connections": item.get("connectionsCount"),
    }

    experiences = []
    for pos in item.get("positions") or []:
        if not isinstance(pos, dict):
            continue
        tp = pos.get("timePeriod") or {}
        company = _position_company(pos)
        # "Bengaluru, Karnataka, India · On-site" -> location + workplace type
        raw_loc = _first(pos.get("locationName"), pos.get("geoLocationName")) or ""
        job_loc, _, workplace = (part.strip() for part in raw_loc.partition("·"))
        experiences.append({
            "title": _first(pos.get("title")),
            "employmentType": _first(pos.get("employmentType")),
            "workplaceType": workplace or None,
            "companyName": company["name"],
            "companyIndustry": company["industry"],
            "companySize": company["size"],
            "companyWebsite": company["website"],
            "companyLink1": company["linkedin"],
            "jobLocation": job_loc or None,
            "jobStartedOn": _month_year(tp.get("startDate")),
            "jobEndedOn": _month_year(tp.get("endDate")),
            "jobStillWorking": not bool(tp.get("endDate")),
            "description": _first(pos.get("description")),
        })
    out["experiences"] = experiences

    educations = []
    for edu in item.get("educations") or []:
        if not isinstance(edu, dict):
            continue
        tp = edu.get("timePeriod") or {}
        degree_name, field = _first(edu.get("degreeName")), _first(edu.get("fieldOfStudy"))
        if field and degree_name and field.lower() == degree_name.lower():
            field = None  # the actor repeats the degree as the field of study
        degree = ", ".join(p for p in (degree_name, field) if p)
        educations.append({
            "title": _first(edu.get("schoolName")),
            "subtitle": degree or None,
            "description": _first(edu.get("description"), edu.get("activities")),
            "period": {"startedOn": _month_year(tp.get("startDate")), "endedOn": _month_year(tp.get("endDate"))},
        })
    out["educations"] = educations

    out["skills"] = [{"title": s if isinstance(s, str) else (s or {}).get("name")} for s in item.get("skills") or []]
    out["licenseAndCertificates"] = [
        {"title": _first(c.get("name")), "subtitle": _first(c.get("authority"), c.get("issuer")),
         "issuedOn": _month_year((c.get("timePeriod") or {}).get("startDate"))}
        for c in item.get("certifications") or [] if isinstance(c, dict)
    ]
    out["languages"] = [{"name": _first(l.get("name")), "proficiency": _first(l.get("proficiency"))}
                        for l in item.get("languages") or [] if isinstance(l, dict)]
    out["honors"] = [{"title": _first(h.get("title")), "issuer": _first(h.get("issuer"))}
                     for h in item.get("honors") or [] if isinstance(h, dict)]
    out["projects"] = [{"title": _first(pr.get("title")), "description": _first(pr.get("description"))}
                       for pr in item.get("projects") or [] if isinstance(pr, dict)]
    return flatten_item(out)


def _linkedin_cookie() -> List[Dict[str, Any]]:
    raw = os.getenv("APIFY_LINKEDIN_COOKIE", "").strip()
    path = os.getenv("APIFY_LINKEDIN_COOKIE_FILE", "").strip()
    if not raw and path:
        try:
            raw = Path(path).read_text().strip()
        except OSError as exc:
            raise ApifyScrapeError(f"APIFY_LINKEDIN_COOKIE_FILE could not be read: {exc}")
    if not raw:
        raise ApifyScrapeError(
            "This actor needs your LinkedIn session: export the cookies with the Cookie-Editor "
            "extension and set APIFY_LINKEDIN_COOKIE (JSON array) or APIFY_LINKEDIN_COOKIE_FILE in .env"
        )
    try:
        cookie = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApifyScrapeError(f"APIFY_LINKEDIN_COOKIE is not valid JSON: {exc}")
    if not isinstance(cookie, list) or not cookie:
        raise ApifyScrapeError("APIFY_LINKEDIN_COOKIE must be the JSON array exported by Cookie-Editor")
    return cookie


def _curious_coder_input(urls: Sequence[str]) -> Dict[str, Any]:
    return {
        "urls": list(urls),
        "cookie": _linkedin_cookie(),
        "userAgent": os.getenv("APIFY_LINKEDIN_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "proxy": {"useApifyProxy": True,
                  "apifyProxyCountry": (os.getenv("APIFY_PROXY_COUNTRY", "IN").strip() or "IN").upper()},
        "scrapeCompany": False,
        "minDelay": int(os.getenv("APIFY_LINKEDIN_MIN_DELAY", "2") or 2),
        "maxDelay": int(os.getenv("APIFY_LINKEDIN_MAX_DELAY", "6") or 6),
    }


def convert_auto(item: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the converter from the record shape: nested ``positions`` means the
    curious_coder/supreme_coder family, anything else is already flat-ish."""
    if isinstance(item.get("positions"), list) or "publicIdentifier" in item:
        return convert_curious_coder_item(item)
    return flatten_item(item)


# One entry per supported actor: (build run input, convert one dataset item).
ACTOR_ADAPTERS: Dict[str, tuple[Callable[[Sequence[str]], Dict[str, Any]], Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
    # Same author family and output shape as curious_coder, but no LinkedIn login.
    # Its ``urls`` field is a request list, so each entry is ``{"url": ...}``.
    "supreme_coder/linkedin-profile-scraper": (lambda urls: {"urls": [{"url": u} for u in urls], "scrapeCompany": False}, convert_auto),
    "curious_coder/linkedin-profile-scraper": (_curious_coder_input, convert_auto),
    "dev_fusion/linkedin-profile-scraper": (lambda urls: {"profileUrls": list(urls)}, flatten_item),
}


def _adapter(actor: str):
    key = actor.replace("~", "/").strip().lower()
    if key not in ACTOR_ADAPTERS:
        raise ApifyScrapeError(f"No adapter for Apify actor {actor!r}; supported: {sorted(ACTOR_ADAPTERS)}")
    return ACTOR_ADAPTERS[key]


def _row_url(row: Dict[str, Any]) -> str:
    return normalize_linkedin(row.get("linkedinPublicUrl") or row.get("linkedinUrl") or "")


def _actor_key(actor: str) -> str:
    return actor.replace("~", "/").strip().lower()


def _has_linkedin_cookie() -> bool:
    return bool(os.getenv("APIFY_LINKEDIN_COOKIE", "").strip() or os.getenv("APIFY_LINKEDIN_COOKIE_FILE", "").strip())


def _fallback_actors(primary: str) -> List[str]:
    """The actors asked, in order, when the main one returns nothing. Unsupported
    actors, the main actor itself, and curious_coder without a LinkedIn cookie (it
    cannot run) are left out."""
    actors: List[str] = []
    for actor in os.getenv("APIFY_LINKEDIN_FALLBACK_ACTOR", DEFAULT_FALLBACK_ACTORS).split(","):
        actor, key = actor.strip(), _actor_key(actor)
        if not actor or key == _actor_key(primary) or key not in ACTOR_ADAPTERS:
            continue
        if key == "curious_coder/linkedin-profile-scraper" and not _has_linkedin_cookie():
            continue
        if key not in {_actor_key(a) for a in actors}:
            actors.append(actor)
    return actors


def _run_actor(actor: str, token: str, urls: Sequence[str], attempts: int) -> List[Dict[str, Any]]:
    """Ask one actor for ``urls``, retrying the misses. Stops early when a run returns
    nothing at all: that is the actor being out of service, not one flaky profile."""
    build_input, convert = _adapter(actor)
    rows: List[Dict[str, Any]] = []
    pending = list(urls)
    for attempt in range(1, attempts + 1):
        resp = requests.post(
            f"{APIFY_BASE}/acts/{actor.replace('/', '~')}/run-sync-get-dataset-items",
            params={"timeout": RUN_TIMEOUT_SECONDS},
            headers={"Authorization": f"Bearer {token}"},
            json=build_input(pending),
            timeout=RUN_TIMEOUT_SECONDS + 30,
        )
        if resp.status_code >= 300:
            raise ApifyScrapeError(f"Apify returned HTTP {resp.status_code}: {resp.text[:300]}")
        batch = [convert(item) for item in resp.json() if isinstance(item, dict)]
        # Some actors answer with a different slug than the one requested (numeric
        # ids, renamed vanity URLs). When the batch lines up one-to-one with what
        # we asked for, trust the order and stamp the requested URL on the row.
        unmatched = [r for r in batch if _row_url(r) not in {normalize_linkedin(u) for u in pending}]
        if unmatched and len(batch) == len(pending):
            for url, row in zip(pending, batch):
                if _row_url(row) != normalize_linkedin(url):
                    row["requestedUrl"], row["linkedinUrl"], row["linkedinPublicUrl"] = url, url, url
        rows += batch
        done = {_row_url(r) for r in rows}
        pending = [u for u in urls if normalize_linkedin(u) not in done]
        if not pending:
            break
        logger.info("apify actor=%s attempt %s: %s of %s profiles still missing", actor, attempt, len(pending),
                    len(urls))
        if not batch:
            break
        if attempt < attempts:
            time.sleep(RETRY_DELAY_SECONDS)
    return rows


def scrape_profiles(linkedin_urls: List[str]) -> Dict[str, Any]:
    """Scrape profiles; returns ``{"rows": [...], "failed": [...], "requested": n}``.

    ``failed`` lists input URLs the actor returned nothing for (private, dead,
    or blocked profiles), so callers can report them per row.
    """
    token, actor, cap = _config()
    _adapter(actor)      # an unsupported actor is a configuration error, not an outage

    urls: List[str] = []
    seen = set()
    for raw in linkedin_urls:
        key = normalize_linkedin(raw)
        if key and key not in seen:
            seen.add(key)
            urls.append(raw.strip())
    if not urls:
        return {"rows": [], "failed": [], "requested": 0}
    if len(urls) > cap:
        raise ApifyScrapeError(
            f"{len(urls)} profiles requested; APIFY_MAX_PROFILES_PER_REQUEST is {cap}"
        )

    # Actors intermittently return nothing for profiles they can fetch a moment
    # later, and failed lookups are not billed, so the misses are retried, then asked
    # of the fallback actor.
    fallbacks = _fallback_actors(actor)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, current in enumerate([actor] + fallbacks):
        missing = [u for u in urls if normalize_linkedin(u) not in {_row_url(r) for r in rows}]
        if not missing:
            break
        if index:
            logger.info("apify: %s of %s profiles still missing; asking %s", len(missing), len(urls), current)
        try:
            rows += _run_actor(current, token, missing, MAX_ATTEMPTS if index == 0 else FALLBACK_ATTEMPTS)
        except (ApifyScrapeError, requests.RequestException) as exc:
            errors.append(f"{current}: {exc}")
            logger.warning("apify actor=%s failed: %s", current, exc)
    # Every actor asked failed outright: that is Apify being unreachable or out of
    # quota, not a profile nobody could return.
    if not rows and len(errors) == 1 + len(fallbacks):
        raise ApifyScrapeError("; ".join(errors)[:600])

    got = {_row_url(r) for r in rows}
    failed = [u for u in urls if normalize_linkedin(u) not in got]
    logger.info("apify actor=%s fallbacks=%s requested=%s returned=%s failed=%s", actor, fallbacks, len(urls),
                len(rows), len(failed))
    return {"rows": rows, "failed": failed, "requested": len(urls)}

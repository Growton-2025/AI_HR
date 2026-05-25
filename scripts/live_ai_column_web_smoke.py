#!/usr/bin/env python3
"""Run live AI-column web-research checks against real cached profiles.

This intentionally calls OpenAI. It loads credentials from the process env and
prints only result metadata, never secrets.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

from backend.api.routes import ai_columns  # noqa: E402
from backend.pipeline import query  # noqa: E402
from backend.services.ai_columns import build_candidate_context  # noqa: E402


OUTPUT_SCHEMA = [
    {"key": "summary", "label": "Summary", "type": "text", "primary": True},
    {"key": "latest_update", "label": "Latest Update", "type": "text", "primary": False},
    {"key": "event_date", "label": "Event Date", "type": "text", "primary": False},
    {"key": "source_url", "label": "Source URL", "type": "text", "primary": False},
]


TESTS = [
    {
        "label": "Freshworks recent layoffs",
        "company": "Freshworks",
        "prompt": (
            "For the candidate's current company, find the latest credible update about recent layoffs, "
            "restructuring, or workforce reduction. Be current as of today. If there is no recent verified "
            "layoff update, say so and give the latest related company update instead."
        ),
    },
    {
        "label": "BrowserStack latest company update",
        "company": "BrowserStack",
        "prompt": (
            "Find the latest credible company update for the candidate's current company, prioritizing "
            "funding, acquisitions, product launches, executive changes, or layoffs."
        ),
    },
    {
        "label": "HubSpot recent AI or layoff update",
        "company": "HubSpot",
        "prompt": (
            "Find the latest credible update about the candidate's current company, especially AI product "
            "updates, go-to-market changes, or workforce reductions."
        ),
    },
    {
        "label": "Salesforce recent layoffs or hiring signal",
        "company": "Salesforce",
        "prompt": (
            "Find the latest credible update about layoffs, hiring changes, or workforce restructuring at "
            "the candidate's current company."
        ),
    },
    {
        "label": "MongoDB latest company news",
        "company": "MongoDB",
        "prompt": (
            "Find the latest credible company update for the candidate's current company, prioritizing "
            "earnings, product, leadership, acquisition, or workforce news."
        ),
    },
]


def _profile_text(profile: Dict[str, Any]) -> str:
    parts = [
        profile.get("name"),
        profile.get("headline"),
        profile.get("company"),
        profile.get("current_company"),
        profile.get("raw_fields"),
    ]
    for role in profile.get("roles") or []:
        parts.append((role or {}).get("company"))
        parts.append((role or {}).get("title"))
    return " ".join(str(part or "") for part in parts)


def _current_role(profile: Dict[str, Any]) -> Dict[str, Any]:
    roles = profile.get("roles") or []
    return (roles[0] or {}) if roles else {}


def _find_profile(company: str, used_ids: Iterable[int]) -> Optional[Dict[str, Any]]:
    used = set(used_ids)
    company_l = company.lower()
    for profile in (query.PROFILES_BY_ID or {}).values():
        if int(profile.get("id") or 0) in used:
            continue
        current_company = str(_current_role(profile).get("company") or "")
        if company_l in current_company.lower():
            return profile
    for profile in (query.PROFILES_BY_ID or {}).values():
        if int(profile.get("id") or 0) in used:
            continue
        if company_l in _profile_text(profile).lower():
            return profile
    return None


def _compact_sources(details: Dict[str, Any]) -> List[str]:
    out = []
    for source in details.get("sources") or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or "").strip()
        if url:
            out.append(f"{title or url} - {url}")
        if len(out) >= 3:
            break
    return out


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    query.initialize_cache()
    print(f"profiles_loaded={len(query.PROFILES_BY_ID or {})}")
    print(f"model={ai_columns._AI_COLUMN_OPENAI_MODEL}")
    print(f"web_tool={ai_columns._AI_COLUMN_WEB_SEARCH_TOOL}")
    print(f"web_context_size={ai_columns._AI_COLUMN_WEB_SEARCH_CONTEXT_SIZE}")
    print()

    used_ids = set()
    failures = 0
    label_filter = os.getenv("LIVE_AI_COLUMN_TEST_FILTER", "").strip().lower()
    selected_tests = [
        test for test in TESTS
        if not label_filter or label_filter in test["label"].lower() or label_filter in test["company"].lower()
    ]
    if not selected_tests:
        print(f"No tests matched LIVE_AI_COLUMN_TEST_FILTER={label_filter!r}", file=sys.stderr)
        return 2

    for test in selected_tests:
        profile = _find_profile(test["company"], used_ids)
        if not profile:
            failures += 1
            print(f"## {test['label']}")
            print(f"missing_profile_for={test['company']}")
            print()
            continue
        used_ids.add(int(profile.get("id") or 0))
        context = build_candidate_context(profile)
        role = _current_role(profile)
        company_matches = test["company"].lower() in str(role.get("company") or "").lower()
        prompt = (
            f"{test['prompt']}\n\n"
            "Return concise structured outputs. Include one source URL in source_url. "
            "Use the candidate row to identify the company, but verify the update on the web.\n"
            "Candidate: {candidate.full_name}. Current company: {role.current_company}."
        )
        result = ai_columns._run_ai_task(
            prompt_template=prompt,
            mode="web_research",
            output_schema=OUTPUT_SCHEMA,
            context=context,
        )
        details = result.get("details") or {}
        outputs = result.get("outputs") or {}
        sources = _compact_sources(details)
        if not outputs.get("summary") or not details.get("searched_at"):
            failures += 1

        print(f"## {test['label']}")
        print(f"profile_id={profile.get('id')} name={profile.get('name')} company={role.get('company')}")
        print(f"current_company_match={company_matches}")
        print(f"searched_at={details.get('searched_at')}")
        print(f"tool={details.get('web_search_tool')} model={details.get('model')} context={details.get('web_search_context_size')}")
        print("outputs=" + json.dumps(outputs, ensure_ascii=False, indent=2))
        print("sources=" + json.dumps(sources, ensure_ascii=False, indent=2))
        print()

    if failures:
        print(f"failures={failures}", file=sys.stderr)
        return 1
    print("live_web_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

""""candidates working in series a listed companies" returned Series B,
Series H and Public employers.

Two causes, both pinned here:

1. The only funding criterion, ``funding_stage_min``, was an open-ended
   minimum, so "Series A" executed as "Series A or later". It now carries an
   optional ``max_stage``; a bare stage in the query is pinned to an exact
   window by ``_apply_explicit_funding_stage_window`` regardless of what the
   planner model emitted.

2. ``STATIC_COMPANY_DETAILS_TAXONOMY["public"]`` lumped "pre-ipo" and "listed"
   with "public", so "listed" became a public-company filter that Pre-IPO
   Series B/H companies satisfied and real Series A companies failed. Pre-IPO
   is now its own bucket and rank, and a bare "listed" next to a named stage
   no longer adds the public filter.

Part A (no network): scorer + guard semantics on synthetic profiles.
Part B (real planner LLM, 15 calls to SCREENING_CRITERIA_MODEL, no DB): 15
recruiter phrasings through planner + coercion, asserting which employer
stages the resulting criteria accept.

Run:  PYTHONPATH=. ./myenv/bin/python -m pytest tests/shortlist_funding_stage_test.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import backend.pipeline.query as q  # noqa: E402

STAGES = ["Seed", "Series A", "Series B", "Series H", "Pre-IPO", "Public"]


def _profile(stage: str, *, current: bool = True) -> Dict[str, Any]:
    role = {
        "company": f"{stage} Co",
        "title": "Account Executive",
        "is_current": current,
        "end_date": None if current else "2023-01-01",
        "duration_years": 2.0,
        "company_details": {"funding_stage": stage, "business_model": "B2B SaaS"},
    }
    if stage == "Pre-IPO":
        # Mirrors the web-research shape: a late-stage private company whose
        # funding_stage is a series letter and whose status is Pre-IPO.
        role["company_details"] = {"funding_stage": "Series B", "company_status": "Pre-IPO", "business_model": "B2B SaaS"}
    return {"id": 1, "name": "Synthetic", "roles": [role]}


def accepted_stages(criteria: Dict[str, Any]) -> List[str]:
    """Which synthetic employer stages pass BOTH the strict funding check and
    the strict company-details check for the given executable criteria."""
    out: List[str] = []
    for stage in STAGES:
        profile = _profile(stage)
        funding = q._strict_funding_stage_result(profile, criteria)
        if funding.get("applicable") and not funding.get("met"):
            continue
        details = criteria.get("required_company_details")
        if details:
            presence = q._strict_presence_result(profile, "required_company_details", details, criteria)
            if presence.get("applicable") and not presence.get("met"):
                continue
        out.append(stage)
    return out


# --------------------------------------------------------------------------
# Part A: deterministic scorer semantics
# --------------------------------------------------------------------------

def test_contract_exposes_max_stage():
    contract = q._executable_criteria_contract()
    assert [k for k in contract if "fund" in k] == ["funding_stage_min"]
    assert "max_stage" in contract["funding_stage_min"]["shape"]


def test_open_ended_minimum_still_accepts_every_later_stage():
    criteria = {"funding_stage_min": {"stage": "series-a", "employment_scope": "current_employer"}}
    assert accepted_stages(criteria) == ["Series A", "Series B", "Series H", "Pre-IPO", "Public"]


def test_exact_window_accepts_only_that_stage():
    criteria = {"funding_stage_min": {"stage": "series a", "max_stage": "series a", "employment_scope": "current_employer"}}
    assert accepted_stages(criteria) == ["Series A"]
    strict = q._strict_funding_stage_result(_profile("Series B"), criteria)
    assert strict["met"] is False
    assert "exactly series a" in strict["missing"][0]
    assert "Series B" in strict["missing"][0]
    ok, *_ = q._score_funding_stage(_profile("Series H"), criteria)
    assert ok is False


def test_closed_range_window_and_reversed_bounds():
    criteria = {"funding_stage_min": {"stage": "series b", "max_stage": "seed"}}
    assert accepted_stages(criteria) == ["Seed", "Series A", "Series B"]


@pytest.mark.parametrize(
    "stage, rank",
    [("Seed", 1), ("Series A", 2), ("series-a", 2), ("Series B", 3), ("Series H", 9), ("Pre-IPO", 11), ("pre ipo", 11), ("Public", 12), ("Publicly listed", 12)],
)
def test_funding_rank_orders_pre_ipo_below_public(stage, rank):
    assert q._funding_rank(stage) == rank


def test_pre_ipo_is_not_a_public_synonym():
    assert "pre-ipo" not in q.STATIC_COMPANY_DETAILS_TAXONOMY["public"]
    assert "pre-ipo" in q.STATIC_COMPANY_DETAILS_TAXONOMY["pre-ipo"]
    criteria = {"required_company_details": {"operator": "OR", "values": ["public", "listed"]}}
    result = q._strict_presence_result(_profile("Pre-IPO"), "required_company_details", criteria["required_company_details"], criteria)
    assert result["met"] is False


def _coerce(hard_filters: Dict[str, Any], query: str) -> Dict[str, Any]:
    plan = {"filter_plan": {"hard_filters": hard_filters}}
    return {k: v for k, v in q._coerce_filter_plan_to_criteria(plan, query).items() if not k.startswith("_")}


@pytest.mark.parametrize(
    "query, planner_stage, expected_min, expected_max",
    [
        ("candidates working in series a listed companies", "series a", "series a", "series a"),
        ("candidates working in series-a companies", "series a", "series a", "series a"),
        ("candidates working at a series b company", "series b", "series b", "series b"),
        ("candidates working at series a or series b companies", "series b", "series a", "series b"),
        ("candidates at startups between seed and series a", "seed", "seed", "series a"),
        ("candidates working in pre-ipo companies", "pre-ipo", "pre ipo", "pre ipo"),
        ("candidates working at series a and above companies", "series a", "series a", None),
        ("candidates working at series c or later companies", "series c", "series c", None),
        ("candidates working at series d+ companies", "series-d", "series-d", None),
        ("at least series b funded companies", "series b", "series b", None),
    ],
)
def test_query_guard_pins_window_to_named_stages(query, planner_stage, expected_min, expected_max):
    criteria = _coerce({"funding_stage_min": {"stage": planner_stage, "employment_scope": "current_employer"}}, query)
    assert criteria["funding_stage_min"]["stage"] == expected_min
    assert criteria["funding_stage_min"].get("max_stage") == expected_max


def test_query_guard_removes_planner_max_when_open_ended():
    criteria = _coerce({"funding_stage_min": {"stage": "series a", "max_stage": "series a"}}, "series a and above")
    assert "max_stage" not in criteria["funding_stage_min"]


def test_query_guard_leaves_criteria_without_stage_mentions_alone():
    criteria = _coerce({"funding_stage_min": {"stage": "growth", "max_stage": "growth"}}, "candidates at growth stage companies")
    assert criteria["funding_stage_min"]["stage"] == "growth"
    assert criteria["funding_stage_min"]["max_stage"] == "growth"


def test_bare_listed_next_to_a_stage_drops_public_filter():
    hard = {
        "funding_stage_min": {"stage": "series a"},
        "required_company_details": {"values": ["public", "publicly traded", "listed"]},
    }
    criteria = _coerce(hard, "candidates working in series a listed companies")
    assert "required_company_details" not in criteria
    assert criteria["funding_stage_min"]["max_stage"] == "series a"
    # "publicly listed" keeps the public filter.
    criteria = _coerce(dict(hard), "candidates working in series a publicly listed companies")
    assert criteria["required_company_details"]["values"]
    # non-public details survive the pruning
    hard["required_company_details"] = {"values": ["saas", "listed"]}
    criteria = _coerce(hard, "series a listed saas companies")
    assert criteria["required_company_details"]["values"] == ["saas"]


# --------------------------------------------------------------------------
# Part B: 15 recruiter phrasings through the real planner
# --------------------------------------------------------------------------

CASES = [
    # (query, stages the recruiter meant)
    ("candidates working in series a listed companies", {"Series A"}),
    ("candidates working in series a companies", {"Series A"}),
    ("candidates currently at series a startups", {"Series A"}),
    ("account executives at series a funded saas companies", {"Series A"}),
    ("candidates working at a series b company", {"Series B"}),
    ("candidates from seed stage startups", {"Seed"}),
    ("candidates working at series a or series b companies", {"Series A", "Series B"}),
    ("candidates at early stage startups between seed and series a", {"Seed", "Series A"}),
    ("candidates working in pre-ipo companies", {"Pre-IPO"}),
    ("candidates working at publicly listed companies", {"Public"}),
    ("candidates at bootstrapped companies", set()),
    ("candidates working at series a and above companies", {"Series A", "Series B", "Series H", "Pre-IPO", "Public"}),
    ("candidates working at series c and above companies", {"Series H", "Pre-IPO", "Public"}),
    ("candidates working at series d+ companies", {"Series H", "Pre-IPO", "Public"}),
    ("worked at a series a company in the past", {"Series A"}),
]


@pytest.fixture(scope="module")
def planned() -> Dict[str, Dict[str, Any]]:
    """Run all 15 queries through the planner in one event loop, no DB."""
    q.get_db_connection = lambda *a, **k: None  # evidence catalog falls back to static shape
    catalog = q.build_db_evidence_catalog(profiles=[], force_refresh=True)
    manifest = q._build_schema_manifest(catalog, scoped_candidate_count=680)
    pack = q._build_terminology_pack()
    tracker = q.TokenCostTracker()

    async def one(query: str) -> Dict[str, Any]:
        normalized = q.normalize_query_with_llm(query)
        raw = await q._generate_schema_aware_filter_plan(normalized, manifest, pack, tracker)
        criteria = q._coerce_filter_plan_to_criteria(raw, normalized)
        public = {k: v for k, v in criteria.items() if not k.startswith("_")}
        return {"raw_hard_filters": (raw.get("filter_plan") or {}).get("hard_filters"), "criteria": public}

    async def run_all():
        results = await asyncio.gather(*(one(query) for query, _ in CASES))
        return dict(zip((query for query, _ in CASES), results))

    results = asyncio.run(run_all())

    print("\n\n=== planner output per query ===")
    rows = []
    for query, expected in CASES:
        crit = results[query]["criteria"]
        got = accepted_stages(crit)
        rows.append((query, sorted(expected), got, crit.get("funding_stage_min"), crit.get("required_company_details")))
    for query, expected, got, fmin, details in rows:
        flag = "OK " if set(got) == set(expected) else "BUG"
        print(f"{flag} | {query}")
        print(f"      wanted   : {expected}")
        print(f"      accepted : {got}")
        print(f"      funding_stage_min={json.dumps(fmin, default=str)} required_company_details={json.dumps(details, default=str)}")
    print(f"planner cost: {tracker.total_tokens} tokens, ${tracker.total_cost:.4f}")
    return results


@pytest.mark.parametrize("query, expected", CASES, ids=[c[0] for c in CASES])
def test_planner_funding_semantics_match_recruiter_intent(planned, query, expected):
    criteria = planned[query]["criteria"]
    got = set(accepted_stages(criteria))
    assert got == expected, (
        f"query={query!r}\n"
        f"  recruiter meant : {sorted(expected)}\n"
        f"  engine accepts  : {sorted(got)}\n"
        f"  criteria        : {json.dumps(criteria, default=str)}"
    )

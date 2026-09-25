"""Run a set of natural-language screens against one role and score them
against regex ground truth computed from the same candidate data.

    ./myenv/bin/python scripts/screening_recall_check.py --role 92 --queries all
    ./myenv/bin/python scripts/screening_recall_check.py --role 92 --queries 1,3,7

Ground truth is deliberately dumb (regex over headline, raw_fields values,
roles): it says who *should* be reachable from the data we hold, so recall
measures the pipeline, not the LLM's opinion. Precision checks that returned
people satisfy the same predicate. Cost comes from the pipeline's tracker.

Uses the real OpenAI key from .env. SCREENING_AUDIT_MODEL defaults to
gpt-4o-mini here (override with --audit-model) to keep 20 runs affordable.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

COMPETITORS = {
    # A company is not its own competitor; "testmu" is LambdaTest's 2025 rebrand.
    # OpenText (UFT One / Micro Focus) and UiPath (Test Suite) sell test
    # automation too and are on every published Testsigma competitor list.
    "testsigma": ["browserstack", "lambdatest", "testmu", "sauce labs", "saucelabs", "katalon", "tricentis", "mabl", "applitools",
                  "perfecto", "functionize", "testrigor", "smartbear", "ranorex", "leapwork", "testim", "kobiton",
                  "headspin", "accelq", "provar", "copado", "qualitia", "opkey", "autify", "opentext", "micro focus", "uipath"],
    "browserstack": ["lambdatest", "testmu", "sauce labs", "saucelabs", "testsigma", "katalon", "kobiton", "headspin", "perfecto", "pcloudy", "testgrid", "experitest"],
    "hevo": ["fivetran", "airbyte", "stitch", "matillion", "informatica", "talend", "rivery", "integrate.io", "estuary", "meltano"],
    "mongodb": ["couchbase", "redis", "datastax", "cockroach", "cassandra", "elastic", "neo4j", "singlestore", "yugabyte", "planetscale"],
}


def _text(profile: dict) -> str:
    parts = [str(profile.get("headline") or ""), str(profile.get("about") or "")]
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    parts.extend(f"{k}: {v}" for k, v in raw.items() if v not in (None, ""))
    for role in profile.get("roles") or []:
        parts.append(f"{role.get('title') or ''} at {role.get('company') or ''}")
    return " ".join(parts).lower()


def _employers(profile: dict) -> List[str]:
    from backend.services.profile_experience import employer_names_from_profiles
    return [n.lower() for n in employer_names_from_profiles([profile])]


def _current_employer(profile: dict) -> str:
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    return str(raw.get("import_company") or raw.get("Company 1 Name") or "").lower()


def _years(profile: dict) -> float:
    from backend.services.profile_experience import raw_total_experience_years
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    return float(profile.get("total_experience_years") or 0) or (raw_total_experience_years(raw) or 0.0)


def _ae_years(profile: dict) -> float:
    from backend.services.profile_experience import raw_function_years, function_years_for
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    return function_years_for(raw_function_years(raw), "account executive", ["AE"]) or 0.0


def _geo(profile: dict) -> str:
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    bits = [str(profile.get("headline") or "")]
    bits.extend(str(v) for k, v in raw.items() if re.search(r"geo|market|region|territory", str(k), re.I))
    return " ".join(bits).lower()


def _loc(profile: dict) -> str:
    # Where the person is, not where they would move to ("Pref Location").
    raw = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    return " ".join(str(x or "") for x in (profile.get("city"), profile.get("location"), raw.get("addressWithCountry"))).lower()


def any_employer(names): return lambda p: any(n in e for e in _employers(p) for n in names) or any(n in _text(p) for n in names)
def current_employer(names): return lambda p: any(n in _current_employer(p) for n in names)
def any_text(*needles): return lambda p: any(n in _text(p) for n in needles)
def _word(needle, text): return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None   # "us" must not hit "australia"
def geo(*needles): return lambda p: any(_word(n, _geo(p)) for n in needles)
def loc(*needles): return lambda p: any(_word(n, _loc(p)) for n in needles)
def AND(*fs): return lambda p: all(f(p) for f in fs)
def NOT(f): return lambda p: not f(p)


QUERIES: List[Dict] = [
    {"id": 1, "q": "Candidates with experience at in testsigma competators", "gt": any_employer(COMPETITORS["testsigma"]), "web": True},
    {"id": 2, "q": "Candidates currently working at a BrowserStack competitor", "gt": current_employer(COMPETITORS["browserstack"]), "web": True},
    {"id": 3, "q": "Account executives who worked at Freshworks", "gt": any_employer(["freshworks"])},
    {"id": 4, "q": "Candidates from Salesforce or Datadog", "gt": any_employer(["salesforce", "datadog"])},
    {"id": 5, "q": "People with experience at Hevo competitors", "gt": any_employer(COMPETITORS["hevo"]), "web": True},
    {"id": 6, "q": "Account executives with 5+ years of overall experience", "gt": lambda p: _years(p) >= 5},
    {"id": 7, "q": "Candidates with at least 3 years as an Account Executive", "gt": lambda p: _ae_years(p) >= 3},
    {"id": 8, "q": "Candidates with EMEA or Europe market experience", "gt": geo("emea", "europe", "uk", "germany", "france", "middle east", "mea", "uae", "gcc", "africa")},   # EMEA = Europe + Middle East + Africa
    {"id": 9, "q": "Candidates who sold into the US market", "gt": geo("us", "usa", "u.s.", "north america", "amer", "namer", "americas", "united states")},
    {"id": 10, "q": "Candidates based in Bangalore", "gt": loc("bangalore", "bengaluru", "blr")},
    {"id": 11, "q": "Candidates located in Mumbai or Pune", "gt": loc("mumbai", "bombay", "pune")},
    {"id": 12, "q": "Account executives with enterprise segment experience", "gt": any_text("enterprise")},
    {"id": 13, "q": "Candidates with mid-market sales experience", "gt": any_text("mid-market", "mid market", "midmarket")},
    {"id": 14, "q": "Candidates with fintech experience", "gt": any_text("fintech", "payments", "banking")},
    {"id": 15, "q": "Immediate joiners or notice period of 30 days or less",
     "gt": lambda p: bool(re.search(r"immediate|\b(?:[1-9]|[12][0-9]|30)\s*days?\b|\b1\s*months?\b", str((p.get("raw_fields") or {}).get("Notice Period") or ""), re.I))},
    {"id": 16, "q": "Senior account executives", "gt": any_text("senior account executive", "senior ae", "sr. account executive", "sr account executive")},
    {"id": 17, "q": "Account executives at Testsigma competitors with EMEA experience", "gt": AND(any_employer(COMPETITORS["testsigma"]), geo("emea", "europe", "middle east", "mea", "uae", "gcc", "africa")), "web": True},
    {"id": 18, "q": "Candidates from Rippling or Freshworks with 5+ years of experience", "gt": AND(any_employer(["rippling", "freshworks"]), lambda p: _years(p) >= 5)},
    {"id": 19, "q": "Enterprise account executives based in Bangalore with US market experience", "gt": AND(any_text("enterprise"), loc("bangalore", "bengaluru", "blr"), geo("us", "usa", "u.s.", "north america", "amer", "namer", "americas", "united states"))},
    {"id": 20, "q": "People who worked at MongoDB competitors", "gt": any_employer(COMPETITORS["mongodb"]), "web": True},
]


async def run_query(entry: Dict, role_id: int, user_id: int) -> Dict:
    from backend.pipeline.query import TokenCostTracker, process_query_main
    tracker = TokenCostTracker()
    statuses: List[str] = []
    final: Dict = {}
    started = time.time()
    async for item in process_query_main(
        entry["q"], f"recall-check-{entry['id']}", tracker,
        screening_user_id=user_id, screening_role="admin",
        source_type="role", source_role_id=role_id, use_web_search=bool(entry.get("web")),
    ):
        if isinstance(item, str):
            statuses.append(item)
        elif isinstance(item, dict) and item.get("type") == "complete":
            final = item
    return {"statuses": statuses, "final": final, "seconds": round(time.time() - started, 1),
            "cost": round(float(getattr(tracker, "total_cost", 0) or 0), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", type=int, required=True)
    ap.add_argument("--user", type=int, default=4)
    ap.add_argument("--queries", default="all")
    ap.add_argument("--audit-model", default=os.getenv("SCREENING_AUDIT_MODEL_FOR_CHECK", "gpt-4o-mini"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("SCREENING_AUDIT_MODEL", args.audit_model)

    from backend.db.connection import get_db_connection, return_db_connection
    from backend.pipeline import query as q
    q.initialize_cache()
    conn = get_db_connection(validate=False, register_pgvector=False)
    with conn.cursor() as cur:
        cur.execute("SELECT candidate_id FROM recruitment_role_candidates WHERE role_id = %s", (args.role,))
        scope_ids = [int(r[0]) for r in cur.fetchall()]
    return_db_connection(conn)
    scope = [q.PROFILES_BY_ID[i] for i in scope_ids if i in q.PROFILES_BY_ID]
    print(f"role {args.role}: {len(scope)} candidates in scope (cache has {len(q.PROFILES_BY_ID)})\n")

    wanted = QUERIES if args.queries == "all" else [e for e in QUERIES if str(e["id"]) in set(args.queries.split(","))]
    rows = []
    # One event loop for every query: the OpenAI async client binds to the
    # loop it first ran on, so a fresh asyncio.run() per query dies with
    # "Event loop is closed" from the second query onwards.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for entry in wanted:
        gt_ids = {p["id"] for p in scope if entry["gt"](p)}
        result = loop.run_until_complete(run_query(entry, args.role, args.user))
        final = result["final"] or {}
        returned = final.get("data") or []
        returned_ids = {int(c.get("id")) for c in returned if c.get("id") is not None}
        debug = final.get("filter_debug") or {}
        passed = debug.get("passed")
        hit = len(returned_ids & gt_ids)
        recall = hit / len(gt_ids) if gt_ids else None
        precision = hit / len(returned_ids) if returned_ids else None
        found = next((s for s in result["statuses"] if s.startswith("Found competitors")), "")
        reasons = debug.get("reject_reason_counts") or {}
        top_reasons = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        # Evidence hygiene: nothing a recruiter reads may carry an internal
        # evidence id, and every returned card must carry its checklist.
        id_re = re.compile(r"\bev\d+\b", re.I)
        answers_with_ids = sum(
            1 for c in returned
            if id_re.search(" ".join([str(c.get("answer") or ""), str(c.get("decision_narrative") or ""),
                                       *[str(i.get("why_it_supports") or "") for i in (c.get("requirement_breakdown") or [])],
                                       *[str(cl.get("text") or "") for cl in (c.get("audit_claims") or [])]]))
        )
        missing_breakdown = sum(1 for c in returned if not c.get("requirement_breakdown"))
        row = {
            "id": entry["id"], "query": entry["q"], "ground_truth": len(gt_ids), "passed_strict": passed,
            "answers_with_ids": answers_with_ids, "missing_breakdown": missing_breakdown,
            "returned": len(returned_ids), "recall": None if recall is None else round(recall, 2),
            "precision": None if precision is None else round(precision, 2),
            "missed_ids": sorted(gt_ids - returned_ids)[:15], "false_positive_ids": sorted(returned_ids - gt_ids)[:15],
            "cost_usd": result["cost"], "seconds": result["seconds"], "competitors": found[:200], "top_reject_reasons": top_reasons,
            "statuses": result["statuses"][-6:],
        }
        rows.append(row)
        print(f"[{row['id']:>2}] gt={row['ground_truth']:<4} strict={passed!s:<5} returned={row['returned']:<4} "
              f"recall={row['recall']!s:<5} precision={row['precision']!s:<5} ${row['cost_usd']:<7} {row['seconds']}s  "
              f"ids_in_text={answers_with_ids} no_breakdown={missing_breakdown}  {entry['q']}", flush=True)
        if found:
            print(f"      {found[:160]}")
        if top_reasons:
            print(f"      rejects: {top_reasons}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

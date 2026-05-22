#!/usr/bin/env python3
"""
Smoke-test AI column backend the same way the UI does (field-catalog → generate).

Uses credentials from the environment (do not hard-code passwords in repo):
  export E2E_ADMIN_EMAIL='admin@gmail.com'
  export E2E_ADMIN_PASSWORD='…'
  export SMOKE_API_BASE='http://127.0.0.1:8000'   # optional; no /api suffix

Example:
  cd AI_HR && PYTHONPATH=. .venv/bin/python scripts/smoke_ai_column_backend.py
"""

from __future__ import annotations

import json
import os
import sys

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    base = os.getenv("SMOKE_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    api = f"{base}/api"
    email = os.getenv("E2E_ADMIN_EMAIL", "").strip()
    password = os.getenv("E2E_ADMIN_PASSWORD", "").strip()
    if not email or not password:
        print("Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD", file=sys.stderr)
        sys.exit(2)

    s = requests.Session()
    r = s.post(f"{api}/login", json={"email": email, "password": password}, timeout=60)
    if r.status_code != 200:
        print("login failed", r.status_code, r.text[:500], file=sys.stderr)
        sys.exit(1)
    token = r.json().get("access_token")
    if not token:
        print("no access_token", r.text[:500], file=sys.stderr)
        sys.exit(1)
    h = {"Authorization": f"Bearer {token}"}

    print("[1] GET /api/health")
    hr = s.get(f"{api}/health", timeout=15)
    print(" ", hr.status_code, hr.text[:120])

    print("[2] GET /api/candidates/browse?view_scope=master&page_size=5")
    br = s.get(
        f"{api}/candidates/browse",
        headers=h,
        params={"view_scope": "master", "page": 1, "page_size": 5},
        timeout=120,
    )
    print(" ", br.status_code)
    if br.status_code != 200:
        print(br.text[:800], file=sys.stderr)
        sys.exit(1)
    cands = br.json().get("candidates") or []
    ids = [c.get("id") for c in cands if c.get("id") is not None][:3]
    print(" ", "sample candidate ids:", ids, "names:", [c.get("name") for c in cands[:3]])

    print("[3] GET /api/ai-columns/field-catalog?view_scope=master")
    fc = s.get(f"{api}/ai-columns/field-catalog", headers=h, params={"view_scope": "master"}, timeout=120)
    print(" ", fc.status_code)
    if fc.status_code != 200:
        print(fc.text[:800], file=sys.stderr)
        sys.exit(1)
    groups = fc.json().get("groups") or []
    print(" ", "groups:", len(groups), "total items:", sum(len(g.get("items") or []) for g in groups))

    print("[4] POST /api/ai-columns/generate (same as UI Generate Config)")
    goal = (
        "Add one column that outputs a single short sentence summarizing title and company "
        "using only row tokens like {candidate.first_name} and {role.current_company}."
    )
    gr = s.post(
        f"{api}/ai-columns/generate",
        headers=h,
        json={"goal": goal, "view_scope": "master", "recruiter_filter_id": None},
        timeout=120,
    )
    print(" ", gr.status_code)
    if gr.status_code != 200:
        print(gr.text[:1200], file=sys.stderr)
        sys.exit(1)
    body = gr.json()
    print(" ", "keys:", list(body.keys()))
    print(" ", "name:", (body.get("name") or "")[:80])
    print(" ", "mode:", body.get("mode"))
    print(" ", "prompt_template (first 200 chars):", (body.get("prompt_template") or "")[:200])
    print(" ", "output_schema:", json.dumps(body.get("output_schema") or [], indent=2)[:400])

    if ids:
        print("[5] GET /api/ai-columns?candidate_ids=… (optional)")
        q = ",".join(str(i) for i in ids)
        lr = s.get(f"{api}/ai-columns", headers=h, params={"view_scope": "master", "candidate_ids": q}, timeout=60)
        print(" ", lr.status_code, "columns:", len((lr.json() or {}).get("columns") or []))

    print("OK — backend paths used by “Generate” completed without hanging.")


if __name__ == "__main__":
    main()

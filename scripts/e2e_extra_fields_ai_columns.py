#!/usr/bin/env python3
"""
Live E2E for uploaded extra Excel/CSV fields in AI Columns.

Flow:
  import disposable rows with custom extra fields + work history
  run verified enrichment
  verify raw storage + AI context tokens
  create/run AI columns for industry, segment, function, geography, and tenure prompts
  always delete marker candidates/uploads/AI columns, then verify cleanup

Required env:
  E2E_ADMIN_EMAIL, E2E_ADMIN_PASSWORD

Optional:
  E2E_BASE_URL (default http://127.0.0.1:8765)
  E2E_SKIP_SERVER=1
  E2E_RECRUITER_A_EMAIL / E2E_RECRUITER_A_PASSWORD
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from backend.db.connection import get_db_connection, return_db_connection  # noqa: E402
from backend.pipeline import query  # noqa: E402
from backend.services.ai_columns import (  # noqa: E402
    build_candidate_context,
    compute_career_facts,
    run_candidate_query_tools,
)

MARKER = "zz-e2e-extra-fields-ai"
FILENAME = "e2e_extra_fields_ai_import.csv"
BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
SKIP_SERVER = os.getenv("E2E_SKIP_SERVER", "").lower() in ("1", "true", "yes")
RECRUITER_A = os.getenv("E2E_RECRUITER_A_EMAIL", "e2e_extra_fields_ai@gmail.com")
RECRUITER_A_PW = os.getenv("E2E_RECRUITER_A_PASSWORD", "adam@123")


def _db_cleanup_marker(*, verify: bool = True) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("DB connection failed during cleanup")
    try:
        with conn.cursor() as cur:
            like = f"%{MARKER}%"
            cur.execute("DELETE FROM ai_column_definitions WHERE slug ILIKE %s OR name ILIKE %s", (like, like))
            cur.execute("SELECT id FROM candidates WHERE linkedin ILIKE %s OR normalized_linkedin LIKE %s", (like, like))
            candidate_ids = [int(row[0]) for row in cur.fetchall()]
            if candidate_ids:
                for table in (
                    "recruitment_role_candidates",
                    "roles",
                    "company_years",
                    "functional_experiences",
                    "industry_experiences",
                    "segment_experiences",
                    "geography_experiences",
                    "education",
                ):
                    try:
                        cur.execute(f"DELETE FROM {table} WHERE candidate_id = ANY(%s)", (candidate_ids,))
                    except Exception:
                        conn.rollback()
                        raise
                cur.execute("DELETE FROM candidates WHERE owner_user_id IS NOT NULL AND id = ANY(%s)", (candidate_ids,))
                cur.execute("DELETE FROM candidates WHERE owner_user_id IS NULL AND id = ANY(%s)", (candidate_ids,))
            cur.execute(
                """
                DELETE FROM candidate_uploads
                WHERE filename = %s OR file_headers::text ILIKE %s
                """,
                (FILENAME, like),
            )
        conn.commit()

        if verify:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM candidates WHERE linkedin ILIKE %s OR normalized_linkedin LIKE %s),
                      (SELECT COUNT(*) FROM candidate_uploads WHERE filename = %s OR file_headers::text ILIKE %s),
                      (SELECT COUNT(*) FROM ai_column_definitions WHERE slug ILIKE %s OR name ILIKE %s)
                    """,
                    (like, like, FILENAME, like, like, like),
                )
                counts = cur.fetchone()
            if counts != (0, 0, 0):
                raise SystemExit(f"Cleanup verification failed; remaining counts={counts!r}")
    finally:
        return_db_connection(conn)


def _safe_delete_test_recruiter(email: str) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            if not row:
                return
            user_id = int(row[0])
            cur.execute(
                """
                SELECT COUNT(*) FROM candidates
                WHERE owner_user_id = %s
                  AND (linkedin NOT ILIKE %s AND (normalized_linkedin IS NULL OR normalized_linkedin NOT LIKE %s))
                """,
                (user_id, f"%{MARKER}%", f"%{MARKER}%"),
            )
            non_marker_candidates = int(cur.fetchone()[0] or 0)
            if non_marker_candidates:
                print(
                    f"[extra-fields-e2e] skip deleting test recruiter {email!r}: "
                    f"has {non_marker_candidates} non-marker candidate rows"
                )
                return
            cur.execute("DELETE FROM candidate_uploads WHERE owner_user_id = %s", (user_id,))
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
    finally:
        return_db_connection(conn)


def _start_uvicorn() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_health(session: requests.Session, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = session.get(f"{BASE_URL}/api/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise SystemExit(f"Server not healthy at {BASE_URL} within {timeout}s")


def _login(session: requests.Session, email: str, password: str) -> str:
    r = session.post(f"{BASE_URL}/api/login", json={"email": email, "password": password}, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"Login failed {email!r}: {r.status_code} {r.text}")
    return r.json()["access_token"]


def _h(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ensure_recruiter(session: requests.Session, admin_token: str, email: str, password: str, name: str) -> int:
    r = session.post(
        f"{BASE_URL}/api/admin/recruiters",
        headers=_h(admin_token),
        json={"name": name, "email": email, "password": password, "phone": "0000000000"},
        timeout=60,
    )
    if r.status_code == 200:
        return int(r.json()["id"])
    if r.status_code == 400 and "already exists" in r.text.lower():
        conn = get_db_connection(validate=False, register_pgvector=False)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                if row:
                    return int(row[0])
        finally:
            return_db_connection(conn)
    raise SystemExit(f"Create recruiter failed {email!r}: {r.status_code} {r.text}")


def _build_mapping(headers: List[str], suggested: Dict[str, str]) -> Dict[str, str]:
    mapping = {header: suggested.get(header, "ignore") for header in headers}
    aliases = {
        "FN": "first_name",
        "LN": "last_name",
        "Person LI": "linkedin",
        "City": "city",
        "Current Role": "title",
        "Company": "company_name",
    }
    for header, target in aliases.items():
        if header in mapping:
            mapping[header] = target
    required = {"first_name", "last_name", "linkedin", "city", "title"}
    missing = required - {target for target in mapping.values() if target and target != "ignore"}
    if missing:
        raise SystemExit(f"Mapping missing {missing}; headers={headers!r}; mapping={mapping!r}")
    return mapping


def _csv_bytes(slug: str) -> bytes:
    rows = [
        [
            "FN",
            "LN",
            "Person LI",
            "City",
            "Current Role",
            "Company",
            "Current CTC",
            "Expected CTC.",
            "Notice Period",
            "Preferred Loc",
            "Focused Geog",
            "Shift timings",
            "Outbound Exp",
            "Targets",
            "CV",
            "Company 1 Name",
            "Title",
            "Start date",
            "End Date",
            "Details",
            "Company 2 Name",
            "Title.1",
            "Start date.1",
            "End Date.1",
            "Details .1",
        ],
        [
            "Extra",
            "Fields",
            f"https://www.linkedin.com/in/{slug}",
            "Bengaluru",
            "Enterprise Account Executive",
            "Acme SaaS",
            "15 LPA",
            "22 LPA",
            "30 days",
            "Bangalore / Remote",
            "US and EMEA",
            "6:30 PM to 3:30 AM",
            "70% outbound",
            "120% quota attainment",
            "https://docs.example.test/cv.pdf",
            "Acme SaaS",
            "Enterprise Account Executive",
            "2021-01",
            "Present",
            "Closed enterprise SaaS new logo deals across EMEA and US with hunting and outbound ownership.",
            "Beta Cloud",
            "Sales Development Representative",
            "2018-01",
            "2020-12",
            "Built SMB pipeline for cloud software customers across India and APAC.",
        ],
    ]
    return "\n".join(",".join(f'"{cell}"' for cell in row) for row in rows).encode("utf-8")


def _upload_and_wait(session: requests.Session, recruiter_token: str, slug: str, timeout_s: float) -> int:
    csv_bytes = _csv_bytes(slug)
    pr = session.post(
        f"{BASE_URL}/api/candidates/upload/preview",
        headers=_h(recruiter_token),
        files={"file": (FILENAME, csv_bytes, "text/csv")},
        data={"use_llm": "false"},
        timeout=120,
    )
    if pr.status_code != 200:
        raise SystemExit(f"preview failed: {pr.status_code} {pr.text}")
    preview = pr.json()
    for header in ("Current CTC", "Expected CTC.", "Notice Period", "Preferred Loc", "Focused Geog", "Shift timings", "CV"):
        if preview.get("suggested_mapping", {}).get(header) != "custom":
            raise SystemExit(f"Expected {header!r} to default to custom; preview={preview!r}")

    mapping = _build_mapping(preview["headers"], preview.get("suggested_mapping") or {})
    cr = session.post(
        f"{BASE_URL}/api/candidates/upload/commit",
        headers=_h(recruiter_token),
        files={"file": (FILENAME, csv_bytes, "text/csv")},
        data={"mapping_json": json.dumps(mapping), "enrichment_mode": "verified_profile"},
        timeout=300,
    )
    if cr.status_code != 200:
        raise SystemExit(f"commit failed: {cr.status_code} {cr.text}")
    upload_id = int(cr.json()["upload_id"])

    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        sr = session.get(f"{BASE_URL}/api/candidates/uploads/{upload_id}", headers=_h(recruiter_token), timeout=60)
        if sr.status_code != 200:
            raise SystemExit(f"upload status failed: {sr.status_code} {sr.text}")
        last = sr.json()
        status = str(last.get("status") or "").lower()
        if status in ("completed", "completed_with_errors", "failed"):
            if status == "failed":
                raise SystemExit(f"upload failed: {last!r}")
            return upload_id
        time.sleep(1.0)
    raise SystemExit(f"Timed out waiting for upload {upload_id}; last={last!r}")


def _master_ids_for_marker(slug: str) -> List[int]:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("DB connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM candidates
                WHERE owner_user_id IS NULL
                  AND (linkedin ILIKE %s OR normalized_linkedin LIKE %s)
                ORDER BY id
                """,
                (f"%{slug}%", f"%{slug}%"),
            )
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        return_db_connection(conn)


def _db_fetchone(sql: str, params: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("DB connection failed")
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        return_db_connection(conn)


def _verify_storage_and_context(candidate_id: int) -> None:
    row = _db_fetchone("SELECT raw_fields FROM candidates WHERE id = %s", (candidate_id,))
    raw = row[0] if row else {}
    if not isinstance(raw, dict):
        raise SystemExit(f"raw_fields was not a dict: {raw!r}")
    extras = raw.get("imported_extra_fields") or {}
    expected = {
        "current_ctc": "15 LPA",
        "expected_ctc": "22 LPA",
        "notice_period": "30 days",
        "preferred_location": "Bangalore / Remote",
        "focused_geography": "US and EMEA",
        "shift_timings": "6:30 PM to 3:30 AM",
        "outbound_experience": "70% outbound",
        "targets": "120% quota attainment",
        "cv": "https://docs.example.test/cv.pdf",
    }
    for key, value in expected.items():
        if extras.get(key, {}).get("value") != value:
            raise SystemExit(f"Missing normalized extra {key!r}: {extras!r}")

    query.initialize_cache()
    profile = query.PROFILES_BY_ID.get(candidate_id)
    if not profile:
        raise SystemExit(f"Candidate {candidate_id} was not loaded into query cache")
    context = build_candidate_context(profile)
    if context.get("raw.Current CTC") != "15 LPA" or context.get("extra.current_ctc") != "15 LPA":
        raise SystemExit(f"AI context missing expected extra tokens: {context!r}")

    facts = compute_career_facts(context)
    if int(facts.get("total_experience_months") or 0) < 60:
        raise SystemExit(f"Expected enriched tenure >= 60 months; facts={facts!r}")
    tools = run_candidate_query_tools(
        "Has enterprise segment, hunting function, SaaS industry, and EMEA geography experience?",
        context,
        facts,
    )
    if int(tools["segment_experience"]["Enterprise"]["months"]) < 1:
        raise SystemExit(f"Enterprise segment evidence missing: {tools!r}")
    if int(tools["functional_experience"]["Hunting"]["months"]) < 1:
        raise SystemExit(f"Hunting function evidence missing: {tools!r}")
    if int(tools["industry_experience"]["SaaS"]["months"]) < 1:
        raise SystemExit(f"SaaS industry evidence missing: {tools!r}")
    if int(tools["geography_experience"]["EMEA"]["months"]) < 1:
        raise SystemExit(f"EMEA geography evidence missing: {tools!r}")


def _fetch_ai_columns(session: requests.Session, admin_token: str, candidate_ids: List[int]) -> List[Dict[str, Any]]:
    params = {"candidate_ids": ",".join(str(i) for i in candidate_ids), "view_scope": "master"}
    r = session.get(f"{BASE_URL}/api/ai-columns", headers=_h(admin_token), params=params, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"list ai-columns failed: {r.status_code} {r.text}")
    return r.json().get("columns") or []


def _poll_run_terminal(
    session: requests.Session,
    admin_token: str,
    column_id: int,
    candidate_ids: List[int],
    timeout_s: float,
) -> Dict[int, Dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        columns = _fetch_ai_columns(session, admin_token, candidate_ids)
        column = next((c for c in columns if int(c.get("id") or 0) == column_id), None)
        if not column:
            time.sleep(1.0)
            continue
        run = column.get("latest_run") or {}
        status = str(run.get("status") or "").lower()
        if status in ("completed", "completed_with_errors", "failed"):
            if status == "failed":
                raise SystemExit(f"AI column run failed: {column!r}")
            return {int(k): v for k, v in (column.get("cells_by_candidate") or {}).items()}
        time.sleep(1.0)
    raise SystemExit(f"Timed out waiting for AI column {column_id}")


def _run_ai_column_check(
    session: requests.Session,
    admin_token: str,
    candidate_id: int,
    *,
    suffix: str,
    prompt: str,
    expected_tool: str,
    expected_term: Optional[str] = None,
    output_key: str = "result",
) -> None:
    name = f"{MARKER} {suffix}"
    sr = session.post(
        f"{BASE_URL}/api/ai-columns",
        headers=_h(admin_token),
        json={
            "name": name,
            "prompt_template": prompt,
            "mode": "content",
            "output_schema": [{"key": output_key, "label": output_key.replace("_", " ").title(), "type": "text", "primary": True}],
            "required_fields": [],
            "only_run_if": {"required_fields": [], "summary": ""},
            "view_scope": "master",
            "recruiter_filter_id": None,
        },
        timeout=60,
    )
    if sr.status_code != 200:
        raise SystemExit(f"save ai-column failed {name!r}: {sr.status_code} {sr.text}")
    column_id = int(sr.json()["id"])

    rr = session.post(
        f"{BASE_URL}/api/ai-columns/run",
        headers=_h(admin_token),
        json={
            "column_definition_id": column_id,
            "selection_mode": "selected_ids",
            "selected_ids": [candidate_id],
            "view_scope": "master",
            "recruiter_filter_id": None,
        },
        timeout=60,
    )
    if rr.status_code != 200:
        raise SystemExit(f"run ai-column failed {name!r}: {rr.status_code} {rr.text}")
    cells = _poll_run_terminal(session, admin_token, column_id, [candidate_id], timeout_s=180)
    cell = cells.get(candidate_id)
    if not cell:
        raise SystemExit(f"Missing cell for {name!r}: {cells!r}")
    if str(cell.get("status") or "").lower() not in ("completed", "skipped", "failed"):
        raise SystemExit(f"Cell not terminal for {name!r}: {cell!r}")

    detail = session.get(f"{BASE_URL}/api/ai-columns/{column_id}/cells/{candidate_id}", headers=_h(admin_token), timeout=60)
    if detail.status_code != 200:
        raise SystemExit(f"cell detail failed {name!r}: {detail.status_code} {detail.text}")
    details = (detail.json() or {}).get("details") or {}
    tools = details.get("tool_results") or {}
    if expected_tool == "career_metrics":
        if int((tools.get("career_metrics") or {}).get("total_experience_months") or 0) < 60:
            raise SystemExit(f"Tenure tool result missing for {name!r}: {tools!r}")
        return
    bucket = tools.get(expected_tool) or {}
    if expected_term and int((bucket.get(expected_term) or {}).get("months") or 0) < 1:
        raise SystemExit(f"Expected {expected_tool}.{expected_term} evidence for {name!r}: {tools!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra imported fields AI-column live E2E.")
    parser.add_argument("--admin-email", default=os.getenv("E2E_ADMIN_EMAIL", "").strip())
    parser.add_argument("--admin-password", default=os.getenv("E2E_ADMIN_PASSWORD", "").strip())
    parser.add_argument("--poll-timeout", type=float, default=240.0)
    args = parser.parse_args()
    if not args.admin_email or not args.admin_password:
        raise SystemExit("Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD (or pass --admin-email / --admin-password).")

    suffix = uuid.uuid4().hex[:8]
    slug = f"{MARKER}-{suffix}"
    proc: Optional[subprocess.Popen] = None
    session = requests.Session()

    print("[extra-fields-e2e] pre-clean marker rows …")
    _db_cleanup_marker(verify=True)
    try:
        if not SKIP_SERVER:
            print("[extra-fields-e2e] starting uvicorn on :8765 …")
            proc = _start_uvicorn()
        _wait_health(session)
        admin_token = _login(session, args.admin_email, args.admin_password)
        _ensure_recruiter(session, admin_token, RECRUITER_A, RECRUITER_A_PW, "E2E Adam")
        recruiter_token = _login(session, RECRUITER_A, RECRUITER_A_PW)

        print("[extra-fields-e2e] uploading dummy extra-field candidate with verified enrichment …")
        _upload_and_wait(session, recruiter_token, slug, args.poll_timeout)
        ids = _master_ids_for_marker(slug)
        if len(ids) != 1:
            raise SystemExit(f"Expected one master candidate for {slug}, got {ids!r}")
        candidate_id = ids[0]

        print("[extra-fields-e2e] verifying raw_fields, extra tokens, and enriched tool evidence …")
        _verify_storage_and_context(candidate_id)

        print("[extra-fields-e2e] running AI-column checks …")
        _run_ai_column_check(
            session,
            admin_token,
            candidate_id,
            suffix="tenure",
            prompt="Calculate total experience, average tenure, and current job tenure in months.",
            expected_tool="career_metrics",
            output_key="total_experience_months",
        )
        _run_ai_column_check(
            session,
            admin_token,
            candidate_id,
            suffix="segment",
            prompt="Has this candidate sold to Enterprise segment customers?",
            expected_tool="segment_experience",
            expected_term="Enterprise",
        )
        _run_ai_column_check(
            session,
            admin_token,
            candidate_id,
            suffix="function",
            prompt="Does this candidate have Hunting function experience?",
            expected_tool="functional_experience",
            expected_term="Hunting",
        )
        _run_ai_column_check(
            session,
            admin_token,
            candidate_id,
            suffix="industry",
            prompt="Does this candidate have SaaS industry experience?",
            expected_tool="industry_experience",
            expected_term="SaaS",
        )
        _run_ai_column_check(
            session,
            admin_token,
            candidate_id,
            suffix="geography",
            prompt="Does this candidate have EMEA geography experience?",
            expected_tool="geography_experience",
            expected_term="EMEA",
        )
        print("[extra-fields-e2e] OK — upload, enrichment, AI context, AI runs verified.")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[extra-fields-e2e] mandatory cleanup …")
        _db_cleanup_marker(verify=True)
        _safe_delete_test_recruiter(RECRUITER_A)
        print("[extra-fields-e2e] cleanup verified.")


if __name__ == "__main__":
    main()

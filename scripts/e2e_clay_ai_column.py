#!/usr/bin/env python3
"""
Clay-style E2E: define an AI column → run on selected spreadsheet rows → poll like the grid.

Mirrors the Talent Pool / Clay flow:
  save column → POST /api/ai-columns/run → poll GET /api/ai-columns?candidate_ids=…
  until latest_run is terminal → assert per-cell statuses and optional primary_output.

- Imports two disposable candidates (LinkedIn slug prefix zz-e2e-clay-ai-*).
- Works without OPENAI_API_KEY (cells complete with the backend fallback message).
- Cleans marker rows, uploads, and the AI column definition.

Required env (same as talent pool smoke):
  E2E_ADMIN_EMAIL, E2E_ADMIN_PASSWORD

Optional:
  E2E_BASE_URL (default http://127.0.0.1:8765)
  E2E_SKIP_SERVER=1
  E2E_RECRUITER_A_EMAIL / E2E_RECRUITER_A_PASSWORD — who performs CSV upload (default adam@gmail.com / adam@123)

Usage:
  cd AI_HR && PYTHONPATH=. .venv/bin/python scripts/e2e_clay_ai_column.py
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

MARKER = "zz-e2e-clay-ai"
BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
SKIP_SERVER = os.getenv("E2E_SKIP_SERVER", "").lower() in ("1", "true", "yes")
RECRUITER_A = os.getenv("E2E_RECRUITER_A_EMAIL", "adam@gmail.com")
RECRUITER_A_PW = os.getenv("E2E_RECRUITER_A_PASSWORD", "adam@123")


def _db_cleanup_clay() -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("DB connection failed during cleanup")
    try:
        with conn.cursor() as cur:
            like = f"%{MARKER}%"
            cur.execute(
                """
                DELETE FROM ai_column_definitions
                WHERE slug ILIKE %s OR name ILIKE %s
                """,
                (like, f"%{MARKER}%"),
            )
            cur.execute(
                """
                DELETE FROM candidates
                WHERE owner_user_id IS NOT NULL
                  AND (linkedin ILIKE %s OR normalized_linkedin LIKE %s)
                """,
                (like, like),
            )
            cur.execute(
                """
                DELETE FROM candidates
                WHERE owner_user_id IS NULL
                  AND (linkedin ILIKE %s OR normalized_linkedin LIKE %s)
                """,
                (like, like),
            )
            cur.execute(
                """
                DELETE FROM candidate_uploads
                WHERE filename = %s OR file_headers::text ILIKE %s
                """,
                ("e2e_clay_ai_import.csv", f"%{MARKER}%"),
            )
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


def _ensure_recruiter(session: requests.Session, admin_token: str, email: str, password: str, name: str) -> None:
    r = session.post(
        f"{BASE_URL}/api/admin/recruiters",
        headers=_h(admin_token),
        json={"name": name, "email": email, "password": password, "phone": "0000000000"},
        timeout=60,
    )
    if r.status_code == 200:
        return
    if r.status_code == 400 and "already exists" in r.text.lower():
        return
    raise SystemExit(f"Create recruiter failed {email!r}: {r.status_code} {r.text}")


def _build_mapping(headers: List[str], suggested: Dict[str, str]) -> Dict[str, str]:
    m = {h: suggested.get(h, "ignore") for h in headers}
    alias = {
        "FN": "first_name",
        "LN": "last_name",
        "Person LI": "linkedin",
        "Metro": "city",
        "Job Title": "title",
        "Co": "company_name",
    }
    for h, t in alias.items():
        if h in m:
            m[h] = t
    required = {"first_name", "last_name", "linkedin", "city", "title"}
    used = {v for v in m.values() if v and v != "ignore"}
    missing = required - used
    if missing:
        raise SystemExit(f"Mapping missing {missing}; headers={headers!r}")
    return m


def _import_two_candidates(session: requests.Session, recruiter_token: str, slug1: str, slug2: str) -> None:
    csv_lines = [
        "FN,LN,Person LI,Metro,Job Title,Co",
        f"Clay,One,https://www.linkedin.com/in/{slug1},Austin,Engineer,AcmeClay",
        f"Clay,Two,https://www.linkedin.com/in/{slug2},Boston,Designer,BetaClay",
    ]
    csv_bytes = "\n".join(csv_lines).encode("utf-8")
    pr = session.post(
        f"{BASE_URL}/api/candidates/upload/preview",
        headers=_h(recruiter_token),
        files={"file": ("e2e_clay_ai_import.csv", csv_bytes, "text/csv")},
        data={"use_llm": "false"},
        timeout=120,
    )
    if pr.status_code != 200:
        raise SystemExit(f"preview failed: {pr.status_code} {pr.text}")
    preview = pr.json()
    mapping = _build_mapping(preview["headers"], preview.get("suggested_mapping") or {})
    cr = session.post(
        f"{BASE_URL}/api/candidates/upload/commit",
        headers=_h(recruiter_token),
        files={"file": ("e2e_clay_ai_import.csv", csv_bytes, "text/csv")},
        data={"mapping_json": json.dumps(mapping)},
        timeout=300,
    )
    if cr.status_code != 200:
        raise SystemExit(f"commit failed: {cr.status_code} {cr.text}")


def _master_ids_for_run(suffix: str) -> List[int]:
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
                (f"%{MARKER}-{suffix}%", f"%{MARKER}-{suffix}%"),
            )
            return [r[0] for r in cur.fetchall()]
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


def _fetch_ai_columns(session: requests.Session, admin_token: str, candidate_ids: List[int]) -> List[Dict[str, Any]]:
    params = {
        "candidate_ids": ",".join(str(i) for i in candidate_ids),
        "view_scope": "master",
    }
    r = session.get(f"{BASE_URL}/api/ai-columns", headers=_h(admin_token), params=params, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"list ai-columns failed: {r.status_code} {r.text}")
    return r.json().get("columns") or []


def _poll_run_terminal(
    session: requests.Session,
    admin_token: str,
    column_id: int,
    candidate_ids: List[int],
    timeout_s: float = 120.0,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    deadline = time.time() + timeout_s
    last_status = ""
    while time.time() < deadline:
        cols = _fetch_ai_columns(session, admin_token, candidate_ids)
        col = next((c for c in cols if int(c.get("id") or 0) == column_id), None)
        if not col:
            time.sleep(1.0)
            continue
        run = col.get("latest_run") or {}
        st = (run.get("status") or "").strip().lower()
        last_status = st or last_status
        cells = col.get("cells_by_candidate") or {}
        # Normalize keys to int for caller
        cells_int: Dict[int, Dict[str, Any]] = {}
        for k, v in cells.items():
            try:
                cells_int[int(k)] = v
            except (TypeError, ValueError):
                continue
        if st in ("completed", "completed_with_errors", "failed"):
            return st, cells_int
        if st in ("running", "queued") or st:
            # still in progress
            pass
        time.sleep(1.5)
    raise SystemExit(f"Timeout waiting for AI run to finish (last_status={last_status!r})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clay-style AI column live E2E (import → column → run → poll).")
    parser.add_argument("--admin-email", default=os.getenv("E2E_ADMIN_EMAIL", "").strip())
    parser.add_argument("--admin-password", default=os.getenv("E2E_ADMIN_PASSWORD", "").strip())
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.admin_email or not args.admin_password:
        raise SystemExit("Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD (or pass --admin-email / --admin-password).")

    suf = uuid.uuid4().hex[:8]
    slug1 = f"{MARKER}-{suf}-a"
    slug2 = f"{MARKER}-{suf}-b"

    print("[clay-e2e] pre-clean marker / prior definitions …")
    _db_cleanup_clay()

    proc: Optional[subprocess.Popen] = None
    if not SKIP_SERVER:
        print("[clay-e2e] starting uvicorn on :8765 …")
        proc = _start_uvicorn()

    session = requests.Session()
    try:
        _wait_health(session)
        admin_tok = _login(session, args.admin_email, args.admin_password)
        _ensure_recruiter(session, admin_tok, RECRUITER_A, RECRUITER_A_PW, "E2E Adam")
        rec_tok = _login(session, RECRUITER_A, RECRUITER_A_PW)

        print("[clay-e2e] importing two disposable candidates …")
        _import_two_candidates(session, rec_tok, slug1, slug2)

        ids = _master_ids_for_run(suf)
        if len(ids) < 2:
            raise SystemExit(f"Expected 2 master candidate ids for marker suffix {suf}, got {ids!r}")

        use_ids = ids[:2]
        print("[clay-e2e] candidate ids:", use_ids)

        # Include MARKER in the display name so DB cleanup can find stray definitions.
        col_name = f"{MARKER} col {suf}"
        payload = {
            "name": col_name,
            "prompt_template": (
                "You are testing a spreadsheet AI column. "
                "Using only the row context, output a very short greeting that includes "
                "the candidate's first name: {candidate.first_name}."
            ),
            "mode": "content",
            "output_schema": [{"key": "result", "label": "Result", "type": "text", "primary": True}],
            "required_fields": [],
            "only_run_if": {"required_fields": [], "summary": ""},
            "view_scope": "master",
            "recruiter_filter_id": None,
        }
        sr = session.post(f"{BASE_URL}/api/ai-columns", headers=_h(admin_tok), json=payload, timeout=60)
        if sr.status_code != 200:
            raise SystemExit(f"save ai-column failed: {sr.status_code} {sr.text}")
        column_id = int(sr.json().get("id") or 0)
        if not column_id:
            raise SystemExit(f"Missing column id in response: {sr.text}")

        print(f"[clay-e2e] created column id={column_id} name={col_name!r}")
        def_row = _db_fetchone(
            """
            SELECT name, mode, prompt_template, output_schema, COALESCE(is_archived, FALSE)
            FROM ai_column_definitions
            WHERE id = %s
            """,
            (column_id,),
        )
        if not def_row or def_row[0] != col_name or def_row[1] != "content" or def_row[4] is not False:
            raise SystemExit(f"definition DB verification failed: {def_row!r}")
        print("[clay-e2e] DB definition verified")

        rr = session.post(
            f"{BASE_URL}/api/ai-columns/run",
            headers=_h(admin_tok),
            json={
                "column_definition_id": column_id,
                "selection_mode": "selected_ids",
                "selected_ids": use_ids,
                "view_scope": "master",
                "recruiter_filter_id": None,
            },
            timeout=60,
        )
        if rr.status_code != 200:
            raise SystemExit(f"run ai-column failed: {rr.status_code} {rr.text}")
        print("[clay-e2e] run queued:", rr.json())

        # First poll should often show queued/running cells (Clay-style feedback)
        time.sleep(0.8)
        early = _fetch_ai_columns(session, admin_tok, use_ids)
        early_col = next((c for c in early if int(c.get("id") or 0) == column_id), None)
        if early_col:
            cells0 = early_col.get("cells_by_candidate") or {}
            print("[clay-e2e] early cell statuses:", {k: (v or {}).get("status") for k, v in cells0.items()})

        terminal, cells = _poll_run_terminal(session, admin_tok, column_id, use_ids, timeout_s=args.poll_timeout)
        print(f"[clay-e2e] terminal run status: {terminal}")
        run_row = _db_fetchone(
            """
            SELECT total, completed, failed, skipped, status
            FROM ai_column_runs
            WHERE column_definition_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (column_id,),
        )
        if not run_row or int(run_row[0]) != len(use_ids) or (run_row[4] or "").lower() != terminal:
            raise SystemExit(f"run DB verification failed: terminal={terminal!r} row={run_row!r}")
        print("[clay-e2e] DB run verified:", run_row)

        for cid in use_ids:
            cell = cells.get(cid)
            if not cell:
                raise SystemExit(f"Missing cell for candidate {cid}; cells={cells!r}")
            st = (cell.get("status") or "").lower()
            if st not in ("completed", "failed", "skipped"):
                raise SystemExit(f"Candidate {cid} expected terminal cell status, got {st!r} body={cell!r}")
            out = (cell.get("primary_output") or "").strip()
            if not out:
                raise SystemExit(f"Candidate {cid} has empty primary_output: {cell!r}")
            cell_row = _db_fetchone(
                """
                SELECT status, primary_output, outputs, last_run_id
                FROM ai_column_cells
                WHERE column_definition_id = %s AND candidate_id = %s
                """,
                (column_id, cid),
            )
            if not cell_row or (cell_row[0] or "").lower() != st or not (cell_row[1] or "").strip():
                raise SystemExit(f"cell DB verification failed candidate={cid}: {cell_row!r}")
            print(f"  [clay-e2e] candidate {cid}: status={st!r} output={out[:120]!r}")

        dr = session.get(
            f"{BASE_URL}/api/ai-columns/{column_id}/cells/{use_ids[0]}",
            headers=_h(admin_tok),
            timeout=30,
        )
        if dr.status_code != 200:
            raise SystemExit(f"cell detail failed: {dr.status_code} {dr.text}")
        print("[clay-e2e] cell detail keys:", list((dr.json() or {}).keys()))

        dr_del = session.delete(f"{BASE_URL}/api/ai-columns/{column_id}", headers=_h(admin_tok), timeout=30)
        if dr_del.status_code != 200:
            print("[clay-e2e] WARNING: delete column:", dr_del.status_code, dr_del.text)
        else:
            archived = _db_fetchone(
                "SELECT COALESCE(is_archived, FALSE) FROM ai_column_definitions WHERE id = %s",
                (column_id,),
            )
            if not archived or archived[0] is not True:
                raise SystemExit(f"delete/archive DB verification failed: {archived!r}")
            after_delete_cols = _fetch_ai_columns(session, admin_tok, use_ids)
            if any(int(c.get("id") or 0) == column_id for c in after_delete_cols):
                raise SystemExit("deleted AI column still appears in list response")
            print("[clay-e2e] archived column definition and verified list hide")

        print("[clay-e2e] OK — Clay-style define → run → poll → verify → cleanup column")

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("[clay-e2e] cleaning DB (marker candidates + uploads + stray defs) …")
    _db_cleanup_clay()
    print("[clay-e2e] done.")


if __name__ == "__main__":
    main()

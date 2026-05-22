#!/usr/bin/env python3
"""
Live E2E smoke against the configured Postgres + running FastAPI server.

- Creates recruiters adam@gmail.com and a second pool for assign checks (if missing).
- Imports a tiny CSV with nonstandard headers (human-confirmed mapping).
- Verifies browse totals for recruiter + admin master scope.
- Assigns one master profile to the second recruiter; verifies pool counts.
- Cleans up ONLY rows with LinkedIn slug prefix zz-e2e-pool-smoke-* and related uploads.
- Deletes test users ONLY if they have no other (non-test) owned candidates.

Required env:
  E2E_ADMIN_EMAIL, E2E_ADMIN_PASSWORD — existing admin login for /api/admin/*.

Optional:
  E2E_BASE_URL (default http://127.0.0.1:8765)
  E2E_SKIP_SERVER=1 — do not spawn uvicorn (expect server already running)

Usage:
  cd AI_HR && PYTHONPATH=. .venv/bin/python scripts/e2e_talent_pool_live_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Repo root = parent of scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from backend.db.connection import get_db_connection, return_db_connection  # noqa: E402

MARKER = "zz-e2e-pool-smoke"
RECRUITER_A = os.getenv("E2E_RECRUITER_A_EMAIL", "adam@gmail.com")
RECRUITER_A_PW = os.getenv("E2E_RECRUITER_A_PASSWORD", "adam@123")
RECRUITER_B = os.getenv("E2E_RECRUITER_B_EMAIL", "e2e_beta_pool_smoke@gmail.com")
RECRUITER_B_PW = os.getenv("E2E_RECRUITER_B_PASSWORD", "adam@123")
BASE_URL = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
SKIP_SERVER = os.getenv("E2E_SKIP_SERVER", "").lower() in ("1", "true", "yes")


def _db_cleanup_marker_and_uploads() -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("DB connection failed during cleanup")
    try:
        with conn.cursor() as cur:
            like = f"%{MARKER}%"
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
                WHERE filename = %s
                   OR file_headers::text ILIKE %s
                """,
                ("e2e_pool_smoke_import.csv", f"%{MARKER}%"),
            )
        conn.commit()
    finally:
        return_db_connection(conn)


def _safe_delete_test_users(emails: List[str]) -> None:
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            for email in emails:
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                if not row:
                    continue
                uid = row[0]
                cur.execute(
                    """
                    SELECT COUNT(*) FROM candidates
                    WHERE owner_user_id = %s
                      AND (linkedin NOT ILIKE %s AND (normalized_linkedin IS NULL OR normalized_linkedin NOT LIKE %s))
                    """,
                    (uid, f"%{MARKER}%", f"%{MARKER}%"),
                )
                other = cur.fetchone()[0]
                if other:
                    print(f"[cleanup] skip deleting user {email!r}: has {other} non-test owned rows")
                    continue
                cur.execute("DELETE FROM candidates WHERE owner_user_id = %s", (uid,))
                cur.execute("DELETE FROM candidate_uploads WHERE owner_user_id = %s", (uid,))
                cur.execute("DELETE FROM users WHERE id = %s", (uid,))
                print(f"[cleanup] deleted test user {email!r}")
        conn.commit()
    finally:
        return_db_connection(conn)


def _start_uvicorn() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ],
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
    r = session.post(
        f"{BASE_URL}/api/login",
        json={"email": email, "password": password},
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit(f"Login failed {email!r}: {r.status_code} {r.text}")
    data = r.json()
    return data["access_token"]


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ensure_recruiter(session: requests.Session, admin_token: str, email: str, password: str, name: str) -> int:
    r = session.post(
        f"{BASE_URL}/api/admin/recruiters",
        headers=_auth_headers(admin_token),
        json={"name": name, "email": email, "password": password, "phone": "0000000000"},
        timeout=60,
    )
    if r.status_code == 200:
        return r.json()["id"]
    if r.status_code == 400 and "already exists" in r.text.lower():
        conn = get_db_connection(validate=False, register_pgvector=False)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                if row:
                    return row[0]
        finally:
            return_db_connection(conn)
        raise SystemExit(f"Recruiter exists but not found in DB: {email}")
    raise SystemExit(f"Create recruiter failed {email!r}: {r.status_code} {r.text}")


def _browse_total(session: requests.Session, token: str, **params: Any) -> int:
    r = session.get(
        f"{BASE_URL}/api/candidates/browse",
        headers=_auth_headers(token),
        params=params,
        timeout=120,
    )
    if r.status_code != 200:
        raise SystemExit(f"browse failed: {r.status_code} {r.text}")
    return int(r.json().get("total", 0))


def _master_ids_for_marker() -> List[int]:
    conn = get_db_connection(validate=False, register_pgvector=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM candidates
                WHERE owner_user_id IS NULL
                  AND (linkedin ILIKE %s OR normalized_linkedin LIKE %s)
                ORDER BY id
                """,
                (f"%{MARKER}%", f"%{MARKER}%"),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        return_db_connection(conn)


def _build_mapping(headers: List[str], suggested: Dict[str, str]) -> Dict[str, str]:
    """Force five required targets; keep suggested for optionals."""
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
        raise SystemExit(f"Mapping still missing {missing}; headers={headers!r} suggested={suggested!r}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Talent pool live E2E smoke (prod-safe marker cleanup).")
    parser.add_argument("--admin-email", default=os.getenv("E2E_ADMIN_EMAIL", "").strip())
    parser.add_argument("--admin-password", default=os.getenv("E2E_ADMIN_PASSWORD", "").strip())
    args = parser.parse_args()
    admin_email = args.admin_email
    admin_password = args.admin_password
    if not admin_email or not admin_password:
        raise SystemExit(
            "Provide admin credentials: --admin-email / --admin-password "
            "or E2E_ADMIN_EMAIL / E2E_ADMIN_PASSWORD in the environment."
        )

    run_suffix = uuid.uuid4().hex[:8]
    slug1 = f"{MARKER}-{run_suffix}-a1"
    slug2 = f"{MARKER}-{run_suffix}-a2"
    slug3 = f"{MARKER}-{run_suffix}-a3"

    print("[e2e] pre-clean marker rows + stale uploads")
    _db_cleanup_marker_and_uploads()

    proc: Optional[subprocess.Popen] = None
    if not SKIP_SERVER:
        print("[e2e] starting uvicorn on :8765 …")
        proc = _start_uvicorn()

    session = requests.Session()
    try:
        _wait_health(session)

        admin_tok = _login(session, admin_email, admin_password)
        adam_id = _ensure_recruiter(session, admin_tok, RECRUITER_A, RECRUITER_A_PW, "E2E Adam")
        beta_id = _ensure_recruiter(session, admin_tok, RECRUITER_B, RECRUITER_B_PW, "E2E Beta")

        adam_tok = _login(session, RECRUITER_A, RECRUITER_A_PW)
        beta_tok = _login(session, RECRUITER_B, RECRUITER_B_PW)

        before_adam = _browse_total(session, adam_tok, page=1, page_size=500, q=MARKER)
        before_beta = _browse_total(session, beta_tok, page=1, page_size=500, q=MARKER)

        csv_lines = [
            "FN,LN,Person LI,Metro,Job Title,Co",
            f"Ann,Alpha,https://www.linkedin.com/in/{slug1},Austin,Eng A,Acme",
            f"Bob,Beta,https://www.linkedin.com/in/{slug2},Boston,Eng B,BetaCo",
            f"Cara,Gamma,https://www.linkedin.com/in/{slug3},Chicago,Eng C,GammaInc",
        ]
        csv_bytes = "\n".join(csv_lines).encode("utf-8")

        pr = session.post(
            f"{BASE_URL}/api/candidates/upload/preview",
            headers=_auth_headers(adam_tok),
            files={"file": ("e2e_pool_smoke_import.csv", csv_bytes, "text/csv")},
            data={"use_llm": "false"},
            timeout=120,
        )
        if pr.status_code != 200:
            raise SystemExit(f"preview failed: {pr.status_code} {pr.text}")
        preview = pr.json()
        headers = preview["headers"]
        suggested = preview.get("suggested_mapping") or {}
        mapping = _build_mapping(headers, suggested)
        mapping_json = json.dumps(mapping)

        cr = session.post(
            f"{BASE_URL}/api/candidates/upload/commit",
            headers=_auth_headers(adam_tok),
            files={"file": ("e2e_pool_smoke_import.csv", csv_bytes, "text/csv")},
            data={"mapping_json": mapping_json},
            timeout=300,
        )
        if cr.status_code != 200:
            raise SystemExit(f"commit failed: {cr.status_code} {cr.text}")
        commit = cr.json()
        print("[e2e] commit response:", commit)
        if commit.get("inserted", 0) + commit.get("updated", 0) != 3:
            print("[e2e] WARNING: expected 3 upserts; check skipped/errors", commit)

        after_adam = _browse_total(session, adam_tok, page=1, page_size=500, q=MARKER)
        master_before_assign = _browse_total(
            session,
            admin_tok,
            page=1,
            page_size=500,
            q=MARKER,
            view_scope="master",
        )

        if after_adam - before_adam != 3:
            raise SystemExit(
                f"Recruiter browse delta mismatch: before={before_adam} after={after_adam} expected +3"
            )
        if master_before_assign < 3:
            raise SystemExit(f"Expected >=3 master rows for marker, got {master_before_assign}")

        mids = _master_ids_for_marker()
        if len(mids) < 3:
            raise SystemExit(f"Expected 3 master ids in DB, got {mids!r}")

        assign_id = mids[-1]
        ar = session.post(
            f"{BASE_URL}/api/admin/candidates/assign-to-recruiter",
            headers=_auth_headers(admin_tok),
            json={"master_candidate_ids": [assign_id], "recruiter_user_id": beta_id},
            timeout=120,
        )
        if ar.status_code != 200:
            raise SystemExit(f"assign failed: {ar.status_code} {ar.text}")
        print("[e2e] assign response:", ar.json())

        after_beta = _browse_total(session, beta_tok, page=1, page_size=500, q=MARKER)
        if after_beta - before_beta != 1:
            raise SystemExit(
                f"Beta pool delta mismatch: before={before_beta} after={after_beta} expected +1"
            )

        uploads = session.get(
            f"{BASE_URL}/api/candidates/uploads",
            headers=_auth_headers(adam_tok),
            timeout=60,
        )
        if uploads.status_code != 200:
            raise SystemExit(f"uploads list failed: {uploads.status_code} {uploads.text}")
        up_json = uploads.json().get("uploads") or []
        latest = next((u for u in up_json if u.get("filename") == "e2e_pool_smoke_import.csv"), None)
        if not latest:
            print("[e2e] WARNING: recent uploads missing our filename", up_json[:3])
        else:
            rc = latest.get("row_count")
            ins = latest.get("inserted_count") or 0
            upd = latest.get("updated_count") or 0
            if rc != 3:
                print(f"[e2e] WARNING: upload row_count={rc} expected 3")
            if ins + upd != 3:
                print(f"[e2e] WARNING: inserted+updated={ins}+{upd} expected 3")

        print("[e2e] OK — import, browse counts, assign, and uploads audit look consistent.")

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("[e2e] cleaning DB (marker candidates + uploads + safe user delete) …")
    _db_cleanup_marker_and_uploads()
    _safe_delete_test_users([RECRUITER_A, RECRUITER_B])
    print("[e2e] done.")


if __name__ == "__main__":
    main()

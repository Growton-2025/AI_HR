"""End-to-end check of the LinkedIn URL import: HTTP route -> Apify -> mapper -> DB.

Boots only the import + auth routers (not ``backend.main``, whose lifespan starts
outreach pollers and registers webhooks), mints a JWT for the owner, posts the
URL, polls the upload status, then reads the result back from the database and
prints exact external call counts (Apify runs, OpenAI calls).

    python scripts/import_linkedin_e2e.py --url https://www.linkedin.com/in/<slug> \
        --owner-email codex.upload.ui.20260522@growton.co --delete-existing
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import requests  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from backend.api.routes import auth, candidate_imports  # noqa: E402
from backend.core.security import create_access_token  # noqa: E402
from backend.db.connection import get_db_connection, return_db_connection  # noqa: E402
from backend.services import apify_scrape, import_enrichment, profile_mapper  # noqa: E402
from backend.services.linkedin_normalize import normalize_linkedin  # noqa: E402

CHILD_TABLES = ("roles", "titles_held", "education", "company_years", "experience_gaps", "education_gaps",
                "industry_gaps", "functional_experiences", "industry_experiences", "segment_experiences",
                "geography_experiences", "recruitment_role_candidates")

CALLS = {"apify_runs": 0, "openai_score_gpt4o": 0, "openai_embedding": 0, "openai_company_web_research": 0,
         "openai_classify_web_research": 0}


def _count(module, name, key):
    real = getattr(module, name)

    def wrapped(*a, **kw):
        CALLS[key] += 1
        return real(*a, **kw)

    setattr(module, name, wrapped)


def _count_apify_posts():
    real_post = apify_scrape.requests.post

    def wrapped(url, *a, **kw):
        if "api.apify.com" in url:
            CALLS["apify_runs"] += 1
        return real_post(url, *a, **kw)

    apify_scrape.requests.post = wrapped


def db():
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise SystemExit("Database connection failed")
    return conn


def existing_rows(cur, key):
    cur.execute("SELECT id, name, owner_user_id, pool_source, created_by, created_at FROM candidates "
                "WHERE normalized_linkedin = %s ORDER BY id", (key,))
    return cur.fetchall()


def child_counts(cur, cid):
    out = {}
    for t in CHILD_TABLES:
        cur.execute(f"SELECT count(*) FROM {t} WHERE candidate_id = %s", (cid,))
        out[t] = cur.fetchone()[0]
    return out


def start_server(port: int) -> str:
    app = FastAPI(title="linkedin-import-e2e")
    app.include_router(auth.router, prefix="/api")
    app.include_router(candidate_imports.router, prefix="/api")
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{base}/api/candidates/uploads/0", timeout=2).status_code in (401, 404):
                return base
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise SystemExit("test server did not start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--owner-email", default="codex.upload.ui.20260522@growton.co")
    ap.add_argument("--delete-existing", action="store_true",
                    help="delete every candidates row (any owner) for this LinkedIn URL before importing")
    ap.add_argument("--duplicate-policy", default="skip", choices=["skip", "replace"])
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    key = normalize_linkedin(args.url)
    print(f"profile key: {key}")

    conn = db()
    try:
        with conn.cursor() as cur:
            before = existing_rows(cur, key)
            print(f"existing rows for this profile: {len(before)}")
            for row in before:
                print("  ", row, child_counts(cur, row[0]))
            if args.delete_existing and before:
                ids = [r[0] for r in before]
                cur.execute("DELETE FROM candidates WHERE id = ANY(%s)", (ids,))
                conn.commit()
                print(f"deleted candidates {ids} (children cascade)")
            cur.execute("SELECT id, email, role FROM users WHERE email = %s", (args.owner_email,))
            owner = cur.fetchone()
            if not owner:
                raise SystemExit(f"no user with email {args.owner_email}")
            print(f"owner: id={owner[0]} email={owner[1]} role={owner[2]}")
    finally:
        return_db_connection(conn)

    _count_apify_posts()
    _count(profile_mapper, "score_candidate", "openai_score_gpt4o")
    _count(profile_mapper, "embed", "openai_embedding")
    _count(profile_mapper, "research_company", "openai_company_web_research")
    _count(import_enrichment, "_web_research_company", "openai_classify_web_research")

    base = start_server(args.port)
    token = create_access_token({"sub": args.owner_email}, timedelta(hours=1))
    headers = {"Authorization": f"Bearer {token}"}
    me = requests.get(f"{base}/api/me", headers=headers, timeout=30)
    print(f"GET /api/me -> {me.status_code} {me.json() if me.ok else me.text[:200]}")
    if not me.ok:
        return 1

    started = time.time()
    resp = requests.post(f"{base}/api/candidates/import/linkedin",
                         json={"linkedin_urls": [args.url], "duplicate_policy": args.duplicate_policy},
                         headers=headers, timeout=60)
    print(f"POST /api/candidates/import/linkedin -> {resp.status_code} {resp.text[:400]}")
    if not resp.ok:
        return 1
    upload_id = resp.json()["upload_id"]

    last = None
    while time.time() - started < args.timeout:
        status = requests.get(f"{base}/api/candidates/uploads/{upload_id}", headers=headers, timeout=30).json()
        line = (status["status"], status["processed_count"], status["inserted"], status["updated"], status["skipped"])
        if line != last:
            print(f"  t+{time.time() - started:5.1f}s status={line[0]} processed={line[1]} inserted={line[2]} "
                  f"replaced={line[3]} skipped={line[4]}")
            last = line
        if status["status"] in ("completed", "completed_with_errors", "failed"):
            break
        time.sleep(3)
    print("final upload status:", json.dumps(status, indent=2, default=str))
    print("external calls:", json.dumps(CALLS))

    conn = db()
    try:
        with conn.cursor() as cur:
            after = existing_rows(cur, key)
            print(f"rows for this profile after import: {len(after)}")
            for row in after:
                cid = row[0]
                print("  ", row)
                print("   children:", child_counts(cur, cid))
                cur.execute("""SELECT name, headline, location, city, total_experience_years, avg_years_in_company,
                                      has_gap_years, has_education_gaps, has_industry_gaps,
                                      functional_experience_score, industry_experience_score, segment_experience_score,
                                      geography_experience_score, team_management_score, max_people_managed,
                                      years_team_management, embedding IS NOT NULL, skills IS NOT NULL,
                                      licenses_and_certifications IS NOT NULL, source_upload_ids, status
                               FROM candidates WHERE id = %s""", (cid,))
                cols = ["name", "headline", "location", "city", "total_experience_years", "avg_years_in_company",
                        "has_gap_years", "has_education_gaps", "has_industry_gaps", "functional_score",
                        "industry_score", "segment_score", "geography_score", "team_mgmt_score",
                        "max_people_managed", "years_team_management", "embedding_stored", "skills_stored",
                        "certs_stored", "source_upload_ids", "status"]
                print("   candidate:", json.dumps(dict(zip(cols, cur.fetchone())), indent=2, default=str))
                cur.execute("""SELECT r.title, c.name, r.duration_years, c.funding_stage, c.business_model, c.headquarters
                               FROM roles r LEFT JOIN companies c ON c.id = r.company_id
                               WHERE r.candidate_id = %s ORDER BY r.id""", (cid,))
                print("   roles/companies:")
                for r in cur.fetchall():
                    print("     ", r)
                cur.execute("SELECT college, degree, start_date, end_date FROM education WHERE candidate_id = %s", (cid,))
                print("   education:", cur.fetchall())
    finally:
        return_db_connection(conn)
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

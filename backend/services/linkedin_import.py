"""Import candidates straight from LinkedIn URLs.

Chain: ``apify_scrape.scrape_profiles`` (LinkedIn -> wide Apify rows) ->
``profile_mapper.map_scraped_profile`` (rows -> every candidate table) ->
``profile_persist.persist_mapped_profile`` (one transaction per candidate).

Progress is written to the same ``candidate_uploads`` row the spreadsheet upload
uses, so ``GET /api/candidates/uploads/{id}`` and the frontend poller work
unchanged. The DB helpers are kept small and module-level so tests can replace
them without a database.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from backend.db.connection import get_db_connection, return_db_connection
from backend.services.apify_scrape import ApifyScrapeError, scrape_profiles
from backend.services.candidate_pool import POOL_SOURCE_LEGACY_MASTER, POOL_SOURCE_RECRUITER_UPLOAD
from backend.services.imported_fields import IMPORTED_EXTRA_FIELDS_KEY
from backend.services.linkedin_normalize import normalize_linkedin
from backend.services.profile_mapper import map_scraped_profile
from backend.services.profile_persist import persist_mapped_profile

logger = logging.getLogger(__name__)

CREATED_BY = "linkedin_import"
DUPLICATE_SKIP = "skip"
DUPLICATE_REPLACE = "replace"
DUPLICATE_POLICIES = (DUPLICATE_SKIP, DUPLICATE_REPLACE)

COMPANY_COLUMNS = (
    "name", "funding_stage", "revenue", "business_model", "product_service",
    "customer_segment", "customer_presence", "culture_type", "headquarters",
)

_threads: Dict[int, threading.Thread] = {}
_threads_lock = threading.Lock()


class LinkedInImportError(ValueError):
    """Bad input for an import request; ``detail`` is safe to return to the client."""

    def __init__(self, detail: Any):
        super().__init__(str(detail))
        self.detail = detail


def prepare_import_urls(urls: Sequence[str]) -> List[str]:
    """Validate, de-duplicate and cap the URLs for one import request."""
    checked = validate_linkedin_urls(urls)
    if checked["invalid"]:
        raise LinkedInImportError({"message": "Not LinkedIn profile URLs (expected linkedin.com/in/...)",
                                   "invalid": checked["invalid"][:50]})
    if not checked["valid"]:
        raise LinkedInImportError("Provide at least one LinkedIn profile URL")
    cap = int(os.getenv("APIFY_MAX_PROFILES_PER_REQUEST", "50") or 50)
    if len(checked["valid"]) > cap:
        raise LinkedInImportError(f"At most {cap} profiles per import; got {len(checked['valid'])}")
    return checked["valid"]


def normalize_duplicate_policy(policy: Optional[str]) -> str:
    value = str(policy or DUPLICATE_SKIP).strip().lower()
    return value if value in DUPLICATE_POLICIES else DUPLICATE_SKIP


def validate_linkedin_urls(urls: Sequence[str]) -> Dict[str, List[str]]:
    """Split input into de-duplicated profile URLs and rejects (not ``/in/`` profiles)."""
    valid: List[str] = []
    invalid: List[str] = []
    seen = set()
    for raw in urls:
        text = str(raw or "").strip()
        if not text:
            continue
        key = normalize_linkedin(text)
        if not key or not key.startswith("/in/"):
            invalid.append(text)
            continue
        if key in seen:
            continue
        seen.add(key)
        valid.append(text)
    return {"valid": valid, "invalid": invalid}


# ----------------------------------------------------------------------- DB helpers
def _with_conn(fn):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            result = fn(cur)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_connection(conn)


def create_upload_row(*, owner_user_id: int, urls: Sequence[str], role_id: Optional[int],
                      duplicate_policy: str) -> int:
    def run(cur):
        cur.execute(
            """
            INSERT INTO candidate_uploads (owner_user_id, filename, file_headers, mapping, row_count,
                                           processed_count, status, role_id)
            VALUES (%s, %s, '[]'::jsonb, %s::jsonb, %s, 0, 'processing', %s)
            RETURNING id
            """,
            (owner_user_id, f"linkedin-import ({len(urls)} profiles)",
             '{"source": "linkedin_urls", "duplicate_policy": "%s"}' % duplicate_policy,
             len(urls), role_id),
        )
        return cur.fetchone()[0]

    return _with_conn(run)


def write_progress(upload_id: int, **updates: Any) -> None:
    if not updates:
        return

    def run(cur):
        fields = ", ".join(f"{k} = %s" for k in updates)
        cur.execute(f"UPDATE candidate_uploads SET {fields}, updated_at = NOW() WHERE id = %s",
                    (*updates.values(), upload_id))

    _with_conn(run)


def existing_candidates(normalized_li: str, owner_user_id: int) -> List[Dict[str, Any]]:
    """Rows this owner already has for the profile. Masters (owner NULL) are not counted."""
    def run(cur):
        cur.execute(
            "SELECT id, pool_source FROM candidates WHERE normalized_linkedin = %s AND owner_user_id = %s",
            (normalized_li, owner_user_id),
        )
        return [{"id": r[0], "pool_source": r[1]} for r in cur.fetchall()]

    return _with_conn(run)


def delete_candidates(ids: Sequence[int]) -> int:
    """Delete owned rows; children cascade. Legacy masters are never deleted."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0

    def run(cur):
        cur.execute("DELETE FROM candidates WHERE id = ANY(%s) AND pool_source <> %s",
                    (ids, POOL_SOURCE_LEGACY_MASTER))
        return cur.rowcount

    return _with_conn(run)


def known_companies(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Existing companies rows keyed by lower-case name, so they are reused not re-researched."""
    wanted = sorted({str(n).strip().lower() for n in names if str(n or "").strip()})
    if not wanted:
        return {}

    def run(cur):
        cur.execute(f"SELECT {', '.join(COMPANY_COLUMNS)} FROM companies WHERE LOWER(name) = ANY(%s)", (wanted,))
        return {row[0].lower(): dict(zip(COMPANY_COLUMNS, row)) for row in cur.fetchall()}

    return _with_conn(run)


def assign_role(role_id: Optional[int], candidate_id: int) -> bool:
    if not role_id:
        return False

    def run(cur):
        cur.execute(
            """
            INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
            VALUES (%s, %s, '--', '')
            ON CONFLICT (role_id, candidate_id) DO NOTHING
            """,
            (role_id, candidate_id),
        )
        return cur.rowcount > 0

    return _with_conn(run)


def refresh_caches(candidate_ids: Sequence[int]) -> None:
    try:
        from backend.api.routes.candidates import invalidate_candidate_count_caches

        invalidate_candidate_count_caches(refresh_profile_ids=sorted({int(i) for i in candidate_ids}))
    except Exception:
        logger.exception("linkedin import could not invalidate count caches")


# ------------------------------------------------------------------------ the run
def _company_names(raw: Dict[str, Any]) -> List[str]:
    out = []
    for i in range(0, 60):
        name = raw.get(f"experiences/{i}/companyName") or raw.get(f"experiences/{i}/subtitle")
        if name:
            out.append(str(name).split("·")[0].strip())
    return out


def _row_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = normalize_linkedin(row.get("linkedinPublicUrl") or row.get("linkedinUrl") or "")
        if key and key not in index:
            index[key] = row
    return index


def _batch_size() -> int:
    try:
        return max(1, int(os.getenv("APIFY_MAX_PROFILES_PER_REQUEST", "50") or 50))
    except ValueError:
        return 50


def apply_row_extras(tables: Dict[str, Any], extras: Optional[Dict[str, Any]]) -> None:
    """Lay one source-sheet row over a scraped profile, in place.

    LinkedIn stays the source for profile fields. The sheet wins for contact
    details and notes, and its custom columns are kept in ``raw_fields`` (under
    ``imported_extra_fields``, same as a spreadsheet upload) so AI columns can read them.
    """
    if not extras:
        return
    cand = tables.get("candidates")
    if not isinstance(cand, dict):
        return
    for field in ("email", "phone", "notes"):
        value = extras.get(field)
        if value not in (None, ""):
            cand[field] = value
    raw_extra = extras.get("raw") or {}
    if raw_extra:
        raw_fields = dict(cand.get("raw_fields") or {})
        existing_block = raw_fields.get(IMPORTED_EXTRA_FIELDS_KEY)
        incoming_block = raw_extra.get(IMPORTED_EXTRA_FIELDS_KEY)
        for key, value in raw_extra.items():
            # never let a sheet column replace a scraped LinkedIn key
            if key != IMPORTED_EXTRA_FIELDS_KEY and key not in raw_fields:
                raw_fields[key] = value
        if isinstance(incoming_block, dict):
            merged = dict(existing_block) if isinstance(existing_block, dict) else {}
            merged.update(incoming_block)
            raw_fields[IMPORTED_EXTRA_FIELDS_KEY] = merged
        cand["raw_fields"] = raw_fields


def run_linkedin_import(*, upload_id: int, urls: Sequence[str], owner_user_id: int,
                        role_id: Optional[int] = None, duplicate_policy: str = DUPLICATE_SKIP,
                        use_llm: bool = True, use_web: bool = True,
                        row_extras: Optional[Dict[str, Dict[str, Any]]] = None,
                        defer_missing: bool = False, finalize: bool = True,
                        quiet_duplicates: bool = False) -> Dict[str, Any]:
    """Scrape, map and persist every URL; returns the final counts. Never raises for a
    single bad profile - those become ``errors`` lines and ``skipped`` counts.

    ``row_extras`` maps a normalized LinkedIn key to the source-sheet row for that
    person (see ``apply_row_extras``). With ``defer_missing`` the profiles Apify could
    not return, or that failed to map, are not counted here; they come back in
    ``deferred`` so the caller can load them from its own data instead. With
    ``finalize=False`` the upload row is left open for the caller's next phase.
    ``quiet_duplicates`` drops the per-row "already in your pool" line, for re-syncs
    where most rows are expected to exist already.
    """
    duplicate_policy = normalize_duplicate_policy(duplicate_policy)
    row_extras = row_extras or {}
    inserted = replaced = skipped = assigned = processed = 0
    errors: List[str] = []
    touched: List[int] = []
    results: List[Dict[str, Any]] = []
    deferred: List[str] = []

    # Apify bills per returned profile, so decide duplicates before scraping and
    # only send the URLs that will actually be loaded.
    dupes_by_key: Dict[str, List[Dict[str, Any]]] = {}
    to_scrape: List[str] = []
    for url in urls:
        key = normalize_linkedin(url)
        dupes_by_key[key] = existing_candidates(key, owner_user_id)
        if dupes_by_key[key] and duplicate_policy == DUPLICATE_SKIP:
            continue
        to_scrape.append(url)

    # One actor run per chunk: the actor caps profiles per request.
    rows: List[Dict[str, Any]] = []
    apify_failed: List[str] = []
    apify_requested = 0
    not_scraped: Dict[str, str] = {}
    size = _batch_size()
    chunks = [to_scrape[i:i + size] for i in range(0, len(to_scrape), size)]
    for index, chunk in enumerate(chunks):
        try:
            write_progress(upload_id, status="scraping")
            scraped = scrape_profiles(chunk)
        except ApifyScrapeError as exc:
            if index == 0:
                write_progress(upload_id, status="failed", completed_at=datetime.now(timezone.utc),
                               error_message=f"apify: {exc}"[:8000])
                return {"status": "failed", "error": str(exc), "inserted": 0, "updated": 0,
                        "skipped": len(urls), "processed": 0, "role_assigned_count": 0,
                        "errors": [f"apify: {exc}"], "results": [], "deferred": []}
            # Keep what earlier chunks already returned (and were billed for); the
            # rest is reported per row so a later sync can pick it up.
            logger.exception("apify chunk %s of %s failed", index + 1, len(chunks))
            for rest in chunks[index:]:
                for url in rest:
                    not_scraped[normalize_linkedin(url)] = str(exc)
            break
        rows += scraped["rows"]
        apify_failed += scraped["failed"]
        apify_requested += scraped["requested"]

    by_key = _row_index(rows)
    write_progress(upload_id, status="enriching")

    for url in urls:
        key = normalize_linkedin(url)
        raw = by_key.get(key)
        dupes = dupes_by_key.get(key, [])
        outcome: Dict[str, Any] = {"url": url}
        counted = True
        try:
            if dupes and duplicate_policy == DUPLICATE_SKIP:
                skipped += 1
                if not quiet_duplicates:
                    errors.append(f"{url}: already in your pool as candidate {dupes[0]['id']} (skipped, not scraped)")
                outcome.update(status="duplicate", candidate_id=dupes[0]["id"])
                # The person is still wanted in this role even though nothing was re-imported.
                if assign_role(role_id, dupes[0]["id"]):
                    assigned += 1
                continue

            if key in not_scraped:
                skipped += 1
                errors.append(f"{url}: not scraped, Apify failed ({not_scraped[key]})"[:500])
                outcome["status"] = "not_scraped"
                continue

            if raw is None:
                outcome["status"] = "not_found"
                if defer_missing:
                    counted = False
                    deferred.append(url)
                    errors.append(f"{url}: no profile returned by Apify; loaded from the sheet values instead")
                else:
                    skipped += 1
                    errors.append(f"{url}: no profile returned by Apify (private, removed, or blocked)")
                continue

            tables = map_scraped_profile(raw, use_llm=use_llm, use_web=use_web,
                                         known_companies=known_companies(_company_names(raw)))
            apply_row_extras(tables, row_extras.get(key))
            if dupes:
                delete_candidates([d["id"] for d in dupes])
            saved = persist_mapped_profile(
                tables, owner_user_id=owner_user_id, pool_source=POOL_SOURCE_RECRUITER_UPLOAD,
                dry_run=False, created_by=CREATED_BY,
                extra_candidate_fields={"source_upload_ids": [upload_id]},
            )
            cid = saved["candidate_id"]
            touched.append(cid)
            if dupes:
                replaced += 1
            else:
                inserted += 1
            if assign_role(role_id, cid):
                assigned += 1
            outcome.update(status="replaced" if dupes else "inserted", candidate_id=cid,
                           counts=saved["counts"], coverage=tables.get("coverage"))
        except Exception as exc:  # one profile must not sink the batch
            logger.exception("linkedin import failed for %s", url)
            outcome.update(status="error", error=str(exc))
            if defer_missing and not dupes:
                counted = False
                deferred.append(url)
                errors.append(f"{url}: LinkedIn enrichment failed ({exc}); loaded from the sheet values instead"[:500])
            else:
                skipped += 1
                errors.append(f"{url}: {exc}"[:500])
        finally:
            if counted:
                processed += 1
            results.append(outcome)
            write_progress(upload_id, processed_count=processed, inserted_count=inserted,
                           updated_count=replaced, skipped_count=skipped, role_assigned_count=assigned,
                           error_message="\n".join(errors)[:8000] or None)

    status = "completed_with_errors" if errors else "completed"
    if finalize:
        write_progress(upload_id, status=status, completed_at=datetime.now(timezone.utc))
    if touched:
        refresh_caches(touched)
    logger.info("linkedin import upload_id=%s inserted=%s replaced=%s skipped=%s deferred=%s", upload_id,
                inserted, replaced, skipped, len(deferred))
    return {"status": status, "inserted": inserted, "updated": replaced, "skipped": skipped,
            "processed": processed, "role_assigned_count": assigned, "errors": errors, "results": results,
            "deferred": deferred, "apify_requested": apify_requested, "apify_failed": apify_failed}


def start_linkedin_import(*, urls: Sequence[str], owner_user_id: int, role_id: Optional[int] = None,
                          duplicate_policy: str = DUPLICATE_SKIP) -> int:
    """Create the tracking row and run the import on a daemon thread. Returns upload_id."""
    duplicate_policy = normalize_duplicate_policy(duplicate_policy)
    upload_id = create_upload_row(owner_user_id=owner_user_id, urls=urls, role_id=role_id,
                                  duplicate_policy=duplicate_policy)

    def run() -> None:
        try:
            run_linkedin_import(upload_id=upload_id, urls=urls, owner_user_id=owner_user_id,
                                role_id=role_id, duplicate_policy=duplicate_policy)
        except Exception as exc:
            logger.exception("linkedin import crashed upload_id=%s", upload_id)
            try:
                write_progress(upload_id, status="failed", completed_at=datetime.now(timezone.utc),
                               error_message=str(exc)[:8000])
            except Exception:
                logger.exception("could not mark linkedin import failed")
        finally:
            with _threads_lock:
                _threads.pop(upload_id, None)

    worker = threading.Thread(target=run, daemon=True, name=f"linkedin-import-{upload_id}")
    with _threads_lock:
        _threads[upload_id] = worker
    worker.start()
    return upload_id

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
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from backend.db.connection import get_db_connection, return_db_connection
from backend.services.apify_scrape import ApifyScrapeError, scrape_profiles
from backend.services.candidate_pool import POOL_SOURCE_LEGACY_MASTER, POOL_SOURCE_RECRUITER_UPLOAD
from backend.services.imported_fields import IMPORTED_EXTRA_FIELDS_KEY
from backend.services.linkedin_normalize import normalize_linkedin
from backend.services.profile_mapper import CompanyCache, map_scraped_profile
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


def existing_candidates_bulk(keys: Sequence[str], owner_user_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """Rows this owner already has for each profile, in one query. Masters (owner NULL)
    are not counted. Every key is present in the result."""
    wanted = sorted({k for k in keys if k})
    found: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    if not wanted:
        return found

    def run(cur):
        cur.execute(
            "SELECT normalized_linkedin, id, pool_source FROM candidates "
            "WHERE normalized_linkedin = ANY(%s) AND owner_user_id = %s ORDER BY id",
            (wanted, owner_user_id),
        )
        return cur.fetchall()

    for key, cid, pool_source in _with_conn(run):
        found.setdefault(key, []).append({"id": cid, "pool_source": pool_source})
    return found


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


def _scrape_workers() -> int:
    """Actor runs in flight at once, after the first one has proved Apify is answering."""
    try:
        return max(1, min(5, int(os.getenv("APIFY_PARALLEL_RUNS", "3") or 3)))
    except ValueError:
        return 3


def _map_workers() -> int:
    """Profiles being mapped (model and web calls) at once."""
    try:
        return max(1, min(16, int(os.getenv("LINKEDIN_IMPORT_MAP_WORKERS", "8") or 8)))
    except ValueError:
        return 8


def _save_workers() -> int:
    """Profiles being written to the database at once. Kept well below the DB pool size."""
    try:
        return max(1, min(6, int(os.getenv("LINKEDIN_IMPORT_SAVE_WORKERS", "3") or 3)))
    except ValueError:
        return 3


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
                        quiet_duplicates: bool = False,
                        report: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """Scrape, map and persist every URL; returns the final counts. Never raises for a
    single bad profile - those become ``errors`` lines and ``skipped`` counts.

    ``row_extras`` maps a normalized LinkedIn key to the source-sheet row for that
    person (see ``apply_row_extras``). With ``defer_missing`` the profiles Apify could
    not return, or that failed to map, are not counted here; they come back in
    ``deferred`` so the caller can load them from its own data instead. With
    ``finalize=False`` the upload row is left open for the caller's next phase.
    ``quiet_duplicates`` drops the per-row "already in your pool" line, for re-syncs
    where most rows are expected to exist already. ``report`` receives the progress
    updates instead of the upload row, for a caller that merges several phases.

    The run is a stream. Actor runs go out side by side; each profile is mapped (model
    and web calls, seconds each) and saved by one worker as soon as its run returns, a
    few saves at a time. Counts move as profiles finish; the returned ``results`` and
    ``errors`` are in the order the URLs were given.
    """
    started = time.monotonic()
    duplicate_policy = normalize_duplicate_policy(duplicate_policy)
    row_extras = row_extras or {}
    report = report or (lambda **updates: write_progress(upload_id, **updates))
    keys = [normalize_linkedin(u) for u in urls]

    # Apify bills per returned profile, so decide duplicates before scraping and
    # only send the URLs that will actually be loaded.
    dupes_by_key = existing_candidates_bulk(keys, owner_user_id)
    skip_as_duplicate = {k for k in keys if dupes_by_key.get(k) and duplicate_policy == DUPLICATE_SKIP}
    to_scrape = [u for u, k in zip(urls, keys) if k not in skip_as_duplicate]
    url_of = {normalize_linkedin(u): u for u in reversed(to_scrape)}

    # ---- shared state: every row ends up with exactly one outcome
    lock = threading.Lock()
    outcomes: Dict[str, Dict[str, Any]] = {}       # key -> {"result", "error", "deferred"}
    counts = {"inserted": 0, "replaced": 0, "skipped": 0, "assigned": 0, "processed": 0}
    touched: List[int] = []
    timing = {"save": 0.0, "scrape": 0.0, "last_report": 0.0}

    def error_lines() -> List[str]:
        return [outcomes[k]["error"] for k in dict.fromkeys(keys) if k in outcomes and outcomes[k].get("error")]

    def settle(key: str, result: Dict[str, Any], *, error: Optional[str] = None, deferred: bool = False,
               **bump: int) -> None:
        """Record one row's outcome and move the counters. Deferred rows are not counted
        here; the caller loads them another way."""
        with lock:
            if key in outcomes:
                return
            outcomes[key] = {"result": {"url": url_of.get(key) or key, **result}, "error": error, "deferred": deferred}
            for name, amount in bump.items():
                counts[name] += amount
            if not deferred:
                counts["processed"] += 1
            now = time.monotonic()
            due = now - timing["last_report"] >= 1.0
            if due:
                timing["last_report"] = now
            snapshot = dict(counts)
            lines = error_lines() if due else []
        if due:      # at most one progress write a second; the final write is unconditional
            report(processed_count=snapshot["processed"], inserted_count=snapshot["inserted"],
                   updated_count=snapshot["replaced"], skipped_count=snapshot["skipped"],
                   role_assigned_count=snapshot["assigned"], error_message="\n".join(lines)[:8000] or None)

    # ---- one job per profile: map it, then save it
    company_cache = CompanyCache()
    save_gate = threading.BoundedSemaphore(_save_workers())     # the DB pool is small
    workers = ThreadPoolExecutor(max_workers=_map_workers(), thread_name_prefix="linkedin-load")
    jobs: Dict[str, Any] = {}

    def load_profile(key: str, raw: Dict[str, Any]) -> None:
        url = url_of[key]
        dupes = dupes_by_key.get(key, [])
        try:
            tables = map_scraped_profile(raw, use_llm=use_llm, use_web=use_web,
                                         known_companies=known_companies(_company_names(raw)),
                                         company_cache=company_cache)
            apply_row_extras(tables, row_extras.get(key))
            with save_gate:
                save_started = time.monotonic()
                if dupes:
                    delete_candidates([d["id"] for d in dupes])
                saved = persist_mapped_profile(
                    tables, owner_user_id=owner_user_id, pool_source=POOL_SOURCE_RECRUITER_UPLOAD,
                    dry_run=False, created_by=CREATED_BY,
                    extra_candidate_fields={"source_upload_ids": [upload_id]},
                )
                cid = saved["candidate_id"]
                joined_role = assign_role(role_id, cid)
                with lock:
                    timing["save"] += time.monotonic() - save_started
                    touched.append(cid)
            settle(key, {"status": "replaced" if dupes else "inserted", "candidate_id": cid,
                         "counts": saved["counts"], "coverage": tables.get("coverage")},
                   **{"replaced" if dupes else "inserted": 1, "assigned": 1 if joined_role else 0})
        except Exception as exc:  # one profile must not sink the batch
            logger.exception("linkedin import failed for %s", url)
            if defer_missing and not dupes:
                settle(key, {"status": "error", "error": str(exc)}, deferred=True,
                       error=f"{url}: LinkedIn enrichment failed ({exc}); loaded from the sheet values instead"[:500])
            else:
                settle(key, {"status": "error", "error": str(exc)}, error=f"{url}: {exc}"[:500], skipped=1)

    def scrape_chunk(chunk: Sequence[str]) -> Dict[str, Any]:
        """One actor run; its profiles start loading the moment they are back."""
        scraped = scrape_profiles(list(chunk))
        with lock:
            for key, row in _row_index(scraped["rows"]).items():
                if key in url_of and key not in jobs:
                    jobs[key] = workers.submit(load_profile, key, row)
        return scraped

    def chunk_done(chunk: Sequence[str], scraped: Optional[Dict[str, Any]], failure: Optional[str]) -> None:
        """Give every URL of a finished run that did not come back its outcome."""
        for url in chunk:
            key = normalize_linkedin(url)
            with lock:
                loading = key in jobs
            if loading:
                continue
            if failure is not None:
                settle(key, {"status": "not_scraped"}, skipped=1,
                       error=f"{url}: not scraped, Apify failed ({failure})"[:500])
            elif defer_missing:
                settle(key, {"status": "not_found"}, deferred=True,
                       error=f"{url}: no profile returned by Apify; loaded from the sheet values instead")
            else:
                settle(key, {"status": "not_found"}, skipped=1,
                       error=f"{url}: no profile returned by Apify (private, removed, or blocked)")

    # ---- duplicates: nothing to scrape, but the person is still wanted in this role
    for key in dict.fromkeys(k for k in keys if k in skip_as_duplicate):
        existing_id = dupes_by_key[key][0]["id"]
        url_of.setdefault(key, next(u for u, k in zip(urls, keys) if k == key))
        settle(key, {"status": "duplicate", "candidate_id": existing_id}, skipped=1,
               assigned=1 if assign_role(role_id, existing_id) else 0,
               error=None if quiet_duplicates else
               f"{url_of[key]}: already in your pool as candidate {existing_id} (skipped, not scraped)")

    # ---- scraping: one actor run per chunk (the actor caps profiles per request)
    apify_failed: List[str] = []
    apify_requested = 0
    size = _batch_size()
    chunks = [to_scrape[i:i + size] for i in range(0, len(to_scrape), size)]
    if chunks:
        report(status="scraping")
        # The first run goes alone: if Apify is down or out of quota the import stops
        # here, before any other run is paid for.
        try:
            first = scrape_chunk(chunks[0])
        except ApifyScrapeError as exc:
            workers.shutdown(wait=False)
            report(status="failed", completed_at=datetime.now(timezone.utc), error_message=f"apify: {exc}"[:8000])
            return {"status": "failed", "error": str(exc), "inserted": 0, "updated": 0,
                    "skipped": len(urls), "processed": 0, "role_assigned_count": 0,
                    "errors": [f"apify: {exc}"], "results": [], "deferred": []}
        report(status="enriching")
        apify_failed += first["failed"]
        apify_requested += first["requested"]
        timing["scrape"] = time.monotonic() - started
        chunk_done(chunks[0], first, None)
        scraper = ThreadPoolExecutor(max_workers=_scrape_workers(), thread_name_prefix="linkedin-scrape")
        later = [(chunk, scraper.submit(scrape_chunk, chunk)) for chunk in chunks[1:]]
        scraper.shutdown(wait=False)
        for index, (chunk, job) in enumerate(later, start=2):
            # A failed run costs only its own profiles: what the other runs returned
            # (and were billed for) is still loaded.
            try:
                scraped = job.result()
            except ApifyScrapeError as exc:
                logger.exception("apify chunk %s of %s failed", index, len(chunks))
                chunk_done(chunk, None, str(exc))
                continue
            apify_failed += scraped["failed"]
            apify_requested += scraped["requested"]
            timing["scrape"] = time.monotonic() - started
            chunk_done(chunk, scraped, None)
    else:
        report(status="enriching")

    # ---- wait for every profile that is still mapping or saving
    workers.shutdown(wait=True)

    errors = error_lines()
    ordered = list(dict.fromkeys(keys))
    results = [outcomes[k]["result"] for k in ordered if k in outcomes]
    deferred = [outcomes[k]["result"]["url"] for k in ordered if k in outcomes and outcomes[k]["deferred"]]
    status = "completed_with_errors" if errors else "completed"
    final = dict(processed_count=counts["processed"], inserted_count=counts["inserted"],
                 updated_count=counts["replaced"], skipped_count=counts["skipped"],
                 role_assigned_count=counts["assigned"], error_message="\n".join(errors)[:8000] or None)
    if finalize:
        final.update(status=status, completed_at=datetime.now(timezone.utc))
    report(**final)
    cache_started = time.monotonic()
    if touched:
        refresh_caches(touched)
    logger.info("linkedin import upload_id=%s inserted=%s replaced=%s skipped=%s deferred=%s", upload_id,
                counts["inserted"], counts["replaced"], counts["skipped"], len(deferred))
    logger.info("linkedin import timing upload_id=%s profiles=%s total=%.1fs scrape=%.1fs save=%.1fs (summed "
                "over %s at a time) cache=%.1fs", upload_id, len(urls), time.monotonic() - started,
                timing["scrape"], timing["save"], _save_workers(), time.monotonic() - cache_started)
    return {"status": status, "inserted": counts["inserted"], "updated": counts["replaced"],
            "skipped": counts["skipped"], "processed": counts["processed"],
            "role_assigned_count": counts["assigned"], "errors": errors, "results": results,
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

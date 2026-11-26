"""Import candidates from a Google Sheet tab.

After the user confirms the column mapping (same screen as a spreadsheet upload),
each row goes one of two ways:

* it has a LinkedIn profile URL  -> ``linkedin_import.run_linkedin_import`` scrapes
  it through Apify and loads the full enriched profile, with the sheet's contact
  details, notes and custom columns laid on top;
* it has no profile URL, or Apify returned nothing for it -> the existing
  spreadsheet row import (``candidate_imports._process_upload_rows``).

Both phases report into one ``candidate_uploads`` row, so the usual progress
polling works. A saved ``sheet_connections`` row keeps the sheet linked: new rows
are imported on their own by ``sheet_auto_sync`` (and "Sync now" re-reads the whole
tab). People already in the pool are skipped before scraping, so a re-sync only
pays for new rows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from backend.api.routes import candidate_imports as ci
from backend.services import google_oauth, linkedin_import
from backend.services.candidate_pool import REQUIRED_IMPORT_TARGETS
from backend.services.linkedin_normalize import normalize_linkedin

logger = logging.getLogger(__name__)

SOURCE = "google_sheet"

_threads: Dict[int, threading.Thread] = {}
_threads_lock = threading.Lock()


class SheetImportError(ValueError):
    """Bad request for a sheet import; the message is safe to show the user."""


# ------------------------------------------------------------------------ mapping
def validate_sheet_mapping(mapping: Any, *, enrich_linkedin: bool) -> Dict[str, str]:
    """Clean the mapping. With LinkedIn enrichment on, the profile supplies name,
    city and title, so only the LinkedIn column is required."""
    if not isinstance(mapping, dict):
        raise SheetImportError("mapping must be an object")
    clean: Dict[str, str] = {}
    for src, tgt in mapping.items():
        target = str(tgt or "ignore")
        clean[str(src)] = target if target in ci.IMPORT_TARGETS else "ignore"
    used = {t for t in clean.values() if t and t != "ignore"}
    required = {"linkedin"} if enrich_linkedin else set(REQUIRED_IMPORT_TARGETS)
    missing = required - used
    if missing:
        raise SheetImportError(f"Missing required field mappings: {sorted(missing)}")
    return clean


def is_profile_url(value: Any) -> bool:
    """True for a LinkedIn profile link. A bare word is not enough: ``normalize_linkedin``
    reads any single token as a vanity slug, which would send cells like "Yes" to Apify."""
    text = str(value or "").strip()
    lowered = text.lower()
    if "linkedin.com/" not in lowered and not lowered.startswith("/in/"):
        return False
    key = normalize_linkedin(text)
    return bool(key) and key.startswith("/in/")


def profile_url(value: Any) -> str:
    """The link to scrape; a stored ``/in/slug`` path is given its host back."""
    text = str(value or "").strip()
    return text if "linkedin.com/" in text.lower() else f"https://www.linkedin.com{normalize_linkedin(text)}"


def linkedin_url_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Per column, how many cells hold a LinkedIn profile URL (columns with none are left out)."""
    counts: Dict[str, int] = {}
    for column in df.columns:
        n = int(sum(1 for value in df[column].tolist() if value is not None and is_profile_url(value)))
        if n:
            counts[str(column)] = n
    return counts


def split_rows(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
    """Sort row positions into LinkedIn-enrichable, sheet-only and repeated-in-sheet."""
    prepared = ci._prepare_import_rows(df, mapping)
    urls: List[str] = []
    position_by_key: Dict[str, int] = {}
    extras: Dict[str, Dict[str, Any]] = {}
    sheet_only: List[int] = []
    repeated: List[int] = []
    for position, row in enumerate(prepared):
        vals = row["vals"]
        key = row["normalized_li"]
        if not (key and is_profile_url(vals.get("linkedin"))):
            sheet_only.append(position)
            continue
        url = profile_url(vals.get("linkedin"))
        if key in position_by_key:
            repeated.append(position)
            continue
        position_by_key[key] = position
        urls.append(url)
        extras[key] = {
            "email": vals.get("email"),
            "phone": vals.get("phone"),
            "notes": vals.get("notes"),
            "raw": row["raw_vals"],
        }
    return {"urls": urls, "position_by_key": position_by_key, "extras": extras,
            "sheet_only": sheet_only, "repeated": repeated}


def row_signatures(df: pd.DataFrame, mapping: Dict[str, str]) -> List[Optional[tuple]]:
    """Per row position, ``(key, digest)``; ``None`` for a blank row.

    ``key`` says who the row is: the LinkedIn profile when it has one (so editing the
    row's other cells later does not make it a new person), else the row's content.
    ``digest`` is the whole row's content, used to tell when a row has stopped changing."""
    prepared = ci._prepare_import_rows(df, mapping)
    signatures: List[Optional[tuple]] = []
    for position, row in enumerate(prepared):
        cells = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
                 for v in df.iloc[position].tolist()]
        if not any(cells):
            signatures.append(None)
            continue
        digest = hashlib.sha1(json.dumps(cells, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        key = row["normalized_li"]
        if key and is_profile_url(row["vals"].get("linkedin")):
            signatures.append((f"li:{key}", digest))
        else:
            signatures.append((f"row:{digest}", digest))
    return signatures


def pick_ready_rows(signatures: List[Optional[tuple]], seen: Any, pending: Any, held: Any = None,
                    now: Optional[datetime] = None) -> tuple:
    """Rows to import now, and the rows to remember for the next read.

    A row is imported once it is not yet imported (``seen``) and reads the same as it
    did on the previous read (``pending``), so a row someone is still typing waits. A
    row ``held`` for LinkedIn data waits until its next attempt is due."""
    seen_keys = set(seen or [])
    before = pending if isinstance(pending, dict) else {}
    not_due = held_until(held, now)
    ready: List[int] = []
    waiting: Dict[str, str] = {}
    for position, signature in enumerate(signatures):
        if signature is None:
            continue
        key, digest = signature
        if key in seen_keys or key in waiting:
            continue
        waiting[key] = digest
        if before.get(key) == digest and key not in not_due:
            ready.append(position)
    return ready, waiting


# ------------------------------------------------------- rows waiting for LinkedIn data
# Minutes to wait after each attempt that returned no profile: soon at first, because
# the scrapers often return a moment later what they just missed. After the last one
# the person is loaded from the sheet values, so a private or removed profile is not lost.
HOLD_RETRY_MINUTES = (2, 5, 15, 30, 60, 120, 240)


def _held(held: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(held, str):
        try:
            held = json.loads(held)
        except ValueError:
            return {}
    return {str(k): v for k, v in held.items() if isinstance(v, dict)} if isinstance(held, dict) else {}


def held_until(held: Any, now: Optional[datetime] = None) -> set:
    """Keys of held rows whose next attempt is still in the future."""
    now = now or datetime.now(timezone.utc)
    waiting = set()
    for key, entry in _held(held).items():
        try:
            due = datetime.fromisoformat(str(entry.get("retry_at")))
        except ValueError:
            continue
        if (due if due.tzinfo else due.replace(tzinfo=timezone.utc)) > now:
            waiting.add(key)
    return waiting


def given_up_keys(held: Any) -> set:
    """Held rows that have used every retry: their next import loads the sheet values."""
    return {key for key, entry in _held(held).items()
            if int(entry.get("attempts") or 0) >= len(HOLD_RETRY_MINUTES)}


def held_after_attempt(held: Any, keys: Sequence[str], now: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
    """The held record after one more attempt returned nothing for ``keys``."""
    now = now or datetime.now(timezone.utc)
    updated = _held(held)
    for key in keys:
        attempts = int((updated.get(key) or {}).get("attempts") or 0) + 1
        wait = HOLD_RETRY_MINUTES[min(attempts, len(HOLD_RETRY_MINUTES)) - 1]
        updated[key] = {"attempts": attempts, "retry_at": (now + timedelta(minutes=wait)).isoformat()}
    return updated


# ------------------------------------------------------------------- a single name column
_FULL_NAME_HEADERS = {"name", "fullname", "yourname", "yourfullname", "candidatename", "applicantname"}
_FIRST_NAME_COLUMN, _LAST_NAME_COLUMN = "__first_name__", "__last_name__"


def with_full_name(df: pd.DataFrame, mapping: Dict[str, str]) -> tuple:
    """``(df, mapping)`` for rows loaded from their sheet values. The import takes a
    first and a last name; a sheet with one "Name" column gets both derived from it,
    unless the user mapped a name column themselves."""
    if df.empty or {"first_name", "last_name"} & set(mapping.values()):
        return df, mapping
    source = next((c for c in df.columns if mapping.get(str(c), "ignore") in {"custom", "ignore"}
                   and "".join(ch for ch in str(c).lower() if ch.isalpha()) in _FULL_NAME_HEADERS), None)
    if source is None:
        return df, mapping
    parts = [("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)).strip().split(None, 1)
             for v in df[source].tolist()]
    named = df.copy()
    named[_FIRST_NAME_COLUMN] = pd.Series([p[0] if p else None for p in parts], index=df.index, dtype=object)
    named[_LAST_NAME_COLUMN] = pd.Series([p[1] if len(p) > 1 else None for p in parts], index=df.index,
                                         dtype=object)
    return named, {**mapping, _FIRST_NAME_COLUMN: "first_name", _LAST_NAME_COLUMN: "last_name"}


# ------------------------------------------------------------------------ the run
class UploadLedger:
    """Running totals for one upload whose phases work at the same time.

    Each phase (LinkedIn rows, sheet-only rows, fallback rows) reports its own counts
    here; the upload row always shows their sum. Only ``finish`` closes the upload, so
    a phase that ends early cannot mark the whole import as done.
    """

    COUNTS = ("processed_count", "inserted_count", "updated_count", "skipped_count", "role_assigned_count")
    LIVE_STATUSES = ("scraping", "enriching")

    def __init__(self, upload_id: int):
        self.upload_id = upload_id
        self._lock = threading.Lock()
        self._phases: Dict[str, Dict[str, Any]] = {}
        self._status: Optional[str] = None

    def _phase(self, name: str) -> Dict[str, Any]:
        return self._phases.setdefault(name, {**{c: 0 for c in self.COUNTS}, "errors": [], "failed": False})

    def _totals(self) -> Dict[str, Any]:
        totals: Dict[str, Any] = {c: sum(p[c] for p in self._phases.values()) for c in self.COUNTS}
        totals["error_message"] = "\n".join(self.errors())[:8000] or None
        return totals

    def errors(self) -> List[str]:
        return [line for phase in self._phases.values() for line in phase["errors"]]

    def failed(self, name: str) -> bool:
        return bool(self._phases.get(name, {}).get("failed"))

    def sink(self, name: str, *, shows_status: bool = False):
        """A progress writer for one phase. ``shows_status`` lets that phase drive the
        stage the user sees (scraping / enriching) while the import runs."""
        def write(**updates: Any) -> None:
            with self._lock:      # held across the write so sums reach the row in order
                phase = self._phase(name)
                for count in self.COUNTS:
                    if count in updates:
                        phase[count] = int(updates[count] or 0)
                if "error_message" in updates:
                    phase["errors"] = [line for line in str(updates["error_message"] or "").splitlines() if line]
                status = updates.get("status")
                if status == "failed":
                    phase["failed"] = True
                row = self._totals()
                if shows_status and status in self.LIVE_STATUSES and status != self._status:
                    self._status = row["status"] = status
                linkedin_import.write_progress(self.upload_id, **row)
        return write

    def add(self, name: str, *, errors: Optional[List[str]] = None, **counts: int) -> None:
        with self._lock:
            phase = self._phase(name)
            for count, amount in counts.items():
                phase[count] += int(amount)
            phase["errors"] += list(errors or [])

    def finish(self) -> str:
        """Write the final totals and close the upload. Returns the final status."""
        with self._lock:
            row = self._totals()
            if any(p["failed"] for p in self._phases.values()):
                status = "failed"
            else:
                status = "completed_with_errors" if row["error_message"] else "completed"
            linkedin_import.write_progress(self.upload_id, **row, status=status,
                                           completed_at=datetime.now(timezone.utc))
        return status


def run_sheet_import(*, upload_id: int, df: pd.DataFrame, mapping: Dict[str, str], owner_user_id: int,
                     user_role: str, role_id: Optional[int], enrich_linkedin: bool = True,
                     enrichment_mode: str = "none", hold_missing: bool = False,
                     load_anyway: Sequence[str] = ()) -> Dict[str, Any]:
    """Run one tab. Rows with a LinkedIn profile are scraped and enriched; rows without
    one load from their sheet values at the same time; rows Apify could not return fall
    back to their sheet values at the end. Returns the LinkedIn result plus the split.

    With ``hold_missing`` the rows Apify could not return are not loaded: their keys
    come back in ``held`` for the caller to try again later, except the ``load_anyway``
    keys, which have waited long enough and fall back to their sheet values."""
    plain = dict(upload_id=upload_id, owner_user_id=owner_user_id, user_role=user_role,
                 role_id=role_id, enrichment_mode=enrichment_mode)

    def load_rows(rows: pd.DataFrame, **more: Any) -> None:
        named, named_mapping = with_full_name(rows, mapping)
        ci._process_upload_rows(df=named, mapping=named_mapping, **plain, **more)

    if not enrich_linkedin:
        load_rows(df)
        return {"linkedin_rows": 0, "sheet_rows": int(len(df.index)), "repeated_rows": 0}

    split = split_rows(df, mapping)
    if not split["urls"]:      # nothing to enrich: it is a plain spreadsheet import
        load_rows(df.iloc[sorted(split["sheet_only"])])
        return {"inserted": 0, "updated": 0, "skipped": 0, "processed": 0, "role_assigned_count": 0,
                "errors": [], "deferred": [], "status": "completed", "linkedin_rows": 0,
                "sheet_rows": len(split["sheet_only"]), "repeated_rows": 0}

    ledger = UploadLedger(upload_id)
    linkedin_sink = ledger.sink("linkedin", shows_status=True)
    sheet_thread: Optional[threading.Thread] = None

    def load_sheet_only() -> None:
        try:
            load_rows(df.iloc[sorted(split["sheet_only"])], progress_sink=ledger.sink("sheet"))
        except Exception as exc:
            logger.exception("sheet-only rows crashed upload_id=%s", upload_id)
            ledger.add("sheet", errors=[f"sheet rows: {exc}"[:500]])

    def report(**updates: Any) -> None:
        # The sheet-only rows start once Apify has answered (status moves to enriching):
        # if Apify is down the import fails as a whole and nothing is half-loaded.
        nonlocal sheet_thread
        linkedin_sink(**updates)
        if updates.get("status") == "enriching" and sheet_thread is None and split["sheet_only"]:
            sheet_thread = threading.Thread(target=load_sheet_only, daemon=True,
                                            name=f"sheet-import-{upload_id}-rows")
            sheet_thread.start()

    phase1 = linkedin_import.run_linkedin_import(
        upload_id=upload_id, urls=split["urls"], owner_user_id=owner_user_id, role_id=role_id,
        duplicate_policy=linkedin_import.DUPLICATE_SKIP, row_extras=split["extras"],
        defer_missing=True, finalize=False, quiet_duplicates=True, report=report,
    )
    if sheet_thread is not None:
        sheet_thread.join()
    repeated = len(split["repeated"])
    if phase1.get("status") == "failed":
        ledger.finish()
        return {**phase1, "linkedin_rows": len(split["urls"]), "sheet_rows": 0, "repeated_rows": repeated}

    if repeated:
        ledger.add("linkedin", processed_count=repeated, skipped_count=repeated, errors=[
            f"{repeated} row(s) repeat a LinkedIn profile already listed earlier in the sheet (skipped)"])

    # Profiles Apify could not return still get loaded, from what the sheet says about them.
    missing = [k for k in dict.fromkeys(normalize_linkedin(u) for u in phase1.get("deferred") or [])
               if k in split["position_by_key"]]
    # A profile that came back but could not be saved (a dropped database connection, a
    # failed model call) is held too: the retry limit bounds what it can cost.
    held = [f"li:{k}" for k in missing if hold_missing and f"li:{k}" not in set(load_anyway)]
    fallback = sorted(split["position_by_key"][k] for k in missing if f"li:{k}" not in held)
    if held:
        ledger.add("linkedin", errors=[
            f"{len(held)} profile(s) not imported yet: LinkedIn enrichment did not complete, they are retried "
            "automatically"])
    if fallback:
        ledger.add("fallback", errors=[f"{len(fallback)} profile(s) loaded from the sheet values instead"])
        try:
            load_rows(df.iloc[fallback], progress_sink=ledger.sink("fallback"))
        except Exception as exc:
            logger.exception("fallback rows crashed upload_id=%s", upload_id)
            ledger.add("fallback", errors=[f"fallback rows: {exc}"[:500]])
    status = ledger.finish()
    return {**phase1, "status": status, "linkedin_rows": len(split["urls"]),
            "sheet_rows": len(split["sheet_only"]) + len(fallback), "repeated_rows": repeated, "held": held}


def start_sheet_import(*, df: pd.DataFrame, mapping: Dict[str, str], owner_user_id: int, user_role: str,
                       role_id: Optional[int], enrich_linkedin: bool, enrichment_mode: str,
                       spreadsheet_id: str, tab_title: str, sheet_title: str,
                       connection_id: Optional[int] = None) -> Dict[str, Any]:
    """Create the tracking row and run the import on a daemon thread."""
    enrichment_mode = "verified_profile" if ci._is_verified_enrichment_mode(enrichment_mode) else "none"
    row_count = int(len(df.index))
    stored_mapping = {**mapping, "__source__": {"source": SOURCE, "spreadsheet_id": spreadsheet_id,
                                                "tab": tab_title, "enrich_linkedin": bool(enrich_linkedin)}}
    upload_id = ci.create_upload_record(
        owner_user_id=owner_user_id, filename=f"gsheet: {sheet_title or spreadsheet_id} / {tab_title}"[:500],
        headers=[str(c) for c in df.columns], mapping=stored_mapping, row_count=row_count, role_id=role_id,
    )
    row_keys: List[str] = []
    hold_missing = False
    load_anyway: set = set()
    if connection_id is not None:
        mark_synced(connection_id, upload_id)
        row_keys = [s[0] for s in row_signatures(df, mapping) if s]
        # Holding a row only makes sense while something will come back for it.
        hold_missing = bool(enrich_linkedin) and auto_sync_seconds() > 0
        if hold_missing:
            load_anyway = given_up_keys(held_rows(connection_id))

    def settle(failure: Optional[str], held: Sequence[str] = ()) -> None:
        """Tell the connection how its rows fared: imported rows are never read as new
        again; a failed run leaves them unmarked so the automatic sync retries them, and
        rows still waiting for LinkedIn data are held for a later attempt."""
        if connection_id is None:
            return
        try:
            if failure:
                set_sync_error(connection_id, f"Import failed: {failure}", retry_minutes=FAILED_IMPORT_RETRY_MINUTES)
            else:
                waiting = set(held)
                mark_rows_seen(connection_id, [k for k in row_keys if k not in waiting])
                if waiting:
                    hold_rows(connection_id, sorted(waiting))
        except Exception:
            logger.exception("could not record sheet import outcome connection_id=%s", connection_id)

    def run() -> None:
        try:
            result = run_sheet_import(upload_id=upload_id, df=df, mapping=mapping, owner_user_id=owner_user_id,
                                      user_role=user_role, role_id=role_id, enrich_linkedin=enrich_linkedin,
                                      enrichment_mode=enrichment_mode, hold_missing=hold_missing,
                                      load_anyway=sorted(load_anyway))
            failed = (result or {}).get("status") == "failed"
            settle(str((result or {}).get("error") or "the import did not finish") if failed else None,
                   (result or {}).get("held") or ())
        except Exception as exc:
            logger.exception("sheet import crashed upload_id=%s", upload_id)
            settle(str(exc))
            try:
                ci._mark_upload_failed(upload_id, str(exc))
            except Exception:
                logger.exception("could not mark sheet import failed")
        finally:
            with _threads_lock:
                _threads.pop(upload_id, None)

    worker = threading.Thread(target=run, daemon=True, name=f"sheet-import-{upload_id}")
    with _threads_lock:
        _threads[upload_id] = worker
    worker.start()
    return {
        "upload_id": upload_id, "row_count": row_count, "processed_count": 0, "inserted": 0, "updated": 0,
        "skipped": 0, "role_assigned_count": 0, "status": "processing", "enrichment_mode": enrichment_mode,
        "enrich_linkedin": bool(enrich_linkedin), "errors": [], "connection_id": connection_id,
    }


def is_running(upload_id: Any) -> bool:
    """Whether this process is still working on the upload, including recording how its
    rows fared after the upload itself is marked finished."""
    with _threads_lock:
        thread = _threads.get(upload_id)
    return bool(thread and thread.is_alive())


# -------------------------------------------------------------------- connections
_CONNECTION_COLUMNS = ("id, owner_user_id, connected_by_user_id, role_id, spreadsheet_id, tab_title, "
                       "sheet_title, mapping, enrich_linkedin, enrichment_mode, last_synced_at, "
                       "last_upload_id, created_at, last_checked_at, last_error, held_rows")

FAILED_IMPORT_RETRY_MINUTES = 15


def _connection_row(row: Any) -> Dict[str, Any]:
    mapping = row[7]
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except ValueError:
            mapping = {}
    return {
        "id": row[0], "owner_user_id": row[1], "connected_by_user_id": row[2], "role_id": row[3],
        "spreadsheet_id": row[4], "tab_title": row[5], "sheet_title": row[6], "mapping": mapping or {},
        "enrich_linkedin": bool(row[8]), "enrichment_mode": row[9] or "none",
        "last_synced_at": row[10].isoformat() if row[10] else None, "last_upload_id": row[11],
        "created_at": row[12].isoformat() if row[12] else None,
        "sheet_url": f"https://docs.google.com/spreadsheets/d/{row[4]}",
        "last_checked_at": row[13].isoformat() if row[13] else None, "last_error": row[14],
        "auto_sync": auto_sync_seconds() > 0, "held_count": len(_held(row[15])),
    }


def auto_sync_seconds() -> int:
    """How often a sheet that is being edited is read for new rows; 0 turns the
    automatic sync off."""
    try:
        return max(0, int(os.getenv("SHEET_AUTO_SYNC_SECONDS", "60") or 0))
    except ValueError:
        return 60


def auto_sync_idle_seconds() -> int:
    """How often a sheet nobody has touched for a day is read. Never faster than the
    active interval."""
    try:
        idle = int(os.getenv("SHEET_AUTO_SYNC_IDLE_SECONDS", "600") or 600)
    except ValueError:
        idle = 600
    return max(idle, auto_sync_seconds())


def sheet_digest(signatures: List[Optional[tuple]]) -> str:
    """One value for the whole tab's content; it changes when any cell or row does."""
    joined = "\n".join(s[1] if s else "" for s in signatures)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def save_connection(*, owner_user_id: int, connected_by_user_id: int, role_id: Optional[int],
                    spreadsheet_id: str, tab_title: str, sheet_title: str, mapping: Dict[str, str],
                    enrich_linkedin: bool, enrichment_mode: str) -> int:
    """Insert or update the saved connection for this owner + role + sheet tab."""
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute(
            """
            INSERT INTO sheet_connections (owner_user_id, connected_by_user_id, role_id, spreadsheet_id,
                                           tab_title, sheet_title, mapping, enrich_linkedin, enrichment_mode,
                                           last_changed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
            ON CONFLICT (owner_user_id, COALESCE(role_id, 0), spreadsheet_id, tab_title) DO UPDATE
              SET connected_by_user_id = EXCLUDED.connected_by_user_id,
                  sheet_title = EXCLUDED.sheet_title,
                  mapping = EXCLUDED.mapping,
                  enrich_linkedin = EXCLUDED.enrich_linkedin,
                  enrichment_mode = EXCLUDED.enrichment_mode,
                  last_changed_at = NOW()
            RETURNING id
            """,
            (owner_user_id, connected_by_user_id, role_id, spreadsheet_id, tab_title[:500],
             (sheet_title or "")[:500], json.dumps(mapping), bool(enrich_linkedin), enrichment_mode),
        )
        return int(cur.fetchone()[0])

    return google_oauth._with_conn(run)


def mark_synced(connection_id: int, upload_id: int) -> None:
    google_oauth._with_conn(lambda cur: cur.execute(
        "UPDATE sheet_connections SET last_synced_at = NOW(), last_upload_id = %s WHERE id = %s",
        (upload_id, connection_id),
    ))


def mark_rows_seen(connection_id: int, keys: List[str]) -> None:
    """Add imported rows to the connection's record and clear any earlier error."""
    google_oauth._with_conn(lambda cur: cur.execute(
        """
        UPDATE sheet_connections
           SET seen_row_keys = COALESCE((
                   SELECT jsonb_agg(DISTINCT k)
                     FROM jsonb_array_elements_text(COALESCE(seen_row_keys, '[]'::jsonb) || %s::jsonb) AS k
               ), '[]'::jsonb),
               held_rows = COALESCE(held_rows, '{}'::jsonb) - %s::text[],
               last_error = NULL, retry_after = NULL
         WHERE id = %s
        """,
        (json.dumps(sorted(set(keys))), sorted(set(keys)), connection_id),
    ))


def held_rows(connection_id: int) -> Dict[str, Dict[str, Any]]:
    """The connection's rows still waiting for LinkedIn data."""
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute("SELECT held_rows FROM sheet_connections WHERE id = %s", (connection_id,))
        row = cur.fetchone()
        return _held(row[0]) if row else {}

    return google_oauth._with_conn(run)


def hold_rows(connection_id: int, keys: Sequence[str]) -> None:
    """Count one more attempt that returned no profile for these rows."""
    def run(cur):
        cur.execute("SELECT held_rows FROM sheet_connections WHERE id = %s FOR UPDATE", (connection_id,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE sheet_connections SET held_rows = %s::jsonb WHERE id = %s",
                        (json.dumps(held_after_attempt(row[0], keys)), connection_id))

    google_oauth._with_conn(run)


def set_sync_error(connection_id: int, message: str, *, retry_minutes: int = 0) -> None:
    """Show the user why the sheet is not syncing, and hold the next attempt back."""
    google_oauth._with_conn(lambda cur: cur.execute(
        "UPDATE sheet_connections SET last_error = %s, "
        "retry_after = CASE WHEN %s > 0 THEN NOW() + make_interval(mins => %s) ELSE NULL END WHERE id = %s",
        (str(message)[:1000], retry_minutes, retry_minutes, connection_id),
    ))


def list_connections(*, user_id: int, role_id: Optional[int] = None, pool_only: bool = False) -> List[Dict[str, Any]]:
    """Connections the user created or owns; narrowed to one role, or to pool-level ones."""
    google_oauth.ensure_tables()

    def run(cur):
        where = "(owner_user_id = %s OR connected_by_user_id = %s)"
        params: List[Any] = [user_id, user_id]
        if role_id is not None:
            where += " AND role_id = %s"
            params.append(role_id)
        elif pool_only:
            where += " AND role_id IS NULL"
        cur.execute(f"SELECT {_CONNECTION_COLUMNS} FROM sheet_connections WHERE {where} "
                    "ORDER BY COALESCE(last_synced_at, created_at) DESC LIMIT 50", params)
        return [_connection_row(r) for r in cur.fetchall()]

    return google_oauth._with_conn(run)


RUNNING_UPLOAD_MINUTES = 30      # an unfinished upload with no progress for longer is dead


def sync_status(*, user_id: int) -> List[Dict[str, Any]]:
    """What the automatic sync is doing right now for each of the user's linked sheets:
    an import in flight, new rows about to be imported, rows waiting for LinkedIn data.
    The app polls this to show progress for imports nobody started by hand."""
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute(
            """
            SELECT c.id, c.role_id, r.name, c.sheet_title, c.tab_title, c.pending_rows, c.held_rows,
                   c.last_checked_at, c.last_error,
                   CASE WHEN u.status NOT IN ('completed', 'completed_with_errors', 'failed')
                             AND u.updated_at > NOW() - make_interval(mins => %s) THEN u.id END,
                   c.seen_row_keys
              FROM sheet_connections c
              LEFT JOIN recruitment_roles r ON r.id = c.role_id
              LEFT JOIN candidate_uploads u ON u.id = c.last_upload_id
             WHERE c.owner_user_id = %s OR c.connected_by_user_id = %s
             ORDER BY c.id
             LIMIT 50
            """,
            (RUNNING_UPLOAD_MINUTES, user_id, user_id))
        return cur.fetchall()

    status: List[Dict[str, Any]] = []
    for row in google_oauth._with_conn(run):
        pending = row[5] if isinstance(row[5], dict) else {}
        if isinstance(row[5], str):
            try:
                pending = json.loads(row[5]) or {}
            except ValueError:
                pending = {}
        held = _held(row[6])
        seen = row[10]
        if isinstance(seen, str):
            try:
                seen = json.loads(seen)
            except ValueError:
                seen = []
        # A row stays in ``pending`` until the next read, so one that has just been
        # imported (seen) or held is not new any more.
        done = set(seen or []) | set(held)
        retries = sorted(str(entry.get("retry_at")) for entry in held.values() if entry.get("retry_at"))
        status.append({
            "id": row[0], "role_id": row[1], "role_name": row[2] or "", "sheet_title": row[3] or "Google Sheet",
            "tab_title": row[4], "new_count": len([k for k in pending if k not in done]),
            "held_count": len(held), "next_retry_at": retries[0] if retries else None,
            "last_checked_at": f"{row[7].isoformat()}+00:00" if row[7] else None, "last_error": row[8],
            "running_upload_id": row[9],
        })
    return status


def get_connection(connection_id: int, *, user_id: int) -> Optional[Dict[str, Any]]:
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute(f"SELECT {_CONNECTION_COLUMNS} FROM sheet_connections WHERE id = %s "
                    "AND (owner_user_id = %s OR connected_by_user_id = %s)", (connection_id, user_id, user_id))
        row = cur.fetchone()
        return _connection_row(row) if row else None

    return google_oauth._with_conn(run)


def delete_connection(connection_id: int, *, user_id: int) -> bool:
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute("DELETE FROM sheet_connections WHERE id = %s AND (owner_user_id = %s OR connected_by_user_id = %s)",
                    (connection_id, user_id, user_id))
        return cur.rowcount > 0

    return google_oauth._with_conn(run)

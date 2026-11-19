"""Import candidates from a Google Sheet tab.

After the user confirms the column mapping (same screen as a spreadsheet upload),
each row goes one of two ways:

* it has a LinkedIn profile URL  -> ``linkedin_import.run_linkedin_import`` scrapes
  it through Apify and loads the full enriched profile, with the sheet's contact
  details, notes and custom columns laid on top;
* it has no profile URL, or Apify returned nothing for it -> the existing
  spreadsheet row import (``candidate_imports._process_upload_rows``).

Both phases report into one ``candidate_uploads`` row, so the usual progress
polling works. A saved ``sheet_connections`` row lets the user re-run it
("Sync now"); people already in the pool are skipped before scraping, so a
re-sync only pays for new rows.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


# ------------------------------------------------------------------------ the run
def run_sheet_import(*, upload_id: int, df: pd.DataFrame, mapping: Dict[str, str], owner_user_id: int,
                     user_role: str, role_id: Optional[int], enrich_linkedin: bool = True,
                     enrichment_mode: str = "none") -> Dict[str, Any]:
    """Run both phases for one tab. Returns the phase-1 result plus the phase split."""
    if not enrich_linkedin:
        ci._process_upload_rows(upload_id=upload_id, df=df, mapping=mapping, owner_user_id=owner_user_id,
                                user_role=user_role, role_id=role_id, enrichment_mode=enrichment_mode)
        return {"linkedin_rows": 0, "sheet_rows": int(len(df.index)), "repeated_rows": 0}

    split = split_rows(df, mapping)
    phase1: Dict[str, Any] = {"inserted": 0, "updated": 0, "skipped": 0, "processed": 0,
                              "role_assigned_count": 0, "errors": [], "deferred": [], "status": "completed"}
    if split["urls"]:
        phase1 = linkedin_import.run_linkedin_import(
            upload_id=upload_id, urls=split["urls"], owner_user_id=owner_user_id, role_id=role_id,
            duplicate_policy=linkedin_import.DUPLICATE_SKIP, row_extras=split["extras"],
            defer_missing=True, finalize=False, quiet_duplicates=True,
        )
        if phase1.get("status") == "failed":
            return {**phase1, "linkedin_rows": len(split["urls"]), "sheet_rows": 0,
                    "repeated_rows": len(split["repeated"])}

    deferred_positions = [split["position_by_key"][normalize_linkedin(u)] for u in phase1.get("deferred") or []
                          if normalize_linkedin(u) in split["position_by_key"]]
    phase2_positions = sorted(set(split["sheet_only"]) | set(deferred_positions))
    repeated = len(split["repeated"])
    errors = list(phase1.get("errors") or [])
    if repeated:
        errors.append(f"{repeated} row(s) repeat a LinkedIn profile already listed earlier in the sheet (skipped)")
    base = {
        "inserted": phase1["inserted"],
        "updated": phase1["updated"],
        "skipped": phase1["skipped"] + repeated,
        "role_assigned": phase1["role_assigned_count"],
        "processed": phase1["processed"] + repeated,
    }

    if phase2_positions:
        ci._process_upload_rows(
            upload_id=upload_id, df=df.iloc[phase2_positions], mapping=mapping, owner_user_id=owner_user_id,
            user_role=user_role, role_id=role_id, enrichment_mode=enrichment_mode,
            base_counts=base, base_errors=errors,
        )
    else:
        linkedin_import.write_progress(
            upload_id, processed_count=base["processed"], skipped_count=base["skipped"],
            status="completed_with_errors" if errors else "completed",
            completed_at=datetime.now(timezone.utc), error_message="\n".join(errors)[:8000] or None,
        )
    return {**phase1, "linkedin_rows": len(split["urls"]), "sheet_rows": len(phase2_positions),
            "repeated_rows": repeated}


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
    if connection_id is not None:
        mark_synced(connection_id, upload_id)

    def run() -> None:
        try:
            run_sheet_import(upload_id=upload_id, df=df, mapping=mapping, owner_user_id=owner_user_id,
                             user_role=user_role, role_id=role_id, enrich_linkedin=enrich_linkedin,
                             enrichment_mode=enrichment_mode)
        except Exception as exc:
            logger.exception("sheet import crashed upload_id=%s", upload_id)
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


# -------------------------------------------------------------------- connections
_CONNECTION_COLUMNS = ("id, owner_user_id, connected_by_user_id, role_id, spreadsheet_id, tab_title, "
                       "sheet_title, mapping, enrich_linkedin, enrichment_mode, last_synced_at, "
                       "last_upload_id, created_at")


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
    }


def save_connection(*, owner_user_id: int, connected_by_user_id: int, role_id: Optional[int],
                    spreadsheet_id: str, tab_title: str, sheet_title: str, mapping: Dict[str, str],
                    enrich_linkedin: bool, enrichment_mode: str) -> int:
    """Insert or update the saved connection for this owner + role + sheet tab."""
    google_oauth.ensure_tables()

    def run(cur):
        cur.execute(
            """
            INSERT INTO sheet_connections (owner_user_id, connected_by_user_id, role_id, spreadsheet_id,
                                           tab_title, sheet_title, mapping, enrich_linkedin, enrichment_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (owner_user_id, COALESCE(role_id, 0), spreadsheet_id, tab_title) DO UPDATE
              SET connected_by_user_id = EXCLUDED.connected_by_user_id,
                  sheet_title = EXCLUDED.sheet_title,
                  mapping = EXCLUDED.mapping,
                  enrich_linkedin = EXCLUDED.enrich_linkedin,
                  enrichment_mode = EXCLUDED.enrichment_mode
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

"""Linked Google Sheets import their new rows on their own.

Every ``SHEET_AUTO_SYNC_SECONDS`` each saved ``sheet_connections`` row is read again.
Rows not imported before, and unchanged since the previous read, go through the same
import as the first one (LinkedIn rows scraped and enriched, the rest loaded from the
sheet) into the connection's role or pool. Nothing is imported twice: the connection
remembers every row it has imported, until the row is removed from the sheet.

A LinkedIn row whose profile was not returned is not imported bare: it is held and
tried again (``sheet_import.HOLD_RETRY_MINUTES``), and loaded from its sheet values
only once those attempts are used up.

A sheet whose content has not changed for ``QUIET_AFTER_HOURS`` is read only every
``SHEET_AUTO_SYNC_IDLE_SECONDS``; the first change puts it back on the fast interval.
Google access tokens are reused until shortly before they expire.

Every web worker (and any laptop on the same database) runs this loop, so a
connection is claimed in one ``UPDATE ... SKIP LOCKED`` and only the claimer reads it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from backend.services import google_oauth, google_sheet_source as sheets, sheet_import
from backend.services.google_oauth import GoogleAuthError
from backend.services.google_sheet_source import SheetSourceError

logger = logging.getLogger(__name__)

START_DELAY_SECONDS = 20
CLAIMS_PER_CYCLE = 5
STALE_IMPORT_MINUTES = 30      # an import with no progress for this long died with its process
# How long to leave a sheet alone after a problem only the user can fix.
BLOCKED_RETRY_MINUTES = 30
QUIET_AFTER_HOURS = 24         # unchanged this long: read on the idle interval instead
TOKEN_REUSE_SECONDS = 45 * 60  # Google access tokens last an hour

_thread: Optional[threading.Thread] = None
_tokens: Dict[int, tuple] = {}     # user id -> (token stamp, access token, use until)


def access_token(user_id: int, stamp: Any) -> str:
    """The user's Google access token, reused while it is fresh. ``stamp`` is when the
    stored connection was last saved (``None`` once it is removed), so a disconnect or
    reconnect made in any process stops the old token being used here."""
    held = _tokens.get(user_id)
    if stamp is not None and held and held[0] == stamp and held[2] > time.monotonic():
        return held[1]
    _tokens.pop(user_id, None)
    token = google_oauth.access_token_for_user(user_id)
    if stamp is not None:
        _tokens[user_id] = (stamp, token, time.monotonic() + TOKEN_REUSE_SECONDS)
    return token


def claim_due_connections(interval_seconds: int, idle_seconds: Optional[int] = None) -> List[Dict[str, Any]]:
    """Take the connections due for a read; no other process gets the same ones."""
    idle_seconds = interval_seconds if idle_seconds is None else idle_seconds

    def run(cur):
        cur.execute(
            """
            UPDATE sheet_connections c
               SET last_checked_at = NOW()
             WHERE c.id IN (
                   SELECT id FROM sheet_connections
                    WHERE (last_checked_at IS NULL OR last_checked_at < NOW() - make_interval(secs =>
                               CASE WHEN COALESCE(last_changed_at, created_at) > NOW() - make_interval(hours => %s)
                                    THEN %s ELSE %s END))
                      AND (retry_after IS NULL OR retry_after < NOW())
                    ORDER BY last_checked_at NULLS FIRST
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED)
            RETURNING c.id, c.owner_user_id, c.connected_by_user_id, c.role_id, c.spreadsheet_id, c.tab_title,
                      c.sheet_title, c.mapping, c.enrich_linkedin, c.enrichment_mode, c.last_upload_id,
                      c.seen_row_keys, c.pending_rows, c.held_rows
            """,
            (QUIET_AFTER_HOURS, max(1, int(interval_seconds * 0.8)), max(1, int(idle_seconds * 0.8)),
             CLAIMS_PER_CYCLE),
        )
        names = ("id", "owner_user_id", "connected_by_user_id", "role_id", "spreadsheet_id", "tab_title",
                 "sheet_title", "mapping", "enrich_linkedin", "enrichment_mode", "last_upload_id",
                 "seen_row_keys", "pending_rows", "held_rows")
        return [dict(zip(names, row)) for row in cur.fetchall()]

    return google_oauth._with_conn(run)


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return default
    return default if value is None else value


def _context(connection: Dict[str, Any]) -> Dict[str, Any]:
    """What the database knows that decides whether this sheet can be imported now."""

    def run(cur):
        cur.execute(
            "SELECT updated_at > NOW() - make_interval(mins => %s) FROM candidate_uploads WHERE id = %s "
            "AND status NOT IN ('completed', 'completed_with_errors', 'failed')",
            (STALE_IMPORT_MINUTES, connection["last_upload_id"]))
        unfinished = cur.fetchone()
        importing = bool(unfinished and unfinished[0])
        # Unfinished with no progress for that long: its process died mid-import.
        stale_upload_id = connection["last_upload_id"] if unfinished and not unfinished[0] else None
        role_exists = True
        if connection["role_id"] is not None:
            cur.execute("SELECT 1 FROM recruitment_roles WHERE id = %s", (connection["role_id"],))
            role_exists = cur.fetchone() is not None
        reader = connection["connected_by_user_id"] or connection["owner_user_id"]
        cur.execute("SELECT role FROM users WHERE id = %s", (reader,))
        row = cur.fetchone()
        cur.execute("SELECT updated_at FROM user_google_tokens WHERE user_id = %s", (reader,))
        token_row = cur.fetchone()
        return {"importing": importing, "stale_upload_id": stale_upload_id, "role_exists": role_exists,
                "reader_user_id": int(reader),
                "user_role": str((row[0] if row else "") or "").strip().lower(),
                "token_stamp": token_row[0] if token_row else None}

    return google_oauth._with_conn(run)


def _save_read(connection_id: int, pending: Dict[str, str], digest: str, forget: Optional[List[str]] = None) -> None:
    """Record a successful read: the rows still waiting, whether the sheet changed, and
    the rows that are no longer in the sheet (``forget``), so adding one back imports it."""
    forget = sorted(set(forget or []))
    google_oauth._with_conn(lambda cur: cur.execute(
        "UPDATE sheet_connections SET pending_rows = %s::jsonb, last_error = NULL, retry_after = NULL, "
        "last_changed_at = CASE WHEN content_digest IS DISTINCT FROM %s THEN NOW() ELSE last_changed_at END, "
        "content_digest = %s, "
        "seen_row_keys = COALESCE((SELECT jsonb_agg(k) FROM jsonb_array_elements_text("
        "COALESCE(seen_row_keys, '[]'::jsonb)) AS k WHERE NOT (k = ANY(%s::text[]))), '[]'::jsonb), "
        "held_rows = COALESCE(held_rows, '{}'::jsonb) - %s::text[] WHERE id = %s",
        (json.dumps(pending), digest, digest, forget, forget, connection_id)))


def _read_tab(connection: Dict[str, Any], context: Dict[str, Any]):
    """The tab as a frame. A held token Google no longer accepts is replaced once."""
    reader, stamp = context["reader_user_id"], context.get("token_stamp")
    try:
        return sheets.read_tab(access_token(reader, stamp), connection["spreadsheet_id"], connection["tab_title"])
    except SheetSourceError as exc:
        if exc.code != "reconnect" or _tokens.pop(reader, None) is None:
            raise
        return sheets.read_tab(access_token(reader, stamp), connection["spreadsheet_id"], connection["tab_title"])


def check_connection(connection: Dict[str, Any]) -> Optional[int]:
    """Read one linked sheet and import its new rows. Returns how many rows were sent
    to import, or ``None`` when the sheet could not be read this time."""
    connection_id = int(connection["id"])
    context = _context(connection)
    # The previous import is still running, or has finished but not yet recorded which
    # of its rows were imported and which are held; reading now would send them again.
    if context["importing"] or sheet_import.is_running(connection["last_upload_id"]):
        return 0
    if context.get("stale_upload_id"):
        try:
            sheet_import.ci._mark_upload_failed(
                int(context["stale_upload_id"]), "Import interrupted; the sheet sync started it again.")
        except Exception:
            logger.exception("could not close the interrupted import %s", context["stale_upload_id"])
    if not context["role_exists"]:
        sheet_import.set_sync_error(connection_id, "The role this sheet was linked to no longer exists.",
                                    retry_minutes=24 * 60)
        return None
    enrich = bool(connection["enrich_linkedin"])
    try:
        mapping = sheet_import.validate_sheet_mapping(
            {k: v for k, v in _json(connection["mapping"], {}).items() if isinstance(v, str)},
            enrich_linkedin=enrich)
        frame = _read_tab(connection, context)
    except (GoogleAuthError, SheetSourceError, sheet_import.SheetImportError) as exc:
        # An expired connection, lost access or an oversized tab needs the user; a
        # network blip does not, so it is retried on the next cycle.
        blocked = getattr(exc, "code", None) in {"not_configured", "not_connected", "reconnect", "no_access",
                                                 "not_found", "too_large", "api_disabled"}
        blocked = blocked or isinstance(exc, sheet_import.SheetImportError)
        sheet_import.set_sync_error(connection_id, str(exc), retry_minutes=BLOCKED_RETRY_MINUTES if blocked else 0)
        return None
    if enrich and not any(h in frame.columns for h, target in mapping.items() if target == "linkedin"):
        sheet_import.set_sync_error(
            connection_id, "The LinkedIn column was renamed or removed in the sheet. Review the mapping to relink it.",
            retry_minutes=BLOCKED_RETRY_MINUTES)
        return None

    signatures = sheet_import.row_signatures(frame, mapping)
    present = {s[0] for s in signatures if s}
    seen = _json(connection["seen_row_keys"], [])
    held = _json(connection.get("held_rows"), {})
    forget = sorted((set(seen) | set(held)) - present)      # rows removed from the sheet
    ready, pending = sheet_import.pick_ready_rows(
        signatures, [k for k in seen if k in present], _json(connection["pending_rows"], {}), held)
    _save_read(connection_id, pending, sheet_import.sheet_digest(signatures), forget)
    if not ready:
        return 0
    result = sheet_import.start_sheet_import(
        df=frame.iloc[ready], mapping=mapping, owner_user_id=int(connection["owner_user_id"]),
        user_role=context["user_role"], role_id=connection["role_id"], enrich_linkedin=enrich,
        enrichment_mode=connection["enrichment_mode"] or "none", spreadsheet_id=connection["spreadsheet_id"],
        tab_title=connection["tab_title"], sheet_title=connection["sheet_title"] or "", connection_id=connection_id)
    logger.info("sheet auto-sync: connection=%s new_rows=%s upload_id=%s", connection_id, len(ready),
                result.get("upload_id"))
    return len(ready)


def run_cycle() -> int:
    """One pass over the connections that are due. Returns rows sent to import."""
    google_oauth.ensure_tables()
    sent = 0
    for connection in claim_due_connections(sheet_import.auto_sync_seconds(),
                                            sheet_import.auto_sync_idle_seconds()):
        try:
            sent += check_connection(connection) or 0
        except Exception as exc:
            logger.exception("sheet auto-sync failed for connection %s", connection.get("id"))
            try:
                sheet_import.set_sync_error(int(connection["id"]), f"Automatic sync failed: {exc}")
            except Exception:
                logger.exception("could not record the sheet auto-sync error")
    return sent


def _loop() -> None:
    time.sleep(START_DELAY_SECONDS)
    while True:
        seconds = sheet_import.auto_sync_seconds()
        if seconds <= 0:
            return
        try:
            if google_oauth.is_configured():
                run_cycle()
        except Exception:
            logger.exception("sheet auto-sync cycle failed")
        time.sleep(seconds)


def start_poller() -> None:
    """Idempotently start the background sync of linked sheets."""
    global _thread
    seconds = sheet_import.auto_sync_seconds()
    if seconds <= 0:
        logger.info("Sheet auto-sync disabled (SHEET_AUTO_SYNC_SECONDS=0)")
        return
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="sheet-auto-sync")
    _thread.start()
    logger.info("Sheet auto-sync started (every %ss)", seconds)

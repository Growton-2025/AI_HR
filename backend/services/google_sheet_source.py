"""Read one tab of a Google Sheet into the same table shape a CSV/XLSX upload gives.

Uses the Sheets REST API directly with a user's access token (see
``google_oauth``). The frame it returns goes through the existing spreadsheet
mapping and import code unchanged.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import pandas as pd

from backend.services import google_oauth

logger = logging.getLogger(__name__)

SHEETS_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
HTTP_TIMEOUT_SECONDS = 30.0

_ID_RE = re.compile(r"/spreadsheets/(?:u/\d+/)?d/([a-zA-Z0-9_-]{20,})")
_GID_RE = re.compile(r"[#?&]gid=(\d+)")
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{25,}$")


class SheetSourceError(ValueError):
    """``code``: bad_link, no_access, not_found, empty, too_large, reconnect, api_disabled, google_error."""

    def __init__(self, message: str, code: str = "google_error"):
        super().__init__(message)
        self.code = code


def max_rows() -> int:
    try:
        return max(1, int(os.getenv("SHEET_IMPORT_MAX_ROWS", "500") or 500))
    except ValueError:
        return 500


def parse_sheet_url(url: str) -> Tuple[str, Optional[int]]:
    """``(spreadsheet_id, gid)`` from a Google Sheets link (or a bare spreadsheet id)."""
    text = str(url or "").strip()
    if not text:
        raise SheetSourceError("Paste a Google Sheets link.", code="bad_link")
    match = _ID_RE.search(text)
    if match:
        gid = _GID_RE.search(text)
        return match.group(1), int(gid.group(1)) if gid else None
    if _BARE_ID_RE.match(text):
        return text, None
    if "/spreadsheets/d/e/" in text:
        raise SheetSourceError(
            "That is a 'published to web' link. Open the sheet normally and copy the link from the address bar.",
            code="bad_link",
        )
    raise SheetSourceError(
        "That does not look like a Google Sheets link (expected docs.google.com/spreadsheets/d/...).",
        code="bad_link",
    )


def _get(url: str, access_token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        resp = google_oauth.http().get(url, params=params, headers={"Authorization": f"Bearer {access_token}"},
                                       timeout=HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise SheetSourceError(f"Could not reach Google Sheets: {exc}")
    if resp.status_code == 200:
        return resp.json()

    message = ""
    status_text = ""
    try:
        err = resp.json().get("error") or {}
        message = str(err.get("message") or "")
        status_text = str(err.get("status") or "")
    except Exception:
        message = resp.text[:300]
    if resp.status_code == 401:
        raise SheetSourceError("Your Google connection expired. Connect Google again.", code="reconnect")
    if resp.status_code == 403:
        if "has not been used" in message or "is disabled" in message or "SERVICE_DISABLED" in message:
            raise SheetSourceError(
                "The Google Sheets API is not enabled for this app's Google Cloud project.", code="api_disabled"
            )
        if "insufficient" in message.lower() or "scope" in message.lower():
            raise SheetSourceError(
                "Google Sheets access was not granted. Connect Google again and allow spreadsheet access.",
                code="reconnect",
            )
        raise SheetSourceError(
            "The connected Google account cannot open this sheet. Ask the owner to share it with that account.",
            code="no_access",
        )
    if resp.status_code == 404:
        raise SheetSourceError("Sheet not found. Check the link.", code="not_found")
    if resp.status_code == 400 and "not supported for this document" in message:
        raise SheetSourceError(
            "This file is an uploaded Excel file, not a Google Sheet. In Google Sheets use "
            "File > Save as Google Sheets, then use the new link.",
            code="bad_link",
        )
    raise SheetSourceError(f"Google Sheets error ({resp.status_code} {status_text}): {message}"[:400])


def list_tabs(access_token: str, spreadsheet_id: str) -> Dict[str, Any]:
    """``{"title": sheet name, "tabs": [{"gid", "title", "index", "rows"}]}``; hidden tabs are left out."""
    data = _get(
        f"{SHEETS_BASE}/{spreadsheet_id}",
        access_token,
        {"fields": "properties.title,sheets.properties(sheetId,title,index,hidden,sheetType,gridProperties.rowCount)"},
    )
    tabs: List[Dict[str, Any]] = []
    for sheet in data.get("sheets") or []:
        props = sheet.get("properties") or {}
        if props.get("hidden") or props.get("sheetType", "GRID") != "GRID":
            continue
        tabs.append({
            "gid": props.get("sheetId"),
            "title": props.get("title") or "",
            "index": props.get("index") or 0,
            "rows": (props.get("gridProperties") or {}).get("rowCount"),
        })
    tabs.sort(key=lambda t: t["index"])
    return {"title": (data.get("properties") or {}).get("title") or "", "tabs": tabs}


def resolve_tab(tabs: List[Dict[str, Any]], *, tab_title: Optional[str] = None,
                gid: Optional[int] = None) -> Dict[str, Any]:
    """Pick the tab by title, else by the link's gid, else the first one."""
    if not tabs:
        raise SheetSourceError("This sheet has no visible tabs.", code="empty")
    if tab_title:
        for tab in tabs:
            if tab["title"] == tab_title:
                return tab
        raise SheetSourceError(f"Tab '{tab_title}' was not found in this sheet (renamed or deleted).",
                               code="not_found")
    if gid is not None:
        for tab in tabs:
            if tab["gid"] == gid:
                return tab
    return tabs[0]


def dedupe_headers(headers: List[str]) -> List[str]:
    """Name columns the way pandas does for a file: blanks become ``Unnamed: n`` and
    repeats get ``.1``, ``.2``, so the work-history header detection sees the same shape."""
    out: List[str] = []
    seen: Dict[str, int] = {}
    taken = set()
    for index, raw in enumerate(headers):
        base = str(raw or "").strip() or f"Unnamed: {index}"
        name = base
        while name in taken:
            seen[base] = seen.get(base, 0) + 1
            name = f"{base}.{seen[base]}"
        taken.add(name)
        out.append(name)
    return out


def rows_to_frame(values: List[List[Any]]) -> pd.DataFrame:
    """First row is the header. Ragged rows are padded, blank rows and wholly blank
    unnamed columns dropped, and empty cells become ``None`` (as NaN would for a file)."""
    rows = [list(r) for r in (values or [])]
    # the header is the first row that has any text
    while rows and not any(str(c or "").strip() for c in rows[0]):
        rows.pop(0)
    if not rows:
        raise SheetSourceError("This tab is empty.", code="empty")
    width = max(len(r) for r in rows)
    header_cells = rows[0] + [""] * (width - len(rows[0]))
    body = []
    for row in rows[1:]:
        cells = [(str(c).strip() if c is not None else "") for c in row] + [""] * (width - len(row))
        if any(cells):
            body.append([c if c else None for c in cells])
    if not body:
        raise SheetSourceError("This tab has a header row but no data rows.", code="empty")

    # name every column by its position in the sheet first, then drop the empty unnamed ones
    names = dedupe_headers(header_cells)
    keep = [i for i in range(width)
            if str(header_cells[i] or "").strip() or any(r[i] is not None for r in body)]
    return pd.DataFrame([[r[i] for i in keep] for r in body], columns=[names[i] for i in keep], dtype=object)


def read_tab(access_token: str, spreadsheet_id: str, tab_title: str) -> pd.DataFrame:
    """The whole tab as a frame of strings. Raises ``too_large`` over ``SHEET_IMPORT_MAX_ROWS``."""
    escaped = tab_title.replace("'", "''")
    rng = quote(f"'{escaped}'", safe="")
    data = _get(
        f"{SHEETS_BASE}/{spreadsheet_id}/values/{rng}",
        access_token,
        {"valueRenderOption": "FORMATTED_VALUE", "majorDimension": "ROWS"},
    )
    frame = rows_to_frame(data.get("values") or [])
    cap = max_rows()
    if len(frame.index) > cap:
        raise SheetSourceError(
            f"This tab has {len(frame.index)} rows; the limit per import is {cap}. "
            "Split it into smaller tabs, or raise SHEET_IMPORT_MAX_ROWS.",
            code="too_large",
        )
    return frame

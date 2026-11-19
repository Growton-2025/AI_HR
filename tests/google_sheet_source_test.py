"""Google Sheet reader: link parsing, table shaping and Google error mapping (no network)."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from backend.services import google_sheet_source as gs


SHEET_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"


def test_parse_sheet_url_reads_id_and_gid():
    assert gs.parse_sheet_url(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=42") == (SHEET_ID, 42)
    assert gs.parse_sheet_url(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?gid=7&usp=sharing") == (SHEET_ID, 7)
    assert gs.parse_sheet_url(f"https://docs.google.com/spreadsheets/u/1/d/{SHEET_ID}/edit") == (SHEET_ID, None)
    assert gs.parse_sheet_url(SHEET_ID) == (SHEET_ID, None)


@pytest.mark.parametrize("link", ["", "https://example.com/file.csv", "https://docs.google.com/document/d/abc/edit",
                                  "https://docs.google.com/spreadsheets/d/e/2PACX-1vT/pubhtml"])
def test_parse_sheet_url_rejects_other_links(link):
    with pytest.raises(gs.SheetSourceError) as exc:
        gs.parse_sheet_url(link)
    assert exc.value.code == "bad_link"


def test_dedupe_headers_matches_pandas_file_naming():
    assert gs.dedupe_headers(["Title", "Company", "Title", "", "Title", "Title.1"]) == [
        "Title", "Company", "Title.1", "Unnamed: 3", "Title.2", "Title.1.1"]


def test_rows_to_frame_pads_ragged_rows_and_drops_blanks():
    frame = gs.rows_to_frame([
        [],                                  # blank lead-in row before the header
        ["Name", "LinkedIn", "", "Notes"],
        ["Ada", "https://linkedin.com/in/ada"],          # short row
        ["", "", "", ""],                                  # blank row
        ["Alan", " https://linkedin.com/in/alan ", "", "call back", "stray"],  # long row
    ])
    # the unnamed, empty third column is dropped; the unnamed fifth one has data so it stays
    assert list(frame.columns) == ["Name", "LinkedIn", "Notes", "Unnamed: 4"]
    assert frame.to_dict(orient="records") == [
        {"Name": "Ada", "LinkedIn": "https://linkedin.com/in/ada", "Notes": None, "Unnamed: 4": None},
        {"Name": "Alan", "LinkedIn": "https://linkedin.com/in/alan", "Notes": "call back", "Unnamed: 4": "stray"},
    ]


@pytest.mark.parametrize("values", [[], [[""], [" "]], [["Name", "LinkedIn"]], [["Name"], [""], [" "]]])
def test_rows_to_frame_rejects_empty_tabs(values):
    with pytest.raises(gs.SheetSourceError) as exc:
        gs.rows_to_frame(values)
    assert exc.value.code == "empty"


class _Resp:
    def __init__(self, status: int, payload: Dict[str, Any]):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _fake_get(monkeypatch, status: int, payload: Dict[str, Any], seen: Dict[str, Any] | None = None):
    def fake(url, params=None, headers=None, timeout=None):
        if seen is not None:
            seen.update(url=url, params=params, headers=headers)
        return _Resp(status, payload)

    monkeypatch.setattr(gs.httpx, "get", fake)


def test_list_tabs_skips_hidden_and_orders_by_index(monkeypatch):
    _fake_get(monkeypatch, 200, {"properties": {"title": "Leads"}, "sheets": [
        {"properties": {"sheetId": 9, "title": "Second", "index": 1, "gridProperties": {"rowCount": 50}}},
        {"properties": {"sheetId": 3, "title": "Old", "index": 2, "hidden": True}},
        {"properties": {"sheetId": 0, "title": "First", "index": 0, "gridProperties": {"rowCount": 10}}},
    ]})
    meta = gs.list_tabs("tok", SHEET_ID)
    assert meta["title"] == "Leads"
    assert [t["title"] for t in meta["tabs"]] == ["First", "Second"]
    assert gs.resolve_tab(meta["tabs"], gid=9)["title"] == "Second"
    assert gs.resolve_tab(meta["tabs"], gid=12345)["title"] == "First"   # unknown gid -> first tab
    assert gs.resolve_tab(meta["tabs"], tab_title="Second")["gid"] == 9
    with pytest.raises(gs.SheetSourceError) as exc:
        gs.resolve_tab(meta["tabs"], tab_title="Renamed")
    assert exc.value.code == "not_found"


def test_read_tab_quotes_the_tab_name_and_enforces_the_row_cap(monkeypatch):
    seen: Dict[str, Any] = {}
    _fake_get(monkeypatch, 200, {"values": [["LinkedIn"], ["a"], ["b"], ["c"]]}, seen)
    frame = gs.read_tab("tok", SHEET_ID, "Bob's leads")
    assert len(frame.index) == 3
    assert seen["url"].endswith("/values/%27Bob%27%27s%20leads%27")
    assert seen["headers"] == {"Authorization": "Bearer tok"}

    monkeypatch.setenv("SHEET_IMPORT_MAX_ROWS", "2")
    with pytest.raises(gs.SheetSourceError) as exc:
        gs.read_tab("tok", SHEET_ID, "Bob's leads")
    assert exc.value.code == "too_large" and "3 rows" in str(exc.value)


@pytest.mark.parametrize("status,message,code", [
    (401, "Request had invalid authentication credentials.", "reconnect"),
    (403, "The caller does not have permission", "no_access"),
    (403, "Google Sheets API has not been used in project 1 before or it is disabled.", "api_disabled"),
    (403, "Request had insufficient authentication scopes.", "reconnect"),
    (404, "Requested entity was not found.", "not_found"),
    (400, "This operation is not supported for this document", "bad_link"),
    (500, "Internal error", "google_error"),
])
def test_google_errors_map_to_user_facing_codes(monkeypatch, status, message, code):
    _fake_get(monkeypatch, status, {"error": {"message": message, "status": "X"}})
    with pytest.raises(gs.SheetSourceError) as exc:
        gs.list_tabs("tok", SHEET_ID)
    assert exc.value.code == code

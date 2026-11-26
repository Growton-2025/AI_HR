"""Automatic sync of linked Google Sheets (no network, no database)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import pytest

from backend.services import google_oauth, google_sheet_source as sheets, sheet_auto_sync as auto, sheet_import as si
from backend.services.google_oauth import GoogleAuthError

ADA = "https://www.linkedin.com/in/ada-lovelace/"
ALAN = "https://linkedin.com/in/alan-turing"
MAPPING = {"Profile": "linkedin", "Mail": "email", "First": "first_name", "Stage": "custom"}


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Profile", "Mail", "First", "Stage"], dtype=object)


def _keys(frame: pd.DataFrame) -> List[Any]:
    return [s and s[0] for s in si.row_signatures(frame, MAPPING)]


def test_linkedin_row_keeps_its_identity_when_other_cells_change():
    before = si.row_signatures(_frame([{"Profile": ADA, "Stage": "Yet to Contact"}]), MAPPING)[0]
    after = si.row_signatures(_frame([{"Profile": "linkedin.com/in/ada-lovelace", "Stage": "Shortlisted"}]), MAPPING)[0]
    assert before[0] == after[0] == "li:/in/ada-lovelace"
    assert before[1] != after[1]


def test_blank_rows_have_no_signature_and_plain_rows_are_keyed_by_content():
    frame = _frame([{"Profile": None, "Mail": " ", "First": None}, {"First": "Noli", "Mail": "n@x.com"},
                    {"First": "Noli", "Mail": "n@x.com"}, {"First": "Noli", "Mail": "other@x.com"}])
    keys = _keys(frame)
    assert keys[0] is None
    assert keys[1] == keys[2] and keys[1].startswith("row:")
    assert keys[3] != keys[1]


def test_a_new_row_waits_one_read_then_imports_once():
    frame = _frame([{"Profile": ADA}, {"Profile": ALAN}])
    signatures = si.row_signatures(frame, MAPPING)
    ready, pending = si.pick_ready_rows(signatures, ["li:/in/ada-lovelace"], {})
    assert ready == [] and list(pending) == ["li:/in/alan-turing"]
    ready, pending = si.pick_ready_rows(signatures, ["li:/in/ada-lovelace"], pending)
    assert ready == [1]
    ready, _ = si.pick_ready_rows(signatures, ["li:/in/ada-lovelace", "li:/in/alan-turing"], pending)
    assert ready == []


def test_a_row_still_being_typed_keeps_waiting_and_repeats_are_taken_once():
    first = si.row_signatures(_frame([{"Profile": ALAN}]), MAPPING)
    _, pending = si.pick_ready_rows(first, [], {})
    edited = si.row_signatures(_frame([{"Profile": ALAN, "Mail": "alan@x.com"}, {"Profile": ALAN}]), MAPPING)
    ready, pending = si.pick_ready_rows(edited, [], pending)
    assert ready == []
    ready, _ = si.pick_ready_rows(edited, [], pending)
    assert ready == [0]


@pytest.fixture
def world(monkeypatch):
    state: Dict[str, Any] = {"frame": _frame([{"Profile": ADA}]), "started": [], "errors": [], "pending": [],
                             "digests": [], "token_calls": [],
                             "context": {"importing": False, "role_exists": True, "reader_user_id": 7,
                                         "user_role": "recruiter", "token_stamp": "t1"}}
    monkeypatch.setattr(auto, "_context", lambda connection: state["context"])
    state["forgot"] = []
    monkeypatch.setattr(auto, "_save_read", lambda cid, pending, digest, forget=None: (
        state["pending"].append(pending), state["digests"].append(digest), state["forgot"].append(forget or [])))
    auto._tokens.clear()
    monkeypatch.setattr(google_oauth, "access_token_for_user",
                        lambda user_id: state["token_calls"].append(user_id) or f"token{len(state['token_calls'])}")
    state["read"] = lambda token, sid, tab: state["frame"]
    monkeypatch.setattr(sheets, "read_tab", lambda token, sid, tab: state["read"](token, sid, tab))
    monkeypatch.setattr(si, "set_sync_error",
                        lambda cid, message, retry_minutes=0: state["errors"].append((message, retry_minutes)))
    monkeypatch.setattr(si, "start_sheet_import",
                        lambda **kw: state["started"].append(kw) or {"upload_id": 900 + len(state["started"])})
    return state


def _connection(**over: Any) -> Dict[str, Any]:
    return {"id": 5, "owner_user_id": 4, "connected_by_user_id": 7, "role_id": 86, "spreadsheet_id": "sid",
            "tab_title": "Form responses 1", "sheet_title": "Test", "mapping": dict(MAPPING), "enrich_linkedin": True,
            "enrichment_mode": "none", "last_upload_id": 81, "seen_row_keys": ["li:/in/ada-lovelace"],
            "pending_rows": {}, **over}


def test_appended_row_is_imported_alone_on_the_second_read(world):
    world["frame"] = _frame([{"Profile": ADA}, {"Profile": ALAN, "First": "Alan"}])
    assert auto.check_connection(_connection()) == 0
    assert world["started"] == []
    assert auto.check_connection(_connection(pending_rows=world["pending"][-1])) == 1
    sent = world["started"][0]
    assert sent["df"]["Profile"].tolist() == [ALAN]
    assert (sent["role_id"], sent["owner_user_id"], sent["connection_id"], sent["user_role"]) == (86, 4, 5, "recruiter")


def test_nothing_new_starts_no_import(world):
    assert auto.check_connection(_connection()) == 0
    assert world["started"] == [] and world["errors"] == []


def test_a_running_import_is_left_to_finish(world):
    world["context"] = {**world["context"], "importing": True}
    world["frame"] = _frame([{"Profile": ALAN}])
    assert auto.check_connection(_connection(pending_rows={"li:/in/alan-turing": "x"})) == 0
    assert world["started"] == [] and world["pending"] == []


def test_expired_google_connection_is_reported_and_held_back(world, monkeypatch):
    def expired(user_id):
        raise GoogleAuthError("Your Google connection expired. Connect Google again.", code="reconnect")
    monkeypatch.setattr(google_oauth, "access_token_for_user", expired)
    assert auto.check_connection(_connection()) is None
    assert world["errors"] == [("Your Google connection expired. Connect Google again.", auto.BLOCKED_RETRY_MINUTES)]


def test_missing_linkedin_column_and_deleted_role_are_reported(world):
    world["frame"] = pd.DataFrame([{"Mail": "a@x.com"}], dtype=object)
    assert auto.check_connection(_connection()) is None
    world["context"] = {**world["context"], "role_exists": False}
    assert auto.check_connection(_connection()) is None
    assert "LinkedIn column" in world["errors"][0][0] and "no longer exists" in world["errors"][1][0]
    assert world["started"] == []


def test_finished_import_marks_its_rows_and_a_failed_one_does_not(monkeypatch):
    calls: Dict[str, Any] = {"seen": [], "errors": []}
    monkeypatch.setattr(si.ci, "create_upload_record", lambda **kw: 77)
    monkeypatch.setattr(si, "mark_synced", lambda cid, upload_id: None)
    monkeypatch.setattr(si, "held_rows", lambda cid: {})
    monkeypatch.setattr(si, "mark_rows_seen", lambda cid, keys: calls["seen"].append(keys))
    monkeypatch.setattr(si, "set_sync_error", lambda cid, message, retry_minutes=0: calls["errors"].append(message))
    outcome = {"status": "completed"}
    monkeypatch.setattr(si, "run_sheet_import", lambda **kw: dict(outcome))

    def run() -> None:
        si.start_sheet_import(df=_frame([{"Profile": ADA}, {"First": None}]), mapping=MAPPING, owner_user_id=4,
                              user_role="recruiter", role_id=86, enrich_linkedin=True, enrichment_mode="none",
                              spreadsheet_id="sid", tab_title="t", sheet_title="s", connection_id=5)
        for thread in list(si._threads.values()):
            thread.join(5)

    run()
    assert calls == {"seen": [["li:/in/ada-lovelace"]], "errors": []}
    outcome.update(status="failed", error="apify down")
    run()
    assert len(calls["seen"]) == 1 and calls["errors"] == ["Import failed: apify down"]


def test_google_token_is_reused_until_the_connection_changes(world):
    for _ in range(3):
        auto.check_connection(_connection())
    assert world["token_calls"] == [7]
    world["context"] = {**world["context"], "token_stamp": "t2"}      # reconnected with another account
    auto.check_connection(_connection())
    world["context"] = {**world["context"], "token_stamp": None}      # disconnected: nothing is held
    auto.check_connection(_connection())
    auto.check_connection(_connection())
    assert world["token_calls"] == [7, 7, 7, 7]


def test_a_held_token_google_rejects_is_replaced_once(world):
    auto.check_connection(_connection())
    seen = []

    def read(token, sid, tab):
        seen.append(token)
        if token == "token1":
            raise sheets.SheetSourceError("Your Google connection expired. Connect Google again.", code="reconnect")
        return world["frame"]

    world["read"] = read
    assert auto.check_connection(_connection()) == 0
    assert seen == ["token1", "token2"] and world["errors"] == []


def test_sheet_digest_changes_with_any_cell(world):
    auto.check_connection(_connection())
    auto.check_connection(_connection())
    world["frame"] = _frame([{"Profile": ADA, "Stage": "Shortlisted"}])
    auto.check_connection(_connection())
    first, second, third = world["digests"]
    assert first == second != third


def test_idle_interval_is_never_faster_than_the_active_one(monkeypatch):
    monkeypatch.setenv("SHEET_AUTO_SYNC_SECONDS", "120")
    monkeypatch.setenv("SHEET_AUTO_SYNC_IDLE_SECONDS", "30")
    assert si.auto_sync_idle_seconds() == 120
    monkeypatch.delenv("SHEET_AUTO_SYNC_IDLE_SECONDS")
    assert si.auto_sync_idle_seconds() == 600


# ------------------------------------------------- rows waiting for LinkedIn data
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
ALAN_KEY = "li:/in/alan-turing"


def test_a_held_row_waits_for_its_next_attempt_then_is_ready_again():
    signatures = si.row_signatures(_frame([{"Profile": ALAN}]), MAPPING)
    pending = {ALAN_KEY: signatures[0][1]}
    held = si.held_after_attempt({}, [ALAN_KEY], now=NOW)
    assert held[ALAN_KEY]["attempts"] == 1
    wait = si.HOLD_RETRY_MINUTES[0]
    assert si.pick_ready_rows(signatures, [], pending, held, now=NOW + timedelta(minutes=wait - 1))[0] == []
    assert si.pick_ready_rows(signatures, [], pending, held, now=NOW + timedelta(minutes=wait + 1))[0] == [0]


def test_retries_back_off_and_then_give_up():
    held: Dict[str, Any] = {}
    waits = []
    for _ in si.HOLD_RETRY_MINUTES:
        assert si.given_up_keys(held) == set()
        held = si.held_after_attempt(held, [ALAN_KEY], now=NOW)
        waits.append((datetime.fromisoformat(held[ALAN_KEY]["retry_at"]) - NOW).total_seconds() / 60)
    assert tuple(waits) == si.HOLD_RETRY_MINUTES
    assert si.given_up_keys(held) == {ALAN_KEY}


def _run_import(monkeypatch, *, phase1: Dict[str, Any], frame: pd.DataFrame, mapping=MAPPING, **kw: Any):
    loaded: List[Dict[str, Any]] = []
    monkeypatch.setattr(si.linkedin_import, "run_linkedin_import", lambda **k: dict(phase1))
    monkeypatch.setattr(si.linkedin_import, "write_progress", lambda upload_id, **u: None)
    monkeypatch.setattr(si.ci, "_process_upload_rows", lambda **k: loaded.append(k))
    result = si.run_sheet_import(upload_id=1, df=frame, mapping=mapping, owner_user_id=4, user_role="recruiter",
                                 role_id=86, **kw)
    return result, loaded


def _phase1(**statuses: str) -> Dict[str, Any]:
    urls = {"ada": ADA, "alan": ALAN}
    return {"status": "completed_with_errors", "errors": [], "deferred": [urls[n] for n in statuses],
            "results": [{"url": urls[n], "status": s} for n, s in statuses.items()]}


def test_a_profile_not_returned_is_held_instead_of_loaded_bare(monkeypatch):
    frame = _frame([{"Profile": ALAN, "First": "Alan"}])
    result, loaded = _run_import(monkeypatch, phase1=_phase1(alan="not_found"), frame=frame, hold_missing=True)
    assert result["held"] == [ALAN_KEY] and loaded == []


def test_a_row_out_of_retries_or_without_holding_loads_from_the_sheet(monkeypatch):
    frame = _frame([{"Profile": ALAN, "First": "Alan"}])
    result, loaded = _run_import(monkeypatch, phase1=_phase1(alan="not_found"), frame=frame, hold_missing=True,
                                 load_anyway=[ALAN_KEY])
    assert result["held"] == [] and loaded[0]["df"]["Profile"].tolist() == [ALAN]
    result, loaded = _run_import(monkeypatch, phase1=_phase1(alan="not_found"), frame=frame)
    assert result["held"] == [] and len(loaded) == 1


def test_a_profile_that_came_back_but_failed_to_save_is_held_too(monkeypatch):
    frame = _frame([{"Profile": ALAN, "First": "Alan"}])
    result, loaded = _run_import(monkeypatch, phase1=_phase1(alan="error"), frame=frame, hold_missing=True)
    assert result["held"] == [ALAN_KEY] and loaded == []


def test_held_rows_are_not_marked_seen(monkeypatch):
    calls: Dict[str, Any] = {"seen": [], "held": []}
    monkeypatch.setattr(si.ci, "create_upload_record", lambda **kw: 78)
    monkeypatch.setattr(si, "mark_synced", lambda cid, upload_id: None)
    monkeypatch.setattr(si, "held_rows", lambda cid: {})
    monkeypatch.setattr(si, "mark_rows_seen", lambda cid, keys: calls["seen"].append(keys))
    monkeypatch.setattr(si, "hold_rows", lambda cid, keys: calls["held"].append(keys))
    monkeypatch.setattr(si, "run_sheet_import", lambda **kw: {"status": "completed_with_errors", "held": [ALAN_KEY]})
    si.start_sheet_import(df=_frame([{"Profile": ADA}, {"Profile": ALAN}]), mapping=MAPPING, owner_user_id=4,
                          user_role="recruiter", role_id=86, enrich_linkedin=True, enrichment_mode="none",
                          spreadsheet_id="sid", tab_title="t", sheet_title="s", connection_id=5)
    for thread in list(si._threads.values()):
        thread.join(5)
    assert calls == {"seen": [["li:/in/ada-lovelace"]], "held": [[ALAN_KEY]]}


# ------------------------------------------------------ rows removed from the sheet
def test_a_row_removed_from_the_sheet_is_forgotten_and_imports_again_when_added_back(world):
    seen = ["li:/in/ada-lovelace", ALAN_KEY]
    held = {"li:/in/gone": {"attempts": 1, "retry_at": "2026-01-01T00:00:00+00:00"}}
    assert auto.check_connection(_connection(seen_row_keys=seen, held_rows=held)) == 0
    assert world["forgot"][-1] == [ALAN_KEY, "li:/in/gone"]
    world["frame"] = _frame([{"Profile": ADA}, {"Profile": ALAN}])
    assert auto.check_connection(_connection()) == 0          # first read after it is back: settling
    assert auto.check_connection(_connection(pending_rows=world["pending"][-1])) == 1
    assert world["started"][0]["df"]["Profile"].tolist() == [ALAN]


def test_a_held_row_is_not_sent_before_its_retry_is_due(world):
    world["frame"] = _frame([{"Profile": ADA}, {"Profile": ALAN}])
    pending = {ALAN_KEY: si.row_signatures(world["frame"], MAPPING)[1][1]}
    later = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    earlier = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    waiting = _connection(pending_rows=pending, held_rows={ALAN_KEY: {"attempts": 1, "retry_at": later}})
    assert auto.check_connection(waiting) == 0 and world["started"] == []
    due = _connection(pending_rows=pending, held_rows={ALAN_KEY: {"attempts": 1, "retry_at": earlier}})
    assert auto.check_connection(due) == 1


def test_an_interrupted_import_is_closed_before_the_rows_go_again(world, monkeypatch):
    closed: List[Any] = []
    monkeypatch.setattr(si.ci, "_mark_upload_failed", lambda upload_id, message: closed.append((upload_id, message)))
    world["context"] = {**world["context"], "stale_upload_id": 85}
    assert auto.check_connection(_connection()) == 0
    assert closed == [(85, "Import interrupted; the sheet sync started it again.")]


# ------------------------------------------------------------ a single name column
def test_one_name_column_supplies_first_and_last_name():
    frame = pd.DataFrame([{"Your Name": "Ashwin G", "Link": ALAN}, {"Your Name": "Cher", "Link": ADA},
                          {"Your Name": None, "Link": ADA}], dtype=object)
    named, mapping = si.with_full_name(frame, {"Your Name": "custom", "Link": "linkedin"})
    assert named[si._FIRST_NAME_COLUMN].tolist() == ["Ashwin", "Cher", None]
    assert named[si._LAST_NAME_COLUMN].tolist() == ["G", None, None]
    assert mapping[si._FIRST_NAME_COLUMN] == "first_name" and mapping["Your Name"] == "custom"
    assert "Your Name" in frame.columns and si._FIRST_NAME_COLUMN not in frame.columns


def test_a_mapped_name_or_no_name_column_is_left_alone():
    frame = _frame([{"Profile": ADA, "First": "Ada"}])
    assert si.with_full_name(frame, MAPPING) == (frame, MAPPING)
    plain = pd.DataFrame([{"Company name": "Acme", "Link": ADA}], dtype=object)
    mapping = {"Company name": "custom", "Link": "linkedin"}
    assert si.with_full_name(plain, mapping) == (plain, mapping)


def test_an_import_still_recording_its_outcome_is_left_to_finish(world, monkeypatch):
    world["frame"] = _frame([{"Profile": ALAN}])
    monkeypatch.setattr(si, "is_running", lambda upload_id: upload_id == 81)
    assert auto.check_connection(_connection(pending_rows={ALAN_KEY: "x"})) == 0
    assert world["started"] == [] and world["pending"] == []


def test_sync_status_reports_new_rows_held_rows_and_a_running_import(monkeypatch):
    from datetime import datetime as dt
    rows = [(5, 86, "role", "Test", "Form responses 1", {ALAN_KEY: "d1", "row:abc": "d2", "row:imported": "d3"},
             {ALAN_KEY: {"attempts": 1, "retry_at": "2026-11-26T12:20:00+00:00"}}, dt(2026, 11, 26, 12, 0), None, None,
             ["row:imported"]),
            (6, None, None, None, "Sheet1", None, None, None, None, 99, None)]
    monkeypatch.setattr(google_oauth, "ensure_tables", lambda: None)
    monkeypatch.setattr(google_oauth, "_with_conn", lambda fn: rows)
    first, second = si.sync_status(user_id=4)
    assert (first["new_count"], first["held_count"], first["next_retry_at"]) == (1, 1, "2026-11-26T12:20:00+00:00")
    assert first["role_name"] == "role" and first["running_upload_id"] is None
    assert (second["new_count"], second["held_count"], second["running_upload_id"]) == (0, 0, 99)
    assert second["sheet_title"] == "Google Sheet"

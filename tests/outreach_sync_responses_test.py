"""Response-sync regressions behind "Failed to sync responses" / "No response yet".

Two independent bugs produced one recruiter-visible symptom:

A. The Roles page called the sync endpoint with a ROOT-RELATIVE api path. In
   production the frontend is an Azure Static Web App on a different host from
   the backend, so that request hit the static host (405) and the sync never
   ran. It passed unnoticed locally because the vite dev server proxies the same
   path to the backend, making the broken and correct forms indistinguishable in
   dev. Hence a Python test reading JSX: it is the only place in this repo that
   runs automatically.

B. The backend's UPDATE nested the LinkedIn fields inside the email fields, and
   only the Smartlead/email branch ever sets `status`. Candidates with a
   HeyReach campaign but no Smartlead campaign therefore had their replies
   silently dropped while still being counted as synced.
"""

import pathlib
import re

from backend.api.routes.outreach import _build_outreach_update_fields

FRONTEND_SRC = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "src"

# The single legitimate site: API_BASE's own definition falls back to '/api'
# for localhost, where same-origin is correct.
RELATIVE_API_ALLOWLIST = {"store/useAppStore.js"}

# A root-relative api path inside a string literal.
RELATIVE_API = re.compile(r"""['"`]/api/""")


def _li_update(**overrides):
    upd = {
        "candidate_id": 1,
        "li_status": "replied",
        "li_sent_count": 2,
        "li_last_action_at": "2026-08-03T10:00:00",
        "li_response_text": "Thanks for reaching out. Here are my digits: 80959 29684",
        "li_response_received_at": "2026-08-03T21:07:17",
        "li_conversation_id": "conv-1",
        "li_account_id": "acct-1",
    }
    upd.update(overrides)
    return upd


def _email_update(**overrides):
    upd = {
        "candidate_id": 1,
        "status": "replied",
        "message_sent_count": 1,
        "last_message_sent_at": "2026-08-03T10:00:00",
        "response_received_at": "2026-08-03T11:00:00",
        "response_text": "interested",
    }
    upd.update(overrides)
    return upd


def _sets(fields, column):
    return any(f.startswith(f"{column} =") for f in fields)


def test_linkedin_only_candidate_gets_their_reply_written():
    """The reported bug. 167 of 390 candidate_outreach rows are LinkedIn-only,
    and every one of them used to have its reply discarded because the LinkedIn
    block sat inside `if "status" in upd`."""
    fields = _build_outreach_update_fields(_li_update())

    assert _sets(fields, "li_response_text")
    assert _sets(fields, "li_sent_count")
    assert _sets(fields, "li_last_action_at")
    # No email data present, so no email columns are touched.
    assert not _sets(fields, "status")
    assert not _sets(fields, "response_text")


def test_email_only_candidate_is_unaffected():
    fields = _build_outreach_update_fields(_email_update())

    assert _sets(fields, "status")
    assert _sets(fields, "response_text")
    assert not _sets(fields, "li_response_text")


def test_both_channels_write_both_sides():
    fields = _build_outreach_update_fields({**_email_update(), **_li_update()})

    assert _sets(fields, "response_text")
    assert _sets(fields, "li_response_text")


def test_a_missing_reply_never_nulls_out_a_stored_one():
    """The batch get_campaign_activities path returns no reply_at and may carry
    no text. Writing that over a previously-synced reply is how a candidate who
    HAS responded reverts to reading "No response yet"."""
    for empty in (None, "", "   "):
        fields = _build_outreach_update_fields(_li_update(li_response_text=empty))
        assert not _sets(fields, "li_response_text"), repr(empty)
        assert not _sets(fields, "li_response_received_at"), repr(empty)
        # The rest of the LinkedIn state is still refreshed.
        assert _sets(fields, "li_sent_count")


def test_reply_timestamp_falls_back_to_the_stored_value():
    """reply_at only exists on the per-lead path, so the batch path would
    otherwise clear a known timestamp."""
    fields = _build_outreach_update_fields(_li_update(li_response_received_at=None))
    received = [f for f in fields if f.startswith("li_response_received_at")]
    assert received and "COALESCE" in received[0]


def test_status_cannot_be_downgraded_from_replied():
    """A stale batch result must not flip 'replied' back to 'message_sent'."""
    fields = _build_outreach_update_fields(_li_update(li_status="message_sent"))
    li_status = [f for f in fields if f.startswith("li_status")]
    assert li_status and "CASE WHEN li_status = 'replied' THEN 'replied'" in li_status[0]


def test_nothing_learned_yields_only_the_timestamp():
    """The caller skips these rather than issuing a no-op UPDATE and counting it
    as a synced response — which is what made the toast claim successes that
    wrote nothing."""
    assert _build_outreach_update_fields({"candidate_id": 1}) == ["updated_at = NOW()"]


def test_no_root_relative_api_paths_in_frontend_source():
    """Guards bug A. Any root-relative api literal works in dev (vite proxies it)
    and silently fails in production (different host, no proxy), so it cannot be
    caught by running the app locally."""
    offenders = []
    for path in FRONTEND_SRC.rglob("*"):
        if path.suffix not in {".js", ".jsx"} or not path.is_file():
            continue
        rel = path.relative_to(FRONTEND_SRC).as_posix()
        if rel in RELATIVE_API_ALLOWLIST:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RELATIVE_API.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Root-relative api paths found. Use API_BASE (or a store action) instead — "
        "these break on the hosted Static Web App:\n" + "\n".join(offenders)
    )

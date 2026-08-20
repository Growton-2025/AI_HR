"""Bulk shortlist must actually send everyone to Clay.

A recruiter shortlisted 30+ people and got 15 phone numbers, against a usual
70-80%. Per-profile "Fetch contact" worked; the bulk action did nothing at all.

Two faults:

1. enrich_targets excluded any candidate already in an active outreach state.
   That guard belongs to ENROLMENT — do not re-enroll someone mid-campaign —
   but enrolment status says nothing about whether we know their phone number.
   Worse, the bulk shortlist enrols candidates itself, so the first run
   disqualified them from every run after it. In one role that left 47 of 47
   phoneless shortlisted candidates unreachable: the button ran, the toast
   said "queued", and no row was ever sent.

2. The callback URL defaulted to a hardcoded, long-expired ngrok tunnel.
   Any deployment that did not set NGROK_URL sent candidates into the
   waterfall with a callback pointing at a dead host: Clay ran the enrichment,
   billed for it, and posted the answer into the void. The symptom is
   indistinguishable from Clay having no data.
"""

import pytest

from backend.api.routes.outreach import _candidate_role_queue_state
from backend.services import clay


def _candidate(**overrides):
    row = {
        "id": 1, "first_name": "Kiran", "last_name": "Mangrulia",
        "name": "Kiran Mangrulia", "email": None, "phone": None,
        "linkedin": "https://www.linkedin.com/in/kiran-mangrulia-5a8a3299",
    }
    row.update(overrides)
    return row


# ── 1. who gets sent ────────────────────────────────────────────────────────

def test_a_candidate_missing_a_phone_needs_enrichment_even_with_an_email():
    # 7 of the 10 sampled had an email and no mobile. Requiring both to be
    # missing would have written them off.
    state = _candidate_role_queue_state(_candidate(email="kiran@example.com"))

    assert state["needs_enrichment"] is True


def test_a_candidate_with_both_contacts_is_left_alone():
    state = _candidate_role_queue_state(
        _candidate(email="kiran@example.com", phone="+919900000000")
    )

    assert state["needs_enrichment"] is False


def test_no_linkedin_means_nothing_to_enrich_from():
    state = _candidate_role_queue_state(_candidate(linkedin=""))

    assert state["needs_enrichment"] is False


def test_enrolment_state_no_longer_decides_enrichment():
    import inspect

    from backend.api.routes import browse

    source = inspect.getsource(browse.bulk_update_status)
    target_block = source[source.rindex("enrich_targets = ["):]
    target_block = target_block[: target_block.index("]")]

    # This is the whole bug: being mid-campaign is not a reason to keep
    # someone's phone number unknown.
    assert "globally_active" not in target_block
    assert "needs_enrichment" in target_block


# ── 2. where Clay is told to send the answer ────────────────────────────────

def test_no_url_is_baked_in_as_a_default():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(clay))
    defaults = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "getenv"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
    ]

    # CLAY_URL/CLAY_AUTH still carry defaults; the callback host must not, or a
    # deployment that forgets to set it silently posts results to a stranger.
    assert not any("ngrok" in str(d) or "trycloudflare" in str(d) for d in defaults)


def test_the_callback_uses_the_hosted_url_when_present(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://hayasa.example.com/")
    monkeypatch.delenv("NGROK_URL", raising=False)

    assert clay._callback_base() == "https://hayasa.example.com"


def test_a_tunnel_is_accepted_when_no_hosted_url_is_set(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.setenv("NGROK_URL", "https://abc.trycloudflare.com")

    assert clay._callback_base() == "https://abc.trycloudflare.com"


def test_nothing_is_sent_when_no_callback_url_is_configured(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.delenv("NGROK_URL", raising=False)
    posted = []
    monkeypatch.setattr(clay.requests, "post", lambda *a, **k: posted.append(a))

    # Spending Clay credits on an answer nobody can receive is worse than
    # failing loudly — and it looks exactly like Clay finding nothing.
    assert clay.trigger_clay("Kiran", "Mangrulia", "https://li/in/x") is False
    assert posted == []


def test_the_callback_url_is_never_guessed_from_a_local_tunnel(monkeypatch):
    import inspect

    source = inspect.getsource(clay._callback_base)

    # The payload carries a candidate's email and phone; auto-detecting a
    # tunnel could hand that to whatever unrelated app happens to be running.
    assert "get_ngrok_url" not in source


# ── 3. sending a whole selection ────────────────────────────────────────────

def test_bulk_sends_every_candidate(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://hayasa.example.com")
    sent = []

    class _Resp:
        ok = True
        text = '{"success":true}'

    monkeypatch.setattr(
        clay.requests, "post",
        lambda url, json=None, headers=None, timeout=None: sent.append(json) or _Resp(),
    )

    rows = [(f"First{i}", f"Last{i}", f"https://li/in/p{i}") for i in range(47)]
    result = clay.trigger_clay_bulk(rows)

    assert result == {"sent": 47, "failed": 0, "skipped": 0}
    assert len(sent) == 47
    assert sent[0]["callback_url"] == "https://hayasa.example.com/results"


def test_bulk_skips_rows_with_no_linkedin(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://hayasa.example.com")

    class _Resp:
        ok = True
        text = "{}"

    monkeypatch.setattr(clay.requests, "post", lambda *a, **k: _Resp())

    result = clay.trigger_clay_bulk([
        ("A", "B", "https://li/in/a"),
        ("C", "D", ""),
        ("E", "F", None),
    ])

    assert result == {"sent": 1, "failed": 0, "skipped": 2}


def test_bulk_reports_failures_rather_than_pretending(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://hayasa.example.com")

    def _boom(*a, **k):
        raise RuntimeError("clay is down")

    monkeypatch.setattr(clay.requests, "post", _boom)

    assert clay.trigger_clay_bulk([("A", "B", "https://li/in/a")]) == {
        "sent": 0, "failed": 1, "skipped": 0,
    }


def test_bulk_is_queued_as_one_task_not_one_per_candidate():
    import inspect

    from backend.api.routes import browse

    source = inspect.getsource(browse.bulk_update_status)

    # Background tasks run in sequence and each trigger blocks on the network,
    # so 47 separate tasks tied up the worker for 47 round trips.
    assert "background_tasks.add_task(trigger_clay_bulk, rows)" in source
    assert "background_tasks.add_task(trigger_clay," not in source


# ── 4. a slow lookup is not an error ────────────────────────────────────────

ROLES_JSX = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "frontend" / "src" / "pages" / "Roles.jsx"
).read_text()


def test_a_still_running_lookup_is_never_reported_as_broken():
    # Clay took over eight minutes in a live batch, while the UI gave up at 40
    # seconds and told the recruiter the callback was broken.
    assert "has not reached Hayasa" not in ROLES_JSX
    assert "Check the Clay callback step" not in ROLES_JSX
    assert "Still enriching" in ROLES_JSX
    assert "the row updates on its own" in ROLES_JSX


def test_the_wait_covers_a_realistic_waterfall():
    import re

    attempts = int(re.search(r"CLAY_POLL_ATTEMPTS = (\d+)", ROLES_JSX).group(1))
    fast, slow = 2000, 5000
    total_ms = sum(fast if a < 10 else slow for a in range(attempts))

    assert total_ms >= 4 * 60 * 1000     # minutes, not seconds


def test_an_unconfigured_callback_is_still_a_real_error():
    import inspect

    from backend.api.routes import enrichment

    source = inspect.getsource(enrichment.enrich_candidate)

    # Clay would run, bill, and post the answer nowhere — worth interrupting
    # the recruiter for, unlike a lookup that is merely slow.
    assert "_callback_base()" in source
    assert "status_code=503" in source

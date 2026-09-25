"""Person linking: which candidate rows are the same human, and the timeline
that follows from it. docs/candidate-history-linking-plan.md, PR 2."""
import datetime as dt

from backend.services import person_identity as pi
from backend.services import person_timeline as pt


# ── keys ───────────────────────────────────────────────────────────────────

def test_linkedin_beats_email_beats_phone():
    assert pi.person_key_for("/in/jane", None, "jane@x.com", "9876543210") == ("/in/jane", "linkedin")
    assert pi.person_key_for(None, "https://linkedin.com/in/Jane/", "jane@x.com", None) == ("/in/jane", "linkedin")
    assert pi.person_key_for(None, None, " Jane@X.com ", "+91 98765 43210") == ("email:jane@x.com", "email")
    assert pi.person_key_for(None, None, None, "+91 98765 43210") == ("phone:9876543210", "phone")
    assert pi.person_key_for(None, None, None, "12345") == (None, "none")


def test_legacy_suffixed_rows_resolve_to_the_same_person():
    assert pi.person_key_for("/in/jane_legacy_1090", None, None, None)[0] == "/in/jane"


# ── weak-key acceptance rules ──────────────────────────────────────────────

def _r(cid, nli=None, li=None, email=None, phone=None):
    return {"id": cid, "normalized_linkedin": nli, "linkedin": li, "email": email, "phone": phone}


def test_email_match_with_a_different_linkedin_is_not_the_same_person():
    # Two people behind one agency mailbox.
    assert pi._accept(_r(2, nli="/in/john", email="agency@x.com"), key="/in/jane", mail="agency@x.com", tail=None, peers_keys={"/in/john"}) is False


def test_email_match_without_linkedin_is_accepted():
    assert pi._accept(_r(2, email="jane@x.com"), key="/in/jane", mail="jane@x.com", tail=None, peers_keys=set()) is True


def test_no_linkedin_on_our_side_accepts_only_agreeing_rows():
    # We only have an email; two matched rows carry two different LinkedIn keys -> ambiguous.
    assert pi._accept(_r(2, nli="/in/john", email="a@x.com"), key=None, mail="a@x.com", tail=None, peers_keys={"/in/john", "/in/jane"}) is False
    assert pi._accept(_r(2, nli="/in/john", email="a@x.com"), key=None, mail="a@x.com", tail=None, peers_keys={"/in/john"}) is True


# ── resolve / lookup against a fake database ───────────────────────────────

class _Cur:
    """Answers each SELECT from a script keyed on a distinctive SQL fragment."""
    def __init__(self, script):
        self.script = script
        self.executed = []
        self._rows = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        for fragment, rows in self.script:
            if fragment in sql:
                self._rows = rows() if callable(rows) else rows
                return
        self._rows = []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _dbrow(cid, owner=None, owner_email=None, name="X", pool="catalog_from_upload", archived=False,
           nli=None, li=None, email=None, phone=None):
    return (cid, owner, owner_email, name, pool, archived, nli, li, email, phone)


def test_resolve_finds_master_recruiter_copies_and_suffixed_rows(monkeypatch):
    pi._table_cache.clear()
    me = _dbrow(14438, nli="/in/arjun", email="arjun@x.com")
    matches = [
        me,
        _dbrow(1090, nli="/in/arjun_legacy_1090"),                 # lost the master dedupe
        _dbrow(20001, owner=19, owner_email="jaya@growton.co", nli="/in/arjun"),
        _dbrow(30001, nli="/in/someone-else", email="arjun@x.com"),  # same mailbox, different person
        _dbrow(30002, email="arjun@x.com"),                         # phone-book row, no LinkedIn
    ]
    cur = _Cur([
        ("to_regclass", [(True,)]),
        ("WHERE c.id = %s", [me]),
        ("WHERE (c.normalized_linkedin", matches),
    ])
    res = pi.resolve_person(cur, 14438)
    assert res.person_key == "/in/arjun" and res.matched_on == "linkedin"
    assert res.candidate_ids == [1090, 14438, 20001, 30002]
    assert 30001 not in res.candidate_ids


def test_lookup_before_saving_reports_prior_rows_and_counts():
    pi._table_cache.clear()
    cur = _Cur([
        ("to_regclass", [(True,)]),
        ("WHERE (c.normalized_linkedin", [_dbrow(2519, name="Nethranand", nli="/in/nethranand")]),
        ("SELECT COUNT(*) FROM calls", [(3, dt.datetime(2026, 9, 24, 13, 47), 12, None, 57, 3, dt.datetime(2026, 9, 2), 1, None)]),
    ])
    out = pi.lookup_person(cur, linkedin="https://www.linkedin.com/in/Nethranand/")
    assert out["known"] is True
    assert out["prior_rows"][0]["id"] == 2519
    assert (out["calls"], out["inbound_calls"], out["linkedin_replies"], out["emails"]) == (3, 12, 57, 3)
    assert out["last_interaction_at"].startswith("2026-09-24T13:47")


def test_lookup_with_nothing_to_go_on():
    assert pi.lookup_person(_Cur([]), linkedin="https://twitter.com/x") == {"known": False, "matched_on": "none"}


def test_summary_parameter_count_matches_placeholders():
    pi._table_cache.clear()
    cur = _Cur([
        ("to_regclass", [(True,)]),
        ("SELECT COUNT(*) FROM calls", [(0, None, 0, None, 0, 0, None, 0, None)]),
    ])
    out = pi.person_summary(cur, [1, 2])
    sql, params = cur.executed[-1]
    assert sql.count("%s") == len(params) == 9
    assert out["last_interaction_at"] is None


# ── timeline ───────────────────────────────────────────────────────────────

def test_timeline_merges_rows_dedupes_messages_and_sorts_newest_first(monkeypatch):
    pi._table_cache.clear()
    rows = [_dbrow(1, owner=None, name="Jane", nli="/in/jane"),
            _dbrow(2, owner=19, owner_email="jaya@growton.co", name="Jane", nli="/in/jane", pool="recruiter_upload")]
    msg = {"id": "m1", "time": "2026-09-20T10:00:00Z", "type": "REPLY", "direction": "inbound", "email_body": "hi"}
    cur = _Cur([
        ("to_regclass", [(True,)]),
        ("WHERE c.id = %s", [rows[0]]),
        ("WHERE (c.normalized_linkedin", rows),
        ("FROM calls c", [
            (10, 1, dt.datetime(2026, 9, 24, 13, 47), "Not Connected - Not Reachable", 0, None, None, None, "Incoming freeze", False, "admin@gmail.com", None, "Jane"),
            (11, 2, dt.datetime(2026, 9, 25, 9, 0), "Connected - Interested", 120, "https://rec", "great", "Lead: hi", None, False, "jaya@growton.co", None, "Jane"),
        ]),
        ("FROM deleted_calls", []),
        ("FROM inbound_calls", []),
        ("FROM candidate_outreach", [(1, 7, "Camp", [msg], None), (2, 7, "Camp", [msg], None)]),   # same message on both rows
        ("FROM recruitment_roles", [(7, "AE - EMEA")]),
        ("FROM candidate_status_history", []),
        ("FROM candidates WHERE id = ANY", [(1, "switched off", dt.datetime(2026, 9, 23))]),
        ("FROM candidate_person_links", []),
    ])
    out = pt.build_timeline(cur, 1, viewer_email="admin@gmail.com", viewer_is_admin=False)
    types = [i["type"] for i in out["items"]]
    assert out["candidate_ids"] == [1, 2]
    assert types == ["call", "call", "note", "linkedin_message"]        # newest first; message deduped once
    assert out["counts"] == {"call": 2, "note": 1, "linkedin_message": 1}
    mine, theirs = out["items"][1], out["items"][0]
    assert mine["you"] is True and "collapsed" not in mine
    assert theirs["you"] is False and theirs["collapsed"] is True     # other recruiter's call: visible, collapsed
    assert out["items"][3]["role"] == "AE - EMEA"


def test_timeline_row_scope_ignores_siblings():
    pi._table_cache.clear()
    cur = _Cur([
        ("to_regclass", [(True,)]),
        ("FROM calls c", []), ("FROM deleted_calls", []), ("FROM inbound_calls", []),
        ("FROM candidate_outreach", []), ("FROM candidate_status_history", []),
        ("FROM candidates WHERE id = ANY", []),
    ])
    out = pt.build_timeline(cur, 5, viewer_email="x@y", viewer_is_admin=True, scope="row")
    assert out["candidate_ids"] == [5] and out["items"] == []

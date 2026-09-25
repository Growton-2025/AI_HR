"""The LinkedIn identity key must be one canonical form everywhere.

On 2026-09-25 production held three formats in candidates.normalized_linkedin
(bare slug, raw URL, '/in/slug'), so the same person's rows could not find
each other and every consumer matched by hand. These pin the canonical rules,
the person key that reads through all historical formats, and the backfill
planner that rewrites the column without violating the unique indexes.
"""
import datetime as dt

import pytest

from backend.services.linkedin_backfill import apply_canonical_keys, plan_canonical_keys
from backend.services.linkedin_normalize import (
    canonical_email,
    is_canonical_linkedin_key,
    normalize_linkedin,
    person_key,
)


@pytest.mark.parametrize("raw, expected", [
    ("https://www.linkedin.com/in/Jane-Doe/?trk=public_profile", "/in/jane-doe"),
    ("http://linkedin.com/in/jane-doe#top", "/in/jane-doe"),
    ("linkedin.com/in/jane-doe", "/in/jane-doe"),
    ("www.linkedin.com/in/jane-doe/", "/in/jane-doe"),
    ("https://uk.linkedin.com/in/jane-doe", "/in/jane-doe"),           # locale mirror
    ("https://in.linkedin.com/pub/jane-doe/1a/2b3/4c5", "/in/jane-doe"),  # old public URL
    ("https://www.linkedin.com/in/jane-doe/details/experience/", "/in/jane-doe"),
    ("https://www.linkedin.com/in/jane%2Ddoe", "/in/jane-doe"),        # percent-encoded
    ("jane-doe-123", "/in/jane-doe-123"),                               # bare slug (legacy pipeline)
    ("/in/jane-doe", "/in/jane-doe"),                                   # what the app stores
    ("/in/Jane-Doe/", "/in/jane-doe"),
    ("  https://www.linkedin.com/in/jane-doe  ", "/in/jane-doe"),
])
def test_every_way_of_writing_a_profile_gives_one_key(raw, expected):
    assert normalize_linkedin(raw) == expected


@pytest.mark.parametrize("raw", [
    None, "", "   ",
    "https://www.linkedin.com/company/acme",     # not a person
    "https://www.linkedin.com/",                 # no profile
    "https://twitter.com/jane",                  # other host
    "https://notlinkedin.com/in/jane",           # look-alike host
    "not a url at all!",
])
def test_non_profiles_have_no_key(raw):
    assert normalize_linkedin(raw) is None


def test_canonical_key_is_a_fixed_point():
    key = normalize_linkedin("https://www.linkedin.com/in/Jane-Doe/")
    assert normalize_linkedin(key) == key
    assert is_canonical_linkedin_key(key)
    assert not is_canonical_linkedin_key("jane-doe")
    assert not is_canonical_linkedin_key("https://www.linkedin.com/in/jane-doe/")


@pytest.mark.parametrize("stored, expected", [
    ("jane-doe", "/in/jane-doe"),                                   # bare slug
    ("https://www.linkedin.com/in/jane-doe/", "/in/jane-doe"),      # raw URL
    ("/in/jane-doe", "/in/jane-doe"),
    ("/in/jane-doe_legacy_14438", "/in/jane-doe"),                  # dedupe suffix
    ("jane-doe_legacy_7", "/in/jane-doe"),
    ("/aroon", "/in/aroon"),                                        # stray leading slash
    (None, None), ("", None),
])
def test_person_key_reads_every_historical_format(stored, expected):
    assert person_key(stored) == expected


def test_emails_compare_lowercased_and_trimmed():
    assert canonical_email("  Jane.Doe@Example.COM ") == "jane.doe@example.com"
    assert canonical_email("") is None and canonical_email(None) is None


# ── backfill planner ───────────────────────────────────────────────────────

T0 = dt.datetime(2026, 1, 1)
T1 = dt.datetime(2026, 6, 1)


def _row(cid, stored, linkedin=None, owner=None, archived=False, updated=T0):
    return (cid, owner, archived, stored, linkedin, updated)


def test_plan_rewrites_non_canonical_keys_and_leaves_canonical_ones():
    plan = plan_canonical_keys([
        _row(1, "jane-doe"),
        _row(2, "https://www.linkedin.com/in/john-roe/"),
        _row(3, "/in/already-fine"),
        _row(4, None, linkedin="https://linkedin.com/in/from-url"),
    ])
    assert sorted(plan.updates) == [(1, "/in/jane-doe"), (2, "/in/john-roe"), (4, "/in/from-url")]
    assert plan.unchanged == 1 and not plan.collisions


def test_collision_keeps_the_canonical_holder_and_suffixes_the_rest():
    # A legacy bare-slug master row and a newer catalog row for the same person.
    plan = plan_canonical_keys([
        _row(1090, "arjun-braria-85743619", updated=T0),
        _row(14438, "/in/arjun-braria-85743619", updated=T1),
    ])
    assert plan.collisions == {(None, "/in/arjun-braria-85743619"): [1090, 14438]}
    assert plan.updates == [(1090, "/in/arjun-braria-85743619_legacy_1090")]


def test_collision_without_a_canonical_holder_keeps_the_most_recently_updated():
    plan = plan_canonical_keys([
        _row(1, "jane-doe", updated=T0),
        _row(2, "https://www.linkedin.com/in/jane-doe", updated=T1),
    ])
    assert set(plan.updates) == {(2, "/in/jane-doe"), (1, "/in/jane-doe_legacy_1")}


def test_scopes_do_not_collide_with_each_other():
    # Master row and a recruiter's copy are allowed to share the key.
    plan = plan_canonical_keys([
        _row(1, "jane-doe", owner=None),
        _row(2, "jane-doe", owner=19),
        _row(3, "jane-doe", owner=None, archived=True),   # archived: outside the index
    ])
    assert not plan.collisions
    assert sorted(plan.updates) == [(1, "/in/jane-doe"), (2, "/in/jane-doe"), (3, "/in/jane-doe")]


def test_previously_suffixed_rows_are_left_alone():
    plan = plan_canonical_keys([
        _row(1, "/in/jane-doe"),
        _row(2, "/in/jane-doe_legacy_2"),
    ])
    assert plan.updates == [] and plan.unchanged == 2


def test_plan_is_idempotent():
    rows = [_row(1, "jane-doe", updated=T0), _row(2, "/in/jane-doe", updated=T1)]
    first = plan_canonical_keys(rows)
    applied = {cid: key for cid, key in first.updates}
    rows_after = [(cid, o, a, applied.get(cid, s), li, u) for cid, o, a, s, li, u in rows]
    second = plan_canonical_keys(rows_after)
    assert second.updates == []


class _Cur:
    def __init__(self): self.executed = []
    def mogrify(self, sql, params): return (sql % tuple(f"'{p}'" for p in params)).encode()
    def execute(self, sql, params=None): self.executed.append(sql)


def test_apply_writes_suffixed_keys_before_keepers():
    plan = plan_canonical_keys([
        _row(1, "jane-doe", updated=T0),
        _row(2, "https://www.linkedin.com/in/jane-doe", updated=T1),
    ])
    cur = _Cur()
    assert apply_canonical_keys(cur, plan) == 2
    # The loser moves off the contested key first; the keeper then takes it.
    assert len(cur.executed) == 2
    assert "_legacy_1" in cur.executed[0] and "_legacy_" not in cur.executed[1]
    assert "'2', '/in/jane-doe'" in cur.executed[1]


def test_apply_batches_rows_instead_of_one_statement_each():
    plan = plan_canonical_keys([_row(i, f"person-{i}") for i in range(1, 1201)])
    cur = _Cur()
    assert apply_canonical_keys(cur, plan) == 1200
    assert len(cur.executed) == 3          # 500 + 500 + 200, not 1,200 round trips

"""A recruiter's softphone must provision on every deployment, not just a laptop.

Inbound calls ring one <User> per registered endpoint, so a recruiter with no
endpoint row is never rung: the incoming-call banner simply never appears, with
nothing in the UI to say why. Three separate defects meant no new endpoint could
be created on the hosted backend at all, and only the one account provisioned
before they landed could receive callbacks.

1. The alias embedded the environment key, which on a hosted deployment is a
   69-character hostname. Plivo caps an alias at 50 characters and allows no
   dots, so endpoints.create() was refused before the database was touched:
       ValidationError: {'alias': ['Ensure that this field has atleast 1
                                   and atmost 50 characters']}
2. The insert named ON CONFLICT (user_id). The table's only unique index is
   (user_id, env_key), so it raised InvalidColumnReference every time.
3. That insert never wrote env_key, so the row defaulted to 'legacy' and the
   reader — which looks up (user_id, env_key) — could never find it again.
"""

import asyncio
import re

import pytest

from backend.integrations import plivo_service as ps

# The real hosted host, verbatim: it is the input that broke, and its length is
# the whole point of the test.
HOSTED = "growton-backend-v2-e3a3hxdmagfggcg9.centralindia-01.azurewebsites.net"

PLIVO_ALIAS_MAX = 50
PLIVO_ALIAS_CHARSET = re.compile(r"[A-Za-z0-9_-]+")


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", f"https://{HOSTED}")
    return HOSTED


@pytest.fixture
def local(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.delenv("NGROK_URL", raising=False)


class _Cursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def _patch_db(monkeypatch, cursor):
    import backend.db.connection as dbc

    conn = _Conn(cursor)
    monkeypatch.setattr(dbc, "get_db_connection", lambda **kw: conn)
    monkeypatch.setattr(dbc, "return_db_connection", lambda _c: None)
    return conn


# ── 1. the alias Plivo will actually accept ─────────────────────────────────

def test_hosted_alias_fits_plivos_limit(hosted):
    alias = ps.endpoint_alias_for_user(19)

    assert len(alias) <= PLIVO_ALIAS_MAX
    assert PLIVO_ALIAS_CHARSET.fullmatch(alias)
    assert "." not in alias


def test_the_alias_that_broke_production_would_still_be_rejected(hosted):
    # Proof the test is measuring the right thing: the old form is 82 chars.
    old_form = f"recruiter_19_{ps._env_key()}"

    assert len(old_form) > PLIVO_ALIAS_MAX
    assert not PLIVO_ALIAS_CHARSET.fullmatch(old_form)
    assert len(ps.endpoint_alias_for_user(19)) <= PLIVO_ALIAS_MAX


def test_local_alias_is_legal_too(local):
    alias = ps.endpoint_alias_for_user(19)

    assert alias.startswith("recruiter_19_local")
    assert len(alias) <= PLIVO_ALIAS_MAX


def test_alias_is_stable_for_the_same_environment(hosted):
    assert ps.endpoint_alias_for_user(19) == ps.endpoint_alias_for_user(19)


def test_alias_differs_per_user_and_per_environment(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", f"https://{HOSTED}")
    hosted_19 = ps.endpoint_alias_for_user(19)
    hosted_29 = ps.endpoint_alias_for_user(29)
    # Two Azure slots sharing a 20-character prefix must not collide: the
    # readable head is truncated, so only the hash tail separates them.
    monkeypatch.setenv("PUBLIC_URL", "https://growton-backend-v2-OTHERSLOT.centralindia-01.azurewebsites.net")
    other_19 = ps.endpoint_alias_for_user(19)

    assert hosted_19 != hosted_29
    assert hosted_19 != other_19


# ── 2 & 3. one writer, with the environment on it ───────────────────────────

def test_persist_uses_the_composite_conflict_target(monkeypatch, hosted):
    cursor = _Cursor()
    conn = _patch_db(monkeypatch, cursor)

    assert ps._persist_endpoint_row(19, "ep-1", "user123", "pw", "app-1") is True
    sql, params = cursor.executed[0]

    # The table's only unique index is (user_id, env_key); naming user_id alone
    # raised InvalidColumnReference on every provisioning attempt.
    assert "ON CONFLICT (user_id, env_key)" in sql
    assert "ON CONFLICT (user_id) " not in sql
    assert "env_key" in sql
    assert HOSTED in params           # the row is stamped with this deployment
    assert conn.committed


def test_no_insert_anywhere_still_names_the_missing_constraint():
    # A second writer drifting out of step with the schema is what caused this
    # outage; there must be exactly one, and it must not be the broken form.
    source = __import__("inspect").getsource(ps)
    # Comments describing the old bug are fine; executable SQL is not.
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "ON CONFLICT (user_id)" not in code


def test_provisioning_reads_back_the_row_it_just_wrote(monkeypatch, hosted):
    # env_key defaulting to 'legacy' meant the reader never matched, so every
    # login minted another endpoint. Reader and writer must agree on the key.
    cursor = _Cursor()
    _patch_db(monkeypatch, cursor)
    ps._persist_endpoint_row(19, "ep-1", "user123", "pw", "app-1")
    _, write_params = cursor.executed[0]

    ps.mark_endpoint_registered(19)
    _, heartbeat_params = cursor.executed[1]

    assert HOSTED in write_params
    assert HOSTED in heartbeat_params


# ── 4. the ring list belongs to one deployment ──────────────────────────────

def test_ring_list_is_scoped_to_this_environment(monkeypatch, hosted):
    cursor = _Cursor(rows=[("user_hosted",)])
    _patch_db(monkeypatch, cursor)

    assert ps.get_registered_endpoint_usernames() == ["user_hosted"]
    sql, params = cursor.executed[0]

    # Without this a production call also rings a developer's laptop, which is
    # bound to a different Plivo Application and cannot connect — burning the
    # whole <Dial timeout> before the caller reaches voicemail.
    assert "env_key = %s" in sql
    assert params[-1] == HOSTED


def test_ring_list_still_excludes_stale_and_busy_endpoints(monkeypatch, hosted):
    cursor = _Cursor(rows=[])
    _patch_db(monkeypatch, cursor)
    ps.get_registered_endpoint_usernames()
    sql, _ = cursor.executed[0]

    assert "last_registered_at >" in sql
    assert "in_call_since" in sql


# ── rollback of an endpoint we could not store ──────────────────────────────

def test_rollback_is_a_noop_without_an_endpoint_id():
    # Nothing to undo, and no credentials needed to decide that.
    asyncio.run(ps._delete_endpoint_quietly(None, 19))


def test_rollback_deletes_the_orphan(monkeypatch):
    deleted = []

    class _Endpoints:
        def delete(self, endpoint_id):
            deleted.append(endpoint_id)

    class _Client:
        endpoints = _Endpoints()

    monkeypatch.setattr(ps, "PLIVO_AUTH_ID", "id")
    monkeypatch.setattr(ps, "PLIVO_AUTH_TOKEN", "token")
    monkeypatch.setattr(ps.plivo, "RestClient", lambda *a, **k: _Client())

    asyncio.run(ps._delete_endpoint_quietly("ep-9", 19))

    assert deleted == ["ep-9"]

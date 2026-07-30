import asyncio
import time

from backend.api.routes import plivo as plivo_routes
from backend.integrations import plivo_service


class _FakeApplicationResponse:
    app_id = "app-123"


class _FakeEndpointResponse:
    username = "endpoint-user"


class _FakeApplications:
    def __init__(self, calls):
        self._calls = calls

    def create(self, **kwargs):
        time.sleep(0.02)
        self._calls.append(("application", kwargs))
        return _FakeApplicationResponse()


class _FakeEndpoints:
    def __init__(self, calls):
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(("endpoint", kwargs))
        return _FakeEndpointResponse()


class _FakeRestClient:
    calls = []

    def __init__(self, auth_id, auth_token):
        self.auth_id = auth_id
        self.auth_token = auth_token
        self.applications = _FakeApplications(self.calls)
        self.endpoints = _FakeEndpoints(self.calls)


class _FakeDialRequest:
    async def form(self):
        return {
            "To": "8618884276",
            "From": "sip:endpointuser@phone.plivo.com",
            "CallUUID": "plivo-call-uuid-1",
        }


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def close(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1


def _reset_plivo_setup_state():
    plivo_service.endpoint_username = ""
    plivo_service.endpoint_password = ""
    plivo_service.endpoint_public_url = ""
    plivo_service.setup_error = ""
    plivo_service.last_calls.clear()
    plivo_service.last_call_states.clear()
    plivo_service.latest_call_uuid = None


def test_recording_callback_waits_for_final_recording_metadata():
    assert plivo_routes._recording_callback_is_final(
        {
            "RecordingDuration": "-1",
            "RecordingDurationMs": "-1",
            "RecordingEndMs": "-1",
        }
    ) is False

    assert plivo_routes._recording_callback_is_final(
        {
            "RecordingDuration": "12",
            "RecordingDurationMs": "12000",
            "RecordingEndMs": "12000",
        }
    ) is True


def test_download_plivo_recording_uses_account_auth(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None, auth=None):
        captured["url"] = url
        captured["timeout"] = timeout
        captured["auth"] = auth

        class _Response:
            status_code = 200
            content = b"audio"

        return _Response()

    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")
    monkeypatch.setattr(plivo_service.requests, "get", fake_get)

    response = plivo_service.download_plivo_recording("https://media.example.com/recording.mp3", timeout=17)

    assert response.status_code == 200
    assert captured == {
        "url": "https://media.example.com/recording.mp3",
        "timeout": 17,
        "auth": ("auth-id", "auth-token"),
    }


def test_setup_plivo_serializes_concurrent_endpoint_creation(monkeypatch):
    _reset_plivo_setup_state()


def test_plivo_dial_maps_endpoint_username_to_call_uuid(monkeypatch, tmp_path):
    _reset_plivo_setup_state()
    # Keep persisted softphone state out of the real data/ directory so the
    # concurrent-setup assertions below always exercise the fresh-create path.
    monkeypatch.setattr(plivo_service, "_PLIVO_STATE_FILE", str(tmp_path / "plivo_state.json"))
    fake_conn = _FakeConnection()

    from backend.api.routes import calls

    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: fake_conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(plivo_service, "get_ngrok_url", lambda: "https://backend.example.com")
    monkeypatch.setattr(plivo_service, "PLIVO_NUMBER", "+918035312881")
    monkeypatch.setattr(plivo_service, "mark_endpoint_busy",
                        lambda user_id=None, username=None: None)

    # Application state now lives in Postgres, so without these the test reads
    # and WRITES the real database — it persisted its mock app_id ("app-123")
    # into plivo_app_state, after which every later run reused the fake app and
    # skipped the create path these assertions depend on.
    monkeypatch.setattr(plivo_service, "_load_app_state", lambda kind: None)
    monkeypatch.setattr(plivo_service, "_save_app_state", lambda *a, **k: True)
    monkeypatch.setattr(plivo_service, "_provision_app_once",
                        lambda kind, answer_url, create_fn: create_fn())

    response = asyncio.run(plivo_routes.plivo_dial(_FakeDialRequest()))

    assert response.status_code == 200
    assert plivo_service.last_calls["endpointuser"] == "plivo-call-uuid-1"
    assert plivo_service.last_call_states["endpointuser"]["to_number"] == "+918618884276"
    assert fake_conn.commits == 1
    query, params = fake_conn.cursor_obj.executed[0]
    assert "plivo_call_uuid" in query
    assert params == ("plivo-call-uuid-1", "endpointuser")
    _FakeRestClient.calls = []

    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")
    monkeypatch.setattr(plivo_service, "get_ngrok_url", lambda: "https://backend.example.com")
    monkeypatch.setattr(plivo_service.plivo, "RestClient", _FakeRestClient)

    async def run_concurrent_setup():
        return await asyncio.gather(plivo_service.setup_plivo(), plivo_service.setup_plivo())

    results = asyncio.run(run_concurrent_setup())

    assert all(result["success"] for result in results)
    assert results[0]["username"] == "endpoint-user"
    assert results[1]["username"] == "endpoint-user"
    assert [kind for kind, _kwargs in _FakeRestClient.calls].count("application") == 1
    assert [kind for kind, _kwargs in _FakeRestClient.calls].count("endpoint") == 1
    endpoint_kwargs = next(kwargs for kind, kwargs in _FakeRestClient.calls if kind == "endpoint")
    assert endpoint_kwargs["username"].isalnum()
    assert 1 <= len(endpoint_kwargs["username"]) <= 25

    _reset_plivo_setup_state()


class _FakeCtxCursor:
    """Cursor supporting the `with conn.cursor() as cur` form used by the
    endpoint-registry helpers."""

    def __init__(self, rows=()):
        self.executed = []
        self._rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self._rows

    def close(self):
        return None


class _FakeCtxConnection:
    def __init__(self, rows=()):
        self.cursor_obj = _FakeCtxCursor(rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _patch_endpoint_db(monkeypatch, conn):
    from backend.db import connection as db_connection

    monkeypatch.setattr(db_connection, "get_db_connection",
                        lambda **kwargs: conn, raising=False)
    monkeypatch.setattr(db_connection, "return_db_connection",
                        lambda c: None, raising=False)


def test_ring_all_skips_recruiters_already_on_a_call(monkeypatch):
    """Inbound must not fork to an endpoint mid-conversation: it rings over a
    live candidate call, and if that recruiter is the only one online the new
    caller burns the whole <Dial timeout> before reaching voicemail."""
    conn = _FakeCtxConnection(rows=[("idleuser",)])
    _patch_endpoint_db(monkeypatch, conn)

    usernames = plivo_service.get_registered_endpoint_usernames()

    assert usernames == ["idleuser"]
    query, params = conn.cursor_obj.executed[0]
    assert "in_call_since IS NULL" in query
    # Busy is only ever cleared by a browser signal, so a crashed tab must not
    # strand an endpoint as permanently unreachable — the flag ages out.
    assert plivo_service.BUSY_STALE_SECONDS in params


def test_busy_flag_set_and_cleared_by_user_id(monkeypatch):
    conn = _FakeCtxConnection()
    _patch_endpoint_db(monkeypatch, conn)
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.delenv("NGROK_URL", raising=False)

    plivo_service.mark_endpoint_busy(7)
    plivo_service.clear_endpoint_busy(7)

    busy_query, busy_params = conn.cursor_obj.executed[0]
    idle_query, idle_params = conn.cursor_obj.executed[1]
    # env_key is part of the key: rows are per (user, environment).
    assert "in_call_since = CURRENT_TIMESTAMP" in busy_query and busy_params == (7, "local")
    assert "in_call_since = NULL" in idle_query and idle_params == (7, "local")
    assert "user_id = %s" in busy_query
    assert conn.commits == 2


def test_dial_webhook_marks_the_dialing_endpoint_busy(monkeypatch, tmp_path):
    """The browser also beacons busy, but this webhook is the one signal that
    cannot be lost to a wedged tab."""
    from backend.api.routes import calls

    _reset_plivo_setup_state()
    monkeypatch.setattr(plivo_service, "_PLIVO_STATE_FILE", str(tmp_path / "plivo_state.json"))
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: _FakeConnection())
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)

    marked = []
    monkeypatch.setattr(plivo_service, "mark_endpoint_busy",
                        lambda user_id=None, username=None: marked.append(username))

    plivo_service.record_browser_dial("endpointuser", "call-uuid-9", "8618884276")

    assert marked == ["endpointuser"]
    _reset_plivo_setup_state()


def _build_credentials_app(monkeypatch):
    from fastapi import FastAPI
    from backend.api import deps, schemas

    app = FastAPI()
    app.dependency_overrides[deps.get_current_user] = lambda: schemas.User(
        id=42, username="owner@example.com", email="owner@example.com",
    )
    app.include_router(plivo_routes.router, prefix="/api/plivo")
    return app


def _get_credentials(app):
    import httpx

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/plivo/credentials")

    return asyncio.run(call())


def test_credentials_flag_shared_endpoint_fallback_as_degraded(monkeypatch):
    """Falling back to the shared endpoint used to be silent. If a second
    recruiter also lands on it they share one SIP username, and call
    attribution then writes one recruiter's recording onto the other's
    candidate — so the recruiter has to be told before they place calls."""
    app = _build_credentials_app(monkeypatch)

    async def _no_endpoint(_user_id):
        return None

    monkeypatch.setattr(plivo_service, "ensure_endpoint_for_user", _no_endpoint)
    monkeypatch.setattr(plivo_service, "setup_plivo",
                        lambda *a, **k: asyncio.sleep(0, result={"success": True}))
    monkeypatch.setattr(plivo_service, "endpoint_username", "sharedendpoint")
    monkeypatch.setattr(plivo_service, "endpoint_password", "sharedpass")

    response = _get_credentials(app)

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "sharedendpoint"
    assert body["degraded"] is True
    assert body["degraded_reason"]


def test_credentials_are_not_degraded_on_the_normal_per_user_path(monkeypatch):
    app = _build_credentials_app(monkeypatch)

    async def _own_endpoint(_user_id):
        return {"username": "ownuser", "password": "ownpass"}

    monkeypatch.setattr(plivo_service, "ensure_endpoint_for_user", _own_endpoint)
    monkeypatch.setattr(plivo_service, "setup_plivo",
                        lambda *a, **k: asyncio.sleep(0, result={"success": True}))

    response = _get_credentials(app)

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "ownuser"
    assert body["degraded"] is False


class _RowcountCursor(_FakeCursor):
    """Cursor that reports how many rows an UPDATE touched, so the token path
    can be told apart from the username fallback."""

    def __init__(self, rowcounts):
        super().__init__()
        self._rowcounts = list(rowcounts)
        self.rowcount = 0

    def execute(self, query, params=None):
        super().execute(query, params)
        self.rowcount = self._rowcounts.pop(0) if self._rowcounts else 0


class _RowcountConnection(_FakeConnection):
    def __init__(self, rowcounts):
        super().__init__()
        self.cursor_obj = _RowcountCursor(rowcounts)


def _dial_test_setup(monkeypatch, conn):
    from backend.api.routes import calls

    _reset_plivo_setup_state()
    plivo_service.dial_token_states.clear()
    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda c: None)
    monkeypatch.setattr(plivo_service, "mark_endpoint_busy",
                        lambda user_id=None, username=None: None)


def test_dial_token_attributes_the_call_to_its_own_row(monkeypatch):
    """The bug this replaces: attribution matched the most recently updated
    `calls` row for a SIP username, so with two attempts sharing a username
    (two recruiters on the shared endpoint, or one fast redial) the recording
    landed on the wrong candidate. A token match must not consult recency."""
    conn = _RowcountConnection([1])
    _dial_test_setup(monkeypatch, conn)

    plivo_service.record_browser_dial("shareduser", "call-A", "8618884276", "token-A")

    query, params = conn.cursor_obj.executed[0]
    assert "WHERE dial_token = %s" in query
    assert params == ("call-A", "token-A")
    # The recency-based query must not run at all when the token matched.
    assert len(conn.cursor_obj.executed) == 1
    assert "ORDER BY updated_at DESC" not in query


def test_concurrent_recruiters_do_not_claim_each_others_rows(monkeypatch):
    """Two dials interleaved on ONE shared SIP username — the exact scenario
    that used to cross-attribute. Each must update via its own token."""
    conn = _RowcountConnection([1, 1])
    _dial_test_setup(monkeypatch, conn)

    plivo_service.record_browser_dial("shareduser", "call-A", "8618884276", "token-A")
    plivo_service.record_browser_dial("shareduser", "call-B", "9876543210", "token-B")

    assert conn.cursor_obj.executed[0][1] == ("call-A", "token-A")
    assert conn.cursor_obj.executed[1][1] == ("call-B", "token-B")


def test_dial_without_a_token_still_falls_back_to_username_matching(monkeypatch):
    """Clients on a pre-token bundle must keep working, even though this is the
    buggy path — it is logged and slated for removal."""
    conn = _RowcountConnection([0])
    _dial_test_setup(monkeypatch, conn)

    plivo_service.record_browser_dial("endpointuser", "call-legacy", "8618884276")

    query, params = conn.cursor_obj.executed[0]
    assert "plivo_endpoint_username = %s" in query
    assert "ORDER BY updated_at DESC" in query
    assert params == ("call-legacy", "endpointuser")


def test_dial_token_state_is_keyed_per_attempt(monkeypatch):
    """The handshake must not be satisfiable by a previous call's UUID."""
    conn = _RowcountConnection([1, 1])
    _dial_test_setup(monkeypatch, conn)

    plivo_service.record_browser_dial("u1", "call-first", "8618884276", "token-first")
    plivo_service.record_browser_dial("u1", "call-second", "8618884276", "token-second")

    assert plivo_service.dial_token_states["token-first"]["call_uuid"] == "call-first"
    assert plivo_service.dial_token_states["token-second"]["call_uuid"] == "call-second"


def test_dial_webhook_extracts_the_token_whatever_plivo_names_it():
    """Plivo documents that X-PH-* headers reach the answer URL but not the
    parameter name it uses, so the extractor must tolerate the spellings."""
    for key in ("X-PH-DialToken", "x-ph-dialtoken", "DialToken", "SipHeader_X-PH-DialToken"):
        assert plivo_routes._extract_dial_token({key: "tok123"}) == "tok123", key
    assert plivo_routes._extract_dial_token({"CallUUID": "abc"}) == ""


def test_shared_endpoint_is_granted_to_one_user_only():
    plivo_service._shared_endpoint_holder["user_id"] = None
    plivo_service._shared_endpoint_holder["at"] = 0.0

    assert plivo_service.claim_shared_endpoint(1) is True
    assert plivo_service.claim_shared_endpoint(1) is True   # renewal by the holder
    assert plivo_service.claim_shared_endpoint(2) is False  # would corrupt attribution

    plivo_service.release_shared_endpoint(1)
    assert plivo_service.claim_shared_endpoint(2) is True
    plivo_service.release_shared_endpoint(2)


def test_shared_endpoint_claim_expires_so_nobody_is_locked_out():
    plivo_service._shared_endpoint_holder["user_id"] = 1
    plivo_service._shared_endpoint_holder["at"] = (
        time.time() - plivo_service.SHARED_ENDPOINT_CLAIM_TTL_SECONDS - 1
    )

    assert plivo_service.claim_shared_endpoint(2) is True
    plivo_service.release_shared_endpoint(2)


def test_endpoint_provisioning_aborts_when_the_registry_is_unreadable(monkeypatch):
    """A DB outage used to look identical to a first-time user, so every login
    created a Plivo endpoint that could not be persisted and was orphaned."""
    from backend.db import connection as db_connection

    monkeypatch.setattr(db_connection, "get_db_connection", lambda **kw: None, raising=False)
    monkeypatch.setattr(db_connection, "return_db_connection", lambda c: None, raising=False)

    created = []

    class _ExplodingRestClient:
        def __init__(self, *a, **k):
            created.append(True)
            raise AssertionError("must not touch the Plivo API when the DB is down")

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _ExplodingRestClient)

    result = asyncio.run(plivo_service.ensure_endpoint_for_user(7))

    assert result is None
    assert created == []


def test_env_key_separates_hosted_from_local(monkeypatch):
    """The same Postgres is shared by a laptop and the hosted backend. Without a
    per-environment key they claim each other's Application, which is how hosted
    ended up serving an endpoint bound to a dead ngrok tunnel."""
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    monkeypatch.delenv("NGROK_URL", raising=False)
    assert plivo_service._env_key() == "local"

    monkeypatch.setenv("PUBLIC_URL", "https://growton-backend-v2.azurewebsites.net")
    assert plivo_service._env_key() == "growton-backend-v2.azurewebsites.net"

    # Trailing path and casing must not produce a different environment.
    monkeypatch.setenv("PUBLIC_URL", "https://Growton-Backend-V2.azurewebsites.net/")
    assert plivo_service._env_key() == "growton-backend-v2.azurewebsites.net"


def test_stale_endpoint_is_rebound_to_the_current_application(monkeypatch):
    """The outage itself: an endpoint bound to an old Application keeps dialling
    through that Application's answer URL."""
    monkeypatch.setattr(plivo_service, "_load_app_state",
                        lambda kind: {"app_id": "NEW_APP"} if kind == "softphone" else None)
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")

    updates = []
    persisted = []

    class _Endpoints:
        def update(self, **kw):
            updates.append(kw)

    class _Client:
        def __init__(self, *a, **k):
            self.endpoints = _Endpoints()

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)
    monkeypatch.setattr(plivo_service, "_persist_endpoint_row",
                        lambda *a: persisted.append(a) or True)

    row = {"username": "u1", "password": "p1", "endpoint_id": "EP1", "app_id": "OLD_APP"}
    result = asyncio.run(plivo_service._rebind_if_stale(4, row))

    assert updates == [{"endpoint_id": "EP1", "app_id": "NEW_APP"}]
    assert result["app_id"] == "NEW_APP"
    assert persisted and persisted[0][4] == "NEW_APP"


def test_endpoint_already_on_the_current_application_is_left_alone(monkeypatch):
    monkeypatch.setattr(plivo_service, "_load_app_state",
                        lambda kind: {"app_id": "APP_1"})

    class _Client:
        def __init__(self, *a, **k):
            raise AssertionError("must not call Plivo when the binding is current")

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)

    row = {"username": "u1", "password": "p1", "endpoint_id": "EP1", "app_id": "APP_1"}
    assert asyncio.run(plivo_service._rebind_if_stale(4, row))["app_id"] == "APP_1"


def test_rebind_failure_still_returns_a_usable_endpoint(monkeypatch):
    """A failed re-bind must not block dialling — stale is no worse than before."""
    monkeypatch.setattr(plivo_service, "_load_app_state", lambda kind: {"app_id": "NEW_APP"})
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")

    class _Client:
        def __init__(self, *a, **k):
            raise RuntimeError("plivo down")

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)

    row = {"username": "u1", "password": "p1", "endpoint_id": "EP1", "app_id": "OLD_APP"}
    result = asyncio.run(plivo_service._rebind_if_stale(4, row))
    assert result["username"] == "u1"
    assert result["app_id"] == "OLD_APP"


def test_provisioning_reuses_the_environments_existing_application(monkeypatch):
    """One Application per environment. Previously the state lived in a
    gitignored file that Azure wiped every deploy, so hosted minted a new
    Application (and endpoint) on each release — 20 of them accumulated."""
    calls_made = []

    def _fake_provision(kind, answer_url, create_fn):
        calls_made.append(kind)
        return {"app_id": "EXISTING_APP", "username": "existinguser", "password": "pw"}

    monkeypatch.setattr(plivo_service, "_load_app_state", lambda kind: None)
    monkeypatch.setattr(plivo_service, "_load_persisted_softphone_state", lambda: None)
    monkeypatch.setattr(plivo_service, "_save_app_state", lambda *a, **k: True)
    monkeypatch.setattr(plivo_service, "_provision_app_once", _fake_provision)
    monkeypatch.setattr(plivo_service, "_persist_softphone_state", lambda *a: None)
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")
    monkeypatch.setattr(plivo_service, "get_ngrok_url", lambda: "https://hosted.example.com")

    class _Client:
        def __init__(self, *a, **k):
            self.applications = None

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)
    _reset_plivo_setup_state()

    result = asyncio.run(plivo_service.setup_plivo(force=True))

    assert result["success"] is True
    assert result["username"] == "existinguser"
    assert calls_made == ["softphone"]
    _reset_plivo_setup_state()


def test_local_cannot_steal_the_inbound_number_from_hosted(monkeypatch):
    """One DID for the account: whoever binds it last owns inbound. A developer
    starting the stack locally used to silently route candidate callbacks to a
    laptop tunnel that dies on sleep."""
    monkeypatch.setenv("PUBLIC_URL", "https://laptop.trycloudflare.com")
    monkeypatch.setattr(plivo_service, "_inbound_number_owner_env",
                        lambda: "growton-backend-v2.azurewebsites.net")
    monkeypatch.delenv("PLIVO_CLAIM_NUMBER", raising=False)
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")
    monkeypatch.setattr(plivo_service, "PLIVO_NUMBER", "+918035312881")
    monkeypatch.setattr(plivo_service, "get_ngrok_url", lambda: "https://laptop.trycloudflare.com")
    monkeypatch.setattr(plivo_service, "_load_app_state",
                        lambda kind: {"app_id": "APP_LOCAL", "answer_url": "https://laptop.trycloudflare.com/api/plivo/incoming"})
    monkeypatch.setattr(plivo_service, "_save_app_state", lambda *a, **k: True)

    rebinds = []

    class _Numbers:
        def update(self, **kw):
            rebinds.append(kw)

    class _Client:
        def __init__(self, *a, **k):
            self.numbers = _Numbers()
            self.applications = None

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)

    asyncio.run(plivo_service.ensure_inbound_application())
    assert rebinds == [], "local must not rebind the shared number"

    # ...unless the takeover is deliberate.
    monkeypatch.setenv("PLIVO_CLAIM_NUMBER", "true")
    asyncio.run(plivo_service.ensure_inbound_application())
    assert rebinds and rebinds[0]["app_id"] == "APP_LOCAL"


def test_registration_and_busy_writes_are_environment_scoped(monkeypatch):
    """A user holds one endpoint row per environment. Marking a laptop's row
    registered or busy must not touch the hosted row — the hosted endpoint would
    otherwise be dropped from the inbound ring-all and callers hit voicemail."""
    conn = _FakeCtxConnection()
    _patch_endpoint_db(monkeypatch, conn)
    monkeypatch.setenv("PUBLIC_URL", "https://hosted.example.com")

    plivo_service.mark_endpoint_registered(4)
    plivo_service.mark_endpoint_busy(4)

    reg_query, reg_params = conn.cursor_obj.executed[0]
    busy_query, busy_params = conn.cursor_obj.executed[1]
    assert "env_key = %s" in reg_query and reg_params == (4, "hosted.example.com")
    assert "env_key = %s" in busy_query and busy_params == (4, "hosted.example.com")


def test_endpoint_adoption_alias_is_environment_scoped(monkeypatch):
    """A bare recruiter_<id> alias let a second environment adopt the FIRST
    environment's live endpoint and reset its password, locking the original out
    of SIP registration entirely."""
    monkeypatch.setenv("PUBLIC_URL", "https://hosted.example.com")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_ID", "auth-id")
    monkeypatch.setattr(plivo_service, "PLIVO_AUTH_TOKEN", "auth-token")

    seen = {}

    class _Endpoints:
        def list(self, limit=20):
            return [type("E", (), {"alias": "recruiter_4", "endpoint_id": "EP1",
                                   "username": "u1"})()]

        def update(self, **kw):
            seen["updated"] = kw

    class _Client:
        def __init__(self, *a, **k):
            self.endpoints = _Endpoints()

    monkeypatch.setattr(plivo_service.plivo, "RestClient", _Client)

    # The only endpoint present belongs to another environment's alias, so it
    # must NOT be adopted and its password must not be reset.
    assert asyncio.run(plivo_service._adopt_orphaned_endpoint(4)) is None
    assert "updated" not in seen

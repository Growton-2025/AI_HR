import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from starlette.requests import Request

from backend.api import schemas
from backend.api.routes import auth
from backend.integrations import frejun as frejun_module


class _FakeCursor:
    def __init__(self, results):
        self._results = list(results)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        if self._results:
            return self._results.pop(0)
        return None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _FakeResponse:
    def __init__(self, status_code, payload, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


def _build_request():
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/auth/frejun-login",
            "headers": [],
            "query_string": b"",
            "scheme": "https",
            "server": ("testserver", 443),
            "client": ("127.0.0.1", 5000),
        }
    )


def test_load_managed_token_falls_back_to_legacy_unmapped_row(monkeypatch):
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    cursor = _FakeCursor(
        [
            None,
            ("access-123", "refresh-123", expires_at, None, "Bearer", None, None),
        ]
    )

    monkeypatch.setattr(frejun_module, "get_db_connection", lambda: _FakeConnection(cursor))
    monkeypatch.setattr(frejun_module, "return_db_connection", lambda conn: None)

    manager = frejun_module.FreJunManager()
    token = manager._load_managed_token("ashwin@growton.co")

    assert token is not None
    assert token["access_token"] == "access-123"
    assert token["refresh_token"] == "refresh-123"
    assert len(cursor.executed) == 2
    assert cursor.executed[0][1] == ("ashwin@growton.co",)


def test_load_managed_token_checks_configured_alias_before_failing(monkeypatch):
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    cursor = _FakeCursor(
        [
            None,
            ("access-999", "refresh-999", expires_at, "ashwin@growton.co", "Bearer", "agent-1", "+919900000000"),
        ]
    )

    monkeypatch.setattr(frejun_module, "get_db_connection", lambda: _FakeConnection(cursor))
    monkeypatch.setattr(frejun_module, "return_db_connection", lambda conn: None)

    manager = frejun_module.FreJunManager()
    token = manager._load_managed_token(
        email="admin@growton.co",
        candidate_emails=["admin@growton.co", "ashwin@growton.co"],
        allow_legacy_unmapped=False,
    )

    assert token is not None
    assert token["access_token"] == "access-999"
    assert len(cursor.executed) == 2
    assert cursor.executed[0][1] == ("admin@growton.co",)
    assert cursor.executed[1][1] == ("ashwin@growton.co",)


def test_frejun_oauth_login_mode_url_returns_auth_url(monkeypatch):
    monkeypatch.setattr(auth, "FREJUN_OAUTH_CLIENT_ID", "client-123")
    monkeypatch.setattr(
        auth,
        "get_frejun_redirect_uri",
        lambda request=None: "https://backend.example.com/api/auth/frejun-callback",
    )
    monkeypatch.setenv("FREJUN_USER_EMAIL", "ashwin@growton.co")

    current_user = schemas.User(
        username="admin@growton.co",
        email="admin@growton.co",
        full_name="Admin",
    )

    result = asyncio.run(
        auth.frejun_oauth_login(
            _build_request(),
            current_user=current_user,
            mode="url",
        )
    )

    assert "auth_url" in result

    parsed = urlparse(result["auth_url"])
    query = parse_qs(parsed.query)
    assert query["client_id"] == ["client-123"]
    assert query["redirect_uri"] == ["https://backend.example.com/api/auth/frejun-callback"]

    state_payload = json.loads(base64.urlsafe_b64decode(query["state"][0].encode("utf-8")).decode("utf-8"))
    assert state_payload["app_user_email"] == "admin@growton.co"
    assert state_payload["frejun_user_email"] == "ashwin@growton.co"


def test_retrieve_voip_user_uses_v2_user_endpoint(monkeypatch):
    manager = frejun_module.FreJunManager()
    requests_made = []

    def fake_get(url, headers=None, params=None, timeout=None):
        requests_made.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return _FakeResponse(
            200,
            {
                "data": {
                    "email": params["email"],
                    "user_id": "agent-123",
                    "bb_calling": True,
                    "virtual_numbers": [],
                }
            },
        )

    monkeypatch.setattr(frejun_module.requests, "get", fake_get)

    result = manager._retrieve_voip_user("ashwin@growton.co", access_token="token-123")

    assert result["success"] is True
    assert requests_made[0]["url"] == "https://api.frejun.com/api/v2/integrations/user/"


def test_enable_browser_calling_uses_v2_user_endpoint(monkeypatch):
    manager = frejun_module.FreJunManager()
    requests_made = []

    def fake_patch(url, headers=None, params=None, json=None, timeout=None):
        requests_made.append(
            {"url": url, "headers": headers, "params": params, "json": json, "timeout": timeout}
        )
        return _FakeResponse(200, {"data": {"browser_calls": True}})

    monkeypatch.setattr(frejun_module.requests, "patch", fake_patch)

    result = manager._enable_browser_calling("token-123", "ashwin@growton.co")

    assert result["success"] is True
    assert result["version"] == "v2"
    assert requests_made[0]["url"] == "https://api.frejun.com/api/v2/integrations/user/"
    assert requests_made[0]["json"] == {"browser_calls": True}


def test_frejun_oauth_callback_uses_v2_token_endpoint(monkeypatch):
    import httpx

    saved_tokens = {}
    requests_made = []

    class _FakeFreJunManager:
        user_email = "ashwin@growton.co"

        @staticmethod
        def _normalize_email(value):
            return (value or "").strip().lower()

        def _save_managed_token(self, access_token, refresh_token, expires_in, email):
            saved_tokens.update(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_in": expires_in,
                    "email": email,
                }
            )

    class _FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, data=None, headers=None):
            requests_made.append({"url": url, "data": data, "headers": headers})
            return _FakeResponse(
                200,
                {
                    "access_token": "access-123",
                    "refresh_token": "refresh-123",
                    "expires_in": 7200,
                },
            )

    monkeypatch.setattr(frejun_module, "FreJunManager", _FakeFreJunManager)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(auth, "FREJUN_OAUTH_CLIENT_ID", "client-123")
    monkeypatch.setattr(auth, "FREJUN_OAUTH_CLIENT_SECRET", "secret-123")
    monkeypatch.setattr(
        auth,
        "get_frejun_redirect_uri",
        lambda request=None: "https://backend.example.com/api/auth/frejun-callback",
    )
    monkeypatch.setattr(
        auth,
        "get_frejun_post_auth_url",
        lambda request=None: "https://frontend.example.com/calls",
    )

    response = asyncio.run(
        auth.frejun_oauth_callback(
            _build_request(),
            code="oauth-code-123",
            email="ashwin@growton.co",
        )
    )

    assert response.status_code == 200
    assert requests_made[0]["url"] == "https://api.frejun.com/api/v2/oauth/token/"
    assert saved_tokens["access_token"] == "access-123"
    assert saved_tokens["email"] == "ashwin@growton.co"


def test_frejun_manager_keeps_call_to_voip_on_v1():
    manager = frejun_module.FreJunManager()

    assert manager.call_to_voip_url == "https://api.frejun.com/api/v1/integrations/call-to-voip/"

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


def test_setup_plivo_serializes_concurrent_endpoint_creation(monkeypatch):
    _reset_plivo_setup_state()


def test_plivo_dial_maps_endpoint_username_to_call_uuid(monkeypatch):
    _reset_plivo_setup_state()
    fake_conn = _FakeConnection()

    from backend.api.routes import calls

    monkeypatch.setattr(calls, "get_calls_db_connection", lambda: fake_conn)
    monkeypatch.setattr(calls, "return_db_connection", lambda conn: None)
    monkeypatch.setattr(plivo_service, "get_ngrok_url", lambda: "https://backend.example.com")
    monkeypatch.setattr(plivo_service, "PLIVO_NUMBER", "+918035312881")

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

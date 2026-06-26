import asyncio

from fastapi import BackgroundTasks

from backend.api import schemas
from backend.api.routes import outreach
from backend.api.routes.outreach import (
    _candidate_role_queue_state,
    _classify_role_email_candidates,
    _render_candidate_template,
    _render_role_template,
)
from backend.services.role_campaigns import campaign_payload
from backend.integrations.smartlead import SmartleadBot
from backend.api.schemas import RoleCreate
from backend.services.role_activation import fetch_role_activation
from backend.services.smartlead_role_dispatcher import _render, _valid_email


def test_role_and_candidate_tokens_render_without_touching_unknown_tokens():
    template = "Hi {{first_name}} {{last_name}}, {{role_name}} / {{company}}"
    rendered = _render_candidate_template(
        template,
        "Enterprise AE",
        {"first_name": "Asha", "last_name": "Rao"},
    )
    assert rendered == "Hi Asha Rao, Enterprise AE / {{company}}"
    assert _render_role_template("Role: {{role_name}}", "BDR") == "Role: BDR"


def test_recipient_classification_is_shortlist_only_and_idempotent():
    candidates = [
        {"id": 1, "status": "shortlisted", "email": "one@example.com"},
        {"id": 2, "status": "shortlisted", "email": ""},
        {"id": 3, "status": "rejected", "email": "three@example.com"},
        {"id": 4, "status": "shortlisted", "email": "four@example.com"},
        {"id": 5, "status": "shortlisted", "email": "not-an-email"},
    ]
    result = _classify_role_email_candidates(candidates, existing_ids={1})
    assert [item["id"] for item in result["shortlisted"]] == [1, 2, 4, 5]
    assert [item["id"] for item in result["missing_email"]] == [2, 5]
    assert [item["id"] for item in result["eligible"]] == [1, 4]
    assert [item["id"] for item in result["pending"]] == [4]


def test_campaign_payload_exposes_saved_setup_and_campaign_state():
    row = (
        "123",
        "BDR",
        "configured",
        None,
        "sender@example.com",
        "Hello {{first_name}}",
        "Body for {{role_name}}",
        "456",
        object(),
    )
    payload = campaign_payload(row)
    assert payload["campaign_id"] == "123"
    assert payload["campaign_configured"] is True
    assert payload["sender_account_id"] == "456"
    assert payload["initial_body"] == "Body for {{role_name}}"
    assert payload["started"] is True


def test_sender_removal_keeps_the_operation_campaign_scoped(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = '{"ok": true}'

        @staticmethod
        def json():
            return {"ok": True}

    def fake_delete(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("backend.integrations.smartlead.requests.delete", fake_delete)
    bot = SmartleadBot(api_key="test")
    bot.campaign_id = 123
    assert bot.remove_email_account("456") == {"ok": True}
    assert "/campaigns/123/email-accounts" in captured["url"]
    assert captured["json"] == {"email_account_ids": [456]}


def test_role_create_requires_complete_outreach_setup():
    request = RoleCreate(
        name="Enterprise AE",
        heyreach_campaign_id=123,
        smartlead_sender_account_id=456,
        email_subject="A {{role_name}} role",
        email_body="Hi {{first_name}}",
    )
    assert request.heyreach_campaign_id == 123
    assert request.smartlead_sender_account_id == 456


def test_activation_is_active_only_when_both_channels_are_configured():
    class Cursor:
        def execute(self, sql, params):
            self.params = params

        @staticmethod
        def fetchone():
            return (
                "smartlead-1", "configured", None, "sender@example.com",
                "Subject", "Body", "sender-1", "heyreach-1", "configured",
                None, "linkedin-sender-1",
            )

    payload = fetch_role_activation(Cursor(), 99)
    assert payload["activation_status"] == "active"
    assert payload["smartlead_campaign_id"] == "smartlead-1"
    assert payload["heyreach_campaign_id"] == "heyreach-1"


def test_pending_email_helpers_validate_and_render_candidate_tokens():
    assert _valid_email("asha@example.com") is True
    assert _valid_email("not-an-email") is False
    assert _render(
        "Hi {{first_name}} {{last_name}} — {{role_name}}",
        "BDR", "Asha", "Rao",
    ) == "Hi Asha Rao — BDR"


def test_bulk_shortlist_queue_state_allows_linkedin_while_email_waits():
    waiting = _candidate_role_queue_state(
        {"email": "", "phone": "", "linkedin": "https://linkedin.com/in/asha"}
    )
    assert waiting == {
        "email_status": "waiting_for_email",
        "linkedin_status": "scheduled",
        "needs_enrichment": True,
    }

    complete = _candidate_role_queue_state(
        {"email": "asha@example.com", "phone": "+919999999999", "linkedin": "https://linkedin.com/in/asha"}
    )
    assert complete == {
        "email_status": "scheduled",
        "linkedin_status": "scheduled",
        "needs_enrichment": False,
    }

    no_linkedin = _candidate_role_queue_state(
        {"email": "asha@example.com", "phone": "", "linkedin": ""}
    )
    assert no_linkedin["linkedin_status"] == "skipped_missing_linkedin"
    assert no_linkedin["needs_enrichment"] is False


def test_repeated_shortlist_repairs_status_without_duplicate_outreach(monkeypatch):
    class Cursor:
        def __init__(self):
            self.last_sql = ""
            self.statements = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.last_sql = " ".join(sql.split())
            self.statements.append((self.last_sql, params))

        def fetchone(self):
            if "FROM recruitment_role_candidates rc" in self.last_sql:
                return (
                    101, "Suvina Rai", "Suvina", "Rai",
                    "suvina@example.com", "+971500000000",
                    "https://linkedin.com/in/suvina",
                )
            if "FROM candidate_outreach" in self.last_sql:
                return ("in_campaign", "in_campaign", "email-campaign", "linkedin-campaign")
            if "FROM role_smartlead_campaigns" in self.last_sql:
                return ("email-campaign",)
            if "FROM role_heyreach_campaigns" in self.last_sql:
                return ("linkedin-campaign",)
            return None

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.commits = 0

        def cursor(self):
            return self.cursor_instance

        def commit(self):
            self.commits += 1

    class ConnectionContext:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, tb):
            return False

    connection = Connection()
    monkeypatch.setattr(
        outreach,
        "get_db_connection_context",
        lambda **_kwargs: ConnectionContext(connection),
    )
    monkeypatch.setattr(outreach, "_get_accessible_role", lambda *_args: (44, "Enterprise AE"))
    monkeypatch.setattr(
        outreach,
        "fetch_role_activation",
        lambda *_args: {"activation_status": "active"},
    )
    background_tasks = BackgroundTasks()

    result = asyncio.run(
        outreach.shortlist_role_candidate(
            role_id=44,
            candidate_id=101,
            background_tasks=background_tasks,
            current_user=schemas.User(id=7, username="recruiter", role="recruiter"),
        )
    )

    statements = [sql for sql, _params in connection.cursor_instance.statements]
    assert result["already_processed"] is True
    assert result["status"] == "Shortlisted"
    assert connection.commits == 1
    assert any("UPDATE candidates SET status='Shortlisted'" in sql for sql in statements)
    assert not any("INSERT INTO candidate_outreach" in sql for sql in statements)
    assert len(background_tasks.tasks) == 1

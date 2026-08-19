import asyncio

from backend.api import schemas
from backend.api.routes import browse, candidates, roles


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _query, _params):
        return None

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _Cursor(self.rows)


def test_role_details_returns_manage_pool_candidate_fields(monkeypatch):
    row = (
        44, "Enterprise AE", "Role description", 7,
        "Recruiter", "recruiter@example.com",
        1, None, 1,
        101, "High", "Strong fit",
        "Aadarsh Goyal", "https://linkedin.com/in/aadarsh", "Bengaluru, India",
        "Account Executive", "Profile summary",
        "aadarsh@example.com", "+919999999999", "Followup / In conversation",
        "Aadarsh", "Goyal", "Bengaluru",
        8.4, 2.1, "Call next Tuesday", "Interested in the role",
        "Enterprise Account Executive", "Example Corp",
    )
    connection = _Connection([row])

    monkeypatch.setattr(roles, "get_db_connection", lambda: connection)
    monkeypatch.setattr(roles, "return_db_connection", lambda _connection: None)
    monkeypatch.setattr(roles, "fetch_role_activation", lambda _cursor, _role_id: {})
    monkeypatch.setattr(roles, "PROFILES_BY_ID", {})
    roles.invalidate_role_detail_cache()

    result = asyncio.run(
        roles.get_role(
            "Enterprise AE",
            current_user=schemas.User(id=7, username="recruiter", role="recruiter"),
        )
    )

    candidate = result["candidates"][0]
    assert candidate["first_name"] == "Aadarsh"
    assert candidate["last_name"] == "Goyal"
    assert candidate["title"] == "Enterprise Account Executive"
    assert candidate["company"] == "Example Corp"
    assert candidate["city"] == "Bengaluru"
    assert candidate["total_experience_years"] == 8.4
    assert candidate["avg_tenure_years"] == 2.1
    assert candidate["email"] == "aadarsh@example.com"
    assert candidate["phone"] == "+919999999999"
    assert candidate["response"] == "Interested in the role"
    assert candidate["notes"] == "Call next Tuesday"
    assert candidate["status"] == "Followup / In conversation"


def test_role_cache_invalidation_removes_every_role_containing_candidate():
    roles._ROLE_DETAIL_CACHE.clear()
    roles._ROLE_DETAIL_CACHE.update({
        "7:recruiter:role-a": (
            1.0,
            {"id": 1, "candidates": [{"id": 101}, {"id": 102}]},
        ),
        "7:recruiter:role-b": (
            1.0,
            {"id": 2, "candidates": [{"id": 101}]},
        ),
        "7:recruiter:role-c": (
            1.0,
            {"id": 3, "candidates": [{"id": 999}]},
        ),
    })

    roles.invalidate_role_detail_cache_for_candidate(101)

    assert set(roles._ROLE_DETAIL_CACHE) == {"7:recruiter:role-c"}
    roles._ROLE_DETAIL_CACHE.clear()


def test_generic_status_update_invalidates_role_and_count_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        browse,
        "PROFILES_BY_ID",
        {101: {"owner_user_id": 7, "status": "To be started"}},
    )
    # Signature gained changed_by/source when status changes started being logged.
    monkeypatch.setattr(browse, "update_candidate_status", lambda candidate_id, status, **kwargs: True)
    monkeypatch.setattr(
        candidates,
        "invalidate_candidate_count_caches",
        lambda **kwargs: calls.append(("counts", kwargs)),
    )
    monkeypatch.setattr(
        roles,
        "invalidate_role_detail_cache_for_candidate",
        lambda candidate_id: calls.append(("roles", candidate_id)),
    )

    result = asyncio.run(
        browse.update_status(
            candidate_id=101,
            update=browse.StatusUpdate(status="Rejected"),
            current_user=schemas.User(id=7, username="recruiter", role="recruiter"),
        )
    )

    assert result["message"] == "Status updated successfully"
    assert ("counts", {"refresh_profile_ids": [101]}) in calls
    assert ("roles", 101) in calls


def test_role_browse_product_filter_uses_deterministic_sql(monkeypatch):
    monkeypatch.setattr(
        browse,
        "_summary_scope_sql",
        lambda *_args, **_kwargs: ("c.owner_user_id = %s", [7]),
    )

    where_sql, params = browse._browse_where_sql(
        schemas.User(id=7, username="recruiter", role="recruiter"),
        effective_scope="recruiter_pools",
        effective_recruiter=7,
        role_id=44,
        product_service="SaaS, Fintech",
    )

    assert "c.raw_fields->>'extracted_industry'" in where_sql
    assert "c.raw_fields->>'services'" in where_sql
    assert "co_product.product_service" in where_sql
    assert params == [
        7,
        "%saas%", "%saas%", "%saas%",
        "%fintech%", "%fintech%", "%fintech%",
    ]
    assert browse._can_use_fast_sql_browse(
        q=None,
        title=None,
        company=None,
        city=None,
        location_type=None,
        product_service="SaaS",
        status=None,
        created_by=None,
        min_exp=None,
        max_exp=None,
        min_avg_tenure=None,
        candidate_ids=None,
    ) is True

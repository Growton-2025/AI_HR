"""Provision and inspect the single Smartlead campaign owned by a role."""

from typing import Any, Dict

from backend.db.connection import get_db_connection_context
from backend.integrations.smartlead import SmartleadBot


def campaign_payload(row) -> Dict[str, Any]:
    if not row:
        return {
            "campaign_id": None,
            "campaign_name": "",
            "campaign_status": "missing",
            "campaign_error": "",
            "campaign_configured": False,
            "sender_account_id": "",
            "sender_email": "",
            "subject": "",
            "initial_body": "",
            "started": False,
        }
    return {
        "campaign_id": row[0],
        "campaign_name": row[1] or "",
        "campaign_status": row[2] or "pending",
        "campaign_error": row[3] or "",
        "campaign_configured": bool(row[0] and row[4]),
        "sender_account_id": row[7] or "",
        "sender_email": row[4] or "",
        "subject": row[5] or "",
        "initial_body": row[6] or "",
        "started": bool(row[8]),
    }


def fetch_role_campaign(cur, role_id: int):
    cur.execute(
        """
        SELECT campaign_id, campaign_name, provisioning_status, provisioning_error,
               sender_email, subject, initial_body, sender_account_id, started_at
        FROM role_smartlead_campaigns
        WHERE recruitment_role_id = %s
        """,
        (role_id,),
    )
    return cur.fetchone()


def provision_role_campaign(role_id: int, role_name: str) -> Dict[str, Any]:
    """Create the empty campaign once. Failed rows are safe to retry.

    ``provisioning`` is an atomic claim, so concurrent role creation/open/retry
    requests cannot create more than one remote campaign.
    """
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return {"campaign_status": "failed", "campaign_error": "Database connection failed"}
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO role_smartlead_campaigns
                    (recruitment_role_id, campaign_name, provisioning_status, created_at, updated_at)
                VALUES (%s, %s, 'pending', NOW(), NOW())
                ON CONFLICT (recruitment_role_id) DO UPDATE
                SET campaign_name = EXCLUDED.campaign_name, updated_at = NOW()
                """,
                (role_id, role_name),
            )
            cur.execute(
                """
                UPDATE role_smartlead_campaigns
                SET provisioning_status = 'provisioning', provisioning_error = NULL,
                    updated_at = NOW()
                WHERE recruitment_role_id = %s AND campaign_id IS NULL
                  AND (
                    provisioning_status <> 'provisioning'
                    OR updated_at < NOW() - INTERVAL '5 minutes'
                  )
                RETURNING campaign_id
                """,
                (role_id,),
            )
            claimed = cur.fetchone() is not None
            existing = fetch_role_campaign(cur, role_id)
            conn.commit()
            if existing and existing[0]:
                return campaign_payload(existing)
            if not claimed:
                return campaign_payload(existing)

    bot = SmartleadBot()
    try:
        campaign_id = bot.create_campaign(role_name)
    except Exception as exc:
        campaign_id = None
        error = str(exc) or "Smartlead campaign creation failed"
    else:
        error = None if campaign_id else "Smartlead campaign creation failed"
    status = "provisioned" if campaign_id else "failed"
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE role_smartlead_campaigns
                    SET campaign_id = COALESCE(campaign_id, %s), provisioning_status = %s,
                        provisioning_error = %s, updated_at = NOW()
                    WHERE recruitment_role_id = %s
                    """,
                    (str(campaign_id) if campaign_id else None, status, error, role_id),
                )
                row = fetch_role_campaign(cur, role_id)
                conn.commit()
                return campaign_payload(row)
    return {"campaign_id": campaign_id, "campaign_status": status, "campaign_error": error or ""}

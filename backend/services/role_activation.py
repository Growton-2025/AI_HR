"""Provision and report the two outreach channels owned by a recruitment role."""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from backend.db.connection import get_db_connection_context
from backend.integrations.heyreach import HeyReachBot
from backend.integrations.smartlead import SmartleadBot
from backend.services.role_campaigns import fetch_role_campaign, provision_role_campaign


def _render_role(value: str, role_name: str) -> str:
    return (value or "").replace("{{role_name}}", role_name)


def fetch_role_activation(cur, role_id: int) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT sc.campaign_id, sc.provisioning_status, sc.provisioning_error,
               sc.sender_email, sc.subject, sc.initial_body, sc.sender_account_id,
               hc.campaign_id, hc.provisioning_status, hc.provisioning_error,
               hc.sender_account_id
        FROM recruitment_roles r
        LEFT JOIN role_smartlead_campaigns sc ON sc.recruitment_role_id = r.id
        LEFT JOIN role_heyreach_campaigns hc ON hc.recruitment_role_id = r.id
        WHERE r.id = %s
        """,
        (role_id,),
    )
    row = cur.fetchone()
    if not row:
        return {
            "activation_status": "inactive",
            "activation_error": "Role not found",
            "smartlead_status": "missing",
            "heyreach_status": "missing",
        }
    smartlead_status = row[1] or "missing"
    heyreach_status = row[8] or "missing"
    active = heyreach_status == "configured" and smartlead_status in ("configured", "skipped")
    errors = [value for value in (row[2], row[9]) if value]
    return {
        "activation_status": "active" if active else "inactive",
        "activation_error": " | ".join(errors),
        "smartlead_status": smartlead_status,
        "smartlead_campaign_id": row[0],
        "smartlead_sender_email": row[3] or "",
        "email_subject": row[4] or "",
        "email_body": row[5] or "",
        "smartlead_sender_account_id": row[6] or "",
        "heyreach_campaign_id": row[7] or "",
        "heyreach_status": heyreach_status,
        "heyreach_sender_account_id": row[10] or "",
    }


def get_role_activation(role_id: int) -> Dict[str, Any]:
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return {
                "activation_status": "inactive",
                "activation_error": "Database connection failed",
                "smartlead_status": "failed",
                "heyreach_status": "failed",
            }
        with conn.cursor() as cur:
            return fetch_role_activation(cur, role_id)


def _save_heyreach_setup(
    role_id: int,
    role_name: str,
    campaign_id: int,
    sender_account_id: Optional[int],
    status: str,
    error: Optional[str] = None,
) -> None:
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO role_heyreach_campaigns
                    (recruitment_role_id, campaign_id, campaign_name, sender_account_id,
                     provisioning_status, provisioning_error, configured_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s,
                        CASE WHEN %s = 'configured' THEN NOW() ELSE NULL END, NOW(), NOW())
                ON CONFLICT (recruitment_role_id) DO UPDATE
                SET campaign_id = EXCLUDED.campaign_id,
                    campaign_name = EXCLUDED.campaign_name,
                    sender_account_id = EXCLUDED.sender_account_id,
                    provisioning_status = EXCLUDED.provisioning_status,
                    provisioning_error = EXCLUDED.provisioning_error,
                    configured_at = CASE WHEN EXCLUDED.provisioning_status = 'configured'
                                         THEN NOW() ELSE role_heyreach_campaigns.configured_at END,
                    updated_at = NOW()
                """,
                (
                    role_id,
                    str(campaign_id),
                    role_name,
                    str(sender_account_id) if sender_account_id else None,
                    status,
                    error,
                    status,
                ),
            )
            conn.commit()


def activate_role(
    role_id: int,
    role_name: str,
    heyreach_campaign_id: int,
    smartlead_sender_account_id: int,
    email_subject: str,
    email_body: str,
) -> Dict[str, Any]:
    """Idempotently configure both role channels, retaining failures for retry."""
    try:
        heyreach_sender = int(os.getenv("HEYREACH_DEFAULT_SENDER_ACCOUNT_ID", "113572") or 0)
    except (TypeError, ValueError):
        heyreach_sender = 0
    _save_heyreach_setup(
        role_id,
        role_name,
        heyreach_campaign_id,
        heyreach_sender or None,
        "provisioning",
    )
    try:
        if not heyreach_sender:
            raise RuntimeError("HEYREACH_DEFAULT_SENDER_ACCOUNT_ID is not configured")
        leads = HeyReachBot().get_campaign_leads(int(heyreach_campaign_id))
        if leads is None:
            raise RuntimeError("HeyReach campaign could not be validated")
        _save_heyreach_setup(
            role_id, role_name, heyreach_campaign_id, heyreach_sender, "configured"
        )
    except Exception as exc:
        _save_heyreach_setup(
            role_id,
            role_name,
            heyreach_campaign_id,
            heyreach_sender or None,
            "failed",
            str(exc)[:1000],
        )

    # If Smartlead is optional and not provided, mark it skipped and finish.
    if smartlead_sender_account_id <= 0:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO role_smartlead_campaigns
                            (recruitment_role_id, campaign_name, provisioning_status, created_at, updated_at)
                        VALUES (%s, %s, 'skipped', NOW(), NOW())
                        ON CONFLICT (recruitment_role_id) DO UPDATE
                        SET provisioning_status = 'skipped', provisioning_error = NULL, updated_at = NOW()
                        """,
                        (role_id, role_name),
                    )
                    conn.commit()
        return get_role_activation(role_id)

    # Always retain the requested Smartlead setup, even when a remote call fails.
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO role_smartlead_campaigns
                        (recruitment_role_id, campaign_name, sender_account_id, subject,
                         initial_body, provisioning_status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', NOW(), NOW())
                    ON CONFLICT (recruitment_role_id) DO UPDATE
                    SET sender_account_id = EXCLUDED.sender_account_id,
                        subject = EXCLUDED.subject, initial_body = EXCLUDED.initial_body,
                        updated_at = NOW()
                    """,
                    (
                        role_id,
                        role_name,
                        str(smartlead_sender_account_id),
                        email_subject,
                        email_body,
                    ),
                )
                conn.commit()

    campaign = provision_role_campaign(role_id, role_name)
    campaign_id = campaign.get("campaign_id")
    if campaign_id:
        bot = SmartleadBot()
        bot.campaign_id = int(campaign_id)
        try:
            accounts = bot.list_email_accounts()
            selected = next(
                (
                    account
                    for account in accounts
                    if str(account.get("id")) == str(smartlead_sender_account_id)
                ),
                None,
            )
            if not selected:
                raise RuntimeError("Selected Smartlead sender does not exist")
            if selected.get("is_smtp_success") is False:
                raise RuntimeError("Selected Smartlead sender is not connected")
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                with conn.cursor() as cur:
                    existing = fetch_role_campaign(cur, role_id)
            already_configured = bool(existing and existing[2] == "configured")
            old_sender_id = str(existing[7]) if existing and existing[7] else ""
            if not already_configured or old_sender_id != str(smartlead_sender_account_id):
                if bot.add_email_account(smartlead_sender_account_id) is None:
                    raise RuntimeError("Could not attach the Smartlead sender")
            if bot.set_email_sequence(
                _render_role(email_subject, role_name),
                _render_role(email_body, role_name),
            ) is None:
                raise RuntimeError("Could not configure the Smartlead email sequence")
            if not already_configured:
                start_time = datetime.now(timezone.utc) + timedelta(minutes=3)
                if bot.set_schedule(
                    tz=os.getenv("SMARTLEAD_DEFAULT_TIMEZONE", "Asia/Kolkata"),
                    start_hour="00:00",
                    end_hour="23:59",
                    start_time=start_time,
                    days_of_the_week=[0, 1, 2, 3, 4, 5, 6],
                ) is None:
                    raise RuntimeError("Could not configure the Smartlead schedule")
                if bot.update_campaign_settings(follow_up_percentage=50) is None:
                    raise RuntimeError("Could not configure Smartlead campaign settings")
            sender_email = selected.get("from_email") or selected.get("username") or ""
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE role_smartlead_campaigns
                        SET sender_account_id=%s, sender_email=%s, subject=%s,
                            initial_body=%s, provisioning_status='configured',
                            provisioning_error=NULL, configured_at=NOW(), updated_at=NOW()
                        WHERE recruitment_role_id=%s
                        """,
                        (
                            str(smartlead_sender_account_id),
                            sender_email,
                            email_subject,
                            email_body,
                            role_id,
                        ),
                    )
                    conn.commit()
        except Exception as exc:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE role_smartlead_campaigns
                            SET provisioning_status='failed', provisioning_error=%s, updated_at=NOW()
                            WHERE recruitment_role_id=%s
                            """,
                            (str(exc)[:1000], role_id),
                        )
                        conn.commit()

    return get_role_activation(role_id)


def retry_role_activation(role_id: int, role_name: str) -> Dict[str, Any]:
    current = get_role_activation(role_id)
    if not current.get("heyreach_campaign_id"):
        current["activation_error"] = "HeyReach campaign ID is missing"
        return current
        
    smartlead_id = current.get("smartlead_sender_account_id") or "0"
    try:
        smartlead_id_int = int(smartlead_id)
    except ValueError:
        smartlead_id_int = 0
        
    if smartlead_id_int > 0:
        required = (current.get("email_subject"), current.get("email_body"))
        if not all(required):
            current["activation_error"] = "The saved role setup is incomplete for Smartlead"
            return current

    return activate_role(
        role_id,
        role_name,
        int(current["heyreach_campaign_id"]),
        smartlead_id_int,
        current.get("email_subject") or "",
        current.get("email_body") or "",
    )

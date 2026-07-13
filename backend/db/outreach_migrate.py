"""Idempotent schema changes for role-based Smartlead and HeyReach outreach."""


def ensure_outreach_migrations(conn) -> None:
    if not conn:
        return
    old_autocommit = getattr(conn, "autocommit", False)
    try:
        if not old_autocommit:
            conn.rollback()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS role_smartlead_campaigns (
                    id SERIAL PRIMARY KEY,
                    recruitment_role_id INTEGER NOT NULL UNIQUE
                        REFERENCES recruitment_roles(id) ON DELETE CASCADE,
                    campaign_id VARCHAR(255),
                    campaign_name VARCHAR(255) NOT NULL,
                    provisioning_status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    provisioning_error TEXT,
                    sender_account_id VARCHAR(255),
                    sender_email VARCHAR(255),
                    subject TEXT,
                    initial_body TEXT,
                    configured_at TIMESTAMP,
                    started_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_role_smartlead_campaign_role ON role_smartlead_campaigns(recruitment_role_id)"
            )
            # Keep upgrades safe if an earlier development build created only
            # part of the role campaign table.
            for column_sql in (
                "ADD COLUMN IF NOT EXISTS campaign_id VARCHAR(255)",
                "ADD COLUMN IF NOT EXISTS campaign_name VARCHAR(255)",
                "ADD COLUMN IF NOT EXISTS provisioning_status VARCHAR(50) NOT NULL DEFAULT 'pending'",
                "ADD COLUMN IF NOT EXISTS provisioning_error TEXT",
                "ADD COLUMN IF NOT EXISTS sender_account_id VARCHAR(255)",
                "ADD COLUMN IF NOT EXISTS sender_email VARCHAR(255)",
                "ADD COLUMN IF NOT EXISTS subject TEXT",
                "ADD COLUMN IF NOT EXISTS initial_body TEXT",
                "ADD COLUMN IF NOT EXISTS configured_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS started_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT NOW()",
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT NOW()",
            ):
                cur.execute(f"ALTER TABLE role_smartlead_campaigns {column_sql}")

            for column_sql in (
                "ADD COLUMN IF NOT EXISTS initial_message TEXT",
                "ADD COLUMN IF NOT EXISTS initial_message_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS response_read_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS li_response_read_at TIMESTAMP",
            ):
                cur.execute(f"ALTER TABLE candidate_outreach {column_sql}")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS role_heyreach_campaigns (
                    id SERIAL PRIMARY KEY,
                    recruitment_role_id INTEGER NOT NULL UNIQUE
                        REFERENCES recruitment_roles(id) ON DELETE CASCADE,
                    campaign_id VARCHAR(255),
                    campaign_name VARCHAR(255) NOT NULL,
                    lead_list_id VARCHAR(255),
                    lead_list_name VARCHAR(255),
                    provisioning_status VARCHAR(50) NOT NULL DEFAULT 'missing',
                    provisioning_error TEXT,
                    sender_account_id VARCHAR(255),
                    sender_account_name VARCHAR(255),
                    connection_note TEXT,
                    follow_up_message TEXT,
                    configured_at TIMESTAMP,
                    started_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_role_heyreach_campaign_role ON role_heyreach_campaigns(recruitment_role_id)"
            )
            for column_sql in (
                "ADD COLUMN IF NOT EXISTS li_scheduled_for TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS li_enrollment_claimed_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS li_enrollment_error TEXT",
                "ADD COLUMN IF NOT EXISTS li_enrolled_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS email_enrollment_claimed_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS email_enrollment_error TEXT",
                "ADD COLUMN IF NOT EXISTS email_enrolled_at TIMESTAMP",
            ):
                cur.execute(f"ALTER TABLE candidate_outreach {column_sql}")
    finally:
        try:
            conn.autocommit = old_autocommit
        except Exception:
            pass

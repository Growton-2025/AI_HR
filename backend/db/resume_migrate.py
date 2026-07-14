"""Idempotent schema changes for candidate resume storage and parsing."""


def ensure_resume_migrations(conn) -> None:
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
                CREATE TABLE IF NOT EXISTS candidate_resumes (
                    id                  SERIAL PRIMARY KEY,
                    candidate_id        INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                    filename            VARCHAR(512) NOT NULL,
                    content_type        VARCHAR(255) NOT NULL DEFAULT 'application/octet-stream',
                    size_bytes          INTEGER NOT NULL DEFAULT 0,
                    checksum_sha256     CHAR(64),
                    storage_backend     VARCHAR(32) NOT NULL DEFAULT 'azure_blob',
                    storage_key         TEXT,
                    file_bytes          BYTEA,
                    extracted_text      TEXT,
                    text_char_count     INTEGER NOT NULL DEFAULT 0,
                    parsed_json         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    summary             TEXT,
                    proposed_changes    JSONB NOT NULL DEFAULT '[]'::jsonb,
                    applied_fields      JSONB NOT NULL DEFAULT '[]'::jsonb,
                    parse_status        VARCHAR(32) NOT NULL DEFAULT 'pending',
                    parse_error         TEXT,
                    parse_model         VARCHAR(100),
                    is_current          BOOLEAN NOT NULL DEFAULT TRUE,
                    uploaded_by_user_id INTEGER REFERENCES users(id),
                    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                    parsed_at           TIMESTAMP,
                    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_candidate_resumes_candidate ON candidate_resumes(candidate_id)"
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_candidate_resumes_current
                    ON candidate_resumes(candidate_id) WHERE is_current
                """
            )
            # Keep upgrades safe if an earlier development build created only
            # part of the resume table.
            for column_sql in (
                "ADD COLUMN IF NOT EXISTS checksum_sha256 CHAR(64)",
                "ADD COLUMN IF NOT EXISTS storage_backend VARCHAR(32) NOT NULL DEFAULT 'azure_blob'",
                "ADD COLUMN IF NOT EXISTS storage_key TEXT",
                "ADD COLUMN IF NOT EXISTS file_bytes BYTEA",
                "ADD COLUMN IF NOT EXISTS extracted_text TEXT",
                "ADD COLUMN IF NOT EXISTS text_char_count INTEGER NOT NULL DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS parsed_json JSONB NOT NULL DEFAULT '{}'::jsonb",
                "ADD COLUMN IF NOT EXISTS summary TEXT",
                "ADD COLUMN IF NOT EXISTS proposed_changes JSONB NOT NULL DEFAULT '[]'::jsonb",
                "ADD COLUMN IF NOT EXISTS applied_fields JSONB NOT NULL DEFAULT '[]'::jsonb",
                "ADD COLUMN IF NOT EXISTS parse_status VARCHAR(32) NOT NULL DEFAULT 'pending'",
                "ADD COLUMN IF NOT EXISTS parse_error TEXT",
                "ADD COLUMN IF NOT EXISTS parse_model VARCHAR(100)",
                "ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT TRUE",
                "ADD COLUMN IF NOT EXISTS uploaded_by_user_id INTEGER REFERENCES users(id)",
                "ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP",
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT NOW()",
            ):
                cur.execute(f"ALTER TABLE candidate_resumes {column_sql}")
    finally:
        try:
            conn.autocommit = old_autocommit
        except Exception:
            pass

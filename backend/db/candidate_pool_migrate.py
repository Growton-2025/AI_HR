"""Idempotent PostgreSQL migrations for candidate pool ownership (existing DBs)."""
import logging

import psycopg2

from backend.services.linkedin_normalize import normalize_linkedin

logger = logging.getLogger(__name__)


# Must match values used in backend.services.candidate_pool (master catalog + recruiter upload).
_POOL_SOURCE_CHECK_LITERALS = (
    "legacy_master",
    "recruiter_upload",
    "admin_assigned",
    "catalog_from_upload",
)
_MIGRATION_LOCK_TIMEOUT_MS = 2000
_MIGRATION_STATEMENT_TIMEOUT_MS = 30000


def _pool_source_check_needs_refresh(definition: str) -> bool:
    """True if CHECK does not allow every pool_source the app writes (e.g. missing catalog_from_upload)."""
    if not definition:
        return True
    return any(lit not in definition for lit in _POOL_SOURCE_CHECK_LITERALS)


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _candidate_linkedin_needs_widen(cur) -> bool:
    cur.execute(
        """
        SELECT data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'candidates'
          AND column_name = 'linkedin'
        """,
    )
    row = cur.fetchone()
    if not row:
        return False
    data_type, max_len = row
    return data_type == "character varying" and max_len is not None and max_len < 1024


def ensure_candidate_pool_migrations(conn) -> None:
    if not conn:
        return
    # DDL + swallowed errors must not leave the connection in "aborted transaction" state.
    old_autocommit = getattr(conn, "autocommit", False)
    try:
        if not old_autocommit:
            try:
                conn.rollback()
            except Exception:
                pass
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = %s", (f"{_MIGRATION_LOCK_TIMEOUT_MS}ms",))
            cur.execute("SET statement_timeout = %s", (f"{_MIGRATION_STATEMENT_TIMEOUT_MS}ms",))
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_uploads (
                    id SERIAL PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id),
                    filename VARCHAR(512),
                    file_headers JSONB DEFAULT '[]',
                    mapping JSONB DEFAULT '{}',
                    row_count INTEGER DEFAULT 0,
                    inserted_count INTEGER DEFAULT 0,
                    updated_count INTEGER DEFAULT 0,
                    skipped_count INTEGER DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
                """
            )

            # Older DBs may have candidate_uploads without newer columns (CREATE IF NOT EXISTS skips ALTER).
            upload_cols = [
                ("filename", "VARCHAR(512)"),
                ("file_headers", "JSONB DEFAULT '[]'::jsonb"),
                ("mapping", "JSONB DEFAULT '{}'::jsonb"),
                ("row_count", "INTEGER DEFAULT 0"),
                ("inserted_count", "INTEGER DEFAULT 0"),
                ("updated_count", "INTEGER DEFAULT 0"),
                ("skipped_count", "INTEGER DEFAULT 0"),
                ("status", "VARCHAR(50) DEFAULT 'pending'"),
                ("error_message", "TEXT"),
                ("role_id", "INTEGER"),
                ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("completed_at", "TIMESTAMP"),
            ]
            for col, typ in upload_cols:
                if not _column_exists(cur, "candidate_uploads", col):
                    cur.execute(f"ALTER TABLE candidate_uploads ADD COLUMN {col} {typ};")

            if not _column_exists(cur, "users", "archived_at"):
                cur.execute("ALTER TABLE users ADD COLUMN archived_at TIMESTAMP;")

            if not _column_exists(cur, "recruitment_roles", "job_description"):
                cur.execute("ALTER TABLE recruitment_roles ADD COLUMN job_description TEXT;")

            cand_cols = [
                ("owner_user_id", "INTEGER REFERENCES users(id)"),
                ("pool_source", "VARCHAR(50) DEFAULT 'legacy_master'"),
                ("source_master_candidate_id", "INTEGER REFERENCES candidates(id)"),
                ("source_upload_id", "INTEGER"),
                ("source_upload_ids", "INTEGER[] DEFAULT '{}'::int[]"),
                ("assigned_by_user_id", "INTEGER REFERENCES users(id)"),
                ("normalized_linkedin", "TEXT"),
                ("is_archived", "BOOLEAN DEFAULT FALSE"),
            ]
            for col, typ in cand_cols:
                if not _column_exists(cur, "candidates", col):
                    cur.execute(f"ALTER TABLE candidates ADD COLUMN {col} {typ};")

            # Migrate existing source_upload_id to the new source_upload_ids array
            cur.execute("""
                UPDATE candidates 
                SET source_upload_ids = ARRAY[source_upload_id] 
                WHERE source_upload_id IS NOT NULL 
                  AND (source_upload_ids IS NULL OR array_length(source_upload_ids, 1) IS NULL);
            """)

            # Assign/import upserts expect these columns; older DBs may only have the base schema.
            legacy_contact_cols = [
                ("email", "VARCHAR(255)"),
                ("phone", "VARCHAR(50)"),
                ("mobile_phone", "VARCHAR(50)"),
                ("notes", "TEXT"),
                ("status", "VARCHAR(100) DEFAULT 'To be started'"),
            ]
            for col, typ in legacy_contact_cols:
                if not _column_exists(cur, "candidates", col):
                    cur.execute(f"ALTER TABLE candidates ADD COLUMN {col} {typ};")

            if _candidate_linkedin_needs_widen(cur):
                logger.info("Widening candidates.linkedin to VARCHAR(1024).")
                cur.execute(
                    "ALTER TABLE candidates ALTER COLUMN linkedin TYPE VARCHAR(1024);"
                )

            cur.execute(
                """
                SELECT id, linkedin FROM candidates
                WHERE linkedin IS NOT NULL AND normalized_linkedin IS NULL
                """
            )
            for cid, li in cur.fetchall():
                norm = normalize_linkedin(li)
                cur.execute(
                    "UPDATE candidates SET normalized_linkedin = %s WHERE id = %s",
                    (norm, cid),
                )

            cur.execute(
                """
                UPDATE candidates SET pool_source = 'legacy_master'
                WHERE pool_source IS NULL AND owner_user_id IS NULL;
                """
            )

            # CSV import writes master rows with catalog_from_upload and pool rows with
            # recruiter_upload. Older CHECK constraints may list only a subset (e.g. omit
            # catalog_from_upload) and still include recruiter_upload — refresh when any
            # required literal is missing.
            cur.execute(
                """
                SELECT pg_get_constraintdef(c.oid)
                FROM pg_constraint c
                JOIN pg_class t ON c.conrelid = t.oid
                JOIN pg_namespace n ON t.relnamespace = n.oid
                WHERE n.nspname = 'public' AND t.relname = 'candidates'
                  AND c.conname = 'candidates_pool_source_check'
                  AND c.contype = 'c'
                """
            )
            _psc_row = cur.fetchone()
            _psc_def = (_psc_row[0] if _psc_row else "") or ""
            if _pool_source_check_needs_refresh(_psc_def):
                cur.execute(
                    "ALTER TABLE candidates DROP CONSTRAINT IF EXISTS candidates_pool_source_check;"
                )
                try:
                    cur.execute(
                        """
                        ALTER TABLE candidates
                        ADD CONSTRAINT candidates_pool_source_check
                        CHECK (
                          pool_source IS NULL
                          OR pool_source IN (
                            'legacy_master',
                            'recruiter_upload',
                            'admin_assigned',
                            'catalog_from_upload'
                          )
                        );
                        """
                    )
                except psycopg2.Error as e:
                    logger.warning(
                        "Could not add candidates_pool_source_check: %s", e, exc_info=True
                    )

            cur.execute(
                """
                UPDATE candidates c
                SET normalized_linkedin = c.normalized_linkedin || '_legacy_' || c.id::text
                FROM (
                    SELECT normalized_linkedin, MIN(id) AS keep_id
                    FROM candidates
                    WHERE owner_user_id IS NULL AND normalized_linkedin IS NOT NULL
                    GROUP BY normalized_linkedin
                    HAVING COUNT(*) > 1
                ) d
                WHERE c.owner_user_id IS NULL
                  AND c.normalized_linkedin = d.normalized_linkedin
                  AND c.id != d.keep_id;
                """
            )

            cur.execute(
                "ALTER TABLE candidates DROP CONSTRAINT IF EXISTS candidates_linkedin_key;"
            )

            try:
                cur.execute(
                    """
                    ALTER TABLE candidates
                    ADD CONSTRAINT candidates_source_upload_id_fkey
                    FOREIGN KEY (source_upload_id) REFERENCES candidate_uploads(id);
                    """
                )
            except psycopg2.Error:
                pass

            cur.execute("DROP INDEX IF EXISTS uq_candidates_master_li;")
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_candidates_master_li
                ON candidates (normalized_linkedin)
                WHERE owner_user_id IS NULL AND normalized_linkedin IS NOT NULL
                  AND COALESCE(is_archived, FALSE) = FALSE;
                """
            )
            cur.execute("DROP INDEX IF EXISTS uq_candidates_recruiter_li;")
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_candidates_recruiter_li
                ON candidates (owner_user_id, normalized_linkedin)
                WHERE owner_user_id IS NOT NULL AND normalized_linkedin IS NOT NULL
                  AND COALESCE(is_archived, FALSE) = FALSE;
                """
            )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_candidates_owner ON candidates (owner_user_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_candidates_norm_li ON candidates (normalized_linkedin);"
            )

            logger.info("Candidate pool migrations applied.")
    except Exception as e:
        logger.error(f"Candidate pool migration failed: {e}")
        raise
    finally:
        try:
            if conn and not conn.closed:
                with conn.cursor() as cur:
                    cur.execute("RESET lock_timeout;")
                    cur.execute("RESET statement_timeout;")
        except Exception:
            pass
        try:
            conn.autocommit = old_autocommit
        except Exception:
            pass

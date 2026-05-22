"""Idempotent PostgreSQL migrations for Talent Pool AI columns."""
from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def ensure_ai_column_migrations(conn) -> None:
    if not conn:
        return

    old_autocommit = getattr(conn, "autocommit", False)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_column_definitions (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) NOT NULL,
                    owner_user_id INTEGER REFERENCES users(id),
                    view_scope VARCHAR(50) NOT NULL DEFAULT 'recruiter_pools',
                    recruiter_filter_id INTEGER REFERENCES users(id),
                    prompt_template TEXT NOT NULL,
                    mode VARCHAR(50) NOT NULL DEFAULT 'auto',
                    output_schema JSONB NOT NULL DEFAULT '[]'::jsonb,
                    required_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
                    only_run_if JSONB NOT NULL DEFAULT '{}'::jsonb,
                    context_inputs JSONB NOT NULL DEFAULT '{}'::jsonb,
                    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_column_runs (
                    id SERIAL PRIMARY KEY,
                    column_definition_id INTEGER NOT NULL REFERENCES ai_column_definitions(id) ON DELETE CASCADE,
                    selection_mode VARCHAR(50) NOT NULL,
                    selection_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    total INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    status VARCHAR(50) NOT NULL DEFAULT 'queued',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_column_cells (
                    id SERIAL PRIMARY KEY,
                    column_definition_id INTEGER NOT NULL REFERENCES ai_column_definitions(id) ON DELETE CASCADE,
                    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                    primary_output TEXT,
                    outputs JSONB NOT NULL DEFAULT '{}'::jsonb,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status VARCHAR(50) NOT NULL DEFAULT 'idle',
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    last_run_id INTEGER REFERENCES ai_column_runs(id) ON DELETE SET NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(column_definition_id, candidate_id)
                );
                """
            )

            defs_cols = [
                ("slug", "VARCHAR(255) DEFAULT ''"),
                ("owner_user_id", "INTEGER REFERENCES users(id)"),
                ("view_scope", "VARCHAR(50) NOT NULL DEFAULT 'recruiter_pools'"),
                ("recruiter_filter_id", "INTEGER REFERENCES users(id)"),
                ("prompt_template", "TEXT DEFAULT ''"),
                ("mode", "VARCHAR(50) NOT NULL DEFAULT 'auto'"),
                ("output_schema", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
                ("required_fields", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
                ("only_run_if", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
                ("context_inputs", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
                ("is_archived", "BOOLEAN NOT NULL DEFAULT FALSE"),
                ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ]
            for col, typ in defs_cols:
                if not _column_exists(cur, "ai_column_definitions", col):
                    cur.execute(f"ALTER TABLE ai_column_definitions ADD COLUMN {col} {typ};")

            runs_cols = [
                ("selection_mode", "VARCHAR(50) NOT NULL DEFAULT 'selected_ids'"),
                ("selection_payload", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
                ("total", "INTEGER NOT NULL DEFAULT 0"),
                ("completed", "INTEGER NOT NULL DEFAULT 0"),
                ("failed", "INTEGER NOT NULL DEFAULT 0"),
                ("skipped", "INTEGER NOT NULL DEFAULT 0"),
                ("status", "VARCHAR(50) NOT NULL DEFAULT 'queued'"),
                ("started_at", "TIMESTAMP"),
                ("completed_at", "TIMESTAMP"),
                ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ]
            for col, typ in runs_cols:
                if not _column_exists(cur, "ai_column_runs", col):
                    cur.execute(f"ALTER TABLE ai_column_runs ADD COLUMN {col} {typ};")

            cells_cols = [
                ("primary_output", "TEXT"),
                ("outputs", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
                ("details", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
                ("status", "VARCHAR(50) NOT NULL DEFAULT 'idle'"),
                ("error_message", "TEXT"),
                ("started_at", "TIMESTAMP"),
                ("completed_at", "TIMESTAMP"),
                ("last_run_id", "INTEGER REFERENCES ai_column_runs(id) ON DELETE SET NULL"),
                ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ]
            for col, typ in cells_cols:
                if not _column_exists(cur, "ai_column_cells", col):
                    cur.execute(f"ALTER TABLE ai_column_cells ADD COLUMN {col} {typ};")

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_column_definitions_slug_scope
                ON ai_column_definitions (
                    COALESCE(owner_user_id, 0),
                    view_scope,
                    COALESCE(recruiter_filter_id, 0),
                    slug
                )
                WHERE COALESCE(is_archived, FALSE) = FALSE;
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_ai_column_runs_definition
                ON ai_column_runs(column_definition_id, created_at DESC);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_ai_column_cells_candidate
                ON ai_column_cells(candidate_id, column_definition_id);
                """
            )

            # Reset the legacy AI-column storage so the new system starts from a clean baseline.
            cur.execute(
                """
                UPDATE candidates
                SET raw_fields = raw_fields - 'ai_columns'
                WHERE raw_fields ? 'ai_columns';
                """
            )
            logger.info("AI column migrations applied.")
    except Exception as exc:
        logger.error("AI column migration failed: %s", exc, exc_info=True)
        raise
    finally:
        try:
            conn.autocommit = old_autocommit
        except Exception:
            pass

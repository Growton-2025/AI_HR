"""A record of who changed a candidate's status, and to what.

Every question that could not be answered while investigating "one call, two
completed entries" was blocked by the absence of this table. Shreya Shroff's
"High CTC" turned into "Not Interested" and the only way to reconstruct that
was to read a synthetic outcome string off a call row and compare timestamps.
Sravani G — closed twice as Not Interested and then Shared with customer —
now reads the active status "Reached out - Phone", and there is simply no way
to find out what changed it.

Deliberately append-only and best effort: a failure to log must never block the
status change the recruiter asked for.
"""

import logging

logger = logging.getLogger(__name__)

# Free-text, so a new caller does not need a migration to describe itself.
SOURCE_STATUS_DROPDOWN = "status_dropdown"
SOURCE_BULK_STATUS = "bulk_status_update"
SOURCE_CALL_OUTCOME = "call_outcome"

_SCHEMA_READY = False


def ensure_status_log_schema(cur) -> None:
    """Create the table on first use. Idempotent, like the other migrations."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_status_history (
            id           SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL,
            old_status   VARCHAR(120),
            new_status   VARCHAR(120),
            changed_by   VARCHAR(255),
            source       VARCHAR(60),
            changed_at   TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candidate_status_history_candidate
            ON candidate_status_history (candidate_id, changed_at DESC)
        """
    )
    _SCHEMA_READY = True


def record_status_change(cur, candidate_id, old_status, new_status, changed_by, source) -> None:
    """Append one status change. Never raises."""
    try:
        ensure_status_log_schema(cur)
        cur.execute(
            """
            INSERT INTO candidate_status_history
                (candidate_id, old_status, new_status, changed_by, source)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (int(candidate_id), old_status or None, new_status or None,
             (changed_by or "").strip().lower() or None, source),
        )
    except Exception as exc:
        logger.warning(
            "Could not log status change for candidate %s (%s -> %s): %s",
            candidate_id, old_status, new_status, exc,
        )


def record_status_changes(cur, candidate_ids, new_status, changed_by, source) -> None:
    """Bulk variant: reads the old values first so the log stays meaningful."""
    ids = sorted({int(cid) for cid in (candidate_ids or [])})
    if not ids:
        return
    try:
        ensure_status_log_schema(cur)
        cur.execute(
            "SELECT id, status FROM candidates WHERE id = ANY(%s::int[])", (ids,)
        )
        previous = dict(cur.fetchall())
    except Exception as exc:
        logger.warning("Could not read previous statuses for the status log: %s", exc)
        previous = {}
    for candidate_id in ids:
        record_status_change(
            cur, candidate_id, previous.get(candidate_id), new_status, changed_by, source,
        )


def fetch_status_history(cur, candidate_id, limit: int = 50) -> list:
    """Newest first. Empty when the table has not been created yet."""
    try:
        cur.execute(
            """
            SELECT old_status, new_status, changed_by, source, changed_at
            FROM candidate_status_history
            WHERE candidate_id = %s
            ORDER BY changed_at DESC
            LIMIT %s
            """,
            (int(candidate_id), int(limit)),
        )
        return [
            {
                "old_status": row[0],
                "new_status": row[1],
                "changed_by": row[2],
                "source": row[3],
                "changed_at": row[4].isoformat() if row[4] else None,
            }
            for row in cur.fetchall()
        ]
    except Exception as exc:
        logger.warning("Could not read status history for candidate %s: %s", candidate_id, exc)
        return []

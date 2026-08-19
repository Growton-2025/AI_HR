import logging

logger = logging.getLogger(__name__)

def sync_shortlisted_to_call_list(cur, role_id: int, candidate_ids: list[int]):
    """
    If the role is linked to an auto-created call list, 
    push all phone-enriched candidates into that call list.
    """
    if not candidate_ids:
        return

    try:
        # Check if role has a linked call list
        cur.execute(
            "SELECT linked_call_list_id FROM recruitment_roles WHERE id = %s",
            (role_id,)
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return
        
        linked_call_list_id = row[0]

        # Get candidates with valid phone numbers who are shortlisted
        cur.execute(
            """
            SELECT id FROM candidates 
            WHERE id = ANY(%s) 
            AND status = 'Shortlisted'
            AND (
                NULLIF(TRIM(phone), '') IS NOT NULL 
                OR NULLIF(TRIM(mobile_phone), '') IS NOT NULL
            )
            """,
            (candidate_ids,)
        )
        enriched_candidate_ids = [r[0] for r in cur.fetchall()]

        if not enriched_candidate_ids:
            return

        # Insert them into the call list (ignoring if they already exist).
        # Import here to avoid a circular import at module load time.
        from backend.api.routes.calls import FIRST_ATTEMPT_TITLE

        cur.execute(
            """
            INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
            SELECT DISTINCT c_id, %s, 'pending', CURRENT_DATE, %s
            FROM UNNEST(%s::int[]) AS c_id
            WHERE NOT EXISTS (
                SELECT 1 FROM calls existing
                WHERE existing.candidate_id = c_id AND existing.list_id = %s
            )
            -- …and not already being called from a DIFFERENT list. A candidate
            -- shortlisted for two roles used to get a parallel "Call 1 - Day 1"
            -- in each list: dialled twice for two jobs, and on retirement a
            -- completed entry appeared in every list, reading as several calls
            -- when only one was made.
            AND NOT EXISTS (
                SELECT 1 FROM calls other
                WHERE other.candidate_id = c_id
                  AND other.list_id <> %s
                  AND other.status IN ('pending', 'in_progress')
            )
            ON CONFLICT (candidate_id, list_id) WHERE status = 'pending' DO NOTHING
            RETURNING candidate_id
            """,
            (linked_call_list_id, FIRST_ATTEMPT_TITLE, enriched_candidate_ids,
             linked_call_list_id, linked_call_list_id)
        )
        added = cur.rowcount or 0
        skipped = len(enriched_candidate_ids) - added
        logger.info(
            f"Auto-synced {added} candidates to call list {linked_call_list_id} for role {role_id}"
            + (f"; skipped {skipped} already in a list" if skipped > 0 else "")
        )
        
    except Exception as e:
        logger.error(f"Failed to auto-sync candidates to call list for role {role_id}: {e}")

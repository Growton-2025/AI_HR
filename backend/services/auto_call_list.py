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

        # Insert them into the call list (ignoring if they already exist)
        cur.execute(
            """
            INSERT INTO calls (candidate_id, list_id, status, due_date, task_title)
            SELECT DISTINCT c_id, %s, 'pending', CURRENT_DATE, 'Call 1 - Day 1'
            FROM UNNEST(%s::int[]) AS c_id
            WHERE NOT EXISTS (
                SELECT 1 FROM calls existing
                WHERE existing.candidate_id = c_id AND existing.list_id = %s
            )
            """,
            (linked_call_list_id, enriched_candidate_ids, linked_call_list_id)
        )
        logger.info(f"Auto-synced {len(enriched_candidate_ids)} candidates to call list {linked_call_list_id} for role {role_id}.")
        
    except Exception as e:
        logger.error(f"Failed to auto-sync candidates to call list for role {role_id}: {e}")

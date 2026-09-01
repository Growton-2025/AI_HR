"""Durable Smartlead enrollment for shortlisted role candidates."""

import re
import threading

from backend.db.connection import get_db_connection_context
from backend.integrations.smartlead import SmartleadBot

_stop = threading.Event()
_thread = None


def _valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))


def _render(value: str, role_name: str, first_name: str, last_name: str) -> str:
    return (
        (value or "")
        .replace("{{role_name}}", role_name or "")
        .replace("{{first_name}}", first_name or "")
        .replace("{{last_name}}", last_name or "")
    )


def dispatch_due_email(limit: int = 20) -> int:
    claimed = []
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT co.candidate_id, co.recruitment_role_id, sc.campaign_id,
                       sc.campaign_name, sc.initial_body, sc.started_at,
                       c.first_name, c.last_name, c.name, c.email
                FROM candidate_outreach co
                JOIN candidates c ON c.id=co.candidate_id
                JOIN role_smartlead_campaigns sc ON sc.recruitment_role_id=co.recruitment_role_id
                WHERE (
                    co.status IN ('scheduled', 'waiting_for_email')
                    OR (co.status='email_enrolling' AND co.email_enrollment_claimed_at < NOW() - INTERVAL '10 minutes')
                    OR (co.status='failed' AND co.updated_at < NOW() - INTERVAL '5 minutes')
                )
                  AND co.email_enrolled_at IS NULL
                  AND sc.provisioning_status='configured'
                  AND c.email IS NOT NULL AND TRIM(c.email) <> ''
                ORDER BY co.updated_at
                FOR UPDATE OF co SKIP LOCKED
                LIMIT %s
                """,
                (limit,),
            )
            claimed = [row for row in cur.fetchall() if _valid_email(row[9])]
            for row in claimed:
                cur.execute(
                    """
                    UPDATE candidate_outreach SET status='email_enrolling',
                        email_enrollment_claimed_at=NOW(), email_enrollment_error=NULL, updated_at=NOW()
                    WHERE candidate_id=%s AND recruitment_role_id=%s
                    """,
                    (row[0], row[1]),
                )
            conn.commit()

    ai_columns_by_candidate = {}
    if claimed:
        roles = {row[1] for row in claimed}
        c_ids = {row[0] for row in claimed}
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    for r_id in roles:
                        view_scope = f"role_{r_id}"
                        cur.execute(
                            """
                            SELECT c.candidate_id, d.name, c.primary_output
                            FROM ai_column_cells c
                            JOIN ai_column_definitions d ON d.id = c.column_definition_id
                            WHERE d.view_scope = %s AND c.candidate_id = ANY(%s)
                            """,
                            (view_scope, list(c_ids))
                        )
                        for cid, col_name, val in cur.fetchall():
                            if val:
                                ai_columns_by_candidate.setdefault(cid, {})[col_name] = val

    completed = 0
    for candidate_id, role_id, campaign_id, campaign_name, body, started_at, first_name, last_name, name, email in claimed:
        try:
            resolved_first = first_name or (name or "Candidate").split()[0]
            resolved_last = last_name or " ".join((name or "").split()[1:])
            bot = SmartleadBot()
            bot.campaign_id = int(campaign_id)
            
            lead_data = {"first_name": resolved_first, "last_name": resolved_last, "email": email.strip()}
            ai_data = ai_columns_by_candidate.get(candidate_id, {})
            lead_data.update(ai_data)
            
            result = bot.add_leads([lead_data])
            if result is None:
                raise RuntimeError("Smartlead rejected the candidate")
            if isinstance(result, dict):
                # Smartlead returns HTTP 200 even when the lead is silently
                # dropped (block list, invalid email, lead limit, ...).
                added = (result.get("total_leads") or 0) + (result.get("already_added_to_campaign") or 0)
                if added <= 0:
                    reasons = {
                        key: result.get(key)
                        for key in ("block_count", "invalid_email_count", "bounce_count", "duplicate_count", "is_lead_limit_exhausted", "lead_import_stopped_count")
                        if result.get(key)
                    }
                    raise RuntimeError(f"Smartlead did not add the lead: {reasons or result}")
            if not started_at and bot.start_campaign() is None:
                raise RuntimeError("Candidate enrolled, but Smartlead campaign could not start")
            initial_message = _render(body, campaign_name, resolved_first, resolved_last)
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE candidate_outreach
                        SET campaign_id=%s, campaign_name=%s, status='in_campaign',
                            initial_message=%s, initial_message_at=COALESCE(initial_message_at,NOW()),
                            email_enrolled_at=NOW(),
                            -- The reply poller ranks by conversation activity; without a
                            -- send stamp a freshly enrolled lead looks stale and falls out
                            -- of the priority tier into the slow round-robin sweep.
                            last_message_sent_at=NOW(), updated_at=NOW()
                        WHERE candidate_id=%s AND recruitment_role_id=%s AND email_enrolled_at IS NULL
                        """,
                        (str(campaign_id), campaign_name, initial_message, candidate_id, role_id),
                    )
                    cur.execute(
                        """UPDATE role_smartlead_campaigns
                           SET started_at=COALESCE(started_at,NOW()), updated_at=NOW()
                           WHERE recruitment_role_id=%s""",
                        (role_id,),
                    )
                    conn.commit()
            completed += 1
        except Exception as exc:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE candidate_outreach SET status='failed',
                                email_enrollment_error=%s, updated_at=NOW()
                            WHERE candidate_id=%s AND recruitment_role_id=%s AND email_enrolled_at IS NULL
                            """,
                            (str(exc)[:1000], candidate_id, role_id),
                        )
                        conn.commit()
    return completed


def _loop():
    while not _stop.wait(5):
        try:
            dispatch_due_email()
        except Exception as exc:
            print(f"SMARTLEAD DISPATCHER ERROR: {exc}")


def start_dispatcher():
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="smartlead-role-dispatcher", daemon=True)
    _thread.start()


def stop_dispatcher():
    _stop.set()

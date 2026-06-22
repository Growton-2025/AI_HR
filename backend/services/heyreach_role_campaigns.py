"""Durable, role-scoped HeyReach campaign enrollment."""

import threading
import time
from datetime import datetime, timedelta

from backend.db.connection import get_db_connection_context
from backend.integrations.heyreach import HeyReachBot

_stop = threading.Event()
_thread = None


def dispatch_due_linkedin(limit: int = 20) -> int:
    claimed = []
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            return 0
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT co.candidate_id, co.recruitment_role_id, hc.campaign_id,
                       hc.sender_account_id, c.first_name, c.last_name, c.name, c.linkedin,
                       hc.follow_up_message, hc.started_at, co.li_status
                FROM candidate_outreach co
                JOIN candidates c ON c.id=co.candidate_id
                JOIN role_heyreach_campaigns hc ON hc.recruitment_role_id=co.recruitment_role_id
                WHERE ((co.li_status='scheduled' AND co.li_scheduled_for <= NOW())
                    OR (co.li_status='enrolling' AND co.li_enrollment_claimed_at < NOW() - INTERVAL '10 minutes')
                    OR (co.li_status='failed' AND co.updated_at < NOW() - INTERVAL '5 minutes'))
                  AND co.li_enrolled_at IS NULL
                  AND hc.provisioning_status='configured'
                ORDER BY co.li_scheduled_for
                FOR UPDATE OF co SKIP LOCKED
                LIMIT %s
                """, (limit,)
            )
            claimed = cur.fetchall()
            for row in claimed:
                cur.execute(
                    """UPDATE candidate_outreach SET li_status='enrolling',
                       li_enrollment_claimed_at=NOW(), li_enrollment_error=NULL, updated_at=NOW()
                       WHERE candidate_id=%s AND recruitment_role_id=%s""",
                    (row[0], row[1]),
                )
            conn.commit()

    bot = HeyReachBot()
    completed = 0
    for candidate_id, role_id, campaign_id, account_id, first_name, last_name, name, linkedin, message, started_at, claimed_from_status in claimed:
        try:
            result = None
            # Recovery after a worker/database interruption: check the external
            # campaign before retrying the side effect.
            if claimed_from_status == "enrolling":
                target = bot._normalize_linkedin_url(linkedin)
                leads = bot.get_campaign_leads(int(campaign_id)) or []
                for entry in leads:
                    profile = entry.get("linkedInUserProfile") or entry.get("lead") or entry
                    remote_url = profile.get("profileUrl") or profile.get("profile_url")
                    if bot._normalize_linkedin_url(remote_url) == target:
                        result = {"alreadyEnrolled": True}
                        break
            if result is None:
                result = bot.push_lead(
                    int(campaign_id), int(account_id),
                    first_name or (name or "Candidate").split()[0],
                    last_name or " ".join((name or "").split()[1:]), linkedin,
                )
            if result is None:
                raise RuntimeError("HeyReach rejected the candidate")
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE candidate_outreach SET heyreach_campaign_id=%s,
                           li_account_id=%s, li_status='in_campaign', initial_li_message=%s,
                           initial_li_message_at=COALESCE(initial_li_message_at,NOW()),
                           li_enrolled_at=NOW(), li_last_action_at=NOW(), updated_at=NOW()
                           WHERE candidate_id=%s AND recruitment_role_id=%s AND li_enrolled_at IS NULL""",
                        (str(campaign_id), str(account_id), message, candidate_id, role_id),
                    )
                    cur.execute("UPDATE role_heyreach_campaigns SET started_at=COALESCE(started_at,NOW()),updated_at=NOW() WHERE recruitment_role_id=%s", (role_id,))
                    conn.commit()
            completed += 1
        except Exception as exc:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """UPDATE candidate_outreach SET li_status='failed',
                               li_enrollment_error=%s, updated_at=NOW()
                               WHERE candidate_id=%s AND recruitment_role_id=%s AND li_enrolled_at IS NULL""",
                            (str(exc)[:1000], candidate_id, role_id),
                        )
                        conn.commit()
    return completed


def _loop():
    while not _stop.wait(5):
        try:
            dispatch_due_linkedin()
        except Exception as exc:
            print(f"HEYREACH DISPATCHER ERROR: {exc}")


def start_dispatcher():
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="heyreach-role-dispatcher", daemon=True)
    _thread.start()


def stop_dispatcher():
    _stop.set()

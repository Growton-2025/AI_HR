import os
import re
import json
import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta, timezone as tz_module
from typing import List, Optional, Dict
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from backend.api import schemas, deps
from backend.db.connection import (
    get_db_connection_context,
)
from backend.integrations.smartlead import SmartleadBot, CampaignNotFoundError
from backend.integrations.heyreach import HeyReachBot
from backend.services.linkedin_normalize import normalize_linkedin
from backend.services.role_campaigns import (
    campaign_payload,
    fetch_role_campaign,
    provision_role_campaign,
)
from backend.services.role_activation import fetch_role_activation
from backend.services.heyreach_role_campaigns import dispatch_due_linkedin
from backend.services.smartlead_role_dispatcher import dispatch_due_email

router = APIRouter()
# This module otherwise uses print(); webhooks need real log records so
# delivery problems are visible in the hosted log stream.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory LinkedIn chat cache
# Structure: {candidate_id: {messages, ts, refreshing}}
# Messages are cached per candidate to avoid redundant API calls
_li_chat_cache: Dict[int, Dict] = {}
_li_chat_lock = threading.Lock()
_LI_CACHE_TTL = 300
_LI_CACHE_STALE_THRESHOLD = 45

# In-memory Email chat cache
_email_chat_cache: Dict[int, Dict] = {}
_email_chat_lock = threading.Lock()
_EMAIL_CACHE_TTL = 3600        # 1 hour for emails (last longer than LI)
# Polls are served instantly from memory; a background Smartlead sync fires
# when the cache is older than this. 60s keeps new replies near-real-time
# without the former poll-storm (a sync every 5s per open modal).
_EMAIL_CACHE_STALE_THRESHOLD = 60

_smartlead_accounts_cache: Dict[str, object] = {"accounts": [], "ts": 0.0}
_SMARTLEAD_ACCOUNTS_TTL = 300



def _update_outreach_identifiers(candidate_id: int, conv_id: str, acc_id: int):
    """Update identifiers in database if they changed."""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                fields = []
                params = []
                if conv_id:
                    fields.append("li_conversation_id = %s")
                    params.append(str(conv_id))
                if acc_id:
                    fields.append("li_account_id = %s")
                    params.append(str(acc_id))

                if fields:
                    params.append(candidate_id)
                    cur.execute(
                        f"UPDATE candidate_outreach SET {', '.join(fields)} WHERE candidate_id = %s",
                        tuple(params),
                    )
                    conn.commit()
                    print(f"DEBUG: Auto-corrected identifiers for cand {candidate_id}")
    except Exception as e:
        print(f"WARNING: ID update failed: {e}")

def _sync_li_messages(
    candidate_id: int,
    profile_url: str,
    campaign_id: Optional[int],
    conversation_id: Optional[str],
    account_id: Optional[int] = None,
) -> List[Dict]:
    """
    Core logic to fetch fresh messages from HeyReach and update the cache.
    Returns the refreshed message list.
    """
    bot = HeyReachBot()
    messages = []
    try:
        res_data = bot.get_li_chat_history(
            profile_url,
            campaign_id=campaign_id,
            conversation_id=conversation_id,
            account_id=account_id,
        )
        messages = res_data.get("messages", [])
        new_conv_id = res_data.get("conversation_id")
        new_acc_id = res_data.get("account_id")

        # Persistent Auto-Correction: If IDs recovered/changed, save to DB
        if (new_conv_id and new_conv_id != conversation_id) or (new_acc_id and str(new_acc_id) != str(account_id)):
             _update_outreach_identifiers(candidate_id, new_conv_id, new_acc_id)

        # Clean messages before caching
        if messages:
            for msg in messages:
                raw_body = msg.get("email_body", "")
                if raw_body:
                    msg["email_body"] = _clean_email_body(raw_body)
    except Exception as e:
        print(f"WARNING: HeyReach sync failed for cand {candidate_id}: {e}")

    with _li_chat_lock:
        # Persistent Cache Protection: If the new fetch is empty, keep old messages
        final_messages = messages
        if not final_messages and candidate_id in _li_chat_cache:
            old_messages = _li_chat_cache[candidate_id].get("messages", [])
            if old_messages:
                print(f"DEBUG: Preserving {len(old_messages)} existing messages for cand {candidate_id} after empty/failed sync.")
                final_messages = old_messages

        previous_li_entry = _li_chat_cache.get(candidate_id) or {}
        new_li_entry = {
            "messages": final_messages or [],
            "ts": time.monotonic(),
            "refreshing": False,
            "db_updated_at": datetime.now(tz_module.utc),
        }
        # Preserve the initial LI message so hot-path responses keep showing it.
        if "initial" in previous_li_entry:
            new_li_entry["initial"] = previous_li_entry.get("initial")
            new_li_entry["initial_at"] = previous_li_entry.get("initial_at")
        _li_chat_cache[candidate_id] = new_li_entry

        # ── PERSISTENT DB CACHE ──────────────────────────────────────────────
        # Save the fetched history to DB so it lives across restarts
        if final_messages:
            # When this live fetch contains a reply, promote it into the columns
            # the candidate LIST reads (li_status / li_response_text). Without
            # this the modal could show a reply pulled straight from HeyReach
            # while the list kept saying "No response yet" indefinitely — the
            # two views read different sources, and only the manual "Sync
            # Responses" button ever wrote these columns.
            #
            # Folded into the existing UPDATE rather than a second statement:
            # connections to this database cost ~2s to establish and ~370ms per
            # round trip, so an extra query here would be expensive.
            newest_reply = max(
                (m for m in final_messages
                 if str(m.get("direction") or "").lower() == "inbound"),
                key=lambda m: str(m.get("time") or ""),
                default=None,
            )
            reply_text = str((newest_reply or {}).get("email_body") or "").strip()
            has_reply = bool(reply_text)
            reply_at = (newest_reply or {}).get("time") or None

            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                    if not conn:
                        raise RuntimeError("Database connection failed while persisting LinkedIn cache")
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE candidate_outreach
                            SET li_chat_history_cache = %s,
                                li_chat_history_updated_at = %s,
                                -- Only ever promote; never clear a stored reply
                                -- from a fetch that happens to come back empty.
                                li_status = CASE WHEN %s THEN 'replied' ELSE li_status END,
                                li_response_text = CASE WHEN %s THEN %s ELSE li_response_text END,
                                li_response_received_at = CASE
                                    WHEN %s THEN COALESCE(%s::timestamptz, li_response_received_at)
                                    ELSE li_response_received_at END
                            WHERE candidate_id = %s
                        """,
                            (
                                json.dumps(final_messages), datetime.now(tz_module.utc),
                                has_reply,
                                has_reply, reply_text or None,
                                has_reply, reply_at,
                                candidate_id,
                            ),
                        )
                        conn.commit()
            except Exception as db_err:
                print(f"WARNING: Failed to persist LI cache to DB: {db_err}")

    print(f"DEBUG: LI cache refreshed for cand {candidate_id}: {len(final_messages or [])} msgs")
    return final_messages or []

def _refresh_li_cache_task(
    candidate_id: int,
    profile_url: str,
    campaign_id: Optional[int],
    conversation_id: Optional[str],
    account_id: Optional[int] = None,
):
    """Background task wrapper."""
    # No longer skipping background sync due to strict filter
    _sync_li_messages(candidate_id, profile_url, campaign_id, conversation_id, account_id)


# ─────────────────────────────────────────────────────────────────────────────


def _sync_email_messages(candidate_id: int, email: str, campaign_id: str) -> List[Dict]:
    """Fetch fresh emails from Smartlead and update DB/memory cache."""
    bot = get_smartlead_bot()
    messages = []
    try:
        messages = bot.get_chat_history(email, campaign_id)
        if messages and isinstance(messages, list):
            for msg in messages:
                raw_body = msg.get("email_body", "")
                if raw_body:
                    msg["email_body"] = _clean_email_body(raw_body)
    except Exception as e:
        print(f"WARNING: Smartlead sync failed for cand {candidate_id}: {e}")

    with _email_chat_lock:
        final_messages = messages
        # If fetch failed/empty, preserve old cache if possible
        previous_entry = _email_chat_cache.get(candidate_id) or {}
        if not final_messages:
            final_messages = previous_entry.get("messages", [])

        new_entry = {
            "messages": final_messages or [],
            "ts": time.monotonic(),
            "refreshing": False,
            "db_updated_at": datetime.now(tz_module.utc),
        }
        # Preserve the sanitized initial message so the hot path keeps working.
        if "initial" in previous_entry:
            new_entry["initial"] = previous_entry.get("initial")
            new_entry["initial_at"] = previous_entry.get("initial_at")
        _email_chat_cache[candidate_id] = new_entry

        # Persist to DB
        if final_messages:
            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                    if not conn:
                        raise RuntimeError("Database connection failed while persisting email cache")
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE candidate_outreach
                            SET email_chat_history_cache = %s,
                                email_chat_history_updated_at = %s
                            WHERE candidate_id = %s AND (campaign_id = %s OR %s IS NULL)
                        """,
                            (
                                json.dumps(final_messages),
                                datetime.now(tz_module.utc),
                                candidate_id,
                                campaign_id,
                                campaign_id,
                            ),
                        )
                        conn.commit()
            except Exception as db_err:
                print(f"WARNING: Failed to persist Email cache to DB: {db_err}")

    return final_messages or []


def _refresh_email_cache_task(candidate_id: int, email: str, campaign_id: str):
    _sync_email_messages(candidate_id, email, campaign_id)


def _prewarm_single(candidate_id: int):
    """Fetch outreach info from DB and pre-warm LinkedIn chat cache for one candidate."""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
                cand_row = cur.fetchone()
                if not cand_row or not cand_row[0]:
                    return
                profile_url = cand_row[0]
                cur.execute(
                    """
                    SELECT heyreach_campaign_id, li_conversation_id, li_account_id
                    FROM candidate_outreach
                    WHERE candidate_id = %s
                    ORDER BY updated_at DESC LIMIT 1
                """,
                    (candidate_id,),
                )
                row = cur.fetchone()
        campaign_id = int(row[0]) if row and row[0] else None
        conv_id = row[1] if row and len(row) > 1 else None
        acc_id = int(row[2]) if row and len(row) > 2 and row[2] else None

        # Only pre-warm if not already cached / refreshing
        with _li_chat_lock:
            cached = _li_chat_cache.get(candidate_id)
            if (
                cached
                and not cached.get("refreshing", False)
                and (time.monotonic() - cached["ts"]) < _LI_CACHE_TTL
            ):
                return  # Already fresh
            if cached and cached.get("refreshing", False):
                return  # Already refreshing

        _sync_li_messages(candidate_id, profile_url, campaign_id, conv_id, acc_id)
    except Exception as e:
        print(f"WARNING: prewarm failed for cand {candidate_id}: {e}")


# --- Request/Response Models ---


# ── Startup & Bulk Warming ────────────────────────────────────────────────────

def bulk_load_chat_caches(rows: List[tuple]):
    """
    Populates in-memory caches from DB rows.
    Row structure must include: candidate_id, li_cache, email_cache
    """
    li_count = 0
    email_count = 0
    now = time.monotonic()

    with _li_chat_lock:
        with _email_chat_lock:
            for row in rows:
                cid, li_cache, email_cache = row
                if li_cache:
                    if isinstance(li_cache, str): li_cache = json.loads(li_cache)
                    _li_chat_cache[cid] = {
                        "messages": li_cache,
                        "ts": now - (_LI_CACHE_TTL / 2),
                        "refreshing": False
                    }
                    li_count += 1
                if email_cache:
                    if isinstance(email_cache, str): email_cache = json.loads(email_cache)
                    _email_chat_cache[cid] = {
                        "messages": email_cache,
                        "ts": now - (_EMAIL_CACHE_TTL / 2),
                        "refreshing": False
                    }
                    email_count += 1

    print(f"DEBUG: Bulk-warmed {li_count} LI chats and {email_count} Email chats into memory.")

def _startup_prewarm():
    """Run once at startup to warm the LI cache for known active candidates."""
    # This will now be handled by bulk_load_chat_caches called from query.py
    pass


_startup_thread = threading.Thread(target=_startup_prewarm, daemon=True)
_startup_thread.start()
# ─────────────────────────────────────────────────────────────────────────────


class OutreachTriggerRequest(BaseModel):
    candidate_ids: List[int]
    role_id: int
    role_name: str


class PrewarmRequest(BaseModel):
    candidate_ids: List[int]


class OutreachStatusResponse(BaseModel):
    candidate_id: int
    status: str
    message_sent_count: int
    last_message_sent_at: Optional[datetime]
    response_received_at: Optional[datetime]
    response_text: Optional[str]
    li_status: Optional[str]
    li_last_action_at: Optional[datetime]
    li_response_text: Optional[str]
    li_sent_count: Optional[int] = 0
    li_response_received_at: Optional[datetime] = None
    li_conversation_id: Optional[str] = None


class HeyReachTriggerRequest(BaseModel):
    candidate_ids: List[int]
    role_id: Optional[int] = 0
    role_name: Optional[str] = None
    campaign_id: int
    sender_account_id: int


class ShortlistOutreachRequest(BaseModel):
    hr_campaign_id: Optional[int] = None  # HeyReach campaign ID (falls back to env var)
    sender_account_id: Optional[int] = (
        None  # HeyReach sender account ID (falls back to env var)
    )


class RoleEmailSetupRequest(BaseModel):
    sender_account_id: int
    campaign_id: int


class BulkRoleShortlistRequest(BaseModel):
    assignments: List[schemas.AssignmentDetail]


def _dispatch_role_outreach_now():
    """Best-effort fast path; lifecycle workers retain the durable retry path."""
    dispatch_due_email()
    dispatch_due_linkedin()


def _refresh_role_shortlist_caches(candidate_id: int, role_id: int):
    """Refresh shared caches after the shortlist response has been returned."""
    try:
        from backend.pipeline import query
        query.refresh_profiles_in_cache([candidate_id])
        from backend.api.routes.roles import (
            invalidate_role_detail_cache,
            invalidate_role_detail_cache_for_candidate,
        )
        from backend.api.routes.candidates import invalidate_candidate_count_caches
        invalidate_role_detail_cache(role_id)
        invalidate_role_detail_cache_for_candidate(candidate_id)
        invalidate_candidate_count_caches()
    except Exception as exc:
        print(f"SHORTLIST CACHE REFRESH WARNING: {exc}")


def _refresh_bulk_role_shortlist_caches(candidate_ids: List[int], role_id: int):
    try:
        from backend.pipeline import query
        query.refresh_profiles_in_cache(candidate_ids)
        from backend.api.routes.roles import invalidate_role_detail_cache
        from backend.api.routes.candidates import invalidate_candidate_count_caches
        invalidate_role_detail_cache(role_id)
        invalidate_candidate_count_caches()
    except Exception as exc:
        print(f"BULK SHORTLIST CACHE REFRESH WARNING: {exc}")


# --- Hardcoded Email Template ---
EMAIL_TEMPLATE = {
    "subject": "Exciting Opportunity at {role_name}",
    "body": """Hi {{first_name}},

I came across your profile and was impressed by your experience. We're currently hiring for a {role_name} position that I believe would be a great fit for your background.

Would you be open to a quick conversation to learn more?

Best regards,
Ashwin
Recruitment Team""",
}

# --- Helper Functions ---


def get_smartlead_bot():
    """Initialize Smartlead bot with environment credentials"""
    return SmartleadBot()


def get_heyreach_bot():
    """Initialize HeyReach bot with environment credentials"""
    return HeyReachBot()


def _role_filter_sql(role_id: int, column: str = "recruitment_role_id"):
    """Role id 0 is used by Talent Pool UI; store/query it as NULL in DB."""
    if role_id == 0:
        return f"{column} IS NULL", ()
    return f"{column} = %s", (role_id,)


def get_candidate_details(candidate_ids: List[int]):
    """Fetch candidate email and name from database"""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, email, first_name, last_name
                    FROM candidates
                    WHERE id = ANY(%s) AND email IS NOT NULL
                """,
                    (candidate_ids,),
                )

                candidates = []
                for row in cur.fetchall():
                    candidates.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "email": row[2],
                            "first_name": row[3] or row[1].split()[0],
                            "last_name": row[4] or "",
                        }
                    )

            return candidates
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {e}")


def _get_accessible_role(cur, role_id: int, current_user: schemas.User):
    if (current_user.role or "").strip().lower() == "admin":
        cur.execute(
            "SELECT id, name, user_id FROM recruitment_roles WHERE id = %s",
            (role_id,),
        )
    else:
        cur.execute(
            "SELECT id, name, user_id FROM recruitment_roles WHERE id = %s AND user_id = %s",
            (role_id, current_user.id),
        )
    role = cur.fetchone()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    return role


def _render_role_template(value: str, role_name: str) -> str:
    return (value or "").replace("{{role_name}}", role_name)


def _render_candidate_template(value: str, role_name: str, candidate: Dict) -> str:
    return (
        _render_role_template(value, role_name)
        .replace("{{first_name}}", candidate.get("first_name") or "")
        .replace("{{last_name}}", candidate.get("last_name") or "")
    )


def _is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))


def _candidate_role_queue_state(candidate: Dict) -> Dict[str, object]:
    has_email = _is_valid_email(candidate.get("email"))
    has_linkedin = bool((candidate.get("linkedin") or "").strip())
    needs_enrichment = bool(
        (not has_email or not candidate.get("phone")) and has_linkedin
    )
    return {
        "email_status": "scheduled" if has_email else "waiting_for_email",
        "linkedin_status": "scheduled" if has_linkedin else "skipped_missing_linkedin",
        "needs_enrichment": needs_enrichment,
    }


def _classify_role_email_candidates(candidates: List[Dict], existing_ids=None) -> Dict[str, List[Dict]]:
    existing_ids = set(existing_ids or [])
    shortlisted = [candidate for candidate in candidates if candidate.get("status") == "shortlisted"]
    missing_email = [candidate for candidate in shortlisted if not _is_valid_email(candidate.get("email"))]
    eligible = [candidate for candidate in shortlisted if _is_valid_email(candidate.get("email"))]
    pending = [candidate for candidate in eligible if candidate.get("id") not in existing_ids]
    return {
        "shortlisted": shortlisted,
        "missing_email": missing_email,
        "eligible": eligible,
        "pending": pending,
    }


# --- API Endpoints ---


@router.get("/smartlead/email-accounts")
async def list_smartlead_email_accounts(
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if (
        _smartlead_accounts_cache["accounts"]
        and time.monotonic() - float(_smartlead_accounts_cache["ts"]) < _SMARTLEAD_ACCOUNTS_TTL
    ):
        return {"accounts": _smartlead_accounts_cache["accounts"], "cached": True}
    try:
        accounts = get_smartlead_bot().list_email_accounts()
    except Exception as exc:
        if _smartlead_accounts_cache["accounts"]:
            return {"accounts": _smartlead_accounts_cache["accounts"], "cached": True, "stale": True}
        raise HTTPException(status_code=502, detail=f"Could not load Smartlead senders: {exc}")

    safe_accounts = []
    for account in accounts:
        account_id = account.get("id")
        email = account.get("from_email") or account.get("username")
        if account_id is None or not email:
            continue
        warmup = account.get("warmup_details") or {}
        safe_accounts.append(
            {
                "id": account_id,
                "email": email,
                "name": account.get("from_name") or "",
                "connected": account.get("is_smtp_success") is not False,
                "warmup_status": warmup.get("status") or "",
            }
        )
    _smartlead_accounts_cache.update(accounts=safe_accounts, ts=time.monotonic())
    return {"accounts": safe_accounts, "cached": False}


@router.get("/roles/{role_id}/email-setup")
async def get_role_email_setup(
    role_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            _get_accessible_role(cur, role_id, current_user)
            return campaign_payload(fetch_role_campaign(cur, role_id))


@router.put("/roles/{role_id}/email-setup")
async def save_role_email_setup(
    role_id: int,
    request: RoleEmailSetupRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            role = _get_accessible_role(cur, role_id, current_user)
            existing = fetch_role_campaign(cur, role_id)

    bot = get_smartlead_bot()
    try:
        accounts = bot.list_email_accounts()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not validate Smartlead sender: {exc}")
    selected_account = next(
        (account for account in accounts if str(account.get("id")) == str(request.sender_account_id)),
        None,
    )
    if not selected_account:
        raise HTTPException(status_code=400, detail="Selected Smartlead sender does not exist")
    if selected_account.get("is_smtp_success") is False:
        raise HTTPException(status_code=400, detail="Selected Smartlead sender is not connected")
    sender_email = selected_account.get("from_email") or selected_account.get("username")

    # Explicit campaign ID links that campaign (sequence/schedule live in
    # Smartlead); with no ID, auto-create/reuse one named after the role.
    if request.campaign_id > 0:
        resolved_campaign_id = int(request.campaign_id)
    else:
        campaign = provision_role_campaign(role_id, role[1])
        if not campaign.get("campaign_id"):
            raise HTTPException(
                status_code=502,
                detail=campaign.get("campaign_error") or "Could not create Smartlead campaign",
            )
        resolved_campaign_id = int(campaign["campaign_id"])
    bot.campaign_id = resolved_campaign_id
    old_sender_id = str(existing[7]) if existing and existing[7] else ""
    old_campaign_id = str(existing[0]) if existing and existing[0] else ""
    try:
        if request.campaign_id > 0 and bot.get_campaign_analytics() is None:
            raise RuntimeError("Smartlead campaign could not be validated — check the campaign ID")
        sender_changed = old_sender_id != str(request.sender_account_id)
        campaign_changed = old_campaign_id != str(resolved_campaign_id)
        if sender_changed or campaign_changed:
            if bot.add_email_account(request.sender_account_id) is None:
                raise RuntimeError("Could not attach the selected Smartlead sender")
        if old_sender_id and sender_changed and not campaign_changed:
            if bot.remove_email_account(old_sender_id) is None:
                raise RuntimeError("New sender was attached, but the previous sender could not be removed")
    except CampaignNotFoundError:
        raise HTTPException(status_code=400, detail="Smartlead campaign was not found — check the campaign ID")
    except HTTPException:
        raise
    except Exception as exc:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE role_smartlead_campaigns
                        SET provisioning_status = 'failed', provisioning_error = %s, updated_at = NOW()
                        WHERE recruitment_role_id = %s
                        """,
                        (str(exc)[:1000], role_id),
                    )
                    conn.commit()
        raise HTTPException(status_code=502, detail=str(exc))

    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO role_smartlead_campaigns
                    (recruitment_role_id, campaign_id, campaign_name, sender_account_id,
                     sender_email, provisioning_status, configured_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, 'configured', NOW(), NOW(), NOW())
                ON CONFLICT (recruitment_role_id) DO UPDATE
                SET campaign_id = EXCLUDED.campaign_id,
                    sender_account_id = EXCLUDED.sender_account_id,
                    sender_email = EXCLUDED.sender_email,
                    provisioning_status = 'configured', provisioning_error = NULL,
                    configured_at = NOW(), updated_at = NOW()
                """,
                (
                    role_id,
                    str(resolved_campaign_id),
                    role[1],
                    str(request.sender_account_id),
                    sender_email,
                ),
            )
            row = fetch_role_campaign(cur, role_id)
            conn.commit()
            return campaign_payload(row)


@router.post("/roles/{role_id}/email/send-shortlisted")
async def send_role_email_to_shortlisted(
    role_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            role = _get_accessible_role(cur, role_id, current_user)
            setup = fetch_role_campaign(cur, role_id)
            if not setup or not (setup[0] and setup[4] and setup[7]):
                raise HTTPException(status_code=409, detail="Configure Smartlead email outreach first")

            cur.execute(
                """
                SELECT c.id, c.name, c.first_name, c.last_name, c.email, c.status
                FROM recruitment_role_candidates rc
                JOIN candidates c ON c.id = rc.candidate_id
                WHERE rc.role_id = %s AND COALESCE(c.is_archived, FALSE) = FALSE
                ORDER BY c.id
                """,
                (role_id,),
            )
            all_candidates = [
                {
                    "id": row[0],
                    "name": row[1] or "",
                    "first_name": row[2] or (row[1] or "Candidate").split()[0],
                    "last_name": row[3] or " ".join((row[1] or "").split()[1:]),
                    "email": (row[4] or "").strip(),
                    "status": (row[5] or "").strip().lower(),
                }
                for row in cur.fetchall()
            ]
            classified = _classify_role_email_candidates(all_candidates)
            shortlisted = classified["shortlisted"]
            missing_email = classified["missing_email"]
            eligible = classified["eligible"]

            existing_ids = set()
            if eligible:
                cur.execute(
                    """
                    SELECT candidate_id
                    FROM candidate_outreach
                    WHERE recruitment_role_id = %s AND campaign_id = %s
                      AND candidate_id = ANY(%s)
                      AND COALESCE(status, '') <> 'failed'
                    """,
                    (role_id, str(setup[0]), [candidate["id"] for candidate in eligible]),
                )
                existing_ids = {row[0] for row in cur.fetchall()}

    pending = _classify_role_email_candidates(all_candidates, existing_ids)["pending"]
    bot = get_smartlead_bot()
    bot.campaign_id = int(setup[0])

    if pending:
        result = bot.add_leads(
            [
                {
                    "first_name": candidate["first_name"],
                    "last_name": candidate["last_name"],
                    "email": candidate["email"],
                }
                for candidate in pending
            ]
        )
        if result is None:
            raise HTTPException(status_code=502, detail="Smartlead rejected the shortlisted candidates")

        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                for candidate in pending:
                    initial_message = _render_candidate_template(setup[6], role[1], candidate)
                    cur.execute(
                        """
                        INSERT INTO candidate_outreach
                            (candidate_id, recruitment_role_id, campaign_id, campaign_name,
                             status, initial_message, initial_message_at, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, 'in_campaign', %s, NOW(), NOW(), NOW())
                        ON CONFLICT (candidate_id, recruitment_role_id)
                        DO UPDATE SET campaign_id = EXCLUDED.campaign_id,
                                      campaign_name = EXCLUDED.campaign_name,
                                      status = 'in_campaign',
                                      initial_message = EXCLUDED.initial_message,
                                      initial_message_at = NOW(), updated_at = NOW()
                        """,
                        (candidate["id"], role_id, str(setup[0]), setup[1], initial_message),
                    )
                conn.commit()

    if not setup[8] and eligible:
        if bot.start_campaign() is None:
            raise HTTPException(
                status_code=502,
                detail="Candidates were enrolled, but the Smartlead campaign could not be started. Retry safely.",
            )
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE role_smartlead_campaigns
                        SET started_at = NOW(), updated_at = NOW()
                        WHERE recruitment_role_id = %s
                        """,
                        (role_id,),
                    )
                    conn.commit()

    return {
        "success": True,
        "campaign_id": str(setup[0]),
        "processed_count": len(all_candidates),
        "shortlisted_count": len(shortlisted),
        "eligible_count": len(eligible),
        "enrolled_count": len(pending),
        "already_enrolled_count": len(existing_ids),
        "skipped_missing_email_count": len(missing_email),
        "skipped_not_shortlisted_count": len(all_candidates) - len(shortlisted),
    }


@router.post("/roles/{role_id}/candidates/{candidate_id}/shortlist")
async def shortlist_role_candidate(
    role_id: int,
    candidate_id: int,
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Shortlist one role candidate and durably queue both outreach channels."""
    candidate = None
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            role = _get_accessible_role(cur, role_id, current_user)
            activation = fetch_role_activation(cur, role_id)
            if activation.get("activation_status") != "active":
                raise HTTPException(
                    status_code=409,
                    detail=activation.get("activation_error") or "Activate this role before shortlisting candidates",
                )
            cur.execute(
                """
                SELECT c.id, c.name, c.first_name, c.last_name, c.email,
                       COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), ''),
                       c.linkedin
                FROM recruitment_role_candidates rc
                JOIN candidates c ON c.id=rc.candidate_id
                WHERE rc.role_id=%s AND c.id=%s AND COALESCE(c.is_archived, FALSE)=FALSE
                """,
                (role_id, candidate_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Candidate is not assigned to this role")
            candidate = {
                "id": row[0], "name": row[1] or "", "first_name": row[2] or "",
                "last_name": row[3] or "", "email": (row[4] or "").strip(),
                "phone": row[5] or "", "linkedin": row[6] or "",
            }
            queue_state = _candidate_role_queue_state(candidate)
            email_status = queue_state["email_status"]
            linkedin_status = queue_state["linkedin_status"]

            # Guardrail: Check if already shortlisted for this role
            cur.execute(
                "SELECT status, li_status, campaign_id, heyreach_campaign_id FROM candidate_outreach WHERE candidate_id=%s AND recruitment_role_id=%s",
                (candidate_id, role_id)
            )
            existing_outreach = cur.fetchone()

            if existing_outreach:
                e_stat, l_stat, existing_camp_id, existing_hr_camp_id = existing_outreach
                
                # Fetch current role campaigns
                cur.execute("SELECT campaign_id FROM role_smartlead_campaigns WHERE recruitment_role_id=%s", (role_id,))
                role_sl = cur.fetchone()
                current_camp_id = role_sl[0] if role_sl else None
                
                cur.execute("SELECT campaign_id FROM role_heyreach_campaigns WHERE recruitment_role_id=%s", (role_id,))
                role_hr = cur.fetchone()
                current_hr_camp_id = role_hr[0] if role_hr else None

                campaign_changed = (
                    (existing_camp_id and current_camp_id and str(existing_camp_id) != str(current_camp_id)) or
                    (existing_hr_camp_id and current_hr_camp_id and str(existing_hr_camp_id) != str(current_hr_camp_id))
                )

                if campaign_changed:
                    print(f"🔄 Campaign ID changed for {candidate['name']} in role {role_id}. Resetting outreach state.")
                    cur.execute("DELETE FROM candidate_outreach WHERE candidate_id=%s AND recruitment_role_id=%s", (candidate_id, role_id))
                else:
                    active_states = {"queued", "scheduled", "started", "in_campaign", "completed"}
                    if e_stat in active_states or l_stat in active_states:
                        cur.execute(
                            "UPDATE candidates SET status='Shortlisted', updated_at=NOW() WHERE id=%s",
                            (candidate_id,),
                        )
                        conn.commit()
                        background_tasks.add_task(
                            _refresh_role_shortlist_caches,
                            candidate_id,
                            role_id,
                        )
                        print(f"⏩ Skipping outreach for {candidate['name']} in role {role_id} - already triggered earlier.")
                        return {
                            "success": True,
                            "candidate_id": candidate_id,
                            "name": candidate["name"],
                            "email": candidate["email"],
                            "phone": candidate["phone"],
                            "linkedin": candidate["linkedin"],
                            "contact_enriching": False,
                            "email_outreach": "started" if e_stat else "not_started",
                            "linkedin_outreach": "started" if l_stat else "not_started",
                            "already_processed": True,
                            "status": "Shortlisted",
                        }

            cur.execute("UPDATE candidates SET status='Shortlisted', updated_at=NOW() WHERE id=%s", (candidate_id,))
            cur.execute(
                """
                INSERT INTO candidate_outreach
                    (candidate_id, recruitment_role_id, status, li_status, li_scheduled_for, created_at, updated_at)
                VALUES (%s, %s, %s, %s,
                        CASE WHEN %s='scheduled' THEN NOW() ELSE NULL END, NOW(), NOW())
                ON CONFLICT (candidate_id, recruitment_role_id) DO UPDATE
                SET status = CASE WHEN candidate_outreach.email_enrolled_at IS NULL
                                  THEN EXCLUDED.status ELSE candidate_outreach.status END,
                    li_status = CASE WHEN candidate_outreach.li_enrolled_at IS NULL
                                    THEN EXCLUDED.li_status ELSE candidate_outreach.li_status END,
                    li_scheduled_for = CASE WHEN candidate_outreach.li_enrolled_at IS NULL
                                            AND EXCLUDED.li_status='scheduled' THEN NOW()
                                            ELSE candidate_outreach.li_scheduled_for END,
                    updated_at=NOW()
                """,
                (candidate_id, role_id, email_status, linkedin_status, linkedin_status),
            )
            
            from backend.services.auto_call_list import sync_shortlisted_to_call_list
            sync_shortlisted_to_call_list(cur, role_id, [candidate_id])
            
            conn.commit()

    contact_enriching = False
    if _candidate_role_queue_state(candidate)["needs_enrichment"]:
        try:
            from backend.services.clay import trigger_clay
            first_name = candidate["first_name"] or (candidate["name"] or "Candidate").split()[0]
            last_name = candidate["last_name"] or " ".join((candidate["name"] or "").split()[1:])
            background_tasks.add_task(trigger_clay, first_name, last_name, candidate["linkedin"])
            contact_enriching = True
        except Exception as exc:
            print(f"SHORTLIST CLAY WARNING: {exc}")

    background_tasks.add_task(_dispatch_role_outreach_now)
    background_tasks.add_task(_refresh_role_shortlist_caches, candidate_id, role_id)
    return {
        "success": True,
        "candidate_id": candidate_id,
        "name": candidate["name"],
        "email": candidate["email"],
        "phone": candidate["phone"],
        "linkedin": candidate["linkedin"],
        "contact_enriching": contact_enriching,
        "email_outreach": email_status,
        "linkedin_outreach": linkedin_status,
        "status": "Shortlisted",
    }


@router.post("/roles/{role_id}/shortlist-selected")
async def shortlist_selected_for_role(
    role_id: int,
    request: BulkRoleShortlistRequest,
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Assign, shortlist, enrich, and queue selected profiles in one transaction."""
    requested = {
        int(item.candidate_id): item
        for item in request.assignments
        if item.candidate_id is not None
    }
    if not requested:
        raise HTTPException(status_code=400, detail="Select at least one candidate")

    candidates = []
    added_ids = []
    already_assigned_ids = []
    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        with conn.cursor() as cur:
            role = _get_accessible_role(cur, role_id, current_user)
            activation = fetch_role_activation(cur, role_id)
            if activation.get("activation_status") != "active":
                raise HTTPException(
                    status_code=409,
                    detail=activation.get("activation_error") or "Activate this role before adding shortlisted profiles",
                )

            requested_ids = list(requested)
            cur.execute(
                """
                SELECT id, name, first_name, last_name, email,
                       COALESCE(NULLIF(TRIM(mobile_phone), ''), NULLIF(TRIM(phone), ''), ''),
                       linkedin
                FROM candidates
                WHERE id = ANY(%s) AND COALESCE(is_archived, FALSE)=FALSE
                """,
                (requested_ids,),
            )
            candidates = [
                {
                    "id": row[0], "name": row[1] or "", "first_name": row[2] or "",
                    "last_name": row[3] or "", "email": (row[4] or "").strip(),
                    "phone": row[5] or "", "linkedin": row[6] or "",
                }
                for row in cur.fetchall()
            ]
            valid_ids = [candidate["id"] for candidate in candidates]

            if valid_ids:
                cur.execute(
                    "SELECT candidate_id FROM recruitment_role_candidates WHERE role_id=%s AND candidate_id=ANY(%s)",
                    (role_id, valid_ids),
                )
                existing_ids = {int(row[0]) for row in cur.fetchall()}
                already_assigned_ids = sorted(existing_ids)

                for candidate_id in valid_ids:
                    if candidate_id in existing_ids:
                        continue
                    item = requested[candidate_id]
                    cur.execute(
                        """
                        INSERT INTO recruitment_role_candidates (role_id, candidate_id, priority, feedback)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (role_id, candidate_id) DO NOTHING
                        RETURNING candidate_id
                        """,
                        (role_id, candidate_id, item.priority or "--", item.feedback or ""),
                    )
                    inserted = cur.fetchone()
                    if inserted:
                        added_ids.append(int(inserted[0]))
                    else:
                        already_assigned_ids.append(candidate_id)

                # Fetch current role campaigns
                cur.execute("SELECT campaign_id FROM role_smartlead_campaigns WHERE recruitment_role_id=%s", (role_id,))
                role_sl = cur.fetchone()
                current_camp_id = role_sl[0] if role_sl else None
                
                cur.execute("SELECT campaign_id FROM role_heyreach_campaigns WHERE recruitment_role_id=%s", (role_id,))
                role_hr = cur.fetchone()
                current_hr_camp_id = role_hr[0] if role_hr else None

                cur.execute(
                    "SELECT candidate_id, status, li_status, campaign_id, heyreach_campaign_id FROM candidate_outreach WHERE recruitment_role_id=%s AND candidate_id=ANY(%s)",
                    (role_id, valid_ids)
                )
                active_states = {"queued", "scheduled", "started", "in_campaign", "completed"}
                active_outreach_ids = set()
                ids_to_reset = []
                for row in cur.fetchall():
                    c_id, e_stat, l_stat, existing_camp_id, existing_hr_camp_id = row
                    
                    campaign_changed = (
                        (existing_camp_id and current_camp_id and str(existing_camp_id) != str(current_camp_id)) or
                        (existing_hr_camp_id and current_hr_camp_id and str(existing_hr_camp_id) != str(current_hr_camp_id))
                    )
                    
                    if campaign_changed:
                        ids_to_reset.append(int(c_id))
                    elif e_stat in active_states or l_stat in active_states:
                        active_outreach_ids.add(int(c_id))

                if ids_to_reset:
                    print(f"🔄 Bulk: Campaign ID changed for {len(ids_to_reset)} candidates in role {role_id}. Resetting outreach state.")
                    cur.execute("DELETE FROM candidate_outreach WHERE recruitment_role_id=%s AND candidate_id=ANY(%s)", (role_id, ids_to_reset))

                cur.execute(
                    "UPDATE candidates SET status='Shortlisted', updated_at=NOW() WHERE id=ANY(%s)",
                    (valid_ids,),
                )
                for candidate in candidates:
                    if candidate["id"] in active_outreach_ids:
                        continue
                    queue_state = _candidate_role_queue_state(candidate)
                    email_status = queue_state["email_status"]
                    linkedin_status = queue_state["linkedin_status"]
                    cur.execute(
                        """
                        INSERT INTO candidate_outreach
                            (candidate_id, recruitment_role_id, status, li_status, li_scheduled_for, created_at, updated_at)
                        VALUES (%s, %s, %s, %s,
                                CASE WHEN %s='scheduled' THEN NOW() ELSE NULL END, NOW(), NOW())
                        ON CONFLICT (candidate_id, recruitment_role_id) DO UPDATE
                        SET status = CASE WHEN candidate_outreach.email_enrolled_at IS NULL
                                          THEN EXCLUDED.status ELSE candidate_outreach.status END,
                            li_status = CASE WHEN candidate_outreach.li_enrolled_at IS NULL
                                            THEN EXCLUDED.li_status ELSE candidate_outreach.li_status END,
                            li_scheduled_for = CASE WHEN candidate_outreach.li_enrolled_at IS NULL
                                                    AND EXCLUDED.li_status='scheduled' THEN NOW()
                                                    ELSE candidate_outreach.li_scheduled_for END,
                            updated_at=NOW()
                        """,
                        (candidate["id"], role_id, email_status, linkedin_status, linkedin_status),
                    )
            
            if valid_ids:
                from backend.services.auto_call_list import sync_shortlisted_to_call_list
                sync_shortlisted_to_call_list(cur, role_id, valid_ids)
            
            conn.commit()

    # Only enrich candidates who aren't already actively being pushed
    enriching = [
        candidate for candidate in candidates
        if candidate["id"] not in active_outreach_ids and _candidate_role_queue_state(candidate)["needs_enrichment"]
    ]
    background_tasks.add_task(_dispatch_role_outreach_now)
    if enriching:
        from backend.services.clay import trigger_clay
        for candidate in enriching:
            first_name = candidate["first_name"] or (candidate["name"] or "Candidate").split()[0]
            last_name = candidate["last_name"] or " ".join((candidate["name"] or "").split()[1:])
            background_tasks.add_task(trigger_clay, first_name, last_name, candidate["linkedin"])
    valid_ids = [candidate["id"] for candidate in candidates]
    background_tasks.add_task(_refresh_bulk_role_shortlist_caches, valid_ids, role_id)

    email_queued = sum(1 for candidate in candidates if _is_valid_email(candidate["email"]))
    linkedin_queued = sum(1 for candidate in candidates if candidate["linkedin"])
    return {
        "success": True,
        "role_id": role_id,
        "role_name": role[1],
        "requested_count": len(requested),
        "processed_count": len(candidates),
        "added_count": len(added_ids),
        "already_assigned_count": len(set(already_assigned_ids)),
        "skipped_count": len(requested) - len(candidates),
        "enriching_count": len(enriching),
        "email_queued_count": email_queued,
        "email_waiting_count": len(candidates) - email_queued,
        "linkedin_queued_count": linkedin_queued,
        "linkedin_skipped_count": len(candidates) - linkedin_queued,
        "candidate_ids": valid_ids,
    }


@router.post("/trigger")
async def trigger_outreach(
    request: OutreachTriggerRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """
    Trigger Smartlead campaign for selected candidates
    """
    # 1. Fetch candidate details
    candidates = get_candidate_details(request.candidate_ids)

    if not candidates:
        raise HTTPException(
            status_code=404, detail="No candidates with valid emails found"
        )

    # 2. Initialize Smartlead bot
    bot = get_smartlead_bot()
    sender_email = os.getenv("SMARTLEAD_SENDER_EMAIL")
    timezone = os.getenv("SMARTLEAD_DEFAULT_TIMEZONE", "Asia/Kolkata")

    # 3. Create campaign
    campaign_name = request.role_name

    campaign_id = bot.create_campaign(campaign_name)

    if not campaign_id:
        raise HTTPException(
            status_code=500, detail="Failed to create Smartlead campaign"
        )

    # 4. Configure campaign
    bot.add_email_account(sender_email)

    subject = EMAIL_TEMPLATE["subject"].format(role_name=request.role_name)
    body = EMAIL_TEMPLATE["body"].format(role_name=request.role_name)
    bot.set_email_sequence(subject, body)

    # Calculate start time: 3 minutes from now
    from datetime import timedelta, timezone as tz_module

    start_time = datetime.now(tz_module.utc) + timedelta(minutes=3)

    bot.set_schedule(
        tz=timezone,
        start_hour="00:00",
        end_hour="23:59",
        start_time=start_time,
        days_of_the_week=[0, 1, 2, 3, 4, 5, 6],
    )
    bot.update_campaign_settings(follow_up_percentage=50)

    # 5. Add leads
    leads = [
        {
            "first_name": c["first_name"],
            "last_name": c["last_name"],
            "email": c["email"],
        }
        for c in candidates
    ]

    bot.add_leads(leads)

    # 6. Start campaign
    bot.start_campaign()

    # 7. Record in database
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur:
                    for candidate in candidates:
                        cur.execute(
                            """
                            INSERT INTO candidate_outreach
                            (candidate_id, recruitment_role_id, campaign_id, campaign_name, status, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, 'in_campaign', NOW(), NOW())
                            ON CONFLICT (candidate_id, recruitment_role_id)
                            DO UPDATE SET
                                campaign_id = EXCLUDED.campaign_id,
                                campaign_name = EXCLUDED.campaign_name,
                                status = 'in_campaign',
                                updated_at = NOW()
                        """,
                            (candidate["id"], request.role_id, campaign_id, campaign_name),
                        )

                conn.commit()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to record outreach: {e}"
        )

    return {
        "success": True,
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "candidates_count": len(candidates),
    }


@router.post("/shortlist/{candidate_id}")
async def shortlist_outreach(
    candidate_id: int,
    request: ShortlistOutreachRequest = ShortlistOutreachRequest(),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """
    Triggered when a candidate is marked as Shortlisted in Talent Pool.
    1. Fetches email, mobile_phone, linkedin from DB
    2. Triggers Smartlead email campaign for this candidate
    3. Pushes lead to HeyReach LinkedIn campaign
    Returns: { email, phone, linkedin, email_outreach, linkedin_outreach }
    """
    # Step 1: Fetch candidate from DB
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, first_name, last_name, email, mobile_phone, linkedin FROM candidates WHERE id = %s",
                    (candidate_id,),
                )
                row = cur.fetchone()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB fetch failed: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found")

    cand_id, name, first_name, last_name, email, mobile_phone, linkedin_url = row
    first_name = first_name or (name.split()[0] if name else "Candidate")
    last_name = last_name or (
        name.split()[-1] if name and len(name.split()) > 1 else ""
    )

    contact_enriching = False
    email_outreach = "not_started"
    linkedin_outreach = "not_started"
    email_campaign_id = None

    # Step 1.5: Guardrail - Check if already shortlisted/outreached recently
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if conn:
                with conn.cursor() as cur_check:
                    cur_check.execute(
                        """
                        SELECT status, li_status, updated_at
                        FROM candidate_outreach
                        WHERE candidate_id = %s AND recruitment_role_id IS NULL
                    """,
                        (cand_id,),
                    )
                    existing = cur_check.fetchone()
                    if existing:
                        e_stat, l_stat, updated_at = existing
                        print(f"⏩ Skipping outreach for {name} - already triggered earlier.")
                        return {
                            "success": True,
                            "candidate_id": cand_id,
                            "name": name,
                            "email": email or "",
                            "phone": mobile_phone or "",
                            "linkedin": linkedin_url or "",
                            "email_outreach": "started" if e_stat else "not_started",
                            "linkedin_outreach": "started" if l_stat else "not_started",
                            "contact_enriching": False,
                            "already_processed": True,
                        }
    except Exception as db_e:
        print(f"Warning: could not check existing outreach: {db_e}")

    # Step 1b: If email or phone missing from DB, trigger Clay enrichment
    if (not email or not mobile_phone) and linkedin_url:
        try:
            from backend.services.clay import trigger_clay

            triggered = trigger_clay(first_name, last_name, linkedin_url)
            if triggered:
                print(
                    f"✅ Clay enrichment triggered for {name} (will populate via /api/results webhook)"
                )
                contact_enriching = True

                # Insert a dummy record immediately so rapid double-clicks are blocked by Step 1.5
                try:
                    with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                        if conn:
                            with conn.cursor() as cur_ins:
                                cur_ins.execute(
                                    """
                                    INSERT INTO candidate_outreach (candidate_id, recruitment_role_id, created_at, updated_at)
                                    SELECT %s, NULL, NOW(), NOW()
                                    WHERE NOT EXISTS (
                                        SELECT 1 FROM candidate_outreach
                                        WHERE candidate_id = %s AND recruitment_role_id IS NULL
                                    )
                                """,
                                    (cand_id, cand_id),
                                )
                                conn.commit()
                except:
                    pass
            else:
                print(f"⚠️ Clay trigger failed for {name}")
        except Exception as clay_e:
            print(f"⚠️ Clay enrichment error for {name}: {clay_e}")

    # Step 2: Trigger Smartlead email campaign (if email exists and is valid)
    def is_valid_email(e):
        return e and str(e).strip().lower() not in ["", "na", "n/a", "none"]

    if is_valid_email(email):
        try:
            bot = get_smartlead_bot()
            sender_email = os.getenv("SMARTLEAD_SENDER_EMAIL")
            timezone = os.getenv("SMARTLEAD_DEFAULT_TIMEZONE", "Asia/Kolkata")
            campaign_name = f"Shortlist - {name}"

            campaign_id = bot.create_campaign(campaign_name)
            if campaign_id:
                bot.add_email_account(sender_email)
                bot.set_email_sequence(
                    subject="Exciting Opportunity",
                    body=f"""Hi {first_name},

I came across your profile and was truly impressed by your experience. We have an exciting opportunity that would be a great fit for your background.

Would you be open to a quick call to explore this further?

Best regards,
Recruitment Team""",
                )
                from datetime import timedelta, timezone as tz_module

                start_time = datetime.now(tz_module.utc) + timedelta(minutes=3)
                bot.set_schedule(
                    tz=timezone,
                    start_hour="00:00",
                    end_hour="23:59",
                    start_time=start_time,
                    days_of_the_week=[0, 1, 2, 3, 4, 5, 6],
                )
                bot.update_campaign_settings(follow_up_percentage=50)
                bot.add_leads(
                    [{"first_name": first_name, "last_name": last_name, "email": email}]
                )
                bot.start_campaign()
                email_campaign_id = campaign_id
                email_outreach = "started"

                # Record in DB as Talent Pool row (NULL recruitment_role_id).
                try:
                    with get_db_connection_context(validate=False, register_pgvector=False) as conn2:
                        if conn2:
                            with conn2.cursor() as cur2:
                                cur2.execute(
                                    "SELECT 1 FROM candidate_outreach WHERE candidate_id = %s AND recruitment_role_id IS NULL",
                                    (cand_id,),
                                )
                                exists = cur2.fetchone() is not None
                                if exists:
                                    cur2.execute(
                                        """
                                        UPDATE candidate_outreach
                                        SET campaign_id = %s,
                                            campaign_name = %s,
                                            status = 'in_campaign',
                                            updated_at = NOW()
                                        WHERE candidate_id = %s AND recruitment_role_id IS NULL
                                    """,
                                        (campaign_id, campaign_name, cand_id),
                                    )
                                else:
                                    cur2.execute(
                                        """
                                        INSERT INTO candidate_outreach
                                        (candidate_id, recruitment_role_id, campaign_id, campaign_name, status, created_at, updated_at)
                                        VALUES (%s, NULL, %s, %s, 'in_campaign', NOW(), NOW())
                                    """,
                                        (cand_id, campaign_id, campaign_name),
                                    )
                                conn2.commit()
                except Exception as db_e:
                    print(
                        f"Warning: could not record email outreach for candidate {cand_id}: {db_e}"
                    )
        except Exception as e:
            print(f"Smartlead outreach failed for candidate {cand_id}: {e}")
            email_outreach = "error"

    # Step 3: Trigger HeyReach LinkedIn campaign (if LinkedIn URL exists)
    if linkedin_url:
        try:
            hr_campaign_id = request.hr_campaign_id or int(
                os.getenv("HEYREACH_DEFAULT_CAMPAIGN_ID", "0")
            )
            sender_account_id = request.sender_account_id or int(
                os.getenv("HEYREACH_DEFAULT_SENDER_ACCOUNT_ID", "113572")
            )

            if hr_campaign_id > 0:
                hr_bot = HeyReachBot()
                result = hr_bot.push_lead(
                    campaign_id=hr_campaign_id,
                    account_id=sender_account_id,
                    first_name=first_name,
                    last_name=last_name,
                    profile_url=linkedin_url,
                )
                linkedin_outreach = "started" if result else "error"

                # Record LinkedIn status in DB
                if linkedin_outreach == "started":
                    try:
                        with get_db_connection_context(validate=False, register_pgvector=False) as conn3:
                            if conn3:
                                with conn3.cursor() as cur3:
                                    cur3.execute(
                                        "SELECT 1 FROM candidate_outreach WHERE candidate_id = %s AND recruitment_role_id IS NULL",
                                        (cand_id,),
                                    )
                                    exists = cur3.fetchone() is not None
                                    if exists:
                                        cur3.execute(
                                            """
                                            UPDATE candidate_outreach
                                            SET heyreach_campaign_id = %s,
                                                li_status = 'in_campaign',
                                                updated_at = NOW()
                                            WHERE candidate_id = %s AND recruitment_role_id IS NULL
                                        """,
                                            (hr_campaign_id, cand_id),
                                        )
                                    else:
                                        cur3.execute(
                                            """
                                            INSERT INTO candidate_outreach
                                            (candidate_id, recruitment_role_id, heyreach_campaign_id, li_status, created_at, updated_at)
                                            VALUES (%s, NULL, %s, 'in_campaign', NOW(), NOW())
                                        """,
                                            (cand_id, hr_campaign_id),
                                        )
                                    conn3.commit()
                    except Exception as db_e:
                        print(
                            f"Warning: could not record LinkedIn outreach for candidate {cand_id}: {db_e}"
                        )
            else:
                linkedin_outreach = "no_campaign_id"
        except Exception as e:
            print(f"HeyReach outreach failed for candidate {cand_id}: {e}")
            linkedin_outreach = "error"

    # Keep in-memory cache synchronized so Talent Pool browse reflects fresh contact data.
    try:
        from backend.pipeline.query import PROFILES_BY_ID

        profile = PROFILES_BY_ID.get(cand_id)
        if profile:
            if email:
                profile["email"] = email
            if mobile_phone:
                profile["phone"] = mobile_phone
            # Also sync HeyReach status if it was triggered
            if "hr_campaign_id" in locals() and hr_campaign_id:
                profile["heyreach_campaign_id"] = str(hr_campaign_id)
                profile["li_status"] = "in_campaign"
    except Exception:
        pass

    return {
        "success": True,
        "candidate_id": cand_id,
        "name": name,
        "email": email or "",
        "phone": mobile_phone or "",
        "linkedin": linkedin_url or "",
        "email_outreach": email_outreach,
        "linkedin_outreach": linkedin_outreach,
        "contact_enriching": contact_enriching,  # True if Clay was triggered to fetch email/phone
    }


@router.get("/status/{role_id}")
async def get_outreach_status(
    role_id: int, current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Get outreach status for all candidates in a role
    """
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "recruitment_role_id")
                cur.execute(
                    f"""
                    SELECT
                        candidate_id,
                        status,
                        message_sent_count,
                        last_message_sent_at,
                        response_received_at,
                        response_text,
                        li_status,
                        li_last_action_at,
                        li_response_text,
                        li_sent_count,
                        li_response_received_at,
                        li_conversation_id,
                        CASE
                            WHEN jsonb_typeof(email_chat_history_cache) = 'array'
                            THEN jsonb_array_length(email_chat_history_cache)
                            ELSE 0
                        END AS email_cached_message_count,
                        CASE
                            WHEN jsonb_typeof(li_chat_history_cache) = 'array'
                            THEN jsonb_array_length(li_chat_history_cache)
                            ELSE 0
                        END AS li_cached_message_count,
                        response_read_at,
                        li_response_read_at
                    FROM candidate_outreach
                    WHERE {role_where}
                """,
                    role_params,
                )

                statuses = {}
                for row in cur.fetchall():
                    email_message_count = max(
                        int(row[2] or 0) + (1 if row[5] else 0),
                        int(row[12] or 0),
                    )
                    li_message_count = max(
                        int(row[9] or 0) + (1 if row[8] else 0),
                        int(row[13] or 0),
                    )
                    # Unread = a response exists and was never read, or a newer
                    # response arrived after the last read.
                    email_unread = (bool(row[4]) or bool(row[5])) and (
                        row[14] is None or (row[4] is not None and row[14] < row[4])
                    )
                    li_unread = (bool(row[10]) or bool(row[8])) and (
                        row[15] is None or (row[10] is not None and row[15] < row[10])
                    )
                    statuses[row[0]] = {
                        "candidate_id": row[0],
                        "status": row[1],
                        "message_sent_count": row[2],
                        "last_message_sent_at": row[3],
                        "response_received_at": row[4],
                        "response_text": row[5],
                        "li_status": row[6],
                        "li_last_action_at": row[7],
                        "li_response_text": row[8],
                        "li_sent_count": row[9],
                        "li_response_received_at": row[10],
                        "li_conversation_id": row[11],
                        "email_message_count": email_message_count,
                        "li_message_count": li_message_count,
                        "message_count": email_message_count + li_message_count,
                        "response_read_at": row[14],
                        "li_response_read_at": row[15],
                        "has_unread_response": email_unread or li_unread,
                    }
        return statuses
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {e}")


@router.post("/mark-response-read/{role_id}/{candidate_id}")
async def mark_response_read(
    role_id: int,
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Mark a candidate's email/LinkedIn responses as read (user opened the conversation)."""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "recruitment_role_id")
                cur.execute(
                    f"""
                    UPDATE candidate_outreach
                    SET response_read_at = NOW(), li_response_read_at = NOW()
                    WHERE candidate_id = %s AND {role_where}
                    """,
                    (candidate_id, *role_params),
                )
            conn.commit()
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark response read: {e}")


@router.get("/chat/email/{role_id}/{candidate_id}")
async def get_email_chat_history(
    role_id: int,
    candidate_id: int,
    force: bool = False,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured Email chat history for a candidate"""
    # NOTE: force/current_user must be passed by keyword — passing current_user
    # positionally lands it in `force`, which permanently disables the cache.
    return await get_chat_history(role_id, candidate_id, force=force, current_user=current_user)


@router.post("/prewarm/linkedin")
async def prewarm_linkedin_cache(
    request: PrewarmRequest, current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Pre-warm the LinkedIn chat cache for a list of candidates in parallel background threads.
    The frontend calls this right after the talent pool table renders so that by the time
    the user clicks 'View Chat' the data is ready and the response is instant.
    """
    queued = 0
    for cid in request.candidate_ids:
        with _li_chat_lock:
            cached = _li_chat_cache.get(cid)
            already_fresh = (
                cached is not None
                and not cached.get("refreshing", False)
                and (time.monotonic() - cached["ts"]) < _LI_CACHE_TTL
            )
            already_running = cached and cached.get("refreshing", False)

            if already_fresh or already_running:
                continue

            # Place a stub so others know we are fetching
            if cached is None:
                _li_chat_cache[cid] = {"messages": [], "ts": 0, "refreshing": True}
            else:
                _li_chat_cache[cid]["refreshing"] = True

        t = threading.Thread(target=_prewarm_single, args=(cid,), daemon=True)
        t.start()
        queued += 1

    print(
        f"DEBUG: Prewarm queued {queued} background fetches for {len(request.candidate_ids)} candidates"
    )
    return {"queued": queued, "total": len(request.candidate_ids)}


@router.get("/chat/linkedin/{role_id}/{candidate_id}")
async def get_linkedin_chat_history(
    role_id: int,
    candidate_id: int,
    force: bool = False,
):
    """Fetch structured LinkedIn chat history for a candidate.

    Uses an in-memory cache so the response is instant on repeat opens.
    A background thread always refreshes the cache after serving.
    """
    import time
    start_t = time.time()
    
    # ── HOT PATH CACHE CHECK ──
    with _li_chat_lock:
        cached = _li_chat_cache.get(candidate_id)
        if cached:
            cache_age = time.monotonic() - cached.get("ts", 0)
            is_stale = cache_age > _LI_CACHE_STALE_THRESHOLD
            already_refreshing = cached.get("refreshing", False)
            if (not is_stale or already_refreshing) and not force:
                final_msgs = cached.get("messages", [])
                initial_li_message = cached.get("initial", None)
                initial_li_message_at = cached.get("initial_at", None)
                if initial_li_message:
                    has_sent = any(msg.get("type") == "SENT" for msg in final_msgs)
                    if not has_sent:
                        clean_init = initial_li_message.strip()
                        _JUNK_LI_INITIALS = {"hii", "hi", "hey", "hello", "test", "linkedin", "msg", "message", "helo", "hello!", "hi!"}
                        if len(clean_init) >= 12 and clean_init.lower() not in _JUNK_LI_INITIALS:
                            entry = {
                                "type": "SENT",
                                "email_body": clean_init,
                                "time": initial_li_message_at.isoformat() if initial_li_message_at else None,
                                "sender_name": "You",
                            }
                            if not final_msgs:
                                final_msgs = [entry]
                            elif not any((m.get("email_body") or "").strip() == clean_init for m in final_msgs):
                                final_msgs = [entry] + final_msgs
                try:
                    final_msgs.sort(key=lambda x: x.get("time", ""))
                except:
                    pass
                return {"messages": final_msgs, "syncing": already_refreshing}
    try:
        t0 = time.time()
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            t1 = time.time()
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
                cand_row = cur.fetchone()

                if role_id == 0:
                    cur.execute(
                        """
                        SELECT updated_at, heyreach_campaign_id, li_conversation_id, initial_li_message, initial_li_message_at, li_account_id, li_response_text, response_received_at,
                               li_chat_history_cache
                        FROM candidate_outreach
                        WHERE candidate_id = %s
                        ORDER BY heyreach_campaign_id DESC NULLS LAST, updated_at DESC LIMIT 1
                    """,
                        (candidate_id,),
                    )
                else:
                    role_where, role_params = _role_filter_sql(role_id, "recruitment_role_id")
                    cur.execute(
                        f"""
                        SELECT updated_at, heyreach_campaign_id, li_conversation_id, initial_li_message, initial_li_message_at, li_account_id, li_response_text, response_received_at,
                               li_chat_history_cache
                        FROM candidate_outreach
                        WHERE candidate_id = %s AND {role_where}
                        ORDER BY heyreach_campaign_id DESC NULLS LAST, updated_at DESC LIMIT 1
                    """,
                        (candidate_id, *role_params),
                    )

                outreach_row = cur.fetchone()
                print(f"DEBUG: Outreach record for cand {candidate_id}: {outreach_row}")

        if not cand_row or not cand_row[0]:
            return {"messages": []}

        profile_url = cand_row[0]
        heyreach_campaign_id = (
            outreach_row[1] if outreach_row and len(outreach_row) > 1 else None
        )
        li_conversation_id = (
            outreach_row[2] if outreach_row and len(outreach_row) > 2 else None
        )
        initial_li_text = (
            outreach_row[3] if outreach_row and len(outreach_row) > 3 else None
        )
        initial_li_at = (
            outreach_row[4] if outreach_row and len(outreach_row) > 4 else None
        )
        li_account_id = (
            outreach_row[5] if outreach_row and len(outreach_row) > 5 else None
        )
        li_response_text = (
            outreach_row[6] if outreach_row and len(outreach_row) > 6 else None
        )
        response_received_at = (
            outreach_row[7] if outreach_row and len(outreach_row) > 7 else None
        )
        # Index 8: li_chat_history_cache — fetched in same query, no extra DB call needed
        li_chat_history_cache_raw = (
            outreach_row[8] if outreach_row and len(outreach_row) > 8 else None
        )
        li_account_id_int = int(li_account_id) if li_account_id else None
        campaign_id_int = int(heyreach_campaign_id) if heyreach_campaign_id else None

        # ---------------------------------------------------------------
        # Cache logic - optimized for performance
        # ---------------------------------------------------------------
        # Strategy: Return cached data ASAP, refresh in background if
        li_chat_history_cache = outreach_row[8] if len(outreach_row) > 8 else None

        t2 = time.time()
        with _li_chat_lock:
            t3 = time.time()
            cached = _li_chat_cache.get(candidate_id)
            cache_ts = cached.get("ts", 0) if cached else 0
            cache_age = time.monotonic() - cache_ts
            is_stale = cache_age > _LI_CACHE_STALE_THRESHOLD
            already_refreshing = cached and cached.get("refreshing", False)

            if force:
                is_stale = True
                already_refreshing = False

        # Common junk/placeholder values that should never be shown as real messages
        _JUNK_LI_INITIALS = {"linkedin", "hi", "hii", "hello", "test", "hey", "helo", "helo", "msg", "message"}

        def _prepend_initial(msgs):
            """Helper: prepend initial_li_text if not already in the list.
            Guards against junk test data being shown as a real sent message.
            """
            if not initial_li_text:
                return msgs
            clean = initial_li_text.strip()
            # No more strict junk guard - if a message exists, show it.
            if not clean:
                return msgs
            clean_key = _msg_dedup_key(clean)
            already = any(
                _msg_dedup_key(m.get("email_body")) == clean_key
                for m in (msgs or [])
            )
            if already:
                return msgs
            entry = {
                "type": "SENT",
                "email_body": clean,
                "time": initial_li_at.isoformat() if initial_li_at else None,
                "sender_name": "You",
            }
            return [entry] + (msgs or [])


        # ── Always-instant strategy ─────────────────────────────────────────
        # Return cached data immediately. Kick off a background refresh if
        # stale. The frontend handles freshness via SWR parallel fetching.

        if (not is_stale or already_refreshing) and not force:
            return {'messages': _prepend_initial(cached['messages'] if cached else []), 'syncing': already_refreshing}

        if not already_refreshing and (is_stale or not cached):
            # ── INSTANT DB CACHE RESTORE ─────────────────────────────────────
            # li_chat_history_cache is already in outreach_row[8] — zero extra DB calls.
            # Restore to memory cache so the response is instant on first load.
            if not cached and li_chat_history_cache_raw:
                try:
                    messages_from_db = li_chat_history_cache_raw
                    if isinstance(messages_from_db, str):
                        messages_from_db = json.loads(messages_from_db)
                    if messages_from_db:
                        print(f"DEBUG: Instantly restored {len(messages_from_db)} msgs from DB cache for cand {candidate_id}")
                        with _li_chat_lock:
                            _li_chat_cache[candidate_id] = {
                                "messages": messages_from_db,
                                # Mark semi-stale so background refresh still runs
                                "ts": time.monotonic() - (_LI_CACHE_TTL / 2),
                                "refreshing": False
                            }
                            cached = _li_chat_cache[candidate_id]
                except Exception as e:
                    print(f"WARNING: DB cache restore parse failed: {e}")

            # Kick off background refresh (always, if stale or first load)
            with _li_chat_lock:
                if candidate_id not in _li_chat_cache:
                    _li_chat_cache[candidate_id] = {"messages": [], "ts": 0, "refreshing": True}
                else:
                    _li_chat_cache[candidate_id]["refreshing"] = True

            t = threading.Thread(
                target=_refresh_li_cache_task,
                args=(candidate_id, profile_url, campaign_id_int, li_conversation_id, li_account_id_int),
                daemon=True,
            )
            t.start()
            print(f"DEBUG: Background HeyReach refresh started for cand {candidate_id} (cached={cached is not None}, stale={is_stale})")

        # Store the initial LI message on the existing cache entry so the hot
        # path can prepend it without a DB read on future polls.
        with _li_chat_lock:
            li_entry = _li_chat_cache.get(candidate_id)
            if li_entry is not None:
                li_entry["initial"] = initial_li_text
                li_entry["initial_at"] = initial_li_at

        current_messages = cached["messages"] if cached else []
        final_msgs = _prepend_initial(current_messages)

        # ── OPTIMISTIC FALLBACK ──────────────────────────────────────────────
        # If the history is STILL empty, but we possess a reply in our DB,
        # synthesize a virtual message so the user sees the content immediately.
        if not final_msgs and li_response_text:
            print(f"DEBUG: Using optimistic fallback for candidate {candidate_id}")
            final_msgs = [{
                "type": "RECEIVED",
                "email_body": li_response_text,
                "time": response_received_at.isoformat() if response_received_at else None,
                "sender_name": "Candidate"
            }]

        result = {
            "messages": final_msgs,
            "syncing": not cached or is_stale or (cached and cached.get("refreshing")),
        }
        t4 = time.time()
        print(f"DEBUG TIMING: get_db_conn={t1-t0:.4f}s, db_query={t2-t1:.4f}s, lock_wait={t3-t2:.4f}s, rest={t4-t3:.4f}s")
        return result



    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch LinkedIn chat history: {e}"
        )


# Narrow HTML detector: avoids misclassifying plain text with a stray "<".
_HTML_HINT = re.compile(
    r"<\s*(br|p|div|span|a|strong|b|i|em|u|ul|ol|li|table|tr|td|blockquote|h[1-6])\b"
    r"|</\s*[a-z]+\s*>|&(nbsp|amp|lt|gt|quot|#39|apos);",
    re.IGNORECASE,
)


def _normalize_body_text(body: str) -> str:
    """Convert an HTML email body to clean plain text with real newlines.

    Only touches bodies that actually look like HTML; plain-text messages pass
    through untouched so existing (already-clean) messages are never disturbed.
    Mirrors the frontend normalization so display is identical everywhere.
    """
    if not body or not isinstance(body, str):
        return body
    if not _HTML_HINT.search(body):
        return body
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", body, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*(p|div|li|tr|h[1-6]|blockquote)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _msg_dedup_key(text: str) -> str:
    """Formatting-agnostic key for comparing two message bodies.

    Normalizes HTML (<br> etc.), collapses whitespace and lowercases so the
    stored initial_message (clean text) and its Smartlead-synced copy (which may
    carry <br> markup) compare equal — preventing the same send from rendering
    twice in the conversation view.
    """
    return re.sub(r"\s+", " ", _normalize_body_text(text or "") or "").strip().lower()


def _clean_email_body(body: str) -> str:
    """Strip only trailing quoted/forwarded sections from OUTBOUND email bodies.

    Uses maxsplit=1 so only the first quote-block is removed, then stops.
    If the result is empty, falls back to plain-text from the original HTML.
    Candidate REPLY / INBOX messages should NOT be passed through this function.
    """
    if not body:
        return ""

    # Markers that begin a quoted section; we split on the FIRST occurrence only
    markers = [
        r'<div[^>]*class=["\'][^"\']*(?:gmail_quote|smartlead-quote)[^"\']*["\']',
        r"<blockquote",
        r"On\s+[A-Za-z]{3},?\s+[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}.*?wrote:",
        r"<div>\s*On\s+.*?wrote:",
        r"<hr[^>]*>\s*<b>From:</b>",
        r"-----\s*Original Message\s*-----",
    ]

    cleaned = body
    for marker in markers:
        parts = re.split(marker, cleaned, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)
        if len(parts) > 1 and parts[0].strip():
            cleaned = parts[0]
            break  # Stop at first match

    # Clean up trailing tags, whitespace, and empty paragraphs
    cleaned = re.sub(
        r"(<br\s*/?>|<p[^>]*>\s*</p>|\s)+$", "", cleaned, flags=re.IGNORECASE
    ).strip()

    # Safety fallback: if cleaning emptied the body, strip HTML tags from original
    if not cleaned:
        cleaned = re.sub(r"<[^>]+>", " ", body).strip()

    # Convert any remaining inline <br>/block tags to real newlines so no HTML
    # markup is stored or shown (this is what fixes the literal "<br><br>").
    return _normalize_body_text(cleaned)


@router.get("/chat/{role_id}/{candidate_id}")
async def get_chat_history(
    role_id: int,
    candidate_id: int,
    force: bool = False,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured chat history (Default to Email)"""
    # ── HOT PATH: serve a fresh in-memory cache without any DB round trip ──
    # Mirrors the LinkedIn endpoint. Each DB round trip costs ~0.6s to the
    # remote region and the conversation modal polls this endpoint every few
    # seconds. Only valid once the sanitized initial message has been cached
    # ("initial" key present) — set by the DB path below.
    if not force:
        with _email_chat_lock:
            hot_entry = _email_chat_cache.get(candidate_id)
            if hot_entry and "initial" in hot_entry:
                hot_age = time.monotonic() - hot_entry.get("ts", 0)
                hot_refreshing = hot_entry.get("refreshing", False)
                if hot_age <= _EMAIL_CACHE_STALE_THRESHOLD or hot_refreshing:
                    hot_msgs = list(hot_entry.get("messages", []))
                    hot_initial = hot_entry.get("initial")
                    hot_initial_at = hot_entry.get("initial_at")
                    if hot_initial:
                        entry = {
                            "type": "SENT",
                            "email_body": hot_initial,
                            "time": hot_initial_at.isoformat() if hot_initial_at else None,
                            "sender_name": "You",
                        }
                        init_key = _msg_dedup_key(hot_initial)
                        if not hot_msgs:
                            hot_msgs = [entry]
                        elif not any(_msg_dedup_key(m.get("email_body")) == init_key for m in hot_msgs):
                            hot_msgs = [entry] + hot_msgs
                    return {"messages": hot_msgs, "syncing": hot_refreshing}

    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
                # Strictly prioritize rows that have a Smartlead campaign_id (Email campaign)
                # to prevent LinkedIn records from leaking into the Email history view.
                cur.execute(
                    f"""
                    SELECT c.email, co.campaign_id, co.initial_message, co.initial_message_at,
                           co.email_chat_history_cache, co.email_chat_history_updated_at
                    FROM candidate_outreach co
                    JOIN candidates c ON c.id = co.candidate_id
                    WHERE co.candidate_id = %s AND {role_where}
                    ORDER BY (co.campaign_id IS NOT NULL) DESC, co.updated_at DESC
                    LIMIT 1
                """,
                    (candidate_id, *role_params),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404, detail="Candidate outreach record not found"
            )

        email, campaign_id, initial_msg_text, initial_msg_at, db_cache, db_updated_at = row
        print(f"DEBUG: Email chat — email={email}, campaign_id={campaign_id}")

        # ── Always-instant strategy for Email ──
        with _email_chat_lock:
            cached = _email_chat_cache.get(candidate_id)
            cache_ts = cached.get("ts", 0) if cached else 0
            cache_age = time.monotonic() - cache_ts
            is_stale = force or (cache_age > _EMAIL_CACHE_STALE_THRESHOLD)
            already_refreshing = cached and cached.get("refreshing", False)

        # ── Guard: junk initial messages or platform crosstalk
        # If we don't have a campaign_id, this record might be a LinkedIn-only record.
        # We should NOT show the initial_message (email col) if there's no email campaign.
        _JUNK_EMAIL_INITIALS = {"hii", "hi", "hey", "hello", "test", "linkedin", "msg", "message", "helo", "hello!", "hi!"}
        if not campaign_id:
            initial_msg_text = None

        if initial_msg_text:
            _clean_init = initial_msg_text.strip()
            if len(_clean_init) < 12 or _clean_init.lower() in _JUNK_EMAIL_INITIALS:
                initial_msg_text = None

        def _prepend_initial_email(msgs):
            if not initial_msg_text: return msgs
            clean_init = initial_msg_text.strip()
            # Already handled earlier
            entry = {
                "type": "SENT",
                "email_body": clean_init,
                "time": initial_msg_at.isoformat() if initial_msg_at else None,
                "sender_name": "You",
            }
            if not msgs: return [entry]
            init_key = _msg_dedup_key(clean_init)
            if any(_msg_dedup_key(m.get("email_body")) == init_key for m in msgs):
                return msgs
            return [entry] + msgs

        if (not is_stale or already_refreshing) and not force:
            return {'messages': _prepend_initial_email(cached['messages'] if cached else []), 'syncing': already_refreshing}

        if not already_refreshing and (is_stale or not cached):
            # Try instant restore from DB cache column
            if not cached and db_cache:
                try:
                    msgs_from_db = db_cache
                    if isinstance(msgs_from_db, str): msgs_from_db = json.loads(msgs_from_db)
                    if msgs_from_db:
                        print(f"DEBUG: Instantly restored {len(msgs_from_db)} email msgs from DB cache for cand {candidate_id}")
                        with _email_chat_lock:
                            _email_chat_cache[candidate_id] = {
                                "messages": msgs_from_db,
                                "ts": time.monotonic() - (_EMAIL_CACHE_TTL / 2),
                                "refreshing": False
                            }
                            cached = _email_chat_cache[candidate_id]
                except Exception as e:
                    print(f"WARNING: Email DB cache restore failed: {e}")

            # Kick off background refresh if we have a campaign
            if campaign_id:
                with _email_chat_lock:
                    if candidate_id not in _email_chat_cache:
                        _email_chat_cache[candidate_id] = {"messages": [], "ts": 0, "refreshing": True}
                    else:
                        _email_chat_cache[candidate_id]["refreshing"] = True

                t = threading.Thread(
                    target=_refresh_email_cache_task,
                    args=(candidate_id, email, campaign_id),
                    daemon=True
                )
                t.start()
                print(f"DEBUG: Background Smartlead refresh started for cand {candidate_id}")

        # Remember the SANITIZED initial message on the cache entry so the hot
        # path above can serve future polls without touching the DB. Create an
        # entry even for empty conversations — polling them repeatedly was one
        # of the main sources of per-poll DB latency.
        with _email_chat_lock:
            entry = _email_chat_cache.get(candidate_id)
            if entry is None:
                entry = {"messages": [], "ts": time.monotonic(), "refreshing": False}
                _email_chat_cache[candidate_id] = entry
            entry["initial"] = initial_msg_text.strip() if initial_msg_text else None
            entry["initial_at"] = initial_msg_at

        # Return whatever we have in cache (or empty) + syncing flag
        current_messages = cached["messages"] if cached else []
        final_msgs = _prepend_initial_email(current_messages)
        return {
            "messages": final_msgs,
            "syncing": not cached or is_stale or (cached and cached.get("refreshing")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Email chat history: {e}"
        )


@router.post("/reply/email/{role_id}/{candidate_id}")
async def send_email_chat_reply(
    role_id: int,
    candidate_id: int,
    request: schemas.ChatReplyRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Send a reply to a lead's email thread"""
    return await send_chat_reply(role_id, candidate_id, request, current_user)


@router.post("/reply/linkedin/{role_id}/{candidate_id}")
async def send_linkedin_chat_reply(
    role_id: int,
    candidate_id: int,
    request: schemas.ChatReplyRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Send a LinkedIn message reply"""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
                row = cur.fetchone()

        if not row or not row[0]:
            raise HTTPException(
                status_code=404, detail="LinkedIn URL not found for candidate"
            )

        profile_url = row[0]

        conv_id = None
        campaign_id = None
        with get_db_connection_context(validate=False, register_pgvector=False) as conn2:
            if conn2:
                with conn2.cursor() as cur2:
                    # If role_id is 0 (Talent Pool), try specific filter, but allow fallback
                    role_where, role_params = _role_filter_sql(
                        role_id, "recruitment_role_id"
                    )
                    cur2.execute(
                        f"""
                        SELECT heyreach_campaign_id, li_conversation_id
                        FROM candidate_outreach
                        WHERE candidate_id = %s AND {role_where}
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (candidate_id, *role_params),
                    )
                    cached = cur2.fetchone()

                    if not cached and role_id == 0:
                        # Talent Pool fallback: find ANY recent conversation for this candidate
                        cur2.execute(
                            """
                            SELECT heyreach_campaign_id, li_conversation_id
                            FROM candidate_outreach
                            WHERE candidate_id = %s AND li_conversation_id IS NOT NULL
                            ORDER BY updated_at DESC
                            LIMIT 1
                            """,
                            (candidate_id,),
                        )
                        cached = cur2.fetchone()

                    if cached:
                        campaign_id = cached[0]
                        conv_id = cached[1]

        bot = HeyReachBot()
        print(
            f"DEBUG: Replying via LinkedIn to candidate {candidate_id} for role {role_id}. Resolved conv_id: {conv_id}"
        )
        success = bot.send_li_message(
            profile_url,
            request.message,
            conversation_id=conv_id,
            campaign_id=int(campaign_id) if campaign_id else None,
        )

        if success:
            # Record initial message if first time (optional but keeps patterns consistent)
            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn3:
                    if conn3:
                        with conn3.cursor() as cur3:
                            role_where, role_params = _role_filter_sql(role_id, "recruitment_role_id")
                            cur3.execute(
                                f"""
                                UPDATE candidate_outreach
                                SET initial_li_message = %s,
                                    initial_li_message_at = NOW(),
                                    updated_at = NOW()
                                WHERE candidate_id = %s AND {role_where}
                                  AND initial_li_message IS NULL
                                """,
                                (request.message, candidate_id, *role_params)
                            )
                            conn3.commit()
            except:
                pass

            with _li_chat_lock:
                if candidate_id in _li_chat_cache:
                    _li_chat_cache[candidate_id]["ts"] = (
                        0  # Invalidate cache so next fetch is fresh
                    )
            return {"success": True}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to send LinkedIn message"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send LinkedIn reply: {e}"
        )


@router.post("/reply/{role_id}/{candidate_id}")
async def send_chat_reply(
    role_id: int,
    candidate_id: int,
    request: schemas.ChatReplyRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Send a reply to a lead (Default to Email)"""
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
                cur.execute(
                    f"""
                    SELECT c.email, co.campaign_id
                    FROM candidate_outreach co
                    JOIN candidates c ON c.id = co.candidate_id
                    WHERE co.candidate_id = %s AND {role_where}
                """,
                    (candidate_id, *role_params),
                )
                row = cur.fetchone()

        # Default behavior: try Email if campaign_id exists
        if row and row[1]:
            email, campaign_id = row
            bot = get_smartlead_bot()
            history = bot.get_chat_history(email, campaign_id)
            if history and isinstance(history, list):
                latest_msg = None
                for msg in history:
                    msg_type = str(msg.get("type", "")).upper()
                    if msg_type in ["INBOX", "REPLY"]:
                        latest_msg = msg
                        break
                if not latest_msg and history:
                    latest_msg = history[0]

                if latest_msg:
                    estats_id = latest_msg.get("email_stats_id") or latest_msg.get(
                        "stats_id"
                    )
                    res = bot.reply_to_email_thread(
                        campaign_id=campaign_id,
                        email_stats_id=str(estats_id) if estats_id else None,
                        message=request.message,
                        reply_message_id=str(latest_msg.get("message_id")),
                        reply_email_time=latest_msg.get("time")
                        or latest_msg.get("created_at"),
                        reply_email_body=latest_msg.get("email_body"),
                    )
                    if res:
                        return {"success": True}

        # Fallback: Campaign doesn't exist OR it exists but has no history yet. Trigger a new one.
        candidates = get_candidate_details([candidate_id])
        if not candidates:
            raise HTTPException(status_code=404, detail="Candidate details not found")
        cand = candidates[0]

        bot = get_smartlead_bot()
        sender_email = os.getenv("SMARTLEAD_SENDER_EMAIL")
        timezone = os.getenv("SMARTLEAD_DEFAULT_TIMEZONE", "Asia/Kolkata")
        campaign_name = f"Quick Chat - {cand['name']}"

        campaign_id = bot.create_campaign(campaign_name)
        if not campaign_id:
            raise HTTPException(
                status_code=500, detail="Failed to create Smartlead campaign"
            )

        bot.add_email_account(sender_email)
        subject = "Following up regarding your profile"
        bot.set_email_sequence(subject, request.message)

        start_time = datetime.now(tz_module.utc) + timedelta(minutes=1)

        bot.set_schedule(
            tz=timezone,
            start_hour="00:00",
            end_hour="23:59",
            start_time=start_time,
            days_of_the_week=[0, 1, 2, 3, 4, 5, 6],
        )
        bot.update_campaign_settings(follow_up_percentage=50)
        bot.add_leads(
            [
                {
                    "first_name": cand["first_name"],
                    "last_name": cand["last_name"],
                    "email": cand["email"],
                }
            ]
        )
        bot.start_campaign()

        # Record in DB
        with get_db_connection_context(validate=False, register_pgvector=False) as conn2:
            if conn2:
                with conn2.cursor() as cur2:
                    role_where, role_params = _role_filter_sql(
                        role_id, "recruitment_role_id"
                    )
                    cur2.execute(
                        f"SELECT 1 FROM candidate_outreach WHERE candidate_id = %s AND {role_where}",
                        (candidate_id, *role_params),
                    )
                    if cur2.fetchone():
                        cur2.execute(
                            f"""
                            UPDATE candidate_outreach
                            SET campaign_id = %s,
                                campaign_name = %s,
                                status = 'in_campaign',
                                initial_message = %s,
                                initial_message_at = NOW(),
                                updated_at = NOW()
                            WHERE candidate_id = %s AND {role_where}
                        """,
                            (
                                campaign_id,
                                campaign_name,
                                request.message,
                                candidate_id,
                                *role_params,
                            ),
                        )
                    else:
                        # Use NULL for recruitment_role_id if role_id is 0
                        real_role_id = role_id if role_id != 0 else None
                        cur2.execute(
                            """
                            INSERT INTO candidate_outreach
                            (candidate_id, recruitment_role_id, campaign_id, campaign_name, status, initial_message, initial_message_at, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, 'in_campaign', %s, NOW(), NOW(), NOW())
                        """,
                            (
                                candidate_id,
                                real_role_id,
                                campaign_id,
                                campaign_name,
                                request.message,
                            ),
                        )
                    conn2.commit()
                    return {"success": True, "triggered": True}

        raise HTTPException(
            status_code=400,
            detail="Could not identify suitable email thread or trigger new outreach",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send reply: {e}")


def _build_outreach_update_fields(upd: dict) -> list:
    """SET fragments for the candidate_outreach sync UPDATE.

    Email and LinkedIn are INDEPENDENT channels. The LinkedIn block used to be
    nested inside `if "status" in upd`, and `status` is only ever set by the
    Smartlead/email branch — so a candidate with a HeyReach campaign but no
    Smartlead campaign_id had their reply silently discarded, while the caller
    still counted the row as synced. That is the majority case here: 167 of 390
    candidate_outreach rows are LinkedIn-only.

    Extracted as a pure function so this stays testable without a database.
    """
    fields = ["updated_at = NOW()"]

    if "status" in upd:
        fields += [
            "status = %(status)s",
            "message_sent_count = %(message_sent_count)s",
            "last_message_sent_at = %(last_message_sent_at)s",
            "response_received_at = %(response_received_at)s",
            "response_text = %(response_text)s",
        ]

    if "li_status" in upd:
        # Never downgrade a known reply. The batch get_campaign_activities path
        # can return an entry with no reply data, which would otherwise flip a
        # 'replied' row back to 'message_sent'. Postgres evaluates the CASE
        # against the pre-UPDATE row, so this needs no extra query.
        fields.append(
            "li_status = CASE WHEN li_status = 'replied' THEN 'replied' ELSE %(li_status)s END"
        )
        fields += [
            "li_last_action_at = %(li_last_action_at)s",
            "li_sent_count = %(li_sent_count)s",
        ]
        # Only touch the reply columns when there is actually a reply. The batch
        # path carries no reply_at and may carry no text, and writing NULL over a
        # previously-synced reply is exactly how a responded candidate reverts to
        # reading "No response yet".
        if str(upd.get("li_response_text") or "").strip():
            fields += [
                "li_response_text = %(li_response_text)s",
                "li_response_received_at = COALESCE(%(li_response_received_at)s, li_response_received_at)",
            ]
        if "li_conversation_id" in upd:
            fields.append("li_conversation_id = %(li_conversation_id)s")
        if "li_account_id" in upd:
            fields.append("li_account_id = %(li_account_id)s")

    return fields


@router.post("/sync-responses/{role_id}")
def sync_responses(
    role_id: int, current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Sync responses from both Smartlead (Email) and HeyReach (LinkedIn)
    """
    candidates_data = []
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
                # Fetch both Smartlead and HeyReach identifiers
                cur.execute(
                    f"""
                    SELECT co.candidate_id, c.email, co.campaign_id, c.linkedin, co.heyreach_campaign_id, co.li_conversation_id, co.li_account_id
                    FROM candidate_outreach co
                    JOIN candidates c ON c.id = co.candidate_id
                    WHERE {role_where}
                """,
                    role_params,
                )

                candidates_data = cur.fetchall()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching candidates: {e}")

    if not candidates_data:
        return {"updated_count": 0}

    sl_bot = get_smartlead_bot()
    hr_bot = get_heyreach_bot()
    updates = []

    # Cache for HeyReach campaign leads to avoid redundant API calls for multiple leads in same campaign
    hr_campaign_cache = {}

    print(f"Syncing responses for {len(candidates_data)} candidates...")

    for c_id, email, sl_campaign_id, linkedin, hr_campaign_id, li_conv_id, li_acc_id in candidates_data:
        update_data = {"candidate_id": c_id, "role_id": role_id}
        has_update = False

        # 1. Sync Smartlead (Email)
        if sl_campaign_id:
            try:
                sl_bot.campaign_id = sl_campaign_id
                activity = sl_bot.get_lead_activity(email)
                if activity:
                    update_data["status"] = (
                        "replied" if activity["is_replied"] else "sent"
                    )
                    update_data["message_sent_count"] = activity["sent_count"]
                    update_data["last_message_sent_at"] = activity["last_sent_at"]
                    update_data["response_received_at"] = activity["reply_at"]
                    update_data["response_text"] = activity["reply_text"]
                    has_update = True
            except Exception as e:
                print(f"Error syncing Smartlead for {email}: {e}")

        # 2. Sync HeyReach (LinkedIn)
        if linkedin:
            # Synchronize responses for everyone
            try:
                activity = None
                if hr_campaign_id:
                    # Batch fetch for campaign to avoid rate limiting
                    hr_camp_int = int(hr_campaign_id)
                    if hr_camp_int not in hr_campaign_cache:
                        print(
                            f"DEBUG: Fetching all activities for HeyReach campaign {hr_camp_int}"
                        )
                        hr_campaign_cache[hr_camp_int] = hr_bot.get_campaign_activities(
                            hr_camp_int
                        )

                    norm_li = hr_bot._normalize_linkedin_url(linkedin)
                    if norm_li in hr_campaign_cache[hr_camp_int]:
                        activity = hr_campaign_cache[hr_camp_int][norm_li]

                # Fallback to single fetch if not in cache or no campaign_id
                if not activity:
                    print(f"DEBUG: Starting HeyReach single sync for {linkedin}")
                    import time

                    time.sleep(0.5)  # small sleep just for single fetches
                    activity = hr_bot.get_lead_activity(
                        linkedin,
                        campaign_id=int(hr_campaign_id) if hr_campaign_id else None,
                        conversation_id=li_conv_id,
                        account_id=int(li_acc_id) if li_acc_id else None,
                    )

                    # Update IDs if recovered
                    if activity and (activity.get("conversation_id") != li_conv_id or str(activity.get("account_id")) != str(li_acc_id)):
                        li_conv_id = activity.get("conversation_id")
                        li_acc_id = activity.get("account_id")

                if activity:
                    # ── NORMALIZE ACTIVITY KEYS ─────────────────────────────
                    # Handle keys from both get_campaign_activities and get_lead_activity
                    is_replied = activity.get("is_replied") or (activity.get("li_status") == "replied")
                    reply_text = activity.get("reply_text") or activity.get("li_response_text")
                    last_action_at = activity.get("last_sent_at") or activity.get("li_last_action_at")
                    sent_count = activity.get("sent_count") or 0
                    recov_conv_id = activity.get("conversation_id") or activity.get("li_conversation_id") or li_conv_id
                    recov_acc_id = activity.get("account_id") or activity.get("li_account_id") or li_acc_id

                    print(f"DEBUG: HeyReach activity found for {linkedin}: {activity}")
                    update_data["li_status"] = (
                        "replied" if is_replied else "message_sent"
                    )
                    update_data["li_sent_count"] = sent_count
                    update_data["li_last_action_at"] = last_action_at
                    update_data["li_response_text"] = reply_text
                    update_data["li_response_received_at"] = activity.get("reply_at")  # reply_at is only in get_lead_activity
                    update_data["li_conversation_id"] = recov_conv_id
                    update_data["li_account_id"] = recov_acc_id
                    has_update = True

                    # Invalidate chat cache if activity was found
                    with _li_chat_lock:
                        if c_id in _li_chat_cache:
                            _li_chat_cache[c_id]["ts"] = 0
            except Exception as e:
                print(f"Error syncing HeyReach for {linkedin}: {e}")

        if has_update:
            updates.append(update_data)

    # 3. Update Database
    updated_count = 0
    if updates:
        try:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if not conn:
                    raise RuntimeError("Database connection failed")

                with conn.cursor() as cur:
                    for upd in updates:
                        # Build dynamic update statement based on what was fetched
                        fields = _build_outreach_update_fields(upd)
                        params = upd

                        # Only `updated_at` means we learned nothing worth
                        # storing. Skip the write rather than issuing a no-op and
                        # reporting it to the user as a synced response.
                        if len(fields) == 1:
                            continue

                        if role_id == 0:
                            role_where_update = "recruitment_role_id IS NULL"
                        else:
                            role_where_update = f"recruitment_role_id = {int(role_id)}"
                        sql = f"""
                            UPDATE candidate_outreach
                            SET {", ".join(fields)}
                            WHERE candidate_id = %(candidate_id)s AND {role_where_update}
                        """
                        cur.execute(sql, params)
                        # Count rows we actually persisted. Incrementing
                        # unconditionally reported "Synced N new responses" even
                        # when the WHERE matched nothing or the write was a no-op.
                        if cur.rowcount:
                            updated_count += 1
                conn.commit()
        except Exception as e:
            print(f"Error updating database: {e}")

        # Keep the in-memory talent pool cache in sync so browse results reflect fresh outreach data.
        try:
            from backend.pipeline.query import update_profile_cache
            from backend.api.routes.browse import _invalidate_browse_cache

            for upd in updates:
                cache_payload = {}
                if "status" in upd:
                    cache_payload["response"] = upd.get("response_text") or ""
                if "li_status" in upd:
                    cache_payload["li_status"] = upd.get("li_status") or ""
                    cache_payload["li_response_text"] = (
                        upd.get("li_response_text") or ""
                    )
                    cache_payload["li_sent_count"] = upd.get("li_sent_count") or 0
                if cache_payload:
                    update_profile_cache(upd["candidate_id"], cache_payload)

            # Invalidate browse result cache so next request reflects updated data
            if updates:
                _invalidate_browse_cache()
        except Exception as cache_e:
            print(f"Warning: could not update in-memory outreach cache: {cache_e}")

    return {"updated_count": updated_count}


@router.post("/heyreach/trigger")
async def trigger_heyreach_outreach(
    request: HeyReachTriggerRequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """
    Trigger HeyReach LinkedIn sequence for selected candidates
    """
    # 1. Fetch candidate details (need LinkedIn URL)
    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, first_name, last_name, name, linkedin
                    FROM candidates
                    WHERE id = ANY(%s)
                      AND linkedin IS NOT NULL
                      AND LOWER(TRIM(COALESCE(status, ''))) = 'shortlisted'
                """,
                    (request.candidate_ids,),
                )

                candidates = []
                for row in cur.fetchall():
                    candidates.append(
                        {
                            "id": row[0],
                            "first_name": row[1] or row[3].split()[0],
                            "last_name": row[2]
                            or (row[3].split()[1] if len(row[3].split()) > 1 else ""),
                            "linkedin": row[4],
                        }
                    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {e}")

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No shortlisted candidates with valid LinkedIn profiles found",
        )

    print(f"DEBUG: Found {len(candidates)} candidates for HeyReach outreach")
    # 2. Push Leads to HeyReach

    bot = get_heyreach_bot()

    campaign_id = request.campaign_id
    if (not campaign_id or campaign_id <= 0) and request.role_name:
        found_id = bot.find_campaign_by_name(request.role_name)
        if found_id:
            print(
                f"DEBUG: Found matching HeyReach campaign '{request.role_name}' with ID {found_id}"
            )
            campaign_id = found_id

    if not campaign_id or campaign_id <= 0:
        raise HTTPException(
            status_code=400, detail="A valid HeyReach campaign ID is required"
        )

    success_count = 0

    for candidate in candidates:
        res = bot.push_lead(
            campaign_id=campaign_id,
            account_id=request.sender_account_id,
            first_name=candidate["first_name"],
            last_name=candidate["last_name"],
            profile_url=candidate["linkedin"],
        )
        print(f"DEBUG: HeyReach push result for candidate {candidate['id']}: {res}")
        if res is not None:
            success_count += 1

            # Handle role_id = 0 for Talent Pool context (save as NULL in DB)
            db_role_id = request.role_id if (request.role_id or 0) > 0 else None

            # Record in DB
            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                    if not conn:
                        raise RuntimeError("Database connection failed")
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO candidate_outreach
                            (candidate_id, recruitment_role_id, heyreach_campaign_id, li_status, created_at, updated_at)
                            VALUES (%s, %s, %s, 'in_campaign', NOW(), NOW())
                            ON CONFLICT (candidate_id, recruitment_role_id)
                            DO UPDATE SET
                                heyreach_campaign_id = EXCLUDED.heyreach_campaign_id,
                                li_status = 'in_campaign',
                                updated_at = NOW()

                        """,
                            (candidate["id"], db_role_id, str(campaign_id)),
                        )
                        conn.commit()
            except Exception as e:
                print(f"Error recording HeyReach outreach: {e}")
                continue

            # Synchronize cache after the DB write succeeds.
            try:
                from backend.pipeline.query import update_profile_cache

                update_profile_cache(
                    candidate["id"],
                    {
                        "heyreach_campaign_id": str(campaign_id),
                        "li_status": "in_campaign",
                    },
                )
            except Exception as cache_e:
                print(f"Warning: could not update profile cache: {cache_e}")

    if success_count == 0:
        raise HTTPException(
            status_code=502,
            detail="Failed to add any candidates to the HeyReach campaign",
        )

    return {
        "success": True,
        "processed_count": len(candidates),
        "success_count": success_count,
    }


@router.get("/heyreach/find-campaign/{role_name}")
async def find_heyreach_campaign(
    role_name: str, current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Find a HeyReach campaign ID by its name
    """
    bot = get_heyreach_bot()
    campaign_id = bot.find_campaign_by_name(role_name)
    if campaign_id:
        return {"campaign_id": campaign_id}
    raise HTTPException(
        status_code=404, detail=f"No campaign found matching '{role_name}'"
    )


@router.post("/smartlead/webhook")
async def smartlead_webhook(payload: Dict):
    """Real-time email replies from Smartlead.

    Counterpart to /heyreach/webhook. Without this, `response_text` (the column
    the candidate list reads for email) is only ever written by someone pressing
    "Sync Responses", so a replied-to candidate can read as "No response yet"
    indefinitely.

    Register with:
        POST https://server.smartlead.ai/api/v1/webhook/create?api_key=...
    selecting the EMAIL_REPLY event and pointing at
        {public_url}/api/outreach/smartlead/webhook

    Smartlead retries 5xx at 1min / 5min / 30min and treats 4xx as permanent, so
    this returns 200 for anything it cannot act on — an unmatched lead is not a
    delivery failure and must not be retried for half an hour.
    """
    # The payload shape is documented but unverified against this account, so
    # log the keys of the first events to confirm before trusting any field.
    logger.info("[SmartleadWebhook] event=%s keys=%s",
                payload.get("event_type"), sorted(payload.keys()))

    event = str(payload.get("event_type") or "").strip().upper()
    if event != "EMAIL_REPLY":
        return {"status": "ignored", "reason": f"unhandled_event:{event or 'none'}"}

    # Smartlead sends the LEAD's address as to_email on a reply event.
    lead_email = str(payload.get("to_email") or "").strip().lower()
    if not lead_email:
        return {"status": "ignored", "reason": "no_lead_email"}

    # preview_text is already plain; reply_body is HTML.
    reply_text = str(payload.get("preview_text") or "").strip()
    if not reply_text:
        raw = str(payload.get("reply_body") or "")
        reply_text = re.sub(r"<[^<]+?>", "", raw).strip()
    replied_at = payload.get("time_replied")

    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return {"status": "error", "reason": "db_connection_failed"}
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM candidates WHERE LOWER(TRIM(email)) = %s LIMIT 1",
                    (lead_email,),
                )
                row = cur.fetchone()
                if not row:
                    return {"status": "ignored", "reason": "candidate_not_found"}
                candidate_id = row[0]

                # Promote only — a malformed event must never blank a stored reply.
                cur.execute(
                    """
                    UPDATE candidate_outreach
                       SET status = 'replied',
                           response_text = COALESCE(NULLIF(%s, ''), response_text),
                           response_received_at = COALESCE(%s::timestamptz, response_received_at, NOW()),
                           updated_at = NOW()
                     WHERE candidate_id = %s
                    """,
                    (reply_text, replied_at, candidate_id),
                )
                updated = cur.rowcount
            conn.commit()

        # Drop the cached thread so the conversation modal shows the reply now
        # rather than after the 300s TTL.
        with _email_chat_lock:
            if candidate_id in _email_chat_cache:
                _email_chat_cache[candidate_id]["ts"] = 0

        try:
            from backend.api.routes.browse import _invalidate_browse_cache
            _invalidate_browse_cache()
        except Exception:
            pass

        logger.info("[SmartleadWebhook] reply stored for candidate %s (%d rows)",
                    candidate_id, updated)
        return {"status": "ok", "candidate_id": candidate_id, "updated": updated}
    except Exception as exc:
        logger.exception("[SmartleadWebhook] failed")
        # 200 with an error body: a 5xx would put Smartlead into a 30-minute
        # retry cycle for what is most likely a permanent parsing problem.
        return {"status": "error", "reason": str(exc)}


@router.post("/heyreach/webhook")
async def heyreach_webhook(request: Dict):
    """
    Handle webhook events from HeyReach
    """
    # Event Mapping
    event = request.get("event_type") or request.get("type") or "unknown_event"
    lead_data = request.get("lead", {})
    if not lead_data:
        lead_data = request  # Fallback

    profile_url = lead_data.get("profile_url") or lead_data.get("profileUrl")

    conv_id_early = request.get("conversation_id") or request.get("conversationId")

    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                return {"status": "error", "reason": "db_connection_failed"}
            with conn.cursor() as cur:
                candidate_id = None
                if conv_id_early:
                    cur.execute(
                        """
                        SELECT candidate_id FROM candidate_outreach
                        WHERE li_conversation_id = %s
                        LIMIT 1
                        """,
                        (str(conv_id_early),),
                    )
                    cr = cur.fetchone()
                    if cr:
                        candidate_id = cr[0]

                if candidate_id is None and profile_url:
                    norm = normalize_linkedin(profile_url)
                    cur.execute(
                        """
                        SELECT c.id FROM candidates c
                        LEFT JOIN candidate_outreach co
                          ON co.candidate_id = c.id AND co.recruitment_role_id IS NULL
                        WHERE (c.normalized_linkedin = %s OR c.linkedin = %s)
                          AND COALESCE(c.is_archived, FALSE) = FALSE
                        ORDER BY
                          CASE WHEN co.li_conversation_id IS NOT NULL THEN 0 ELSE 1 END,
                          c.id
                        LIMIT 1
                        """,
                        (norm, profile_url),
                    )
                    candidate_row = cur.fetchone()
                    if candidate_row:
                        candidate_id = candidate_row[0]

                if candidate_id is None:
                    return {"status": "ignored", "reason": "candidate_not_found"}

                new_status = None
                new_response = None

                event_lower = event.lower()

                # Check if it's explicitly a reply
                if "reply" in event_lower or "replied" in event_lower:
                    new_status = "replied"
                    # Extract message
                    recent = request.get("recent_messages", [])
                    if recent and isinstance(recent, list):
                        last_msg = recent[-1]
                        # Trust is_reply flag if it exists, otherwise assume the latest is the reply
                        if last_msg.get("is_reply", True):
                            new_response = last_msg.get("message", "")
                    else:
                        new_response = request.get("messageText") or request.get("message")

                elif "message" in event_lower and "sent" in event_lower:
                    new_status = "message_sent"

                elif "connection" in event_lower:
                    if "accepted" in event_lower:
                        new_status = "connection_accepted"
                    else:
                        new_status = "connection_sent"

                elif "action" in event_lower:
                    action = request.get("actionType", "")
                    if "message" in action.lower() or "send" in action.lower():
                        new_status = "message_sent"

                # Extract Identifiers from webhook for future direct sync optimization
                conv_id = request.get("conversation_id") or request.get("conversationId")
                acc_id = request.get("accountId") or request.get("linkedInAccountId")

                # Invalidate in-memory chat cache immediately so UI refresh shows the reply NOW.
                with _li_chat_lock:
                    if candidate_id in _li_chat_cache:
                        _li_chat_cache[candidate_id]["ts"] = 0
                        print(f"DEBUG: Webhook invalidated LI chat cache for cand {candidate_id} due to event: {event}")

                if new_status:
                    # Update candidate_outreach
                    # Since we don't have role_id in webhook, we update all entries for this candidate
                    fields = [
                        "li_status = %s",
                        "li_last_action_at = NOW()",
                        "updated_at = NOW()"
                    ]
                    params = [new_status]

                    if new_response:
                        fields.append("li_response_text = %s")
                        params.append(new_response)

                    if conv_id:
                        fields.append("li_conversation_id = %s")
                        params.append(str(conv_id))

                    if acc_id:
                        fields.append("li_account_id = %s")
                        params.append(str(acc_id))

                    update_sql = f"UPDATE candidate_outreach SET {', '.join(fields)} WHERE candidate_id = %s"
                    params.append(candidate_id)

                    cur.execute(update_sql, tuple(params))
                    conn.commit()

        return {"status": "success"}
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"status": "error", "reason": str(e)}

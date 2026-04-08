import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone as tz_module
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.api import schemas, deps
from backend.db.connection import get_db_connection, return_db_connection
from backend.integrations.smartlead import SmartleadBot
from backend.integrations.heyreach import HeyReachBot

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory LinkedIn chat cache
# Structure: {candidate_id: {messages, ts, refreshing}}
# Messages are cached per candidate to avoid redundant API calls
_li_chat_cache: Dict[int, Dict] = {}  # {candidate_id: {messages, ts, refreshing}}
_li_chat_lock = threading.Lock()
_LI_CACHE_TTL = 300  # 5 minutes - increased from 60s for better performance
_LI_CACHE_STALE_THRESHOLD = 120  # 2 minutes - trigger background refresh after this


def _refresh_li_cache(
    candidate_id: int,
    profile_url: str,
    campaign_id: Optional[int],
    conversation_id: Optional[str],
):
    """Background thread: fetch fresh messages and update cache."""
    bot = HeyReachBot()
    messages = []
    try:
        messages = bot.get_li_chat_history(
            profile_url, campaign_id=campaign_id, conversation_id=conversation_id
        )
        # Clean messages before caching
        if messages:
            for msg in messages:
                raw_body = msg.get("email_body", "")
                if raw_body:
                    msg["email_body"] = _clean_email_body(raw_body)
    except Exception as e:
        print(
            f"WARNING: Background LI cache refresh failed for cand {candidate_id}: {e}"
        )
    finally:
        with _li_chat_lock:
            # We store the monotonic timestamp for TTL and the count for verification
            _li_chat_cache[candidate_id] = {
                "messages": messages or [],
                "ts": time.monotonic(),
                "refreshing": False,
                "db_updated_at": datetime.now(
                    tz_module.utc
                ),  # Track when this was fetched
            }
        print(
            f"DEBUG: LI cache refreshed for cand {candidate_id}: {len(messages or [])} msgs"
        )


def _prewarm_single(candidate_id: int):
    """Fetch outreach info from DB and pre-warm LinkedIn chat cache for one candidate."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
        cand_row = cur.fetchone()
        if not cand_row or not cand_row[0]:
            cur.close()
            return_db_connection(conn)
            return
        profile_url = cand_row[0]
        cur.execute(
            """
            SELECT heyreach_campaign_id, li_conversation_id
            FROM candidate_outreach
            WHERE candidate_id = %s
            ORDER BY updated_at DESC LIMIT 1
        """,
            (candidate_id,),
        )
        row = cur.fetchone()
        cur.close()
        return_db_connection(conn)
        campaign_id = int(row[0]) if row and row[0] else None
        conv_id = row[1] if row and len(row) > 1 else None

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

        _refresh_li_cache(candidate_id, profile_url, campaign_id, conv_id)
    except Exception as e:
        print(f"WARNING: prewarm failed for cand {candidate_id}: {e}")
        if conn:
            try:
                return_db_connection(conn)
            except Exception:
                pass


# --- Request/Response Models ---


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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
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

        cur.close()
        cur.close()
        return_db_connection(conn)
        return candidates
    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {e}")


# --- API Endpoints ---


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
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
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
            cur.close()
            conn.commit()
            cur.close()
            return_db_connection(conn)
        except Exception as e:
            if conn:
                return_db_connection(conn)
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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # Step 1: Fetch candidate from DB
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, first_name, last_name, email, mobile_phone, linkedin FROM candidates WHERE id = %s",
            (candidate_id,),
        )
        row = cur.fetchone()
        cur.close()
        return_db_connection(conn)
    except Exception as e:
        if conn:
            return_db_connection(conn)
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
        conn = get_db_connection()
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
    finally:
        if conn:
            return_db_connection(conn)

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
                    conn = get_db_connection()
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
                finally:
                    if conn:
                        return_db_connection(conn)
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
                conn2 = get_db_connection()
                if conn2:
                    try:
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
                    finally:
                        return_db_connection(conn2)
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
                conn3 = get_db_connection()
                if conn3 and linkedin_outreach == "started":
                    try:
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
                    finally:
                        return_db_connection(conn3)
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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
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
                li_conversation_id
            FROM candidate_outreach
            WHERE {role_where}
        """,
            role_params,
        )

        statuses = {}
        for row in cur.fetchall():
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
            }

        cur.close()
        return_db_connection(conn)
        return statuses
    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {e}")


@router.get("/chat/email/{role_id}/{candidate_id}")
async def get_email_chat_history(
    role_id: int,
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured Email chat history for a candidate"""
    return await get_chat_history(role_id, candidate_id, current_user)


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
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured LinkedIn chat history for a candidate.

    Uses an in-memory cache so the response is instant on repeat opens.
    A background thread always refreshes the cache after serving.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
        cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
        cand_row = cur.fetchone()

        if role_id == 0:
            cur.execute(
                """
                SELECT updated_at, heyreach_campaign_id, li_conversation_id
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
                SELECT updated_at, heyreach_campaign_id, li_conversation_id
                FROM candidate_outreach
                WHERE candidate_id = %s AND {role_where}
                ORDER BY heyreach_campaign_id DESC NULLS LAST, updated_at DESC LIMIT 1
            """,
                (candidate_id, *role_params),
            )

        outreach_row = cur.fetchone()
        print(f"DEBUG: Outreach record for cand {candidate_id}: {outreach_row}")
        cur.close()
        return_db_connection(conn)

        if not cand_row or not cand_row[0]:
            return {"messages": []}

        profile_url = cand_row[0]
        heyreach_campaign_id = (
            outreach_row[1] if outreach_row and len(outreach_row) > 1 else None
        )
        li_conversation_id = (
            outreach_row[2] if outreach_row and len(outreach_row) > 2 else None
        )
        campaign_id_int = int(heyreach_campaign_id) if heyreach_campaign_id else None

         # ---------------------------------------------------------------
         # Cache logic - optimized for performance
         # ---------------------------------------------------------------
         # Strategy: Return cached data ASAP, refresh in background if needed
         # Two-tier cache: Fresh (< 2min) returns immediately, Stale (2-5min) returns + background refresh
         db_updated_at = outreach_row[0] if outreach_row else None

         with _li_chat_lock:
             cached = _li_chat_cache.get(candidate_id)
             cache_ts = cached.get("ts", 0) if cached else 0
             cache_age = time.monotonic() - cache_ts

             # Two-tier strategy:
             # is_very_stale: cache > 5min old, needs synchronous fetch
             # is_stale: cache 2-5min old, return cached + trigger background refresh
             is_very_stale = cache_age > _LI_CACHE_TTL
             is_stale = cache_age > _LI_CACHE_STALE_THRESHOLD
             already_refreshing = cached and cached.get("refreshing", False)

        if already_refreshing:
            # For polling UI: If we are already refreshing, return what we have (stale or empty)
            # DO NOT wait in a loop, it makes the frontend feel sluggish/stuck.
            print(
                f"DEBUG: LI cache is currently refreshing for cand {candidate_id}. Serving current state."
            )
            return {"messages": cached["messages"] if cached else [], "syncing": True}

         if cached and not is_very_stale and not is_stale:
             # Fresh cache (< 2min) — return immediately, no refresh needed
             print(
                 f"DEBUG: LI cache HIT (age {cache_age:.0f}s) for cand {candidate_id}: {len(cached['messages'])} msgs"
             )
             return {"messages": cached["messages"]}

         if cached and is_stale and not is_very_stale:
             # Stale cache (2-5min) but HAS data — return immediately, trigger background refresh
             print(
                 f"DEBUG: LI cache STALE (age {cache_age:.0f}s) for cand {candidate_id}, refreshing in background"
             )
             with _li_chat_lock:
                 _li_chat_cache[candidate_id]["refreshing"] = True

             t = threading.Thread(
                 target=_refresh_li_cache,
                 args=(candidate_id, profile_url, campaign_id_int, li_conversation_id),
                 daemon=True,
             )
             t.start()
             return {"messages": cached["messages"], "syncing": True}

         # No cache or very stale — perform synchronous fetch and prime the cache
         print(f"DEBUG: LI cache MISS/VERY_STALE for cand {candidate_id}, fetching live...")
         with _li_chat_lock:
             # Prevent other requests from fetching simultaneously
             if candidate_id not in _li_chat_cache:
                 _li_chat_cache[candidate_id] = {
                     "messages": [],
                     "ts": 0,
                     "refreshing": True,
                 }
             else:
                 _li_chat_cache[candidate_id]["refreshing"] = True

         bot = HeyReachBot()
         messages = []
         try:
             messages = bot.get_li_chat_history(
                 profile_url,
                 campaign_id=campaign_id_int,
                 conversation_id=li_conversation_id,
             )
             messages = messages or []

             # Clean messages before returning/caching
             for msg in messages:
                 raw_body = msg.get("email_body", "")
                 if raw_body:
                     msg["email_body"] = _clean_email_body(raw_body)
         except Exception as e:
             print(
                 f"ERROR: Sync LinkedIn chat fetch failed for cand {candidate_id}: {e}"
             )
        finally:
            with _li_chat_lock:
                _li_chat_cache[candidate_id] = {
                    "messages": messages,
                    "ts": time.monotonic(),
                    "refreshing": False,
                }

        print(
            f"DEBUG: HeyReach returned {len(messages)} messages for candidate {candidate_id}"
        )
        return {"messages": messages}

    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch LinkedIn chat history: {e}"
        )


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

    return cleaned


@router.get("/chat/{role_id}/{candidate_id}")
async def get_chat_history(
    role_id: int,
    candidate_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured chat history (Default to Email)"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
        role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
        # Use ORDER BY + LIMIT 1 to get the most recent outreach row with a campaign_id if possible
        cur.execute(
            f"""
            SELECT c.email, co.campaign_id, co.initial_message, co.initial_message_at
            FROM candidate_outreach co
            JOIN candidates c ON c.id = co.candidate_id
            WHERE co.candidate_id = %s AND {role_where}
            ORDER BY co.campaign_id DESC NULLS LAST, co.updated_at DESC
            LIMIT 1
        """,
            (candidate_id, *role_params),
        )
        row = cur.fetchone()
        cur.close()
        return_db_connection(conn)

        if not row:
            raise HTTPException(
                status_code=404, detail="Candidate outreach record not found"
            )

        email, campaign_id, initial_msg_text, initial_msg_at = row
        print(f"DEBUG: Email chat — email={email}, campaign_id={campaign_id}")

        if not campaign_id:
            # No campaign yet — return initial message if stored
            if initial_msg_text:
                return {
                    "messages": [
                        {
                            "type": "SENT",
                            "email_body": initial_msg_text,
                            "time": initial_msg_at.isoformat()
                            if initial_msg_at
                            else None,
                            "sender_name": "You",
                        }
                    ]
                }
            return {"messages": []}

        bot = get_smartlead_bot()
        print(f"DEBUG: Fetching Email history for {email} (campaign_id: {campaign_id})")
        messages = bot.get_chat_history(email, campaign_id)
        print(f"DEBUG: Found {len(messages) if messages else 0} Emails")

        # Prepend the initial outreach message if Smartlead hasn't stored it yet
        if initial_msg_text:
            initial_entry = {
                "type": "SENT",
                "email_body": initial_msg_text,
                "time": initial_msg_at.isoformat() if initial_msg_at else None,
                "sender_name": "You",
            }
            if not messages:
                messages = [initial_entry]
            else:
                # Only prepend if no message with same text already exists (avoid duplicates)
                already_present = any(
                    (m.get("email_body") or "").strip() == initial_msg_text.strip()
                    for m in messages
                )
                if not already_present:
                    messages = [initial_entry] + messages

        if messages and isinstance(messages, list):
            for msg in messages:
                raw_body = msg.get("email_body", "")
                if raw_body:
                    cleaned_text = _clean_email_body(raw_body)
                    # For emails containing raw HTML entities (like <email@domain.com>) fix them so the frontend
                    # strips them cleanly without rendering literal "&lt;".
                    cleaned_text = (
                        cleaned_text.replace("&lt;", "<")
                        .replace("&gt;", ">")
                        .replace("&nbsp;", " ")
                        .replace("&amp;", "&")
                        .replace("&quot;", '"')
                    )
                    msg["email_body"] = cleaned_text

        return {"messages": messages or []}
    except HTTPException:
        if conn:
            return_db_connection(conn)
        raise
    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch chat history: {e}"
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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
        cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
        row = cur.fetchone()
        cur.close()
        return_db_connection(conn)

        if not row or not row[0]:
            raise HTTPException(
                status_code=404, detail="LinkedIn URL not found for candidate"
            )

        profile_url = row[0]

        conv_id = None
        campaign_id = None
        conn2 = get_db_connection()
        if conn2:
            try:
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
            finally:
                return_db_connection(conn2)

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
            return {"success": True}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to send LinkedIn message"
            )
    except Exception as e:
        if conn:
            return_db_connection(conn)
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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
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
        cur.close()
        return_db_connection(conn)

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
        conn2 = get_db_connection()
        if conn2:
            try:
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
            finally:
                return_db_connection(conn2)

        raise HTTPException(
            status_code=400,
            detail="Could not identify suitable email thread or trigger new outreach",
        )

    except HTTPException:
        if conn:
            return_db_connection(conn)
        raise
    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Failed to send reply: {e}")


@router.post("/sync-responses/{role_id}")
def sync_responses(
    role_id: int, current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Sync responses from both Smartlead (Email) and HeyReach (LinkedIn)
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    candidates_data = []
    try:
        cur = conn.cursor()
        role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
        # Fetch both Smartlead and HeyReach identifiers
        cur.execute(
            f"""
            SELECT co.candidate_id, c.email, co.campaign_id, c.linkedin, co.heyreach_campaign_id
            FROM candidate_outreach co
            JOIN candidates c ON c.id = co.candidate_id
            WHERE {role_where}
        """,
            role_params,
        )

        candidates_data = cur.fetchall()
        cur.close()
    except Exception as e:
        print(f"Error fetching candidates: {e}")
    finally:
        return_db_connection(conn)

    if not candidates_data:
        return {"updated_count": 0}

    sl_bot = get_smartlead_bot()
    hr_bot = get_heyreach_bot()
    updates = []

    # Cache for HeyReach campaign leads to avoid redundant API calls for multiple leads in same campaign
    hr_campaign_cache = {}

    print(f"Syncing responses for {len(candidates_data)} candidates...")

    for c_id, email, sl_campaign_id, linkedin, hr_campaign_id in candidates_data:
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
            try:
                print(f"DEBUG: Starting HeyReach sync for {linkedin}")
                activity = hr_bot.get_lead_activity(
                    linkedin,
                    campaign_id=int(hr_campaign_id) if hr_campaign_id else None,
                )
                if activity:
                    print(f"DEBUG: HeyReach activity found for {linkedin}: {activity}")
                    update_data["li_status"] = (
                        "replied" if activity["is_replied"] else "message_sent"
                    )
                    update_data["li_sent_count"] = activity["sent_count"]
                    update_data["li_last_action_at"] = activity["last_sent_at"]
                    update_data["li_response_text"] = activity["reply_text"]
                    update_data["li_response_received_at"] = activity["reply_at"]
                    update_data["li_conversation_id"] = activity["conversation_id"]
                    has_update = True
            except Exception as e:
                print(f"Error syncing HeyReach for {linkedin}: {e}")

        if has_update:
            updates.append(update_data)

    # 3. Update Database
    updated_count = 0
    if updates:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                for upd in updates:
                    # Build dynamic update statement based on what was fetched
                    fields = ["updated_at = NOW()"]
                    params = upd

                    if "status" in upd:
                        fields.append("status = %(status)s")
                        fields.append("message_sent_count = %(message_sent_count)s")
                        fields.append("last_message_sent_at = %(last_message_sent_at)s")
                        fields.append("response_received_at = %(response_received_at)s")
                        fields.append("response_text = %(response_text)s")

                    if "li_status" in upd:
                        fields.append("li_status = %(li_status)s")
                        fields.append("li_response_text = %(li_response_text)s")
                        fields.append("li_last_action_at = %(li_last_action_at)s")
                        fields.append("li_sent_count = %(li_sent_count)s")
                        fields.append(
                            "li_response_received_at = %(li_response_received_at)s"
                        )
                        fields.append("li_conversation_id = %(li_conversation_id)s")

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
                    updated_count += 1
            conn.commit()
        except Exception as e:
            print(f"Error updating database: {e}")
        finally:
            return_db_connection(conn)

        # Keep the in-memory talent pool cache in sync so browse results reflect fresh outreach data.
        try:
            from backend.pipeline.query import update_profile_cache

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
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, first_name, last_name, name, linkedin
            FROM candidates
            WHERE id = ANY(%s) AND linkedin IS NOT NULL
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
        cur.close()
        return_db_connection(conn)
    except Exception as e:
        if conn:
            return_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {e}")

    if not candidates:
        raise HTTPException(
            status_code=404, detail="No candidates with valid LinkedIn profiles found"
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
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
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
                    cur.close()
                    return_db_connection(conn)

                    # Synchronize cache
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

                except Exception as e:
                    print(f"Error recording HeyReach outreach: {e}")
                    if conn:
                        return_db_connection(conn)

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
    if not profile_url:
        return {"status": "ignored", "reason": "no_profile_url"}

    # Find candidate by LinkedIn URL
    conn = get_db_connection()
    if not conn:
        return {"status": "error", "reason": "db_connection_failed"}

    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM candidates WHERE linkedin = %s", (profile_url,))
        candidate_row = cur.fetchone()
        if not candidate_row:
            cur.close()
            return_db_connection(conn)
            return {"status": "ignored", "reason": "candidate_not_found"}

        candidate_id = candidate_row[0]

        new_status = None
        new_response = None

        event_lower = event.lower()
        if "reply" in event_lower or "message" in event_lower:
            new_status = "replied"
            # Extract message
            recent = request.get("recent_messages", [])
            if recent and isinstance(recent, list):
                last_msg = recent[-1]
                if last_msg.get("is_reply"):
                    new_response = last_msg.get("message", "")
            else:
                new_response = request.get("messageText") or request.get("message")

        elif "connection" in event_lower:
            if "accepted" in event_lower:
                new_status = "connection_accepted"
            else:
                new_status = "connection_sent"

        elif "action" in event_lower:
            action = request.get("actionType", "")
            if "message" in action.lower() or "send" in action.lower():
                new_status = "message_sent"

        if new_status:
            # Update candidate_outreach
            # Since we don't have role_id in webhook, we update all entries for this candidate
            # Or we could try to find the active campaign
            update_sql = """
                UPDATE candidate_outreach
                SET li_status = %s,
                    li_last_action_at = NOW(),
                    updated_at = NOW()
            """
            params = [new_status]

            if new_response:
                update_sql += ", li_response_text = %s"
                params.append(new_response)

            update_sql += " WHERE candidate_id = %s"
            params.append(candidate_id)

            cur.execute(update_sql, tuple(params))
            conn.commit()

        cur.close()
        return_db_connection(conn)
        return {"status": "success"}
    except Exception as e:
        print(f"Webhook error: {e}")
        if conn:
            return_db_connection(conn)
        return {"status": "error", "reason": str(e)}

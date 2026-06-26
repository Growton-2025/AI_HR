import re

with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

# 1. Fix get_linkedin_chat_history
def patch_li(m):
    original = m.group(0)
    # We want to insert the cache check AT THE VERY TOP of the function!
    new_code = '''async def get_linkedin_chat_history(
    role_id: int,
    candidate_id: int,
    force: bool = False,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured LinkedIn chat history for a candidate."""
    import time
    start_t = time.time()
    
    # ── HOT PATH CACHE CHECK ──
    # If we already have the messages and it's not stale, return IMMEDIATELY
    # without touching the database to avoid Azure PostgreSQL connection latency (~250ms+).
    with _li_chat_lock:
        cached = _li_chat_cache.get(candidate_id)
        if cached:
            cache_age = time.monotonic() - cached.get("ts", 0)
            is_stale = cache_age > _LI_CACHE_STALE_THRESHOLD
            already_refreshing = cached.get("refreshing", False)
            if not is_stale and not force:
                final_msgs = cached.get("messages", [])
                initial_li_message = cached.get("initial", None)
                initial_li_message_at = cached.get("initial_at", None)
                
                # Prepend initial if needed
                if initial_li_message:
                    has_sent = any(msg.get("type") == "SENT" for msg in final_msgs)
                    if not has_sent:
                        clean_init = initial_li_message.strip()
                        # Quick junk filter
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
                    
                return {
                    "messages": final_msgs,
                    "syncing": already_refreshing
                }

    try:
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            with conn.cursor() as cur:
                cur.execute("SELECT linkedin FROM candidates WHERE id = %s", (candidate_id,))
                cand_row = cur.fetchone()
                if not cand_row or not cand_row[0]:
                    raise HTTPException(status_code=404, detail="Candidate has no LinkedIn profile")
                profile_url = cand_row[0]

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
                    cur.execute(
                        """
                        SELECT updated_at, heyreach_campaign_id, li_conversation_id, initial_li_message, initial_li_message_at, li_account_id, li_response_text, response_received_at,
                               li_chat_history_cache
                        FROM candidate_outreach
                        WHERE candidate_id = %s AND recruitment_role_id = %s
                        ORDER BY updated_at DESC LIMIT 1
                    """,
                        (candidate_id, role_id),
                    )
                outreach_row = cur.fetchone()

        if not outreach_row:
            return {"messages": [], "syncing": False}

        db_updated_at = outreach_row[0]
        campaign_id_int = int(outreach_row[1]) if outreach_row[1] else None
        li_conversation_id = outreach_row[2]
        initial_li_message = outreach_row[3]
        initial_li_message_at = outreach_row[4]
        li_account_id_int = int(outreach_row[5]) if outreach_row[5] else None
        li_response_text = outreach_row[6]
        response_received_at = outreach_row[7]
        li_chat_history_cache = outreach_row[8] if len(outreach_row) > 8 else None

        with _li_chat_lock:
            cached = _li_chat_cache.get(candidate_id)
            cache_ts = cached.get("ts", 0) if cached else 0
            cache_age = time.monotonic() - cache_ts
            is_stale = cache_age > _LI_CACHE_STALE_THRESHOLD
            already_refreshing = cached and cached.get("refreshing", False)

            if force:
                if cached:
                    cached["refreshing"] = True
                is_stale = True
                already_refreshing = False

            # If not cached but DB has it, load from DB cache!
            if not cached and li_chat_history_cache:
                _li_chat_cache[candidate_id] = {
                    "messages": li_chat_history_cache,
                    "ts": time.monotonic(),
                    "refreshing": False,
                    "initial": initial_li_message,
                    "initial_at": initial_li_message_at
                }
                cached = _li_chat_cache[candidate_id]
                is_stale = False

            if not already_refreshing and (is_stale or not cached):
                if candidate_id not in _li_chat_cache:
                    _li_chat_cache[candidate_id] = {
                        "messages": [], 
                        "ts": 0, 
                        "refreshing": True,
                        "initial": initial_li_message,
                        "initial_at": initial_li_message_at
                    }
                else:
                    _li_chat_cache[candidate_id]["refreshing"] = True
                    _li_chat_cache[candidate_id]["initial"] = initial_li_message
                    _li_chat_cache[candidate_id]["initial_at"] = initial_li_message_at

                t = threading.Thread(
                    target=_refresh_li_cache_task,
                    args=(candidate_id, profile_url, campaign_id_int, li_conversation_id, li_account_id_int),
                    daemon=True,
                )
                t.start()

        current_messages = cached["messages"] if cached else []
        
        # Guard: junk initial messages
        _JUNK_LI_INITIALS = {"hii", "hi", "hey", "hello", "test", "linkedin", "msg", "message", "helo", "hello!", "hi!"}
        if initial_li_message:
            _clean_init = initial_li_message.strip()
            if len(_clean_init) < 12 or _clean_init.lower() in _JUNK_LI_INITIALS:
                initial_li_message = None

        def _prepend_initial(msgs: List[Dict]) -> List[Dict]:
            has_sent = any(m.get("type") == "SENT" for m in msgs)
            if initial_li_message and not has_sent:
                clean_init = initial_li_message.strip()
                entry = {
                    "type": "SENT",
                    "email_body": clean_init,
                    "time": initial_li_message_at.isoformat() if initial_li_message_at else None,
                    "sender_name": "You",
                }
                if not msgs:
                    return [entry]
                if any((m.get("email_body") or "").strip() == clean_init for m in msgs):
                    return msgs
                return [entry] + msgs
            return msgs

        final_msgs = _prepend_initial(current_messages)
        try:
            final_msgs.sort(key=lambda x: x.get("time", ""))
        except:
            pass

        result = {
            "messages": final_msgs,
            "syncing": not cached or is_stale or (cached and cached.get("refreshing")),
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching linkedin chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
    return new_code

code = re.sub(r'async def get_linkedin_chat_history\(.*?\):.*?raise HTTPException\(status_code=500, detail=str\(e\)\)', patch_li, code, flags=re.DOTALL)

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)
print("patched")

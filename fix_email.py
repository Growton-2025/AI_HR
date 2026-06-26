import re

with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

def patch_email(m):
    original = m.group(0)
    new_code = '''async def get_chat_history(
    role_id: int,
    candidate_id: int,
    force: bool = False,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """Fetch structured chat history (Default to Email)"""
    import time
    
    # ── HOT PATH CACHE CHECK ──
    with _email_chat_lock:
        cached = _email_chat_cache.get(candidate_id)
        if cached:
            cache_age = time.monotonic() - cached.get("ts", 0)
            is_stale = cache_age > _EMAIL_CACHE_STALE_THRESHOLD
            already_refreshing = cached.get("refreshing", False)
            if not is_stale and not force:
                final_msgs = cached.get("messages", [])
                initial_msg_text = cached.get("initial", None)
                initial_msg_at = cached.get("initial_at", None)
                
                if initial_msg_text:
                    clean_init = initial_msg_text.strip()
                    _JUNK_EMAIL_INITIALS = {"hii", "hi", "hey", "hello", "test", "linkedin", "msg", "message", "helo", "hello!", "hi!"}
                    if len(clean_init) >= 12 and clean_init.lower() not in _JUNK_EMAIL_INITIALS:
                        entry = {
                            "type": "SENT",
                            "email_body": clean_init,
                            "time": initial_msg_at.isoformat() if initial_msg_at else None,
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
        with get_db_connection_context(validate=False, register_pgvector=False) as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                role_where, role_params = _role_filter_sql(role_id, "co.recruitment_role_id")
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

        with _email_chat_lock:
            cached = _email_chat_cache.get(candidate_id)
            cache_ts = cached.get("ts", 0) if cached else 0
            cache_age = time.monotonic() - cache_ts
            is_stale = force or (cache_age > _EMAIL_CACHE_STALE_THRESHOLD)
            already_refreshing = cached and cached.get("refreshing", False)

            if force:
                if cached:
                    cached["refreshing"] = True
                is_stale = True
                already_refreshing = False

            if not cached and db_cache:
                _email_chat_cache[candidate_id] = {
                    "messages": db_cache,
                    "ts": time.monotonic(),
                    "refreshing": False,
                    "initial": initial_msg_text,
                    "initial_at": initial_msg_at
                }
                cached = _email_chat_cache[candidate_id]
                is_stale = False

            if not already_refreshing and (is_stale or not cached):
                if candidate_id not in _email_chat_cache:
                    _email_chat_cache[candidate_id] = {
                        "messages": [], 
                        "ts": 0, 
                        "refreshing": True,
                        "initial": initial_msg_text,
                        "initial_at": initial_msg_at
                    }
                else:
                    _email_chat_cache[candidate_id]["refreshing"] = True
                    _email_chat_cache[candidate_id]["initial"] = initial_msg_text
                    _email_chat_cache[candidate_id]["initial_at"] = initial_msg_at

                t = threading.Thread(
                    target=_refresh_email_cache_task,
                    args=(candidate_id, email, campaign_id),
                    daemon=True,
                )
                t.start()

        _JUNK_EMAIL_INITIALS = {"hii", "hi", "hey", "hello", "test", "linkedin", "msg", "message", "helo", "hello!", "hi!"}
        if not campaign_id:
            initial_msg_text = None

        if initial_msg_text:
            _clean_init = initial_msg_text.strip()
            if len(_clean_init) < 12 or _clean_init.lower() in _JUNK_EMAIL_INITIALS:
                initial_msg_text = None

        current_messages = cached["messages"] if cached else []

        def _prepend_initial_email(msgs):
            if not initial_msg_text: return msgs
            clean_init = initial_msg_text.strip()
            entry = {
                "type": "SENT",
                "email_body": clean_init,
                "time": initial_msg_at.isoformat() if initial_msg_at else None,
                "sender_name": "You",
            }
            if not msgs: return [entry]
            if any((m.get("email_body") or "").strip() == clean_init for m in msgs):
                return msgs
            return [entry] + msgs

        final_msgs = _prepend_initial_email(current_messages)
        try:
            final_msgs.sort(key=lambda x: x.get("time", ""))
        except:
            pass

        return {
            "messages": final_msgs,
            "syncing": not cached or is_stale or (cached and cached.get("refreshing")),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching email chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
    return new_code

code = re.sub(r'async def get_chat_history\(.*?\):.*?raise HTTPException\(status_code=500, detail=str\(e\)\)', patch_email, code, flags=re.DOTALL)

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)
print("patched")

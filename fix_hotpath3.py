with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

import re
code = re.sub(r'    """Fetch structured chat history \(Default to Email\)"""\n    import time', r'''    """Fetch structured chat history (Default to Email)"""
    import time
    
    # ── HOT PATH CACHE CHECK ──
    with _email_chat_lock:
        cached = _email_chat_cache.get(candidate_id)
        if cached:
            cache_age = time.monotonic() - cached.get("ts", 0)
            is_stale = cache_age > _EMAIL_CACHE_STALE_THRESHOLD
            already_refreshing = cached.get("refreshing", False)
            if (not is_stale or already_refreshing) and not force:
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
                return {"messages": final_msgs, "syncing": already_refreshing}''', code, count=1)

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)
print("Hotpath injected for email.")

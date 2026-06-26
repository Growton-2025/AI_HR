with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

# I will insert the hotpath directly after `start_t = time.time()`
import re
code = re.sub(r'    start_t = time\.time\(\)', r'''    start_t = time.time()
    
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
                return {"messages": final_msgs, "syncing": already_refreshing}''', code, count=1)

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)
print("Hotpath injected.")

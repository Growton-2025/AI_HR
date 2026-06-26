import re

with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

# For LinkedIn
code = code.replace("""        if final_messages:
            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                    if not conn:
                        raise RuntimeError("DB Connection failed")
                    with conn.cursor() as cur:
                        import json
                        cur.execute(
                            "UPDATE candidate_outreach SET li_chat_history_cache = %s, li_chat_history_updated_at = NOW() WHERE candidate_id = %s",
                            (json.dumps(final_messages), candidate_id)
                        )
                        conn.commit()
            except Exception as e:
                print(f"DEBUG: Failed to update DB cache: {e}")""", "")

code = code.replace("""        _li_chat_cache[candidate_id] = {
            "messages": final_messages,
            "ts": time.monotonic(),
            "refreshing": False
        }

        # ── PERSISTENT DB CACHE ──────────────────────────────────────────────
        # Save the fetched history to DB so it lives across restarts""", """        _li_chat_cache[candidate_id] = {
            "messages": final_messages,
            "ts": time.monotonic(),
            "refreshing": False
        }

    # ── PERSISTENT DB CACHE ──────────────────────────────────────────────
    # Save the fetched history to DB OUTSIDE THE LOCK so hot path isn't blocked!
    if final_messages:
        try:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if not conn:
                    raise RuntimeError("DB Connection failed")
                with conn.cursor() as cur:
                    import json
                    cur.execute(
                        "UPDATE candidate_outreach SET li_chat_history_cache = %s, li_chat_history_updated_at = NOW() WHERE candidate_id = %s",
                        (json.dumps(final_messages), candidate_id)
                    )
                    conn.commit()
        except Exception as e:
            print(f"DEBUG: Failed to update DB cache: {e}")""")

# For Email
code = code.replace("""        if final_messages:
            try:
                with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                    if not conn:
                        raise RuntimeError("DB Connection failed")
                    with conn.cursor() as cur:
                        import json
                        cur.execute(
                            "UPDATE candidate_outreach SET email_chat_history_cache = %s, email_chat_history_updated_at = NOW() WHERE candidate_id = %s",
                            (json.dumps(final_messages), candidate_id)
                        )
                        conn.commit()
            except Exception as e:
                print(f"DEBUG: Failed to update DB cache: {e}")""", "")

code = code.replace("""        _email_chat_cache[candidate_id] = {
            "messages": final_messages,
            "ts": time.monotonic(),
            "refreshing": False
        }

        # ── PERSISTENT DB CACHE ──────────────────────────────────────────────
        # Save the fetched history to DB so it lives across restarts""", """        _email_chat_cache[candidate_id] = {
            "messages": final_messages,
            "ts": time.monotonic(),
            "refreshing": False
        }

    # ── PERSISTENT DB CACHE ──────────────────────────────────────────────
    # Save the fetched history to DB OUTSIDE THE LOCK!
    if final_messages:
        try:
            with get_db_connection_context(validate=False, register_pgvector=False) as conn:
                if not conn:
                    raise RuntimeError("DB Connection failed")
                with conn.cursor() as cur:
                    import json
                    cur.execute(
                        "UPDATE candidate_outreach SET email_chat_history_cache = %s, email_chat_history_updated_at = NOW() WHERE candidate_id = %s",
                        (json.dumps(final_messages), candidate_id)
                    )
                    conn.commit()
        except Exception as e:
            print(f"DEBUG: Failed to update DB cache: {e}")""")

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)

print("Patched sync methods.")

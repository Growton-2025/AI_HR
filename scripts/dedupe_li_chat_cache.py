"""Reconcile stored LinkedIn threads with HeyReach and drop redundant entries.

candidate_outreach.li_chat_history_cache accumulated duplicate entries: the
reply webhook was registered four times (every reply echoed 4x, plus retries)
and the sync paths preferred whichever thread was LONGER, so nothing ever
converged back to HeyReach's real thread. Adjacent duplicates were collapsed
earlier; what remains are local echoes (`local_echo`, ids `local-*`) that sit
apart from their original — typically an inbound echo landing after our next
outbound message.

Uses the SAME merge the runtime now uses (_merge_li_thread): HeyReach's fetch
is truth, echoes it already ingested are dropped and replaced by the real
entries, real messages are never removed. A candidate genuinely repeating
"hi" keeps every "hi".

    python scripts/dedupe_li_chat_cache.py                    # dry run
    python scripts/dedupe_li_chat_cache.py --only 13918,13846
    python scripts/dedupe_li_chat_cache.py --apply
    python scripts/dedupe_li_chat_cache.py --restore <backup.json>
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.connection import get_db_connection_context
from backend.api.routes.outreach import (
    _clean_email_body,
    _dedupe_consecutive_messages,
    _is_echo,
    _merge_li_thread,
    _msg_key,
)
from backend.integrations.heyreach import HeyReachBot

BACKUP_DIR = os.getenv(
    "LI_CACHE_BACKUP_DIR",
    os.path.join(tempfile.gettempdir(), "ai_hr_li_cache_backups"),
)


def _needs_work(thread):
    if any(_is_echo(m) for m in thread):
        return True
    counts = Counter(_msg_key(m) for m in thread)
    return any(n > 1 for k, n in counts.items() if k[1])


def _fetch_real_thread(bot, linkedin, conversation_id, account_id, sleep_s):
    try:
        res = bot.get_li_chat_history(
            linkedin or None,
            conversation_id=conversation_id,
            account_id=int(account_id) if account_id else None,
        )
    except Exception as e:
        print(f"    fetch failed ({e}); retrying once in 5s")
        time.sleep(5)
        try:
            res = bot.get_li_chat_history(
                linkedin or None,
                conversation_id=conversation_id,
                account_id=int(account_id) if account_id else None,
            )
        except Exception as e2:
            print(f"    fetch failed again ({e2}); using stored-only rule")
            return None
    finally:
        time.sleep(sleep_s)
    msgs = (res or {}).get("messages") or []
    for m in msgs:
        if str(m.get("direction") or "").lower() != "inbound" and m.get("email_body"):
            m["email_body"] = _clean_email_body(m["email_body"])
    return _dedupe_consecutive_messages(msgs)


def _fmt(m):
    return f"{m.get('direction'):8} | {str(m.get('time'))[:24]:24} | echo={str(_is_echo(m)):5} | {(m.get('email_body') or '')[:60]!r}"


def _sig(m):
    # What a recruiter can see: direction, text, second-precision time, echo-ness.
    # Provider ids differ between the listing and chatroom copies of the same
    # message, so comparing whole dicts would report every row as changed.
    return (_msg_key(m), str(m.get("time") or "")[:19], _is_echo(m))


def _diff(old, new):
    old_c = Counter(_sig(m) for m in old)
    new_c = Counter(_sig(m) for m in new)
    removed_sigs = old_c - new_c
    added_sigs = new_c - old_c
    removed, added = [], []
    for m in old:
        if removed_sigs[_sig(m)] > 0:
            removed_sigs[_sig(m)] -= 1
            removed.append(m)
    for m in new:
        if added_sigs[_sig(m)] > 0:
            added_sigs[_sig(m)] -= 1
            added.append(m)
    return removed, added


def restore(path):
    with open(path) as f:
        backup = json.load(f)
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            for cid, blob in backup.items():
                cur.execute(
                    "UPDATE candidate_outreach SET li_chat_history_cache = %s::jsonb, li_chat_history_updated_at = NOW() WHERE candidate_id = %s",
                    (json.dumps(blob), int(cid)),
                )
                print(f"restored candidate {cid} ({len(blob)} msgs)")
        conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-reconcile", action="store_true", help="never call HeyReach; stored-only rule")
    ap.add_argument("--only", default="", help="comma-separated candidate ids")
    ap.add_argument("--skip", default="", help="comma-separated candidate ids")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--restore", default="")
    args = ap.parse_args()

    if args.restore:
        restore(args.restore)
        return

    only = {int(x) for x in args.only.split(",") if x.strip()}
    skip = {int(x) for x in args.skip.split(",") if x.strip()}

    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT co.candidate_id, c.name, co.li_chat_history_cache,
                       c.linkedin, co.li_conversation_id, co.li_account_id,
                       md5(co.li_chat_history_cache::text)
                FROM candidate_outreach co
                JOIN candidates c ON c.id = co.candidate_id
                WHERE co.li_chat_history_cache IS NOT NULL
                  AND jsonb_array_length(co.li_chat_history_cache) > 0
                ORDER BY co.candidate_id
                """
            )
            rows = cur.fetchall()
    print(f"Scanned {len(rows)} threads.")

    bot = HeyReachBot()
    backups = {}
    changes = []
    skipped_invariant = []
    for cid, name, thread, linkedin, conv_id, acc_id, md5 in rows:
        if only and cid not in only:
            continue
        if cid in skip:
            continue
        thread = thread or []
        if not _needs_work(thread):
            continue

        fetched = None
        if not args.no_reconcile:
            fetched = _fetch_real_thread(bot, linkedin, conv_id, acc_id, args.sleep)
        new = _merge_li_thread(thread, fetched or [])

        before = Counter(_msg_key(m) for m in thread if not _is_echo(m))
        after = Counter(_msg_key(m) for m in new)
        lost = {k: n for k, n in before.items() if after[k] < n}
        if lost:
            skipped_invariant.append(cid)
            print(f"\n!!! {cid} {name}: would lose real messages {list(lost)[:3]} — SKIPPED")
            continue

        removed, added = _diff(thread, new)
        if not removed and not added:
            continue
        changes.append((cid, md5, new))
        backups[str(cid)] = thread
        print(f"\n=== {cid} {name}: {len(thread)} -> {len(new)} (fetched {len(fetched) if fetched is not None else 'n/a'})")
        for m in removed:
            print("  - " + _fmt(m))
        for m in added:
            print("  + " + _fmt(m))

    print(f"\nCandidates to change: {len(changes)}; skipped by invariant: {skipped_invariant}")
    if not changes:
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(BACKUP_DIR, f"li_chat_cache_backup_{stamp}.json")
    with open(backup_path, "w") as f:
        json.dump(backups, f, indent=2, default=str)
    print(f"Backup written to {backup_path}")

    if not args.apply:
        print("DRY RUN — re-run with --apply to write.")
        return

    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            for cid, md5, new in changes:
                cur.execute(
                    """
                    UPDATE candidate_outreach
                    SET li_chat_history_cache = %s::jsonb, li_chat_history_updated_at = NOW()
                    WHERE candidate_id = %s AND md5(li_chat_history_cache::text) = %s
                    """,
                    (json.dumps(new), cid, md5),
                )
                if cur.rowcount:
                    print(f"updated {cid}")
                else:
                    print(f"!!! {cid} changed underneath — re-run")
                conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()

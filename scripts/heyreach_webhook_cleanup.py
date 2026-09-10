"""Collapse duplicate HeyReach reply-webhook registrations.

HeyReach fans every reply out to EVERY active webhook registered for the
event, and retries failed deliveries for 24h. Four identical
EVERY_MESSAGE_REPLY_RECEIVED hooks for our URL meant each candidate reply hit
the backend four times. This lists what's registered, and with --apply keeps
the oldest active hook for our URL+event, deletes the other duplicates, and
optionally deactivates extra hook ids (e.g. an older MESSAGE_REPLY_RECEIVED
hook on the same URL). Hooks pointing anywhere else are never touched.

    python scripts/heyreach_webhook_cleanup.py                 # list only
    python scripts/heyreach_webhook_cleanup.py --apply --deactivate 24358
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.db.connection  # noqa: F401  (loads .env)
from backend.integrations.heyreach import HeyReachBot

DEFAULT_URL = (
    "https://growton-backend-v2-e3a3hxdmagfggcg9.centralindia-01.azurewebsites.net"
    "/api/outreach/heyreach/webhook"
)


def _print(hooks):
    for h in sorted(hooks, key=lambda h: int(h.get("id") or 0)):
        print(
            f"  id={h.get('id')} active={h.get('isActive')} event={h.get('eventType')} "
            f"name={h.get('webhookName')!r} campaigns={h.get('campaignIds')} url={h.get('webhookUrl')}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("HEYREACH_WEBHOOK_URL") or DEFAULT_URL)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--deactivate", type=int, nargs="*", default=[], help="extra hook ids to set inactive")
    args = ap.parse_args()

    bot = HeyReachBot()
    hooks = bot.list_webhooks()
    print(f"{len(hooks)} webhooks registered:")
    _print(hooks)

    ours = sorted(
        (
            h for h in hooks
            if h.get("webhookUrl") == args.url
            and h.get("eventType") == HeyReachBot.REPLY_WEBHOOK_EVENT
            and h.get("isActive", True)
        ),
        key=lambda h: int(h.get("id") or 0),
    )
    by_id = {int(h.get("id")): h for h in hooks if h.get("id") is not None}
    print(f"\nActive {HeyReachBot.REPLY_WEBHOOK_EVENT} hooks for our URL: {[h['id'] for h in ours]}")
    if not ours:
        print("Nothing to keep — the backend registers one on its next startup.")
    extras = ours[1:]
    print(f"Would delete: {[h['id'] for h in extras]}  (keeping {ours[0]['id'] if ours else None})")

    deactivate = []
    for hid in args.deactivate:
        h = by_id.get(hid)
        if not h:
            print(f"  skip {hid}: not registered")
        elif h.get("webhookUrl") != args.url:
            print(f"  refuse {hid}: points at {h.get('webhookUrl')}, not our URL")
        elif not h.get("isActive", True):
            print(f"  skip {hid}: already inactive")
        else:
            deactivate.append(h)
    print(f"Would deactivate: {[h['id'] for h in deactivate]}")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to change HeyReach.")
        return

    for h in extras:
        bot.delete_webhook(h["id"])
        print(f"deleted {h['id']}")
    for h in deactivate:
        bot.set_webhook_active(h["id"], False)
        print(f"deactivated {h['id']}")

    hooks = bot.list_webhooks()
    remaining = [
        h for h in hooks
        if h.get("webhookUrl") == args.url
        and h.get("eventType") == HeyReachBot.REPLY_WEBHOOK_EVENT
        and h.get("isActive", True)
    ]
    print(f"\nAfter: {len(remaining)} active {HeyReachBot.REPLY_WEBHOOK_EVENT} hook(s) for our URL")
    _print(hooks)
    if len(remaining) != 1:
        print("WARNING: expected exactly one")
        sys.exit(1)


if __name__ == "__main__":
    main()

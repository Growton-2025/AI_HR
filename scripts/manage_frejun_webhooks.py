#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from backend.integrations.frejun import FreJunManager

load_dotenv()

DEFAULT_EVENTS = ["call.status", "call.recording", "call.summary"]


def _resolve_callback_url(value: Optional[str]) -> str:
    callback_url = (value or os.getenv("FREJUN_WEBHOOK_CALLBACK_URL") or "").strip()
    if not callback_url:
        raise ValueError("A callback URL is required. Pass --callback-url or set FREJUN_WEBHOOK_CALLBACK_URL.")
    if not callback_url.startswith("https://"):
        raise ValueError("FreJun webhooks require a public HTTPS callback URL.")
    return callback_url


def main() -> int:
    parser = argparse.ArgumentParser(description="List or register FreJun webhooks for completed-call events.")
    parser.add_argument(
        "action",
        choices=["list", "ensure"],
        help="List configured FreJun webhooks or create the missing ones for the callback URL.",
    )
    parser.add_argument(
        "--callback-url",
        help="Public HTTPS callback URL for FreJun webhook delivery. Defaults to FREJUN_WEBHOOK_CALLBACK_URL.",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=DEFAULT_EVENTS,
        help="Webhook events to ensure. Defaults to call.status call.recording call.summary.",
    )
    args = parser.parse_args()

    manager = FreJunManager()
    if args.action == "list":
        result = manager.list_webhooks()
    else:
        callback_url = _resolve_callback_url(args.callback_url)
        result = manager.ensure_webhooks(callback_url=callback_url, events=args.events or DEFAULT_EVENTS)

    if result.get("success"):
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(json.dumps(result, indent=2, default=str), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

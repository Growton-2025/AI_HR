"""One chronological timeline for a *person*, built from every candidate row
that person_identity resolves to: outbound calls (live and archived), inbound
callbacks, LinkedIn and email messages, status changes, notes and the links
themselves. Deduped, newest first.

Visibility: everyone sees every event (that is the point — "when did *we*
last contact this person"). Another recruiter's call still carries its
recording and transcript, flagged ``you=false`` so the UI can collapse them
by default; admins get them expanded.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from dateutil import parser as dateparser

from backend.services.call_artifacts import extract_transcript_text
from backend.services.person_identity import resolve_person, table_exists

logger = logging.getLogger(__name__)


def _ts(value) -> Optional[datetime]:
    """Any timestamp we store (naive UTC datetimes, ISO strings from provider
    caches) as an aware UTC datetime; None when unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = dateparser.parse(str(value))
    except Exception:
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value) -> Optional[str]:
    dt = _ts(value)
    return dt.isoformat() if dt else None


def _is_inbound(msg: dict) -> bool:
    if msg.get("direction") == "inbound":
        return True
    if msg.get("direction") == "outbound":
        return False
    return str(msg.get("type") or "").upper() in {"REPLY", "REPLIED", "INBOX", "INCOMING", "LEAD"}


def _message_key(platform: str, msg: dict) -> str:
    mid = msg.get("id") or msg.get("message_id") or msg.get("stats_id")
    if mid:
        return f"{platform}:{mid}"
    body = str(msg.get("email_body") or msg.get("text") or "")[:80]
    return f"{platform}:{msg.get('time')}:{_is_inbound(msg)}:{body}"


def build_timeline(cur, candidate_id: int, *, viewer_email: str, viewer_is_admin: bool,
                   scope: str = "person") -> dict:
    viewer = (viewer_email or "").strip().lower()
    resolution = resolve_person(cur, candidate_id) if scope == "person" else None
    if resolution:
        ids = resolution.candidate_ids
        rows = resolution.rows
        person_key = resolution.person_key
    else:
        ids, rows, person_key = [candidate_id], [], None

    owner_by_row: Dict[int, Optional[str]] = {r["id"]: r.get("owner_email") for r in rows}
    items: List[dict] = []

    # ── outbound calls ────────────────────────────────────────────────────
    cur.execute(
        """
        SELECT c.id, c.candidate_id, COALESCE(c.completed_at, c.updated_at, c.created_at),
               c.outcome, c.duration, c.recording_url, c.summary, c.transcript, c.notes,
               COALESCE(c.likely_voicemail, FALSE), LOWER(COALESCE(cl.created_by, '')),
               c.plivo_hangup_cause, cand.name
        FROM calls c
        JOIN call_lists cl ON cl.id = c.list_id
        JOIN candidates cand ON cand.id = c.candidate_id
        WHERE c.candidate_id = ANY(%s) AND c.status = 'completed'
          AND COALESCE(c.outcome, '') NOT LIKE 'Closed - %%'
        """,
        (ids,),
    )
    for r in cur.fetchall():
        by = r[10]
        items.append({
            "type": "call", "id": f"call:{r[0]}", "call_id": r[0], "candidate_id": r[1],
            "occurred_at": _iso(r[2]), "outcome": r[3], "duration_seconds": r[4] or 0,
            "recording_url": r[5], "summary": (r[6] or "").strip() or None,
            "transcript": extract_transcript_text(r[7], candidate_name=r[12]) if r[7] else None,
            "notes": (r[8] or "").strip() or None, "likely_voicemail": bool(r[9]),
            "hangup_cause": r[11], "by": by or None, "you": bool(by) and by == viewer,
            "archived": False,
        })

    # ── archived (deleted) call rows keep the fact that we called ─────────
    deleted_rows = []
    if table_exists(cur, "deleted_calls"):
        cur.execute(
            """
            SELECT call_row, deleted_at FROM deleted_calls
            WHERE (call_row->>'candidate_id')::int = ANY(%s)
              AND call_row->>'status' = 'completed'
              AND COALESCE(call_row->>'outcome', '') NOT LIKE 'Closed - %%'
            """,
            (ids,),
        )
        deleted_rows = cur.fetchall()
    for call_row, deleted_at in deleted_rows:
        if True:
            row = call_row or {}
            items.append({
                "type": "call", "id": f"call:{row.get('id')}", "call_id": row.get("id"),
                "candidate_id": row.get("candidate_id"),
                "occurred_at": _iso(row.get("completed_at") or row.get("updated_at")),
                "outcome": row.get("outcome"), "duration_seconds": row.get("duration") or 0,
                "recording_url": row.get("recording_url"), "summary": row.get("summary"),
                "transcript": None, "notes": row.get("notes"), "likely_voicemail": False,
                "hangup_cause": row.get("plivo_hangup_cause"), "by": None, "you": False,
                "archived": True, "deleted_at": _iso(deleted_at),
            })

    # ── inbound callbacks ────────────────────────────────────────────────
    cur.execute(
        """
        SELECT i.id, i.candidate_id, i.received_at, i.answered_at, i.duration, i.hangup_cause,
               i.call_status, i.status, i.note, i.recording_url, i.transcript, i.from_number,
               LOWER(COALESCE(u.email, ''))
        FROM inbound_calls i LEFT JOIN users u ON u.id = i.answered_by_user_id
        WHERE i.candidate_id = ANY(%s)
        """,
        (ids,),
    )
    for r in cur.fetchall():
        by = r[12] or None
        items.append({
            "type": "inbound_call", "id": f"inbound:{r[0]}", "candidate_id": r[1],
            "occurred_at": _iso(r[2]), "answered_at": _iso(r[3]), "duration_seconds": r[4] or 0,
            "hangup_cause": r[5], "call_status": r[6], "status": r[7],
            "notes": (r[8] or "").strip() or None, "recording_url": r[9],
            "transcript": r[10], "from_number": r[11], "by": by, "you": bool(by) and by == viewer,
        })

    # ── LinkedIn and email threads (deduped across rows) ─────────────────
    role_names: Dict[int, str] = {}
    cur.execute(
        """
        SELECT o.candidate_id, o.recruitment_role_id, o.campaign_name,
               o.li_chat_history_cache, o.email_chat_history_cache
        FROM candidate_outreach o WHERE o.candidate_id = ANY(%s)
        """,
        (ids,),
    )
    outreach = cur.fetchall()
    role_ids = sorted({r[1] for r in outreach if r[1]})
    if role_ids:
        cur.execute("SELECT id, name FROM recruitment_roles WHERE id = ANY(%s)", (role_ids,))
        role_names = {int(i): n for i, n in cur.fetchall()}
    seen_messages = set()
    for cand_id, role_id, campaign_name, li_cache, email_cache in outreach:
        for platform, cache in (("linkedin", li_cache), ("email", email_cache)):
            if not isinstance(cache, list):
                continue
            for msg in cache:
                if not isinstance(msg, dict):
                    continue
                key = _message_key(platform, msg)
                if key in seen_messages:
                    continue
                seen_messages.add(key)
                inbound = _is_inbound(msg)
                items.append({
                    "type": f"{platform}_message", "id": key, "candidate_id": cand_id,
                    "occurred_at": _iso(msg.get("time") or msg.get("created_at")),
                    "direction": "inbound" if inbound else "outbound",
                    "subject": msg.get("subject"), "body": msg.get("email_body") or msg.get("text") or "",
                    "sender_name": msg.get("sender_name"),
                    "role": role_names.get(int(role_id)) if role_id else None,
                    "campaign": campaign_name, "you": False,
                })

    # ── threads fetched from the providers for this person (PR 3) ────────
    pending: List[str] = []
    if person_key and table_exists(cur, "person_provider_threads"):
        from backend.services.person_history_backfill import pending_providers, provider_threads
        for provider, thread_ref, messages in provider_threads(cur, person_key):
            platform = "linkedin" if provider == "heyreach" else "email"
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                key = _message_key(platform, msg)
                if key in seen_messages:
                    continue
                seen_messages.add(key)
                inbound = _is_inbound(msg)
                items.append({
                    "type": f"{platform}_message", "id": key, "candidate_id": candidate_id,
                    "occurred_at": _iso(msg.get("time") or msg.get("created_at")),
                    "direction": "inbound" if inbound else "outbound",
                    "subject": msg.get("subject"), "body": msg.get("email_body") or msg.get("text") or "",
                    "sender_name": msg.get("sender_name"), "role": None,
                    "campaign": thread_ref, "you": False, "source": provider,
                })
        if table_exists(cur, "person_history_jobs"):
            pending = pending_providers(cur, person_key)

    # ── status changes ───────────────────────────────────────────────────
    status_rows = []
    if table_exists(cur, "candidate_status_history"):
        cur.execute(
            """
            SELECT id, candidate_id, old_status, new_status, changed_by, source, changed_at
            FROM candidate_status_history WHERE candidate_id = ANY(%s)
            """,
            (ids,),
        )
        status_rows = cur.fetchall()
    for r in status_rows:
        if True:
            by = (r[4] or "").lower() or None
            items.append({
                "type": "status_change", "id": f"status:{r[0]}", "candidate_id": r[1],
                "occurred_at": _iso(r[6]), "old_status": r[2], "new_status": r[3],
                "source": r[5], "by": by, "you": bool(by) and by == viewer,
            })

    # ── notes (current value per row) and the links themselves ───────────
    cur.execute(
        "SELECT id, notes, updated_at FROM candidates WHERE id = ANY(%s) AND COALESCE(TRIM(notes), '') <> ''",
        (ids,),
    )
    for cid, notes, updated_at in cur.fetchall():
        items.append({
            "type": "note", "id": f"note:{cid}", "candidate_id": cid,
            "occurred_at": _iso(updated_at), "body": notes,
            "by": owner_by_row.get(cid), "you": (owner_by_row.get(cid) or "") == viewer,
        })
    link_rows = []
    if len(ids) > 1 and table_exists(cur, "candidate_person_links"):
        cur.execute(
            "SELECT candidate_id, matched_on, linked_at, linked_by FROM candidate_person_links WHERE candidate_id = ANY(%s)",
            (ids,),
        )
        link_rows = cur.fetchall()
    if link_rows:
        for cid, matched_on, linked_at, linked_by in link_rows:
            items.append({
                "type": "link", "id": f"link:{cid}", "candidate_id": cid,
                "occurred_at": _iso(linked_at), "matched_on": matched_on, "by": linked_by,
                "you": False,
            })

    for item in items:
        item["owner_email"] = owner_by_row.get(item.get("candidate_id"))
        if not viewer_is_admin and not item.get("you") and item["type"] in ("call", "inbound_call"):
            item["collapsed"] = True   # visible, but the UI hides transcript/recording behind a click

    items.sort(key=lambda i: (_ts(i.get("occurred_at")) or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    counts: Dict[str, int] = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    return {
        "candidate_id": candidate_id, "scope": scope, "person_key": person_key,
        "candidate_ids": ids,
        "rows": [
            {"id": r["id"], "name": r["name"], "owner_email": r["owner_email"],
             "pool_source": r["pool_source"], "is_archived": r["is_archived"]}
            for r in rows
        ],
        "counts": counts, "items": items, "pending_backfill": pending,
    }

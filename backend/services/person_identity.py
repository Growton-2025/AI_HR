"""Who is the same person across candidate rows.

Every history store (calls, inbound calls, outreach threads, status history,
notes) is keyed on ``candidates.id``, and the same person legitimately exists
as several rows: one master-library row plus one copy per recruiter pool. A
new row therefore starts with no history even when a sibling row has months of
it. This module resolves a row to its *person* and records the link, so the
timeline (person_timeline.py) can answer for the person rather than the row.

Rows are never merged: ownership, cadences and role links hang off the row id.

Identity precedence: LinkedIn (canonical ``/in/slug``, see linkedin_normalize)
> email > phone. Email and phone are weak keys — an agency mailbox or a family
phone must never glue two people together — so they only link rows that do
not disagree on a LinkedIn key.

Round trips matter: the hosted database answers in ~0.3–0.6s per statement,
so resolution is two statements and the summary is one. Optional tables are
checked once per process rather than probed with savepoints on every call.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from backend.services.linkedin_normalize import (
    canonical_email,
    normalize_linkedin,
    person_key,
)

logger = logging.getLogger(__name__)

MATCHED_LINKEDIN = "linkedin"
MATCHED_EMAIL = "email"
MATCHED_PHONE = "phone"
MATCHED_NONE = "none"

LINKS_TABLE = "candidate_person_links"
_schema_ready = False
_table_cache: Dict[str, bool] = {}


def table_exists(cur, name: str) -> bool:
    """Per-process memo: a table that exists never stops existing."""
    if _table_cache.get(name):
        return True
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{name}",))
    exists = bool(cur.fetchone()[0])
    if exists:
        _table_cache[name] = True
    return exists


def ensure_person_links_schema(cur) -> None:
    global _schema_ready
    if _schema_ready:
        return
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_person_links (
            candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
            person_key   TEXT NOT NULL,
            matched_on   VARCHAR(16) NOT NULL,
            linked_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            linked_by    VARCHAR(255)
        );
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_person_links_key ON candidate_person_links (person_key);"
    )
    _schema_ready = True
    _table_cache[LINKS_TABLE] = True


def phone_tail(value: Optional[str]) -> Optional[str]:
    """Last ten digits — the same rule inbound-call matching uses."""
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[-10:] if len(digits) >= 10 else None


def person_key_for(normalized_linkedin, linkedin, email, phone) -> tuple:
    """(key, matched_on) for a row's identity fields. Keys are namespaced so an
    email-only person can never collide with a LinkedIn slug."""
    key = person_key(normalized_linkedin) or normalize_linkedin(linkedin)
    if key:
        return key, MATCHED_LINKEDIN
    mail = canonical_email(email)
    if mail:
        return f"email:{mail}", MATCHED_EMAIL
    tail = phone_tail(phone)
    if tail:
        return f"phone:{tail}", MATCHED_PHONE
    return None, MATCHED_NONE


@dataclass
class PersonResolution:
    candidate_id: int
    person_key: Optional[str]
    matched_on: str
    candidate_ids: List[int] = field(default_factory=list)   # includes candidate_id
    rows: List[dict] = field(default_factory=list)           # id, owner_user_id, owner_email, name, pool_source, is_archived


_ROW_SQL = """
    SELECT c.id, c.owner_user_id, u.email, c.name, c.pool_source, COALESCE(c.is_archived, FALSE),
           c.normalized_linkedin, c.linkedin, c.email,
           COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''))
    FROM candidates c
    LEFT JOIN users u ON u.id = c.owner_user_id
"""
_PHONE_TAIL_SQL = "RIGHT(REGEXP_REPLACE(COALESCE(NULLIF(TRIM(c.mobile_phone), ''), NULLIF(TRIM(c.phone), ''), ''), '[^0-9]', '', 'g'), 10)"


def _row_dict(r) -> dict:
    return {
        "id": int(r[0]), "owner_user_id": r[1], "owner_email": r[2], "name": r[3],
        "pool_source": r[4], "is_archived": bool(r[5]),
        "normalized_linkedin": r[6], "linkedin": r[7], "email": r[8], "phone": r[9],
    }


def _matching_rows(cur, *, key: Optional[str], mail: Optional[str], tail: Optional[str]) -> List[dict]:
    """Every row that could be this person, in ONE statement: by LinkedIn key
    (including rows carrying a "<key>_legacy_<id>" dedupe suffix and rows
    linked manually), by email, by phone tail."""
    clauses, params = [], []
    if key:
        clauses.append("c.normalized_linkedin = %s OR c.normalized_linkedin LIKE %s")
        params += [key, key + r"\_legacy\_%"]
        if table_exists(cur, LINKS_TABLE):
            clauses.append("c.id IN (SELECT candidate_id FROM candidate_person_links WHERE person_key = %s)")
            params.append(key)
    if mail:
        clauses.append("LOWER(TRIM(c.email)) = %s")
        params.append(mail)
    if tail:
        clauses.append(f"{_PHONE_TAIL_SQL} = %s")
        params.append(tail)
    if not clauses:
        return []
    cur.execute(_ROW_SQL + " WHERE " + " OR ".join(f"({c})" for c in clauses), params)
    return [_row_dict(r) for r in cur.fetchall()]


def _accept(row: dict, *, key: Optional[str], mail: Optional[str], tail: Optional[str],
            peers_keys: set) -> bool:
    """Does a matched row belong to the person? LinkedIn agreement wins; email/
    phone only count when the row does not carry a *different* LinkedIn key."""
    row_key = person_key(row["normalized_linkedin"]) or normalize_linkedin(row["linkedin"])
    if key and row_key == key:
        return True
    if row_key and key and row_key != key:
        return False
    if row_key and not key:
        # We have no LinkedIn; the match came via email/phone. Accept only if
        # the matched rows agree on a single LinkedIn key between them.
        return len(peers_keys) <= 1
    same_mail = mail and canonical_email(row["email"]) == mail
    same_phone = tail and phone_tail(row["phone"]) == tail
    return bool(same_mail or same_phone)


def _resolve_rows(cur, me: Optional[dict], *, key: Optional[str], mail: Optional[str], tail: Optional[str]) -> List[dict]:
    matched = _matching_rows(cur, key=key, mail=mail, tail=tail)
    peers_keys = {person_key(r["normalized_linkedin"]) or normalize_linkedin(r["linkedin"]) for r in matched}
    peers_keys.discard(None)
    found: Dict[int, dict] = {}
    if me:
        found[me["id"]] = me
    for row in matched:
        if row["id"] in found:
            continue
        if _accept(row, key=key, mail=mail, tail=tail, peers_keys=peers_keys):
            found[row["id"]] = row
    return [found[i] for i in sorted(found)]


def resolve_person(cur, candidate_id: int) -> Optional[PersonResolution]:
    cur.execute(_ROW_SQL + " WHERE c.id = %s", (candidate_id,))
    r = cur.fetchone()
    if not r:
        return None
    me = _row_dict(r)
    key, matched_on = person_key_for(me["normalized_linkedin"], me["linkedin"], me["email"], me["phone"])
    mail = canonical_email(me["email"])
    tail = phone_tail(me["phone"])
    rows = _resolve_rows(cur, me, key=key, mail=mail, tail=tail)
    return PersonResolution(
        candidate_id=candidate_id, person_key=key, matched_on=matched_on,
        candidate_ids=[x["id"] for x in rows], rows=rows,
    )


def link_candidates_bulk(cur, candidate_ids: Sequence[int], *, by: str = "system") -> int:
    """Record the person key for many rows in two statements. Used by every
    write path (single add, import, admin assign) so a row is linked the moment
    it exists; never raises — an unlinked row is only a missing shortcut."""
    ids = sorted({int(i) for i in candidate_ids if i})
    if not ids:
        return 0
    try:
        cur.execute("SAVEPOINT person_link")
        ensure_person_links_schema(cur)
        cur.execute(
            """
            SELECT id, normalized_linkedin, linkedin, email,
                   COALESCE(NULLIF(TRIM(mobile_phone), ''), NULLIF(TRIM(phone), ''))
            FROM candidates WHERE id = ANY(%s)
            """,
            (ids,),
        )
        values = []
        for cid, nli, li, mail, phone in cur.fetchall():
            key, matched_on = person_key_for(nli, li, mail, phone)
            if key:
                values.append((int(cid), key, matched_on, by))
        if values:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO candidate_person_links (candidate_id, person_key, matched_on, linked_by)
                VALUES %s
                ON CONFLICT (candidate_id) DO UPDATE
                SET person_key = EXCLUDED.person_key, matched_on = EXCLUDED.matched_on,
                    linked_at = CURRENT_TIMESTAMP, linked_by = EXCLUDED.linked_by
                """,
                values,
            )
        cur.execute("RELEASE SAVEPOINT person_link")
        return len(values)
    except Exception as exc:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT person_link")
        except Exception:
            pass
        logger.warning("Could not link candidates %s: %s", ids[:5], exc)
        return 0


# ── the summary the UI shows the moment a row is saved (optimistic) ─────────

_EMPTY_SUMMARY = {"prior_candidate_ids": [], "calls": 0, "inbound_calls": 0,
                  "linkedin_replies": 0, "emails": 0, "status_changes": 0, "last_interaction_at": None}


def person_summary(cur, candidate_ids: Sequence[int], *, exclude: Optional[int] = None) -> dict:
    ids = [int(i) for i in candidate_ids if i and i != exclude]
    if not ids:
        return dict(_EMPTY_SUMMARY)
    has_status = table_exists(cur, "candidate_status_history")
    status_sql = (
        "(SELECT COUNT(*) FROM candidate_status_history WHERE candidate_id = ANY(%s)), "
        "(SELECT MAX(changed_at) FROM candidate_status_history WHERE candidate_id = ANY(%s))"
        if has_status else "0, NULL::timestamp"
    )
    cur.execute(
        f"""
        SELECT
          (SELECT COUNT(*) FROM calls WHERE candidate_id = ANY(%s) AND status = 'completed'
             AND COALESCE(outcome, '') NOT LIKE 'Closed - %%'),
          (SELECT MAX(completed_at) FROM calls WHERE candidate_id = ANY(%s) AND status = 'completed'),
          (SELECT COUNT(*) FROM inbound_calls WHERE candidate_id = ANY(%s)),
          (SELECT MAX(received_at) FROM inbound_calls WHERE candidate_id = ANY(%s)),
          (SELECT COALESCE(SUM((SELECT COUNT(*) FROM jsonb_array_elements(COALESCE(li_chat_history_cache, '[]'::jsonb)) m
                                WHERE m->>'direction' = 'inbound' OR m->>'type' = 'REPLY')), 0)
             FROM candidate_outreach WHERE candidate_id = ANY(%s)),
          (SELECT COALESCE(SUM((SELECT COUNT(*) FROM jsonb_array_elements(
                                CASE WHEN jsonb_typeof(email_chat_history_cache) = 'array' THEN email_chat_history_cache ELSE '[]'::jsonb END) m
                                WHERE m->>'type' = 'REPLY' OR m->>'direction' = 'inbound')), 0)
             FROM candidate_outreach WHERE candidate_id = ANY(%s)),
          (SELECT MAX(GREATEST(li_response_received_at, response_received_at, li_last_action_at, last_message_sent_at))
             FROM candidate_outreach WHERE candidate_id = ANY(%s)),
          {status_sql}
        """,
        (ids,) * (9 if has_status else 7),
    )
    calls, last_call, inbound, last_inbound, li, em, last_outreach, status_n, last_status = cur.fetchone()
    lasts = [t for t in (last_call, last_inbound, last_outreach, last_status) if t is not None]
    return {
        "prior_candidate_ids": ids, "calls": int(calls or 0), "inbound_calls": int(inbound or 0),
        "linkedin_replies": int(li or 0), "emails": int(em or 0), "status_changes": int(status_n or 0),
        "last_interaction_at": max(lasts).isoformat() if lasts else None,
    }


def _public_rows(rows: List[dict], exclude: Optional[int] = None) -> List[dict]:
    return [
        {"id": r["id"], "name": r["name"], "owner_email": r["owner_email"],
         "pool_source": r["pool_source"], "is_archived": r["is_archived"]}
        for r in rows if r["id"] != exclude
    ]


def link_candidate(cur, candidate_id: int, *, by: str = "system") -> dict:
    """Link one freshly written row and describe what we already know about
    the person — the payload the Add Candidate modal shows immediately."""
    resolution = resolve_person(cur, candidate_id)
    link_candidates_bulk(cur, [candidate_id], by=by)
    if not resolution:
        return {"known": False}
    prior = _public_rows(resolution.rows, exclude=candidate_id)
    summary = person_summary(cur, resolution.candidate_ids, exclude=candidate_id)
    return {"known": bool(prior), "person_key": resolution.person_key,
            "matched_on": resolution.matched_on, "prior_rows": prior, **summary}


def lookup_person(cur, *, linkedin: Optional[str] = None, email: Optional[str] = None,
                  phone: Optional[str] = None) -> dict:
    """What the Add Candidate modal asks on field blur, before anything is
    saved: is this person already in Hayasa, and what have we done with them?"""
    key, matched_on = person_key_for(None, linkedin, email, phone)
    if not key:
        return {"known": False, "matched_on": MATCHED_NONE}
    li_key = key if matched_on == MATCHED_LINKEDIN else None
    rows = _resolve_rows(cur, None, key=li_key, mail=canonical_email(email), tail=phone_tail(phone))
    summary = person_summary(cur, [r["id"] for r in rows])
    return {"known": bool(rows), "person_key": key, "matched_on": matched_on,
            "prior_rows": _public_rows(rows), **summary}

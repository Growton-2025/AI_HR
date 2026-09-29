"""Merge duplicate candidates created by the legacy migration.

The migration appended "_legacy_<id>" to LinkedIn URLs, so a person
re-uploaded later got a second profile. The two records hold different
things: the upload has the recruiter work (role memberships, outreach, calls,
status history); the legacy record usually has the richer profile (uploaded
fields, work history).

Plan per pair: keep the upload record, copy in the profile data it is
missing from the legacy record, move the legacy record's recruiter work onto
it (skipping anything that would clash), and archive the legacy record
(never delete it; raw_fields._merged_into records where it went).

Categories — only AUTO_merge is ever applied:
  AUTO_merge                 same person, no conflicting recruiter data
  REVIEW_conflict            same person, but statuses / notes / role or
                             outreach rows conflict: a person must decide
  SKIP_recruiter_copy        the upload is a recruiter's own pool copy
  EXCLUDE_different_person   the LinkedIn URLs collide but these are two
                             people (judged by the LLM from name, headline,
                             company, location)

Default is a dry run: nothing is written, a CSV report is produced.
  python3 scripts/merge_legacy_duplicates.py
  python3 scripts/merge_legacy_duplicates.py --apply --confirm <AUTO_merge count>
"""
import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.config import settings  # noqa: E402,F401  (loads .env)
from backend.db.connection import get_db_connection, return_db_connection  # noqa: E402
from backend.services.ai_columns import call_openai_json  # noqa: E402

JUDGE_MODEL = os.getenv("DUPLICATE_JUDGE_MODEL", "gpt-4o")
DEFAULT_REPORT = ROOT / "data" / "_reports" / "duplicate_merge_report.csv"

ACTIVITY_TABLES = [
    "recruitment_role_candidates", "candidate_outreach", "calls", "inbound_calls", "candidate_resumes",
    "ai_column_cells", "candidate_status_history", "candidate_person_links",
]
PROFILE_TABLES = [
    "roles", "education", "titles_held", "company_years", "functional_experiences", "geography_experiences",
    "industry_experiences", "segment_experiences", "experience_gaps", "education_gaps", "industry_gaps",
]
FILL_COLUMNS = [
    "email", "phone", "mobile_phone", "location", "city", "headline", "about", "first_name", "last_name",
    "total_experience_years", "max_people_managed", "years_team_management", "avg_years_in_company", "notes",
]
DEFAULT_STATUSES = {"", "to be started"}


def find_pairs(cur):
    cur.execute(
        """
        SELECT l.id, n.id
        FROM candidates l
        JOIN candidates n
          ON n.normalized_linkedin = regexp_replace(l.normalized_linkedin, '_legacy_[0-9]+$', '')
         AND n.id <> l.id
        WHERE l.normalized_linkedin ~ '_legacy_[0-9]+$'
          AND NOT COALESCE(l.is_archived, false)
          AND NOT COALESCE(n.is_archived, false)
        ORDER BY l.id, n.id
        """
    )
    return cur.fetchall()


def load_candidates(cur, ids):
    cur.execute(
        """SELECT id, name, email, COALESCE(mobile_phone, phone), status, notes, owner_user_id, pool_source,
                  raw_fields, headline, location
           FROM candidates WHERE id = ANY(%s)""",
        (ids,),
    )
    keys = ["id", "name", "email", "phone", "status", "notes", "owner_user_id", "pool_source", "raw_fields", "headline", "location"]
    return {row[0]: dict(zip(keys, row)) for row in cur.fetchall()}


def current_role(cur, ids):
    cur.execute(
        """SELECT DISTINCT ON (r.candidate_id) r.candidate_id, r.title, c.name
           FROM roles r JOIN companies c ON c.id = r.company_id
           WHERE r.candidate_id = ANY(%s) ORDER BY r.candidate_id, r.id""",
        (ids,),
    )
    return {cid: f"{title or ''} at {company or ''}" for cid, title, company in cur.fetchall()}


def _name_key(name):
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(t for t in re.split(r"[^a-z]+", text) if t)


def judge_same_person(pairs_info):
    """LLM decides whether two records whose names differ are one person."""
    verdicts = {}
    for start in range(0, len(pairs_info), 20):
        batch = pairs_info[start:start + 20]
        lines = "\n".join(
            f"{i}. A: {a['name']} | {a.get('headline') or ''} | {a.get('role') or ''} | {a.get('location') or ''}\n"
            f"   B: {b['name']} | {b.get('headline') or ''} | {b.get('role') or ''} | {b.get('location') or ''}"
            for i, (a, b) in enumerate(batch, 1)
        )
        result = call_openai_json(
            "Two candidate records share a LinkedIn URL. For each numbered pair decide whether A and B are the same "
            "person (name spellings, nicknames, emoji or a company appended to a name are the same person; a "
            "different first or last name with an unrelated headline is a different person). "
            'Return JSON only: {"same": [<numbers of pairs that are the same person>]}.',
            lines, model=JUDGE_MODEL, use_web=False, temperature=0.0, timeout=90.0,
            response_format={"type": "json_object"},
        )
        same = {int(n) for n in (result.get("same") or []) if str(n).isdigit()} if isinstance(result, dict) else set()
        for i, (a, b) in enumerate(batch, 1):
            verdicts[(a["id"], b["id"])] = i in same
    return verdicts


def build_plan(cur):
    pairs = find_pairs(cur)
    ids = sorted({i for pair in pairs for i in pair})
    cands = load_candidates(cur, ids)
    roles_of = current_role(cur, ids)
    counts = {}
    for table in ACTIVITY_TABLES + PROFILE_TABLES:
        cur.execute(f"SELECT candidate_id, count(*) FROM {table} WHERE candidate_id = ANY(%s) GROUP BY 1", (ids,))
        counts[table] = dict(cur.fetchall())
    cur.execute("SELECT candidate_id, role_id FROM recruitment_role_candidates WHERE candidate_id = ANY(%s)", (ids,))
    role_ids = {}
    for cid, rid in cur.fetchall():
        role_ids.setdefault(cid, set()).add(rid)
    cur.execute("SELECT candidate_id, recruitment_role_id FROM candidate_outreach WHERE candidate_id = ANY(%s)", (ids,))
    outreach_roles = {}
    for cid, rid in cur.fetchall():
        outreach_roles.setdefault(cid, set()).add(rid)

    to_judge = []
    for legacy, keep in pairs:
        if _name_key(cands[legacy]["name"]) != _name_key(cands[keep]["name"]):
            a = {**cands[legacy], "role": roles_of.get(legacy)}
            b = {**cands[keep], "role": roles_of.get(keep)}
            to_judge.append((a, b))
    same_person = judge_same_person(to_judge) if to_judge else {}

    plan = []
    for legacy, keep in pairs:
        L, K = cands[legacy], cands[keep]
        flags = []
        if (legacy, keep) in same_person and not same_person[(legacy, keep)]:
            category = "EXCLUDE_different_person"
        elif K["owner_user_id"] is not None:
            category = "SKIP_recruiter_copy"
        else:
            if role_ids.get(legacy, set()) & role_ids.get(keep, set()):
                flags.append("same_role_on_both")
            if outreach_roles.get(legacy, set()) & outreach_roles.get(keep, set()):
                flags.append("outreach_same_role_on_both")
            ls, ks = str(L["status"] or "").strip().lower(), str(K["status"] or "").strip().lower()
            if ls not in DEFAULT_STATUSES and ks not in DEFAULT_STATUSES and ls != ks:
                flags.append("status_differs")
            if str(L["notes"] or "").strip() and str(K["notes"] or "").strip() and L["notes"].strip() != K["notes"].strip():
                flags.append("notes_on_both")
            category = "REVIEW_conflict" if flags else "AUTO_merge"
        lraw = L["raw_fields"] if isinstance(L["raw_fields"], dict) else {}
        kraw = K["raw_fields"] if isinstance(K["raw_fields"], dict) else {}
        plan.append({
            "category": category,
            "legacy_id": legacy, "keep_id": keep,
            "legacy_name": L["name"], "keep_name": K["name"],
            "legacy_status": L["status"], "keep_status": K["status"],
            "fields_to_copy": sum(1 for k, v in lraw.items() if str(v or "").strip() and not str(kraw.get(k) or "").strip()),
            **{f"move_{t}": counts[t].get(legacy, 0) for t in ACTIVITY_TABLES},
            **{f"profile_{t}": f"{counts[t].get(legacy, 0)}->{counts[t].get(keep, 0)}" for t in ("roles", "education")},
            "flags": ";".join(flags),
        })
    return plan


def merge_pair(cur, legacy, keep):
    cur.execute("SELECT raw_fields, " + ", ".join(FILL_COLUMNS) + " FROM candidates WHERE id = %s FOR UPDATE", (legacy,))
    lrow = cur.fetchone()
    cur.execute("SELECT raw_fields, " + ", ".join(FILL_COLUMNS) + " FROM candidates WHERE id = %s FOR UPDATE", (keep,))
    krow = cur.fetchone()
    lraw = lrow[0] if isinstance(lrow[0], dict) else {}
    kraw = krow[0] if isinstance(krow[0], dict) else {}
    # Legacy fills gaps; the kept record's own non-empty values always win.
    merged = {**lraw, **{k: v for k, v in kraw.items() if str(v or "").strip()}}
    fills = {
        col: lval for col, lval, kval in zip(FILL_COLUMNS, lrow[1:], krow[1:])
        if lval not in (None, "", 0) and kval in (None, "", 0)
    }
    sets = ", ".join(["raw_fields = %s::jsonb"] + [f"{c} = %s" for c in fills])
    cur.execute(f"UPDATE candidates SET {sets} WHERE id = %s", [json.dumps(merged, default=str), *fills.values(), keep])

    # Profile tables: moved only when the kept record has none of its own.
    for table in PROFILE_TABLES:
        cur.execute(f"SELECT 1 FROM {table} WHERE candidate_id = %s LIMIT 1", (keep,))
        if not cur.fetchone():
            cur.execute(f"UPDATE {table} SET candidate_id = %s WHERE candidate_id = %s", (keep, legacy))

    # Recruiter work: moved unless the kept record already has the same row.
    cur.execute(
        """UPDATE recruitment_role_candidates SET candidate_id = %s WHERE candidate_id = %s
           AND role_id NOT IN (SELECT role_id FROM recruitment_role_candidates WHERE candidate_id = %s)""",
        (keep, legacy, keep))
    cur.execute(
        """UPDATE candidate_outreach o SET candidate_id = %s WHERE o.candidate_id = %s
           AND NOT EXISTS (SELECT 1 FROM candidate_outreach k WHERE k.candidate_id = %s
                           AND k.recruitment_role_id IS NOT DISTINCT FROM o.recruitment_role_id)""",
        (keep, legacy, keep))
    cur.execute(
        """UPDATE calls c SET candidate_id = %s WHERE c.candidate_id = %s
           AND NOT (c.status = 'pending' AND EXISTS (SELECT 1 FROM calls k WHERE k.candidate_id = %s
                    AND k.status = 'pending' AND k.list_id IS NOT DISTINCT FROM c.list_id))""",
        (keep, legacy, keep))
    cur.execute("UPDATE inbound_calls SET candidate_id = %s WHERE candidate_id = %s", (keep, legacy))
    cur.execute("SELECT 1 FROM candidate_resumes WHERE candidate_id = %s AND is_current", (keep,))
    if cur.fetchone():
        cur.execute("UPDATE candidate_resumes SET is_current = false WHERE candidate_id = %s", (legacy,))
    cur.execute("UPDATE candidate_resumes SET candidate_id = %s WHERE candidate_id = %s", (keep, legacy))
    cur.execute(
        """UPDATE ai_column_cells a SET candidate_id = %s WHERE a.candidate_id = %s
           AND NOT EXISTS (SELECT 1 FROM ai_column_cells k WHERE k.candidate_id = %s
                           AND k.column_definition_id = a.column_definition_id)""",
        (keep, legacy, keep))
    cur.execute("UPDATE candidate_status_history SET candidate_id = %s WHERE candidate_id = %s", (keep, legacy))
    cur.execute("SELECT 1 FROM candidate_person_links WHERE candidate_id = %s", (keep,))
    if not cur.fetchone():
        cur.execute("UPDATE candidate_person_links SET candidate_id = %s WHERE candidate_id = %s", (keep, legacy))

    # Archive, never delete: the record and anything left on it stay recoverable.
    cur.execute(
        """UPDATE candidates SET is_archived = true,
               raw_fields = COALESCE(raw_fields, '{}'::jsonb) || jsonb_build_object('_merged_into', %s)
           WHERE id = %s""",
        (keep, legacy))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="CSV report path (contains personal data; keep out of git)")
    parser.add_argument("--apply", action="store_true", help="merge the AUTO_merge pairs (default: dry run)")
    parser.add_argument("--confirm", type=int, help="with --apply: the exact AUTO_merge count from the dry run")
    args = parser.parse_args()

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            plan = build_plan(cur)
        conn.rollback()  # the planning phase never writes
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(plan[0]) if plan else ["category"])
            writer.writeheader()
            writer.writerows(plan)
        summary = {}
        for row in plan:
            summary[row["category"]] = summary.get(row["category"], 0) + 1
        auto = [r for r in plan if r["category"] == "AUTO_merge"]
        print(f"pairs: {len(plan)}  {summary}")
        print(f"report: {report}")
        if not args.apply:
            print("dry run: nothing was changed")
            return
        if args.confirm != len(auto):
            sys.exit(f"--confirm must equal the AUTO_merge count ({len(auto)}); nothing was changed")
        merged = failed = 0
        for row in auto:
            try:
                with conn.cursor() as cur:
                    merge_pair(cur, row["legacy_id"], row["keep_id"])
                conn.commit()
                merged += 1
            except Exception as exc:  # one pair failing must not block or half-apply the others
                conn.rollback()
                failed += 1
                print(f"pair {row['legacy_id']}->{row['keep_id']} failed and was rolled back: {exc}")
        print(f"merged {merged}, failed {failed}; restart the backend (or refresh its profile cache) to see the result")
    finally:
        return_db_connection(conn)


if __name__ == "__main__":
    main()

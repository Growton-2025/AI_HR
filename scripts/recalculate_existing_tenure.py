import sys
import os
import json
from datetime import datetime

# Adjust Python path to include the backend directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.connection import get_db_connection_context

def parse_date(date_str):
    if not date_str:
        return None
    try:
        clean_str = date_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(clean_str)
        return dt.replace(tzinfo=None)
    except Exception:
        try:
            date_part = date_str.split('T')[0].split()[0]
            dt = datetime.strptime(date_part, "%Y-%m-%d")
            return dt.replace(tzinfo=None)
        except Exception:
            return None

def merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged

def calculate_candidate_tenure(roles, raw_total_exp):
    """Compute total_experience_years and avg_tenure_years from a list of DB role dicts.

    Bug-2 fix: average tenure is computed only over *completed* (past) companies,
    matching the canonical calculate_tenure_metrics logic in import_enrichment.py.
    Bug-3 fix: the numerator for average tenure is the sum of per-company months
    (not the merged total career span), also matching the canonical logic.
    Bug-1 fix: months_between now uses the corrected formula (subtract 1 when
    end.day < start.day) instead of the old inflating +1.
    """
    # 1. Identify the most recent company (canonical current company)
    most_recent_company = None
    latest_start = None

    for r in roles:
        details_text = r.get("details") or ""
        if details_text.strip().startswith('{'):
            try:
                parsed = json.loads(details_text)
                sd = parse_date(parsed.get("start_date"))
                if sd:
                    if latest_start is None or sd > latest_start:
                        latest_start = sd
                        most_recent_company = (r.get("company_name") or "").lower().strip()
            except Exception:
                pass

    # If no start date was found but we have roles, fall back to the first role's company
    if not most_recent_company and roles:
        for r in roles:
            comp = (r.get("company_name") or "").lower().strip()
            if comp:
                most_recent_company = comp
                break

    intervals = []
    undated_months = 0
    today = datetime.now()

    # Per-company tracking: company_norm -> {months, is_current}
    company_data: dict = {}

    for r in roles:
        company = (r.get("company_name") or "").lower().strip()
        if not company:
            continue

        duration_years = r.get("duration_years") or 0.0
        details_text = r.get("details") or ""

        start_dt = None
        end_dt = None

        if details_text.strip().startswith('{'):
            try:
                parsed = json.loads(details_text)
                start_dt = parse_date(parsed.get("start_date"))
                end_dt = parse_date(parsed.get("end_date"))
            except Exception:
                pass

        if start_dt:
            if not end_dt:
                end_dt = today
            intervals.append((start_dt, end_dt))
            # Per-company month accumulation
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if end_dt.day < start_dt.day:   # Bug-1 fix
                months -= 1
            months = max(months, 0)
        else:
            # Undated role: use duration_years as fallback
            months = int(round(duration_years * 12.0))
            undated_months += months

        bucket = company_data.setdefault(company, {"months": 0, "is_current": False})
        bucket["months"] += months
        if company == most_recent_company:
            bucket["is_current"] = True

    merged = merge_intervals(intervals)
    dated_months = 0
    for start, end in merged:
        if end >= start:
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if end.day < start.day:   # Bug-1 fix
                months -= 1
            dated_months += max(months, 0)

    total_months = dated_months + undated_months
    date_total_years = round(total_months / 12.0, 2)

    total_experience_years = date_total_years
    if raw_total_exp is not None and raw_total_exp > date_total_years:
        total_experience_years = round(raw_total_exp, 2)
        total_months = int(round(raw_total_exp * 12.0))

    # Bug-2 & Bug-3 fix: average over completed companies only, using per-company sums.
    completed = [data for data in company_data.values() if not data["is_current"]]
    completed_company_count = len(completed)
    completed_company_months = sum(d["months"] for d in completed)
    avg_years = (
        round((completed_company_months / 12.0) / completed_company_count, 2)
        if completed_company_count > 0
        else 0.0
    )

    return total_experience_years, avg_years

MAX_TOTAL_EXP_YEARS = 40  # Safety cap: skip update if computed exp exceeds this
BATCH_COMMIT_SIZE = 500   # Commit every N updates to avoid one giant transaction



def main(owner_user_id: int = None, dry_run: bool = False):
    scope_label = f"owner_user_id = {owner_user_id}" if owner_user_id else "ALL candidates"
    print(f"Connecting to database... (scope: {scope_label}, dry_run={dry_run})")

    with get_db_connection_context() as conn:
        with conn.cursor() as cur:

            # 1. Fetch candidates (all, or filtered by owner)
            if owner_user_id:
                print(f"Fetching candidates for owner_user_id = {owner_user_id}...")
                cur.execute("""
                    SELECT id, raw_fields, total_experience_years, avg_years_in_company
                    FROM candidates
                    WHERE owner_user_id = %s
                      AND COALESCE(is_archived, FALSE) = FALSE
                    ORDER BY id;
                """, (owner_user_id,))
            else:
                print("Fetching ALL non-archived candidates...")
                cur.execute("""
                    SELECT id, raw_fields, total_experience_years, avg_years_in_company
                    FROM candidates
                    WHERE COALESCE(is_archived, FALSE) = FALSE
                    ORDER BY id;
                """)
            candidates = cur.fetchall()
            print(f"Loaded {len(candidates)} candidates.")

            # 2. Fetch all roles (filtered to match candidate scope)
            print("Fetching roles...")
            if owner_user_id:
                cur.execute("""
                    SELECT r.candidate_id, c.name as company_name, r.duration_years, r.details
                    FROM roles r
                    LEFT JOIN companies c ON r.company_id = c.id
                    WHERE r.candidate_id IN (
                        SELECT id FROM candidates
                        WHERE owner_user_id = %s
                          AND COALESCE(is_archived, FALSE) = FALSE
                    );
                """, (owner_user_id,))
            else:
                cur.execute("""
                    SELECT r.candidate_id, c.name as company_name, r.duration_years, r.details
                    FROM roles r
                    LEFT JOIN companies c ON r.company_id = c.id
                    WHERE r.candidate_id IN (
                        SELECT id FROM candidates
                        WHERE COALESCE(is_archived, FALSE) = FALSE
                    );
                """)
            all_roles = cur.fetchall()
            print(f"Loaded {len(all_roles)} roles.")

            # Group roles by candidate_id
            roles_by_candidate = {}
            for r in all_roles:
                cid = r[0]
                roles_by_candidate.setdefault(cid, []).append({
                    "company_name": r[1],
                    "duration_years": float(r[2]) if r[2] is not None else 0.0,
                    "details": r[3],
                })

            # 3. Process each candidate
            print(f"\nRecalculating tenures{'  [DRY RUN — no writes]' if dry_run else ''}...")
            update_count = 0
            skipped_count = 0
            decreased_log = []   # exp went DOWN (correction of inflation)
            increased_log = []   # exp went UP (undercounted before)
            capped_log = []      # skipped: computed value > MAX_TOTAL_EXP_YEARS
            total_count = len(candidates)
            pending_commit = 0

            for idx, candidate in enumerate(candidates):
                cid = candidate[0]
                raw_fields_str = candidate[1]
                old_total_exp = float(candidate[2]) if candidate[2] is not None else 0.0
                old_avg_tenure = float(candidate[3]) if candidate[3] is not None else 0.0

                if (idx + 1) % 500 == 0 or (idx + 1) == total_count:
                    print(f"  Processed {idx + 1}/{total_count} candidates"
                          f" | updated so far: {update_count} ...")

                # Parse raw total experience from raw_fields fallback if present
                raw_total_exp = None
                if raw_fields_str:
                    try:
                        raw_fields = json.loads(raw_fields_str) if isinstance(raw_fields_str, str) else raw_fields_str
                        raw_total_exp = float(raw_fields.get("totalExperienceYears") or 0.0) or None
                    except Exception:
                        pass

                roles = roles_by_candidate.get(cid, [])
                new_total_exp, new_avg_tenure = calculate_candidate_tenure(roles, raw_total_exp)

                # Safety cap: skip candidates with implausibly high experience (bad role dates)
                if new_total_exp > MAX_TOTAL_EXP_YEARS:
                    capped_log.append(
                        f"  Candidate ID {cid}: computed {new_total_exp:.2f} yrs > cap {MAX_TOTAL_EXP_YEARS} — SKIPPED"
                    )
                    skipped_count += 1
                    continue

                changed = (
                    abs(old_total_exp - new_total_exp) > 0.01
                    or abs(old_avg_tenure - new_avg_tenure) > 0.01
                )
                if not changed:
                    skipped_count += 1
                    continue

                log_line = (
                    f"  Candidate ID {cid}: "
                    f"Exp {old_total_exp:.2f} -> {new_total_exp:.2f} yrs | "
                    f"AvgTenure {old_avg_tenure:.2f} -> {new_avg_tenure:.2f} yrs"
                )
                if new_total_exp < old_total_exp:
                    decreased_log.append(log_line)
                else:
                    increased_log.append(log_line)

                if not dry_run:
                    executed = False
                    for attempt in range(3):
                        try:
                            cur.execute("""
                                UPDATE candidates
                                SET total_experience_years = %s,
                                    avg_years_in_company    = %s,
                                    updated_at              = NOW()
                                WHERE id = %s;
                            """, (new_total_exp, new_avg_tenure, cid))
                            executed = True
                            break
                        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                            print(f"\n[Warning] Connection error during update for Candidate ID {cid}: {e}. Reconnecting...")
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            try:
                                cur.close()
                                conn.close()
                            except Exception:
                                pass
                            try:
                                from backend.db.connection import get_db_connection
                                conn = get_db_connection(validate=True)
                                cur = conn.cursor()
                                pending_commit = 0
                            except Exception as re_err:
                                print(f"[Error] Reconnection failed: {re_err}")
                                import time
                                time.sleep(5)

                    if not executed:
                        print(f"[Error] Failed to update Candidate ID {cid} after retries. Skipping.")
                        continue

                    pending_commit += 1
                    update_count += 1

                    # Batch commit to avoid one giant transaction
                    if pending_commit >= BATCH_COMMIT_SIZE:
                        committed = False
                        for commit_attempt in range(3):
                            try:
                                conn.commit()
                                committed = True
                                break
                            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                                print(f"\n[Warning] Connection error during commit: {e}. Reconnecting...")
                                try:
                                    cur.close()
                                    conn.close()
                                except Exception:
                                    pass
                                try:
                                    from backend.db.connection import get_db_connection
                                    conn = get_db_connection(validate=True)
                                    cur = conn.cursor()
                                    pending_commit = 0
                                except Exception as re_err:
                                    print(f"[Error] Reconnection failed: {re_err}")
                                    import time
                                    time.sleep(5)
                        if committed:
                            print(f"    [batch commit after {update_count} updates]")
                            pending_commit = 0
                else:
                    update_count += 1  # count as "would update" in dry run

            # Final commit for remaining rows
            if not dry_run and pending_commit:
                try:
                    conn.commit()
                except Exception as e:
                    print(f"[Warning] Failed final commit: {e}")

            # ── Summary ──────────────────────────────────────────────────────
            action = "Would update" if dry_run else "Updated"
            print(f"\n{'='*60}")
            print(f"{'DRY RUN COMPLETE' if dry_run else 'COMPLETE'} — scope: {scope_label}")
            print(f"  Candidates checked : {total_count}")
            print(f"  Unchanged (skipped): {skipped_count - len(capped_log)}")
            print(f"  Capped anomalies   : {len(capped_log)} (exp > {MAX_TOTAL_EXP_YEARS} yrs — not written)")
            print(f"  {action}           : {update_count}")
            print(f"    ↓ Decreased (over-inflation fixed): {len(decreased_log)}")
            print(f"    ↑ Increased (under-count fixed)  : {len(increased_log)}")
            print(f"{'='*60}")

            if capped_log:
                print(f"\n--- Capped Anomalies ({len(capped_log)}) — NOT written ---")
                for line in capped_log:
                    print(line)

            if decreased_log:
                print(f"\n--- Decreased ({len(decreased_log)}) ---")
                for line in decreased_log[:50]:
                    print(line)
                if len(decreased_log) > 50:
                    print(f"  ... and {len(decreased_log) - 50} more")

            if increased_log:
                print(f"\n--- Increased ({len(increased_log)}) ---")
                for line in increased_log[:50]:
                    print(line)
                if len(increased_log) > 50:
                    print(f"  ... and {len(increased_log) - 50} more")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Recalculate tenure and experience for all (or filtered) candidates."
    )
    parser.add_argument(
        "--owner-user-id", type=int, default=None,
        help="Only process candidates belonging to this owner_user_id. Omit for ALL candidates."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Calculate and log diffs but do NOT write to the database."
    )
    args = parser.parse_args()
    main(owner_user_id=args.owner_user_id, dry_run=args.dry_run)

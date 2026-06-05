import sys
import os
import json
from datetime import datetime

# Adjust Python path to include the backend directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.connection import get_db_connection_context

MAX_TOTAL_EXP_YEARS = 40  # Safety cap

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
    company_data = {}

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
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if end_dt.day < start_dt.day:
                months -= 1
            months = max(months, 0)
        else:
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
            if end.day < start.day:
                months -= 1
            dated_months += max(months, 0)

    total_months = dated_months + undated_months
    date_total_years = round(total_months / 12.0, 2)

    total_experience_years = date_total_years
    if raw_total_exp is not None and raw_total_exp > date_total_years:
        total_experience_years = round(raw_total_exp, 2)

    completed = [data for data in company_data.values() if not data["is_current"]]
    completed_company_count = len(completed)
    completed_company_months = sum(d["months"] for d in completed)
    avg_years = (
        round((completed_company_months / 12.0) / completed_company_count, 2)
        if completed_company_count > 0
        else 0.0
    )

    return total_experience_years, avg_years

def main():
    print("Connecting to database for verification...")
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            # Fetch candidates
            cur.execute("""
                SELECT id, raw_fields, total_experience_years, avg_years_in_company
                FROM candidates
                WHERE COALESCE(is_archived, FALSE) = FALSE
                ORDER BY id;
            """)
            candidates = cur.fetchall()
            print(f"Loaded {len(candidates)} candidates.")

            # Fetch roles
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

            print("\nVerifying calculations...")
            verified_count = 0
            matching_count = 0
            capped_anomaly_count = 0
            mismatch_log = []

            for candidate in candidates:
                cid = candidate[0]
                raw_fields_str = candidate[1]
                db_total_exp = float(candidate[2]) if candidate[2] is not None else 0.0
                db_avg_tenure = float(candidate[3]) if candidate[3] is not None else 0.0

                # Parse raw total experience
                raw_total_exp = None
                if raw_fields_str:
                    try:
                        raw_fields = json.loads(raw_fields_str) if isinstance(raw_fields_str, str) else raw_fields_str
                        raw_total_exp = float(raw_fields.get("totalExperienceYears") or 0.0) or None
                    except Exception:
                        pass

                roles = roles_by_candidate.get(cid, [])
                computed_total_exp, computed_avg_tenure = calculate_candidate_tenure(roles, raw_total_exp)

                # Check if it was skipped due to safety cap (> 40 years)
                if computed_total_exp > MAX_TOTAL_EXP_YEARS:
                    capped_anomaly_count += 1
                    verified_count += 1
                    continue

                # Compare values with floating point tolerance
                matches = (
                    abs(db_total_exp - computed_total_exp) <= 0.01
                    and abs(db_avg_tenure - computed_avg_tenure) <= 0.01
                )

                if matches:
                    matching_count += 1
                else:
                    mismatch_log.append({
                        "id": cid,
                        "db_exp": db_total_exp,
                        "computed_exp": computed_total_exp,
                        "db_avg": db_avg_tenure,
                        "computed_avg": computed_avg_tenure
                    })

                verified_count += 1

            # Print Summary
            print(f"\n{'='*60}")
            print("VERIFICATION SUMMARY")
            print(f"  Total Candidates Checked : {verified_count}")
            print(f"  Matching Calculations    : {matching_count}")
            print(f"  Capped Anomalies (Skipped): {capped_anomaly_count}")
            print(f"  Mismatches Detected      : {len(mismatch_log)}")
            print(f"{'='*60}")

            if mismatch_log:
                print("\n--- Detected Mismatches ---")
                for m in mismatch_log[:20]:
                    print(f"  Candidate ID {m['id']}:")
                    print(f"    Total Exp  - DB: {m['db_exp']:.2f} | Computed: {m['computed_exp']:.2f}")
                    print(f"    Avg Tenure - DB: {m['db_avg']:.2f} | Computed: {m['computed_avg']:.2f}")
                if len(mismatch_log) > 20:
                    print(f"  ... and {len(mismatch_log) - 20} more mismatches.")
                sys.exit(1)
            else:
                print("\n✅ Success: All calculated database values match the metrics formulas perfectly!")
                sys.exit(0)

if __name__ == "__main__":
    main()

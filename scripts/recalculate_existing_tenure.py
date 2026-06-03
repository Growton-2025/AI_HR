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
    intervals = []
    undated_months = 0
    unique_companies = set()
    today = datetime.now()
    
    for r in roles:
        company = r.get("company_name")
        if company:
            unique_companies.add(company.lower().strip())
            
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
        else:
            undated_months += int(round(duration_years * 12.0))
            
    merged = merge_intervals(intervals)
    dated_months = 0
    for start, end in merged:
        if end >= start:
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if end.day >= start.day:
                months += 1
            dated_months += max(months, 0)
            
    total_months = dated_months + undated_months
    date_total_years = round(total_months / 12.0, 2)
    
    total_experience_years = date_total_years
    if raw_total_exp is not None and raw_total_exp > date_total_years:
        total_experience_years = round(raw_total_exp, 2)
        total_months = int(round(raw_total_exp * 12.0))
        
    company_count = len(unique_companies)
    avg_years = round((total_months / 12.0) / company_count, 2) if company_count > 0 else 0.0
    
    return total_experience_years, avg_years

def main():
    print("Connecting to database...")
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            # 1. Fetch only recently uploaded candidate records (owner_user_id = 29)
            print("Fetching recently uploaded candidates (owner_user_id = 29)...")
            cur.execute("""
                SELECT id, raw_fields, total_experience_years, avg_years_in_company
                FROM candidates
                WHERE owner_user_id = 29
                ORDER BY id;
            """)
            candidates = cur.fetchall()
            print(f"Loaded {len(candidates)} candidates.")
            
            # 2. Fetch roles only for these candidates
            print("Fetching candidate roles...")
            cur.execute("""
                SELECT r.candidate_id, c.name as company_name, r.duration_years, r.details
                FROM roles r
                LEFT JOIN companies c ON r.company_id = c.id
                WHERE r.candidate_id IN (SELECT id FROM candidates WHERE owner_user_id = 29);
            """)
            all_roles = cur.fetchall()
            print(f"Loaded {len(all_roles)} roles.")
            
            # Group roles by candidate ID
            roles_by_candidate = {}
            for r in all_roles:
                cid = r[0]
                roles_by_candidate.setdefault(cid, []).append({
                    "company_name": r[1],
                    "duration_years": float(r[2]) if r[2] is not None else 0.0,
                    "details": r[3]
                })
                
            # 3. Process each candidate
            print("Recalculating candidate tenures...")
            update_count = 0
            adjusted_log = []
            
            total_count = len(candidates)
            for idx, candidate in enumerate(candidates):
                cid = candidate[0]
                raw_fields_str = candidate[1]
                old_total_exp = float(candidate[2]) if candidate[2] is not None else 0.0
                old_avg_tenure = float(candidate[3]) if candidate[3] is not None else 0.0
                
                # Print progress periodically
                if (idx + 1) % 100 == 0 or (idx + 1) == total_count:
                    print(f"Processed {idx + 1}/{total_count} candidates...")
                
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
                
                if abs(old_total_exp - new_total_exp) > 0.01 or abs(old_avg_tenure - new_avg_tenure) > 0.01:
                    cur.execute("""
                        UPDATE candidates
                        SET total_experience_years = %s, avg_years_in_company = %s, updated_at = NOW()
                        WHERE id = %s;
                    """, (new_total_exp, new_avg_tenure, cid))
                    update_count += 1
                    
                    if old_total_exp > new_total_exp:
                        adjusted_log.append(
                            f"Candidate ID {cid}: Exp: {old_total_exp} -> {new_total_exp} yrs | "
                            f"Avg Tenure: {old_avg_tenure} -> {new_avg_tenure} yrs"
                        )
            
            conn.commit()
            print("\nDatabase transaction committed successfully!")
            print(f"Total candidates checked: {total_count}")
            print(f"Total candidates updated: {update_count}")
            print(f"Total candidates with reduced/corrected overlaps: {len(adjusted_log)}")
            
            if adjusted_log:
                print("\nAdjustment Details:")
                for log in adjusted_log:
                    print(log)

if __name__ == "__main__":
    main()

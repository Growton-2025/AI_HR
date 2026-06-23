from backend.db.connection import get_db_connection_context
from backend.services.import_enrichment import parse_roles_from_raw, calculate_tenure_metrics

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, raw_fields 
                FROM candidates 
                WHERE owner_user_id = 29 AND total_experience_years = 0;
            """)
            rows = cur.fetchall()
            fixed_count = 0
            
            for row in rows:
                c_id, raw_data = row
                if not raw_data:
                    continue
                    
                roles = parse_roles_from_raw(raw_data, None)
                tenure = calculate_tenure_metrics(roles, raw_total_exp_years=float(raw_data.get("totalExperienceYears") or 0))
                
                total_exp = tenure.get("total_experience_years") or 0
                avg_tenure = tenure.get("avg_tenure_years") or 0
                
                if total_exp > 0:
                    cur.execute("""
                        UPDATE candidates 
                        SET total_experience_years = %s, avg_years_in_company = %s 
                        WHERE id = %s;
                    """, (total_exp, avg_tenure, c_id))
                    fixed_count += 1
            
            conn.commit()
            print(f"Successfully recalculated and fixed {fixed_count} candidates!")

if __name__ == "__main__":
    main()

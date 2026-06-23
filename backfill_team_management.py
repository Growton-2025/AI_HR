import json
from backend.db.connection import get_db_connection_context
from backend.services.import_enrichment import extract_profile_claims

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, raw_fields 
                FROM candidates 
                WHERE owner_user_id = 29;
            """)
            rows = cur.fetchall()
            fixed_count = 0
            
            print(f"Found {len(rows)} candidates to backfill.")
            for row in rows:
                c_id, raw_data = row
                if not raw_data:
                    continue
                    
                # Wrap the raw data so extract_profile_claims can read it properly!
                candidate_dict = {"raw_fields": raw_data}
                claims = extract_profile_claims(candidate_dict)
                
                max_people = claims.get("max_people_managed", 0)
                years_team = claims.get("years_team_management", 0)
                
                if max_people > 0 or years_team > 0:
                    cur.execute("""
                        UPDATE candidates 
                        SET max_people_managed = %s, years_team_management = %s 
                        WHERE id = %s;
                    """, (max_people, years_team, c_id))
                    fixed_count += 1
            
            conn.commit()
            print(f"Successfully ran LLM extraction and updated {fixed_count} candidates with team management metrics!")

if __name__ == "__main__":
    main()

from backend.db.connection import get_db_connection_context
import json

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, raw_fields 
                FROM candidates 
                WHERE owner_user_id = 29 AND max_people_managed = 0;
            """)
            rows = cur.fetchall()
            found_suspicious = 0
            
            for row in rows:
                c_id, name, raw_data = row
                if not raw_data: continue
                
                # convert to string to search
                raw_str = json.dumps(raw_data).lower()
                
                # Look for numbers near 'team' or 'manage'
                import re
                if re.search(r'(team of \d+|managed \d+|leading \d+)', raw_str):
                    print(f"SUSPICIOUS MISS - Candidate: {name} (ID {c_id})")
                    found_suspicious += 1
            
            print(f"Total suspicious misses found by Regex: {found_suspicious}")

if __name__ == "__main__":
    main()

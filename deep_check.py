from backend.db.connection import get_db_connection_context
import json, re

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, max_people_managed, raw_fields 
                FROM candidates 
                WHERE owner_user_id = 29;
            """)
            rows = cur.fetchall()
            
            for row in rows:
                c_id, name, db_max_people, raw_data = row
                if not raw_data: continue
                
                raw_str = json.dumps(raw_data).lower()
                
                # Check if the DB has them at 0
                if db_max_people == 0:
                    # Robust search for any sign of numbers related to team/managing
                    matches = re.findall(r'(.{0,30}(?:team of|managed|managing|lead|leading|direct reports).{0,30}\d+.*?)', raw_str)
                    if matches:
                        print(f"\n--- POTENTIAL MISS: {name} (ID {c_id}) ---")
                        for m in set(matches):
                            print(f"Context: ...{m}...")

if __name__ == "__main__":
    main()

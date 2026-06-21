from backend.db.connection import get_db_connection_context

def check_counts():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM candidates;")
            total = cur.fetchone()[0]
            print(f"Total candidates: {total}")
            
            cur.execute("SELECT COUNT(*) FROM candidates WHERE total_experience_years IS NOT NULL;")
            enriched = cur.fetchone()[0]
            print(f"Candidates with computed experience (enriched): {enriched}")

            cur.execute("SELECT COUNT(*) FROM roles;")
            roles_count = cur.fetchone()[0]
            print(f"Total roles: {roles_count}")

            cur.execute("SELECT COUNT(DISTINCT candidate_id) FROM roles;")
            c_roles = cur.fetchone()[0]
            print(f"Candidates with roles: {c_roles}")
            
if __name__ == "__main__":
    check_counts()

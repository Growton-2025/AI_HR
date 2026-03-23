
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def check_roles(user_id):
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require"
        )
        cur = conn.cursor()
        
        print(f"--- FETCHING ROLES FOR USER {user_id} ---")
        cur.execute("""
            SELECT r.id, r.name, COUNT(rc.candidate_id) as candidate_count
            FROM recruitment_roles r
            LEFT JOIN recruitment_role_candidates rc ON r.id = rc.role_id
            WHERE r.user_id = %s
            GROUP BY r.id, r.name
            ORDER BY r.created_at DESC
        """, (user_id,))
        
        rows = cur.fetchall()
        roles_list = []
        for row in rows:
            roles_list.append({
                "id": row[0],
                "name": row[1],
                "candidate_count": row[2]
            })
            print(row)
            
        print("\nResult JSON structure:")
        print({"roles": roles_list})
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_roles(4)

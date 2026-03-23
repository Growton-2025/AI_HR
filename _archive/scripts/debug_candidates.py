import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "ai_hr_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None

def debug_candidates():
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            # Get role id for 'test'
            cur.execute("SELECT id FROM recruitment_roles WHERE name = 'test'")
            role = cur.fetchone()
            if not role:
                print("Role 'test' not found")
                return
            role_id = role[0]
            print(f"Role ID: {role_id}")

            # Get candidates
            cur.execute(f"""
                SELECT c.id, c.name, c.about, c.headline 
                FROM candidates c
                JOIN recruitment_role_candidates rc ON c.id = rc.candidate_id
                WHERE rc.role_id = %s
            """, (role_id,))
            
            rows = cur.fetchall()
            print(f"Found {len(rows)} candidates:")
            for row in rows:
                print(f"ID: {row[0]}, Name: {row[1]}")
                print(f"About (len): {len(row[2]) if row[2] else 0}")
                print(f"Headline: {row[3]}")
                print("-" * 20)

    finally:
        conn.close()

if __name__ == "__main__":
    debug_candidates()


import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def check_outreach():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "growton"),
            user=os.getenv("DB_USER", "growton"),
            password=os.getenv("DB_PASSWORD", "Postgres-2026"),
            host=os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com"),
            port=os.getenv("DB_PORT", "5432"),
            sslmode="require"
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        print("Checking outreach mapping for candidate 2519 in role 44...")
        cur.execute("""
            SELECT co.*, c.linkedin 
            FROM candidate_outreach co
            JOIN candidates c ON c.id = co.candidate_id
            WHERE co.candidate_id = 2519 AND co.recruitment_role_id = 44;
        """)
        
        row = cur.fetchone()
        if row:
            print("\nFOUND RECORD:")
            for key, val in row.items():
                print(f"  {key}: {val}")
        else:
            print("\n❌ NO RECORD FOUND in candidate_outreach for this candidate/role.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Database check failed: {e}")

if __name__ == "__main__":
    check_outreach()

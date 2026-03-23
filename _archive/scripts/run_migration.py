import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def run_migration():
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
        
        print("Running migration: add_outreach_tracking.sql")
        
        with open('backend/db/migrations/add_outreach_tracking.sql', 'r') as f:
            sql = f.read()
        
        cur.execute(sql)
        conn.commit()
        
        print("✅ Migration successful!")
        print("   Table 'candidate_outreach' created")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    run_migration()


import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def apply_fix():
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
        
        print("Adding missing columns to candidate_outreach...")
        
        sql = """
        ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS heyreach_campaign_id VARCHAR(255);
        ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_status VARCHAR(50);
        ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_last_action_at TIMESTAMP;
        ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_response_text TEXT;
        """
        
        cur.execute(sql)
        conn.commit()
        
        print("✅ Database fix successful!")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Fix failed: {e}")

if __name__ == "__main__":
    apply_fix()

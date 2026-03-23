import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def migrate():
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
        
        print("Adding new columns to candidate_outreach table...")
        
        # Add li_sent_count
        cur.execute("ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_sent_count INTEGER DEFAULT 0;")
        # Add li_response_received_at
        cur.execute("ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_response_received_at TIMESTAMP;")
        # Add li_conversation_id
        cur.execute("ALTER TABLE candidate_outreach ADD COLUMN IF NOT EXISTS li_conversation_id VARCHAR(255);")
        
        conn.commit()
        print("✅ Database migration successful!")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    migrate()


import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def update_linkedin():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "growton"),
            user=os.getenv("DB_USER", "growton"),
            password=os.getenv("DB_PASSWORD", "Postgres-2026"),
            host=os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com"),
            port=os.getenv("DB_PORT", "5432"),
            sslmode="require"
        )
        cur = conn.cursor()
        
        real_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"
        print(f"Updating candidate 2519 with real URL: {real_url}")
        
        cur.execute("UPDATE candidates SET linkedin = %s WHERE id = 2519;", (real_url,))
        conn.commit()
        print("✅ Candidates table updated.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Database update failed: {e}")

if __name__ == "__main__":
    update_linkedin()

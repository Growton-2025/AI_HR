import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def check_db():
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
        
        print("Checking candidate 2519...")
        cur.execute("SELECT id, name, linkedin FROM candidates WHERE id = 2519;")
        row = cur.fetchone()
        if row:
            print(f"ID: {row[0]}")
            print(f"Name: {row[1]}")
            print(f"LinkedIn: {row[2]}")
        else:
            print("Candidate 2519 not found.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB Check failed: {e}")

if __name__ == "__main__":
    check_db()

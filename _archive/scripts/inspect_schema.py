import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def list_tables():
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
        
        print("--- COMPANIES SCHEMA ---")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'companies'
        """)
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")

        print("\n--- CANDIDATE_OUTREACH SCHEMA ---")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'candidate_outreach'
        """)
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")

            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_tables()

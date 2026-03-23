
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variables or defaults matching connection.py
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "growton"),
    user=os.getenv("DB_USER", "growton"),
    password=os.getenv("DB_PASSWORD", "Postgres-2026"),
    host=os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com"),
    port=os.getenv("DB_PORT", "5432"),
    sslmode="require"
)

with conn.cursor() as cur:
    try:
        cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS email VARCHAR(255);")
        cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS mobile_phone VARCHAR(50);")
        conn.commit()
        print("Columns 'email' and 'mobile_phone' added successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")

conn.close()

import os
import psycopg2
import time
from dotenv import load_dotenv

# Load env from up two levels (standard project structure)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

print(f"Connecting to {DB_HOST}...")
try:
    start = time.time()
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    print(f"Connected in {time.time() - start:.2f}s")
    
    cur = conn.cursor()
    print("Querying candidates count...")
    cur.execute("SELECT count(*) FROM candidates")
    count = cur.fetchone()[0]
    print(f"Total candidates: {count}")
    
    print("Querying all profiles (simulating load)...")
    start = time.time()
    cur.execute("SELECT id, name FROM candidates")
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} rows in {time.time() - start:.2f}s")
    
    cur.close()
    conn.close()
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

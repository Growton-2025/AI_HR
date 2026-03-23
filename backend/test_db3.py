import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=os.getenv("DB_PORT", "5432")
    )
    cur = conn.cursor()
    cur.execute("SELECT raw_fields->>'department' FROM candidates WHERE raw_fields->>'department' IS NOT NULL LIMIT 5")
    print("Dept:", cur.fetchall())
    
    cur.execute("SELECT raw_fields->>'functions' FROM candidates WHERE raw_fields->>'functions' IS NOT NULL LIMIT 5")
    print("Funcs:", cur.fetchall())
    
    cur.execute("SELECT raw_fields->>'job_family' FROM candidates WHERE raw_fields->>'job_family' IS NOT NULL LIMIT 5")
    print("Job Family:", cur.fetchall())

except Exception as e:
    print("DB error:", e)

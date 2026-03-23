import sqlite3

try:
    conn = sqlite3.connect('db/aihr.db')
    cur = conn.cursor()
    cur.execute("SELECT raw_fields FROM candidates LIMIT 1")
    row = cur.fetchone()
    if row:
        print("Candidate raw_fields:", row[0])
except Exception as e:
    print("sqlite error:", e)

try:
    import psycopg2
    import os
    from dotenv import load_dotenv
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=os.getenv("DB_PORT", "5432")
    )
    cur = conn.cursor()
    cur.execute("SELECT raw_fields FROM candidates LIMIT 1")
    row = cur.fetchone()
    if row:
        print("Candidate raw_fields from PG:", type(row[0]), row[0].keys() if isinstance(row[0], dict) else row[0])
except Exception as e:
    print("pg error:", e)

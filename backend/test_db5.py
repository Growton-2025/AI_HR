import sys
import os
import json

try:
    with open('db/aihr.db') as f:
        pass
    import sqlite3
    conn = sqlite3.connect('db/aihr.db')
    cur = conn.cursor()
    cur.execute("SELECT location, raw_fields FROM candidates LIMIT 20")
    for row in cur.fetchall():
        rf = json.loads(row[1]) if row[1] else {}
        print(f"Loc: {row[0]}")
        print(f"Func Exp: {rf.get('functional_experience')}")
        print(f"Func: {rf.get('functions')}")
        print("---")
except Exception as e:
    import psycopg2
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
    cur.execute("SELECT location, raw_fields FROM candidates LIMIT 20")
    for row in cur.fetchall():
        rf = row[1]
        if isinstance(rf, str): rf = json.loads(rf)
        print(f"Loc: {row[0]}")
        print(f"Func Exp: {rf.get('functional_experience', 'N/A')}")
        print(f"Func: {rf.get('functions', 'N/A')}")
        print("---")

from backend.db.connection import get_db_connection
conn = get_db_connection()
cur = conn.cursor()
cur.execute("""
SELECT name, roles->>0 as role
FROM candidates
WHERE raw_fields::text ILIKE '%product manager%' AND raw_fields::text ILIKE '%fintech%'
LIMIT 5;
""")
print(cur.fetchall())

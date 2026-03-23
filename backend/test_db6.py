import sys
import os

# Add backend directory to module path so imports work
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

from backend.db.connection import get_db_connection

def main():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT location, raw_fields->>'functional_experience', raw_fields->>'functions', raw_fields->>'department', raw_fields FROM candidates LIMIT 20")
        for row in cur.fetchall():
            print(f"Loc: {row[0]}")
            print(f"Func Exp: {row[1]}")
            print(f"Functions: {row[2]}")
            print(f"Dept: {row[3]}")
            print("---")
    except Exception as e:
        print("Error:", e)
        
if __name__ == "__main__":
    main()

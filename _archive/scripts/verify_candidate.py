import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def verify_candidate():
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
        
        print(f"--- QUERYING FOR NETHRANAND P S ---")
        
        # Using ILIKE for flexibility
        query = """
            SELECT id, name, email, headline, location, total_experience_years, created_at 
            FROM candidates 
            WHERE name ILIKE '%Nethranand%'
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        if not rows:
            print("No matching candidate found.")
        else:
            for row in rows:
                print(f"ID: {row[0]}")
                print(f"Name: {row[1]}")
                print(f"Email: {row[2]}")
                print(f"Headline: {row[3]}")
                print(f"Location: {row[4]}")
                print(f"Experience: {row[5]} years")
                print(f"Created At: {row[6]}")
                print("-" * 30)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    verify_candidate()

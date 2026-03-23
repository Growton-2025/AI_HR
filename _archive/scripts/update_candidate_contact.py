import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def update_contact():
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
        
        # Update Nethranand P S's contact details
        cur.execute("""
            UPDATE candidates
            SET email = %s,
                mobile_phone = %s,
                first_name = 'Nethranand',
                last_name = 'P S',
                updated_at = NOW()
            WHERE id = 2519
            RETURNING id, name, email, mobile_phone
        """, ('nethranandps2001@gmail.com', '+918618884276'))
        
        result = cur.fetchone()
        if result:
            print(f"✅ Updated Candidate ID: {result[0]}")
            print(f"   Name: {result[1]}")
            print(f"   Email: {result[2]}")
            print(f"   Phone: {result[3]}")
            conn.commit()
        else:
            print("❌ Candidate not found")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    update_contact()

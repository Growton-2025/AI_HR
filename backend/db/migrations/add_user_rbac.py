
import os
import psycopg2
from backend.db.connection import get_db_connection, return_db_connection

def migrate():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return

    try:
        with conn.cursor() as cur:
            print("Adding role and permissions columns to users table...")
            cur.execute("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'recruiter',
                ADD COLUMN IF NOT EXISTS permissions JSONB DEFAULT '{}';
            """)
            
            # Make the first user an admin if roles weren't set
            cur.execute("UPDATE users SET role = 'admin' WHERE id = (SELECT min(id) FROM users) AND (role IS NULL OR role = 'recruiter');")
            
            conn.commit()
            print("Migration successful")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    migrate()

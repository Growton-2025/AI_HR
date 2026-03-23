
import psycopg2
from backend.db.connection import get_db_connection, return_db_connection

def migrate():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return

    try:
        with conn.cursor() as cur:
            print("Adding status column to candidates table...")
            cur.execute("""
                ALTER TABLE candidates 
                ADD COLUMN IF NOT EXISTS status VARCHAR(100) DEFAULT 'To be started';
            """)
            conn.commit()
            print("Migration successful: status column added to candidates table.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    migrate()

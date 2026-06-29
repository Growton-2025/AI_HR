import os
import sys
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.db.connection import get_db_connection

def migrate():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        return

    try:
        with conn.cursor() as cur:
            # Check if auto_create_call_list exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='recruitment_roles' AND column_name='auto_create_call_list';
            """)
            if not cur.fetchone():
                print("Adding auto_create_call_list column to recruitment_roles...")
                cur.execute("ALTER TABLE recruitment_roles ADD COLUMN auto_create_call_list BOOLEAN DEFAULT FALSE;")
            else:
                print("auto_create_call_list column already exists.")

            # Check if linked_call_list_id exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='recruitment_roles' AND column_name='linked_call_list_id';
            """)
            if not cur.fetchone():
                print("Adding linked_call_list_id column to recruitment_roles...")
                cur.execute("ALTER TABLE recruitment_roles ADD COLUMN linked_call_list_id INTEGER REFERENCES call_lists(id) ON DELETE SET NULL;")
            else:
                print("linked_call_list_id column already exists.")

            conn.commit()
            print("Migration successful.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()

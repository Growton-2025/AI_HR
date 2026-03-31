
import os
import sys

# Add parent directory to path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.db.connection import get_db_connection, return_db_connection

def migrate():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return

    try:
        cur = conn.cursor()
        
        # 1. Add unique constraint on (candidate_id, list_id) in calls
        # This is required for ON CONFLICT to work effectively and for fast uniqueness checks
        print("Adding unique constraint on calls(candidate_id, list_id)...")
        cur.execute("""
            ALTER TABLE calls 
            ADD CONSTRAINT unique_candidate_list UNIQUE (candidate_id, list_id);
        """)

        # 2. Add performance indexes
        print("Adding performance indexes...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_list_id ON calls(list_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_candidate_id ON calls(candidate_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_status_due_date ON calls(status, due_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_call_lists_created_by ON call_lists(created_by);")
        
        conn.commit()
        cur.close()
        print("Migration successful: Performance indexes and constraints added")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__": migrate()


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
        
        # 1. Create call_lists table
        print("Creating call_lists table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS call_lists (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(255)
            );
        """)
        
        # 2. Create calls table
        print("Creating calls table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id SERIAL PRIMARY KEY,
                candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                list_id INTEGER NOT NULL REFERENCES call_lists(id) ON DELETE CASCADE,
                status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed'
                outcome VARCHAR(100), -- 'Left Voicemail', 'Connected - Interested', etc.
                notes TEXT,
                duration INTEGER DEFAULT 0, -- in seconds
                due_date DATE DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)
        
        # 3. Add updated_at trigger for calls
        cur.execute("""
            DROP TRIGGER IF EXISTS sync_updated_at_calls ON calls;
            CREATE TRIGGER sync_updated_at_calls
            BEFORE UPDATE ON calls
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)
        
        conn.commit()
        cur.close()
        print("Migration successful")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__": migrate()

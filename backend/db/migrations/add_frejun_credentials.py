
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
        
        print("Creating frejun_oauth_credentials table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frejun_oauth_credentials (
                id SERIAL PRIMARY KEY,
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                frejun_user_email VARCHAR(255),
                token_type VARCHAR(50) DEFAULT 'Bearer',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Add updated_at trigger
        cur.execute("""
            DROP TRIGGER IF EXISTS sync_updated_at_frejun_creds ON frejun_oauth_credentials;
            CREATE TRIGGER sync_updated_at_frejun_creds
            BEFORE UPDATE ON frejun_oauth_credentials
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)
        
        conn.commit()
        cur.close()
        print("Migration successful: frejun_oauth_credentials table created.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    migrate()

from backend.db.connection import get_db_connection_context

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            # Get user ID
            cur.execute("SELECT id FROM users WHERE name ILIKE '%ashwin%';")
            user = cur.fetchone()
            if not user:
                print("User ashwin not found")
                return
            user_id = user[0]
            print(f"Deleting data for user_id: {user_id}")
            
            # Delete candidates (cascades to roles, education, etc.)
            cur.execute("DELETE FROM candidates WHERE owner_user_id = %s;", (user_id,))
            deleted_candidates = cur.rowcount
            print(f"Deleted {deleted_candidates} candidates.")
            
            # Delete candidate uploads
            cur.execute("DELETE FROM candidate_uploads WHERE owner_user_id = %s;", (user_id,))
            deleted_uploads = cur.rowcount
            print(f"Deleted {deleted_uploads} candidate uploads.")
            
            conn.commit()
            print("Successfully deleted all ashwin data.")

if __name__ == "__main__":
    main()

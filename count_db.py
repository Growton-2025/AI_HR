from backend.db.connection import get_db_connection_context
def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM candidates WHERE owner_user_id = 29;")
            cnt = cur.fetchone()[0]
            print(f"Total uploaded for Ashwin: {cnt}")
if __name__ == "__main__":
    main()

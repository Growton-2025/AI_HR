from backend.db.connection import get_db_connection_context

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candidates WHERE pool_source = 'catalog_from_upload';")
            deleted = cur.rowcount
            conn.commit()
            print(f"Successfully deleted {deleted} master candidates from the catalog upload.")

if __name__ == "__main__":
    main()

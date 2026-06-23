from backend.db.connection import get_db_connection_context

def main():
    tables_to_check = [
        'candidates', 'companies', 'roles', 'education', 
        'candidate_uploads', 'ai_column_cells', 'ai_column_runs',
        'candidate_outreach', 'calls'
    ]
    
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            for table in tables_to_check:
                print(f"### Table: `{table}`")
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                """, (table,))
                columns = cur.fetchall()
                for col in columns:
                    print(f"- `{col[0]}` ({col[1]})")
                print("\n")

if __name__ == "__main__":
    main()

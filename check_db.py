from backend.db.connection import get_db_connection_context

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN total_experience_years = 0 THEN 1 ELSE 0 END) as zero_exp,
                    SUM(CASE WHEN total_experience_years IS NULL THEN 1 ELSE 0 END) as null_exp,
                    SUM(CASE WHEN avg_years_in_company = 0 THEN 1 ELSE 0 END) as zero_avg_tenure,
                    SUM(CASE WHEN avg_years_in_company IS NULL THEN 1 ELSE 0 END) as null_avg_tenure
                FROM candidates 
                WHERE owner_user_id = 29;
            """)
            row = cur.fetchone()
            print(f"Total Candidates: {row[0]}")
            print(f"0 Experience: {row[1]}")
            print(f"NULL Experience: {row[2]}")
            print(f"0 Avg Tenure: {row[3]}")
            print(f"NULL Avg Tenure: {row[4]}")
            
            # Show a few examples of the 0 experience ones
            cur.execute("""
                SELECT id, name, total_experience_years, avg_years_in_company, created_at 
                FROM candidates 
                WHERE owner_user_id = 29 AND total_experience_years = 0
                LIMIT 5;
            """)
            examples = cur.fetchall()
            print("\nExamples of 0 Experience Candidates:")
            for e in examples:
                print(f"ID {e[0]} | Name: {e[1]} | Exp: {e[2]} | Avg Tenure: {e[3]} | Created At: {e[4]}")

if __name__ == "__main__":
    main()

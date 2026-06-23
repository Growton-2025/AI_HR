from backend.db.connection import get_db_connection_context

def main():
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    name, 
                    location,
                    city,
                    skills,
                    total_experience_years, 
                    avg_years_in_company,
                    has_gap_years,
                    has_education_gaps,
                    has_industry_gaps,
                    max_people_managed,
                    years_team_management
                FROM candidates WHERE id = 11079;
            """)
            row = cur.fetchone()
            if row:
                print(f"Name: {row[0]}")
                print(f"Location: {row[1]}")
                print(f"City: {row[2]}")
                print(f"Skills: {row[3][:100]}..." if row[3] else "Skills: None")
                print(f"Total Exp: {row[4]} years")
                print(f"Avg Tenure: {row[5]} years")
                print(f"Has Gap Years: {row[6]}")
                print(f"Has Education Gaps: {row[7]}")
                print(f"Has Industry Gaps: {row[8]}")
                print(f"Max People Managed: {row[9]}")
                print(f"Years Team Management: {row[10]}")

if __name__ == "__main__":
    main()

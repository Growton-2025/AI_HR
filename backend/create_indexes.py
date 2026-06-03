import os
import psycopg2
from dotenv import load_dotenv

load_dotenv('/Users/nethranand/Downloads/AI_HR/backend/.env')

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-restore-may26.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    sslmode="require"
)
conn.autocommit = True

indexes = [
    "CREATE INDEX IF NOT EXISTS idx_roles_candidate ON roles(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_roles_company ON roles(company_id);",
    "CREATE INDEX IF NOT EXISTS idx_education_candidate ON education(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_company_years_candidate ON company_years(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_experience_gaps_candidate ON experience_gaps(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_education_gaps_candidate ON education_gaps(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_industry_gaps_candidate ON industry_gaps(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_functional_experiences_candidate ON functional_experiences(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_func_exp_roles_fkey ON functional_experience_roles(functional_experience_id);",
    "CREATE INDEX IF NOT EXISTS idx_industry_experiences_candidate ON industry_experiences(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_ind_exp_roles_fkey ON industry_experience_roles(industry_experience_id);",
    "CREATE INDEX IF NOT EXISTS idx_segment_experiences_candidate ON segment_experiences(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_seg_exp_roles_fkey ON segment_experience_roles(segment_experience_id);",
    "CREATE INDEX IF NOT EXISTS idx_geography_experiences_candidate ON geography_experiences(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_geo_exp_regions_fkey ON geography_experience_regions(geography_experience_id);",
    "CREATE INDEX IF NOT EXISTS idx_titles_held_candidate ON titles_held(candidate_id);",
    "CREATE INDEX IF NOT EXISTS idx_recruitment_roles_user ON recruitment_roles(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_recruitment_role_candidates_candidate ON recruitment_role_candidates(candidate_id);"
]

try:
    with conn.cursor() as cur:
        for idx in indexes:
            print(f"Executing: {idx}")
            cur.execute(idx)
    print("All indexes created successfully.")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()

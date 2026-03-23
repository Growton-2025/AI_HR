import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def add_role():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require"
        )
        cur = conn.cursor()
        
        # 1. Check/Insert Company
        company_name = "Rootent Technologies"
        cur.execute("SELECT id FROM companies WHERE name = %s", (company_name,))
        row = cur.fetchone()
        
        if row:
            company_id = row[0]
            print(f"Company '{company_name}' exists with ID: {company_id}")
        else:
            print(f"Adding company '{company_name}'...")
            cur.execute("""
                INSERT INTO companies (name, funding_stage, business_model, product_service, culture_type, customer_presence, created_at, updated_at)
                VALUES (%s, 'Bootstrapped', 'B2B', 'SaaS Services', 'Startup', ARRAY['India'], NOW(), NOW())
                RETURNING id
            """, (company_name,))
            company_id = cur.fetchone()[0]
            print(f"Added company with ID: {company_id}")
            
        # 2. Add Role for Candidate 2519
        candidate_id = 2519
        title = "AIML Engineer"
        # Using extraction from profile text
        details = "Developing AI solutions, leveraging AWS services (SageMaker, Lambda, EC2), building RAG pipelines with Langchain and Llamaindex."
        duration_years = 0.2
        
        cur.execute("""
            INSERT INTO roles (candidate_id, company_id, title, details, duration_years)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (candidate_id, company_id, title, details, duration_years))
        
        role_id = cur.fetchone()[0]
        conn.commit()
        print(f"Added Role '{title}' with ID: {role_id} for Candidate {candidate_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        if cur: cur.close()
        if conn: conn.close()

if __name__ == "__main__":
    add_role()

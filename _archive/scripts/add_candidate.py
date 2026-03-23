import os
import psycopg2
from dotenv import load_dotenv
import json

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def add_candidate():
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
        
        candidate_data = {
            "name": "Nethranand P S",
            "first_name": "Nethranand",
            "last_name": "P S",
            "email": "nethranand.ps@example.com", # Placeholder
            "headline": "AIML Engineer | AI Engineer",
            "location": "Bengaluru, Karnataka, India",
            "city": "Bengaluru",
            "about": "Experienced AI/ML Engineer with a background in Generative AI development and data-driven SaaS solutions. Skilled in AWS, Langchain, Llamaindex, and predictive modeling.",
            "skills": "Machine Learning, Python, FastAPI, MongoDB, AWS (SageMaker, Lambda, EC2), Langchain, Llamaindex, OpenCV, PyTorch",
            "total_experience_years": 1.75,
            "linkedin": "https://www.linkedin.com/in/nethranand-p-s-placeholder"
        }
        
        insert_query = """
            INSERT INTO candidates (
                name, first_name, last_name, email, headline, 
                location, city, about, skills, total_experience_years, linkedin,
                created_at, updated_at
            ) VALUES (
                %(name)s, %(first_name)s, %(last_name)s, %(email)s, %(headline)s,
                %(location)s, %(city)s, %(about)s, %(skills)s, %(total_experience_years)s, %(linkedin)s,
                NOW(), NOW()
            ) RETURNING id;
        """
        
        cur.execute(insert_query, candidate_data)
        new_id = cur.fetchone()[0]
        conn.commit()
        
        print(f"Candidate added successfully with ID: {new_id}")
        
        # Verify
        cur.execute("SELECT * FROM candidates WHERE id = %s", (new_id,))
        row = cur.fetchone()
        print("Verified Record:", row)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_candidate()

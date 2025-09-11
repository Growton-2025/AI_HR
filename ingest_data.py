import os
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURE YOUR DETAILS HERE ---
load_dotenv()
DB_NAME = "candidate_search"
DB_USER = "postgres"       # e.g., 'postgres'
DB_PASSWORD = "postgres"   # Your PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"
JSON_FILE_PATH = "enriched_candidate_profiles.json"
EMBEDDING_MODEL = "text-embedding-3-small"



def ingest_data():
    logging.info("Starting data ingestion...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    register_vector(conn)
    logging.info("Successfully connected to the database.")

    with open(JSON_FILE_PATH, 'r') as f:
        profiles_data = json.load(f)
    logging.info(f"Loaded {len(profiles_data)} profiles.")

    for i, profile in enumerate(profiles_data):
        try:
            # Create a rich text document for embedding
            roles_summary = " ".join([f"{r.get('title', '')} {r.get('details', '')}" for r in profile.get('roles', [])])
            skills = profile.get('raw_fields', {}).get('Skills', '')
            document_text = (f"Name: {profile.get('name', '')}. Headline: {profile.get('headline', '')}. "
                             f"About: {profile.get('about', '')}. Experience: {roles_summary}. Skills: {skills}.")
            
            embedding_vector = embeddings.embed_query(document_text)

            # Insert into 'candidates' table
            cur.execute("""
                INSERT INTO candidates (
                    name, linkedin, location, city, headline, about, total_experience_years,
                    avg_years_in_company, company_years, has_gap_years, gaps, has_education_gaps,
                    education_gaps, has_industry_gaps, industry_gaps, functional_experience,
                    team_management, industry_experience, segment_experience, geography_experience,
                    raw_fields, titles_held, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                profile.get('name'), profile.get('linkedin'), profile.get('location'), profile.get('city'),
                profile.get('headline'), profile.get('about'), profile.get('total_experience_years'),
                profile.get('avg_years_in_company'), json.dumps(profile.get('company_years')),
                profile.get('has_gap_years'), json.dumps(profile.get('gaps')), profile.get('has_education_gaps'),
                json.dumps(profile.get('education_gaps')), profile.get('has_industry_gaps'),
                json.dumps(profile.get('industry_gaps')), json.dumps(profile.get('functional_experience')),
                json.dumps(profile.get('team_management')), json.dumps(profile.get('industry_experience')),
                json.dumps(profile.get('segment_experience')), json.dumps(profile.get('geography_experience')),
                json.dumps(profile.get('raw_fields')), json.dumps(profile.get('titles_held')), embedding_vector
            ))
            candidate_id = cur.fetchone()[0]

            # Batch insert roles
            roles = profile.get('roles', [])
            if roles:
                roles_to_insert = [(candidate_id, r.get('company'), r.get('title'), r.get('details'),
                                    r.get('duration_years'), json.dumps(r.get('company_details'))) for r in roles]
                execute_values(cur, """
                    INSERT INTO roles (candidate_id, company, title, details, duration_years, company_details)
                    VALUES %s
                """, roles_to_insert)

            # Batch insert education
            education_history = profile.get('education', [])
            if education_history:
                edu_to_insert = [(candidate_id, e.get('college'), e.get('degree'), e.get('start'), e.get('end'))
                                 for e in education_history]
                execute_values(cur, """
                    INSERT INTO education (candidate_id, college, degree, start, "end")
                    VALUES %s
                """, edu_to_insert)

            conn.commit()
            logging.info(f"Successfully inserted profile {i+1}/{len(profiles_data)}: {profile.get('name')}")

        except Exception as e:
            logging.error(f"Error on profile {profile.get('name')}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    logging.info("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
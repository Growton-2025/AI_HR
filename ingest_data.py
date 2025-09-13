import os
import json
from datetime import datetime
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
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
JSON_FILE_PATH = "enriched_candidate_profiles.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def parse_date(date_str):
    """Parse date strings from JSON, handling various formats."""
    if not date_str or date_str.strip() == "" or date_str.lower() in ["present", "current"]:
        return None  # Treat 'Present' and 'current' as NULL
    try:
        # Handle YYYY-MM-DD HH:MM:SS
        if ' ' in date_str:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()
        # Handle YYYY-MM-DD
        elif len(date_str.split('-')) == 3:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        # Handle YYYY-MM
        elif len(date_str.split('-')) == 2:
            return datetime.strptime(date_str + "-01", "%Y-%m-%d").date()
        # Handle YYYY
        elif len(date_str) == 4 and date_str.isdigit():
            return datetime.strptime(date_str + "-01-01", "%Y-%m-%d").date()
        else:
            logging.warning(f"Invalid date format: {date_str}")
            return None
    except ValueError:
        logging.warning(f"Invalid date format: {date_str}")
        return None

def create_schema(cur):
    """Create the normalized database schema."""
    logging.info("Creating database schema...")
    schema_statements = [
        """
        CREATE EXTENSION IF NOT EXISTS vector;
        """,
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            linkedin VARCHAR(255),
            location TEXT,
            city VARCHAR(100),
            headline TEXT,
            about TEXT,
            skills TEXT,
            licenses_and_certifications TEXT,
            total_experience_years NUMERIC,
            avg_years_in_company NUMERIC,
            has_gap_years BOOLEAN,
            has_education_gaps BOOLEAN,
            has_industry_gaps BOOLEAN,
            functional_experience_score INTEGER,
            functional_experience_rationale TEXT,
            industry_experience_score INTEGER,
            industry_experience_rationale TEXT,
            segment_experience_score INTEGER,
            segment_experience_rationale TEXT,
            geography_experience_score INTEGER,
            geography_experience_rationale TEXT,
            team_management_score INTEGER,
            team_management_rationale TEXT,
            max_people_managed INTEGER,
            years_team_management NUMERIC,
            raw_fields JSONB,
            embedding VECTOR(1536)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            company VARCHAR(255),
            title VARCHAR(255),
            details TEXT,
            duration_years NUMERIC,
            company_details_product_service TEXT,
            company_details_customer_segment TEXT[],
            company_details_customer_presence TEXT[],
            company_details_funding_stage VARCHAR(255),  -- Increased to 255 to handle longer values
            company_details_revenue TEXT,
            company_details_culture_type VARCHAR(255),  -- Increased to 255
            company_details_headquarters VARCHAR(255),
            company_details_business_model VARCHAR(255)  -- Increased to 255
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS education (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            college VARCHAR(255),
            degree VARCHAR(255),
            start_date DATE,
            end_date DATE,
            details TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS company_years (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            company VARCHAR(255),
            years NUMERIC
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS experience_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS education_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS industry_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS functional_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS functional_experience_roles (
            id SERIAL PRIMARY KEY,
            functional_experience_id INTEGER REFERENCES functional_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            activity_type VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS industry_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS industry_experience_roles (
            id SERIAL PRIMARY KEY,
            industry_experience_id INTEGER REFERENCES industry_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            industry VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS segment_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS segment_experience_roles (
            id SERIAL PRIMARY KEY,
            segment_experience_id INTEGER REFERENCES segment_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            segment VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS geography_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS geography_experience_regions (
            id SERIAL PRIMARY KEY,
            geography_experience_id INTEGER REFERENCES geography_experiences(id) ON DELETE CASCADE,
            region VARCHAR(100)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS titles_held (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            title VARCHAR(255),
            company VARCHAR(255),
            start_date DATE,
            end_date DATE
        );
        """
    ]
    try:
        for statement in schema_statements:
            cur.execute(statement)
        logging.info("Schema created successfully.")
    except Exception as e:
        logging.error(f"Error creating schema: {e}")
        raise

def ingest_data():
    logging.info("Starting JSON data ingestion...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        register_vector(conn)
        logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

    # Create schema
    create_schema(cur)
    conn.commit()

    # Load JSON data
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            profiles_data = json.load(f)
        logging.info(f"Loaded {len(profiles_data)} profiles from {JSON_FILE_PATH}.")
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        cur.close()
        conn.close()
        raise

    for i, profile in enumerate(profiles_data):
        try:
            # Create embedding text
            roles_summary = " ".join([f"{r.get('title', '')} {r.get('details', '')}" for r in profile.get('roles', [])])
            skills = profile.get('raw_fields', {}).get('Skills', '')
            document_text = (
                f"Name: {profile.get('name', '')}. Headline: {profile.get('headline', '')}. "
                f"About: {profile.get('about', '')}. Experience: {roles_summary}. Skills: {skills}."
            )
            embedding_vector = embeddings.embed_query(document_text)

            # Extract raw_fields for flattened columns
            raw_fields = profile.get('raw_fields', {})

            # Prepare parameters for candidates table
            candidate_params = (
                profile.get('name'),
                raw_fields.get('First Name'),
                raw_fields.get('Last Name'),
                profile.get('linkedin'),
                profile.get('location'),
                profile.get('city'),
                profile.get('headline'),
                profile.get('about'),
                skills,
                raw_fields.get('Licenses and certifications'),
                profile.get('total_experience_years'),
                profile.get('avg_years_in_company'),
                profile.get('has_gap_years'),
                profile.get('has_education_gaps'),
                profile.get('has_industry_gaps'),
                profile.get('functional_experience', {}).get('functional_experience_score'),
                profile.get('functional_experience', {}).get('rationale'),
                profile.get('industry_experience', {}).get('industry_experience_score'),
                profile.get('industry_experience', {}).get('rationale'),
                profile.get('segment_experience', {}).get('segment_experience_score'),
                profile.get('segment_experience', {}).get('rationale'),
                profile.get('geography_experience', {}).get('geography_experience_score'),
                profile.get('geography_experience', {}).get('rationale'),
                profile.get('team_management', {}).get('team_management_score'),
                profile.get('team_management', {}).get('rationale'),
                profile.get('team_management', {}).get('max_people_managed'),
                profile.get('team_management', {}).get('years_team_management'),
                json.dumps(raw_fields) if raw_fields else None,
                embedding_vector
            )

            # Dynamically generate placeholders
            placeholders = ", ".join(["%s"] * len(candidate_params))

            # Insert into candidates table
            cur.execute(f"""
                INSERT INTO candidates (
                    name, first_name, last_name, linkedin, location, city, headline, about, skills,
                    licenses_and_certifications, total_experience_years, avg_years_in_company,
                    has_gap_years, has_education_gaps, has_industry_gaps,
                    functional_experience_score, functional_experience_rationale,
                    industry_experience_score, industry_experience_rationale,
                    segment_experience_score, segment_experience_rationale,
                    geography_experience_score, geography_experience_rationale,
                    team_management_score, team_management_rationale,
                    max_people_managed, years_team_management, raw_fields, embedding
                ) VALUES ({placeholders})
                RETURNING id;
            """, candidate_params)
            candidate_id = cur.fetchone()[0]

            # Insert into company_years
            company_years_data = profile.get('company_years', {})
            if company_years_data:
                company_years_to_insert = [(candidate_id, company, years) for company, years in company_years_data.items()]
                execute_values(cur, """
                    INSERT INTO company_years (candidate_id, company, years)
                    VALUES %s
                """, company_years_to_insert)

            # Insert into experience_gaps
            gaps = profile.get('gaps', [])
            if gaps:
                gaps_to_insert = [
                    (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason'))
                    for g in gaps
                ]
                execute_values(cur, """
                    INSERT INTO experience_gaps (candidate_id, from_date, to_date, duration_months, reason)
                    VALUES %s
                """, gaps_to_insert)

            # Insert into education_gaps
            education_gaps = profile.get('education_gaps', [])
            if education_gaps:
                edu_gaps_to_insert = [
                    (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason'))
                    for g in education_gaps
                ]
                execute_values(cur, """
                    INSERT INTO education_gaps (candidate_id, from_date, to_date, duration_months, reason)
                    VALUES %s
                """, edu_gaps_to_insert)

            # Insert into industry_gaps
            industry_gaps = profile.get('industry_gaps', [])
            if industry_gaps:
                ind_gaps_to_insert = [
                    (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason'))
                    for g in industry_gaps
                ]
                execute_values(cur, """
                    INSERT INTO industry_gaps (candidate_id, from_date, to_date, duration_months, reason)
                    VALUES %s
                """, ind_gaps_to_insert)

            # Insert into functional_experiences and its roles
            func_exp = profile.get('functional_experience', {})
            if func_exp:
                cur.execute("""
                    INSERT INTO functional_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id;
                """, (
                    candidate_id,
                    func_exp.get('functional_experience_score'),
                    func_exp.get('rationale')
                ))
                func_exp_id = cur.fetchone()[0]

                func_roles = func_exp.get('roles', [])
                if func_roles:
                    func_roles_to_insert = [
                        (func_exp_id, r.get('company'), r.get('activity_type'), r.get('reason'), r.get('duration_years'))
                        for r in func_roles
                    ]
                    execute_values(cur, """
                        INSERT INTO functional_experience_roles (functional_experience_id, company, activity_type, reason, duration_years)
                        VALUES %s
                    """, func_roles_to_insert)

            # Insert into industry_experiences and its roles
            ind_exp = profile.get('industry_experience', {})
            if ind_exp:
                cur.execute("""
                    INSERT INTO industry_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id;
                """, (
                    candidate_id,
                    ind_exp.get('industry_experience_score'),
                    ind_exp.get('rationale')
                ))
                ind_exp_id = cur.fetchone()[0]

                ind_roles = ind_exp.get('roles', [])
                if ind_roles:
                    ind_roles_to_insert = [
                        (ind_exp_id, r.get('company'), r.get('industry'), r.get('reason'), r.get('duration_years'))
                        for r in ind_roles
                    ]
                    execute_values(cur, """
                        INSERT INTO industry_experience_roles (industry_experience_id, company, industry, reason, duration_years)
                        VALUES %s
                    """, ind_roles_to_insert)

            # Insert into segment_experiences and its roles
            seg_exp = profile.get('segment_experience', {})
            if seg_exp:
                cur.execute("""
                    INSERT INTO segment_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id;
                """, (
                    candidate_id,
                    seg_exp.get('segment_experience_score'),
                    seg_exp.get('rationale')
                ))
                seg_exp_id = cur.fetchone()[0]

                seg_roles = seg_exp.get('roles', [])
                if seg_roles:
                    seg_roles_to_insert = [
                        (seg_exp_id, r.get('company'), r.get('segment'), r.get('reason'), r.get('duration_years'))
                        for r in seg_roles
                    ]
                    execute_values(cur, """
                        INSERT INTO segment_experience_roles (segment_experience_id, company, segment, reason, duration_years)
                        VALUES %s
                    """, seg_roles_to_insert)

            # Insert into geography_experiences and its regions
            geo_exp = profile.get('geography_experience', {})
            if geo_exp:
                cur.execute("""
                    INSERT INTO geography_experiences (candidate_id, score, rationale)
                    VALUES (%s, %s, %s) RETURNING id;
                """, (
                    candidate_id,
                    geo_exp.get('geography_experience_score'),
                    geo_exp.get('rationale')
                ))
                geo_exp_id = cur.fetchone()[0]

                regions = geo_exp.get('regions', [])
                if regions:
                    regions_to_insert = [(geo_exp_id, region) for region in regions]
                    execute_values(cur, """
                        INSERT INTO geography_experience_regions (geography_experience_id, region)
                        VALUES %s
                    """, regions_to_insert)

            # Insert into roles with flattened company_details, handling arrays and truncation
            roles = profile.get('roles', [])
            if roles:
                roles_to_insert = []
                for r in roles:
                    company_details = r.get('company_details', {})
                    customer_segment = company_details.get('customer_segment', [])
                    if isinstance(customer_segment, str):
                        customer_segment = [customer_segment.strip()]

                    customer_presence = company_details.get('customer_presence', [])
                    if isinstance(customer_presence, str):
                        # Split if contains commas or spaces
                        presence_list = [p.strip() for p in customer_presence.split(',') if p.strip()]
                        if not presence_list:
                            presence_list = [customer_presence]
                        customer_presence = presence_list

                    funding_stage = company_details.get('funding_stage', '')[:255]
                    culture_type = company_details.get('culture_type', '')[:255]
                    business_model = company_details.get('business_model', '')[:255]

                    roles_to_insert.append(
                        (
                            candidate_id, r.get('company'), r.get('title'), r.get('details'), r.get('duration_years'),
                            company_details.get('product_service'),
                            customer_segment,
                            customer_presence,
                            funding_stage,
                            company_details.get('revenue'),
                            culture_type,
                            company_details.get('headquarters'),
                            business_model
                        )
                    )
                execute_values(cur, """
                    INSERT INTO roles (
                        candidate_id, company, title, details, duration_years,
                        company_details_product_service, company_details_customer_segment,
                        company_details_customer_presence, company_details_funding_stage,
                        company_details_revenue, company_details_culture_type,
                        company_details_headquarters, company_details_business_model
                    ) VALUES %s
                """, roles_to_insert)

            # Insert into education with parsed dates
            education_history = profile.get('education', [])
            if education_history:
                edu_to_insert = [
                    (
                        candidate_id, e.get('college'), e.get('degree'),
                        parse_date(e.get('start')), parse_date(e.get('end')), e.get('details')
                    ) for e in education_history
                ]
                execute_values(cur, """
                    INSERT INTO education (candidate_id, college, degree, start_date, end_date, details)
                    VALUES %s
                """, edu_to_insert)

            # Insert into titles_held with parsed dates
            titles = profile.get('titles_held', [])
            if titles:
                titles_to_insert = [
                    (
                        candidate_id, t.get('title'), t.get('company'),
                        parse_date(t.get('start')), parse_date(t.get('end'))
                    ) for t in titles
                ]
                execute_values(cur, """
                    INSERT INTO titles_held (candidate_id, title, company, start_date, end_date)
                    VALUES %s
                """, titles_to_insert)

            conn.commit()
            logging.info(f"Successfully inserted profile {i+1}/{len(profiles_data)}: {profile.get('name')}")

        except Exception as e:
            logging.error(f"Error on profile {profile.get('name')}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    logging.info("JSON data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
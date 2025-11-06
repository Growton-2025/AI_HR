import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

# Load dataset
try:
    with open("/home/nethranand-ps/AI_HR/individual/enriched_candidate_profiles.json", "r") as f:
        profiles = json.load(f)
except FileNotFoundError:
    print("Error: enriched_candidate_profiles.json not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in enriched_candidate_profiles.json")
    exit(1)

# Function to calculate years
def calculate_years(start_date: str, end_date: str, current_date: str = "2025-08-06") -> float:
    try:
        date_formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"]
        start, end = None, None
        for fmt in date_formats:
            try:
                start = datetime.strptime(start_date, fmt)
                end = datetime.strptime(end_date, fmt) if end_date and end_date.lower() != "present" else datetime.strptime(current_date, "%Y-%m-%d")
                break
            except ValueError:
                continue
        if not start or not end:
            raise ValueError("No valid date format")
        return round((end - start).days / 365.25, 2)
    except Exception as e:
        print(f"Date parsing error: {e}, start_date={start_date}, end_date={end_date}")
        return 0.0

# Function to flatten dictionaries
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list) and k not in ['education', 'titles_held', 'embedding', 'roles', 'segments', 'regions', 'gaps', 'education_gaps', 'industry_gaps']:
            items.append((new_key, ', '.join([str(item) for item in v if item])))
        else:
            items.append((new_key, str(v)))
    return dict(items)

# Function to estimate token count
def estimate_tokens(text):
    return len(text) // 4

# Function to truncate text
def truncate_text(text, max_tokens=2000):
    max_chars = max_tokens * 4
    if isinstance(text, str) and len(text) > max_chars:
        return text[:max_chars] + "... [Truncated]"
    return text

# Create documents
docs = []
for profile in profiles:
    name = profile.get("name", "Unknown")
    if name == "Unknown":
        print(f"Warning: Profile with missing name found: {profile.get('linkedin', 'No LinkedIn')}")
        continue

    flattened_profile = flatten_dict(profile)

    # Process functional_experience roles
    role_descriptions = []
    total_experience_by_type = {}
    roles = profile.get('functional_experience', {}).get('roles', [])
    for role in roles:
        if isinstance(role, dict):
            title = role.get('title', 'Unknown Title')
            company = role.get('company', 'Unknown Company')
            start_date = role.get('start_date', '')
            end_date = role.get('end_date', '')
            activity_type = role.get('activity_type', 'Unknown')
            details = truncate_text(role.get('details', ''), max_tokens=500)
            duration = role.get('duration_years', None)
            if duration is None or not isinstance(duration, (int, float)):
                duration = calculate_years(start_date, end_date)
            if not title.strip() and not company.strip():
                print(f"Warning: Empty role in profile {name}: {role}")
                continue
            role_str = f"Activity Type: {activity_type}, Title: {title}, Company: {company}, Duration: {duration:.2f} years"
            if details:
                role_str += f", Details: {details}"
            role_descriptions.append(role_str)
            total_experience_by_type[activity_type] = total_experience_by_type.get(activity_type, 0.0) + float(duration)
            print(f"Profile {name}, Role: {role_str}")  # Log role details
        else:
            print(f"Warning: Non-dictionary role found in profile {name}: {role}")
            title = str(role).strip()
            if title:
                role_descriptions.append(f"Activity Type: Unknown, Title: {title}, Company: Unknown Company, Duration: Unknown")

    # Summarize total experience by activity_type
    experience_summary = "\n".join([f"Activity Type: {activity_type}, Total Duration: {years:.2f} years" for activity_type, years in total_experience_by_type.items()])
    print(f"Profile {name}, Experience Summary:\n{experience_summary}")  # Log summary
    print(f"Profile {name}, Functional Experience JSON:\n{json.dumps(profile.get('functional_experience', {}), indent=2)}")  # Log full functional_experience

    # Process raw_fields roles
    raw_role_descriptions = []
    for i in range(1, 11):
        company_key = f"Company {i} Name" if i == 1 else f"Company {i} Name"
        title_key = "Title" if i == 1 else f"Title.{i-1}"
        start_key = "Start date" if i == 1 else f"Start date.{i-1}"
        end_key = "End Date" if i == 1 else f"End Date.{i-1}"
        details_key = "Details" if i == 1 else f"Details .{i-1}"
        company = flattened_profile.get(f"raw_fields.{company_key}", '')
        title = flattened_profile.get(f"raw_fields.{title_key}", '')
        start_date = flattened_profile.get(f"raw_fields.{start_key}", '')
        end_date = flattened_profile.get(f"raw_fields.{end_key}", '')
        details = truncate_text(flattened_profile.get(f"raw_fields.{details_key}", ''), max_tokens=500)
        if company and title:
            duration = calculate_years(start_date, end_date)
            role_str = f"Title: {title}, Company: {company}, Duration: {duration:.2f} years"
            if details:
                role_str += f", Details: {details}"
            raw_role_descriptions.append(role_str)

    # Build page_content
    page_content = (
        f"Name: {name}\n"
        f"Functional Experience Summary:\n{experience_summary}\n"
        f"Functional Experience Roles:\n{'; '.join(role_descriptions) if role_descriptions else 'None'}\n"
        f"Location: {truncate_text(profile.get('location', ''), max_tokens=100)}\n"
        f"Headline: {truncate_text(profile.get('headline', ''), max_tokens=200)}\n"
        f"About: {truncate_text(profile.get('about', ''), max_tokens=1000)}\n"
        f"Raw Fields Roles:\n{'; '.join(raw_role_descriptions) if raw_role_descriptions else 'None'}\n"
        f"Regions: {', '.join(profile.get('geography_experience', {}).get('regions', []))}\n"
        f"Details: {truncate_text(', '.join([d for d in profile.get('details', []) if d]), max_tokens=500)}\n"
        f"Skills: {truncate_text(flattened_profile.get('raw_fields.Skills', ''), max_tokens=500)}\n"
        f"Certifications: {truncate_text(flattened_profile.get('raw_fields.Licenses and certifications', ''), max_tokens=500)}\n"
        f"Total Experience Years: {profile.get('total_experience_years', '')}\n"
        f"Average Years in Company: {profile.get('avg_years_in_company', '')}\n"
        f"Company Years: {', '.join([f'{k}: {v} years' for k, v in profile.get('company_years', {}).items()])}\n"
        f"Education: {', '.join([f'{edu.get('college', '')}: {edu.get('degree', '')}' for edu in profile.get('education', [])])}\n"
        f"Titles Held: {', '.join([f'{t.get('title', '')} at {t.get('company', '')}' if isinstance(t, dict) else t for t in profile.get('titles_held', []) if t])}\n"
        f"Full Row: {truncate_text(profile.get('full_row', ''), max_tokens=2000)}\n"
    )

    token_count = estimate_tokens(page_content)
    if token_count > 100000:
        print(f"Warning: Document for {name} is large: ~{token_count} tokens")
        page_content = truncate_text(page_content, max_tokens=50000)

    metadata = {"name": name, "full_row": json.dumps(profile), **flattened_profile}
    docs.append(Document(page_content=page_content, metadata=metadata))

# Initialize embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Batch processing
batch_size = 5
vector_store = None
for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)
    print(f"Processing batch {i//batch_size + 1} with {len(batch)} documents, ~{total_tokens} tokens")
    if total_tokens > 300000:
        print(f"Warning: Batch {i//batch_size + 1} exceeds 300,000 tokens. Truncating...")
        for doc in batch:
            if estimate_tokens(doc.page_content) > 100000:
                doc.page_content = truncate_text(doc.page_content, max_tokens=50000)
        total_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)
        print(f"New batch token count: ~{total_tokens} tokens")
    try:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embedding)
        else:
            vector_store.add_documents(batch)
        print(f"Batch {i//batch_size + 1} processed successfully")
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")

# Save FAISS index
if vector_store:
    vector_store.save_local("faiss_index")
    print("FAISS index saved successfully")
else:
    print("Error: No vector store created")



----------------------------------------ingest_data is below----------------------------------------


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
DB_NAME = "growton_ai"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"
JSON_FILE_PATH = "enriched_candidate_profiles.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def parse_date(date_str):
    """
    Parse date strings from JSON, handling various formats and stripping time components.
    This fix addresses the 'Invalid date format: YYYY-MM-DD 00:00:00' warnings.
    """
    if not date_str or date_str.strip() == "" or date_str.lower() in ["present", "current"]:
        return None  # Treat 'Present' and 'current' as NULL

    # NEW FIX: Strip the time component (like 00:00:00) if a space is present,
    # because PostgreSQL DATE type only needs YYYY-MM-DD.
    if ' ' in date_str:
        date_str = date_str.split(' ')[0]
        
    try:
        # Handle YYYY-MM-DD (This now handles the stripped format from above)
        if len(date_str.split('-')) == 3:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        # Handle YYYY-MM
        elif len(date_str.split('-')) == 2:
            # Append day '01' to create a valid date object
            return datetime.strptime(date_str + "-01", "%Y-%m-%d").date()
        # Handle YYYY
        elif len(date_str) == 4 and date_str.isdigit():
            # Append month/day '01-01' to create a valid date object
            return datetime.strptime(date_str + "-01-01", "%Y-%m-%d").date()
        else:
            logging.warning(f"Invalid date format (unknown format): {date_str}")
            return None
    except ValueError:
        # Catch unexpected errors during parsing of the simplified date_str
        logging.warning(f"Invalid date format (Value Error): {date_str}")
        return None


def drop_all_tables(cur, conn):
    """
    Drops all user-defined tables in the public schema using CASCADE.
    This ensures a clean start and removes any tables from previous, incomplete runs.
    """
    logging.info("Attempting to drop all existing tables in the 'public' schema...")

    # Query for all tables in the 'public' schema
    cur.execute("""
        SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tableowner != 'postgres';
    """)
    tables = [row[0] for row in cur.fetchall()]
    
    if not tables:
        logging.info("No user tables found to drop.")
        return

    # Generate DROP TABLE statements with CASCADE to handle foreign key dependencies
    drop_statements = [f"DROP TABLE IF EXISTS {table} CASCADE;" for table in tables]
    
    try:
        for statement in drop_statements:
            cur.execute(statement)
        conn.commit()
        logging.info(f"Successfully dropped {len(tables)} tables with CASCADE.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error dropping tables: {e}")
        # We raise the error here to stop the script if the database connection/permissions are fundamentally broken
        raise

def create_schema(cur, conn):
    """
    Create the normalized database schema. 
    NOTE: The order is crucial to satisfy Foreign Key dependencies. 
    'candidates' and 'companies' must be created before 'roles' and other referencing tables.
    Each statement is executed and committed individually for maximum robustness.
    """
    # This logging line confirms you are running the new logic!
    logging.info("Starting robust schema creation (17 tables expected)...") 
    schema_statements = [
        
        # ----------------------------------------------------
        # 1. BASE TABLE: candidates (Most tables depend on this)
        # ----------------------------------------------------
        ("candidates", 
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
        """),
        
        # ----------------------------------------------------
        # 2. BASE TABLE: companies (roles table depends on this)
        # ----------------------------------------------------
        ("companies",
        """
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL, 
            funding_stage VARCHAR(255),
            revenue TEXT,
            business_model VARCHAR(255),
            product_service TEXT,
            customer_segment TEXT[],
            customer_presence TEXT[],
            culture_type VARCHAR(255),
            headquarters VARCHAR(255)
        );
        """),

        # ----------------------------------------------------
        # 3. LINKING TABLE: roles (Requires candidates and companies to exist)
        # ----------------------------------------------------
        ("roles",
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE, 
            title VARCHAR(255),
            details TEXT,
            duration_years NUMERIC
        );
        """),
        
        # ----------------------------------------------------
        # 4-17. Other tables (Mostly referencing candidates)
        # ----------------------------------------------------
        ("education",
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
        """),
        ("company_years",
        """
        CREATE TABLE IF NOT EXISTS company_years (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            company VARCHAR(255),
            years NUMERIC
        );
        """),
        ("experience_gaps",
        """
        CREATE TABLE IF NOT EXISTS experience_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("education_gaps",
        """
        CREATE TABLE IF NOT EXISTS education_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("industry_gaps",
        """
        CREATE TABLE IF NOT EXISTS industry_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("functional_experiences",
        """
        CREATE TABLE IF NOT EXISTS functional_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("functional_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS functional_experience_roles (
            id SERIAL PRIMARY KEY,
            functional_experience_id INTEGER REFERENCES functional_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            activity_type VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("industry_experiences",
        """
        CREATE TABLE IF NOT EXISTS industry_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("industry_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS industry_experience_roles (
            id SERIAL PRIMARY KEY,
            industry_experience_id INTEGER REFERENCES industry_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            industry VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("segment_experiences",
        """
        CREATE TABLE IF NOT EXISTS segment_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("segment_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS segment_experience_roles (
            id SERIAL PRIMARY KEY,
            segment_experience_id INTEGER REFERENCES segment_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            segment VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("geography_experiences",
        """
        CREATE TABLE IF NOT EXISTS geography_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("geography_experience_regions",
        """
        CREATE TABLE IF NOT EXISTS geography_experience_regions (
            id SERIAL PRIMARY KEY,
            geography_experience_id INTEGER REFERENCES geography_experiences(id) ON DELETE CASCADE,
            region VARCHAR(100)
        );
        """),
        ("titles_held",
        """
        CREATE TABLE IF NOT EXISTS titles_held (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
            title VARCHAR(255),
            company VARCHAR(255),
            start_date DATE,
            end_date DATE
        );
        """)
    ]
    
    successful_creations = 0
    total_statements = len(schema_statements)

    for i, (table_name, statement) in enumerate(schema_statements):
        try:
            logging.info(f"Executing statement {i+1}/{total_statements}: CREATE TABLE {table_name}...")
            cur.execute(statement)
            conn.commit() # Commit after each statement
            successful_creations += 1
            logging.info(f"SUCCESS: Table '{table_name}' created/checked.")
        except psycopg2.Error as e:
            conn.rollback() # Rollback the failed transaction only
            logging.error(f"FAILURE on statement {i+1}/{total_statements}: Table '{table_name}' failed to create.")
            logging.error(f"Database Error: {e.pgcode} - {e.pgerror}")
            logging.error(f"Failing Query: {statement.strip()}")
            # DO NOT raise here, let it continue attempting the remaining tables

    logging.info(f"Schema creation finished. {successful_creations} out of {total_statements} tables were successfully created/checked.")


def ingest_data():
    logging.info("Starting JSON data ingestion...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise
        
    # --- 1. DROP ALL EXISTING TABLES FOR A CLEAN START ---
    drop_all_tables(cur, conn)

    # --- 2. ENSURE PGVECTOR EXTENSION IS ENABLED ---
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        logging.info("pgvector extension enabled.")
    except Exception as e:
        logging.error(f"Error enabling pgvector extension. Please ensure you have installed pgvector on your PostgreSQL server: {e}")
        conn.rollback()
        raise

    # Register vector type handler for psycopg2
    try:
        register_vector(conn)
    except Exception as e:
        logging.error(f"Failed to register pgvector type: {e}")
        raise

    # --- 3. CREATE SCHEMA WITH CORRECTED TABLE ORDER (All 17 tables) ---
    create_schema(cur, conn) # Pass conn to allow for per-statement commits

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
        
    # Helper function to correctly format array fields (handle strings -> list conversion)
    def format_array_field(data):
        if isinstance(data, list):
            return data
        if isinstance(data, str) and data:
            return [s.strip() for s in data.split(',') if s.strip()]
        return []

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

            # Insert into roles with normalized company data
            roles = profile.get('roles', [])
            if roles:
                roles_to_insert = []
                for r in roles:
                    company_name = r.get('company')
                    # Skip role if company name is missing, as it's required for upsert
                    if not company_name:
                        logging.warning(f"Skipping role for candidate {candidate_id} due to missing company name.")
                        continue
                        
                    company_details = r.get('company_details', {})
                    
                    # --- 1. Prepare & Upsert Company Data ---
                    
                    customer_segment = format_array_field(company_details.get('customer_segment'))
                    customer_presence = format_array_field(company_details.get('customer_presence'))

                    # Truncate strings for VARCHAR fields
                    funding_stage = company_details.get('funding_stage', '')[:255]
                    culture_type = company_details.get('culture_type', '')[:255]
                    business_model = company_details.get('business_model', '')[:255]
                    headquarters = company_details.get('headquarters', '')[:255]

                    # Upsert into companies table to get company_id
                    cur.execute("""
                        INSERT INTO companies (
                            name, funding_stage, revenue, business_model, product_service, 
                            customer_segment, customer_presence, culture_type, headquarters
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            funding_stage = EXCLUDED.funding_stage, 
                            revenue = EXCLUDED.revenue,
                            business_model = EXCLUDED.business_model,
                            product_service = EXCLUDED.product_service,
                            customer_segment = EXCLUDED.customer_segment,
                            customer_presence = EXCLUDED.customer_presence,
                            culture_type = EXCLUDED.culture_type,
                            headquarters = EXCLUDED.headquarters
                        RETURNING id;
                    """, (
                        company_name, company_details.get('funding_stage'), company_details.get('revenue'), 
                        company_details.get('business_model'), company_details.get('product_service'), 
                        customer_segment, customer_presence, 
                        company_details.get('culture_type'), company_details.get('headquarters')
                    ))
                    company_id = cur.fetchone()[0]

                    # --- 2. Prepare Role Data (Now just storing company_id) ---
                    roles_to_insert.append(
                        (
                            candidate_id, 
                            company_id, # Use company_id instead of company_name
                            r.get('title'), 
                            r.get('details'), 
                            r.get('duration_years')
                        )
                    )
                
                # --- 3. Insert Role Data ---
                if roles_to_insert:
                    execute_values(cur, """
                        INSERT INTO roles (
                            candidate_id, company_id, title, details, duration_years
                        ) VALUES %s
                    """, roles_to_insert)

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

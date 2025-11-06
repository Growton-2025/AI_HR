import pandas as pd
import openai
import json
import logging
import os
import asyncio
import time
from datetime import datetime, timezone # Import timezone
from dateutil import parser
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Database & Embedding Imports ---
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from langchain_openai import OpenAIEmbeddings

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Load Environment Variables ===
load_dotenv()

# === Configuration ===
# --- Source File ---
EXCEL_PATH = "/home/nethranand-ps/AI_HR/individual/Sales Leader - Sample Dataset.xlsx"

# --- AI & Concurrency Settings ---
ACCURATE_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONCURRENCY_LIMIT = 20  # Max parallel AI calls AND max DB connections

# postgresql://growton_ai_user:QDIt4TrRF57WmjjycR7hGU8Uojg1xBw0@dpg-d46deqa4d50c73avjb1g-a.singapore-postgres.render.com/growton_ai_zcaz

# --- Database Configuration ---
DB_NAME = "growton_ai_zcaz"
DB_USER = "growton_ai_user"
DB_PASSWORD = "QDIt4TrRF57WmjjycR7hGU8Uojg1xBw0"
DB_HOST = "dpg-d46deqa4d50c73avjb1g-a.singapore-postgres.render.com"
DB_PORT = "5432"

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    exit(1)

# === Database Helper Functions (from ingest_data.py) ===


def get_db_connection_params():
    """Returns a dictionary of DB connection parameters."""
    return {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
        "sslmode": "require"
    }

def create_schema(conn):
    """Create the database schema with audit columns and unique constraints."""
    schema_statements = [
        ("candidates",
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            linkedin VARCHAR(255) UNIQUE,
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
            max_people_managed INTEGER, -- Allows NULL
            years_team_management NUMERIC, -- Allows NULL
            raw_fields JSONB,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            created_by VARCHAR(255)
        );
        """),
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
            headquarters VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            created_by VARCHAR(255)
        );
        """),
        ("roles",
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            title VARCHAR(255),
            details TEXT,
            duration_years NUMERIC
        );
        """),
        ("education",
        """
        CREATE TABLE IF NOT EXISTS education (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            company VARCHAR(255),
            years NUMERIC
        );
        """),
        ("experience_gaps",
        """
        CREATE TABLE IF NOT EXISTS experience_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("functional_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS functional_experience_roles (
            id SERIAL PRIMARY KEY,
            functional_experience_id INTEGER NOT NULL REFERENCES functional_experiences(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("industry_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS industry_experience_roles (
            id SERIAL PRIMARY KEY,
            industry_experience_id INTEGER NOT NULL REFERENCES industry_experiences(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("segment_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS segment_experience_roles (
            id SERIAL PRIMARY KEY,
            segment_experience_id INTEGER NOT NULL REFERENCES segment_experiences(id) ON DELETE CASCADE,
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
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("geography_experience_regions",
        """
        CREATE TABLE IF NOT EXISTS geography_experience_regions (
            id SERIAL PRIMARY KEY,
            geography_experience_id INTEGER NOT NULL REFERENCES geography_experiences(id) ON DELETE CASCADE,
            region VARCHAR(100)
        );
        """),
        ("titles_held",
        """
        CREATE TABLE IF NOT EXISTS titles_held (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            title VARCHAR(255),
            company VARCHAR(255),
            start_date DATE,
            end_date DATE
        );
        """)
    ]

    with conn.cursor() as cur:
        # Enable pgvector
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logger.info("pgvector extension enabled.")
        except Exception as e:
            logger.error(f"Error enabling pgvector extension: {e}")
            conn.rollback()
            raise

        register_vector(conn)

        for table_name, statement in schema_statements:
            try:
                cur.execute(statement)
                conn.commit()
                logger.info(f"SUCCESS: Table '{table_name}' created/checked.")
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"FAILURE: Table '{table_name}' failed to create: {e}")
                raise

        # Add updated_at trigger
        try:
            cur.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at() RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS sync_updated_at ON candidates;
                CREATE TRIGGER sync_updated_at
                BEFORE UPDATE ON candidates
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS sync_updated_at_companies ON companies;
                CREATE TRIGGER sync_updated_at_companies
                BEFORE UPDATE ON companies
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
            """)
            conn.commit()
            logger.info("Triggers created/updated successfully.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Error creating triggers: {e}")
            raise

def parse_date(date_str):
    """
    Parse date strings from JSON, handling various formats and stripping time components.
    This is now more robust to handle 'NaT', 'nat', None, and empty strings. Returns None on failure.
    """
    if not date_str: # Handles None
        return None

    # Robustly clean and check for "NaT" or other null-like strings
    clean_str = str(date_str).strip().lower()
    # Check explicitly for variations indicating current/invalid
    if not clean_str or clean_str in ["present", "current", "na", "nat", "nat::timestamp"]:
        return None

    # Handle if it's already a datetime object (e.g., from pandas)
    if isinstance(date_str, datetime):
        # Ensure it's naive or convert to naive UTC before getting date
        if date_str.tzinfo is not None and date_str.tzinfo.utcoffset(date_str) is not None:
             dt_naive_utc = date_str.astimezone(timezone.utc).replace(tzinfo=None)
             return dt_naive_utc.date()
        else: # Already naive
             return date_str.date()
    if isinstance(date_str, pd.Timestamp):
         if pd.isna(date_str): # Check if it's NaT
              return None
         # Convert pandas Timestamp to standard datetime, then get date
         # Make timezone naive before getting date if it's aware, to avoid issues with DATE type
         try:
             dt_aware = date_str.to_pydatetime()
             # Convert aware datetime to naive UTC representation
             if dt_aware.tzinfo is not None and dt_aware.tzinfo.utcoffset(dt_aware) is not None:
                  dt_naive_utc = dt_aware.astimezone(timezone.utc).replace(tzinfo=None)
                  return dt_naive_utc.date()
             else: # If already naive
                  return dt_aware.date()
         except Exception as e:
              logger.warning(f"Error converting pandas Timestamp to date: {e}. Timestamp: {date_str}")
              return None


    # date_str is now a non-empty string that isn't null-like
    date_str_val = str(date_str) # Ensure it's a string for splitting
    if ' ' in date_str_val: # Handle cases like '2023-05-01 00:00:00'
        date_str_val = date_str_val.split(' ')[0]
    try:
        # Try parsing full YYYY-MM-DD
        if len(date_str_val.split('-')) == 3:
            return datetime.strptime(date_str_val, "%Y-%m-%d").date()
        # Try parsing YYYY-MM
        elif len(date_str_val.split('-')) == 2:
            return datetime.strptime(date_str_val + "-01", "%Y-%m-%d").date()
        # Try parsing YYYY
        elif len(date_str_val) == 4 and date_str_val.isdigit():
            return datetime.strptime(date_str_val + "-01-01", "%Y-%m-%d").date()
        # Try pandas parser as a fallback for other formats (like MM/DD/YYYY etc.)
        else:
            # errors='coerce' will return NaT (Not a Time) for unparseable formats
            dt = pd.to_datetime(date_str_val, errors='coerce')
            if pd.isna(dt): # Check if coercion resulted in NaT
                logger.warning(f"Pandas failed to parse date, returning None: {date_str_val}")
                return None
            # Convert pandas Timestamp to standard Python date
            return dt.date()
    except Exception as e:
        logger.warning(f"Error parsing date format, returning None: {date_str_val}. Error: {e}")
        return None


def format_array_field(data):
    """Helper to format list data for Postgres array fields."""
    if isinstance(data, list):
        # Ensure all elements are strings for DB compatibility
        return [str(item) for item in data]
    if isinstance(data, str) and data:
        return [s.strip() for s in data.split(',') if s.strip()]
    return []

# === Profile Processing Helper Functions (from individual.py) ===

def clean(x):
    result = str(x).strip() if pd.notnull(x) else ""
    return result

def get_datetime(val):
    """Convert value to pd.Timestamp (UTC-aware), returning NaT on failure."""
    if pd.isnull(val):
        return pd.NaT

    val_str = str(val).strip().lower()
    if not val_str or val_str in ['na', '', 'present', 'current']:
        return pd.NaT

    dt_naive = pd.NaT # Initialize as NaT

    # 1. Handle existing datetime/timestamp objects
    if isinstance(val, (datetime, pd.Timestamp)):
        # Convert to pandas Timestamp to easily handle timezone
        dt_pd = pd.Timestamp(val)
        if dt_pd.tzinfo is None:
             # If naive, localize to UTC
             try:
                 return dt_pd.tz_localize('UTC')
             except Exception as tz_err:
                  logger.error(f"Failed to localize existing naive timestamp {dt_pd}: {tz_err}")
                  return pd.NaT
        else:
             # If already aware, convert to UTC
             return dt_pd.tz_convert('UTC')

    # 2. Handle numeric values (Excel serial dates or potentially just years)
    else:
        try:
            num = float(val_str)
            if 1900 <= num <= 2100: # Likely a year
                dt_naive = pd.Timestamp(f"{int(num)}-01-01")
            elif num <= 2958465: # Check range for Excel serial dates
                # Excel origin '1899-12-30'. Result is naive.
                dt_naive = pd.Timestamp(num, unit='d', origin='1899-12-30')
            else: # Number too large
                logger.debug(f"Numeric value {num} outside valid date range, returning NaT.")
                return pd.NaT
        except (ValueError, TypeError):
            # 3. Handle string parsing if not numeric
            try:
                # pd.to_datetime attempts various formats. Returns naive by default.
                dt_naive = pd.to_datetime(val_str, errors='coerce')
                # Check explicitly if coercion failed
                if pd.isna(dt_naive):
                    logger.debug(f"Failed to parse date string '{val_str}' with pd.to_datetime.")
                    return pd.NaT # Return NaT immediately if parsing failed
            except Exception as e:
                logger.error(f"Unexpected error parsing date string '{val_str}': {e}, returning NaT.")
                return pd.NaT # Return NaT on unexpected error

    # 4. Localize to UTC if we successfully obtained a naive timestamp
    if pd.notnull(dt_naive):
        try:
            # Ensure it's naive before localizing
            if dt_naive.tzinfo is None:
                return dt_naive.tz_localize('UTC')
            else:
                 # Should not happen if logic above is correct, but convert just in case
                 logger.warning(f"Timestamp {dt_naive} unexpectedly aware before localization. Converting.")
                 return dt_naive.tz_convert('UTC')
        except Exception as tz_err:
            logger.error(f"Failed to localize timestamp {dt_naive}: {tz_err}")
            return pd.NaT
    else:
        # If dt_naive is still NaT after all attempts
        return pd.NaT


def merge_periods(periods):
    if not periods: return []
    # Filter out periods with NaT dates before sorting
    valid_periods = [p for p in periods if pd.notnull(p[0]) and pd.notnull(p[1])]
    if not valid_periods: return []

    valid_periods = sorted(valid_periods, key=lambda x: x[0])
    merged = [valid_periods[0]]
    for curr in valid_periods[1:]:
        prev = merged[-1]
        # Ensure comparison is valid (should be safe now with consistent timezones)
        if pd.notnull(curr[0]) and pd.notnull(prev[1]) and curr[0] <= prev[1]:
            # Ensure max works correctly
            merged[-1] = (prev[0], max(prev[1], curr[1]) if pd.notnull(curr[1]) else prev[1])
        else:
            merged.append(curr)
    return merged

def extract_gap_years(row, current_date):
    periods = []
    for i in range(1, 11):
        idx = '' if i == 1 else f'.{i-1}'
        start = get_datetime(row.get(f"Start date{idx}"))
        raw_end = row.get(f"End Date{idx}") # Get raw value first

        # Determine the end date: current_date if ongoing, parsed date if valid, else NaT
        end = pd.NaT # Default to NaT
        if pd.isnull(raw_end) or str(raw_end).strip().lower() in ['na', '', 'present', 'current']:
             end = current_date # Use current date (already UTC-aware)
        else:
            parsed_end_dt = get_datetime(raw_end) # Attempt parsing (expecting UTC-aware or NaT)
            if pd.notnull(parsed_end_dt): # Check if parsing was successful
                 end = parsed_end_dt

        # Only add period if both start and end are valid Timestamps and end > start
        # Comparison is now safe as both should be UTC-aware or NaT
        if pd.notnull(start) and pd.notnull(end) and end > start:
            periods.append((start, end))

    merged = merge_periods(periods)
    if not merged: return False, []

    gaps = []
    for j in range(len(merged) - 1):
        gap_start, gap_end = merged[j][1], merged[j+1][0]
        # Check if dates involved in gap calculation are valid
        if pd.notnull(gap_start) and pd.notnull(gap_end) and gap_end > gap_start:
            days = (gap_end - gap_start).days
            if days > 180: # More than ~6 months
                gaps.append({
                    "from": gap_start.strftime("%Y-%m"),
                    "to": gap_end.strftime("%Y-%m"),
                    "duration_months": round(days / 30),
                    "reason": "unknown"
                })
        else:
            # Refined Warning: Log only if dates are non-NaT but comparison fails
             if pd.notnull(gap_start) and pd.notnull(gap_end) and gap_end <= gap_start:
                 logger.warning(f"Skipping potential gap calculation: Next start date ({gap_end.date()}) is not after previous end date ({gap_start.date()}).")
             # else: (Implicitly handles cases where one or both are NaT - no warning needed)


    return len(gaps) > 0, gaps


def extract_education_gaps(row, current_date):
    periods = []
    for i in range(1, 4):
        college = clean(row.get(f"Education {i} - College Name"))
        if not college: continue # Skip if no college name

        s = get_datetime(row.get(f"Start date.{9+i}"))
        raw_e = row.get(f"End Date.{9+i}")

        e = pd.NaT
        if pd.isnull(raw_e) or str(raw_e).strip().lower() in ['na', '', 'present', 'current']:
             e = current_date
        else:
            parsed_end_dt = get_datetime(raw_e)
            if pd.notnull(parsed_end_dt):
                 e = parsed_end_dt

        if pd.notnull(s) and pd.notnull(e) and e > s:
            periods.append((s, e))

    merged = merge_periods(periods)
    if not merged: return False, []

    gaps = []
    for a, b in zip(merged, merged[1:]):
        # Ensure dates are valid for gap calculation
        if pd.notnull(a[1]) and pd.notnull(b[0]) and b[0] > a[1]:
            days = (b[0] - a[1]).days
            if days > 180:
                gaps.append({
                    "from": a[1].strftime("%Y-%m"), "to": b[0].strftime("%Y-%m"),
                    "duration_months": round(days / 30), "reason": "education"
                })
        else:
             # Refined Warning
             if pd.notnull(a[1]) and pd.notnull(b[0]) and b[0] <= a[1]:
                 logger.warning(f"Skipping potential education gap calculation: Next start date ({b[0].date()}) is not after previous end date ({a[1].date()}).")


    return bool(gaps), gaps

def extract_industry_gaps(roles, ind_roles):
    # Check if ind_roles is valid and has the expected structure
    if not isinstance(ind_roles, list): # Basic check: ind_roles should be a list
        logger.warning(f"Cannot calculate industry gaps: ind_roles is not a list ({type(ind_roles)}).")
        return False, []
    if not ind_roles or len(roles) != len(ind_roles):
        logger.warning(f"Mismatch or empty ind_roles ({len(ind_roles)}) for industry gaps. Roles count: {len(roles)}. Cannot calculate.")
        return False, []

    stints = []
    for r, ir in zip(roles, ind_roles):
        # **Crucial Check**: Ensure 'ir' is a dictionary before accessing keys
        if not isinstance(ir, dict):
            logger.warning(f"Skipping role for industry gap calculation: Expected dict in ind_roles, got {type(ir)}. Role: {r.get('title')} at {r.get('company')}")
            continue # Skip this iteration if ir is not a dictionary

        # Now safe to use .get()
        start_dt = r.get("start_dt")
        end_dt = r.get("end_dt")
        industry = ir.get("industry") # Use .get() for safety

        if pd.notnull(start_dt) and pd.notnull(end_dt) and end_dt > start_dt and industry:
            stints.append((start_dt, end_dt, industry))
        else:
             logger.debug(f"Skipping role for industry gap calculation due to missing/invalid data: Start={start_dt}, End={end_dt}, Industry={industry}")


    if not stints:
        logger.debug("No valid stints found for industry gap calculation after checks.")
        return False, []

    # Sort valid stints by start date
    stints.sort(key=lambda x: x[0])

    gaps = []
    # Iterate through pairs of consecutive stints
    for (s1, e1, ind1), (s2, e2, ind2) in zip(stints, stints[1:]):
        # Ensure dates are valid for gap calculation (now safe to compare directly)
        if pd.notnull(e1) and pd.notnull(s2) and s2 > e1:
            days = (s2 - e1).days
            # Check for gap duration and different industries
            if days > 180 and ind1 != ind2:
                gaps.append({
                    "from": e1.strftime("%Y-%m"),
                    "to": s2.strftime("%Y-%m"),
                    "duration_months": round(days / 30),
                    "from_industry": ind1,
                    "to_industry": ind2,
                    "reason": "industry"
                })
        else:
            # Refined Warning: Log only if dates are non-NaT but comparison fails (s2 <= e1)
             if pd.notnull(e1) and pd.notnull(s2) and s2 <= e1:
                 logger.warning(f"Skipping potential industry gap calculation: Next start date ({s2.date()}) is not after previous end date ({e1.date()}).")
             # else: (Implicitly handles cases where one or both are NaT - no warning needed)


    return len(gaps) > 0, gaps



# === Asynchronous AI Helper Functions (from individual.py) ===

async def call_gpt_async(client, model, data, prompt, retries=3, delay=5):
    """
    Asynchronously call OpenAI and robustly parse JSON response.
    Includes exponential backoff for rate limits.
    """
    # Only log prompt start for brevity, not the full data payload
    logger.info(f"Calling {model} for prompt starting with: {prompt[:80]}...")
    messages=[
        {"role": "system", "content": "You MUST return *only* a single valid JSON object—no extra text, no markdown, no explanations. Use `null` for missing integer/float values where appropriate (e.g., if team size is not explicitly mentioned, return `null`, not 0). Accurately extract and sum numbers when requested."},
        {"role": "user", "content": f"{prompt}\n\nData:\n{json.dumps(data, indent=2, default=str)}"} # Use default=str for dates
    ]

    # Log the full request structure if needed for deep debugging
    # logger.debug(f"Full OpenAI Request Messages: {json.dumps(messages, indent=2)}")

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048, # Increased slightly
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content.strip()
            # Log raw response for debugging JSON issues
            # logger.debug(f"Raw GPT response content: {content}")

            # Basic JSON validation before parsing
            if not content.startswith('{') or not content.endswith('}'):
                 logger.warning(f"Response does not look like JSON: {content[:100]}...")
                 # Attempt to extract if possible (optional, might be risky)
                 start = content.find("{")
                 end = content.rfind("}")
                 if start != -1 and end != -1:
                     content = content[start:end+1]
                 else:
                     raise ValueError("Valid JSON object boundaries not found in response.")

            parsed_json = json.loads(content)
            logger.info(f"Successfully received and parsed JSON from {model}.")
            return parsed_json

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit calling {model}. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
        except json.JSONDecodeError as e:
            logger.error(f"Error PARSING JSON from {model}: {e}. Response content was: {content[:500]}...") # Log more content on error
            return None # Return None on parsing error
        except Exception as e:
            # Log specific error type and message
            logger.error(f"Error calling OpenAI ({type(e).__name__}): {e}")
            return None  # Return None on non-retryable error

    logger.error(f"Failed to call OpenAI model {model} after all retries (RateLimit likely).")
    return None


async def get_or_fetch_company_details(client, companies_needed, company_cache, lock):
    """
    Asynchronously checks in-memory cache for company details, fetches missing ones,
    and updates the *global* cache. This is lock-protected to prevent race conditions.
    """
    new_companies_to_fetch = []

    # Check cache to see what we're missing (thread-safe)
    async with lock:
        # Filter out empty strings or None before adding to fetch list
        valid_companies_needed = [comp for comp in companies_needed if comp and isinstance(comp, str)]
        for comp in valid_companies_needed:
            if comp not in company_cache:
                new_companies_to_fetch.append(comp)
                company_cache[comp] = {} # Add placeholder to prevent re-fetching

    if not new_companies_to_fetch:
        return

    logger.info(f"Fetching details for {len(new_companies_to_fetch)} new companies using {FAST_MODEL}...")
    comp_prompt = (
        'For each company in the list, provide firmographic details. Your response must be a single JSON object where keys are the company names.'
        'Each value should be an object with keys: "product_service", "customer_segment" (list), "customer_presence" (list), "funding_stage", "revenue", "culture_type", "headquarters", "business_model". '
        'If info is unavailable, use null or empty string/list.'
    )

    # API call is outside the lock
    # Make sure the data payload is just the list of names
    comp_res = await call_gpt_async(client, FAST_MODEL, {"companies": new_companies_to_fetch}, comp_prompt)

    if comp_res:
         # Check if the response itself is a dictionary (as expected)
        if isinstance(comp_res, dict):
            # Lock again to safely update the global cache
            async with lock:
                logger.info(f"Updating company cache with details for {len(comp_res)} companies.")
                 # Iterate through the fetched results and update cache
                for company_name, details in comp_res.items():
                    if company_name in company_cache: # Update existing placeholder or entry
                         company_cache[company_name] = details
                    else:
                         logger.warning(f"Received details for unexpected company '{company_name}' - ignoring.")
        else:
            logger.error(f"Failed to fetch company details: API response was not a dictionary. Response: {str(comp_res)[:200]}")

    else:
        logger.error(f"Failed to fetch any new company details for: {new_companies_to_fetch}. API call returned None.")


# === NEW: Database Ingestion Function (Replaces save_profile_incrementally) ===

def ingest_profile_to_db(profile: dict, conn, embeddings_client: OpenAIEmbeddings, user_id="system_ingest"):
    """
    Ingests a single, fully formed profile dictionary into the database.
    This function is BLOCKING and should be run in an executor.
    It combines all the logic from ingest_data.py's main loop.
    """

    # This function assumes `conn` is a valid connection from the pool.
    # It will commit or rollback within a transaction.
    with conn.cursor() as cur:
        try:
            # 1. Create embedding
            roles_summary = " ".join([f"{r.get('title', '')} {r.get('details', '')}" for r in profile.get('roles', [])])
            skills = profile.get('raw_fields', {}).get('Skills', '')
            document_text = (
                f"Name: {profile.get('name', '')}. Headline: {profile.get('headline', '')}. "
                f"About: {profile.get('about', '')}. Experience: {roles_summary}. Skills: {skills}."
            )
            # Ensure document_text is not empty before embedding
            if not document_text.strip():
                 logger.warning(f"Skipping embedding for profile {profile.get('name')} due to empty content.")
                 embedding_vector = None # Or handle as needed, e.g., default vector
            else:
                 embedding_vector = embeddings_client.embed_query(document_text)

            raw_fields = profile.get('raw_fields', {})

            # 2. Insert or update candidate
            # Ensure numeric fields default to None if missing/invalid, not 0 unless appropriate
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
                profile.get('total_experience_years'), # Already calculated
                profile.get('avg_years_in_company'), # Already calculated
                profile.get('has_gap_years'),
                profile.get('has_education_gaps'),
                profile.get('has_industry_gaps'),
                profile.get('functional_experience', {}).get('score'),
                profile.get('functional_experience', {}).get('rationale'),
                profile.get('industry_experience', {}).get('score'),
                profile.get('industry_experience', {}).get('rationale'),
                profile.get('segment_experience', {}).get('score'),
                profile.get('segment_experience', {}).get('rationale'),
                profile.get('geography_experience', {}).get('score'),
                profile.get('geography_experience', {}).get('rationale'),
                profile.get('team_management', {}).get('score'),
                profile.get('team_management', {}).get('rationale'),
                profile.get('team_management', {}).get('max_people_managed'), # Should default to null/None based on prompt
                profile.get('team_management', {}).get('years_team_management'), # Should default to null/None based on prompt
                json.dumps(raw_fields) if raw_fields else None,
                embedding_vector, # Can be None now
                datetime.now(),
                user_id
            )

            cur.execute("""
                INSERT INTO candidates (
                    name, first_name, last_name, linkedin, location, city, headline, about, skills,
                    licenses_and_certifications, total_experience_years, avg_years_in_company,
                    has_gap_years, has_education_gaps, has_industry_gaps,
                    functional_experience_score, functional_experience_rationale,
                    industry_experience_score, industry_experience_rationale,
                    segment_experience_score, segment_experience_rationale,
                    geography_experience_score, geography_experience_rationale,
                    team_management_score, team_management_rationale,
                    max_people_managed, years_team_management, raw_fields, embedding,
                    created_at, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (linkedin) DO UPDATE SET
                    name = EXCLUDED.name,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    location = EXCLUDED.location,
                    city = EXCLUDED.city,
                    headline = EXCLUDED.headline,
                    about = EXCLUDED.about,
                    skills = EXCLUDED.skills,
                    licenses_and_certifications = EXCLUDED.licenses_and_certifications,
                    total_experience_years = EXCLUDED.total_experience_years,
                    avg_years_in_company = EXCLUDED.avg_years_in_company,
                    has_gap_years = EXCLUDED.has_gap_years,
                    has_education_gaps = EXCLUDED.has_education_gaps,
                    has_industry_gaps = EXCLUDED.has_industry_gaps,
                    functional_experience_score = EXCLUDED.functional_experience_score,
                    functional_experience_rationale = EXCLUDED.functional_experience_rationale,
                    industry_experience_score = EXCLUDED.industry_experience_score,
                    industry_experience_rationale = EXCLUDED.industry_experience_rationale,
                    segment_experience_score = EXCLUDED.segment_experience_score,
                    segment_experience_rationale = EXCLUDED.segment_experience_rationale,
                    geography_experience_score = EXCLUDED.geography_experience_score,
                    geography_experience_rationale = EXCLUDED.geography_experience_rationale,
                    team_management_score = EXCLUDED.team_management_score,
                    team_management_rationale = EXCLUDED.team_management_rationale,
                    max_people_managed = EXCLUDED.max_people_managed,
                    years_team_management = EXCLUDED.years_team_management,
                    raw_fields = EXCLUDED.raw_fields,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP,
                    created_by = EXCLUDED.created_by
                RETURNING id;
            """, candidate_params)

            candidate_id = cur.fetchone()[0]

            # --- Refreshed Logic for Child Records ---
            # Instead of deleting all, we query existing ones and only insert/update/delete as needed.
            # This is more complex but better for preserving IDs if they matter elsewhere.
            # FOR SIMPLICITY HERE, we stick to the DELETE ALL then INSERT pattern.
            # If preserving child record IDs becomes important, this section needs significant rework.

            # 3. Delete existing child records to refresh them
            child_tables = [
                'roles', 'education', 'company_years', 'experience_gaps', 'education_gaps',
                'industry_gaps', 'titles_held'
            ]
            for table in child_tables:
                cur.execute(f"DELETE FROM {table} WHERE candidate_id = %s;", (candidate_id,))

            # Delete related experience records (these have their own tables now)
            cur.execute("DELETE FROM functional_experiences WHERE candidate_id = %s;", (candidate_id,)) # This will cascade delete functional_experience_roles
            cur.execute("DELETE FROM industry_experiences WHERE candidate_id = %s;", (candidate_id,))   # This will cascade delete industry_experience_roles
            cur.execute("DELETE FROM segment_experiences WHERE candidate_id = %s;", (candidate_id,))    # This will cascade delete segment_experience_roles
            cur.execute("DELETE FROM geography_experiences WHERE candidate_id = %s;", (candidate_id,))  # This will cascade delete geography_experience_regions


            # 4. Insert roles and companies (UPSERT logic for companies is good)
            roles = profile.get('roles', [])
            if roles:
                roles_to_insert = []
                # --- Company Cache & Upsert ---
                company_name_to_id = {} # Cache IDs for this transaction

                for r in roles:
                    company_name = r.get('company')
                    if not company_name:
                        logger.warning(f"Skipping role for candidate {candidate_id} due to missing company name: {r.get('title')}")
                        continue

                    company_id = company_name_to_id.get(company_name)
                    if not company_id:
                        company_details = r.get('company_details', {})
                        # Ensure lists are formatted correctly for DB array type
                        customer_segment = format_array_field(company_details.get('customer_segment'))
                        customer_presence = format_array_field(company_details.get('customer_presence'))

                        cur.execute("""
                            INSERT INTO companies (
                                name, funding_stage, revenue, business_model, product_service,
                                customer_segment, customer_presence, culture_type, headquarters,
                                created_at, created_by
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (name) DO UPDATE SET
                                funding_stage = COALESCE(EXCLUDED.funding_stage, companies.funding_stage),
                                revenue = COALESCE(EXCLUDED.revenue, companies.revenue),
                                business_model = COALESCE(EXCLUDED.business_model, companies.business_model),
                                product_service = COALESCE(EXCLUDED.product_service, companies.product_service),
                                customer_segment = COALESCE(EXCLUDED.customer_segment, companies.customer_segment),
                                customer_presence = COALESCE(EXCLUDED.customer_presence, companies.customer_presence),
                                culture_type = COALESCE(EXCLUDED.culture_type, companies.culture_type),
                                headquarters = COALESCE(EXCLUDED.headquarters, companies.headquarters),
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING id;
                        """, (
                            company_name, company_details.get('funding_stage'), company_details.get('revenue'),
                            company_details.get('business_model'), company_details.get('product_service'),
                            customer_segment, customer_presence, company_details.get('culture_type'),
                            company_details.get('headquarters'), datetime.now(), user_id
                        ))
                        company_id = cur.fetchone()[0]
                        company_name_to_id[company_name] = company_id # Cache the ID

                    # Add role data using the obtained company_id
                    roles_to_insert.append((
                        candidate_id,
                        company_id,
                        r.get('title'),
                        r.get('details'),
                        r.get('duration_years')
                     ))

                if roles_to_insert:
                    execute_values(cur, """
                        INSERT INTO roles (candidate_id, company_id, title, details, duration_years)
                        VALUES %s
                    """, roles_to_insert)

            # 5. Insert other child tables (experience scores, gaps, etc.)
            # --- Functional Experience ---
            func_exp = profile.get('functional_experience', {})
            if func_exp and isinstance(func_exp, dict): # Check it's a dict
                cur.execute("INSERT INTO functional_experiences (candidate_id, score, rationale) VALUES (%s, %s, %s) RETURNING id;",
                            (candidate_id, func_exp.get('score'), func_exp.get('rationale')))
                func_exp_id = cur.fetchone()[0]
                func_roles = func_exp.get('roles', [])
                if func_roles and isinstance(func_roles, list): # Check it's a list
                    func_roles_to_insert = [
                        (func_exp_id, r.get('company'), r.get('activity_type'), r.get('reason'), r.get('duration_years'))
                        for r in func_roles if isinstance(r, dict) # Ensure each role is a dict
                    ]
                    if func_roles_to_insert:
                        execute_values(cur, "INSERT INTO functional_experience_roles (functional_experience_id, company, activity_type, reason, duration_years) VALUES %s", func_roles_to_insert)

            # --- Industry Experience ---
            ind_exp = profile.get('industry_experience', {})
            if ind_exp and isinstance(ind_exp, dict):
                cur.execute("INSERT INTO industry_experiences (candidate_id, score, rationale) VALUES (%s, %s, %s) RETURNING id;",
                            (candidate_id, ind_exp.get('score'), ind_exp.get('rationale')))
                ind_exp_id = cur.fetchone()[0]
                ind_roles = ind_exp.get('roles', [])
                if ind_roles and isinstance(ind_roles, list):
                    ind_roles_to_insert = [
                        (ind_exp_id, r.get('company'), r.get('industry'), r.get('reason'), r.get('duration_years'))
                        for r in ind_roles if isinstance(r, dict)
                    ]
                    if ind_roles_to_insert:
                         execute_values(cur, "INSERT INTO industry_experience_roles (industry_experience_id, company, industry, reason, duration_years) VALUES %s", ind_roles_to_insert)


            # --- Segment Experience ---
            seg_exp = profile.get('segment_experience', {})
            if seg_exp and isinstance(seg_exp, dict):
                cur.execute("INSERT INTO segment_experiences (candidate_id, score, rationale) VALUES (%s, %s, %s) RETURNING id;",
                            (candidate_id, seg_exp.get('score'), seg_exp.get('rationale')))
                seg_exp_id = cur.fetchone()[0]
                seg_roles = seg_exp.get('roles', [])
                if seg_roles and isinstance(seg_roles, list):
                    seg_roles_to_insert = [
                        (seg_exp_id, r.get('company'), r.get('segment'), r.get('reason'), r.get('duration_years'))
                         for r in seg_roles if isinstance(r, dict)
                    ]
                    if seg_roles_to_insert:
                         execute_values(cur, "INSERT INTO segment_experience_roles (segment_experience_id, company, segment, reason, duration_years) VALUES %s", seg_roles_to_insert)

            # --- Geography Experience ---
            geo_exp = profile.get('geography_experience', {})
            if geo_exp and isinstance(geo_exp, dict):
                cur.execute("INSERT INTO geography_experiences (candidate_id, score, rationale) VALUES (%s, %s, %s) RETURNING id;",
                            (candidate_id, geo_exp.get('score'), geo_exp.get('rationale')))
                geo_exp_id = cur.fetchone()[0]
                regions = geo_exp.get('regions', [])
                 # Ensure regions is a list before processing
                if regions and isinstance(regions, list):
                    # Filter out non-string elements just in case
                    valid_regions = [region for region in regions if isinstance(region, str)]
                    if valid_regions:
                         regions_to_insert = [(geo_exp_id, region) for region in valid_regions]
                         execute_values(cur, "INSERT INTO geography_experience_regions (geography_experience_id, region) VALUES %s", regions_to_insert)

            # --- Gaps, Education, Titles (using execute_values) ---
            company_years_data = profile.get('company_years', {})
            if company_years_data and isinstance(company_years_data, dict):
                company_years_to_insert = [(candidate_id, company, years) for company, years in company_years_data.items()]
                if company_years_to_insert:
                     execute_values(cur, "INSERT INTO company_years (candidate_id, company, years) VALUES %s", company_years_to_insert)


            gaps = profile.get('gaps', [])
            if gaps and isinstance(gaps, list):
                gaps_to_insert = [
                     (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason'))
                     for g in gaps if isinstance(g, dict)
                ]
                # Filter out rows with None dates before inserting for gaps
                gaps_to_insert_filtered = [g for g in gaps_to_insert if g[1] is not None and g[2] is not None]
                if gaps_to_insert_filtered:
                    execute_values(cur, "INSERT INTO experience_gaps (candidate_id, from_date, to_date, duration_months, reason) VALUES %s", gaps_to_insert_filtered)


            education_gaps = profile.get('education_gaps', [])
            if education_gaps and isinstance(education_gaps, list):
                edu_gaps_to_insert = [
                     (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason'))
                     for g in education_gaps if isinstance(g, dict)
                ]
                 # Filter out rows with None dates
                edu_gaps_to_insert_filtered = [g for g in edu_gaps_to_insert if g[1] is not None and g[2] is not None]
                if edu_gaps_to_insert_filtered:
                    execute_values(cur, "INSERT INTO education_gaps (candidate_id, from_date, to_date, duration_months, reason) VALUES %s", edu_gaps_to_insert_filtered)

            industry_gaps = profile.get('industry_gaps', [])
            if industry_gaps and isinstance(industry_gaps, list):
                ind_gaps_to_insert = [
                     (candidate_id, parse_date(g.get('from')), parse_date(g.get('to')), g.get('duration_months'), g.get('reason')) # Assuming 'reason' is correct, might need 'from_industry', 'to_industry' depending on schema
                     for g in industry_gaps if isinstance(g, dict)
                ]
                 # Filter out rows with None dates
                ind_gaps_to_insert_filtered = [g for g in ind_gaps_to_insert if g[1] is not None and g[2] is not None]
                if ind_gaps_to_insert_filtered:
                    execute_values(cur, "INSERT INTO industry_gaps (candidate_id, from_date, to_date, duration_months, reason) VALUES %s", ind_gaps_to_insert_filtered)


            education_history = profile.get('education', [])
            if education_history and isinstance(education_history, list):
                edu_to_insert = [
                    (candidate_id, e.get('college'), e.get('degree'), parse_date(e.get('start')), parse_date(e.get('end')), e.get('details'))
                     for e in education_history if isinstance(e, dict)
                ]
                # Allow NULL dates for education start/end - NO filtering needed here
                if edu_to_insert:
                    execute_values(cur, "INSERT INTO education (candidate_id, college, degree, start_date, end_date, details) VALUES %s", edu_to_insert)

            titles = profile.get('titles_held', [])
            if titles and isinstance(titles, list):
                titles_to_insert = [
                     (candidate_id, t.get('title'), t.get('company'), parse_date(t.get('start')), parse_date(t.get('end')))
                     for t in titles if isinstance(t, dict)
                ]
                # Allow NULL dates for titles start/end - NO filtering needed here
                if titles_to_insert:
                    execute_values(cur, "INSERT INTO titles_held (candidate_id, title, company, start_date, end_date) VALUES %s", titles_to_insert)


            # 6. Commit the transaction
            conn.commit()

        except psycopg2.Error as db_err:
             # Log detailed database errors
             logger.error(f"Database error ingesting profile {profile.get('name')} (Candidate ID: {candidate_id if 'candidate_id' in locals() else 'N/A'}): {db_err}")
             # Check if diag has message_detail, otherwise log pgerror
             detail = db_err.diag.message_detail if hasattr(db_err, 'diag') and hasattr(db_err.diag, 'message_detail') else db_err.pgerror
             logger.error(f"SQLSTATE: {db_err.pgcode}, DETAIL/ERROR: {detail}")
             conn.rollback()
             raise # Re-raise after logging
        except Exception as e:
             # Catch other unexpected errors during ingestion
             logger.error(f"Unexpected error ingesting profile {profile.get('name')}: {type(e).__name__} - {e}")
             conn.rollback()
             raise # Re-raise the exception to be caught by the async processor


# === UPDATED: Asynchronous Candidate Processing Function ===
async def process_candidate_async(row, client, semaphore, current_date,
                                  company_cache, company_lock,
                                  db_pool, embeddings_client):
    """
    Processes a single candidate row asynchronously using SEPARATE prompts,
    fetches company data, assembles the full profile, and INGESTS it directly to the database.
    Calculates total_experience_years based on overall date range, handling overlaps.
    Calculates avg_years_in_company based on average duration per company.
    """
    async with semaphore: # Limit concurrency for accurate model calls
        profile_name = f"{clean(row.get('First Name'))} {clean(row.get('Last Name'))}".strip()
        conn = None
        loop = asyncio.get_running_loop() # Get current running loop
        try:
            logger.info(f"--- Processing profile: {profile_name} ---")
            raw_fields = {col: clean(row[col]) for col in row.index}

            # --- Corrected Experience Calculation Logic ---
            roles = []
            company_years = {} # Stores total duration (in years) per company
            unique_companies = set()
            last_valid_company = ""
            overall_earliest_start = pd.NaT
            overall_latest_end = pd.NaT

            for i in range(1, 11):
                idx_str = '' if i == 1 else f'.{i-1}'
                comp = clean(row.get(f"Company {i} Name"))
                title = clean(row.get(f"Title{idx_str}"))

                if not comp and title: comp = last_valid_company
                elif comp: last_valid_company = comp
                if not comp or not title: continue

                unique_companies.add(comp)
                if comp not in company_years:
                    company_years[comp] = 0.0

                raw_start = row.get(f"Start date{idx_str}")
                raw_end = row.get(f"End Date{idx_str}")
                details_key = f"Details .{i-1}" if i > 1 else "Details "
                details_text = clean(row.get(details_key))

                start_dt = get_datetime(raw_start)
                end_dt = pd.NaT
                if pd.isnull(raw_end) or str(raw_end).strip().lower() in ['na', '', 'present', 'current']:
                    end_dt = current_date
                else:
                    parsed_end_dt = get_datetime(raw_end)
                    if pd.notnull(parsed_end_dt):
                        end_dt = parsed_end_dt

                yrs = 0.0 # Default duration for this role
                # Calculate role duration and track overall dates IF dates are valid
                if pd.notnull(start_dt) and pd.notnull(end_dt) and end_dt >= start_dt: # Allow same start/end
                    # Track overall min/max dates
                    if pd.isna(overall_earliest_start) or start_dt < overall_earliest_start:
                        overall_earliest_start = start_dt
                    if pd.isna(overall_latest_end) or end_dt > overall_latest_end:
                        overall_latest_end = end_dt

                    # Calculate duration for this specific role and add to company total
                    days = (end_dt - start_dt).days
                    # Handle potential edge case of very short roles resulting in 0 days but valid dates
                    # Or roles starting/ending on same day
                    if days >= 0:
                        yrs = round(days / 365.25, 2)
                        company_years[comp] += yrs
                    else: # Should not happen with end_dt >= start_dt check, but just in case
                         logger.warning(f"Negative duration calculated for role at {comp} for {profile_name}. Start: {start_dt}, End: {end_dt}. Setting duration to 0.")


                else:
                    if raw_start or raw_end:
                        logger.warning(f"Invalid dates for role at {comp} for {profile_name}. Start: {start_dt}, End: {end_dt}. Duration calculation skipped for this role.")

                # Add role info (always)
                roles.append({
                    "company": comp, "title": title,
                    "details": details_text,
                    "duration_years": yrs,
                    "start_dt": start_dt, "end_dt": end_dt,
                    "start": str(raw_start), "end": str(raw_end)
                })

            # --- Calculate Total Experience (Career Span) ---
            total_exp = 0.0
            if pd.notnull(overall_earliest_start) and pd.notnull(overall_latest_end) and overall_latest_end >= overall_earliest_start:
                total_span_days = (overall_latest_end - overall_earliest_start).days
                # Ensure minimum of 0 days if start/end are same
                if total_span_days >= 0:
                    total_exp = round(total_span_days / 365.25, 2)
            else:
                 logger.warning(f"Could not determine valid overall date range for {profile_name}. total_experience_years set to 0.")


            # --- Calculate Average Tenure Per Company ---
            total_company_durations = sum(company_years.values())
            # Count only companies where *some* valid duration was recorded
            valid_company_count = sum(1 for yrs in company_years.values() if yrs > 0)
            avg_tenure = round(total_company_durations / valid_company_count, 2) if valid_company_count > 0 else 0


            # Gaps (Pre-AI - refined functions handle NaT checks)
            has_gap, gaps = extract_gap_years(row, current_date)
            has_edu_gap, edu_gaps = extract_education_gaps(row, current_date)

            about_sections = [raw_fields.get("about"), raw_fields.get("Details"), raw_fields.get("Details.1")]
            combined_about_text = " ".join(filter(None, about_sections))

            # --- Prepare Data for Separate API Calls ---
            gpt_data_base = {
                "headline": clean(row.get("headline")),
                "about": combined_about_text,
                # Include roles *with* duration for context, even if calling separately
                "roles": [{"company": r["company"], "title": r["title"], "details": r["details"], "duration_years": r["duration_years"]} for r in roles],
                "location": clean(row.get("addressWithCountry")),
            }

            # --- Define Separate Prompts (Keep Team Management Strict) ---
            func_prompt = (
                 "Evaluate the candidate's functional sales experience (e.g., prospecting, closing, account management) based *only* on the provided headline, about, and role details. "
                 "Output JSON: {\"score\": int (1-10), \"rationale\": str (brief justification), \"roles\": list of {\"company\": str, \"activity_type\": str (e.g., 'Account Management', 'New Business Development'), \"reason\": str (why this role contributes), \"duration_years\": float}}."
            )
            ind_prompt = (
                 "Identify the primary industry domain for each role based *only* on the provided company, title, and details. Avoid guessing. If unclear, state 'Unknown'. "
                 "Output JSON: {\"score\": int (1-10, based on consistency/depth in specific industries), \"rationale\": str (brief justification), \"roles\": list of {\"company\": str, \"industry\": str (e.g., 'SaaS', 'Finance', 'Healthcare', 'Unknown'), \"reason\": str (why this industry classification), \"duration_years\": float}}."
             )
            seg_prompt = (
                 "Identify the customer segment (e.g., Enterprise, SMB, Mid-Market, Public Sector) targeted in each role based *only* on explicit mentions in the headline, about, or role details. Avoid guessing. If unclear, state 'Unknown'. "
                 "Output JSON: {\"score\": int (1-10, based on clarity/consistency of segment focus), \"rationale\": str (brief justification), \"roles\": list of {\"company\": str, \"segment\": str (e.g., 'Enterprise', 'SMB', 'Unknown'), \"reason\": str (evidence for segment), \"duration_years\": float}}."
            )
            geo_prompt = (
                 "Identify the geographic regions of responsibility or experience based *only* on explicit mentions in the location, headline, about, or role details (e.g., 'APAC', 'North America', 'India'). Do not infer from location alone unless specified as territory. "
                 "Output JSON: {\"score\": int (1-10, based on breadth/depth of specified regions), \"rationale\": str (brief justification), \"regions\": list of str (e.g., ['India', 'Southeast Asia'])}."
            )
            tm_prompt = (
                 "Strictly extract team management experience based *only* on explicit statements in headline, about, or role details. DO NOT infer from titles (like Director or Manager). "
                 "Identify *all* explicitly mentioned numbers of people managed (e.g., 'led 15 reps', 'managed 10 account managers and 5 BDRs'). "
                 "Output JSON: {"
                 "\"score\": int (1-10, based ONLY on extracted evidence),"
                 "\"rationale\": str (justify score based on evidence or lack thereof),"
                 "\"max_people_managed\": int (SUM of *all* explicitly mentioned numbers found across all texts. Default to null if no specific number(s) mentioned anywhere),"
                 "\"years_team_management\": float (Sum duration_years ONLY for roles EXPLICITLY stating *any* people management responsibility (e.g., 'led team', 'managed reports', or specific counts). Default to null if no role explicitly mentions management)"
                 "}"
             )


            # --- Execute Separate API Calls Concurrently (within semaphore limit) ---
            results = await asyncio.gather(
                call_gpt_async(client, ACCURATE_MODEL, gpt_data_base, func_prompt),
                call_gpt_async(client, ACCURATE_MODEL, gpt_data_base, ind_prompt),
                call_gpt_async(client, ACCURATE_MODEL, gpt_data_base, seg_prompt),
                call_gpt_async(client, ACCURATE_MODEL, gpt_data_base, geo_prompt),
                call_gpt_async(client, ACCURATE_MODEL, gpt_data_base, tm_prompt),
                return_exceptions=True
            )

            # --- Process Results, Handle Potential Failures ---
            func_res, ind_res, seg_res, geo_res, tm_res = {}, {}, {}, {}, {} # Default

            if isinstance(results[0], Exception) or results[0] is None: logger.error(f"Functional Exp call failed for {profile_name}: {results[0]}")
            else: func_res = results[0]

            if isinstance(results[1], Exception) or results[1] is None: logger.error(f"Industry Exp call failed for {profile_name}: {results[1]}")
            else: ind_res = results[1]

            if isinstance(results[2], Exception) or results[2] is None: logger.error(f"Segment Exp call failed for {profile_name}: {results[2]}")
            else: seg_res = results[2]

            if isinstance(results[3], Exception) or results[3] is None: logger.error(f"Geography Exp call failed for {profile_name}: {results[3]}")
            else: geo_res = results[3]

            if isinstance(results[4], Exception) or results[4] is None: logger.error(f"Team Management call failed for {profile_name}: {results[4]}")
            else: tm_res = results[4]


            # --- Industry Gap Calculation (after ind_res is available) ---
            industry_roles_list = ind_res.get('roles') if isinstance(ind_res, dict) else None
            has_ind_gap, ind_gaps = extract_industry_gaps(roles, industry_roles_list if isinstance(industry_roles_list, list) else [])



            # --- ASYNC API CALL (Conditional): Company Details ---
            await get_or_fetch_company_details(client, list(unique_companies), company_cache, company_lock)

            # Education
            education = []
            for j in range(1, 4):
                if college := clean(row.get(f"Education {j} - College Name")):
                    deg_idx = '' if j == 1 else f'.{j-1}'
                    start_date_str = str(row.get(f"Start date.{9 + j}"))
                    end_date_str = str(row.get(f"End Date.{9 + j}"))
                    education.append({
                        "college": college,
                        "degree": clean(row.get(f"Degree Name{deg_idx}")),
                        "start": start_date_str, "end": end_date_str,
                        "details": ""
                    })


            # Final Profile Assembly
            location_str = clean(row.get("addressWithCountry"))
            city = location_str.split(",")[0].strip() if location_str else ""

            for r in roles:
                 company_name = r.get("company")
                 if company_name and isinstance(company_name, str):
                     r["company_details"] = company_cache.get(company_name, {})
                 else:
                     r["company_details"] = {}


            profile = {
                "name": profile_name,
                "linkedin": clean(row.get("Person Linkedin Url")),
                "location": location_str,
                "city": city,
                "headline": clean(row.get("headline")),
                "about": combined_about_text,
                "roles": [{k: v for k, v in r.items() if k not in ["start_dt", "end_dt"]} for r in roles],
                "raw_fields": raw_fields,
                "total_experience_years": total_exp, # Use the career span calculation
                "avg_years_in_company": avg_tenure, # Use the corrected average tenure
                "company_years": company_years,
                "has_gap_years": has_gap, "gaps": gaps,
                "has_education_gaps": has_edu_gap, "education_gaps": edu_gaps,
                "has_industry_gaps": has_ind_gap, "industry_gaps": ind_gaps,
                "functional_experience": func_res,
                "industry_experience": ind_res,
                "segment_experience": seg_res,
                "geography_experience": geo_res,
                "team_management": tm_res,
                "education": education,
                "titles_held": [{"title": r["title"], "company": r["company"], "start": r["start"], "end": r["end"]} for r in roles],
            }

            # --- Asynchronous Database Ingestion ---
            try:
                conn = await loop.run_in_executor(None, db_pool.getconn)
                await loop.run_in_executor(None, ingest_profile_to_db, profile, conn, embeddings_client)
                logger.info(f"Successfully ingested profile to DB: {profile.get('name', 'N/A')}")
            except Exception as e:
                failed_name = profile.get('name', 'N/A')
                logger.error(f"Failed to ingest profile {failed_name} to DB: {type(e).__name__} - {e}")

        except Exception as e:
            logger.error(f"CRITICAL error processing profile {profile_name}: {type(e).__name__} - {e}")
            # import traceback # Optional for detailed debugging
            # logger.error(traceback.format_exc())

        finally:
            if conn:
                try:
                    await loop.run_in_executor(None, db_pool.putconn, conn)
                except Exception as pool_e:
                    logger.error(f"Error returning connection to pool: {pool_e}")


# === UPDATED: Main Orchestrator Function ===
async def main():
    logger.info("🚀 Asynchronous processing pipeline starting…")
    start_time = time.time()

    # --- 1. Initialize DB Connection Pool & Embeddings ---
    db_conn_params = get_db_connection_params()
    db_pool = None # Initialize pool to None
    try:
        # Create a thread-safe connection pool
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=CONCURRENCY_LIMIT + 5, # Allow a few extra connections for flexibility
             **db_conn_params
        )
        embeddings_client = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("Database connection pool and embeddings client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize DB pool or embeddings: {e}")
        if db_pool: # Close pool if partially initialized
             db_pool.closeall()
        return


    # --- 2. Setup Database Schema ---
    conn = None
    try:
        conn = db_pool.getconn()
        create_schema(conn)
        # No need to commit here, create_schema handles commits per table
    except Exception as e:
        logger.error(f"Failed to create/verify schema: {e}")
        # Ensure connection is returned even if schema creation fails
    finally:
         if conn: # Always return connection if obtained
             try:
                 db_pool.putconn(conn)
             except Exception as pool_e:
                 logger.error(f"Error returning connection after schema check: {pool_e}")
         # Do not close the pool here yet


    # --- 3. Load Excel Data ---
    try:
        df = pd.read_excel(EXCEL_PATH)
        # Pre-process DataFrame: Fill NaN with empty string for consistent comparison later
        df = df.fillna('')
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        db_pool.closeall()
        return

    logger.info(f"Loaded Excel file with {len(df)} rows.")

    # --- 4. Check for Existing Profiles (Resume Logic) ---
    processed_profiles_data = {}
    conn = None # Reset conn variable
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            # Fetching raw_fields as JSONB (dict) directly
            cur.execute("SELECT linkedin, raw_fields FROM candidates WHERE linkedin IS NOT NULL AND raw_fields IS NOT NULL")
            # Store as {linkedin_url: raw_fields_dict}
            # Ensure raw_fields loaded from JSONB is treated correctly (should be dict)
            processed_profiles_data = {row[0]: row[1] for row in cur.fetchall() if row[0] and isinstance(row[1], dict)}
        logger.info(f"Found {len(processed_profiles_data)} existing profile 'signatures' in DB for change detection.")
    except Exception as e:
        logger.error(f"Failed to fetch existing profiles: {e}")
        # Ensure connection returned on error
    finally:
        if conn:
            try:
                db_pool.putconn(conn)
            except Exception as pool_e:
                logger.error(f"Error returning connection after fetching profiles: {pool_e}")
        # Do not close pool yet


    current_date = pd.to_datetime("now", utc=True) # Use timezone-aware current time

    # --- 5. Setup Async Components ---
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    company_cache = {} # In-memory cache for this run
    company_lock = asyncio.Lock()

    tasks = []

    # --- 6. Create Candidate Processing Tasks ---
    logger.info("Starting task creation loop...")
    skipped_unchanged = 0
    queued_for_processing = 0
    skipped_missing_linkedin = 0

    for index, row in df.iterrows(): # Use index from iterrows
        linkedin_url = clean(row.get("Person Linkedin Url"))

        if not linkedin_url:
            logger.warning(f"Skipping Excel row index {index} due to missing LinkedIn URL.")
            skipped_missing_linkedin += 1
            continue

        # Create raw_fields dict from the current Excel row using clean()
        # This ensures consistent string representation (handling None, NaN -> '')
        current_raw_fields = {col: clean(row[col]) for col in df.columns}

        # Get the stored signature (raw_fields dict) from the DB
        stored_raw_fields_dict = processed_profiles_data.get(linkedin_url)

        # --- Change Detection Logic ---
        needs_processing = False
        if not stored_raw_fields_dict:
            logger.info(f"Found new profile (Excel index {index}): {linkedin_url}. Queuing for ingestion.")
            needs_processing = True
        else:
             # Direct comparison should work if both are dicts with consistent string values
             if current_raw_fields != stored_raw_fields_dict:
                 logger.info(f"Detected changes for existing profile (Excel index {index}): {linkedin_url}. Queuing for update.")
                 needs_processing = True
             else:
                 skipped_unchanged += 1 # Increment skip counter
                 # logger.debug(f"Skipping unchanged profile (Excel index {index}): {linkedin_url}")


        if needs_processing:
            queued_for_processing += 1
            tasks.append(process_candidate_async(
                row, client, semaphore, current_date,
                company_cache, company_lock,
                db_pool, embeddings_client
            ))

    logger.info(f"Task Creation Summary: Queued={queued_for_processing}, Skipped (Unchanged)={skipped_unchanged}, Skipped (No LinkedIn)={skipped_missing_linkedin}")


    num_to_process = len(tasks)
    if num_to_process == 0:
        logger.info("✅ No new or changed profiles needed processing. Exiting.")
        db_pool.closeall()
        return

    logger.info(f"Starting processing for {num_to_process} profiles...")


    # --- 7. Run all tasks concurrently ---
    logger.info("Running all tasks concurrently...")
    start_gather_time = time.time()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    end_gather_time = time.time()
    logger.info(f"asyncio.gather completed in {end_gather_time - start_gather_time:.2f} seconds.")


    # --- 8. Final Tally ---
    failed_tasks = 0
    successful_tasks = 0
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            # Error should have been logged within process_candidate_async or ingest_profile_to_db
            logger.error(f"Task {i+1} completed with an exception (see logs above for details): {result}")
            failed_tasks += 1
        # No explicit check for None needed if tasks don't return None on success/handled failure
        # Assume successful if no exception was raised
        else:
             successful_tasks += 1


    logger.info(f"--- Task execution summary: {successful_tasks} successful, {failed_tasks} failed (out of {num_to_process} queued). ---")

    # --- 9. Cleanup ---
    logger.info("Closing database connection pool...")
    db_pool.closeall()
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"✅ Total pipeline execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")
    logger.info(f"Data processing complete. Check '{DB_NAME}' database and logs for details.")


# --- Run the main asynchronous function ---
if __name__ == "__main__":
    # Optional: Add argument parsing here if needed
    asyncio.run(main())


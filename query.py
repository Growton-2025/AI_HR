import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
import psycopg2
from pgvector.psycopg2 import register_vector
import redis
import hashlib
from datetime import datetime
import logging
import tiktoken
import asyncio
from typing import List, Dict, Any, AsyncIterator, Tuple

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Database Configuration ---
DB_NAME = "growton_ai"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- OpenAI and Redis Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.ConnectionError as e:
    st.error(f"Failed to connect to Redis: {e}")
    st.stop()

# --- LLM and Embeddings Initialization ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
streaming_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)
specialist_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Dynamic Taxonomy Generation ---

def safe_json_loads(json_str: str, default_val: Any = None) -> Any:
    """Safely loads a JSON string, stripping markdown and handling errors."""
    if default_val is None:
        default_val = {}
    try:
        # Clean the string from common LLM output artifacts
        cleaned_str = json_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:].rstrip("```").strip()
        elif cleaned_str.startswith("`"):
            cleaned_str = cleaned_str.strip("`").strip()

        if not cleaned_str:
            return default_val
        return json.loads(cleaned_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Could not parse JSON string: '{json_str}'")
        return default_val

@st.cache_data(ttl=86400) # Cache the taxonomy for 24 hours
def generate_dynamic_taxonomy(seed_taxonomy: dict, category: str) -> dict:
    """
    Uses an LLM to expand a seed taxonomy with more synonyms and related terms.
    The result is cached to avoid repeated LLM calls.
    """
    logger.info(f"Generating dynamic taxonomy for category: {category}... (This will be cached)")

    prompt_template = PromptTemplate(
        input_variables=["seed_taxonomy_json", "category"],
        template="""
        You are an expert HR and recruitment analyst specializing in {category}.
        Your task is to expand the given seed taxonomy with a comprehensive list of synonyms, related job titles, and common variations.

        Maintain the original keys as the canonical names. For each key, significantly expand its list of values.
        For example, if the key is 'Hunting', the values should include terms like 'new business development', 'net new logo acquisition', 'hunter', 'closer', 'Account Executive (New Business)', etc.

        The final output MUST be a valid JSON object with the exact same structure as the input seed.

        **Seed Taxonomy for {category}:**
        {seed_taxonomy_json}

        **Expanded JSON Output:**
        """
    )

    try:
        formatted_prompt = prompt_template.format(
            seed_taxonomy_json=json.dumps(seed_taxonomy, indent=2),
            category=category
        )
        response = generation_llm.invoke(formatted_prompt)
        expanded_taxonomy = safe_json_loads(response.content, default_val=seed_taxonomy)

        if expanded_taxonomy == seed_taxonomy:
            logger.warning(f"LLM failed to generate an expanded taxonomy for {category}. Falling back to the static seed.")
            return seed_taxonomy

        logger.info(f"Successfully generated and cached dynamic taxonomy for {category}.")
        return expanded_taxonomy
    except Exception as e:
        logger.error(f"An error occurred during taxonomy generation for {category}: {e}")
        return seed_taxonomy

# --- Seed Taxonomies (Used for LLM-powered expansion) ---
STATIC_SALES_TAXONOMY = {
    'Hunting': ['Hunting', 'new accounts', 'net new', 'New Closures', 'Account Executive'],
    'Farming': ['Account management', 'Account manager', 'Farming', 'Retention'],
    'Sales Development': ['Sales Development', 'Business Development', 'inside sales', 'SDR', 'BDR', 'account development', 'client development'],
    'Partner Sales': ['Partner Sales', 'Partner Development', 'Channel Sales', 'alliance management'],
    'Customer Success': ['Customer Success', 'customer retention']
}

STATIC_SEGMENT_SYNONYMS = {
    "enterprise": ["enterprise", "large enterprise", "large customers"],
    "mid-market": ["mid-market", "medium size customers"],
    "smb": ["smb", "small business", "small and medium business", "sme"]
}

STATIC_COMPANY_DETAILS_TAXONOMY = {
    "bootstrapped": ["bootstrapped", "self-funded"],
    "seed": ["seed", "seed stage", "pre-seed"],
    "series-a": ["series a", "series-a"],
    "series-b": ["series b", "series-b"],
    "public": ["public", "publicly traded", "ipo"],
    "b2b": ["b2b", "business-to-business"],
    "b2c": ["b2c", "business-to-consumer"],
    "saas": ["saas", "software as a service"]
}

# --- NEW: Culture Taxonomy ---
STATIC_CULTURE_TAXONOMY = {
    "startup": ["startup", "fast-paced", "agile environment", "high-growth", "early-stage"],
    "corporate": ["corporate", "mnc", "multinational", "large enterprise", "structured environment", "established company"],
    "remote": ["remote-first", "fully remote", "distributed team"]
}


# --- ✨ DYNAMIC TAXONOMY INITIALIZATION ✨ ---
SALES_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SALES_TAXONOMY,
    category="Sales Functions"
)

SEGMENT_SYNONYMS = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SEGMENT_SYNONYMS,
    category="Customer Segments"
)

COMPANY_DETAILS_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_COMPANY_DETAILS_TAXONOMY,
    category="Company Attributes (Funding, Business Model)"
)

# --- NEW: Initialize Culture Taxonomy ---
CULTURE_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_CULTURE_TAXONOMY,
    category="Company Culture Types"
)


# Log the results to see what the LLM created
logger.info(f"Loaded Sales Taxonomy with {sum(len(v) for v in SALES_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Segment Taxonomy with {sum(len(v) for v in SEGMENT_SYNONYMS.values())} total terms.")
logger.info(f"Loaded Company Details Taxonomy with {sum(len(v) for v in COMPANY_DETAILS_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Culture Taxonomy with {sum(len(v) for v in CULTURE_TAXONOMY.values())} total terms.")


# --- Database Connection Pool ---
# Using a simple connection function for this script's structure
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Database connection failed: {e}")
        logger.error(f"Database connection failed: {e}")
        st.stop()

# --- Caching Data from Database ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_all_company_names_from_db():
    """Loads all unique company names from the database."""
    logger.info("Loading all unique company names from the database into cache...")
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM companies ORDER BY name")
            company_names = [row[0] for row in cur.fetchall()]
            logger.info(f"Successfully loaded and cached {len(company_names)} unique company names.")
            return company_names
    except psycopg2.Error as e:
        logger.error(f"Failed to load company names from DB: {e}")
        return []
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=3600) # Cache for 1 hour
def load_all_profiles_from_db():
    """
    Loads all candidate profiles and their roles from the database based on the new schema.
    This function adapts the new schema's structure into the nested dictionary format
    that the rest of the application's logic expects.
    """
    logger.info("Loading all profiles from the database into cache...")
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all candidates using the core fields needed for the app
    cur.execute("SELECT id, name, linkedin, location, headline, about, total_experience_years, max_people_managed FROM candidates")
    candidates_raw = cur.fetchall()

    # Fetch all roles and join with companies table to get company details
    cur.execute("""
        SELECT
            r.candidate_id,
            c.name as company_name,
            r.title,
            r.details,
            r.duration_years,
            c.funding_stage,
            c.revenue,
            c.business_model,
            c.product_service,
            c.customer_segment,
            c.customer_presence,
            c.culture_type,
            c.headquarters
        FROM roles r
        JOIN companies c ON r.company_id = c.id
    """)
    roles_raw = cur.fetchall()
    roles_by_candidate = {}
    for role in roles_raw:
        # Map by index based on the new SELECT statement
        candidate_id = role[0]
        if candidate_id not in roles_by_candidate:
            roles_by_candidate[candidate_id] = []

        # Construct the company_details dictionary with all the fetched fields
        company_details = {
            "funding_stage": role[5],
            "revenue": role[6],
            "business_model": role[7],
            "product_service": role[8],
            "customer_segment": role[9] if role[9] is not None else [],
            "customer_presence": role[10] if role[10] is not None else [],
            "culture_type": role[11],
            "headquarters": role[12],
            # Adding 'industry' for compatibility with downstream functions, mapping it from product_service.
            "industry": role[8] or ""
        }

        roles_by_candidate[candidate_id].append({
            "company": role[1], # company_name
            "title": role[2],
            "details": role[3],
            "duration_years": float(role[4]) if role[4] is not None else 0.0,
            "company_details": company_details
        })

    # Combine candidates and their roles
    profiles = []
    for cand in candidates_raw:
        candidate_id = cand[0]
        profiles.append({
            "id": candidate_id,
            "name": cand[1],
            "linkedin": cand[2],
            "location": cand[3],
            "headline": cand[4],
            "about": cand[5],
            "total_experience_years": float(cand[6]) if cand[6] is not None else 0.0,
            "max_people_managed": cand[7] or 0,
            "roles": roles_by_candidate.get(candidate_id, [])
        })

    cur.close()
    conn.close()
    logger.info(f"Successfully loaded and cached {len(profiles)} profiles.")
    return profiles

# Load data into a global variable for the app session
PROFILES_BY_ID = {p['id']: p for p in load_all_profiles_from_db()}
ALL_COMPANY_NAMES = load_all_company_names_from_db()

# --- Core Logic ---

def normalize_query_with_llm(query: str) -> str:
    """Uses LLM to normalize common synonyms in the query."""
    logger.info(f"Normalizing query... Search Query: {query}")
    return query.lower().replace("sme", "smb").replace("mid market", "mid-market")

def get_cache_key(prefix: str, text: str) -> str:
    """Generates a consistent cache key."""
    return f"{prefix}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

# --- Specific Experience Calculation Functions ---

def calculate_functional_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles that meet the functional criteria."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(v in role_text for v in req_values):
            duration = role.get('duration_years', 0.0) or 0.0
            total_duration += duration
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': duration
            })
    return total_duration, contributing_roles

def calculate_industry_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles that meet the industry criteria."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        role_text = (
            f"{(role.get('company') or '').lower()} "
            f"{(company_details.get('industry', '') or '').lower()} "
            f"{(company_details.get('product_service', '') or '').lower()}"
        )
        if any(v in role_text for v in req_values):
            duration = role.get('duration_years', 0.0) or 0.0
            total_duration += duration
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': duration
            })
    return total_duration, contributing_roles

def calculate_segment_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles that meet the segment criteria."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    all_search_terms = {}
    for v in req_values:
        # Use the SEGMENT_SYNONYMS for expansion
        all_search_terms[v] = SEGMENT_SYNONYMS.get(v, [v])

    for role in profile.get('roles', []):
        company_segments = role.get("company_details", {}).get("customer_segment", [])
        company_segments_lower = ' '.join([cs.lower() for cs in company_segments])
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()} {company_segments_lower}"

        for original_value, synonyms in all_search_terms.items():
            if any(s in role_text for s in synonyms):
                duration = role.get('duration_years', 0.0) or 0.0
                total_duration += duration
                contributing_roles.append({
                    'company': role.get('company', ''),
                    'title': role.get('title', ''),
                    'duration_years': duration
                })
                break # Avoid double counting a role if it matches multiple synonyms
    return total_duration, contributing_roles

def calculate_geography_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles that meet the geography criteria."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(v in role_text for v in req_values):
            duration = role.get('duration_years', 0.0) or 0.0
            total_duration += duration
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': duration
            })
    return total_duration, contributing_roles

def calculate_company_details_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles that meet the company detail criteria."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        details_text = (
            f"{(company_details.get('funding_stage') or '').lower()} "
            f"{(company_details.get('business_model') or '').lower()} "
            f"{(company_details.get('product_service') or '').lower()}"
        )
        if any(v in details_text for v in req_values):
            duration = role.get('duration_years', 0.0) or 0.0
            total_duration += duration
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': duration
            })
    return total_duration, contributing_roles


# --- Presence Check Functions ---

def check_company_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a specifically named company."""
    # This key is a simple list, not a dict with operator/values
    required_companies = criteria.get("required_companies")
    if not required_companies or not isinstance(required_companies, list):
        return True # No requirement, so it passes

    # Case-insensitive matching
    required_companies_lower = [c.lower() for c in required_companies]
    found_companies = set()

    for company_name in required_companies_lower:
        for role in profile.get('roles', []):
            if company_name in (role.get('company') or '').lower():
                found_companies.add(company_name)
                break # Move to the next required company

    # For companies, we assume AND logic. The user wants experience in ALL specified companies.
    is_met = found_companies == set(required_companies_lower)
    if is_met:
        profile['evidence_log'].append({
            "criterion": "company_presence (AND)",
            "source_text": f"Profile confirms experience in all required companies: {', '.join(required_companies)}."
        })
    return is_met

def check_industry_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a company matching the industry criteria, supporting AND/OR logic."""
    criteria_obj = criteria.get("required_industries")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            company_details = role.get('company_details', {})
            role_text = (
                f"{(role.get('company') or '').lower()} "
                f"{(company_details.get('industry', '') or '').lower()} "
                f"{(company_details.get('product_service', '') or '').lower()}"
            )
            if v in role_text:
                found_values.add(v)
                break

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "industry_presence (AND)",
                "source_text": f"Profile confirms experience in all required industries: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "industry_presence (OR)",
                "source_text": f"Profile confirms experience in at least one required industry. Found: {', '.join(found_values)}."
            })
        return is_met

def check_functional_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a role matching the functional criteria, supporting AND/OR logic."""
    criteria_obj = criteria.get("required_functions")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            if v in role_text:
                found_values.add(v)
                break

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "functional_presence (AND)",
                "source_text": f"Profile confirms experience in all required functions: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "functional_presence (OR)",
                "source_text": f"Profile confirms experience in at least one required function. Found: {', '.join(found_values)}."
            })
        return is_met

def check_customer_segments(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has experience in the required customer segments, supporting AND/OR logic."""
    criteria_obj = criteria.get("required_segments")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    # Expand synonyms for all values
    all_search_terms = {}
    for v in values:
        all_search_terms[v] = SEGMENT_SYNONYMS.get(v, [v])

    found_values = set()
    for original_value, synonyms in all_search_terms.items():
        found_synonym = False
        for role in profile.get('roles', []):
            company_segments = role.get("company_details", {}).get("customer_segment", [])
            company_segments_lower = ' '.join([cs.lower() for cs in company_segments])
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()} {company_segments_lower}"

            if any(s in role_text for s in synonyms):
                found_values.add(original_value)
                found_synonym = True
                break
        if found_synonym:
            profile['evidence_log'].append({
                "criterion": "segment_presence (OR)",
                "source_text": f"Profile confirms experience in at least one required segment. Found: {', '.join(found_values)}."
            })

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "segment_presence (AND)",
                "source_text": f"Profile confirms experience in all required segments: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        return is_met

def check_location_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate's location matches the required locations, supporting AND/OR logic."""
    criteria_obj = criteria.get("required_locations")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    profile_location = (profile.get('location') or '').lower()
    found_values = set()
    for v in values:
        if v in profile_location:
            found_values.add(v)

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "location_presence (AND)",
                "source_text": f"Profile location confirms all required locations: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "location_presence (OR)",
                "source_text": f"Profile location confirms at least one required location. Found: {', '.join(found_values)}."
            })
        return is_met

def check_geography_experience(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has experience in the required geographies, supporting AND/OR logic."""
    criteria_obj = criteria.get("required_geographies")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            if v in role_text:
                found_values.add(v)
                break

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "geography_presence (AND)",
                "source_text": f"Profile confirms experience in all required geographies: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "geography_presence (OR)",
                "source_text": f"Profile confirms experience in at least one required geography. Found: {', '.join(found_values)}."
            })
        return is_met

def check_company_details(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a company matching the company detail criteria."""
    criteria_obj = criteria.get("required_company_details")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            company_details = role.get('company_details', {})
            # Combine multiple company fields into one text block for searching
            details_text = (
                f"{(company_details.get('funding_stage') or '').lower()} "
                f"{(company_details.get('business_model') or '').lower()} "
                f"{(company_details.get('product_service') or '').lower()}"
            )
            if v in details_text:
                found_values.add(v)
                break

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "company_details_presence (AND)",
                "source_text": f"Profile confirms experience in companies with all required attributes: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "company_details_presence (OR)",
                "source_text": f"Profile confirms experience in a company with at least one required attribute. Found: {', '.join(found_values)}."
            })
        return is_met

def check_company_culture_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a company matching the culture type criteria."""
    criteria_obj = criteria.get("required_culture_type")
    if not criteria_obj:
        return True

    op = "OR"
    values = []
    if isinstance(criteria_obj, dict):
        op = criteria_obj.get("operator", "OR").upper()
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            culture_type = (role.get('company_details', {}).get('culture_type') or '').lower()
            if v in culture_type:
                found_values.add(v)
                break

    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "culture_type_presence (AND)",
                "source_text": f"Profile confirms experience in companies with all required culture types: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "culture_type_presence (OR)",
                "source_text": f"Profile confirms experience in a company with at least one required culture type. Found: {', '.join(found_values)}."
            })
        return is_met


# --- Strict Filtering Function ---

async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs strict, deterministic filtering in Python with the new flexible logic and sorts by specified criterion duration."""
    logger.info("Applying detailed filters with reasoning...")
    matching_candidates = []

    # Determine the primary criterion for sorting (e.g., required_segments for SMB experience)
    sort_criterion = None
    if criteria.get("required_segments"):
        sort_criterion = "required_segments"
    elif criteria.get("required_functions"):
        sort_criterion = "required_functions"
    elif criteria.get("required_industries"):
        sort_criterion = "required_industries"
    elif criteria.get("required_geographies"):
        sort_criterion = "required_geographies"
    elif criteria.get("required_company_details"):
        sort_criterion = "required_company_details"
    elif criteria.get("required_culture_type"):
        sort_criterion = "required_culture_type"
    else:
        sort_criterion = "required_functions"  # Default to functions if none specified

    for profile in profiles:
        profile['evidence_log'] = []
        profile['contributing_roles_details'] = {}
        profile['calculated_experience'] = {}
        all_criteria_met = True

        # 1. Check Total Experience (if specified)
        min_total_exp = criteria.get("min_total_experience")
        if min_total_exp and (profile.get("total_experience_years") or 0) < min_total_exp:
            all_criteria_met = False

        # 2. Check Team Management Size (if specified)
        min_managed = criteria.get("min_people_managed")
        if min_managed and (profile.get("max_people_managed") or 0) < min_managed:
            all_criteria_met = False

        # 3. Check for presence in required fields (AND/OR logic)
        if all_criteria_met and not check_company_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_functional_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_industry_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_customer_segments(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_location_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_geography_experience(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_company_details(profile, criteria):
            all_criteria_met = False
        # --- NEW: Added culture check ---
        if all_criteria_met and not check_company_culture_presence(profile, criteria):
            all_criteria_met = False

        # 4. Check for dynamic year requirements and calculate durations for sorting
        if all_criteria_met:
            for key, calc_func in [
                ("required_functions", calculate_functional_experience_duration),
                ("required_industries", calculate_industry_experience_duration),
                ("required_segments", calculate_segment_experience_duration),
                ("required_geographies", calculate_geography_experience_duration),
                ("required_company_details", calculate_company_details_experience_duration)
            ]:
                crit_obj = criteria.get(key)
                if crit_obj and isinstance(crit_obj, dict) and crit_obj.get("min_years"):
                    min_y = crit_obj["min_years"]
                    duration, roles = calc_func(profile, crit_obj)

                    profile['calculated_experience'][key] = {
                        "duration": duration,
                        "roles": roles,
                        "label": ", ".join(crit_obj.get("values",[])),
                        "required": min_y
                    }

                    if duration < min_y:
                        all_criteria_met = False
                        break
                elif crit_obj and isinstance(crit_obj, dict):
                    # Calculate duration even if no min_years, for sorting purposes
                    duration, roles = calc_func(profile, crit_obj)
                    profile['calculated_experience'][key] = {
                        "duration": duration,
                        "roles": roles,
                        "label": ", ".join(crit_obj.get("values",[])),
                        "required": 0.0
                    }

        if all_criteria_met:
            # If specific experience was calculated, use it for the breakdown. Otherwise, fall back to default functional breakdown.
            if profile['calculated_experience']:
                # Prioritize the first calculated experience for the main display breakdown
                first_calc_key = next(iter(profile['calculated_experience']))
                profile['contributing_roles_details'] = {'roles': profile['calculated_experience'][first_calc_key]['roles']}
            else:
                _, roles_list = calculate_functional_experience_duration(profile, criteria.get("required_functions", {}))
                profile['contributing_roles_details'] = {'roles': roles_list}

            matching_candidates.append(profile)

    # Sort candidates by the duration of the primary criterion (e.g., SMB experience)
    if matching_candidates and sort_criterion:
        matching_candidates.sort(
            key=lambda x: x['calculated_experience'].get(sort_criterion, {}).get('duration', 0.0),
            reverse=True
        )

    # Limit to top_n candidates, or return all if top_n is 0 or not specified
    top_n = criteria.get("top_n")
    if top_n is None or top_n == 0:
        logger.info(f"No top_n specified or top_n is 0, returning all {len(matching_candidates)} candidates.")
    else:
        matching_candidates = matching_candidates[:top_n]
        logger.info(f"Found {len(matching_candidates)} candidates after strict filtering and sorting by {sort_criterion} duration, limited to top {top_n}.")

    return matching_candidates

async def generate_response_with_evidence(query: str, matching_profiles: List[dict], criteria: Dict[str, Any]) -> AsyncIterator[str]:
    """
    Generates the final response with detailed, evidence-based reasoning.
    This function now processes one profile at a time to avoid context length errors.
    """
    if not matching_profiles:
        yield "No candidates were found that strictly match all criteria with explicit evidence in their profiles."
        return

    yield f"Found {len(matching_profiles)} matching candidates. Here are the details:\n\n"

    final_answer_prompt_template = PromptTemplate(
        input_variables=["query", "criteria_json", "matching_profile_json"],
        template="""
You are an expert recruitment analyst. Your task is to generate a concise, evidence-based summary for the single candidate provided, based on the user's query and the filtering criteria.

**Original User Query:** {query}
**Filtering Criteria Used (JSON):** {criteria_json}
**Matching Candidate (JSON):** {matching_profile_json}

**CRITICAL INSTRUCTIONS FOR DYNAMIC REASONING:**

1.  **Candidate Header:** Start with the candidate's name as a large Markdown header (e.g., # Name). Then, list their LinkedIn, Location, Headline, Total Experience Years, and Max People Managed.

2.  **Create a "Reasoning" Section:**
    - You **MUST** create a bullet point for **ONLY the keys present** in the `Filtering Criteria Used` JSON.
    - **Do NOT** mention criteria like `min_total_experience` or `min_people_managed` if they are not in the `Filtering Criteria Used` JSON.

3.  **Generate Specific, Evidence-Based Reasoning:**
    - For each criterion (e.g., `required_segments`), look inside the candidate's `calculated_experience` object.
    - Find the matching key (e.g., `calculated_experience.required_segments`).
    - Use the `duration` and `roles` from that object to construct a detailed reason.
    - **Example Sentence Structure:** "The candidate meets the **SMB Experience** requirement with **[Duration] years** of experience, gained primarily from their roles at **[Company A]** and **[Company B]**."
    - This creates a direct link between the requirement, the experience duration, and the companies where it was gained.

4.  **Create a Correct "Experience Breakdown" Section:**
    - Look for the `contributing_roles_details.roles` list in the candidate's JSON.
    - If it exists, you **MUST** format it as a bulleted list under the heading "Relevant Experience".
    - **Use this exact format for each role:** `* **{{company}}**: {{title}} ({{duration_years}} years)`
    - This section should only list the specific roles that contributed to meeting the primary search criterion.

5. **No Conclusion:** Do not add any concluding remarks.

---
**Your Turn. Generate the response for the single candidate provided.**
"""
    )    
    
    for i, profile in enumerate(matching_profiles):
        profile['display_number'] = i + 1
        
        # Format the prompt for each individual profile
        final_prompt_formatted = final_answer_prompt_template.format(
            query=query,
            criteria_json=json.dumps(criteria, indent=2),
            matching_profile_json=json.dumps(profile, indent=2) # Only one profile
        )

        # Stream the response for this single profile
        async for chunk in streaming_llm.astream(final_prompt_formatted):
            yield chunk.content
        yield "\n---\n\n" # Add a separator between candidates


async def process_query_main(query: str, session_id: str) -> AsyncIterator[str]:
    """
    Main processing pipeline for a user query.
    """
    normalized_query = normalize_query_with_llm(query)

    # 1. Extract Criteria using LLM with detailed definitions
    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "sales_taxonomy_keys", "segment_taxonomy_keys", "company_details_taxonomy_keys", "culture_taxonomy_keys"],
        template="""
You are an expert assistant tasked with extracting structured filtering criteria from a user's query for a candidate search system. Your goal is to categorize user intent into functions, segments, industries, etc.

**DEFINITIONS & CANONICAL KEYS:**
- `required_companies`: A list of specific company names. Use this when the user asks for experience *at* a company (e.g., "worked at Google", "from Microsoft").
- `required_functions`: Sales roles. **Map user input to one of these keys:** {sales_taxonomy_keys}
- `required_segments`: Customer types. **Map user input to one of these keys:** {segment_taxonomy_keys}
- `required_company_details`: Company attributes like funding stage or business model. **Map to keys:** {company_details_taxonomy_keys}
- `required_culture_type`: Company environment. **Map to keys:** {culture_taxonomy_keys}
- `required_industries`: Broad industries (e.g., "SaaS", "Fintech"). Do NOT put specific company names here.
- `competitors_of`: A list of companies for which to find competitors.
- `required_geographies`: Regions of sales experience.
- `required_locations`: Candidate's physical base.
- `top_n`: Integer for the number of candidates to return.

**JSON STRUCTURE RULES:**
- For each criterion, use an object with "operator" ("AND"/"OR") and "values" (list of mapped canonical keys or strings).
- `required_companies` should be a simple list of strings.
- If years of experience are mentioned with a criterion (e.g., "10 years in inside sales"), include a "min_years" (float) key in that criterion's object.
- Use `min_total_experience` (float) only for general experience.

**EXAMPLES TABLE (Follow this logic exactly):**
| User Query                                    | Correct JSON Output                                                                                               |
|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| "experience in HCL AND Tech Mahindra"         | `{{"required_companies": ["HCL", "Tech Mahindra"]}}`                                                              |
| "candidates from the SaaS industry"           | `{{"required_industries": {{"operator": "OR", "values": ["SaaS"]}}}}`                                             |
| "candidates with startup culture experience"  | `{{"required_culture_type": {{"operator": "OR", "values": ["startup"]}}}}`                                        |
| "more than 10 years in inside sales"          | `{{"required_functions": {{"operator": "OR", "values": ["Sales Development"], "min_years": 10.0}}}}`               |
| "candidates who have worked at competitors of Oracle" | `{{"competitors_of": ["Oracle"]}}`                                                                        |
| "top 10 profiles with SMB experience"         | `{{"required_segments": {{"operator": "OR", "values": ["smb"]}}, "top_n": 10}}`                                   |

**Available criteria keys:**
- `min_total_experience` (float)
- `min_people_managed` (integer)
- `required_locations` (list of strings)
- `required_geographies` (object)
- `required_companies` (list of strings)
- `required_industries` (object)
- `required_functions` (object)
- `required_segments` (object)
- `required_company_details` (object)
- `required_culture_type` (object)
- `competitors_of` (list of strings)
- `top_n` (integer)

**CRITICAL INSTRUCTION**: Use `required_companies` for specific company names and `required_industries` for general categories.

**User Query:** {query}

**JSON Criteria:**
"""
    )
    try:
        yield "Extracting criteria... "
        criteria_response = await llm.ainvoke(criteria_extraction_prompt.format(
            query=normalized_query,
            sales_taxonomy_keys=json.dumps(list(SALES_TAXONOMY.keys())),
            segment_taxonomy_keys=json.dumps(list(SEGMENT_SYNONYMS.keys())),
            company_details_taxonomy_keys=json.dumps(list(COMPANY_DETAILS_TAXONOMY.keys())),
            culture_taxonomy_keys=json.dumps(list(CULTURE_TAXONOMY.keys()))
        ))
        criteria = safe_json_loads(criteria_response.content, {})
        if not criteria:
            raise ValueError("Failed to parse criteria.")

        # Log the raw LLM response for debugging
        logger.info(f"Raw LLM criteria response: {criteria_response.content}")

        # Post-extraction override: Remove top_n for blunt queries, set to 0 for "all"
        normalized_query_lower = normalized_query.lower()
        if "all" in normalized_query_lower:
            criteria["top_n"] = 0
            logger.info("Detected 'all' in query; setting top_n to 0 to return all matches.")
        elif not any(word in normalized_query_lower for word in ["top", "one", "maximum"]):
            if "top_n" in criteria:
                logger.info(f"Blunt query detected; removing top_n (was {criteria['top_n']}) to return all matches.")
                del criteria["top_n"]
        else:
            logger.info(f"Keeping top_n as {criteria.get('top_n')} for explicit top count query.")

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error parsing criteria: {e}")
        yield "I had trouble understanding the criteria in your query. Could you please rephrase it?"
        return

    # --- NEW: Competitor Identification using dedicated LLM node ---
    if "competitors_of" in criteria and criteria.get("competitors_of"):
        company_to_find_competitors_for = criteria["competitors_of"][0]
        
        # Determine the dynamic task for the LLM based on the user's query
        competitor_task = "identify all direct competitors for the given company"
        if "top" in normalized_query.lower() and criteria.get("top_n"):
             competitor_task = f"identify the top {criteria['top_n']} direct competitors for the given company"

        yield f"Identifying competitors for **{company_to_find_competitors_for}** using the company database as context... "

        competitors_found = False
        try:
            # This is the new dedicated "LLM node" for competitor analysis with a dynamic task
            competitor_identification_prompt = PromptTemplate(
                input_variables=["company_name", "company_list_json", "competitor_task"],
                template="""
                You are an expert business analyst. Your task is to {competitor_task}.
                You MUST ONLY select competitors from the provided JSON list of companies available in our database.

                **Target Company:**
                {company_name}

                **List of all available companies in the database:**
                {company_list_json}

                Analyze the list and return a JSON list of names that are direct competitors of the target company. If you cannot find any direct competitors in the list, return an empty list.

                **JSON List of Competitors (must be names from the list above):**
                """
            )

            formatted_prompt = competitor_identification_prompt.format(
                company_name=company_to_find_competitors_for,
                company_list_json=json.dumps(ALL_COMPANY_NAMES),
                competitor_task=competitor_task
            )
            
            response = await llm.ainvoke(formatted_prompt)
            competitors = safe_json_loads(response.content, [])

            if competitors:
                yield f"Found potential competitors in DB via LLM analysis: `{', '.join(competitors)}`. Now searching for candidates...\n"
                
                # Funnel competitors into 'required_industries' to leverage existing OR logic
                if "required_industries" not in criteria:
                    criteria["required_industries"] = {"operator": "OR", "values": []}
                
                existing_industries = set(criteria["required_industries"].get("values", []))
                competitor_set = set(competitors)
                # Ensure the original company is not included in the competitor search
                competitor_set.discard(company_to_find_competitors_for)
                criteria["required_industries"]["values"] = list(existing_industries.union(competitor_set))
                
                competitors_found = True
            else:
                yield f"The LLM could not identify any direct competitors for '{company_to_find_competitors_for}' from within the database list. "

        except Exception as e:
            logger.error(f"LLM error while finding competitors: {e}")
            yield "There was an issue analyzing competitors with the LLM. "

        # Fallback logic: if LLM fails, use the original company name as an industry search term
        if not competitors_found:
            yield "Falling back to use the original company name as a search keyword.\n"
            if "required_industries" not in criteria:
                criteria["required_industries"] = {"operator": "OR", "values": []}
            if company_to_find_competitors_for not in criteria["required_industries"]["values"]:
                 criteria["required_industries"]["values"].append(company_to_find_competitors_for)

        # Always remove the trigger key
        del criteria["competitors_of"]
    # --- END OF STEP ---

   
    keyword_expansion_prompt = PromptTemplate(
    input_variables=["keywords", "category"],
    template="""
    You are an expert business analyst. Your task is to generate a JSON list of 5-7 semantically similar keywords or synonyms for the initial keywords provided.
    The category is '{category}'.
    
    **CRITICAL RULE: You MUST NOT include any of the original 'Initial Keywords' in your final JSON list output.** Your goal is to find *alternatives* or *competitors*, not to repeat the input.

    Initial Keywords: {keywords}
    
    JSON List of Alternatives/Synonyms (excluding initial keywords):
    """
)
    location_expansion_prompt = PromptTemplate(
        input_variables=["locations"],
        template="""
        You are a geography expert. For the given list of countries or regions, generate a JSON list containing the original names plus up to 5 major cities or states within them to improve search recall.

        For example, if the input is ["USA"], the output should be a JSON list like: ["USA", "United States", "New York", "California", "Texas", "Illinois"].
        If the input is ["Malaysia"], the output should be a JSON list like: ["Malaysia", "Kuala Lumpur", "Penang", "Johor Bahru", "Selangor"].

        Initial Locations: {locations}

        JSON List:
        """
    )
    try:
        yield "Expanding keywords... "
        company_keywords = criteria.pop("required_companies", [])

        def get_values_from_criteria(crit_val):
            if isinstance(crit_val, dict): return crit_val.get("values", [])
            if isinstance(crit_val, list): return crit_val
            return []

        # Expand industries
        if criteria.get("required_industries"):
            industry_keywords = get_values_from_criteria(criteria["required_industries"])
            if industry_keywords:
                industry_keywords_response = await llm.ainvoke(keyword_expansion_prompt.format(keywords=industry_keywords, category="Industry"))
                expanded_industries = safe_json_loads(industry_keywords_response.content, [])
                industry_keywords.extend(expanded_industries)
                if isinstance(criteria["required_industries"], dict):
                    criteria["required_industries"]["values"] = list(set(industry_keywords))

        # Expand functions with a fallback for unknown terms
        if criteria.get("required_functions"):
            function_keywords = get_values_from_criteria(criteria["required_functions"])
            if function_keywords:
                expanded_functions = []
                unknown_functions = []

                for func in function_keywords:
                    if func in SALES_TAXONOMY:
                        expanded_functions.extend(SALES_TAXONOMY.get(func, [func]))
                    else:
                        unknown_functions.append(func)

                if unknown_functions:
                    logger.info(f"Found unknown function terms, expanding them on the fly: {unknown_functions}")
                    unknown_functions_response = await llm.ainvoke(keyword_expansion_prompt.format(
                        keywords=unknown_functions,
                        category="Sales Job Titles"
                    ))
                    expanded_unknown = safe_json_loads(unknown_functions_response.content, [])
                    expanded_functions.extend(unknown_functions)
                    expanded_functions.extend(expanded_unknown)

                if isinstance(criteria["required_functions"], dict):
                    all_funcs = list(set(expanded_functions))
                    criteria["required_functions"]["values"] = all_funcs

        # Expand segments with a fallback for unknown terms
        if criteria.get("required_segments"):
            segment_keywords = get_values_from_criteria(criteria["required_segments"])
            if segment_keywords:
                expanded_segments = []
                unknown_segments = []

                for seg in segment_keywords:
                    if seg in SEGMENT_SYNONYMS:
                        expanded_segments.extend(SEGMENT_SYNONYMS.get(seg, [seg]))
                    else:
                        unknown_segments.append(seg)

                if unknown_segments:
                    logger.info(f"Found unknown segment terms, expanding them on the fly: {unknown_segments}")
                    unknown_segments_response = await llm.ainvoke(keyword_expansion_prompt.format(
                        keywords=unknown_segments,
                        category="Customer Segments"
                    ))
                    expanded_unknown = safe_json_loads(unknown_segments_response.content, [])
                    expanded_segments.extend(unknown_segments)
                    expanded_segments.extend(expanded_unknown)

                if isinstance(criteria["required_segments"], dict):
                    all_segs = list(set(expanded_segments))
                    criteria["required_segments"]["values"] = all_segs

        # Expand company details with a fallback for unknown terms
        if criteria.get("required_company_details"):
            cd_keywords = get_values_from_criteria(criteria["required_company_details"])
            if cd_keywords:
                expanded_details = []
                unknown_details = []

                for detail in cd_keywords:
                    if detail in COMPANY_DETAILS_TAXONOMY:
                        expanded_details.extend(COMPANY_DETAILS_TAXONOMY.get(detail, [detail]))
                    else:
                        unknown_details.append(detail)
                
                if unknown_details:
                    logger.info(f"Found unknown company detail terms, expanding them on the fly: {unknown_details}")
                    unknown_details_response = await llm.ainvoke(keyword_expansion_prompt.format(
                        keywords=unknown_details,
                        category="Company Attributes (e.g., funding, business model)"
                    ))
                    expanded_unknown = safe_json_loads(unknown_details_response.content, [])
                    expanded_details.extend(unknown_details)
                    expanded_details.extend(expanded_unknown)

                if isinstance(criteria["required_company_details"], dict):
                    all_details = list(set(expanded_details))
                    criteria["required_company_details"]["values"] = all_details

        # --- NEW: Expand culture types ---
        if criteria.get("required_culture_type"):
            culture_keywords = get_values_from_criteria(criteria["required_culture_type"])
            if culture_keywords:
                expanded_cultures = []
                unknown_cultures = []
                for culture in culture_keywords:
                    if culture in CULTURE_TAXONOMY:
                        expanded_cultures.extend(CULTURE_TAXONOMY.get(culture, [culture]))
                    else:
                        unknown_cultures.append(culture)
                
                if unknown_cultures:
                    logger.info(f"Found unknown culture terms, expanding them on the fly: {unknown_cultures}")
                    unknown_cultures_response = await llm.ainvoke(keyword_expansion_prompt.format(
                        keywords=unknown_cultures,
                        category="Company Culture (e.g., startup, corporate)"
                    ))
                    expanded_unknown = safe_json_loads(unknown_cultures_response.content, [])
                    expanded_cultures.extend(unknown_cultures)
                    expanded_cultures.extend(expanded_unknown)

                if isinstance(criteria["required_culture_type"], dict):
                    all_cultures = list(set(expanded_cultures))
                    criteria["required_culture_type"]["values"] = all_cultures


        # Expand locations
        if criteria.get("required_locations"):
            locations_to_expand = criteria["required_locations"]
            if locations_to_expand:
                location_response = await llm.ainvoke(location_expansion_prompt.format(locations=json.dumps(locations_to_expand)))
                expanded_locations = safe_json_loads(location_response.content, [])
                criteria["required_locations"] = list(set(locations_to_expand + expanded_locations))

        # Centralized cleanup logic for all expandable fields
        for key in ["required_industries", "required_functions", "required_geographies", "required_segments", "required_company_details", "required_culture_type"]:
            if key in criteria and isinstance(criteria[key], dict) and "values" in criteria[key]:
                original_values = criteria[key]["values"]
                # Clean the "keywords" literal and any empty strings
                cleaned_values = [v for v in original_values if v and v.lower() != 'keywords']
                criteria[key]["values"] = cleaned_values
        
        if company_keywords:
            criteria["required_companies"] = company_keywords

        logger.info(f"Full Criteria after expansion and filtering: {json.dumps(criteria)}")
        yield f"Full Criteria: `{json.dumps(criteria)}`\n"

    except Exception as e:
        logger.error(f"Error expanding keywords: {e}")

    # 3. Initial Semantic Search against PostgreSQL
    yield "Performing initial semantic search... "
    def get_values_from_criteria_for_search(crit_val):
        if isinstance(crit_val, dict): return crit_val.get("values", [])
        if isinstance(crit_val, list): return crit_val
        return []

    search_query_text = " ".join(
        (criteria.get("required_companies") or []) + 
        get_values_from_criteria_for_search(criteria.get("required_industries")) +
        get_values_from_criteria_for_search(criteria.get("required_functions")) +
        get_values_from_criteria_for_search(criteria.get("required_segments")) +
        get_values_from_criteria_for_search(criteria.get("required_geographies")) +
        get_values_from_criteria_for_search(criteria.get("required_company_details")) +
        get_values_from_criteria_for_search(criteria.get("required_culture_type")) # --- NEW: Added culture to search ---
    )


    # A query is only too broad if it has no semantic keywords AND no hard filters at all.
    hard_filters_present = (
        criteria.get("required_locations") or
        criteria.get("min_people_managed") is not None or
        criteria.get("min_total_experience") is not None or
        criteria.get("required_companies")
    )
    if not search_query_text and not hard_filters_present:
        yield "Your query is too broad. Please specify industries, functions, segments, geographies, or locations."
        return

    # If there's semantic text, perform the search. Otherwise, we'll just filter the whole dataset.
    if search_query_text:
        query_embedding = embeddings.embed_query(search_query_text)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM candidates ORDER BY embedding <=> %s LIMIT 333",
            (str(query_embedding),)
        )
        initial_candidate_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not initial_candidate_ids:
            yield "No potential matches found in the initial search."
            return

        initial_candidate_pool = [PROFILES_BY_ID[id] for id in initial_candidate_ids if id in PROFILES_BY_ID]
        yield f"Found {len(initial_candidate_pool)} potential matches. "
    else:
        # If the search is only by hard filters, the initial pool is all candidates.
        initial_candidate_pool = list(PROFILES_BY_ID.values())


    # 4. Strict Python Filtering
    final_candidates = await filter_candidates_by_criteria(initial_candidate_pool, criteria)

    # 5. Generate Final Response
    async for token in generate_response_with_evidence(query, final_candidates, criteria):
        yield token

# --- Streamlit UI ---
st.set_page_config(page_title="Growton AI - Candidate Search", layout="wide")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 280px !important;  /* Adjust this value as needed for your preferred width */
        min-width: 250px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] > div {  /* Targets the inner content wrapper for better control */
        height: 100%;
        overflow-y: auto;  /* Adds scrolling if content overflows, preventing expansion */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 30px;">
        <img src="https://media.licdn.com/dms/image/v2/D560BAQF7O3De5SQ1vA/company-logo_200_200/company-logo_200_200/0/1708433749265/letsgrowton_logo?e=2147483647&v=beta&t=GerSYeinV4BZI9iFhaAo1dfHFDS1Ym5cwhYYwQXEWJo"
             style="width:50px;height:50px;">
        <h1 style="margin: 0; font-size: 2.2em;">Growton AI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar summary
st.sidebar.subheader("📊 Dataset Summary")
total_profiles = len(PROFILES_BY_ID)
total_exp = sum(p.get("total_experience_years") or 0 for p in PROFILES_BY_ID.values())
avg_experience = total_exp / total_profiles if total_profiles > 0 else 0
st.sidebar.markdown(f"**Total Profiles:** {total_profiles}")
st.sidebar.markdown(f"**Avg. Experience:** {round(avg_experience, 1)} years")

# Session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(os.urandom(32)).hexdigest()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.write_stream(process_query_main(prompt, st.session_state.session_id))


import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
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
import copy
import pandas as pd
import io

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Pricing and Token Configuration ---
# Prices are per 1 million tokens in USD (as of late 2024 for these models)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0} # Embeddings are priced on input
}
tokenizer = tiktoken.get_encoding("cl100k_base")

class TokenCostTracker:
    """A helper class to track token usage and associated costs."""
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.session_details = []

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates the cost for a given model and token counts."""
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def add_usage(self, model: str, input_text: str = "", output_text: str = "", usage_type: str = "LLM Call"):
        """Adds usage details and updates totals."""
        input_tokens = len(tokenizer.encode(input_text)) if input_text else 0
        output_tokens = len(tokenizer.encode(output_text)) if output_text else 0
        
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost
        
        self.session_details.append({
            "type": usage_type,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })

    def get_summary(self) -> str:
        """Returns a formatted markdown summary of the session's usage."""
        if self.total_tokens == 0:
            return ""

        summary_md = f"\n---\n\n**Session Usage Summary:**\n"
        summary_md += f"- **Total Tokens:** `{self.total_tokens}`\n"
        summary_md += f"- **Estimated Cost:** `${self.total_cost:.6f} USD`\n"
        
        return summary_md

# --- Database Configuration ---
# DB_NAME = "growton_ai"
# DB_USER = "postgres"
# DB_PASSWORD = "postgres"
# DB_HOST = "localhost"
# DB_PORT = "5433"
DB_NAME = "growton_ai"
DB_USER = "growton_ai_user"
DB_PASSWORD = "j8BpdJ42APcQPfQsuZMiBCoE7nxHNfOM"
DB_HOST = "dpg-d46agkchg0os73eev130-a.singapore-postgres.render.com"
DB_PORT = "5432"

# --- OpenAI and Redis Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

try:
    redis_client = redis.Redis(host='red-d46duqur433s73ckm440', port=6379, db=0, decode_responses=True)
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
        For example, if the key is 'Hunting', the values should include terms like 'new business development', 'net new logo acquisition', 'hunter', 'closer', 'Account Executive (New Business)', 'Sales Executive', etc.[MANDATORYstrea]

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

STATIC_CULTURE_TAXONOMY = {
    "startup": ["startup", "fast-paced", "agile environment", "high-growth", "early-stage"],
    "corporate": ["corporate", "mnc", "multinational", "large enterprise", "structured environment", "established company"],
    "remote": ["remote-first", "fully remote", "distributed team"]
}

# --- Geography Mapping for Region Expansion ---
STATIC_GEOGRAPHY_MAP = {
    # APAC
    "singapore": "apac", "malaysia": "apac", "indonesia": "apac", "thailand": "apac", "vietnam": "apac",
    "philippines": "apac", "australia": "apac", "new zealand": "apac", "japan": "apac", "south korea": "apac",
    "india": "apac", "hong kong": "apac",
    # EMEA
    "united kingdom": "emea", "uk": "emea", "germany": "emea", "france": "emea", "spain": "emea", "italy": "emea",
    "netherlands": "emea", "sweden": "emea", "norway": "emea", "denmark": "emea", "finland": "emea",
    "united arab emirates": "emea", "uae": "emea", "saudi arabia": "emea", "south africa": "emea", "israel": "emea",
    # Americas / NA / LATAM
    "united states": "americas", "usa": "americas", "us": "americas", "canada": "americas", "mexico": "americas",
    "brazil": "latam", "argentina": "latam", "colombia": "latam"
}


# --- Dynamic Taxonomy Initialization ---
SALES_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SALES_TAXONOMY,
    category="Sales Functions"
)

SEGMENT_SYNONYMS = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SEGMENT_SYNONYMS,
    category="Customer Seggents"
)

COMPANY_DETAILS_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_COMPANY_DETAILS_TAXONOMY,
    category="Company Attributes (Funding, Business Model)"
)

CULTURE_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_CULTURE_TAXONOMY,
    category="Company Culture Types"
)

@st.cache_data(ttl=86400) # Cache for 24 hours
def generate_dynamic_geography_map(seed_map: dict) -> dict:
    """
    Uses an LLM to expand a seed geography map with more countries and variations.
    """
    logger.info("Generating dynamic geography map... (This will be cached)")

    prompt_template = PromptTemplate(
        input_variables=["seed_map_json"],
        template="""
        You are an expert geographer. Your task is to expand the given seed JSON map of countries to regions (like apac, emea).
        Add many more countries for each region. The keys should be lowercase country names or common abbreviations, and values should be the lowercase region name (e.g., "apac", "emea", "latam", "americas").
        Ensure the final output is a single, flat, valid JSON object.

        **Seed Map:**
        {seed_map_json}

        **Expanded JSON Output:**
        """
    )
    try:
        formatted_prompt = prompt_template.format(seed_map_json=json.dumps(seed_map, indent=2))
        response = generation_llm.invoke(formatted_prompt)
        expanded_map = safe_json_loads(response.content, default_val=seed_map)

        if expanded_map == seed_map:
            logger.warning("LLM failed to generate an expanded geography map. Falling back to the static seed.")
            return seed_map

        logger.info("Successfully generated and cached dynamic geography map.")
        return expanded_map
    except Exception as e:
        logger.error(f"An error occurred during geography map generation: {e}")
        return seed_map

GEOGRAPHY_COUNTRY_TO_REGION_MAP = generate_dynamic_geography_map(STATIC_GEOGRAPHY_MAP)

logger.info(f"Loaded Sales Taxonomy with {sum(len(v) for v in SALES_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Segment Taxonomy with {sum(len(v) for v in SEGMENT_SYNONYMS.values())} total terms.")
logger.info(f"Loaded Company Details Taxonomy with {sum(len(v) for v in COMPANY_DETAILS_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Culture Taxonomy with {sum(len(v) for v in CULTURE_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Geography Map with {len(GEOGRAPHY_COUNTRY_TO_REGION_MAP)} total entries.")


# --- Database Connection Pool ---
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
    Loads all candidate profiles and their roles from the database.
    """
    logger.info("Loading all profiles from the database into cache...")
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, name, linkedin, location, headline, about, total_experience_years, max_people_managed FROM candidates")
    candidates_raw = cur.fetchall()

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
        candidate_id = role[0]
        if candidate_id not in roles_by_candidate:
            roles_by_candidate[candidate_id] = []

        company_details = {
            "funding_stage": role[5],
            "revenue": role[6],
            "business_model": role[7],
            "product_service": role[8],
            "customer_segment": role[9] if role[9] is not None else [],
            "customer_presence": role[10] if role[10] is not None else [],
            "culture_type": role[11],
            "headquarters": role[12],
            "industry": role[8] or ""
        }

        roles_by_candidate[candidate_id].append({
            "company": role[1],
            "title": role[2],
            "details": role[3],
            "duration_years": float(role[4]) if role[4] is not None else 0.0,
            "company_details": company_details
        })

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
                break
    return total_duration, contributing_roles

def calculate_geography_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculates the total duration for roles that meet the geography criteria.
    """
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []
    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    regions_for_req_values = {v: GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v) for v in req_values}
    regions_to_check = {region for region in regions_for_req_values.values() if region}

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"

        company_details = role.get('company_details', {})
        company_office_locations = company_details.get('customer_presence', [])
        company_locations_text = ' '.join([loc.lower() for loc in company_office_locations])
        company_hq_text = (company_details.get('headquarters') or '').lower()

        combined_search_text = f"{role_text} {company_locations_text} {company_hq_text}"

        direct_match = any(v in combined_search_text for v in req_values)
        region_match = any(r in combined_search_text for r in regions_to_check)

        if direct_match or region_match:
            duration = role.get('duration_years', 0.0) or 0.0
            if not any(cr['company'] == role.get('company', '') and cr['title'] == role.get('title', '') for cr in contributing_roles):
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


def calculate_company_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculates the total duration for roles at a specific company."""
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    company_name = criteria_obj.get("company_name")
    if not company_name:
        return 0.0, []
    
    company_name_lower = company_name.lower()

    for role in profile.get('roles', []):
        role_company = (role.get('company') or '').lower()
        
        # Using 'in' for flexibility (e.g., "HCL" vs "HCL Technologies Ltd")
        if company_name_lower in role_company:
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
    required_companies = criteria.get("required_companies")
    if not required_companies or not isinstance(required_companies, list):
        return True

    required_companies_lower = [c.lower() for c in required_companies]
    found_companies = set()

    for company_name in required_companies_lower:
        for role in profile.get('roles', []):
            if company_name in (role.get('company') or '').lower():
                found_companies.add(company_name)
                break

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
    """
    Checks if a candidate has any experience in the required geographies.
    """
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
    evidence_reasons = [] 

    for v in values:
        region_for_v = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v)

        profile_location_lower = (profile.get('location') or '').lower()
        if v in profile_location_lower:
            if v not in found_values:
                evidence_reasons.append(f"candidate's primary location is '{profile.get('location')}' which directly matches the search for '{v}'")
            found_values.add(v)
            continue 

        if region_for_v and region_for_v in profile_location_lower:
            if v not in found_values:
                evidence_reasons.append(f"candidate's primary location is '{profile.get('location')}' which contains the region '{region_for_v}', matching the search for '{v}'")
            found_values.add(v)
            continue

        is_found_in_roles = False
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            
            company_details = role.get('company_details', {})
            company_office_locations = company_details.get('customer_presence', [])
            company_locations_text = ' '.join([loc.lower() for loc in company_office_locations])
            company_hq_text = (company_details.get('headquarters') or '').lower()
            
            combined_search_text = f"{role_text} {company_locations_text} {company_hq_text}"
            
            if v in combined_search_text:
                if v not in found_values:
                    evidence_reasons.append(f"experience at '{role.get('company')}' shows presence in '{v}', as inferred from company/role details")
                found_values.add(v)
                is_found_in_roles = True
                break 

            if region_for_v and region_for_v in combined_search_text:
                if v not in found_values:
                     evidence_reasons.append(f"experience at '{role.get('company')}' implies '{region_for_v}' presence, matching the search for '{v}'")
                found_values.add(v)
                is_found_in_roles = True
                break
        
        if is_found_in_roles:
            continue

    is_met = False
    if op == "AND":
        is_met = found_values == set(values)
    else: # OR
        is_met = bool(found_values)

    if is_met:
        source_text = f"Profile confirms experience in {', '.join(found_values)}. Reasons: {'; '.join(evidence_reasons)}."
        profile['evidence_log'].append({
            "criterion": f"geography_presence ({op})",
            "source_text": source_text
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
            details_text = (
                f"{(company_details.get('funding_stage') or '').lower()} "
                f"{(company_details.get('business_model') or '').lower()} "
                f"{(company_details.get('product_service', '') or '').lower()}"
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

def check_excluded_geography_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """
    Checks if a candidate has experience in any of the excluded geographies.
    Returns True if there is NO match (i.e., the candidate passes the exclusion check).
    """
    criteria_obj = criteria.get("excluded_geographies")
    if not criteria_obj:
        return True

    values = []
    if isinstance(criteria_obj, dict):
        values = [v.lower() for v in criteria_obj.get("values", [])]
    elif isinstance(criteria_obj, list):
        values = [v.lower() for v in criteria_obj]

    if not values:
        return True

    profile_location = (profile.get('location') or '').lower()
    if any(v in profile_location for v in values):
        return False

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(v in role_text for v in values):
            return False

    return True

def check_tenure_in_latest_role(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks tenure in the most recent role against min, max, and exact requirements."""
    min_tenure = criteria.get("min_tenure_in_latest_role")
    exact_tenure = criteria.get("exact_tenure_in_latest_role")
    max_tenure = criteria.get("max_tenure_in_latest_role")

    if not min_tenure and not exact_tenure and not max_tenure:
        return True  # No tenure criteria for latest role

    roles = profile.get('roles', [])
    if not roles:
        return False  # No roles, so fails any check

    latest_role = roles[0]
    latest_role_duration = latest_role.get('duration_years', 0.0)

    if min_tenure and latest_role_duration < min_tenure:
        return False  # Fails min check

    if max_tenure and latest_role_duration > max_tenure:
        return False  # Fails max check

    if exact_tenure:
        # Use a small buffer (e.g., 0.1 years) for float comparison
        if not (exact_tenure - 0.1 <= latest_role_duration <= exact_tenure + 0.1):
            return False  # Fails exact check

    # If we passed all checks, log it and return True
    evidence_parts = []
    if min_tenure: evidence_parts.append(f"min {min_tenure} yrs")
    if exact_tenure: evidence_parts.append(f"exactly {exact_tenure} yrs")
    if max_tenure: evidence_parts.append(f"max {max_tenure} yrs")
    
    profile['evidence_log'].append({
        "criterion": "tenure_in_latest_role",
        "source_text": f"Candidate's latest role at {latest_role.get('company')} lasted {latest_role_duration:.1f} years, meeting the requirement ({', '.join(evidence_parts)})."
    })
    return True

def check_avg_tenure_in_last_n_roles(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate's average tenure in their last N roles meets the minimum requirement."""
    tenure_criteria = criteria.get("avg_tenure_in_last_n_roles")
    if not tenure_criteria or not isinstance(tenure_criteria, dict):
        return True
        
    avg_years = tenure_criteria.get("avg_years")
    num_roles = tenure_criteria.get("num_roles")
    
    if not avg_years or not num_roles:
        return True

    roles = profile.get('roles', [])
    if len(roles) < num_roles:
        return False

    last_n_roles = roles[:num_roles]
    total_duration = sum(role.get('duration_years', 0.0) for role in last_n_roles)
    calculated_avg = total_duration / num_roles
    
    is_met = calculated_avg >= avg_years
    if is_met:
        company_names = ", ".join([role.get('company', 'N/A') for role in last_n_roles])
        profile['evidence_log'].append({
            "criterion": "avg_tenure_in_last_n_roles",
            "source_text": f"Candidate's average tenure in their last {num_roles} roles ({company_names}) is {calculated_avg:.1f} years, meeting the minimum average of {avg_years} years."
        })
    return is_met

# --- Strict Filtering Function ---

async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs strict, deterministic filtering in Python and sorts by specified criterion duration."""
    logger.info("Applying detailed filters with reasoning...")
    matching_candidates = []

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
        sort_criterion = "required_functions"

    for profile in profiles:
        profile['evidence_log'] = []
        profile['contributing_roles_details'] = {}
        profile['calculated_experience'] = {}
        all_criteria_met = True

        # --- Total Experience Checks (Min, Max, Exact) ---
        total_exp = profile.get("total_experience_years") or 0.0
        
        min_total_exp = criteria.get("min_total_experience")
        if min_total_exp and total_exp < min_total_exp:
            all_criteria_met = False

        exact_total_exp = criteria.get("exact_total_experience")
        if all_criteria_met and exact_total_exp:
            # Use a small buffer (e.g., 0.1 years) for float comparison
            if not (exact_total_exp - 0.1 <= total_exp <= exact_total_exp + 0.1):
                all_criteria_met = False
        
        max_total_exp = criteria.get("max_total_experience")
        if all_criteria_met and max_total_exp and total_exp > max_total_exp:
            all_criteria_met = False
        # --- End Total Experience Checks ---

        min_managed = criteria.get("min_people_managed")
        if all_criteria_met and min_managed and (profile.get("max_people_managed") or 0) < min_managed:
            all_criteria_met = False

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
        if all_criteria_met and not check_company_culture_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_excluded_geography_presence(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_tenure_in_latest_role(profile, criteria):
            all_criteria_met = False
        if all_criteria_met and not check_avg_tenure_in_last_n_roles(profile, criteria):
            all_criteria_met = False
            
        if all_criteria_met:
            for key, calc_func in [
                ("required_functions", calculate_functional_experience_duration),
                ("required_industries", calculate_industry_experience_duration),
                ("required_segments", calculate_segment_experience_duration),
                ("required_geographies", calculate_geography_experience_duration),
                ("required_company_details", calculate_company_details_experience_duration)
            ]:
                crit_obj = criteria.get(key)
                if crit_obj and isinstance(crit_obj, dict):
                    duration, roles = calc_func(profile, crit_obj)
                    
                    # Store calculation regardless of filter, for sorting
                    profile['calculated_experience'][key] = {
                        "duration": duration,
                        "roles": roles,
                        "label": ", ".join(crit_obj.get("values",[])),
                        # Store all requirements for potential display/reasoning
                        "required_min": crit_obj.get("min_years"),
                        "required_exact": crit_obj.get("exact_years"),
                        "required_max": crit_obj.get("max_years"),
                    }

                    # Now, apply filters
                    min_y = crit_obj.get("min_years")
                    if min_y and duration < min_y:
                        all_criteria_met = False
                        break

                    exact_y = crit_obj.get("exact_years")
                    if all_criteria_met and exact_y:
                         # Use a small buffer (e.t., 0.1 years) for float comparison
                        if not (exact_y - 0.1 <= duration <= exact_y + 0.1):
                            all_criteria_met = False
                            break
                    
                    max_y = crit_obj.get("max_years")
                    if all_criteria_met and max_y and duration > max_y:
                        all_criteria_met = False
                        break

            # --- NEW: Handle Company-Specific Tenure ---
            if all_criteria_met: # Check again in case the loop above set it to False
                company_tenure_criteria = criteria.get("company_tenure")
                if company_tenure_criteria and isinstance(company_tenure_criteria, list):
                    for i, tenure_obj in enumerate(company_tenure_criteria):
                        if not isinstance(tenure_obj, dict) or not tenure_obj.get("company_name"):
                            continue
                        
                        duration, roles = calculate_company_experience_duration(profile, tenure_obj)
                        
                        # Use a unique key for calculated_experience
                        key = f"company_tenure_{i}"
                        profile['calculated_experience'][key] = {
                            "duration": duration,
                            "roles": roles,
                            "label": tenure_obj.get("company_name"),
                            "required_min": tenure_obj.get("min_years"),
                            "required_exact": tenure_obj.get("exact_years"),
                            "required_max": tenure_obj.get("max_years"),
                        }

                        # Apply filters
                        min_y = tenure_obj.get("min_years")
                        if min_y and duration < min_y:
                            all_criteria_met = False
                            break # Fails this company tenure obj

                        exact_y = tenure_obj.get("exact_years")
                        if all_criteria_met and exact_y:
                            if not (exact_y - 0.1 <= duration <= exact_y + 0.1):
                                all_criteria_met = False
                                break # Fails this company tenure obj

                        max_y = tenure_obj.get("max_years")
                        if all_criteria_met and max_y and duration > max_y:
                            all_criteria_met = False
                            break # Fails this company tenure obj
                    
                    if not all_criteria_met:
                        continue # Go to the next profile
            # --- END NEW BLOCK ---

        if all_criteria_met:
            if profile['calculated_experience']:
                first_calc_key = next(iter(profile['calculated_experience']))
                profile['contributing_roles_details'] = {'roles': profile['calculated_experience'][first_calc_key]['roles']}
            else:
                _, roles_list = calculate_functional_experience_duration(profile, criteria.get("required_functions", {}))
                profile['contributing_roles_details'] = {'roles': roles_list}

            matching_candidates.append(profile)

    if matching_candidates and sort_criterion:
        matching_candidates.sort(
            key=lambda x: x['calculated_experience'].get(sort_criterion, {}).get('duration', 0.0),
            reverse=True
        )

    top_n = criteria.get("top_n")
    if top_n is None or top_n == 0:
        logger.info(f"No top_n specified or top_n is 0, returning all {len(matching_candidates)} candidates.")
    else:
        matching_candidates = matching_candidates[:top_n]
        logger.info(f"Found {len(matching_candidates)} candidates after strict filtering and sorting by {sort_criterion} duration, limited to top {top_n}.")

    return matching_candidates

async def generate_reasoning_for_profile(profile: Dict[str, Any], original_criteria: Dict[str, Any], tracker: TokenCostTracker) -> str:
    """Generates the reasoning text for a single profile as a complete string."""
    prompt_template = PromptTemplate(
        input_variables=["original_criteria_json", "matching_profile_json"],
        template="""
        You are an expert recruitment analyst. Synthesize a concise, single-paragraph summary explaining why this candidate is a good match based on the original search criteria.

        **Original Filtering Criteria (JSON):** {original_criteria_json}
        **Matching Candidate (JSON):** {matching_profile_json}

        **Instructions:**
        - Use the `calculated_experience` and `evidence_log` from the candidate's JSON to find evidence.
        - For tenure criteria (like 'tenure_in_latest_role' or 'avg_tenure'), use the `source_text` from the `evidence_log`.
        - For other criteria (like 'required_functions' or 'company_tenure'), mention the duration from `calculated_experience`.
        - **DO NOT** mention static details like "Total Experience Years" or "Max People Managed" unless they were the *only* criteria.
        - The entire reasoning must be a single, flowing paragraph without bullet points or newlines.
        - Do not include the markdown pipe `|` characters. Just the text.

        **Reasoning Paragraph:**
        """
    )
    
    formatted_prompt = prompt_template.format(
        original_criteria_json=json.dumps(original_criteria, indent=2),
        matching_profile_json=json.dumps(profile, indent=2)
    )

    response = await specialist_llm.ainvoke(formatted_prompt)
    full_response_content = response.content.replace('\n', ' ').replace('|', '')
    
    tracker.add_usage(specialist_llm.model_name, formatted_prompt, response.content, "Reasoning Generation")
    return full_response_content


async def process_query_main(query: str, session_id: str, tracker: TokenCostTracker) -> AsyncIterator[Any]:
    """
    Main processing pipeline. Yields status messages and profile chunks.
    """

    def get_values_from_criteria(crit_val):
        values = []
        if isinstance(crit_val, dict):
            values = crit_val.get("values", [])
        elif isinstance(crit_val, list):
            values = crit_val

        flat_values = []
        for item in values:
            if isinstance(item, str):
                flat_values.append(item)
            elif isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, str):
                        flat_values.append(sub_item)
        return flat_values

    def get_list_from_llm_json(llm_json_response: Any) -> List[str]:
        """Robustly extracts a list of strings from a JSON object that could be a list or a dict."""
        if isinstance(llm_json_response, list):
            return [str(item) for item in llm_json_response if isinstance(item, str)]
        if isinstance(llm_json_response, dict):
            for value in llm_json_response.values():
                if isinstance(value, list):
                    return [str(item) for item in value if isinstance(item, str)]
        return []

    normalized_query = normalize_query_with_llm(query)

    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "sales_taxonomy_json", "segment_taxonomy_json", "company_details_taxonomy_json", "culture_taxonomy_json"],
        template="""
You are an expert assistant tasked with extracting structured filtering criteria from a user's query for a candidate search system. Your goal is to categorize user intent into functions, segments, industries, etc., and correctly associate any specified durations or tenure requirements.

**DEFINITIONS, TAXONOMIES & CANONICAL KEYS:**
- `required_companies`: List of specific company names. **Use this for simple presence checks only.**
- `company_tenure`: A list of objects for company-specific duration checks. Each object has "company_name" and duration keys (e.g., "min_years", "exact_years").
- `required_functions`: Sales roles. **MUST** map to a key from the sales taxonomy.
  - **Sales Taxonomy:** {sales_taxonomy_json}
- `required_segments`: Customer types. **MUST** map to a key from the segment taxonomy.
  - **Segment Taxonomy:** {segment_taxonomy_json}
- `required_company_details`: Company attributes. **MUST** map to a key from the company details taxonomy.
  - **Company Details Taxonomy:** {company_details_taxonomy_json}
- `required_culture_type`: Company environment. **MUST** map to a key from the culture taxonomy.
  - **Culture Taxonomy:** {culture_taxonomy_json}
- `required_industries`: Broad industries (e.g., "SaaS", "Fintech").
- `competitors_of`: A list of companies for which to find competitors.
- `required_geographies`: Regions of sales experience (e.g., "APAC", "EMEA", "India").
- `excluded_geographies`: Regions or countries to specifically exclude.
- `required_locations`: Candidate's physical base.
- `top_n`: Integer for the number of candidates to return.
- `avg_tenure_in_last_n_roles`: An object with `avg_years` and `num_roles` for average tenure calculations.

**NEW DURATION & TENURE KEYS:**
- `min_total_experience`, `exact_total_experience`, `max_total_experience` (float)
- `min_tenure_in_latest_role`, `exact_tenure_in_latest_role`, `max_tenure_in_latest_role` (float)

**NEW LOCATION RULE:**
- Queries like "Candidates in [Location]", "Find people in [Location]", or "[Job Title] in [Location]" **MUST** be mapped to `required_locations`.
- `required_geographies` is **ONLY** for experience-based queries, such as "experience in [Region]", "sold into [Region]", or "managed [Region]".

**JSON STRUCTURE & DURATION RULES:**
- For inclusion criteria (required_*), use an object with "operator" ("AND"/"OR") and "values".
- **Exception**: `required_companies` (for presence check) can be a simple list of strings.
- **Duration Rule:** If a duration is mentioned with a function, industry, or segment, capture it appropriately inside that criterion's object:
    - 'at least 5 years', '5+ years' -> `"min_years": 5.0`
    - 'exactly 5 years', '5 years' -> `"exact_years": 5.0`
    - 'at most 5 years', 'up to 5 years' -> `"max_years": 5.0`
- **Company Tenure Rule:** If a duration is linked to a *specific company*, use the `company_tenure` list.
- **Tenure Rule:** Capture specific tenure requests using the new tenure keys (e.g., `exact_tenure_in_latest_role`).

**EXAMPLES TABLE (Follow this logic exactly):**
| User Query                                                    | Correct JSON Output                                                                                                                                                             |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "12 years as an Account Executive"                            | `{{"required_functions": {{"operator": "OR", "values": ["Hunting"], "min_years": 12.0}}}}`                                                                                       |
| "exactly 10 years total experience"                           | `{{"exact_total_experience": 10.0}}`                                                                                                                                            |
| "max 15 years total experience"                               | `{{"max_total_experience": 15.0}}`                                                                                                                                              |
| "exactly 3 years in SaaS"                                     | `{{"required_industries": {{"operator": "OR", "values": ["SaaS"], "exact_years": 3.0}}}}`                                                                                       |
| "worked in recent company for exactly 2 yrs"                  | `{{"exact_tenure_in_latest_role": 2.0}}`                                                                                                                                        |
| "at most 4 years in their last role"                          | `{{"max_tenure_in_latest_role": 4.0}}`                                                                                                                                          |
| "exactly 3 yrs of exp in HCL Technologies Ltd"                | `{{"company_tenure": [{{"company_name": "HCL Technologies Ltd", "exact_years": 3.0}}]}}`                                                                                        |
| "at least 5 years at Google"                                  | `{{"company_tenure": [{{"company_name": "Google", "min_years": 5.0}}]}}`                                                                                                        |
| "avg work exp of 3 yrs in last 2 companies"                   | `{{"avg_tenure_in_last_n_roles": {{"avg_years": 3.0, "num_roles": 2}}}}`                                                                                                        |
| "inside sales with 5 years exp and avg tenure of 2y in last 3 roles" | `{{"required_functions": {{"operator": "OR", "values": ["Sales Development"], "min_years": 5.0}}, "avg_tenure_in_last_n_roles": {{"avg_years": 2.0, "num_roles": 3}}}}` |
| "Candidates in Florida"                                       | `{{"required_locations": {{"operator": "OR", "values": ["Florida"]}}}}`                                                                                                         |
| "Find people in the US"                                       | `{{"required_locations": {{"operator": "OR", "values": ["US"]}}}}`                                                                                                               |
| "Sales leaders with APAC experience"                          | `{{"required_functions": {{"operator": "OR", "values": ["Sales Development"]}}, "required_geographies": {{"operator": "OR", "values": ["APAC"]}}}}`                             |


**Available criteria keys:**
- `min_total_experience`, `exact_total_experience`, `max_total_experience` (float)
- `min_people_managed` (integer)
- `required_locations` (object)
- `required_geographies` (object)
- `excluded_geographies` (object)
- `required_companies` (list of strings)
- `company_tenure` (list of objects)
- `required_industries` (object)
- `required_functions` (object)
- `required_segments` (object)
- `required_company_details` (object)
- `required_culture_type` (object)
- `competitors_of` (list of strings)
- `top_n` (integer)
- `min_tenure_in_latest_role`, `exact_tenure_in_latest_role`, `max_tenure_in_latest_role` (float)
- `avg_tenure_in_last_n_roles` (object with "avg_years" and "num_roles")

**CRITICAL INSTRUCTION**: Map user terms to their canonical keys using the provided taxonomies.

**User Query:** {query}

**JSON Criteria:**
"""
    )
    try:
        yield "Extracting criteria..."
        prompt_text = criteria_extraction_prompt.format(
                query=normalized_query,
                sales_taxonomy_json=json.dumps(SALES_TAXONOMY, indent=2),
                segment_taxonomy_json=json.dumps(SEGMENT_SYNONYMS, indent=2),
                company_details_taxonomy_json=json.dumps(COMPANY_DETAILS_TAXONOMY, indent=2),
                culture_taxonomy_json=json.dumps(CULTURE_TAXONOMY, indent=2)
            )
        criteria_response = await llm.ainvoke(prompt_text)
        tracker.add_usage(llm.model_name, prompt_text, criteria_response.content, "Criteria Extraction")
        
        criteria = safe_json_loads(criteria_response.content, {})
        original_criteria = copy.deepcopy(criteria)

        if not criteria and normalized_query:
            logger.warning(f"LLM failed to extract structured criteria. Treating '{normalized_query}' as a general keyword search.")
            if "worked at" in normalized_query or "worked in" in normalized_query or "from" in normalized_query:
                company_name = normalized_query.split(" in ")[-1].split(" at ")[-1].split(" from ")[-1].strip()
                criteria["required_companies"] = [company_name]
            else:
                criteria["required_industries"] = {"operator": "OR", "values": [normalized_query]}

        if not criteria:
            raise ValueError("Failed to parse criteria from query.")

        logger.info(f"Raw LLM criteria response: {criteria_response.content}")

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

    final_competitors_list = []
    if "competitors_of" in criteria and criteria.get("competitors_of"):
        company_to_find_competitors_for = criteria["competitors_of"][0]
        
        competitor_task = "identify all direct competitors for the given company"
        if "top" in normalized_query.lower() and criteria.get("top_n"):
             competitor_task = f"identify the top {criteria['top_n']} direct competitors for the given company"

        yield f"Identifying competitors for **{company_to_find_competitors_for}** using AI Brainstorm + DB Validation..."

        competitors_found = False
        try:
            brainstorm_prompt = PromptTemplate(
                input_variables=["company_name", "competitor_task"],
                template="""
                You are an expert business analyst with deep knowledge of corporate landscapes. Your task is to {competitor_task}.
                When asked for 'all' competitors, you should provide a comprehensive and extensive list, including both major and niche competitors. Do not limit the list to only the most obvious ones.

                Provide the output as a JSON-formatted list of company names.

                **Target Company:**
                {company_name}
                
                **JSON List of Competitors:**
                """
            )
            formatted_prompt = brainstorm_prompt.format(
                company_name=company_to_find_competitors_for,
                competitor_task=competitor_task
            )
            response = await specialist_llm.ainvoke(formatted_prompt)
            tracker.add_usage(specialist_llm.model_name, formatted_prompt, response.content, "Competitor ID")
            llm_competitors = safe_json_loads(response.content, [])

            if llm_competitors:
                db_company_names_lower = {name.lower() for name in ALL_COMPANY_NAMES}
                validated_competitors = []
                for competitor in llm_competitors:
                    if competitor.lower() in db_company_names_lower:
                        validated_competitors.append(competitor)
                
                if validated_competitors:
                    target_company_lower = company_to_find_competitors_for.lower()
                    final_competitors_list = [c for c in validated_competitors if target_company_lower not in c.lower()]
                    
                    if final_competitors_list:
                        competitors_found = True

        except Exception as e:
            logger.error(f"Error during competitor identification: {e}")
            yield "There was an issue identifying competitors. "
        
        if not competitors_found:
            yield "Could not identify any valid competitors from the database. Halting search."
            return

        logger.info(f"Found and verified competitors: {', '.join(final_competitors_list)}")
        yield f"**Found Competitors:** `{', '.join(final_competitors_list)}`\n\nNow searching for candidates from these companies...\n"
        
        if "required_industries" not in criteria:
            criteria["required_industries"] = {"operator": "OR", "values": []}
        
        existing_industries = set(get_values_from_criteria(criteria["required_industries"]))
        competitor_set = set(final_competitors_list)
        criteria["required_industries"]["values"] = list(existing_industries.union(competitor_set))

        del criteria["competitors_of"]
   
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
        You are a geography expert. For the given list of countries, states, or regions, generate a JSON list containing the original names plus up to 5 major cities AND common abbreviations (like state codes or "USA").

        For example:
        - If the input is ["USA"], the output should be a JSON list like: ["USA", "US", "United States", "New York", "California", "Texas"].
        - If the input is ["Malaysia"], the output should be a JSON list like: ["Malaysia", "MY", "Kuala Lumpur", "Penang", "Johor Bahru"].
        - If the input is ["Florida"], the output should be a JSON list like: ["Florida", "FL", "Miami", "Orlando", "Tampa"].

        Initial Locations: {locations}

        JSON List:
        """
    )
    geography_expansion_prompt = PromptTemplate(
        input_variables=["geographies"],
        template="""
        You are a geography and business market expert. For the given list of business regions (like APAC, EMEA, NA), expand them into a comprehensive JSON object. The JSON object must have a single key, "geographies", with a value that is a list of constituent countries and major business hubs.

        **SPECIAL RULE**: When expanding "APAC", you MUST exclude "China" from the list of countries, as China experience is typically specified separately.

        For example:
        - If the input is ["APAC"], the output should be: {{"geographies": ["APAC", "Asia Pacific", "India", "Japan", "Australia", "Singapore", "Malaysia", "Indonesia"]}}
        - If the input is ["India"], the output should be: {{"geographies": ["India", "Mumbai", "Bangalore", "Delhi", "Pune"]}}

        The goal is to maximize search recall.

        Initial Geographies: {geographies}

        JSON Output:
        """
    )
    try:
        company_keywords = criteria.pop("required_companies", [])
        competitor_search_was_run = bool(final_competitors_list)

        if not competitor_search_was_run and criteria.get("required_industries"):
            yield "Expanding keywords..."
            industry_keywords = get_values_from_criteria(criteria["required_industries"])
            if industry_keywords:
                prompt_text = keyword_expansion_prompt.format(keywords=industry_keywords, category="Industry")
                industry_keywords_response = await llm.ainvoke(prompt_text)
                tracker.add_usage(llm.model_name, prompt_text, industry_keywords_response.content, "Keyword Expansion")
                expanded_industries_raw = safe_json_loads(industry_keywords_response.content, [])
                expanded_industries = get_list_from_llm_json(expanded_industries_raw)
                industry_keywords.extend(expanded_industries)
                if isinstance(criteria["required_industries"], dict):
                    criteria["required_industries"]["values"] = list(set(industry_keywords))

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
                    prompt_text = keyword_expansion_prompt.format(keywords=unknown_functions, category="Sales Job Titles")
                    unknown_functions_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, unknown_functions_response.content, "Keyword Expansion")
                    expanded_unknown_raw = safe_json_loads(unknown_functions_response.content, [])
                    expanded_unknown = get_list_from_llm_json(expanded_unknown_raw)
                    expanded_functions.extend(unknown_functions)
                    expanded_functions.extend(expanded_unknown)

                if isinstance(criteria["required_functions"], dict):
                    all_funcs = list(set(expanded_functions))
                    criteria["required_functions"]["values"] = all_funcs

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
                    prompt_text = keyword_expansion_prompt.format(keywords=unknown_segments, category="Customer Segments")
                    unknown_segments_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, unknown_segments_response.content, "Keyword Expansion")
                    expanded_unknown_raw = safe_json_loads(unknown_segments_response.content, [])
                    expanded_unknown = get_list_from_llm_json(expanded_unknown_raw)
                    expanded_segments.extend(unknown_segments)
                    expanded_segments.extend(expanded_unknown)

                if isinstance(criteria["required_segments"], dict):
                    all_segs = list(set(expanded_segments))
                    criteria["required_segments"]["values"] = all_segs

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
                    prompt_text = keyword_expansion_prompt.format(keywords=unknown_details, category="Company Attributes (e.g., funding, business model)")
                    unknown_details_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, unknown_details_response.content, "Keyword Expansion")
                    expanded_unknown_raw = safe_json_loads(unknown_details_response.content, [])
                    expanded_unknown = get_list_from_llm_json(expanded_unknown_raw)
                    expanded_details.extend(unknown_details)
                    expanded_details.extend(expanded_unknown)

                if isinstance(criteria["required_company_details"], dict):
                    all_details = list(set(expanded_details))
                    criteria["required_company_details"]["values"] = all_details

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
                    prompt_text = keyword_expansion_prompt.format(keywords=unknown_cultures, category="Company Culture (e.g., startup, corporate)")
                    unknown_cultures_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, unknown_cultures_response.content, "Keyword Expansion")
                    expanded_unknown_raw = safe_json_loads(unknown_cultures_response.content, [])
                    expanded_unknown = get_list_from_llm_json(expanded_unknown_raw)
                    expanded_cultures.extend(unknown_cultures)
                    expanded_cultures.extend(expanded_unknown)

                if isinstance(criteria["required_culture_type"], dict):
                    all_cultures = list(set(expanded_cultures))
                    criteria["required_culture_type"]["values"] = all_cultures
        
        if criteria.get("required_geographies"):
            geographies_to_expand_obj = criteria["required_geographies"]
            if geographies_to_expand_obj and isinstance(geographies_to_expand_obj, dict):
                geographies_to_expand = geographies_to_expand_obj.get("values", [])
                if geographies_to_expand and isinstance(geographies_to_expand, list):
                    yield "Expanding geographies..."
                    prompt_text = geography_expansion_prompt.format(geographies=json.dumps(geographies_to_expand))
                    geography_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, geography_response.content, "Geography Expansion")
                    logger.info(f"Raw geography expansion response from LLM: {geography_response.content}")
                    expanded_geographies_raw = safe_json_loads(geography_response.content, {})
                    expanded_geographies = get_list_from_llm_json(expanded_geographies_raw)
                    criteria["required_geographies"]["values"] = list(set(geographies_to_expand + expanded_geographies))

        if criteria.get("required_locations"):
            locations_to_expand_obj = criteria["required_locations"]
            locations_to_expand = []
            if isinstance(locations_to_expand_obj, dict):
                locations_to_expand = locations_to_expand_obj.get("values", [])
            elif isinstance(locations_to_expand_obj, list):
                locations_to_expand = locations_to_expand_obj

            if locations_to_expand:
                yield "Expanding locations..."
                prompt_text = location_expansion_prompt.format(locations=json.dumps(locations_to_expand))
                location_response = await llm.ainvoke(prompt_text)
                tracker.add_usage(llm.model_name, prompt_text, location_response.content, "Location Expansion")
                logger.info(f"Raw location expansion response from LLM: {location_response.content}")
                expanded_locations_raw = safe_json_loads(location_response.content, [])
                expanded_locations = get_list_from_llllm_json(expanded_locations_raw)
                
                all_locations = list(set(locations_to_expand + expanded_locations))
                
                if isinstance(criteria["required_locations"], dict):
                    criteria["required_locations"]["values"] = all_locations
                elif isinstance(criteria["required_locations"], list):
                     criteria["required_locations"] = {"operator": "OR", "values": all_locations} # Normalize to object

        for key in ["required_industries", "required_functions", "required_geographies", "required_segments", "required_company_details", "required_culture_type", "excluded_geographies"]:
            if key in criteria and isinstance(criteria[key], dict) and "values" in criteria[key]:
                original_values = criteria[key]["values"]
                cleaned_values = [v for v in original_values if v and v.lower() != 'keywords']
                criteria[key]["values"] = cleaned_values
        
        if company_keywords:
            criteria["required_companies"] = company_keywords

        logger.info(f"Full Criteria after expansion and filtering: {json.dumps(criteria)}")

    except Exception as e:
        logger.error(f"Error expanding keywords: {e}")

    yield "Performing initial semantic search..."
    
    # --- FIX 1: Add company_tenure names to search_query_text ---
    company_tenure_names = []
    if criteria.get("company_tenure"):
        for tenure_obj in criteria.get("company_tenure", []):
            if isinstance(tenure_obj, dict) and tenure_obj.get("company_name"):
                company_tenure_names.append(tenure_obj["company_name"])

    search_query_text = " ".join(
        (criteria.get("required_companies") or []) + 
        company_tenure_names + # <-- ADDED
        get_values_from_criteria(criteria.get("required_industries")) +
        get_values_from_criteria(criteria.get("required_functions")) +
        get_values_from_criteria(criteria.get("required_segments")) +
        get_values_from_criteria(criteria.get("required_geographies")) +
        get_values_from_criteria(criteria.get("required_company_details")) +
        get_values_from_criteria(criteria.get("required_culture_type"))
    )

    # --- FIX 2: Add company_tenure to hard_filters_present check ---
    hard_filters_present = (
        criteria.get("required_locations") or
        criteria.get("min_people_managed") is not None or
        criteria.get("min_total_experience") is not None or
        criteria.get("exact_total_experience") is not None or
        criteria.get("max_total_experience") is not None or
        criteria.get("required_companies") or
        criteria.get("company_tenure") # <-- ADDED
    )
    
    if not search_query_text and not hard_filters_present:
        yield "Your query is too broad. Please specify industries, functions, segments, geographies, or locations."
        return

    if search_query_text:
        query_embedding = embeddings.embed_query(search_query_text)
        tracker.add_usage(embeddings.model, search_query_text, usage_type="Embedding")
        
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
        # This branch will now be correctly reached if only hard filters (like company_tenure) are present
        initial_candidate_pool = list(PROFILES_BY_ID.values())


    final_candidates = await filter_candidates_by_criteria(initial_candidate_pool, criteria)
    
    if not final_candidates:
        # If no candidates, just send the summary and stop
        yield tracker.get_summary()
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    # 1. Send a "progress_start" message so the UI can draw a progress bar
    yield {"type": "progress_start", "total": len(final_candidates)}
    
    # --- NEW: Parallel Processing with asyncio.Semaphore and as_completed ---
    CONCURRENCY_LIMIT = 10
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def get_reasoning_with_semaphore(profile):
        """Wrapper to apply semaphore to the reasoning generation."""
        async with semaphore:
            if st.session_state.get("stop_signal", False):
                return None  # Stop processing this profile
            reasoning = await generate_reasoning_for_profile(profile, original_criteria, tracker)
            profile['reasoning'] = reasoning
            return profile

    tasks = [get_reasoning_with_semaphore(profile) for profile in final_candidates]
    
    processed_candidates = []
    
    for i, future in enumerate(asyncio.as_completed(tasks)):
        if st.session_state.get("stop_signal", False):
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            yield "\n\nGeneration stopped by user."
            break

        try:
            processed_profile = await future
            if processed_profile:  # Will be None if it was stopped before starting
                processed_candidates.append(processed_profile)
                yield {
                    "type": "profile_chunk",
                    "data": processed_profile,
                    "current": len(processed_candidates),
                    "total": len(final_candidates)
                }
        except asyncio.CancelledError:
            logger.info("A reasoning task was cancelled.")
            continue

    # Re-sort the results to match the original ranking from filtering
    original_order_map = {p['id']: i for i, p in enumerate(final_candidates)}
    processed_candidates.sort(key=lambda p: original_order_map.get(p['id'], float('inf')))

    # 4. After the loop finishes (or is stopped), send the final 'complete' message
    yield {"type": "complete", "data": processed_candidates, "summary": tracker.get_summary()}
    
    # --- NEW: Parallel Processing with asyncio.Semaphore and as_completed ---
    CONCURRENCY_LIMIT = 10
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def get_reasoning_with_semaphore(profile):
        """Wrapper to apply semaphore to the reasoning generation."""
        async with semaphore:
            if st.session_state.get("stop_signal", False):
                return None  # Stop processing this profile
            reasoning = await generate_reasoning_for_profile(profile, original_criteria, tracker)
            profile['reasoning'] = reasoning
            return profile

    tasks = [get_reasoning_with_semaphore(profile) for profile in final_candidates]
    
    processed_candidates = []
    
    for i, future in enumerate(asyncio.as_completed(tasks)):
        if st.session_state.get("stop_signal", False):
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            yield "\n\nGeneration stopped by user."
            break

        try:
            processed_profile = await future
            if processed_profile:  # Will be None if it was stopped before starting
                processed_candidates.append(processed_profile)
                yield {
                    "type": "profile_chunk",
                    "data": processed_profile,
                    "current": len(processed_candidates),
                    "total": len(final_candidates)
                }
        except asyncio.CancelledError:
            logger.info("A reasoning task was cancelled.")
            continue

    # Re-sort the results to match the original ranking from filtering
    original_order_map = {p['id']: i for i, p in enumerate(final_candidates)}
    processed_candidates.sort(key=lambda p: original_order_map.get(p['id'], float('inf')))

    # 4. After the loop finishes (or is stopped), send the final 'complete' message
    yield {"type": "complete", "data": processed_candidates, "summary": tracker.get_summary()}
# --- Excel Export Helper ---
def profiles_to_excel(profiles_dict: Dict[str, Any]) -> bytes:
    """Converts a dictionary of selected profiles to an Excel file in memory."""
    if not profiles_dict:
        return b""
    
    profiles_list = list(profiles_dict.values())
    flat_data = []
    for p in profiles_list:
        # Simple role summarization
        roles_summary = []
        for role in p.get('roles', []):
            company = role.get('company', 'N/A')
            title = role.get('title', 'N/A')
            duration = role.get('duration_years', 0)
            roles_summary.append(f"{title} at {company} ({duration:.1f} yrs)")

        flat_data.append({
            "Name": p.get("name"),
            "LinkedIn": p.get("linkedin"),
            "Location": p.get("location"),
            "Relevance Summary": p.get("reasoning", "N/A"),
            "Total Experience (Yrs)": p.get("total_experience_years"),
            "Max People Managed": p.get("max_people_managed"),
            "Roles": " | ".join(roles_summary)
        })
    
    df = pd.DataFrame(flat_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Selected Profiles')
        # Auto-adjust column widths
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Selected Profiles'].column_dimensions[chr(65 + col_idx)].width = column_length + 2

    return output.getvalue()

# --- Enhanced UI Helper Function ---
def display_profile_with_checkbox(profile: Dict[str, Any], container):
    """Helper to render a single profile with modern card design and enhanced features."""
    is_selected = profile['id'] in st.session_state.selected_profiles
    
    # Modern Card Design
    card_class = "candidate-card selected" if is_selected else "candidate-card"
    
    with container:
        # Professional card layout with selection highlighting
        if is_selected:
            st.markdown(
                """
                <div style="border: 2px solid #1a1a1a; border-radius: 8px; padding: 1rem; background: #f8f9fa; margin-bottom: 1rem;">
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background: white; margin-bottom: 1rem;">
                """,
                unsafe_allow_html=True
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {profile.get('name', 'N/A')}")
            st.markdown(f"**Location:** {profile.get('location', 'N/A')}")
            st.markdown(f"**Experience:** {profile.get('total_experience_years', 0):.1f} years")
            st.markdown(f"**Managed:** {profile.get('max_people_managed', 0)} people")
            
            # AI Analysis
            with st.expander("AI Match Analysis", expanded=False):
                st.markdown(profile.get('reasoning', 'Analysis not available.'))
        
        with col2:
            # Selection button
            if st.button("Select" if not is_selected else "Selected", 
                        key=f"select_{profile['id']}", 
                        type="primary" if is_selected else "secondary",
                        use_container_width=True):
                if profile['id'] not in st.session_state.selected_profiles:
                    st.session_state.selected_profiles[profile['id']] = profile
                    st.success(f"Selected {profile.get('name', 'N/A')}")
                else:
                    del st.session_state.selected_profiles[profile['id']]
                    st.info(f"Deselected {profile.get('name', 'N/A')}")
                st.rerun()

            # LinkedIn button
            if profile.get('linkedin'):
                st.link_button("LinkedIn", profile.get('linkedin'), use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
# --- Streamlit UI ---
st.set_page_config(
    page_title="Growton AI - Advanced Candidate Search", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://growton.ai/help',
        'Report a bug': 'https://growton.ai/bug-report',
        'About': 'Growton AI - Advanced HR Analytics Platform'
    }
)

# Force sidebar to be visible
st.markdown(
    """
    <style>
    /* Force sidebar visibility */
    .css-1d391kg {
        width: 350px !important;
        min-width: 350px !important;
    }
    
    /* Hide the sidebar toggle button */
    .css-1rs6os {
        display: none !important;
    }
    
    /* Ensure main content doesn't overlap */
    .main .block-container {
        padding-left: 380px !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
    section[data-testid="stSidebar"] {
            width: 100% !important;
            min-width: 100% !important;
        }
        
        .header-content {
            flex-direction: column;
            gap: 1rem;
        }
        
        .header-stats {
            flex-direction: row;
            gap: 1rem;
            justify-content: center;
        }
    }
    
    /* Ensure proper spacing */
    .stApp > div {
        padding-top: 0 !important;
    }
    
    /* Fix sidebar positioning */
    .css-1d391kg {
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        z-index: 1000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Advanced CSS Styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-section img {
        width: 60px;
        height: 60px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .logo-section h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-stats {
        display: flex;
        gap: 2rem;
        color: white;
    }
    
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 400px !important;
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] > div {
        height: 100vh;
        overflow-y: auto;
        padding: 1rem;
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Ensure sidebar is visible */
    .css-1d391kg {
        width: 350px !important;
    }
    
    /* Main content area adjustment */
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }
    
    /* Search Interface */
    .search-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .search-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    /* Progress Indicators */
    .progress-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1a1a1a;
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .progress-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 1.1rem;
    }
    
    .progress-percentage {
        font-weight: 700;
        color: #3b82f6;
        font-size: 1.2rem;
    }
    
    /* Control Buttons - GPT Style */
    .control-buttons {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    /* Chat Input Area Styling */
    .stChatInput > div {
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Control buttons above chat input */
    .stChatInput {
        margin-top: 1rem;
    }
    
    /* GPT-style corner buttons */
    .corner-buttons {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        gap: 10px;
    }
    
    .corner-btn {
        background: #1a1a1a;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .corner-btn:hover {
        background: #333;
        transform: translateY(-1px);
    }
    
    .corner-btn-secondary {
        background: #6b7280;
    }
    
    .corner-btn-secondary:hover {
        background: #4b5563;
    }
    
    .control-btn {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        flex: 1;
    }
    
    .btn-primary {
        background: #1a1a1a;
        color: white;
        border: 1px solid #333;
    }
    
    .btn-primary:hover {
        background: #333;
        transform: translateY(-1px);
    }
    
    .btn-danger {
        background: #dc2626;
        color: white;
        border: 1px solid #b91c1c;
    }
    
    .btn-danger:hover {
        background: #b91c1c;
        transform: translateY(-1px);
    }
    
    .btn-success {
        background: #059669;
        color: white;
        border: 1px solid #047857;
    }
    
    .btn-success:hover {
        background: #047857;
        transform: translateY(-1px);
    }
    
    .btn-secondary {
        background: #6b7280;
        color: white;
        border: 1px solid #4b5563;
    }
    
    .btn-secondary:hover {
        background: #4b5563;
        transform: translateY(-1px);
    }
    
    /* Candidate Cards */
    .candidate-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #1a1a1a;
    }
    
    .candidate-card.selected {
        border-color: #1a1a1a;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    
    .candidate-info h3 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .candidate-meta {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .candidate-reasoning {
        background: #f9fafb;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 3px solid #1a1a1a;
    }
    
    .reasoning-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .reasoning-text {
        color: #475569;
        line-height: 1.5;
        font-size: 0.9rem;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-processing {
        background: #fef3c7;
        color: #d97706;
    }
    
    .status-completed {
        background: #d1fae5;
        color: #059669;
    }
    
    .status-stopped {
        background: #fee2e2;
        color: #dc2626;
    }
    
    /* Export Section */
    .export-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .export-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .export-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .selection-count {
        background: #1a1a1a;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    
    /* GPT-style typing indicator */
    @keyframes typing {
        0%, 20% { opacity: 0; }
        50% { opacity: 1; }
        100% { opacity: 0; }
    }
    
    .typing-indicator {
        animation: typing 1.5s infinite;
    }
    
    /* Control buttons hover effect */
    .control-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            gap: 1rem;
        }
        
        .header-stats {
            flex-direction: column;
            gap: 1rem;
        }
        
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Compact Header
st.markdown(
    f"""
    <div style="
        background: #1a1a1a;
        padding: 12px 20px;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 8px 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
        <img src="https://media.licdn.com/dms/image/v2/D560BAQF7O3De5SQ1vA/company-logo_200_200/company-logo_200_200/0/1708433749265/letsgrowton_logo?e=2147483647&v=beta&t=GerSYeinV4BZI9iFhaAo1dfHFDS1Ym5cwhYYwQXEWJo"
                 alt="Growton AI Logo" style="width: 28px; height: 28px; border-radius: 4px;">
            <h1 style="color: white; font-size: 18px; font-weight: 600; margin: 0;">Growton AI</h1>
        </div>
        <div style="display: flex; align-items: center; gap: 20px; color: #9ca3af; font-size: 14px;">
            <span>{len(PROFILES_BY_ID)} profiles</span>
            <span>{sum(p.get('total_experience_years', 0) for p in PROFILES_BY_ID.values()) / len(PROFILES_BY_ID):.1f}y avg</span>
            <span>{st.session_state.get('search_time', 0.0):.1f}s</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Corner buttons removed - moved to sidebar

# Page Navigation Logic
current_page = st.session_state.get('current_page', 'search')

# Professional Sidebar
st.sidebar.markdown("###  Quick Actions")
st.sidebar.markdown("Manage your search session and selections.")

# New Chat and Clear Selection in Sidebar
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🆕 New Chat", help="Start a fresh conversation", use_container_width=True, type="primary"):
        # Clear all session state for new chat
        st.session_state.messages = []
        st.session_state.last_results = []
        st.session_state.selected_profiles = {}
        st.session_state.generating = False
        st.session_state.stop_signal = False
        st.session_state.paused = False
        st.success("New chat started!")
        st.rerun()

with col2:
    if st.button("🗑️ Clear Selection", help="Clear all selected candidates", use_container_width=True, type="secondary"):
        st.session_state.selected_profiles = {}
        st.info("Selection cleared")
        st.rerun()

# Navigation Buttons
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")

# Page navigation buttons
if st.sidebar.button("🔍 Search", help="Main search interface", use_container_width=True, type="primary" if current_page == 'search' else "secondary"):
    st.session_state.current_page = 'search'
    st.rerun()

if st.sidebar.button("🤖 Co-pilot", help="AI-powered query splitting", use_container_width=True, type="primary" if current_page == 'copilot' else "secondary"):
    st.session_state.current_page = 'copilot'
    st.rerun()

if st.sidebar.button("📊 Analytics", help="Dataset analysis and insights", use_container_width=True, type="primary" if current_page == 'analytics' else "secondary"):
    st.session_state.current_page = 'analytics'
    st.rerun()

# Simplified sidebar for all pages
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.markdown(f"**Total Profiles:** {len(PROFILES_BY_ID)}")
st.sidebar.markdown(f"**Avg Experience:** {sum(p.get('total_experience_years', 0) for p in PROFILES_BY_ID.values()) / len(PROFILES_BY_ID):.1f}y")

# Page-specific sidebar content
if current_page == 'search':
    # Search History
    st.sidebar.markdown("### 🔍 Recent Searches")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    if st.session_state.search_history:
        for i, search in enumerate(st.session_state.search_history[-5:]):
            if st.sidebar.button(f"🔍 {search[:30]}...", key=f"history_{i}"):
                st.session_state.messages.append({"role": "user", "content": search})
                st.session_state.generating = True
                st.rerun()
    else:
        st.sidebar.markdown("*No recent searches*")

    # Clear History
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.search_history = []
        st.rerun()
elif current_page == 'copilot':
    st.sidebar.markdown("### 🤖 Co-pilot Features")
    st.sidebar.markdown("Split complex queries into multiple criteria for better results.")
elif current_page == 'analytics':
    st.sidebar.markdown("### 📊 Analytics Features")
    st.sidebar.markdown("Comprehensive analysis of your candidate database.")

# --- Session State Management ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(os.urandom(32)).hexdigest()

# Initialize current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'token_tracker' not in st.session_state:
    st.session_state.token_tracker = TokenCostTracker()
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'stop_signal' not in st.session_state:
    st.session_state.stop_signal = False
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'selected_profiles' not in st.session_state:
    st.session_state.selected_profiles = {} # Stores profile data by ID

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Page Content Logic ---
if current_page == 'search':
    # Main search page content
    pass
elif current_page == 'copilot':
    # Co-pilot page content - Criteria Management Interface
    st.markdown("## 🤖 AI Co-pilot")
    st.markdown("Define and manage your search criteria with intelligent prioritization.")
    
    # Initialize criteria in session state
    if 'copilot_criteria' not in st.session_state:
        st.session_state.copilot_criteria = []
    
    
    # Criteria management section
    st.markdown("### 📋 Search Criteria")
    
    # Initialize criteria if empty
    if not st.session_state.copilot_criteria:
        st.session_state.copilot_criteria = [
            {"text": "GenAI Experience", "importance": "MOST IMPORTANT"},
            {"text": "Experience with cloud platforms (e.g., AWS, GCP, Azure)", "importance": "HIGH"},
            {"text": "Focused on data pipeline optimization in recent role", "importance": "MEDIUM"},
            {"text": "Contributed to open-source Python libraries", "importance": "LEAST IMPORTANT"}
        ]
    
    # Display criteria with drag-and-drop interface
    st.markdown("""
    <style>
    .criteria-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    .criteria-item {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .criteria-number {
        background: #6b7280;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 12px;
        margin-right: 12px;
    }
    .drag-handle {
        width: 20px;
        height: 20px;
        background: #e5e7eb;
        border-radius: 4px;
        margin-right: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: move;
    }
    .drag-dots {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2px;
        width: 8px;
        height: 8px;
    }
    .drag-dot {
        width: 2px;
        height: 2px;
        background: #6b7280;
        border-radius: 50%;
    }
    .importance-label {
        background: #8b5cf6;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 12px;
    }
    .remove-btn {
        background: #ef4444;
        color: white;
        border: none;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display criteria items with proper removal functionality
    for i, criterion in enumerate(st.session_state.copilot_criteria):
        importance_color = {
            "MOST IMPORTANT": "#8b5cf6",
            "HIGH": "#3b82f6", 
            "MEDIUM": "#10b981",
            "LEAST IMPORTANT": "#6b7280"
        }.get(criterion.get('importance', 'MEDIUM'), '#6b7280')
        
        col1, col2, col3, col4, col5 = st.columns([0.5, 0.5, 4, 1.5, 0.5])
        
        with col1:
            st.markdown(f"""
            <div style="
                background: #6b7280;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 12px;
                margin: 0 auto;
            ">{i+1}</div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                width: 20px;
                height: 20px;
                background: #e5e7eb;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: move;
                margin: 0 auto;
            ">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2px; width: 8px; height: 8px;">
                    <div style="width: 2px; height: 2px; background: #6b7280; border-radius: 50%;"></div>
                    <div style="width: 2px; height: 2px; background: #6b7280; border-radius: 50%;"></div>
                    <div style="width: 2px; height: 2px; background: #6b7280; border-radius: 50%;"></div>
                    <div style="width: 2px; height: 2px; background: #6b7280; border-radius: 50%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            criterion_text = st.text_input(
                f"Criterion {i+1}",
                value=criterion['text'],
                key=f"criterion_text_{i}",
                label_visibility="collapsed"
            )
            # Update the criterion text in session state
            if criterion_text != criterion['text']:
                st.session_state.copilot_criteria[i]['text'] = criterion_text
        
        with col4:
            importance_options = ["MOST IMPORTANT", "HIGH", "MEDIUM", "LEAST IMPORTANT"]
            current_importance = criterion.get('importance', 'MEDIUM')
            new_importance = st.selectbox(
                "Importance",
                options=importance_options,
                index=importance_options.index(current_importance),
                key=f"criterion_importance_{i}",
                label_visibility="collapsed"
            )
            # Update the importance in session state
            if new_importance != current_importance:
                st.session_state.copilot_criteria[i]['importance'] = new_importance
        
        with col5:
            if st.button("×", key=f"remove_criterion_{i}", help="Remove this criterion"):
                st.session_state.copilot_criteria.pop(i)
                st.rerun()
    
    # Add new criterion button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("➕ Add Criterion", use_container_width=True, type="secondary"):
            st.session_state.copilot_criteria.append({
                "text": "New criterion",
                "importance": "MEDIUM"
            })
            st.rerun()
    
    with col2:
        if st.button("🚀 Confirm & Start Autopilot", use_container_width=True, type="primary"):
            # Process criteria and start search
            criteria_text = [c['text'] for c in st.session_state.copilot_criteria]
            search_query = f"Find candidates with: {', '.join(criteria_text)}"
            st.session_state.messages.append({"role": "user", "content": search_query})
            st.session_state.generating = True
            st.session_state.current_page = 'search'
            st.rerun()
    
    # JavaScript for interactive functionality
    st.markdown("""
    <script>
    function removeCriterion(index) {
        // This would remove the criterion at the given index
        console.log('Remove criterion at index:', index);
        // In a real implementation, this would trigger a Streamlit rerun
    }
    
    function reorderCriteria(fromIndex, toIndex) {
        // This would reorder criteria
        console.log('Reorder from', fromIndex, 'to', toIndex);
        // In a real implementation, this would trigger a Streamlit rerun
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Criteria importance explanation
    st.markdown("---")
    st.markdown("### 📊 Criteria Importance Levels")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="color: #8b5cf6; font-weight: 600; font-size: 12px;">MOST IMPORTANT</div>
            <div style="font-size: 11px; color: #6b7280;">Critical requirements</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="color: #3b82f6; font-weight: 600; font-size: 12px;">HIGH</div>
            <div style="font-size: 11px; color: #6b7280;">Important factors</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="color: #10b981; font-weight: 600; font-size: 12px;">MEDIUM</div>
            <div style="font-size: 11px; color: #6b7280;">Nice to have</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="color: #6b7280; font-weight: 600; font-size: 12px;">LEAST IMPORTANT</div>
            <div style="font-size: 11px; color: #6b7280;">Optional criteria</div>
        </div>
        """, unsafe_allow_html=True)

elif current_page == 'analytics':
    # Analytics page content
    st.markdown("## 📊 Dataset Analytics")
    st.markdown("Comprehensive analysis of your candidate database with interactive visualizations.")
    
    # Import required libraries for visualizations
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.error("Please install required packages: pip install plotly folium streamlit-folium")
        st.stop()
    
    # Analytics tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "🌍 World Map", "📊 Charts", "💼 Experience", "🏢 Segment", "⚙️ Functional"])
    
    with tab1:
        st.markdown("### 📈 Dataset Overview")
        
        # Key metrics with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Profiles", len(PROFILES_BY_ID), delta="+12% from last month")
        
        with col2:
            avg_exp = sum(p.get('total_experience_years', 0) for p in PROFILES_BY_ID.values()) / len(PROFILES_BY_ID)
            st.metric("Avg Experience", f"{avg_exp:.1f} years", delta="+0.5 years")
        
        with col3:
            total_managed = sum(p.get('max_people_managed', 0) for p in PROFILES_BY_ID.values())
            st.metric("Total Managed", f"{total_managed:,}", delta="+150")
        
        with col4:
            unique_locations = len(set(p.get('location', 'Unknown') for p in PROFILES_BY_ID.values()))
            st.metric("Unique Locations", unique_locations, delta="+3")
    
    with tab2:
        st.markdown("### 🌍 World Map - Candidate Distribution")
        
        # Prepare location data for world map from actual database
        locations = {}
        for profile in PROFILES_BY_ID.values():
            loc = profile.get('location', 'Unknown')
            if loc and loc != 'Unknown' and loc.strip():
                locations[loc] = locations.get(loc, 0) + 1
        
        st.info(f"Found {len(locations)} unique locations with {sum(locations.values())} total candidates")
        
        # Create a comprehensive world map with location markers
        if locations:
            # Create map centered on Asia (where most of your candidates are)
            m = folium.Map(location=[15, 80], zoom_start=4, tiles='OpenStreetMap')
            
            # Enhanced location coordinates with more countries and specific cities
            location_coords = {
                # India - Major cities
                'India': [20.5937, 78.9629],
                'Bengaluru': [12.9716, 77.5946],
                'Bangalore': [12.9716, 77.5946],
                'Karnataka': [12.9716, 77.5946],
                'Bengaluru, Karnataka, India': [12.9716, 77.5946],
                'Greater Bengaluru Area': [12.9716, 77.5946],
                'Bengaluru South, Karnataka, India': [12.9716, 77.5946],
                'Noida': [28.5355, 77.3910],
                'Noida, Uttar Pradesh, India': [28.5355, 77.3910],
                'Uttar Pradesh': [28.5355, 77.3910],
                
                # Malaysia - Major cities
                'Malaysia': [4.2105, 101.9758],
                'Kuala Lumpur': [3.1390, 101.6869],
                'Greater Kuala Lumpur': [3.1390, 101.6869],
                
                # USA - Major cities
                'USA': [39.8283, -98.5795],
                'United States': [39.8283, -98.5795],
                'Florida': [27.7663, -82.6404],
                'Winter Haven, Florida, United States': [28.0222, -81.7329],
                'Winter Haven': [28.0222, -81.7329],
                
                # Other countries
                'Singapore': [1.3521, 103.8198],
                'Australia': [-25.2744, 133.7751],
                'UK': [55.3781, -3.4360],
                'United Kingdom': [55.3781, -3.4360],
                'Germany': [51.1657, 10.4515],
                'Canada': [56.1304, -106.3468],
                'Japan': [36.2048, 138.2529],
                'China': [35.8617, 104.1954],
                'France': [46.2276, 2.2137],
                'Netherlands': [52.1326, 5.2913],
                'Sweden': [60.1282, 18.6435],
                'Norway': [60.4720, 8.4689],
                'Denmark': [56.2639, 9.5018],
                'Finland': [61.9241, 25.7482],
                'Switzerland': [46.8182, 8.2275],
                'Austria': [47.5162, 14.5501],
                'Belgium': [50.5039, 4.4699],
                'Ireland': [53.4129, -8.2439],
                'Spain': [40.4637, -3.7492],
                'Italy': [41.8719, 12.5674],
                'Portugal': [39.3999, -8.2245],
                'Poland': [51.9194, 19.1451],
                'Czech Republic': [49.8175, 15.4730],
                'Hungary': [47.1625, 19.5033],
                'Romania': [45.9432, 24.9668],
                'Bulgaria': [42.7339, 25.4858],
                'Greece': [39.0742, 21.8243],
                'Turkey': [38.9637, 35.2433],
                'Russia': [61.5240, 105.3188],
                'Ukraine': [48.3794, 31.1656],
                'Brazil': [-14.2350, -51.9253],
                'Argentina': [-38.4161, -63.6167],
                'Mexico': [23.6345, -102.5528],
                'Chile': [-35.6751, -71.5430],
                'Colombia': [4.5709, -74.2973],
                'Peru': [-9.1900, -75.0152],
                'South Africa': [-30.5595, 22.9375],
                'Nigeria': [9.0820, 8.6753],
                'Kenya': [-0.0236, 37.9062],
                'Egypt': [26.0975, 30.0444],
                'Israel': [31.0461, 34.8516],
                'UAE': [23.4241, 53.8478],
                'Saudi Arabia': [23.8859, 45.0792],
                'Thailand': [15.8700, 100.9925],
                'Vietnam': [14.0583, 108.2772],
                'Philippines': [12.8797, 121.7740],
                'Indonesia': [-0.7893, 113.9213],
                'South Korea': [35.9078, 127.7669],
                'Taiwan': [23.6978, 120.9605],
                'Hong Kong': [22.3193, 114.1694],
                'New Zealand': [-40.9006, 174.8860]
            }
            
            # Add markers for each location with dynamic sizing
            max_count = max(locations.values()) if locations else 1
            
            for location, count in locations.items():
                # Dynamic location matching with intelligent fallbacks
                coords = None
                location_lower = location.lower()
                
                # First try exact matches
                if location in location_coords:
                    coords = location_coords[location]
                else:
                    # Try case-insensitive exact match
                    for key, coord in location_coords.items():
                        if key.lower() == location_lower:
                            coords = coord
                            break
                
                # If no exact match, try intelligent partial matching
                if not coords:
                    # Extract key words from location
                    location_words = [word.strip() for word in location_lower.replace(',', ' ').split() if len(word.strip()) > 2]
                    
                    best_match = None
                    best_score = 0
                    
                    for key, coord in location_coords.items():
                        key_lower = key.lower()
                        key_words = [word.strip() for word in key_lower.replace(',', ' ').split() if len(word.strip()) > 2]
                        
                        # Calculate match score
                        score = 0
                        
                        # Check for word matches
                        for loc_word in location_words:
                            for key_word in key_words:
                                if loc_word in key_word or key_word in loc_word:
                                    score += 1
                                elif len(loc_word) > 3 and len(key_word) > 3:
                                    # Check for partial matches
                                    if loc_word[:4] in key_word or key_word[:4] in loc_word:
                                        score += 0.5
                        
                        # Bonus for city names
                        city_indicators = ['bengaluru', 'bangalore', 'kuala', 'lumpur', 'noida', 'winter', 'haven']
                        for indicator in city_indicators:
                            if indicator in location_lower and indicator in key_lower:
                                score += 2
                        
                        # Bonus for country/state matches
                        country_indicators = ['india', 'malaysia', 'usa', 'united states', 'karnataka', 'uttar pradesh', 'florida']
                        for indicator in country_indicators:
                            if indicator in location_lower and indicator in key_lower:
                                score += 1.5
                        
                        if score > best_score:
                            best_score = score
                            best_match = coord
                    
                    if best_score > 0:
                        coords = best_match
                
                # Final fallback - try to extract country/region
                if not coords:
                    if 'india' in location_lower or 'karnataka' in location_lower or 'uttar pradesh' in location_lower:
                        coords = [20.5937, 78.9629]  # India center
                    elif 'malaysia' in location_lower or 'kuala' in location_lower or 'lumpur' in location_lower:
                        coords = [3.1390, 101.6869]  # Kuala Lumpur
                    elif 'usa' in location_lower or 'united states' in location_lower or 'florida' in location_lower:
                        coords = [39.8283, -98.5795]  # USA center
                
                if coords:
                    # Dynamic radius based on count (minimum 5, maximum 50)
                    radius = max(5, min(50, (count / max_count) * 40 + 10))
                    
                    # Color based on count
                    if count >= max_count * 0.8:
                        color = '#dc2626'  # Red for highest
                    elif count >= max_count * 0.6:
                        color = '#ea580c'  # Orange
                    elif count >= max_count * 0.4:
                        color = '#d97706'  # Amber
                    elif count >= max_count * 0.2:
                        color = '#16a34a'  # Green
                    else:
                        color = '#2563eb'  # Blue for lowest
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=radius,
                        popup=f"""
                        <div style="font-family: Arial, sans-serif;">
                            <h3 style="margin: 0 0 8px 0; color: #1f2937;">{location}</h3>
                            <p style="margin: 0; font-size: 14px;"><strong>{count}</strong> candidates</p>
                            <p style="margin: 4px 0 0 0; font-size: 12px; color: #6b7280;">
                                {count/len(PROFILES_BY_ID)*100:.1f}% of total
                            </p>
                        </div>
                        """,
                        color='white',
                        weight=2,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
                else:
                    # Log unmatched locations for debugging
                    st.write(f"⚠️ No coordinates found for: {location}")
            
            # Add a legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Candidate Distribution</b></p>
            <p><i class="fa fa-circle" style="color:#dc2626"></i> High Density</p>
            <p><i class="fa fa-circle" style="color:#ea580c"></i> Medium-High</p>
            <p><i class="fa fa-circle" style="color:#d97706"></i> Medium</p>
            <p><i class="fa fa-circle" style="color:#16a34a"></i> Medium-Low</p>
            <p><i class="fa fa-circle" style="color:#2563eb"></i> Low Density</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Display the map - Full width and larger
            st.markdown("""
            <style>
            .folium-map {
                width: 100% !important;
                height: 600px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st_folium(m, width=1200, height=600)
            
            # Show location summary
            st.markdown("### 📍 Location Summary")
            sorted_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Locations:**")
                for location, count in sorted_locations[:10]:
                    percentage = (count / len(PROFILES_BY_ID)) * 100
                    st.markdown(f"• **{location}:** {count} candidates ({percentage:.1f}%)")
            
            with col2:
                st.markdown("**Geographic Distribution:**")
                # Group by continent/region
                regions = {
                    'Asia': ['India', 'Singapore', 'Malaysia', 'Japan', 'China', 'Thailand', 'Vietnam', 'Philippines', 'Indonesia', 'South Korea', 'Taiwan', 'Hong Kong'],
                    'Europe': ['UK', 'Germany', 'France', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Switzerland', 'Austria', 'Belgium', 'Ireland', 'Spain', 'Italy', 'Portugal', 'Poland', 'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Greece', 'Turkey', 'Russia', 'Ukraine'],
                    'North America': ['USA', 'Canada', 'Mexico'],
                    'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
                    'Africa': ['South Africa', 'Nigeria', 'Kenya', 'Egypt'],
                    'Middle East': ['Israel', 'UAE', 'Saudi Arabia'],
                    'Oceania': ['Australia', 'New Zealand']
                }
                
                region_counts = {}
                for location, count in locations.items():
                    for region, countries in regions.items():
                        if any(country.lower() in location.lower() for country in countries):
                            region_counts[region] = region_counts.get(region, 0) + count
                            break
                    else:
                        region_counts['Other'] = region_counts.get('Other', 0) + count
                
                for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(PROFILES_BY_ID)) * 100
                    st.markdown(f"• **{region}:** {count} candidates ({percentage:.1f}%)")
        else:
            st.info("No location data available for mapping.")
    
    with tab3:
        st.markdown("### 📊 Interactive Charts")
        
        # Experience distribution chart
        exp_data = []
        for profile in PROFILES_BY_ID.values():
            exp = profile.get('total_experience_years', 0)
            exp_data.append(exp)
        
        # Create histogram
        fig_hist = px.histogram(
            x=exp_data, 
            nbins=20,
            title="Experience Distribution",
            labels={'x': 'Years of Experience', 'y': 'Number of Candidates'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Location bar chart
        location_data = {}
        for profile in PROFILES_BY_ID.values():
            loc = profile.get('location', 'Unknown')
            location_data[loc] = location_data.get(loc, 0) + 1
        
        if location_data:
            df_locations = pd.DataFrame(list(location_data.items()), columns=['Location', 'Count'])
            df_locations = df_locations.sort_values('Count', ascending=True)
            
            fig_bar = px.bar(
                df_locations, 
                x='Count', 
                y='Location',
                title="Candidates by Location",
                orientation='h',
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab4:
        st.markdown("### 💼 Experience Analysis")
        
        # Experience distribution
        exp_ranges = {
            "0-2 years": 0,
            "3-5 years": 0,
            "6-10 years": 0,
            "11-15 years": 0,
            "16+ years": 0
        }
        
        for profile in PROFILES_BY_ID.values():
            exp = profile.get('total_experience_years', 0)
            if exp <= 2:
                exp_ranges["0-2 years"] += 1
            elif exp <= 5:
                exp_ranges["3-5 years"] += 1
            elif exp <= 10:
                exp_ranges["6-10 years"] += 1
            elif exp <= 15:
                exp_ranges["11-15 years"] += 1
            else:
                exp_ranges["16+ years"] += 1
        
        # Create pie chart
        df_exp = pd.DataFrame(list(exp_ranges.items()), columns=['Range', 'Count'])
        fig_pie = px.pie(
            df_exp, 
            values='Count', 
            names='Range',
            title="Experience Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Display detailed breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Detailed Breakdown:**")
            for range_name, count in exp_ranges.items():
                percentage = (count / len(PROFILES_BY_ID)) * 100
                st.markdown(f"• **{range_name}:** {count} candidates ({percentage:.1f}%)")
        
        with col2:
            # Management experience analysis
            st.markdown("**Management Experience:**")
            managed_0 = sum(1 for p in PROFILES_BY_ID.values() if p.get('max_people_managed', 0) == 0)
            managed_1_5 = sum(1 for p in PROFILES_BY_ID.values() if 1 <= p.get('max_people_managed', 0) <= 5)
            managed_6_20 = sum(1 for p in PROFILES_BY_ID.values() if 6 <= p.get('max_people_managed', 0) <= 20)
            managed_20_plus = sum(1 for p in PROFILES_BY_ID.values() if p.get('max_people_managed', 0) > 20)
            
            st.markdown(f"• **No management:** {managed_0} ({managed_0/len(PROFILES_BY_ID)*100:.1f}%)")
            st.markdown(f"• **1-5 people:** {managed_1_5} ({managed_1_5/len(PROFILES_BY_ID)*100:.1f}%)")
            st.markdown(f"• **6-20 people:** {managed_6_20} ({managed_6_20/len(PROFILES_BY_ID)*100:.1f}%)")
            st.markdown(f"• **20+ people:** {managed_20_plus} ({managed_20_plus/len(PROFILES_BY_ID)*100:.1f}%)")
    
    with tab5:
        st.markdown("### 🏢 Segment Analysis")
        
        # Load enriched profiles from JSON file for analytics
        try:
            with open("../enriched_candidate_profiles.json", "r") as f:
                enriched_profiles = json.load(f)
        except FileNotFoundError:
            st.error("Enriched profiles file not found. Please run the individual.py script first to generate enriched data.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading enriched profiles: {e}")
            st.stop()
        
        # Segment analysis from enriched profiles
        segments = {}
        segment_scores = {}
        
        for profile in enriched_profiles:
            segment_exp = profile.get('segment_experience', {})
            if segment_exp and 'roles' in segment_exp:
                for role in segment_exp['roles']:
                    segment = role.get('segment', 'Unknown')
                    segments[segment] = segments.get(segment, 0) + 1
                    
                    # Track segment scores
                    if segment not in segment_scores:
                        segment_scores[segment] = []
                    segment_scores[segment].append(segment_exp.get('segment_experience_score', 0))
        
        if segments:
            # Create a pie chart for segments
            df_segment = pd.DataFrame(list(segments.items()), columns=['Segment', 'Count'])
            fig_pie = px.pie(
                df_segment, 
                values='Count', 
                names='Segment',
                title="Candidates by Customer Segment",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Create a bar chart for segments
            fig_bar = px.bar(
                df_segment, 
                x='Segment', 
                y='Count',
                title="Segment Distribution",
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No segment data available.")

    with tab6:
        st.markdown("### ⚙️ Functional Experience Analysis")
        
        # Load enriched profiles from JSON file for analytics
        try:
            with open("../enriched_candidate_profiles.json", "r") as f:
                enriched_profiles = json.load(f)
        except FileNotFoundError:
            st.error("Enriched profiles file not found. Please run the individual.py script first to generate enriched data.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading enriched profiles: {e}")
            st.stop()
        
        # Functional experience analysis from enriched profiles
        functional_areas = {}
        functional_scores = {}
        
        for profile in enriched_profiles:
            func_exp = profile.get('functional_experience', {})
            if func_exp and 'roles' in func_exp:
                for role in func_exp['roles']:
                    activity_type = role.get('activity_type', 'Unknown')
                    functional_areas[activity_type] = functional_areas.get(activity_type, 0) + 1
                    
                    # Track functional scores
                    if activity_type not in functional_scores:
                        functional_scores[activity_type] = []
                    functional_scores[activity_type].append(func_exp.get('functional_experience_score', 0))
        
        if functional_areas:
            # Get top 15 functional areas for better visualization
            sorted_areas = sorted(functional_areas.items(), key=lambda x: x[1], reverse=True)[:15]
            df_functional = pd.DataFrame(sorted_areas, columns=['Functional Area', 'Count'])
            
            # Create horizontal bar chart for better readability
            fig_hbar = px.bar(
                df_functional, 
                x='Count', 
                y='Functional Area',
                orientation='h',
                title="Top 15 Functional Areas",
                color='Count',
                color_continuous_scale='Plasma'
            )
            fig_hbar.update_layout(showlegend=False, height=600)
            st.plotly_chart(fig_hbar, use_container_width=True)
            
            # Create a treemap for functional areas
            fig_treemap = px.treemap(
                df_functional, 
                path=['Functional Area'], 
                values='Count',
                title="Functional Areas Treemap",
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Score distribution histogram
            all_scores = []
            for profile in enriched_profiles:
                func_exp = profile.get('functional_experience', {})
                if func_exp and 'functional_experience_score' in func_exp:
                    all_scores.append(func_exp['functional_experience_score'])
            
            if all_scores:
                df_scores = pd.DataFrame({'Score': all_scores})
                fig_scores = px.histogram(
                    df_scores, 
                    x='Score',
                    title="Functional Experience Score Distribution",
                    nbins=10,
                    color_discrete_sequence=['#2E86AB']
                )
                st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.warning("No functional experience data available.")

# --- Advanced Generation Logic ---
if st.session_state.get("generating") and current_page == 'search':
    # Show generation status with typing indicator
    with st.chat_message("assistant"):
        st.markdown("**Searching for candidates...**")
        st.markdown("_AI is analyzing your requirements and finding the best matches_")
    
    
    # --- [NEW] Helper function for dynamic buttons ---
    def draw_criteria_buttons(criteria: dict, container):
        """Draws the dynamic criteria buttons into the container with green highlighting."""
        
        # Helper to format values
        def format_value(val):
            if isinstance(val, dict):
                vals = val.get("values", [])
                return vals[0] if vals else ""
            if isinstance(val, list):
                return val[0] if val else ""
            return str(val)

        # Map keys to (Label, Priority) - higher priority = more important
        PRIORITY_MAP = {
            "required_geographies": ("Geography", 1),
            "required_locations": ("Geography", 1),
            "min_total_experience": ("Experience", 2),
            "required_industries": ("Industry", 3),
            "required_functions": ("Role", 4),
            "required_segments": ("Role", 4),
            "min_tenure_in_latest_role": ("Experience", 2),
            "avg_tenure_in_last_n_roles": ("Experience", 2),
            "required_companies": ("Industry", 3),
            "competitors_of": ("Industry", 3),
            "required_company_details": ("Industry", 3),
            "required_culture_type": ("Role", 4),
        }
        
        with container.container():
            # Create the main criteria buttons (Geography, Experience, Industry, Role)
            main_criteria = {
                "Geography": {"extracted": False, "value": "Pending"},
                "Experience": {"extracted": False, "value": "Pending"},
                "Industry": {"extracted": False, "value": "Pending"},
                "Role": {"extracted": False, "value": "Pending"}
            }
            
            # Process extracted criteria
            for key, value in criteria.items():
                if key in PRIORITY_MAP and value:
                    label, priority = PRIORITY_MAP[key]
                    
                    # Map to main criteria
                    if priority == 1:  # Geography
                        main_criteria["Geography"]["extracted"] = True
                        main_criteria["Geography"]["value"] = format_value(value)
                    elif priority == 2:  # Experience
                        main_criteria["Experience"]["extracted"] = True
                        if key == "min_total_experience":
                            main_criteria["Experience"]["value"] = f"{value}+ years"
                        else:
                            main_criteria["Experience"]["value"] = format_value(value)
                    elif priority == 3:  # Industry
                        main_criteria["Industry"]["extracted"] = True
                        main_criteria["Industry"]["value"] = format_value(value)
                    elif priority == 4:  # Role
                        main_criteria["Role"]["extracted"] = True
                        main_criteria["Role"]["value"] = format_value(value)
            
            # Render the main criteria buttons
            st.markdown("""
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0;">
            """, unsafe_allow_html=True)
            
            for label, data in main_criteria.items():
                if data["extracted"]:
                    # Green button for extracted criteria
                    button_html = f"""
                    <div style="
                        background: #10b981;
                        color: white; 
                        padding: 8px 12px; 
                        border-radius: 20px; 
                        text-align: center; 
                        font-size: 12px; 
                        font-weight: 500;
                        margin: 4px;
                        display: inline-block;
                        animation: popIn 0.3s ease-out forwards;
                        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
                    ">{label}: {data['value']}</div>
                    """
                else:
                    # Transparent button for pending criteria
                    button_html = f"""
                    <div style="
                        background: rgba(0,0,0,0.1); 
                        color: #6b7280; 
                        padding: 8px 12px; 
                        border-radius: 20px; 
                        text-align: center; 
                        font-size: 12px; 
                        font-weight: 500;
                        margin: 4px;
                        display: inline-block;
                        border: 1px solid #e5e7eb;
                        transition: all 0.3s ease;
                    ">{label}: Pending</div>
                    """
                
                st.markdown(button_html, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add the animation style
            st.markdown("""
            <style>
            @keyframes popIn {
                0% { opacity: 0; transform: translateY(10px) scale(0.9); }
                100% { opacity: 1; transform: translateY(0) scale(1); }
            }
            </style>
            """, unsafe_allow_html=True)
    # --- [END NEW] Helper function ---


    # --- [MODIFIED] Dynamic Criteria Buttons ---
    st.markdown("### 🎯 Extracted Criteria")
    # This is the new placeholder that the helper function will fill
    criteria_placeholder = st.empty()
    
    # Show initial transparent criteria buttons
    with criteria_placeholder.container():
        st.markdown("""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0;">
            <div style="
                background: rgba(0,0,0,0.1); 
                color: #6b7280; 
                padding: 8px 12px; 
                border-radius: 20px; 
                text-align: center; 
                font-size: 12px; 
                font-weight: 500;
                border: 1px solid #e5e7eb;
                transition: all 0.3s ease;
            ">Geography: Pending</div>
            <div style="
                background: rgba(0,0,0,0.1); 
                color: #6b7280; 
                padding: 8px 12px; 
                border-radius: 20px; 
                text-align: center; 
                font-size: 12px; 
                font-weight: 500;
                border: 1px solid #e5e7eb;
                transition: all 0.3s ease;
            ">Experience: Pending</div>
            <div style="
                background: rgba(0,0,0,0.1); 
                color: #6b7280; 
                padding: 8px 12px; 
                border-radius: 20px; 
                text-align: center; 
                font-size: 12px; 
                font-weight: 500;
                border: 1px solid #e5e7eb;
                transition: all 0.3s ease;
            ">Industry: Pending</div>
            <div style="
                background: rgba(0,0,0,0.1); 
                color: #6b7280; 
                padding: 8px 12px; 
                border-radius: 20px; 
                text-align: center; 
                font-size: 12px; 
                font-weight: 500;
                border: 1px solid #e5e7eb;
                transition: all 0.3s ease;
            ">Role: Pending</div>
        </div>
        """, unsafe_allow_html=True)
    # --- [END MODIFIED] ---
    

    # Status and Progress Indicators
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    # This container will hold the results as they stream in
    st.markdown("---")
    st.markdown("### Search Results")
    results_container = st.container()
    
    prompt = st.session_state.messages[-1]["content"]
    
    async def run_generation_and_display():
        """Runs the query and displays results as they arrive."""
        
        # Clear previous results from state for this new run
        st.session_state.last_results = []
        
        # Start timing
        import time
        start_time = time.time()
        
        generator = process_query_main(prompt, st.session_state.session_id, st.session_state.token_tracker)
        
        async for item in generator:
            if st.session_state.get("stop_signal", False):
                logger.info("Stop signal received, halting generation.")
                status_placeholder.warning("Generation stopped by user.")
                break
            
            if isinstance(item, str):
                # Handle status strings
                status_placeholder.info(item)
            
            elif isinstance(item, dict):
                msg_type = item.get("type")
                
                # --- [NEW] Handle the criteria message ---
                if msg_type == "criteria":
                    criteria_data = item.get("data")
                    if criteria_data:
                        draw_criteria_buttons(criteria_data, criteria_placeholder)
                # --- [END NEW] ---

                elif msg_type == "progress_start":
                    total = item.get("total", 0)
                    if total > 0:
                        status_placeholder.info(f"Found {total} candidates. Generating summaries...")
                        progress_placeholder.progress(0.0, text="0%")

                elif msg_type == "profile_chunk":
                    profile = item.get("data")
                    if profile:
                        # Add to state
                        st.session_state.last_results.append(profile)
                        
                        # Update progress with visual indicators
                        current = item.get("current", 0)
                        total = item.get("total", 1) # Avoid division by zero
                        progress_percent = current / total
                        
                        # Create visual progress with dots
                        progress_placeholder.markdown(
                            f"""
                            <div style="
                                background: #f8f9fa; 
                                padding: 12px; 
                                border-radius: 8px; 
                                border: 1px solid #e5e7eb;
                                margin: 10px 0;
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span style="font-weight: 600; color: #1a1a1a;">Processing Candidates</span>
                                    <span style="color: #6b7280; font-size: 14px;">{current}/{total}</span>
                                </div>
                                <div style="font-size: 18px; letter-spacing: 2px; color: #10b981;">{"●" * current}{"○" * (total - current)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Use our new helper to draw the profile *immediately*
                        display_profile_with_checkbox(profile, results_container)

                elif msg_type == "complete":
                    final_data = item.get("data", [])
                    final_summary = item.get("summary", "")
                    
                    # Calculate search time
                    elapsed_time = time.time() - start_time
                    
                    # Update header with search time
                    st.markdown(
                        f"""
                        <script>
                        const searchTimeElement = document.getElementById('search-time');
                        if (searchTimeElement) {{
                            searchTimeElement.textContent = '{elapsed_time:.1f}s';
                        }}
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Final sync of session state
                    st.session_state.last_results = final_data
                    
                    if not final_data:
                        status_placeholder.info("No candidates were found that strictly match all criteria.")
                    
                    if final_summary:
                        summary_placeholder.markdown(final_summary)

                    # Add final summary to chat history
                    assistant_message = f"Found {len(final_data)} candidates in {elapsed_time:.1f} seconds."
                    if final_summary:
                        assistant_message += f"\n\n{final_summary}"
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                    
                    break # Generation is finished

    # Run the streaming function
    asyncio.run(run_generation_and_display())
    
    # Generation is complete, update state and rerun to show the 'Export' section
    st.session_state.generating = False
    st.session_state.stop_signal = False
    st.rerun()


# --- Enhanced Results Display ---
if st.session_state.last_results:
    st.markdown("---")
    
    # Results Summary
    total_found = len(st.session_state.last_results)
    selected_count = len(st.session_state.selected_profiles)

    st.markdown(
        f"""
        <div class="export-section">
            <div class="export-header">
                <div class="export-title">Search Results</div>
                <div class="selection-count">{selected_count} Selected</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Results Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Found", total_found)
    with col2:
        st.metric("Selected", selected_count)
    with col3:
        selection_rate = (selected_count / total_found * 100) if total_found > 0 else 0
        st.metric("Selection Rate", f"{selection_rate:.1f}%")
    with col4:
        avg_exp = sum(p.get('total_experience_years', 0) for p in st.session_state.selected_profiles.values()) / max(selected_count, 1)
        st.metric("Avg Experience", f"{avg_exp:.1f}y")
    
    # Results Container
    results_container = st.container()
    for profile in st.session_state.last_results:
        display_profile_with_checkbox(profile, results_container)

    # --- [MODIFIED] Enhanced Export Section ---
    # Only show export/bulk actions if profiles are selected
    if selected_count > 0:
        st.markdown("---")
        st.markdown(
            """
            <div class="export-section">
                <div class="export-header">
                    <div class="export-title">Export Options</div>
                    <div class="selection-count">Ready to Export</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Export Options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Preview Selection**")
            if st.button("Preview Selected Candidates", use_container_width=True):
                preview_data = []
                for p in st.session_state.selected_profiles.values():
                    preview_data.append({
                        "Name": p['name'],
                        "LinkedIn": p['linkedin'],
                        "Location": p['location'],
                        "Experience": f"{p.get('total_experience_years', 0):.1f}y",
                        "Managed": p.get('max_people_managed', 0)
                    })
                st.dataframe(preview_data, use_container_width=True)

        with col2:
            st.markdown("**Excel Export**")
        excel_data = profiles_to_excel(st.session_state.selected_profiles)
        st.download_button(
                label="Download Excel",
            data=excel_data,
                file_name=f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col3:
            st.markdown("**CSV Export**")
            # Create CSV data
            csv_data = []
            for p in st.session_state.selected_profiles.values():
                csv_data.append({
                    "Name": p['name'],
                    "LinkedIn": p['linkedin'],
                    "Location": p['location'],
                    "Experience_Years": p.get('total_experience_years', 0),
                    "People_Managed": p.get('max_people_managed', 0),
                    "AI_Analysis": p.get('reasoning', 'N/A')
                })

            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Bulk Actions
        st.markdown("---")
        st.markdown("### Bulk Actions")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Refresh Results", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.selected_profiles = {}
                st.rerun()

        with col3:
            if st.button("Save Search", use_container_width=True):
                if st.session_state.messages:
                    # Find the last user message to save
                    last_user_message = next((m['content'] for m in reversed(st.session_state.messages) if m['role'] == 'user'), None)
                    if last_user_message and last_user_message not in st.session_state.search_history:
                        st.session_state.search_history.append(last_user_message)
                        st.success("Search saved to history!")
                        st.rerun()
                    elif last_user_message in st.session_state.search_history:
                        st.info("This search is already in your history.")
                    else:
                        st.warning("No user query found to save.")

    # --- [END MODIFIED] ---


# Full page utilization - no extra text

# Professional Control Buttons
if st.session_state.get("generating"):
    # Show control buttons right above the chat input
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Stop", type="primary", use_container_width=True):
            st.session_state.stop_signal = True
            st.session_state.generating = False
            st.rerun()
    
    with col2:
        if st.session_state.get("paused", False):
            if st.button("Resume", type="secondary", use_container_width=True):
                st.session_state.paused = False
                st.info("Search resumed")
        else:
            if st.button("Pause", type="secondary", use_container_width=True):
                st.session_state.paused = True
                st.info("Search paused")
    
    with col3:
        if st.button("Clear", type="secondary", use_container_width=True):
            st.session_state.last_results = []
            st.session_state.selected_profiles = {}
            st.info("Results cleared")

# Enhanced Chat Input
if prompt := st.chat_input("🔍 Ask me to find the perfect candidates...", disabled=st.session_state.get("generating")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.generating = True
    st.session_state.stop_signal = False
    st.session_state.last_results = []
    st.session_state.selected_profiles = {}
    
    # Add to search history
    if prompt not in st.session_state.search_history:
        st.session_state.search_history.append(prompt)
    
    st.rerun()

# Add JavaScript for interactive controls and dynamic updates
st.markdown(
    """
    <script>
    // Update header stats with actual values
    function updateHeaderStats() {
        const totalProfiles = document.getElementById('total-profiles');
        const avgExperience = document.getElementById('avg-experience');
        const searchTime = document.getElementById('search-time');
        
        if (totalProfiles) {
            totalProfiles.textContent = '333';
        }
        if (avgExperience) {
            avgExperience.textContent = '15.0y';
        }
        if (searchTime) {
            // This will be updated by Streamlit's st.markdown injection
        }
    }
    
    // Update stats on page load
    setTimeout(updateHeaderStats, 100);
    
    // Corner button functions
    function newChat() {
        // This would trigger a new chat in Streamlit
        console.log('New chat requested');
        // In a real implementation, this would trigger a Streamlit rerun
    }
    
    function clearSelection() {
        // This would clear selections in Streamlit
        console.log('Clear selection requested');
        // In a real implementation, this would trigger a Streamlit rerun
    }
    
    function stopGeneration() {
        console.log('Stop generation requested');
    }
    
    function pauseGeneration() {
        console.log('Pause generation requested');
    }
    
    function resumeGeneration() {
        console.log('Resume generation requested');
    }
    
    function toggleSelection(profileId) {
        console.log('Toggle selection for profile:', profileId);
    }
    
    </script>
    """,
    unsafe_allow_html=True
)


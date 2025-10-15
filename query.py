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
import copy

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
DB_NAME = "growton_ai"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"

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

STATIC_CULTURE_TAXONOMY = {
    "startup": ["startup", "fast-paced", "agile environment", "high-growth", "early-stage"],
    "corporate": ["corporate", "mnc", "multinational", "large enterprise", "structured environment", "established company"],
    "remote": ["remote-first", "fully remote", "distributed team"]
}

# --- NEW CODE BLOCK: Geography Mapping for Region Expansion ---
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
    "united states": "americas", "usa": "americas", "canada": "americas", "mexico": "americas",
    "brazil": "latam", "argentina": "latam", "colombia": "latam"
}
# --- END NEW CODE BLOCK ---


# --- Dynamic Taxonomy Initialization ---
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

CULTURE_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_CULTURE_TAXONOMY,
    category="Company Culture Types"
)

# --- NEW CODE BLOCK: Geography Map Generation ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def generate_dynamic_geography_map(seed_map: dict) -> dict:
    """
    Uses an LLM to expand a seed geography map with more countries and variations.
    The result is cached to avoid repeated LLM calls.
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
# --- END NEW CODE BLOCK ---


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
    Loads all candidate profiles and their roles from the database based on the new schema.
    This function adapts the new schema's structure into the nested dictionary format
    that the rest of the application's logic expects.
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

# --- MODIFIED FUNCTION: calculate_geography_experience_duration ---
def calculate_geography_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculates the total duration for roles that meet the geography criteria.
    This now includes reverse lookup logic: if a user searches for a country (e.g., "Singapore"),
    it will match roles that mention the broader region (e.g., "APAC"). It also retains the
    original logic of inferring experience from company office locations.
    """
    total_duration = 0.0
    contributing_roles = []
    if not criteria_obj or not isinstance(criteria_obj, dict):
        return 0.0, []

    # Note: The `geography_expansion_prompt` already expands regions in the query.
    # e.g., a search for "APAC" becomes a search for ["APAC", "Singapore", "India", ...]
    # The logic here handles the opposite case: a search for "Singapore" matching a profile with "APAC".
    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values:
        return 0.0, []

    # Pre-calculate regions for all search terms to be efficient
    regions_for_req_values = {v: GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v) for v in req_values}
    regions_to_check = {region for region in regions_for_req_values.values() if region}

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"

        company_details = role.get('company_details', {})
        company_office_locations = company_details.get('customer_presence', [])
        company_locations_text = ' '.join([loc.lower() for loc in company_office_locations])
        company_hq_text = (company_details.get('headquarters') or '').lower()

        combined_search_text = f"{role_text} {company_locations_text} {company_hq_text}"

        # Check if any of the required values (e.g., "singapore") are in the text
        direct_match = any(v in combined_search_text for v in req_values)

        # Check if any of the corresponding regions (e.g., "apac") are in the text
        region_match = any(r in combined_search_text for r in regions_to_check)

        if direct_match or region_match:
            duration = role.get('duration_years', 0.0) or 0.0
            # To avoid double-counting, we add this role only once per criteria check
            if not any(cr['company'] == role.get('company', '') and cr['title'] == role.get('title', '') for cr in contributing_roles):
                total_duration += duration
                contributing_roles.append({
                    'company': role.get('company', ''),
                    'title': role.get('title', ''),
                    'duration_years': duration
                })

    return total_duration, contributing_roles
# --- END MODIFIED FUNCTION ---

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

# --- MODIFIED FUNCTION: check_geography_experience ---
def check_geography_experience(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """
    Checks if a candidate has any experience in the required geographies.
    This now includes reverse lookup logic and adds detailed evidence to the profile's evidence_log.
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
    evidence_reasons = [] # To store the detailed reasons

    for v in values:
        # Get the corresponding region for the search term, if it exists.
        # e.g., if v is "singapore", region_for_v becomes "apac"
        region_for_v = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v)

        # 1. Check candidate's primary location
        profile_location_lower = (profile.get('location') or '').lower()
        if v in profile_location_lower:
            if v not in found_values:
                evidence_reasons.append(f"candidate's primary location is '{profile.get('location')}' which directly matches the search for '{v}'")
            found_values.add(v)
            continue # Move to the next required value

        if region_for_v and region_for_v in profile_location_lower:
            if v not in found_values:
                evidence_reasons.append(f"candidate's primary location is '{profile.get('location')}' which contains the region '{region_for_v}', matching the search for '{v}'")
            found_values.add(v)
            continue

        # 2. If not in primary location, check roles
        is_found_in_roles = False
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            
            company_details = role.get('company_details', {})
            company_office_locations = company_details.get('customer_presence', [])
            company_locations_text = ' '.join([loc.lower() for loc in company_office_locations])
            company_hq_text = (company_details.get('headquarters') or '').lower()
            
            combined_search_text = f"{role_text} {company_locations_text} {company_hq_text}"
            
            # Check for direct match (e.g., "singapore" in text) OR
            # region match (e.g., "apac" in text if searching for "singapore")
            if v in combined_search_text:
                if v not in found_values:
                    evidence_reasons.append(f"experience at '{role.get('company')}' shows presence in '{v}', as inferred from company/role details")
                found_values.add(v)
                is_found_in_roles = True
                break # Found this value, move to the next required value

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
        # Combine reasons into a single log entry.
        source_text = f"Profile confirms experience in {', '.join(found_values)}. Reasons: {'; '.join(evidence_reasons)}."
        profile['evidence_log'].append({
            "criterion": f"geography_presence ({op})",
            "source_text": source_text
        })
    
    return is_met
# --- END MODIFIED FUNCTION ---

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

def check_excluded_geography_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """
    Checks if a candidate has experience in any of the excluded geographies.
    Returns True if there is NO match (i.e., the candidate passes the exclusion check).
    Returns False if there IS a match (i.e., the candidate should be excluded).
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


# --- Strict Filtering Function ---

async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs strict, deterministic filtering in Python with the new flexible logic and sorts by specified criterion duration."""
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

        min_total_exp = criteria.get("min_total_experience")
        if min_total_exp and (profile.get("total_experience_years") or 0) < min_total_exp:
            all_criteria_met = False

        min_managed = criteria.get("min_people_managed")
        if min_managed and (profile.get("max_people_managed") or 0) < min_managed:
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
                    duration, roles = calc_func(profile, crit_obj)
                    profile['calculated_experience'][key] = {
                        "duration": duration,
                        "roles": roles,
                        "label": ", ".join(crit_obj.get("values",[])),
                        "required": 0.0
                    }

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

async def generate_response_with_evidence(query: str, matching_profiles: List[dict], criteria: Dict[str, Any], original_criteria: Dict[str, Any], tracker: TokenCostTracker) -> AsyncIterator[str]:
    """
    Generates the final response with detailed, evidence-based reasoning.
    This function now processes one profile at a time to avoid context length errors.
    """
    if not matching_profiles:
        yield "No candidates were found that strictly match all criteria with explicit evidence in their profiles."
        return

    yield f"Found {len(matching_profiles)} matching candidates. Here are the details:\n\n"

    final_answer_prompt_template = PromptTemplate(
        input_variables=["query", "criteria_json", "matching_profile_json", "original_criteria_json"],
        template="""
You are an expert recruitment analyst. Your task is to generate a concise, evidence-based summary for the single candidate provided, based on the user's query and the filtering criteria.

**Original User Query:** {query}
**Original Filtering Criteria (JSON):** {original_criteria_json}
**Expanded Filtering Criteria Used (JSON):** {criteria_json}
**Matching Candidate (JSON):** {matching_profile_json}

**CRITICAL INSTRUCTIONS FOR DYNAMIC REASONING:**

1.  **Candidate Header:** Start with the candidate's name as a large Markdown header (e.g., # Name). Then, list their LinkedIn, Location, Headline, Total Experience Years, and Max People Managed.

2.  **Create a "Reasoning" Section:**
    - You **MUST** create a bullet point for **ONLY the keys present** in the `Original Filtering Criteria` JSON. Do not mention keys that were only added during expansion.
    - **Do NOT** mention criteria like `min_total_experience` or `min_people_managed` if they are not in the `Original Filtering Criteria` JSON.
    - If there are `excluded_geographies` in the `Original Filtering Criteria`, you MUST mention this as a reasoning point.

3.  **Generate Specific, Evidence-Based Reasoning:**
    - For each criterion (e.g., `required_segments`), look inside the candidate's `calculated_experience` object.
    - Find the matching key (e.g., `calculated_experience.required_segments`).
    - Use the `duration` and `roles` from that object to construct a detailed reason.
    - **Example Sentence Structure:** "The candidate meets the **SMB Experience** requirement with **[Duration] years** of experience, gained primarily from their roles at **[Company A]** and **[Company B]**."

4.  **SPECIAL INSTRUCTION FOR GEOGRAPHY:**
    - The reasoning for a geography match is provided in the `evidence_log` with the criterion `geography_presence`.
    - You **MUST** use the `source_text` from this log entry to construct your reasoning sentence. Do not invent reasons.
    - **Example 1 (Region Match):** If the log says "Reasons: candidate's primary location is 'Singapore, APAC' which contains the region 'apac', matching the search for 'singapore'", your output should be something like: "Meets the **Singapore Experience** requirement because their primary location is listed as APAC."
    - **Example 2 (Company Operations):** If the log says "Reasons: experience at 'Acme Inc' shows presence in 'singapore', as inferred from company/role details", your output should be: "Meets the **Singapore Experience** requirement due to experience at Acme Inc, which has operations in that region."
    - Combine this reason with the duration from `calculated_experience.required_geographies` to form a complete sentence like: "The candidate has **[Duration] years** of **[Geography] Experience**, meeting the requirement because their primary location is in APAC."
    

5.  **Create a Correct "Experience Breakdown" Section:**
    - Look for the `contributing_roles_details.roles` list in the candidate's JSON.
    - If it exists, you **MUST** format it as a bulleted list under the heading "Relevant Experience".
    - **Use this exact format for each role:** `* **{{company}}**: {{title}} ({{duration_years}} years)`
    - This section should only list the specific roles that contributed to meeting the primary search criterion.

6. **No Conclusion:** Do not add any concluding remarks.

---
**Your Turn. Generate the response for the single candidate provided.**
"""
    )    
    
    full_response_content = ""
    for i, profile in enumerate(matching_profiles):
        profile['display_number'] = i + 1
        
        final_prompt_formatted = final_answer_prompt_template.format(
            query=query,
            criteria_json=json.dumps(criteria, indent=2),
            original_criteria_json=json.dumps(original_criteria, indent=2),
            matching_profile_json=json.dumps(profile, indent=2)
        )

        profile_response_content = ""
        async for chunk in streaming_llm.astream(final_prompt_formatted):
            yield chunk.content
            profile_response_content += chunk.content
        
        full_response_content += profile_response_content
        tracker.add_usage("gpt-4o-mini", final_prompt_formatted, profile_response_content, "Final Summary")

        yield "\n---\n\n"


async def process_query_main(query: str, session_id: str, tracker: TokenCostTracker) -> AsyncIterator[str]:
    """
    Main processing pipeline for a user query.
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

    # --- THIS IS THE CORRECTED PROMPT ---
    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "sales_taxonomy_json", "segment_taxonomy_json", "company_details_taxonomy_json", "culture_taxonomy_json"],
        template="""
You are an expert assistant tasked with extracting structured filtering criteria from a user's query for a candidate search system. Your goal is to categorize user intent into functions, segments, industries, etc., and correctly associate any specified durations.

**DEFINITIONS, TAXONOMIES & CANONICAL KEYS:**
- `required_companies`: A list of specific company names. Use this when the user asks for experience *at* a company (e.g., "worked at Google", "from Microsoft").
- `required_functions`: Sales roles. You **MUST** map user input to a canonical key from the sales taxonomy provided below.
  - **Sales Taxonomy (map terms to the key):** {sales_taxonomy_json}
- `required_segments`: Customer types. **You MUST** map user input to a canonical key from the segment taxonomy provided below.
  - **Segment Taxonomy (map terms to the key):** {segment_taxonomy_json}
- `required_company_details`: Company attributes. **You MUST** map user input to a canonical key from the company details taxonomy provided below.
  - **Company Details Taxonomy (map terms to the key):** {company_details_taxonomy_json}
- `required_culture_type`: Company environment. **You MUST** map user input to a canonical key from the culture taxonomy provided below.
  - **Culture Taxonomy (map terms to the key):** {culture_taxonomy_json}
- `required_industries`: Broad industries (e.g., "SaaS", "Fintech"). Do NOT put specific company names here.
- `competitors_of`: A list of companies for which to find competitors.
- `required_geographies`: Regions of sales experience (e.g., "APAC", "EMEA", "India").
- `excluded_geographies`: Regions or countries to specifically exclude from the search.
- `required_locations`: Candidate's physical base.
- `top_n`: Integer for the number of candidates to return.
- `min_total_experience`: The candidate's entire career duration. Only use this if the query explicitly says "total experience" or "overall experience".

**JSON STRUCTURE & DURATION RULES:**
- For inclusion criteria (required_*), use an object with "operator" ("AND"/"OR") and "values" (list of strings).
- For exclusion criteria (excluded_*), use an object with "values" (a list of strings to exclude).
- `required_companies` should be a simple list of strings.
- **Duration Rule:** When a duration (e.g., '10 years', '5+ years') is mentioned with a specific function, industry, or segment, it **MUST** be captured using the `min_years` (float) key inside that specific criterion's object. Do **NOT** use `min_total_experience` unless explicitly stated.

**EXAMPLES TABLE (Follow this logic exactly):**
| User Query                                          | Correct JSON Output                                                                                                                                                             |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "APAC experience, but not India"                    | `{{"required_geographies": {{"operator": "OR", "values": ["APAC"]}}, "excluded_geographies": {{"values": ["India"]}}}}`                                                         |
| "12 years as an Account Executive"                  | `{{"required_functions": {{"operator": "OR", "values": ["Hunting"], "min_years": 12.0}}}}`                                                                                       |
| "candidates who have worked at competitors of Oracle" | `{{"competitors_of": ["Oracle"]}}`                                                                                                                                              |
| "top 10 profiles with SMB experience"               | `{{"required_segments": {{"operator": "OR", "values": ["smb"]}}, "top_n": 10}}`                                                                                                  |
| "SaaS candidates with 10+ years of hunting experience in the US" | `{{"required_industries": {{"operator": "OR", "values": ["saas"]}}, "required_functions": {{"operator": "OR", "values": ["Hunting"], "min_years": 10.0}}, "required_geographies": {{"operator": "OR", "values": ["US"]}}}}` |
| "25+ years of inside sales experience in India"       | `{{"required_functions": {{"operator": "OR", "values": ["Sales Development"], "min_years": 25.0}}, "required_geographies": {{"operator": "OR", "values": ["India"]}}}}`        |


**Available criteria keys:**
- `min_total_experience` (float)
- `min_people_managed` (integer)
- `required_locations` (object)
- `required_geographies` (object)
- `excluded_geographies` (object)
- `required_companies` (list of strings)
- `required_industries` (object)
- `required_functions` (object)
- `required_segments` (object)
- `required_company_details` (object)
- `required_culture_type` (object)
- `competitors_of` (list of strings)
- `top_n` (integer)

**CRITICAL INSTRUCTION**: Based on the taxonomies provided, you **MUST** correctly map user terms like "Account Executive", "inside sales", "hunter", etc., to their canonical keys (`Hunting`, `Sales Development`, etc.) inside the `required_functions` object.

**User Query:** {query}

**JSON Criteria:**
"""
    )
    try:
        yield "Extracting criteria... "
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

        yield f"Identifying competitors for **{company_to_find_competitors_for}** using AI Brainstorm + DB Validation... "

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
        You are a geography expert. For the given list of countries or regions, generate a JSON list containing the original names plus up to 5 major cities or states within them to improve search recall.

        For example, if the input is ["USA"], the output should be a JSON list like: ["USA", "United States", "New York", "California", "Texas", "Illinois"].
        If the input is ["Malaysia"], the output should be a JSON list like: ["Malaysia", "Kuala Lumpur", "Penang", "Johor Bahru", "Selangor"].

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
            yield "Expanding keywords... "
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
                    yield "Expanding geographies... "
                    prompt_text = geography_expansion_prompt.format(geographies=json.dumps(geographies_to_expand))
                    geography_response = await llm.ainvoke(prompt_text)
                    tracker.add_usage(llm.model_name, prompt_text, geography_response.content, "Geography Expansion")
                    logger.info(f"Raw geography expansion response from LLM: {geography_response.content}")
                    expanded_geographies_raw = safe_json_loads(geography_response.content, {})
                    expanded_geographies = get_list_from_llm_json(expanded_geographies_raw)
                    criteria["required_geographies"]["values"] = list(set(geographies_to_expand + expanded_geographies))

        if criteria.get("required_locations"):
            locations_to_expand = get_values_from_criteria(criteria["required_locations"])
            if locations_to_expand:
                prompt_text = location_expansion_prompt.format(locations=json.dumps(locations_to_expand))
                location_response = await llm.ainvoke(prompt_text)
                tracker.add_usage(llm.model_name, prompt_text, location_response.content, "Location Expansion")
                expanded_locations_raw = safe_json_loads(location_response.content, [])
                expanded_locations = get_list_from_llm_json(expanded_locations_raw)
                if isinstance(criteria["required_locations"], dict):
                    criteria["required_locations"]["values"] = list(set(locations_to_expand + expanded_locations))
                elif isinstance(criteria["required_locations"], list):
                     criteria["required_locations"] = list(set(locations_to_expand + expanded_locations))

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

    yield "Performing initial semantic search... "
    search_query_text = " ".join(
        (criteria.get("required_companies") or []) + 
        get_values_from_criteria(criteria.get("required_industries")) +
        get_values_from_criteria(criteria.get("required_functions")) +
        get_values_from_criteria(criteria.get("required_segments")) +
        get_values_from_criteria(criteria.get("required_geographies")) +
        get_values_from_criteria(criteria.get("required_company_details")) +
        get_values_from_criteria(criteria.get("required_culture_type"))
    )

    hard_filters_present = (
        criteria.get("required_locations") or
        criteria.get("min_people_managed") is not None or
        criteria.get("min_total_experience") is not None or
        criteria.get("required_companies")
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
        initial_candidate_pool = list(PROFILES_BY_ID.values())


    final_candidates = await filter_candidates_by_criteria(initial_candidate_pool, criteria)

    async for token in generate_response_with_evidence(query, final_candidates, criteria, original_criteria, tracker):
        yield token
        
    yield tracker.get_summary()

# --- Streamlit UI ---
st.set_page_config(page_title="Growton AI - Candidate Search", layout="wide")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 250px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] > div {
        height: 100%;
        overflow-y: auto;
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

st.sidebar.subheader("📊 Dataset Summary")
total_profiles = len(PROFILES_BY_ID)
total_exp = sum(p.get("total_experience_years") or 0 for p in PROFILES_BY_ID.values())
avg_experience = total_exp / total_profiles if total_profiles > 0 else 0
st.sidebar.markdown(f"**Total Profiles:** {total_profiles}")
st.sidebar.markdown(f"**Avg. Experience:** {round(avg_experience, 1)} years")

# --- Session State Management ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(os.urandom(32)).hexdigest()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'token_tracker' not in st.session_state:
    st.session_state.token_tracker = TokenCostTracker()
# Add state for controlling the generation flow
if "generating" not in st.session_state:
    st.session_state.generating = False
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = False

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core Logic for Generation and Stopping ---

# This block runs when a response is being generated
if st.session_state.get("generating", False):
    # Display a stop button. If clicked, it just sets the signal.
    if st.button("Stop Generation"):
        st.session_state.stop_signal = True
        
    # The actual generation process
    with st.chat_message("assistant"):
        prompt = st.session_state.messages[-1]["content"]
        
        # Wrapper to check for the stop signal
        async def stoppable_generator(generator):
            """A wrapper that checks the session state for a stop signal."""
            async for chunk in generator:
                if st.session_state.get("stop_signal", False):
                    logger.info("Stop signal received, halting generation.")
                    break
                yield chunk

        # Create the main generator and wrap it
        response_generator = process_query_main(prompt, st.session_state.session_id, st.session_state.token_tracker)
        wrapped_gen = stoppable_generator(response_generator)
        
        # Stream the output. This will now stop gracefully when the button is pressed.
        full_response = st.write_stream(wrapped_gen)
        
        # Append the full (or partial) response to the messages list
        # only if some content was generated.
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Clean up state and rerun to reset the UI for the next prompt.
    st.session_state.generating = False
    st.session_state.stop_signal = False
    st.rerun()

# --- Chat Input Form ---

# The input is disabled while a response is being generated
if prompt := st.chat_input("Search for candidates...", disabled=st.session_state.get("generating", False)):
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Set the generating flag to True to start the process
    st.session_state.generating = True
    # Ensure the stop signal is reset for the new run
    st.session_state.stop_signal = False
    # Rerun the script to enter the "generating" block above
    st.rerun()



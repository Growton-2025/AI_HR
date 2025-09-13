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
DB_NAME = "candidate_search"
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
specialist_llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Use a powerful model for validation
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.2) # Powerful model for one-time taxonomy generation
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

# --- ✨ DYNAMIC TAXONOMY INITIALIZATION ✨ ---
SALES_TAXONOMY = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SALES_TAXONOMY,
    category="Sales Functions"
)

SEGMENT_SYNONYMS = generate_dynamic_taxonomy(
    seed_taxonomy=STATIC_SEGMENT_SYNONYMS,
    category="Customer Segments"
)

# Log the results to see what the LLM created
logger.info(f"Loaded Sales Taxonomy with {sum(len(v) for v in SALES_TAXONOMY.values())} total terms.")
logger.info(f"Loaded Segment Taxonomy with {sum(len(v) for v in SEGMENT_SYNONYMS.values())} total terms.")


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
    
    # Fetch all roles and map them by candidate_id, selecting from the new schema columns
    cur.execute("""
        SELECT 
            candidate_id, 
            company, 
            title, 
            details, 
            duration_years, 
            company_details_product_service, 
            company_details_customer_segment 
        FROM roles
    """)
    roles_raw = cur.fetchall()
    roles_by_candidate = {}
    for role in roles_raw:
        candidate_id = role[0]
        if candidate_id not in roles_by_candidate:
            roles_by_candidate[candidate_id] = []
        
        # Manually construct the 'company_details' dictionary to maintain compatibility
        # with downstream functions (e.g., check_industry_presence, check_customer_segments).
        company_details = {
            # The new schema provides 'company_details_product_service'. We map this to both 'industry'
            # and 'product_service' keys to ensure the industry check function works as expected.
            "industry": role[5] or "",
            "product_service": role[5] or "",
            # The 'company_details_customer_segment' is a TEXT[] array, which is handled correctly.
            "customer_segment": role[6] if role[6] is not None else []
        }

        roles_by_candidate[candidate_id].append({
            "company": role[1],
            "title": role[2],
            "details": role[3],
            "duration_years": float(role[4]) if role[4] is not None else 0.0,
            "company_details": company_details  # Use the constructed dictionary
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

# --- Core Logic (No changes needed here as data structure is preserved) ---

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


# --- Presence Check Functions ---

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
                "source_text": f"Profile confirms experience in all required companies/industries: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "industry_presence (OR)",
                "source_text": f"Profile confirms experience in at least one required company/industry. Found: {', '.join(found_values)}."
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
            continue
    
    if op == "AND":
        is_met = found_values == set(values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "segment_experience (AND)",
                "source_text": f"Profile confirms experience in all required segments: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "segment_experience (OR)",
                "source_text": f"Profile confirms experience in at least one required segment. Found: {', '.join(found_values)}."
            })
        return is_met

def check_location_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate is in one of the required locations."""
    req_locations = [loc.lower() for loc in criteria.get("required_locations", [])]
    if not req_locations:
        return True

    candidate_location = (profile.get('location') or "").lower()
    if not candidate_location:
        return False

    if any(loc in candidate_location for loc in req_locations):
        profile['evidence_log'].append({
            "criterion": "location_presence",
            "source_text": f"Candidate is located in '{profile.get('location')}', which matches the search criteria."
        })
        return True
    
    logger.debug(f"{profile.get('name')} filtered out: location '{profile.get('location')}' not in {req_locations}")
    return False

def check_geography_experience(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has sales experience in a required geography, supporting AND/OR logic."""
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
                "criterion": "geography_experience (AND)",
                "source_text": f"Profile confirms experience in all required geographies: {', '.join(values)}."
            })
        return is_met
    else: # OR
        is_met = bool(found_values)
        if is_met:
            profile['evidence_log'].append({
                "criterion": "geography_experience (OR)",
                "source_text": f"Profile confirms experience in at least one required geography. Found: {', '.join(found_values)}."
            })
        return is_met

async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs strict, deterministic filtering in Python with the new flexible logic."""
    logger.info("Applying detailed filters with reasoning...")
    matching_candidates = []

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

        # 4. Check for dynamic year requirements
        if all_criteria_met:
            for key, calc_func in [
                ("required_functions", calculate_functional_experience_duration),
                ("required_industries", calculate_industry_experience_duration),
                ("required_segments", calculate_segment_experience_duration),
                ("required_geographies", calculate_geography_experience_duration)
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

    logger.info(f"Found {len(matching_candidates)} candidates after strict filtering.")
    return matching_candidates

async def generate_response_with_evidence(query: str, matching_profiles: List[dict], criteria: Dict[str, Any]) -> AsyncIterator[str]:
    """
    Generates the final response with detailed, evidence-based reasoning.
    """
    if not matching_profiles:
        yield "No candidates were found that strictly match all criteria with explicit evidence in their profiles."
        return

    yield f"Found {len(matching_profiles)} matching candidates. Here are the details:\n\n"

    final_answer_prompt_template = PromptTemplate(
        input_variables=["query", "criteria_json", "matching_profiles_json"],
        template="""
You are an expert recruitment analyst. Your task is to generate a concise, evidence-based summary for each matching candidate based on the user's query and the specific criteria they were filtered by.

**Original User Query:** {query}
**Filtering Criteria Used (JSON):** {criteria_json}
**Matching Candidates (JSON):** {matching_profiles_json}

**CRITICAL INSTRUCTIONS FOR DYNAMIC REASONING:**

1.  **Address Each Criterion:** For each candidate, you **MUST** create a "**Reasoning**" section. This section **MUST** contain a separate bullet point for **EACH KEY** present in the `Filtering Criteria Used` JSON.
2.  **Be Specific and Cite Evidence:**
    - For `min_total_experience`: State how the candidate meets the requirement by mentioning their total experience.
    - For `min_people_managed`: State how the candidate meets the management requirement by mentioning their max team size.
    - For `required_locations`: Cite their location as evidence.
    - For other criteria (`required_functions`, `required_industries`, etc.):
        - First, cite the evidence from the `evidence_log` that they have presence in the required field.
        - **If a specific `min_years` was required for that criterion**, you **MUST** also cite the calculated duration from the `calculated_experience` field in the candidate's JSON.
        - **Example for Specific Duration:** "Meets Industry Experience: Has worked in FinTech (evidence: role at Stripe). Satisfies the 10-year minimum with 12.5 years of calculated FinTech experience."
3.  **Transparent Experience Calculation:**
    - If the candidate's JSON data includes a `contributing_roles_details` section with a `roles` list, you **MUST** display it as "Experience Breakdown".

---
**Your Turn. Generate the response now based on the provided query, criteria, and profiles.**
"""
    )
    
    for i, profile in enumerate(matching_profiles):
        profile['display_number'] = i + 1

    final_prompt_formatted = final_answer_prompt_template.format(
        query=query,
        criteria_json=json.dumps(criteria, indent=2),
        matching_profiles_json=json.dumps(matching_profiles, indent=2)
    )

    async for chunk in streaming_llm.astream(final_prompt_formatted):
        yield chunk.content

async def process_query_main(query: str, session_id: str) -> AsyncIterator[str]:
    """
    Main processing pipeline for a user query.
    """
    normalized_query = normalize_query_with_llm(query)
    
    # 1. Extract Criteria using LLM with detailed definitions
    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "sales_taxonomy_keys", "segment_taxonomy_keys"],
        template="""
You are an expert assistant. Extract structured filtering criteria from a user's query. Your primary goal is to correctly categorize user intent into functions, segments, industries, etc.

**DEFINITIONS & CANONICAL KEYS:**
- `required_functions`: Describes a sales role. **Map user input to one of these keys:** {sales_taxonomy_keys}
- `required_segments`: Describes a customer type. **Map user input to one of these keys:** {segment_taxonomy_keys}
- `required_industries`: Companies or industries worked in.
- `required_geographies`: Regions of sales experience.
- `required_locations`: Candidate's physical base.

**JSON STRUCTURE RULES:**
- For each criterion, use an object with "operator" ("AND"/"OR") and "values" (a list of the mapped canonical keys or strings).
- If the query specifies years for a criterion, add a "min_years" (float) key to that criterion's object.
- **CRITICAL RULE:** If years of experience are mentioned directly with a function, industry, or segment (e.g., "10 years in inside sales"), you **MUST** capture this as `min_years` inside that specific criterion's object. Only use the top-level `min_total_experience` if the years are mentioned generally (e.g., "candidate with 10 years of experience").
- `min_total_experience` and `min_people_managed` are top-level keys.
- `required_locations` is a simple list of strings.

**EXAMPLES TABLE (Follow this logic exactly):**
| User Query                                                  | Correct JSON Output                                                                                                                                                 |
|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "more than 10 years in inside sales"                        | `{{"required_functions": {{"operator": "OR", "values": ["Sales Development"], "min_years": 10.0}}}}` (Years correctly attached to the function)                          |
| "10 years of SMB sales and managed 30 people"               | `{{"min_people_managed": 30, "required_segments": {{"operator": "OR", "values": ["smb"], "min_years": 10.0}}}}` (Correctly identified 'smb' as a segment)         |
| "experience hunting in the enterprise space"                | `{{"required_functions": {{"operator": "OR", "values": ["Hunting"]}}, "required_segments": {{"operator": "OR", "values": ["enterprise"]}}}}`                         |
| "experience in HCL AND Tech Mahindra"                       | `{{"required_industries": {{"operator": "AND", "values": ["HCL", "Tech Mahindra"]}}}}` (Correctly identified the AND operator)                                      |
| "at least 15 years of team management experience"           | `{{"min_total_experience": 15.0, "min_people_managed": 1}}`                                                                                                          |

**Available criteria keys:**
- `min_total_experience` (float)
- `min_people_managed` (integer)
- `required_locations` (list of strings)
- `required_geographies` (object)
- `required_industries` (object)
- `required_functions` (object)
- `required_segments` (object)
        
**User Query:** {query}
        
**JSON Criteria:**
        """
    )
    try:
        yield "Extracting criteria... "
        criteria_response = await llm.ainvoke(criteria_extraction_prompt.format(
            query=normalized_query,
            sales_taxonomy_keys=json.dumps(list(SALES_TAXONOMY.keys())),
            segment_taxonomy_keys=json.dumps(list(SEGMENT_SYNONYMS.keys()))
        ))
        criteria = safe_json_loads(criteria_response.content, {})
        if not criteria:
            raise ValueError("Failed to parse criteria.")
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error parsing criteria: {e}")
        yield "I had trouble understanding the criteria in your query. Could you please rephrase it?"
        return

    # 2. Expand Keywords using LLM for better search recall
    keyword_expansion_prompt = PromptTemplate(
        input_variables=["keywords", "category"],
        template="""
        Generate a JSON list of 5-7 semantically similar keywords or synonyms for candidate profile searches based on the initial keywords.
        The category is '{category}'.
        Initial Keywords: {keywords}
        JSON List:
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

        # Expand locations
        if criteria.get("required_locations"):
            locations_to_expand = criteria["required_locations"]
            if locations_to_expand:
                location_response = await llm.ainvoke(location_expansion_prompt.format(locations=json.dumps(locations_to_expand)))
                expanded_locations = safe_json_loads(location_response.content, [])
                criteria["required_locations"] = list(set(locations_to_expand + expanded_locations))
        
        # Centralized cleanup logic for all expandable fields
        for key in ["required_industries", "required_functions", "required_geographies", "required_segments"]:
            if key in criteria and isinstance(criteria[key], dict) and "values" in criteria[key]:
                original_values = criteria[key]["values"]
                # Clean the "keywords" literal and any empty strings
                cleaned_values = [v for v in original_values if v and v.lower() != 'keywords']
                criteria[key]["values"] = cleaned_values

        logger.info(f"Full Criteria after expansion: {json.dumps(criteria)}")
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
        get_values_from_criteria_for_search(criteria.get("required_industries")) + 
        get_values_from_criteria_for_search(criteria.get("required_functions")) + 
        get_values_from_criteria_for_search(criteria.get("required_segments")) +
        get_values_from_criteria_for_search(criteria.get("required_geographies"))
    )
    
    # A query is only too broad if it has no semantic keywords AND no hard filters at all.
    hard_filters_present = (
        criteria.get("required_locations") or 
        criteria.get("min_people_managed") is not None or 
        criteria.get("min_total_experience") is not None
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
        # If the search is only by location, the initial pool is all candidates.
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


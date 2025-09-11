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
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Static Taxonomies (as in the old script) ---
SALES_TAXONOMY = {
    'Hunting': ['Hunting', 'new accounts', 'net new', 'New Closures', 'Account Executive'],
    'Farming': ['Account management', 'Account manager', 'Farming', 'Retention'],
    'Sales Development': ['Sales Development', 'Business Development', 'inside sales', 'SDR', 'BDR', 'account development', 'client development'],
    'Partner Sales': ['Partner Sales', 'Partner Development', 'Channel Sales', 'alliance management'],
    'Customer Success': ['Customer Success', 'customer retention']
}

SEGMENT_SYNONYMS = {
    "enterprise": ["enterprise", "large enterprise", "large customers"],
    "mid-market": ["mid-market", "medium size customers"],
    "smb": ["smb", "small business", "small and medium business", "sme"]
}


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
    Loads all candidate profiles and their roles from the database.
    This is suitable for moderately sized datasets. For very large datasets,
    this should be replaced with on-demand fetching.
    """
    logger.info("Loading all profiles from the database into cache...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Fetch all candidates
    cur.execute("SELECT id, name, linkedin, location, headline, about, total_experience_years FROM candidates")
    candidates_raw = cur.fetchall()
    
    # Fetch all roles and map them by candidate_id
    cur.execute("SELECT candidate_id, company, title, details, duration_years, company_details FROM roles")
    roles_raw = cur.fetchall()
    roles_by_candidate = {}
    for role in roles_raw:
        candidate_id = role[0]
        if candidate_id not in roles_by_candidate:
            roles_by_candidate[candidate_id] = []
        roles_by_candidate[candidate_id].append({
            "company": role[1],
            "title": role[2],
            "details": role[3],
            "duration_years": role[4],
            "company_details": json.loads(role[5]) if isinstance(role[5], str) else role[5]
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
            "total_experience_years": cand[6],
            "roles": roles_by_candidate.get(candidate_id, [])
        })
        
    cur.close()
    conn.close()
    logger.info(f"Successfully loaded and cached {len(profiles)} profiles.")
    return profiles

# Load data into a global variable for the app session
PROFILES_BY_ID = {p['id']: p for p in load_all_profiles_from_db()}

# --- Core Logic (Adapted from old, effective script) ---

def normalize_query_with_llm(query: str) -> str:
    """Uses LLM to normalize common synonyms in the query."""
    logger.info(f"Normalizing query... Search Query: {query}")
    return query.lower().replace("sme", "smb").replace("mid market", "mid-market")

def get_cache_key(prefix: str, text: str) -> str:
    """Generates a consistent cache key."""
    return f"{prefix}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

def safe_json_loads(json_str: str, default_val: Any = None) -> Any:
    """Safely loads a JSON string, stripping markdown and handling errors."""
    if default_val is None:
        default_val = []
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

def calculate_functional_experience_duration(profile: dict, criteria: dict) -> Tuple[float, list]:
    """Calculates the total duration for roles that meet the functional criteria."""
    total_duration = 0.0
    contributing_roles = []
    req_funcs = [f.lower() for f in criteria.get("required_functions", [])]
    if not req_funcs:
        return 0.0, []

    for role in profile.get('roles', []):
        role_text = f"{role.get('title', '')} {role.get('details', '')}".lower()
        if any(kw in role_text for kw in req_funcs):
            duration = role.get('duration_years', 0.0) or 0.0
            total_duration += duration
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': duration
            })
    return total_duration, contributing_roles

def check_industry_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a company matching the industry criteria."""
    req_industries = [i.lower() for i in criteria.get("required_industries", [])]
    if not req_industries:
        return True

    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        company_industry_text = f"{company_details.get('industry', '')} {company_details.get('product_service', '')}".lower()
        if any(kw in company_industry_text for kw in req_industries):
            profile['evidence_log'].append({
                "criterion": "industry_presence",
                "source_text": f"Worked at {role.get('company')}, a company in the {', '.join(criteria.get('required_industries',[]))} industry."
            })
            return True
    return False

def check_functional_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has ever worked in a role matching the functional criteria."""
    req_funcs = [f.lower() for f in criteria.get("required_functions", [])]
    if not req_funcs:
        return True

    for role in profile.get('roles', []):
        role_text = f"{role.get('title', '')} {role.get('details', '')}".lower()
        if any(kw in role_text for kw in req_funcs):
            profile['evidence_log'].append({
                "criterion": "functional_presence",
                "source_text": f"Held a role as '{role.get('title')}' at {role.get('company')} which matches the required function."
            })
            return True
    return False


def check_customer_segments(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Checks if the candidate has experience in the required customer segments."""
    req_segments = criteria.get("required_segments")
    if not req_segments:
        return True

    for req_segment in req_segments:
        req_segment_lower = req_segment.lower()
        synonyms = SEGMENT_SYNONYMS.get(req_segment_lower, [req_segment_lower])
        
        found_match_for_segment = False
        
        for role in profile.get("roles", []):
            role_text = f"{role.get('title', '')} {role.get('details', '')}".lower()
            if any(syn in role_text for syn in synonyms):
                profile['evidence_log'].append({
                    "criterion": f"segment_experience: {req_segment}",
                    "source_text": f"Found mention of '{req_segment}' in role at {role.get('company')}."
                })
                found_match_for_segment = True
                break
        
        if found_match_for_segment:
            continue

        for role in profile.get("roles", []):
            company_segments = role.get("company_details", {}).get("customer_segment", [])
            company_segments_lower = [cs.lower() for cs in company_segments]
            if any(syn in company_segments_lower for syn in synonyms):
                profile['evidence_log'].append({
                    "criterion": f"segment_experience: {req_segment}",
                    "source_text": f"Worked at {role.get('company')}, a company serving the '{req_segment}' segment."
                })
                found_match_for_segment = True
                break
        
        if not found_match_for_segment:
            logger.debug(f"{profile.get('name')} filtered out: missing segment {req_segment}")
            return False

    return True

async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs strict, deterministic filtering in Python with the new flexible logic."""
    logger.info("Applying detailed filters with reasoning...")
    matching_candidates = []

    for profile in profiles:
        profile['evidence_log'] = []
        profile['contributing_roles_details'] = {}
        all_criteria_met = True

        # 1. Check Total Experience (if specified)
        min_total_exp = criteria.get("min_total_experience")
        if min_total_exp and (profile.get("total_experience_years") or 0) < min_total_exp:
            all_criteria_met = False

        # 2. Check for presence in the required function
        if all_criteria_met and not check_functional_presence(profile, criteria):
            all_criteria_met = False
            
        # 3. Check for presence in the required industry
        if all_criteria_met and not check_industry_presence(profile, criteria):
            all_criteria_met = False
        
        # 4. Check Customer Segments
        if all_criteria_met and not check_customer_segments(profile, criteria):
            all_criteria_met = False

        if all_criteria_met:
            # If all criteria are met, calculate the experience breakdown for display
            _, roles_list = calculate_functional_experience_duration(profile, criteria)
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

1.  **Address Each Criterion:** For each candidate, you **MUST** create a "**Reasoning**" section. This section **MUST** contain a separate bullet point for **EACH KEY** present in the `Filtering Criteria Used` JSON (e.g., a bullet for `min_total_experience`, a bullet for `required_functions`, etc.).
2.  **Be Specific and Cite Evidence:**
    - For `min_total_experience`: State how the candidate meets the requirement by mentioning their total experience.
    - For `required_functions`: State that the candidate has the required functional experience and cite the evidence from the `evidence_log`.
    - For `required_industries`: State that the candidate has the required industry experience and cite the evidence from the `evidence_log`.
3.  **Transparent Experience Calculation:**
    - If the candidate's JSON data includes a `contributing_roles_details` section with a `roles` list, you **MUST** display it as "Experience Breakdown".

**Example of Perfect, Dynamic Reasoning:**
*If the criteria were `{{"min_total_experience": 10, "required_industries": ["FinTech"]}}`:*

**1. Jane Doe**
- **LinkedIn:** [URL]
- **Location:** [Location]

**Reasoning:**
- **Meets Total Experience Requirement:** She satisfies the **10-year experience** minimum with **15 years** of total experience.
- **Meets Industry Requirement:** She has experience in the **FinTech industry**, as shown by her role at Stripe.

**Experience Breakdown:**
- Stripe, Enterprise Account Executive: 5.5 years
- Adyen, Account Manager: 7.0 years

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
        input_variables=["query"],
        template="""
You are an expert assistant. Extract structured filtering criteria from a user's query. Follow the mapping rules and examples PERFECTLY.

**DEFINITIONS:**
- **Segment Experience**: The type of customers sold to (e.g., enterprise, smb). Maps to `required_segments`.
- **Functional Experience**: The type of sales role (e.g., Hunting, Farming). Maps to `required_functions`.

**CRITICAL RULE: `min_role_experience` vs. `min_total_experience`**
- Use `min_total_experience` when a query mentions years of experience, as this provides a better pool of candidates.
- Ensure `required_functions` is also populated to check for skill presence.

**EXAMPLES TABLE (Follow this logic exactly):**
| User Query                                             | Correct JSON Output                                                                 |
|--------------------------------------------------------|-------------------------------------------------------------------------------------|
| "10 years of account development experience"           | `{{"min_total_experience": 10.0, "required_functions": ["Sales Development"]}}`       |
| "10 years of inside sales experience"                  | `{{"min_total_experience": 10.0, "required_functions": ["Sales Development"]}}`       |
| "experience in customer engagement and 10 years hunting"| `{{"min_total_experience": 10.0, "required_functions": ["Hunting"], "required_industries": ["customer engagement"]}}` |
| "at least 15 years of total experience"                | `{{"min_total_experience": 15.0}}`                                                    |
| "experience in smb segment"                            | `{{"required_segments": ["smb"]}}`                                                     |


**Available criteria keys:**
- `min_total_experience` (float)
- `required_industries` (list of strings)
- `required_functions` (list of strings from: 'Hunting', 'Farming', 'Sales Development', 'Partner Sales', 'Customer Success')
- `required_segments` (list of strings from: 'enterprise', 'mid-market', 'smb')
        
**User Query:** {query}
        
**JSON Criteria:**
        """
    )
    try:
        yield "Extracting criteria... "
        criteria_response = await llm.ainvoke(criteria_extraction_prompt.format(query=normalized_query))
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
    try:
        yield "Expanding keywords... "
        # Expand industries
        if criteria.get("required_industries"):
            industry_keywords_response = await llm.ainvoke(keyword_expansion_prompt.format(keywords=criteria["required_industries"], category="Industry"))
            expanded_industries = safe_json_loads(industry_keywords_response.content)
            criteria["required_industries"].extend(expanded_industries)
        
        # FIX: Directly use the full list of synonyms from the taxonomy instead of a fragile LLM call
        if criteria.get("required_functions"):
            expanded_functions = []
            for func in criteria["required_functions"]:
                expanded_functions.extend(SALES_TAXONOMY.get(func, [func]))
            criteria["required_functions"] = expanded_functions
        
        # Ensure unique keywords and clean up any "keywords" literal from LLM
        if criteria.get("required_industries"):
            criteria["required_industries"] = [k for k in set(criteria["required_industries"]) if k.lower() != 'keywords']
        if criteria.get("required_functions"):
            criteria["required_functions"] = [k for k in set(criteria["required_functions"]) if k.lower() != 'keywords']

        logger.info(f"Full Criteria after expansion: {json.dumps(criteria)}")
        yield f"Full Criteria: `{json.dumps(criteria)}`\n"

    except Exception as e:
        logger.error(f"Error expanding keywords: {e}")
        
    # 3. Initial Semantic Search against PostgreSQL
    yield "Performing initial semantic search... "
    search_query_text = " ".join(criteria.get("required_industries", []) + criteria.get("required_functions", []) + criteria.get("required_segments",[]))
    if not search_query_text:
        yield "Your query is too broad. Please specify industries, functions, or segments."
        return

    query_embedding = embeddings.embed_query(search_query_text)
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM candidates ORDER BY embedding <=> %s LIMIT 100",
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
if prompt := st.chat_input("Find me a sales leader with 10+ years of experience in FinTech..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.write_stream(process_query_main(prompt, st.session_state.session_id))

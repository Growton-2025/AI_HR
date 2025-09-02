import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import redis
import hashlib
from datetime import datetime
import logging
import tiktoken
import numpy as np
import asyncio
from typing import List, Optional, AsyncIterator, Tuple


# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Initialize Redis client for caching and history
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True) # Using port 6379 as per your default setup
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.ConnectionError as e:
    st.error(f"Failed to connect to Redis. Please ensure Redis server is running: {e}")
    st.stop()

# Clear existing query cache on app startup (optional, good for fresh testing)
try:
    keys = redis_client.keys("query:*")
    if keys:
        for key in keys:
            redis_client.delete(key)
        logger.info("Cleared all previous query cache keys.")
except Exception as e:
    logger.error(f"Error clearing Redis query cache: {e}")
    st.warning("Failed to clear Redis query cache. Cache might contain old entries.")

# Define maximum number of profiles to process in a single LLM chunk for the final response
MAX_PROFILES_PER_FINAL_LLM_CHUNK = 5 # Adjust based on token limits, 5-10 is often good for detailed summaries per profile

# Define maximum number of concurrent LLM calls for industry classification
# This helps manage API rate limits and prevents overwhelming the LLM.
MAX_INDUSTRY_CLASSIFICATIONS_PER_LLM_BATCH = 1000 # You can adjust this value
# Max concurrent LLM calls for sales category determination
MAX_SALES_CATEGORY_CLASSIFICATIONS_PER_LLM_BATCH = 1000 # Adjusted for potentially more frequent calls
# Max concurrent LLM calls for sales specialist validation
MAX_SALES_SPECIALIST_VALIDATIONS_PER_LLM_BATCH = 20 # Keep this low for detailed LLM review


# Define a semantic similarity threshold for product/service keywords
SEMANTIC_SIMILARITY_THRESHOLD = 0.75

# Use st.cache_resource to load heavy components only once
@st.cache_resource
def load_llm():
    """Loads and caches the ChatOpenAI language model for direct calls."""
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.0, max_tokens=1000)

@st.cache_resource
def load_streaming_llm():
    """Loads and caches the ChatOpenAI language model for streaming calls."""
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.0, streaming=True, max_tokens=1000)

@st.cache_resource
def load_async_llm():
    """Loads and caches the async ChatOpenAI language model for batch calls (non-streaming)."""
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.0, max_tokens=1000)

@st.cache_resource
def load_sales_specialist_llm():
    """Loads and caches a specialized LLM for sales validation."""
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.3, max_tokens=1500) # Higher max_tokens for detailed analysis

@st.cache_resource
def load_embeddings():
    """Loads and caches the OpenAIEmbeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

@st.cache_resource
def load_vector_store(_embedding_model):
    """Loads and caches the FAISS vector store."""
    try:
        return FAISS.load_local("faiss_index", embeddings=_embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index. Please ensure 'faiss_index' directory exists: {e}")
        st.stop()

@st.cache_resource
def load_profiles_data():
    """Loads and caches the candidate profiles data."""
    try:
        with open("enriched_candidate_profiles.json", "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} profiles from enriched_candidate_profiles.json.")
        return data
    except FileNotFoundError:
        st.error("enriched_candidate_profiles.json not found. Please ensure it's in the same directory.")
    except Exception as e:
        st.error(f"Error loading enriched_candidate_profiles.json: {e}")
    st.stop() # Stop if profiles data cannot be loaded, preventing further errors

# Initialize cached components
llm = load_llm()
streaming_llm = load_streaming_llm() # New streaming LLM instance
async_llm = load_async_llm()
sales_specialist_llm = load_sales_specialist_llm() # Dedicated sales specialist LLM
embedding = load_embeddings()
vector_store = load_vector_store(embedding)
profiles_data = load_profiles_data()
profiles_dict = {p["name"]: p for p in profiles_data}

# New function to generate dynamic SALES_TAXONOMY - Now directly defined based on user's request
@st.cache_resource
def get_sales_taxonomy():
    """Returns the static SALES_TAXONOMY based on user's defined categories."""
    logger.info("Using static SALES_TAXONOMY as per user's definition.")
    sales_taxonomy = {
        'Hunting': ['Hunting', 'new accounts', 'net new', 'New Closures', 'Account Executive'],
        'Farming': ['Account management', 'Account manager', 'Farming', 'Retention'],
        'Sales Development': ['Sales Development', 'Business Development', 'inside sales', 'SDR', 'BDR', 'Account development'],
        'Partner Sales': ['Partner Sales', 'Partner Development', 'Channel Sales', 'alliance management'],
        'Customer Success': ['Customer Success', 'customer retention']
    }
    # Create the inverted taxonomy from the static sales_taxonomy
    inverted_taxonomy = {keyword.lower(): category for category, keywords in sales_taxonomy.items() for keyword in keywords}
    logger.info(f"Loaded static SALES_TAXONOMY: {json.dumps(sales_taxonomy, indent=2)}")
    logger.info(f"Generated INVERTED_TAXONOMY with {len(inverted_taxonomy)} keywords.")
    return sales_taxonomy, inverted_taxonomy

# Get the static SALES_TAXONOMY and INVERTED_TAXONOMY
SALES_TAXONOMY, INVERTED_TAXONOMY = get_sales_taxonomy()

# New function to generate dynamic SEGMENT_SYNONYMS
@st.cache_resource
def generate_dynamic_segments_synonyms(_profiles_data):
    """Generates dynamic SEGMENT_SYNONYMS using LLM based on profiles data."""
    logger.info("Generating dynamic SEGMENT_SYNONYMS using LLM.")

    # Define a default segments synonyms as a fallback
    default_segments = {
        "smb": ["smb", "small and medium business", "mid-market", "small business", "sme"],
        "enterprise": ["enterprise", "large enterprise"],
        "consumer": ["consumer", "b2c"]
    }

    sample_size = min(50, len(_profiles_data))
    sampled_profiles = _profiles_data[:sample_size]
    profile_summary = ""
    for profile in sampled_profiles:
        for role in profile.get('roles', []):
            if role.get('title'):
                profile_summary += f" {role['title'].lower()}"
            if role.get('details'):
                profile_summary += f" {role['details'].lower()}"
    if 'raw_fields' in profile and 'Skills' in profile['raw_fields']:
        profile_summary += f" {profile['raw_fields']['Skills'].lower()}"

    segments_prompt = PromptTemplate(
        input_variables=["profile_summary"],
        template="""
You are an AI assistant tasked with generating a dynamic dictionary of customer segments and their synonyms based on a summary of candidate profiles. The output must be a JSON object where each key is a primary segment name (e.g., "smb", "enterprise", "consumer") and each value is a list of common keywords or phrases that represent that segment in professional profiles.

**MANDATORY OUTPUT FORMAT:**
- Return a JSON object.
- Keys should be primary segment names (e.g., "smb", "enterprise", "consumer").
- Values should be lists of keywords/phrases (strings), that appear in job titles, details, or skills.
- Ensure keywords are unique across segments (case-insensitive).
- Limit to 3-5 primary segments to keep the taxonomy manageable.
- Each segment should have 3-7 keywords.

**Example Output:**
```json
{{
  "smb": ["smb", "small and medium business", "mid-market", "small business", "sme"],
  "enterprise": ["enterprise", "large enterprise", "corporate accounts"],
  "consumer": ["consumer", "b2c", "direct-to-consumer"]
}}
```
Instructions:
Analyze the profile summary to identify common customer segments and their associated terms.
If a term could fit multiple segments, assign it to the most specific one.
Return only the JSON object, no additional text or markdown.

Profile Summary: {profile_summary}
"""
    )
    try:
        formatted_prompt = segments_prompt.format(profile_summary=profile_summary[:10000])
        logger.info(f"Sending segments generation prompt to LLM (first 1000 chars): {formatted_prompt[:1000]}...")
        segments_response = llm.invoke(formatted_prompt).content.strip()
        if segments_response.startswith("json"):
            segments_response = segments_response.lstrip("json\n").strip()
        elif segments_response.startswith("```json"):
            segments_response = segments_response.lstrip("```json").rstrip("```").strip()
        dynamic_segments = json.loads(segments_response)
        logger.info(f"Generated dynamic SEGMENT_SYNONYMS: {json.dumps(dynamic_segments, indent=2)}")
        return dynamic_segments
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for dynamic SEGMENT_SYNONYMS: {e}. Raw response: {segments_response}")
        logger.info("Using default SEGMENT_SYNONYMS due to parsing error.")
        return default_segments
    except Exception as e:
        logger.error(f"Error during dynamic SEGMENT_SYNONYMS generation: {e}")
        return default_segments

# New function to generate dynamic EUROPEAN_COUNTRIES (and other relevant geographies)
@st.cache_resource
def generate_dynamic_geographies(_profiles_data):
    """Generates dynamic geographic lists using LLM based on profiles data."""
    logger.info("Generating dynamic geographic lists using LLM.")

    # Define default geographies as a fallback
    default_european_countries = ["united kingdom", "ireland", "germany", "france", "spain", "sweden", "italy", "netherlands", "belgium"]
    # You could also define a broader 'GLOBAL_REGIONS' if you had a concept for that.
    
    sample_size = min(50, len(_profiles_data))
    sampled_profiles = _profiles_data[:sample_size]
    location_summary = ""
    for profile in sampled_profiles:
        if profile.get('location'):
            location_summary += f" {profile['location'].lower()}"
        for role in profile.get('roles', []):
            if role.get('company_details', {}).get('customer_presence'):
                location_summary += f" {' '.join([loc.lower() for loc in role['company_details']['customer_presence']])}"
        if profile.get('geography_experience', {}).get('regions'):
             location_summary += f" {' '.join([loc.lower() for loc in profile['geography_experience']['regions']])}"

    geography_prompt = PromptTemplate(
        input_variables=["location_summary"],
        template="""
You are an AI assistant tasked with identifying and listing common European countries (and potentially other significant global regions/countries) mentioned in candidate profiles. The output must be a JSON object containing lists of countries/regions.

**MANDATORY OUTPUT FORMAT:**
- Return a JSON object.
- One key should be "european_countries" with a list of European countries (strings).
- You can add other keys for broader regions if they are distinctly mentioned (e.g., "asia_pacific_regions", "north_american_countries").
- Values should be lists of countries or regions (strings), lowercase.
- Ensure all items in lists are unique.
- Prioritize countries explicitly mentioned in profile locations or customer presence details.

**Example Output:**
```json
{{
  "european_countries": ["united kingdom", "ireland", "germany", "france", "spain"],
  "asia_pacific_global_regions": ["india", "singapore", "australia"]
}}
```
Instructions:
Analyze the location summary to extract common European countries and other significant global regions/countries.
Return only the JSON object, no additional text or markdown.

Location Summary: {location_summary}
"""
    )
    try:
        formatted_prompt = geography_prompt.format(location_summary=location_summary[:10000])
        logger.info(f"Sending geography generation prompt to LLM (first 1000 chars): {formatted_prompt[:1000]}...")
        geography_response = llm.invoke(formatted_prompt).content.strip()
        if geography_response.startswith("json"):
            geography_response = geography_response.lstrip("json\n").strip()
        elif geography_response.startswith("```json"):
            geography_response = geography_response.lstrip("```json").rstrip("```").strip()
        dynamic_geographies = json.loads(geography_response)
        
        # Extract European countries, ensuring it's a list
        dynamic_european_countries = dynamic_geographies.get("european_countries", [])
        if not isinstance(dynamic_european_countries, list):
            dynamic_european_countries = [] # Ensure it's a list even if LLM gives wrong type
        
        logger.info(f"Generated dynamic Geographies: {json.dumps(dynamic_geographies, indent=2)}")
        return dynamic_european_countries # Return only the list of European countries
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for dynamic Geographies: {e}. Raw response: {geography_response}")
        logger.info("Using default EUROPEAN_COUNTRIES due to parsing error.")
        return default_european_countries
    except Exception as e:
        logger.error(f"Error during dynamic Geography generation: {e}")
        return default_european_countries

# Generate dynamic SEGMENT_SYNONYMS and EUROPEAN_COUNTRIES
SEGMENT_SYNONYMS = generate_dynamic_segments_synonyms(profiles_data)
EUROPEAN_COUNTRIES = generate_dynamic_geographies(profiles_data)


# Helper Functions
def calculate_years(start_date: str, end_date: str, current_date: str) -> float:
    """Calculate years between start_date and end_date."""
    try:
        date_formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%m/%d/%Y", "%b %Y", "%Y"]
        start, end = None, None
        for fmt in date_formats:
            try:
                start = datetime.strptime(start_date, fmt)
                break
            except Exception:
                continue
        if not start:
            raise ValueError(f"Could not parse start_date: {start_date}")

        if end_date and str(end_date).lower() != "present":
            for fmt in date_formats:
                try:
                    end = datetime.strptime(end_date, fmt)
                    break
                except Exception:
                    continue
            if not end:
                raise ValueError(f"Could not parse end_date: {end_date}")
        else:
            end = datetime.strptime(current_date, "%Y-%m-%d")

        return round((end - start).days / 365.25, 2)
    except Exception as e:
        logger.error(f"Date parsing error: {e}, start_date='{start_date}', end_date='{end_date}'")
        return 0.0

def get_cache_key(query: str) -> str:
    """Generate a unique cache key for the query."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# FIX: New helper function to calculate experience duration for roles that meet ALL specified criteria
def calculate_conjunctive_experience_duration(profile: dict, criteria: dict, query_focus: str) -> Tuple[float, list]:
    """
    Calculates the total duration for roles that meet a combination of functional,
    segment, and industry criteria simultaneously.
    """
    total_duration = 0.0
    contributing_roles = []
    
    req_funcs = criteria.get("required_functional_experience", [])
    req_segments = criteria.get("required_segments", [])
    req_industries = criteria.get("required_industries", [])

    # Pre-compile keywords for matching, expanding from the sales taxonomy
    func_keywords = set()
    for f in req_funcs:
        f_lower = f.lower()
        func_keywords.add(f_lower)
        # Check if the functional requirement is a key in the sales taxonomy
        for category, keywords in SALES_TAXONOMY.items():
            if f_lower in [kw.lower() for kw in keywords]:
                func_keywords.update([kw.lower() for kw in keywords])
                break

    segment_keywords = {s.lower() for s in req_segments}
    industry_keywords = {i.lower() for i in req_industries}

    for role in profile.get('roles', []):
        # FIX: Combine title and details for a comprehensive check of the person's role.
        role_text = f"{role.get('title', '')} {role.get('details', '')}".lower()
        company_details = role.get('company_details', {})
        company_industry_text = f"{company_details.get('industry', '')} {company_details.get('product_service', '')}".lower()
        company_segment_text = ' '.join([seg.lower() for seg in company_details.get('customer_segment', [])])

        # A role must match all specified criteria types (functional, segment, industry)
        # If a criterion type is not specified in the query, it's considered a match by default.
        matches_func = not req_funcs or any(kw in role_text for kw in func_keywords)
        matches_segment = not req_segments or any(kw in role_text for kw in segment_keywords) or (query_focus != "person" and any(kw in company_segment_text for kw in segment_keywords))
        matches_industry = not req_industries or (query_focus != "person" and any(kw in company_industry_text for kw in industry_keywords))

        if matches_func and matches_segment and matches_industry:
            total_duration += role.get('duration_years', 0.0)
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': role.get('duration_years', 0.0)
            })

    # Remove duplicate roles
    unique_roles = []
    seen = set()
    for r in contributing_roles:
        identifier = (r['company'], r['title'])
        if identifier not in seen:
            unique_roles.append(r)
            seen.add(identifier)

    return total_duration, unique_roles


# Function to build LLM prompt for sales category classification for a single profile
def build_sales_category_classification_prompt(profile_text: str, sales_taxonomy_json: str):
    """Builds a single prompt for LLM classification of sales category."""
    sales_category_classification_prompt_template = PromptTemplate(
        input_variables=["profile_text", "sales_taxonomy_json"],
        template="""
You are a highly precise Sales Domain Expert. Your task is to **rigorously and strictly** categorize the candidate's primary sales experience based on the provided profile text and sales taxonomy.

Sales Categories and associated keywords:
{sales_taxonomy_json}

Profile Text Summary:
{profile_text}

Instructions:
- Based on the profile text, determine if the candidate's primary experience **unequivocally** aligns with 'Hunting', 'Farming', 'Sales Development', 'Partner Sales', 'Customer Success'.
- **If a profile does not unequivocally fit one of the defined categories, classify it as 'Other'. Avoid any loose interpretations or assumptions.**
- Consider job titles, roles, responsibilities, and skills.
- Be mindful of synonyms and functional differences (e.g., 'Channel' often maps to 'Partner Sales').
- **Specifically, understand that general 'Business Development' or 'Account Development' typically implies lead generation (Sales Development), unless the text explicitly details new customer acquisition, 'net new', or 'new logos' indicating a hunting role.**
- Each category is **distinct**. Do not combine or assume overlap unless the taxonomy explicitly states it.
- Respond with only the name of the best-matching sales category (e.g., "Hunting", "Farming", "Partner Sales"). Do NOT include any other text or explanation.
"""
    )
    return sales_category_classification_prompt_template.format(
        profile_text=profile_text[:2000], # Limit profile text to avoid token overflow
        sales_taxonomy_json=sales_taxonomy_json
    )

# Async function to determine sales categories in batches
async def batch_determine_sales_categories(profiles: List[dict], sales_taxonomy: dict):
    """
    Uses LLM to determine the sales category for multiple profiles,
    processing in controlled batches to manage API limits and caching results.
    """
    all_classification_tasks = []
    
    sales_taxonomy_json = json.dumps(sales_taxonomy, indent=2)

    for profile in profiles:
        profile_name = profile.get("name", "Unknown Profile")
        profile_sales_cache_key = f"profile_sales_category:{hashlib.md5(profile_name.encode()).hexdigest()}"
        
        cached_category = redis_client.get(profile_sales_cache_key)
        if cached_category:
            profile['sales_category'] = cached_category
            logger.debug(f"Cached sales category for '{profile_name}': {cached_category}")
        else:
            # FIX: Use a more comprehensive text for classification, including titles.
            profile_text = f"{profile.get('headline', '')} {profile.get('about', '')}"
            for role in profile.get('roles', []):
                profile_text += f" {role.get('title', '')} {role.get('details', '')}"
            if 'Skills' in profile.get('raw_fields', {}):
                profile_text += f" {profile['raw_fields']['Skills']}"

            prompt = build_sales_category_classification_prompt(profile_text, sales_taxonomy_json)
            all_classification_tasks.append((prompt, profile_name, profile_sales_cache_key))
    
    if not all_classification_tasks:
        logger.info("No new sales category classifications needed (all cached).")
        return

    logger.info(f"Total {len(all_classification_tasks)} sales category classification tasks to perform.")

    # Process tasks in smaller, controlled batches
    for i in range(0, len(all_classification_tasks), MAX_SALES_CATEGORY_CLASSIFICATIONS_PER_LLM_BATCH):
        batch_tasks_info = all_classification_tasks[i:i + MAX_SALES_CATEGORY_CLASSIFICATIONS_PER_LLM_BATCH]
        llm_tasks = [async_llm.ainvoke(task_info[0]) for task_info in batch_tasks_info]

        logger.info(f"Sending batch of {len(llm_tasks)} sales category classification prompts to LLM.")
        responses = await asyncio.gather(*llm_tasks)

        for j, response in enumerate(responses):
            original_prompt, profile_name, redis_key = batch_tasks_info[j]
            classified_category = response.content.strip()
            
            # Basic validation: ensure the classified category is one of the expected ones
            if classified_category not in sales_taxonomy.keys() and classified_category != 'Other': # 'Other' is a valid fallback
                logger.warning(f"LLM returned unexpected sales category '{classified_category}' for '{profile_name}'. Defaulting to 'Other'.")
                classified_category = 'Other'

            redis_client.setex(redis_key, 86400, classified_category) # Cache for 24 hours
            logger.debug(f"LLM classified '{profile_name}' as '{classified_category}'")

            # Update the profile in the original list (by reference)
            for profile in profiles:
                if profile.get("name") == profile_name:
                    profile['sales_category'] = classified_category
                    break
    logger.info("Finished all sales category classification batches.")


# Function to build LLM prompts for batching
def build_industry_classification_prompt(company_details: dict, target_industry: str):
    """Builds a single prompt for LLM classification."""
    company_name = company_details.get("company", company_details.get("name", "Unknown Company"))
    industry = company_details.get("industry", "N/A")
    product_service = company_details.get("product_service", "N/A")

    industry_classification_prompt_template = PromptTemplate(
        input_variables=["company_name", "industry", "product_service", "target_industry"],
        template="""
Based on the provided company details, determine if the company '{company_name}'
(Industry: '{industry}', Product/Service: '{product_service}')
is primarily and fundamentally operating within the '{target_industry}' sector.
CRITICAL GUIDELINES:

A company is considered to be in the '{target_industry}' sector ONLY if its core business, main products, or primary services are directly and fundamentally tied to that industry.
Do NOT classify a company as '{target_industry}' if it merely provides services or technology to companies in that industry, or if it has a small, non-core division in that industry.
For 'Financial Technology' (FinTech), this means the company's main offering must be technology that automates, enhances, or delivers financial services. A traditional bank is NOT considered a FinTech company.
Respond with only "YES" or "NO".
"""
    )
    return industry_classification_prompt_template.format(
        company_name=company_name,
        industry=industry,
        product_service=product_service,
        target_industry=target_industry
    )

# Async function to classify company industries in batches
async def batch_classify_industries(profiles, required_industries):
    """
    Uses LLM to classify company industries for multiple profiles,
    processing in controlled batches to manage API limits.
    """
    all_classification_tasks = []
    prompts_to_cache_map = {} # Maps (company_name, req_industry) to redis_key

    for profile in profiles:
        for role in profile.get("roles", []):
            company_details = role.get("company_details", {})
            company_name = company_details.get("company", company_details.get("name", "Unknown Company"))
            for req_industry in required_industries:
                cache_key_data = f"{company_name}|{company_details.get('industry', 'N/A')}|{company_details.get('product_service', 'N/A')}|{req_industry.lower()}"
                cache_key_hash = hashlib.md5(cache_key_data.encode('utf-8')).hexdigest()
                redis_industry_key = f"industry_company_cache:{cache_key_hash}"
                cached_result = redis_client.get(redis_industry_key)

                if cached_result:
                    # Update profile with cached result immediately
                    profile['cached_industry_result'] = profile.get('cached_industry_result', {})
                    profile['cached_industry_result'][f"{company_name}|{req_industry}"] = cached_result.lower() == "yes"
                else:
                    # Create prompt and store mapping for later processing
                    prompt = build_industry_classification_prompt(company_details, req_industry)
                    all_classification_tasks.append((prompt, (company_name, req_industry), redis_industry_key))
    
    if not all_classification_tasks:
        logger.info("No new industry classifications needed (all cached or no requirements).")
        return

    logger.info(f"Total {len(all_classification_tasks)} industry classification tasks to perform.")

    # Process tasks in smaller, controlled batches
    for i in range(0, len(all_classification_tasks), MAX_INDUSTRY_CLASSIFICATIONS_PER_LLM_BATCH):
        batch_tasks_info = all_classification_tasks[i:i + MAX_INDUSTRY_CLASSIFICATIONS_PER_LLM_BATCH]
        llm_tasks = [async_llm.ainvoke(task_info[0]) for task_info in batch_tasks_info]

        logger.info(f"Sending batch of {len(llm_tasks)} industry classification prompts to LLM.")
        responses = await asyncio.gather(*llm_tasks)

        for j, response in enumerate(responses):
            original_prompt, (company_name, req_industry), redis_key = batch_tasks_info[j]
            llm_response_raw = response.content.strip().upper()
            is_in_industry = llm_response_raw == "YES"
            redis_client.setex(redis_key, 86400, "YES" if is_in_industry else "NO") # Cache for 24 hours
            #logger.debug(f"LLM classified '{profile_name}' in '{req_industry}': {is_in_industry} (Raw: {llm_response_raw})")

            # Update all profiles that match this company_name with the classification result
            for profile in profiles: # Iterate through all profiles to update relevant ones
                if any(r.get("company_details", {}).get("company") == company_name or \
                       r.get("company_details", {}).get("name") == company_name for r in profile.get("roles", [])):
                    profile['cached_industry_result'] = profile.get('cached_industry_result', {})
                    profile['cached_industry_result'][f"{company_name}|{req_industry}"] = is_in_industry
    logger.info("Finished all industry classification batches.")

# Filtering Functions

def check_role_specific_experience(profile, criteria, current_date, query_focus):
    """
    Checks if a candidate has the required years of experience for specific
    functional, segment, or industry roles, calculated conjunctively.
    """
    # MORE ROBUST FIX: Use 'or' to provide a default value if criteria returns None, then cast to float.
    min_exp = float(criteria.get("min_role_experience") or 0.0)
    max_exp = float(criteria.get("max_role_experience") or 'inf')
    
    # Only calculate duration if there are specific criteria to match against
    if criteria.get("required_functional_experience") or criteria.get("required_segments") or criteria.get("required_industries"):
        total_duration, roles_list = calculate_conjunctive_experience_duration(profile, criteria, query_focus)
        
        # Store details for the final response
        profile['contributing_roles_details'] = {
            'category': 'Combined Experience',
            'total_duration': total_duration,
            'roles': roles_list
        }
        
        if not (min_exp <= total_duration <= max_exp):
            logger.debug(f"{profile.get('name')} filtered out by conjunctive duration: {total_duration}yrs not in [{min_exp}, {max_exp}]")
            return False
        
        # Log evidence
        profile['evidence_log'] = profile.get('evidence_log', [])
        profile['evidence_log'].append({
            "criterion": "role_specific_experience",
            "source_type": "calculated_duration",
            "source_text": f"Candidate has a calculated {total_duration} years of experience matching the specific role criteria."
        })

    return True


def check_total_experience(profile, criteria, query_focus):
    """Checks total experience and logs evidence."""
    if "min_total_experience" not in criteria and "max_total_experience" not in criteria:
        return True
        
    # MORE ROBUST FIX: Use 'or' to provide a default value if criteria returns None, then cast to float.
    min_exp = float(criteria.get("min_total_experience") or 0.0)
    max_exp = float(criteria.get("max_total_experience") or 'inf')

    total_exp = profile.get("total_experience_years", 0.0)
    if not (min_exp <= total_exp <= max_exp):
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: total_experience_years {total_exp} not within {min_exp} to {max_exp}")
        return False
    
    # Log evidence
    profile['evidence_log'] = profile.get('evidence_log', [])
    profile['evidence_log'].append({
        "criterion": "total_experience",
        "source_type": "profile_summary",
        "source_text": f"Candidate has {total_exp} years of total experience."
    })
    return True

def check_functional_experience(profile, criteria, query_focus):
    """Checks functional experience and logs evidence."""
    req_funcs = criteria.get("required_functional_experience")
    if not req_funcs:
        return True

    profile_functional_texts = []
    evidence_sources = {} # To store where the keyword was found

    # Collect text from enriched functional roles
    for role in profile.get("functional_experience", {}).get("roles", []):
        for field in ["activity_type", "title", "details"]:
            if role.get(field):
                text = role[field].lower()
                profile_functional_texts.append(text)
                evidence_sources[text] = f"functional_role_{field}"

    # Collect text from main roles' titles and details
    for i, role in enumerate(profile.get("roles", [])):
        if role.get("title"):
            text = role["title"].lower()
            profile_functional_texts.append(text)
            evidence_sources[text] = f"role_{i}_title"
        if role.get("details"):
            text = role["details"].lower()
            profile_functional_texts.append(text)
            evidence_sources[text] = f"role_{i}_details"

    for req_func in req_funcs:
        req_func_lower = req_func.lower()
        found_match = False
        # Strict matching: requires the exact phrase or all keywords of the phrase
        for text, source in evidence_sources.items():
            if req_func_lower in text:
                profile['evidence_log'] = profile.get('evidence_log', [])
                profile['evidence_log'].append({
                    "criterion": req_func,
                    "source_type": source,
                    "source_text": text
                })
                found_match = True
                break # Found evidence for this required function
        if found_match:
            continue # Move to the next required function

    # If after checking all required functions, one is missing, the candidate fails
    logged_funcs = {log['criterion'].lower() for log in profile.get('evidence_log', []) if log['criterion'] in req_funcs}
    if not all(req.lower() in logged_funcs for req in req_funcs):
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_functional_experience {req_funcs}")
        return False
        
    return True

def check_customer_segments(profile, criteria, query_focus):
    """Checks customer segments and logs evidence."""
    req_segments = criteria.get("required_segments")
    if not req_segments:
        return True

    profile['evidence_log'] = profile.get('evidence_log', [])
    
    person_sources = {
        "headline": profile.get("headline", "").lower(),
        "about": profile.get("about", "").lower()
    }
    for i, role in enumerate(profile.get("roles", [])):
        person_sources[f"role_{i}_details"] = role.get("details", "").lower()
        person_sources[f"role_{i}_title"] = role.get("title", "").lower()

    company_sources = {}
    for i, role in enumerate(profile.get("roles", [])):
        company_sources[f"role_{i}_company_segment"] = ' '.join(
            [seg.lower() for seg in role.get("company_details", {}).get("customer_segment", [])]
        )

    for req_segment in req_segments:
        req_segment_lower = req_segment.lower()
        found_match_for_segment = False
        
        for source_name, source_text in person_sources.items():
            if any(syn.lower() in source_text for syn in SEGMENT_SYNONYMS.get(req_segment_lower, [])):
                profile['evidence_log'].append({
                    "criterion": req_segment,
                    "source_type": source_name.replace(f"_{source_name.split('_')[-1]}", "") if "_" in source_name else source_name,
                    "source_text": source_text
                })
                found_match_for_segment = True
                break
        
        if found_match_for_segment:
            continue

        if query_focus != "person":
            for source_name, source_text in company_sources.items():
                if any(syn.lower() in source_text for syn in SEGMENT_SYNONYMS.get(req_segment_lower, [])):
                    role_index = int(source_name.split('_')[1])
                    company_name = profile['roles'][role_index].get('company')
                    profile['evidence_log'].append({
                        "criterion": req_segment,
                        "source_type": "company_segment",
                        "source_text": f"at {company_name}, a company serving the {req_segment} segment."
                    })
                    found_match_for_segment = True
                    break
        
        if not found_match_for_segment:
            logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_segment {req_segment}")
            return False
            
    return True


def check_industries(profile, criteria, query_focus):
    """Checks industries and logs evidence."""
    req_industries = criteria.get("required_industries")
    if not req_industries:
        return True

    profile['evidence_log'] = profile.get('evidence_log', [])

    for req_industry in req_industries:
        req_industry_lower = req_industry.lower()
        found_industry_match_for_this_req = False
        
        for role in profile.get("industry_experience", {}).get("roles", []):
            if role.get("industry") and req_industry_lower in role["industry"].lower():
                profile['evidence_log'].append({ "criterion": req_industry, "source_type": "industry_role", "source_text": role["industry"] })
                found_industry_match_for_this_req = True
                break
        if found_industry_match_for_this_req:
            continue

        if query_focus != "person":
            for role in profile.get("roles", []):
                company_details = role.get("company_details", {})
                company_name = company_details.get("company", company_details.get("name", "Unknown Company"))
                company_industry_text = (company_details.get("industry", "") + " " + company_details.get("product_service", "")).lower()
                
                if req_industry_lower in company_industry_text:
                    profile['evidence_log'].append({ "criterion": req_industry, "source_type": "company_details", "source_text": company_industry_text })
                    found_industry_match_for_this_req = True
                    break
                
                cached_result_key = f"{company_name}|{req_industry}"
                if profile.get('cached_industry_result', {}).get(cached_result_key) is True:
                    profile['evidence_log'].append({ "criterion": req_industry, "source_type": "llm_classification", "source_text": f"LLM classified {company_name} as being in the {req_industry} industry." })
                    found_industry_match_for_this_req = True
                    break
        
        if not found_industry_match_for_this_req:
            logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_industry '{req_industry}'")
            return False
    return True

def check_people_managed(profile, criteria, query_focus):
    """Checks people managed and logs evidence."""
    if "min_people_managed" not in criteria and "max_people_managed" not in criteria:
        return True
        
    # MORE ROBUST FIX: Use 'or' to provide a default value if criteria returns None, then cast to int/float.
    min_people = int(criteria.get("min_people_managed") or 0)
    max_people = float(criteria.get("max_people_managed") or 'inf')

    max_managed = profile.get("team_management", {}).get("max_people_managed", 0)
    if not (min_people <= max_managed <= max_people):
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: max_people_managed {max_managed} not within {min_people} to {max_people}")
        return False
        
    # Log evidence
    profile['evidence_log'] = profile.get('evidence_log', [])
    profile['evidence_log'].append({
        "criterion": "people_managed",
        "source_type": "team_management_summary",
        "source_text": f"Candidate has managed a maximum of {max_managed} people."
    })
    return True

def check_sales_category(profile, criteria, query_focus):
    """Checks sales category and logs evidence."""
    req_category = criteria.get("required_sales_category")
    if not req_category:
        return True
    
    profile_category = profile.get("sales_category")
    if profile_category and profile_category.lower() == req_category.lower():
        profile['evidence_log'] = profile.get('evidence_log', [])
        profile['evidence_log'].append({
            "criterion": "sales_category",
            "source_type": "llm_classification",
            "source_text": f"Profile classified under sales category: {profile_category}"
        })
        return True
    
    logger.debug(f"{profile.get('name', 'Unknown')} filtered out: category '{profile_category}' does not match required '{req_category}'")
    return False

# ... (Keep other check functions like geographies, funding_stage, etc., but ensure they also log to evidence_log if they are used for filtering) ...

def check_geographies(profile, criteria, query_focus):
    req_geos = criteria.get("required_geographies")
    if not req_geos:
        return True

    profile_geographies = set()
    # Always check personal location
    if profile.get("location"):
        profile_geographies.add(profile["location"].lower())
        location_parts = [p.strip().lower() for p in profile["location"].replace(",", " ").replace("/", " ").split()]
        profile_geographies.update(location_parts)

    # Conditionally check company customer_presence and geography_experience regions
    # Only consider if query is not strictly person-centric
    if query_focus != "person":
        for role in profile.get("roles", []):
            if role.get("company_details", {}).get("customer_presence"):
                profile_geographies.update([g.lower() for g in role["company_details"]["customer_presence"]])
        if profile.get("geography_experience", {}).get("regions"):
            profile_geographies.update([g.lower() for g in profile["geography_experience"]["regions"]])


    for req_geo in req_geos:
        req_geo_lower = req_geo.lower()
        found_geo_match = False
        # Use the dynamically loaded EUROPEAN_COUNTRIES
        if req_geo_lower == "european country":
            if any(country in profile_geographies for country in EUROPEAN_COUNTRIES):
                found_geo_match = True
        else:
            if any(req_geo_lower in p_geo for p_geo in profile_geographies):
                found_geo_match = True
        
        if not found_geo_match:
            logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_geography {req_geo}")
            return False
    return True

def check_funding_stage(profile, criteria, query_focus):
    # Only relevant if query is company-centric or hybrid
    if query_focus == "person" or "required_funding_stage" not in criteria:
        return True
    
    req_stage = criteria.get("required_funding_stage")
    
    found_match = False
    for role in profile.get("roles", []):
        if role.get("company_details", {}).get("funding_stage", "").lower() == req_stage.lower():
            found_match = True
            break
    
    if not found_match:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_funding_stage {req_stage}")
        return False
    return True

def check_company_revenue_range(profile, criteria, query_focus):
    # Only relevant if query is company-centric or hybrid
    if query_focus == "person" or "required_company_revenue_range" not in criteria:
        return True
    
    rev_range = criteria.get("required_company_revenue_range")
    
    # FIX: Add a guard to prevent TypeError if rev_range is None
    if not rev_range:
        return True # If no range is specified, don't filter

    min_rev = rev_range[0] if len(rev_range) > 0 and rev_range[0] is not None else 0.0
    max_rev = rev_range[1] if len(rev_range) > 1 and rev_range[1] is not None else float('inf')
    
    found_match = False
    for role in profile.get("roles", []):
        revenue_str = role.get("company_details", {}).get("revenue", "")
        if revenue_str:
            try:
                revenue_value = float(revenue_str.replace("US$", "").replace(",", "").split()[0])
                if min_rev <= revenue_value <= max_rev:
                    found_match = True
                    break
            except (ValueError, IndexError):
                pass
    
    if not found_match:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: no revenue in range {min_rev} to {max_rev}")
        return False
    return True

def check_culture_type(profile, criteria, query_focus):
    # Only relevant if query is company-centric or hybrid
    if query_focus == "person" or "required_culture_type" not in criteria:
        return True
    
    req_culture = criteria.get("required_culture_type")
    
    found_match = False
    for role in profile.get("roles", []):
        if role.get("company_details", {}).get("culture_type", "").lower() == req_culture.lower():
            found_match = True
            break
    
    if not found_match:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_culture_type {req_culture}")
        return False
    return True

def check_product_service_keywords(profile, criteria, query_focus):
    """
    Checks for required product/service keywords using both exact matching and semantic similarity.
    Only relevant if query is company-centric or hybrid.
    """
    if query_focus == "person" or "required_product_service_keywords" not in criteria:
        return True

    req_keywords = criteria.get("required_product_service_keywords")

    profile_product_service_texts = [
        role.get("company_details", {}).get("product_service", "").lower()
        for role in profile.get("roles", [])
    ]
    # Filter out empty strings before embedding to avoid errors with empty inputs
    profile_embeddings = [embedding.embed_query(text) for text in profile_product_service_texts if text]

    for req_keyword in req_keywords:
        req_keyword_lower = req_keyword.lower()
        req_keyword_words = req_keyword_lower.split()

        # Exact match check
        if any(all(word in text for word in req_keyword_words) for text in profile_product_service_texts):
            logger.debug(f"{profile.get('name')} matched by exact keyword: '{req_keyword}'")
            return True

        # Semantic similarity check
        try:
            req_keyword_embedding = embedding.embed_query(req_keyword_lower)
            for ps_embedding in profile_embeddings:
                similarity = cosine_similarity(req_keyword_embedding, ps_embedding)
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    logger.debug(f"{profile.get('name')} matched by semantic similarity for '{req_keyword}' (score: {similarity:.2f})")
                    return True
        except Exception as e:
            logger.error(f"Error during semantic similarity check for '{req_keyword}': {e}")
            
    logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_product_service_keywords {req_keywords} (no exact or semantic match)")
    return False

def check_degrees(profile, criteria, query_focus):
    # This is a person-centric check, always relevant if specified
    req_degrees = criteria.get("required_degrees")
    if not req_degrees:
        return True

    profile_degrees = set()
    for edu in profile.get("education", []):
        if edu.get("degree"):
            profile_degrees.add(edu["degree"].lower())

    found_any_match = False
    for req_degree in req_degrees:
        if req_degree.lower() in profile_degrees:
            found_any_match = True
            break
    
    if not found_any_match:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_degrees {req_degrees}")
        return False
    return True

def check_skills(profile, criteria, query_focus):
    # This is a person-centric check, always relevant if specified
    req_skills = criteria.get("required_skills")
    if not req_skills:
        return True

    profile_skills_raw = profile.get("raw_fields", {}).get("Skills", "").lower()
    found_all_skills = True
    for req_skill in req_skills:
        if req_skill.lower() not in profile_skills_raw:
            found_all_skills = False
            break
    
    if not found_all_skills:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_skills {req_skills}")
        return False
    return True

def check_business_model(profile, criteria, query_focus):
    # Only relevant if query is company-centric or hybrid
    if query_focus == "person" or "required_business_model" not in criteria:
        return True
    
    req_model = criteria.get("required_business_model")
    
    found_match = False
    for role in profile.get("roles", []):
        if role.get("company_details", {}).get("business_model", "").lower() == req_model.lower():
            found_match = True
            break
    
    if not found_match:
        logger.debug(f"{profile.get('name', 'Unknown')} filtered out: missing required_business_model {req_model}")
        return False
    return True

# FIX: Refactored the filtering logic into a single comprehensive function
# This ensures all criteria are checked for each candidate.
async def filter_candidates_by_criteria(profiles: list, current_date: str, query_focus: str = "both", **criteria) -> list:
    """
    Filters candidate profiles based on a comprehensive set of criteria.
    This function acts as a reliable Python tool for precise filtering.
    """
    # First, dynamically determine sales categories for all profiles in the pool
    await batch_determine_sales_categories(profiles, SALES_TAXONOMY)

    matching_candidates = []
    
    # Store LLM classification results for industries on profiles directly
    if "required_industries" in criteria and criteria["required_industries"] and query_focus != "person":
        await batch_classify_industries(profiles, criteria["required_industries"])

    # A list of all check functions to be applied
    ALL_FILTER_CHECKS = [
        check_role_specific_experience,
        check_total_experience,
        check_functional_experience,
        check_customer_segments,
        check_industries,
        check_people_managed,
        check_geographies,
        check_funding_stage,
        check_company_revenue_range,
        check_culture_type,
        check_product_service_keywords,
        check_degrees,
        check_skills,
        check_business_model,
        check_sales_category
    ]

    for profile in profiles:
        # Reset evidence log for each profile before filtering
        profile['evidence_log'] = []
        all_criteria_met = True
        for check_func in ALL_FILTER_CHECKS:
            # The check_func is called with (profile, criteria, ...)
            # We need to adapt the call for check_role_specific_experience
            if check_func == check_role_specific_experience:
                if not check_func(profile, criteria, current_date, query_focus):
                    all_criteria_met = False
                    break
            else:
                if not check_func(profile, criteria, query_focus):
                    all_criteria_met = False
                    break
        
        if all_criteria_met:
            matching_candidates.append(profile)
            logger.info(f"{profile.get('name', 'Unknown')} matches all criteria.")
        else:
            logger.info(f"{profile.get('name', 'Unknown')} did NOT match all criteria.")
            
    logger.info(f"Found {len(matching_candidates)} matching candidates after detailed filtering. Full criteria: {json.dumps(criteria)}")
    return matching_candidates


# FIX: Updated to log the reason for acceptance or rejection from the LLM.
# This prompt is now extremely strict about person vs. company focus.
async def sales_specialist_llm_validation(original_query: str, extracted_criteria: dict, candidates: List[dict]) -> List[dict]:
    """
    Acts as a 'sales super specialist' to review and validate the top candidates
    based on the original query and extracted criteria.
    """
    if not candidates:
        return []

    validated_candidates = []
    
    for i in range(0, len(candidates), MAX_SALES_SPECIALIST_VALIDATIONS_PER_LLM_BATCH):
        batch = candidates[i:i + MAX_SALES_SPECIALIST_VALIDATIONS_PER_LLM_BATCH]
        
        validation_tasks = []
        for candidate in batch:
            # Make a copy of the candidate profile and remove irrelevant keys before sending to LLM
            # This reduces token count and focuses the LLM on the most important data.
            profile_for_llm = {
                "name": candidate.get("name"),
                "headline": candidate.get("headline"),
                "about": candidate.get("about"),
                "total_experience_years": candidate.get("total_experience_years"),
                "roles": candidate.get("roles"),
                "team_management": candidate.get("team_management"),
                "segment_experience": candidate.get("segment_experience"),
                "contributing_roles_details": candidate.get("contributing_roles_details"),
                "evidence_log": candidate.get("evidence_log", []) # Pass the evidence log
            }
            profile_summary = json.dumps(profile_for_llm, indent=2)
            
            validation_prompt_template = PromptTemplate(
                input_variables=["original_query", "extracted_criteria", "profile_summary"],
                template="""
You are an extremely meticulous and precise Sales Super Specialist. Your task is to act as a final, strict verifier.
You must review a candidate's profile against the user's query and its extracted criteria.
Your decision must be based *only* on the evidence presented in the `evidence_log`.

**Original User Query:** "{original_query}"

**Extracted Filtering Criteria (JSON):**
{extracted_criteria}

**Candidate Profile (JSON with evidence_log):**
{profile_summary}

**MANDATORY INSTRUCTIONS FOR VALIDATION:**

1.  **Evidence is Paramount:** Your decision **MUST** be based *exclusively* on the `evidence_log` provided within the candidate's JSON. Do not infer or assume any information that is not explicitly logged as evidence.
2.  **Verify Every Criterion:** For every criterion in the `extracted_criteria` (e.g., "required_segments", "min_people_managed"), you must find a corresponding entry in the `evidence_log`. If any criterion is not supported by an entry in the `evidence_log`, you **MUST** decide "NO".
3.  **Check Query Focus:** Pay strict attention to the `query_focus`.
    - If `query_focus` is "person", the `source_type` for the evidence in the `evidence_log` **MUST** be one of `['headline', 'about', 'role_title', 'role_details', 'team_management_summary', 'profile_summary', 'calculated_duration']` or similar person-centric sources. If the only evidence for a person-focused criterion comes from a `source_type` of `company_segment`, you **MUST** decide "NO".
4.  **Verify Numerical Criteria:** Strictly verify numerical requirements (e.g., `min_people_managed`, `min_role_experience`) against the candidate's JSON data (`team_management` or `contributing_roles_details`). The `evidence_log` should also confirm this. If these are not met, the decision is "NO".
5.  **No Exceptions:** If there is any doubt or if the evidence is not explicit and directly traceable via the `evidence_log`, you **MUST** reject the candidate to prevent false positives.

**Output Format:** Respond **ONLY** with a valid JSON object containing two keys:
    - "decision": "YES" or "NO".
    - "reason": A brief, one-sentence explanation for your decision, explicitly stating which criterion was not met or confirming that all criteria were met based on the `evidence_log`.

**Example of a "NO" Decision:**
```json
{{
  "decision": "NO",
  "reason": "The `evidence_log` lacks an entry to support the 'min_people_managed: 5' criterion."
}}
```
**Example of a "YES" Decision:**
```json
{{
  "decision": "YES",
  "reason": "All criteria were met and explicitly supported by entries in the `evidence_log` with appropriate source types."
}}
```
**Final Answer (JSON Object):**
"""
            )
            validation_prompt = validation_prompt_template.format(
                original_query=original_query,
                extracted_criteria=json.dumps(extracted_criteria, indent=2),
                profile_summary=profile_summary
            )
            validation_tasks.append((validation_prompt, candidate))

        llm_tasks = [sales_specialist_llm.ainvoke(task_info[0]) for task_info in validation_tasks]
        logger.info(f"Sending batch of {len(llm_tasks)} sales specialist validation prompts to LLM.")
        responses = await asyncio.gather(*llm_tasks)

        for j, response in enumerate(responses):
            _, candidate = validation_tasks[j]
            try:
                # Clean the response for JSON parsing
                response_content = response.content.strip()
                if response_content.startswith("```json"):
                    response_content = response_content.lstrip("```json").rstrip("```").strip()
                
                validation_data = json.loads(response_content)
                decision = validation_data.get("decision", "NO").upper()
                reason = validation_data.get("reason", "No reason provided.")

                if decision == "YES":
                    validated_candidates.append(candidate)
                    logger.info(f"Sales specialist ACCEPTED: {candidate.get('name', 'Unknown')} (Reason: {reason})")
                else:
                    logger.info(f"Sales specialist REJECTED: {candidate.get('name', 'Unknown')} (Reason: {reason})")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Failed to parse sales specialist validation response for {candidate.get('name', 'Unknown')}: {e}. Raw response: {response.content.strip()}")
                logger.info(f"Sales specialist REJECTED (due to parsing error): {candidate.get('name', 'Unknown')}")
                
    return validated_candidates


# FIX: Updated prompt for more detailed and source-attributed reasoning
async def generate_response_in_batches(query: str, matching_profiles: List[dict], current_date: str) -> AsyncIterator[str]:
    """
    Generates the final response by processing matching profiles in batches,
    streaming the output, and numbering each profile.
    """
    if not matching_profiles:
        yield "No candidates were found that strictly match the given criteria with explicit evidence in their profiles."
        return

    yield f"Found {len(matching_profiles)} matching candidates. Here are the details:\n\n"

    candidate_number = 1 # Initialize candidate counter
    # Split profiles into chunks for batch processing
    for i in range(0, len(matching_profiles), MAX_PROFILES_PER_FINAL_LLM_CHUNK):
        chunk = matching_profiles[i:i + MAX_PROFILES_PER_FINAL_LLM_CHUNK]
        
        # Add temporary numbering to each profile in the chunk for LLM's prompt
        numbered_chunk = []
        for profile in chunk:
            numbered_profile = profile.copy() # Avoid modifying original profile
            numbered_profile['display_number'] = candidate_number
            # Pass the `contributing_roles_details` and `evidence_log` to the LLM
            numbered_profile['contributing_roles_details'] = profile.get('contributing_roles_details', {})
            numbered_profile['evidence_log'] = profile.get('evidence_log', [])
            candidate_number += 1 # Increment for the next profile
            numbered_chunk.append(numbered_profile)
        

        chunk_json = json.dumps(numbered_chunk, indent=2)

        # This prompt is heavily modified for perfect, transparent reasoning with source attribution.
        final_answer_prompt_template = PromptTemplate(
            input_variables=["query", "matching_profiles_json", "current_date"],
            template="""
You are a helpful chatbot providing concise and precise answers about candidate profiles.
Based on the user's query and the provided list of matching candidate profiles (in JSON format), generate a clear and satisfying answer.

**CRITICAL INSTRUCTIONS FOR ANSWER GENERATION:**

1.  **Basic Information:** For each candidate, present their **Name**, **LinkedIn URL**, and **Location**.
2.  **Evidence-Based Reasoning:** For each candidate, you **MUST** provide a section titled "**Reasoning**". In this section:
    - Synthesize the evidence into a narrative. Do not just list facts.
    - For each criterion from the query (e.g., "SMB sales", "managed 30 people"), create a single, clear sentence that explains how the candidate meets it.
    - **Source Attribution is MANDATORY.** You must state *exactly where* in the profile the evidence was found. Use the `evidence_log` provided for each candidate to find the `source_type`.
        - If `source_type` is 'headline', 'about', 'role_title', or 'role_details', state it clearly and explicitly.
        - **Example of perfect reasoning:** "He meets the criteria for **SMB Sales** as mentioned in the **details of his role** as a Sales Manager at Acme Inc., which specifies a focus on the small business segment."
        - **Another example:** "She satisfies the requirement of managing **at least 30 people**, as stated in the **details of her role** at Globex Corp where she led a team of 45."
        - If the evidence is from the company's focus (and the query allows it), state that. Example: "...based on their time at connecTECH, a company that serves the **SMB segment**."
    - Be direct and reference exact values from their profile. If no explicit evidence exists for a criterion, do not mention the candidate.

3.  **Transparent Experience Calculation:**
    - If the candidate's JSON data includes a `contributing_roles_details` section with a `total_duration` > 0, you **MUST** use it to explain their relevant experience.
    - First, state the total duration clearly. Example: "**Total Experience in SMB Sales: 12.50 years**"
    - Immediately below, create a sub-list titled "**Experience Breakdown:**" and list each company, role title, and `duration_years` from the `roles` list within `contributing_roles_details`.

4.  **Formatting:**
    - Present each candidate in a clear, distinct section, prefixed with their display_number (e.g., "**1. John Doe**").
    - Use Markdown for formatting (e.g., bolding, bullet points).
    - Ensure strong visual separation between each candidate's summary.

**Example Output Structure:**

**1. Santosh D'Souza**
- **LinkedIn:** [URL]
- **Location:** [Location]

**Reasoning:**
- He meets the criteria for **SMB Sales** from his role as Director of Sales at LinkedIn, as the **details of his role** mention leading a team responsible for customer acquisition including SMB clients.
- He has managed **at least 30 people**, as the **details of his role** at LinkedIn state he leads a team of 45 sales reps.

**Total Experience in SMB Sales: 21.68 years**
**Experience Breakdown:**
- LinkedIn, Director Sales, Talent Solutions: 0.85 years
- LinkedIn, Head of Sales - Talent Solutions: 3.09 years
- ... and so on.

---

Original Query: {query}
Matching Candidates (JSON) Chunk: {matching_profiles_json}
Current Date: {current_date}

Answer for this chunk:
"""
        )
        
        final_prompt_formatted = final_answer_prompt_template.format(
            query=query,
            matching_profiles_json=chunk_json,
            current_date=current_date
        )
        
        logger.info(f"Sending final answer prompt chunk to LLM (for {len(chunk)} profiles):\n{final_prompt_formatted[:500]}...")
        final_prompt_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(final_prompt_formatted))
        logger.info(f"Final answer prompt chunk tokens: {final_prompt_tokens}")

        # Stream the response for each chunk
        stream_response = streaming_llm.stream(final_prompt_formatted)
        for chunk_response in stream_response:
            yield chunk_response.content
        # After each chunk is done, yield a status message if there are more chunks
        # Only yield the "Fetching more" message if there are indeed more profiles to process
        if i + MAX_PROFILES_PER_FINAL_LLM_CHUNK < len(matching_profiles): 
             yield "\n\n_Fetching more candidate summaries..._\n\n" # Textual indicator for a new batch being processed
        yield "\n\n---\n\n" # Separator between batches

# Main Query Processing Logic (now asynchronous)
async def process_query(query: str, session_id: str) -> AsyncIterator[str]:
    """Process user query using LLM for criteria extraction and Python for filtering."""
    if not query.strip():
        yield "Please enter a valid query."
        return
    
    cache_key = get_cache_key(query)
    cached_result = redis_client.get(f"query:{cache_key}")
    if cached_result:
        try:
            decoded_result = cached_result
            logger.info(f"Cache hit for query key: query:{cache_key}")
            yield decoded_result
            return
        except UnicodeDecodeError as e:
            logger.error(f"UnicodeDecodeError for cache key query:{cache_key}: {e}. Deleting corrupted cache entry.")
            redis_client.delete(f"query:{cache_key}")
            yield f"Invalid cache data for query '{query}'. Cleared cache and processing afresh.\n"

    history_key = f"history:{session_id}"
    # Fetch previous turns for context. Limiting to last 2 turns (user Q + assistant A)
    # The list is [Q1, A1, Q2, A2, ..., Q_last, A_last]
    raw_history = redis_client.lrange(history_key, -4, -1) # Get last 2 Q&A pairs
    # Format for prompt: [{"role": "user", "content": "..."}]
    formatted_history = []
    for i in range(0, len(raw_history), 2):
        if i + 1 < len(raw_history):
            formatted_history.append({"role": "user", "content": raw_history[i].replace("Q: ", "")})
            formatted_history.append({"role": "assistant", "content": raw_history[i+1].replace("A: ", "")})

    current_date = datetime.now().strftime("%Y-%m-%d")

    # FIX: Prompt updated with clearer definitions to improve mapping accuracy.
    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "chat_history"],
        template="""
You are an expert assistant designed to extract structured filtering criteria from a user's query about candidate profiles.
You MUST strictly adhere to the JSON format and the exact values/types shown in the examples.

**Sales Role Definitions (for `required_sales_category`):**

- **Hunting:** Acquiring new customers (e.g., "Account Executive", "net new", "new logos"). Map queries about new customer acquisition here.
- **Farming:** Managing and selling to existing customers (e.g., "Account Manager", "retention", "renewals").
- **Sales Development:** Lead generation, not closing deals (e.g., "SDR", "BDR"). **IMPORTANT:** General "Business Development" or "Account Development" maps to this category unless the query explicitly mentions "new accounts" or "net new logos".
- **Partner Sales:** Selling through partners or channels (e.g., "Partner Sales", "Channel Sales", "alliance management"). Queries with these terms map here.
- **Customer Success:** Focus on customer retention and adoption post-sale.

**Instructions:**

- For experience duration in a specific area (e.g., "10 years in SMB sales"), use `min_role_experience` and combine `required_functional_experience` and `required_segments`.
- For total career duration, use `min_total_experience`.
- For phrases like "more than X years" or "at least X years", set the `min_` value (e.g., `min_role_experience: X.0`).
- **`query_focus` is CRITICAL.** You must determine if the query is about the **person's** direct experience, the **company's** attributes, or **both**.
  - **"person":** Query about individual skills, education, total experience, team management, or roles they personally held (e.g., "candidates with an MBA", "managed 5 people"). For these queries, analysis should be strictly based on the `details` of their roles, not general company info.
  - **"company":** Query about attributes of their employers (e.g., "worked at Series B companies", "experience in SaaS companies").
  - **"both":** Query combines both (e.g., "sales experience in tech companies").

**Available criteria keys:**

min_role_experience (float), max_role_experience (float), min_total_experience (float), max_total_experience (float),
required_functional_experience (list of strings), required_segments (list of strings), required_industries (list of strings),
min_team_management_score (float), min_people_managed (int), max_people_managed (int),
required_geographies (list of strings), required_funding_stage (string), required_company_revenue_range (list of two floats),
required_culture_type (string), required_product_service_keywords (list of strings), required_degrees (list of strings),
required_skills (list of strings), required_business_model (string),
required_sales_category (string): One of "Hunting", "Farming", "Sales Development", "Partner Sales", "Customer Success".
query_focus (string): STRICTLY one of "person", "company", or "both".

**Example:**
User Query: "Find hunters with at least 5 years in technology companies who have managed more than 3 people."
JSON Criteria:
```json
{{
  "required_sales_category": "Hunting",
  "min_role_experience": 5.0,
  "required_industries": ["Technology"],
  "min_people_managed": 3,
  "query_focus": "both"
}}
```
Chat History:
{chat_history}

Current User Query: {query}
JSON Criteria:
"""
    )
    
    try:
        criteria_prompt_formatted = criteria_extraction_prompt.format(query=query, chat_history=formatted_history)
        logger.info(f"Sending criteria extraction prompt to LLM:\n{criteria_prompt_formatted[:1000]}...")
        prompt_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(criteria_prompt_formatted))
        logger.info(f"Criteria extraction prompt tokens: {prompt_tokens}")

        llm_response_raw = llm.invoke(criteria_prompt_formatted).content.strip()
        logger.info(f"LLM raw criteria response: {llm_response_raw}")

        # Clean the response for JSON parsing
        if llm_response_raw.startswith("```json"):
            llm_response_raw = llm_response_raw.lstrip("```json").rstrip("```").strip()
        elif llm_response_raw.startswith("json"):
            llm_response_raw = llm_response_raw.lstrip("json").strip()

        extracted_criteria = json.loads(llm_response_raw)
        logger.info(f"Extracted criteria JSON for filtering: {json.dumps(extracted_criteria, indent=2)}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON criteria from LLM: {e}. Raw response: {llm_response_raw}")
        yield ("I couldn't parse the filtering criteria from your query. Please try rephrasing it "
                "(e.g., 'Candidates with at least 10 years in SMB sales and managed at least 30 people').")
        return
    except Exception as e:
        logger.error(f"Error during criteria extraction LLM call: {e}")
        yield f"An error occurred while extracting criteria from your query: {e}"
        return

    # --- Start of Keyword Expansion Logic ---
    # This logic is wrapped to handle cases where no keywords need expansion
    
    # Helper function to create and invoke the keyword expansion prompt
    async def expand_keywords(keywords_to_expand, keyword_type_log):
        # Dynamically build the template string to correctly include dynamic data
        allowed_categories = ", ".join(f'"{cat}"' for cat in SALES_TAXONOMY.keys())
        sales_taxonomy_for_prompt = json.dumps(SALES_TAXONOMY, indent=2).replace("{", "{{").replace("}", "}}")

        # Note: In an f-string, {{ and }} create literal { and } for the final string.
        # The placeholder for LangChain {initial_keywords} needs to be written as {{initial_keywords}}.
        # The JSON examples need to be escaped as well, e.g., {{"keyword":...}}
        keyword_expansion_template_str = f"""
You are an AI assistant specialized in generating semantically similar keywords and synonyms for candidate profile searches.
Given a list of initial keywords from a user's query, produce a JSON array of expanded keywords geared specifically for matching fields on professional profiles (role titles, activity phrases, functional labels, and skills). Use the SALES taxonomy below to map each expansion.

**MANDATORY OUTPUT FORMAT:**

Return ONLY valid JSON (no markdown, no commentary, no extra text).
JSON must be an array of objects (maximum 12 objects).
Each object must contain exactly these keys:

"keyword" (string) — the expanded term exactly as it might appear in a profile (e.g., "Account Executive", "SDR", "net new").
"category" (string) — one of: {allowed_categories}, or "Other".
"type" (string) — one of: "title", "activity", "skill", "abbrev".
"confidence" (number) — a value between 0.00 and 1.00 (use two decimal places) expressing how confident you are that this keyword belongs to the category.


Do not return more than 12 objects.
No duplicate keywords (case-insensitive). If two candidates differ only by punctuation/case, keep one canonical form.
Prioritise profile-realistic terms that commonly appear in LinkedIn/resumes. Avoid marketing copy.
Return items ordered roughly by descending confidence (highest confidence first).

SALES TAXONOMY (use these verbatim for mapping):
{sales_taxonomy_for_prompt}

REGIONAL / DATASET NOTES (important):

Include common regional/UK abbreviations and title variants likely in your dataset (e.g., "BDM", "KAM" for Key Account Manager, "Key Account Manager", "Sales Director", "Territory Manager", "Account Director").
Prefer canonical title text (e.g., "Account Executive" vs. "Account Exec") but include common abbreviations as separate objects where helpful (type "abbrev").

TYPE RULES:

"abbrev" — for short abbreviations (e.g., "AE", "SDR", "BDR", "CSM", "AM", "BDM", "KAM").
"title" — for multi-word job titles (e.g., "Account Manager", "Sales Director").
"activity" — for short activity phrases (e.g., "net new", "new logos", "renewals", "client retention", "upsell").
"skill" — for explicit skill phrases common in profile Skills sections (e.g., "pipeline management", "renewal management").

CONFIDENCE GUIDELINES:

Use two decimal places. Examples:

Very precise and canonical (e.g., "Account Executive") -> 0.95
Strong but slightly broader (e.g., "net new") -> 0.90
Ambiguous or context-dependent (e.g., "Business Development") -> <= 0.60 and category "Other" or the most-likely category with lower confidence.


If the keyword is commonly used in multiple categories but you must pick one, choose the best single category and lower confidence accordingly.

BEHAVIOR RULES (summary):

Keep max 12 objects; pick the most useful, high-precision terms first.
No duplicates (case-insensitive).
For ambiguous inputs from {{initial_keywords}}, expand conservatively: favor high-precision role titles and abbreviations, then high-value activity phrases, then broader skills if space remains.
Include at least one abbreviation where highly common in profiles (e.g., AE/AM/SDR) if applicable.
Order output by descending confidence.

EXAMPLES (follow this output structure exactly):
Input: ["hunting","new accounts"]
Output:
[
{{"keyword":"Account Executive","category":"Hunting","type":"title","confidence":0.95}},
{{"keyword":"net new","category":"Hunting","type":"activity","confidence":0.90}},
{{"keyword":"AE","category":"Hunting","type":"abbrev","confidence":0.85}},
{{"keyword":"new logos","category":"Hunting","type":"activity","confidence":0.88}}
]
Input: ["farming","retention"]
Output:
[
{{"keyword":"Account Manager","category":"Farming","type":"title","confidence":0.95}},
{{"keyword":"client retention","category":"Farming","type":"activity","confidence":0.90}},
{{"keyword":"renewals","category":"Farming","type":"activity","confidence":0.85}}
]
Now expand the following initial keywords into a JSON array following the rules above.
Initial Keywords: {{initial_keywords}}
"""
        # Create the PromptTemplate object with the dynamically generated string
        keyword_expansion_prompt = PromptTemplate(
            input_variables=["initial_keywords"],
            template=keyword_expansion_template_str.replace("{{initial_keywords}}", "{initial_keywords}")
        )
        
        try:
            expanded_keywords_raw = llm.invoke(keyword_expansion_prompt.format(initial_keywords=keywords_to_expand)).content.strip()
            logger.info(f"LLM raw expanded {keyword_type_log} keywords response: {expanded_keywords_raw}")
            if expanded_keywords_raw.startswith("```json"):
                expanded_keywords_raw = expanded_keywords_raw.lstrip("```json").rstrip("```").strip()
            elif expanded_keywords_raw.startswith("json"):
                expanded_keywords_raw = expanded_keywords_raw.lstrip("json").strip()
            expanded_keywords_list = json.loads(expanded_keywords_raw)
            # Replace semantic similarity with exact keyword matching after expansion
            combined_keywords = list(set(keywords_to_expand + [item['keyword'] for item in expanded_keywords_list if 'keyword' in item]))
            logger.info(f"Expanded {keyword_type_log} keywords to: {combined_keywords}")
            return combined_keywords
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for expanded {keyword_type_log} keywords: {e}. Raw response: {expanded_keywords_raw}")
            return keywords_to_expand # Return original keywords on failure
        except Exception as e:
            logger.error(f"Error during {keyword_type_log} keyword expansion LLM call: {e}")
            return keywords_to_expand # Return original keywords on failure

    if "required_product_service_keywords" in extracted_criteria and extracted_criteria["required_product_service_keywords"]:
        extracted_criteria["required_product_service_keywords"] = await expand_keywords(
            extracted_criteria["required_product_service_keywords"], 
            "product/service"
        )

    if "keywords" in extracted_criteria and extracted_criteria["keywords"]:
        extracted_criteria["keywords"] = await expand_keywords(
            extracted_criteria["keywords"], 
            "sales"
        )
    # --- End of Keyword Expansion Logic ---

    initial_candidate_names = set()
    search_query_parts = []

    if "required_product_service_keywords" in extracted_criteria and extracted_criteria["required_product_service_keywords"]:
        search_query_parts.extend(extracted_criteria["required_product_service_keywords"])
    if "required_industries" in extracted_criteria and extracted_criteria["required_industries"]:
        search_query_parts.extend(extracted_criteria["required_industries"])
    if "required_sales_category" in extracted_criteria:
        category = extracted_criteria["required_sales_category"]
        # Use SALES_TAXONOMY keywords for initial FAISS search if a category is specified
        if category in SALES_TAXONOMY:
            search_query_parts.extend(SALES_TAXONOMY[category])
    if "required_functional_experience" in extracted_criteria:
        search_query_parts.extend(extracted_criteria["required_functional_experience"])

    if search_query_parts:
        combined_search_query = " ".join(search_query_parts)
        logger.info(f"Performing initial FAISS search with query: '{combined_search_query}'")
        # Change k to len(profiles_data) to retrieve all relevant candidates initially
        initial_search_results = vector_store.similarity_search(combined_search_query, k=len(profiles_data))

        for doc in initial_search_results:
            if getattr(doc, "metadata", None) and "name" in doc.metadata and doc.metadata["name"] in profiles_dict:
                initial_candidate_names.add(doc.metadata["name"])

        if initial_candidate_names:
            candidates_for_detailed_filter = [profiles_dict[name] for name in initial_candidate_names]
            logger.info(f"Reduced candidate pool to {len(candidates_for_detailed_filter)} based on initial FAISS search.")
        else:
            candidates_for_detailed_filter = profiles_data
            logger.info("Initial FAISS search yielded no results, falling back to full profile data for filtering.")
    else:
        candidates_for_detailed_filter = profiles_data
        logger.info("No specific keywords or industries for initial search, filtering on full profile data.")

    # Get query focus to pass to filter functions
    query_focus = extracted_criteria.pop('query_focus', 'both') # Extract and remove query_focus

    # Perform batch industry classification if required_industries are present AND query is not strictly person-centric
    if "required_industries" in extracted_criteria and extracted_criteria["required_industries"] and query_focus != "person":
        await batch_classify_industries(candidates_for_detailed_filter, extracted_criteria["required_industries"])

    # This is now an async call
    matching_profiles = await filter_candidates_by_criteria(
        profiles=candidates_for_detailed_filter,
        current_date=current_date,
        query_focus=query_focus, # Pass query_focus here
        **extracted_criteria
    )
    logger.info(f"Found {len(matching_profiles)} matching candidates after detailed filtering.")

    # --- Sales Specialist LLM Validation Step ---
    if matching_profiles:
        yield "_Performing final validation with Sales Specialist AI..._\n\n"
        final_validated_candidates = await sales_specialist_llm_validation(query, extracted_criteria, matching_profiles)
        logger.info(f"Sales specialist validated {len(final_validated_candidates)} out of {len(matching_profiles)} candidates.")
        matching_profiles = final_validated_candidates
    # --- End Sales Specialist LLM Validation Step ---


    # All matching profiles are now passed to generate_response_in_batches.
    response_generator = generate_response_in_batches(query, matching_profiles, current_date)
    
    # Concatenate streamed parts to store in cache and history
    full_response_content = []
    async for part in response_generator:
        full_response_content.append(part)
        yield part # Yield parts to Streamlit for real-time display

    final_response_str = "".join(full_response_content)
    # Cache the full concatenated response
    redis_client.setex(f"query:{cache_key}", 3600, final_response_str) # Cache for 1 hour

    # Store chat history
    redis_client.rpush(history_key, f"Q: {query}", f"A: {final_response_str}")
    redis_client.expire(history_key, 86400) # Expire history after 24 hours
    logger.info(f"Stored history for session: {session_id}")


# Streamlit UI
st.set_page_config(page_title="Universal Candidate Chatbot", layout="wide")
st.title("🤖 GROWTON AI")
st.markdown("Ask any question about candidate profiles (e.g., experience, skills, education, location, etc.)")

if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(os.urandom(16)).encode()).hexdigest()
    logger.info(f"New session created: {st.session_state.session_id}")

if st.button("🔄 Clear Chat & Cache"):
    try:
        redis_client.delete(f"history:{st.session_state.session_id}")
        keys = redis_client.keys("query:*")
        if keys:
            for key in keys:
                redis_client.delete(key)
        st.success("Chat history and query cache cleared successfully!")
        logger.info("Cleared chat history and query cache via UI button.")
        st.session_state.messages = [] # Clear Streamlit messages
    except Exception as e:
        st.error(f"Error clearing chat history and cache: {e}")
        logger.error(f"Error clearing chat history and cache via UI button: {e}")

# Initialize messages list for Streamlit chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load chat history from Redis if it exists and sync with session_state
history_key = f"history:{st.session_state.session_id}"
current_redis_history = redis_client.lrange(history_key, 0, -1)

# Only sync if Redis history is different from current session_state messages
if len(st.session_state.messages) * 2 != len(current_redis_history):
    st.session_state.messages = [] # Clear current messages to re-sync
    for i in range(0, len(current_redis_history), 2):
        user_query = current_redis_history[i].replace("Q: ", "")
        assistant_response = current_redis_history[i+1].replace("A: ", "")
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Type your question here...")

if query:
    # Add user message to UI immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process query and display assistant response
    with st.chat_message("assistant"):
        # Use st.write_stream to handle the async generator output
        response_generator = process_query(query, st.session_state.session_id)
        st.write_stream(response_generator)
    
    # After the streaming is complete (st.write_stream finishes),
    # refresh the session state messages from Redis to ensure consistency.
    history_key_after_process = f"history:{st.session_state.session_id}"
    current_redis_history_after_process = redis_client.lrange(history_key_after_process, 0, -1)
    
    # Find the last assistant response
    if current_redis_history_after_process:
        last_assistant_response_from_redis = current_redis_history_after_process[-1].replace("A: ", "")
        # Check if the last message in session_state is already this, to avoid duplicates
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != last_assistant_response_from_redis:
            # We already added the user query above. Now add the full assistant response.
            st.session_state.messages.append({"role": "assistant", "content": last_assistant_response_from_redis})

# Display dataset summary in sidebar
st.sidebar.subheader("📊 Dataset Summary")
st.sidebar.write(f"Total Candidates: {len(profiles_data)}")


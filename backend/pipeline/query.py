import asyncio
import os
import json
import logging
import hashlib
import redis
import tiktoken
import copy
import pandas as pd
import io
import psycopg2
from typing import List, Dict, Any, AsyncIterator, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from backend.db.connection import get_db_connection, return_db_connection

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Pricing and Token Configuration ---
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0}
}
tokenizer = tiktoken.get_encoding("cl100k_base")

class TokenCostTracker:
    """A helper class to track token usage and associated costs."""
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.session_details = []

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def add_usage(self, model: str, input_text: str = "", output_text: str = "", usage_type: str = "LLM Call"):
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
        if self.total_tokens == 0:
            return ""
        summary_md = f"\n---\n\n**Session Usage Summary:**\n"
        summary_md += f"- **Total Tokens:** `{self.total_tokens}`\n"
        summary_md += f"- **Estimated Cost:** `${self.total_cost:.6f} USD`\n"
        return summary_md

# --- OpenAI and Redis Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not found. Please set it in the .env file.")

# Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6380"))
redis_password = os.getenv("REDIS_PASSWORD")
redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=0,
        decode_responses=True,
        ssl=redis_ssl,
        ssl_cert_reqs=None if redis_ssl else None
    )
    # Ping to check connection (optional, can be noisy)
    # redis_client.ping()
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# --- LLM and Embeddings Initialization ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
specialist_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Helper Functions ---
def safe_json_loads(json_str: str, default_val: Any = None) -> Any:
    """Safely extracts and loads a JSON object from a string that might contain Markdown or other text."""
    if default_val is None:
        default_val = {}
    
    if not json_str:
        return default_val

    # 1. Try simple strict parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 2. Try to find the first outer-most brace pair
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')

    if start_idx != -1 and end_idx != -1:
        candidate = json_str[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    # 3. Use Regex to find markdown blocks ```json ... ```
    import re
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, json_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON string from LLM response. Length: {len(json_str)}")
    return default_val

# --- Taxonomy Definitions ---
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

STATIC_GEOGRAPHY_MAP = {
    "singapore": "apac", "malaysia": "apac", "indonesia": "apac", "thailand": "apac", "vietnam": "apac",
    "philippines": "apac", "australia": "apac", "new zealand": "apac", "japan": "apac", "south korea": "apac",
    "india": "apac", "hong kong": "apac",
    "united kingdom": "emea", "uk": "emea", "germany": "emea", "france": "emea", "spain": "emea", "italy": "emea",
    "netherlands": "emea", "sweden": "emea", "norway": "emea", "denmark": "emea", "finland": "emea",
    "united arab emirates": "emea", "uae": "emea", "saudi arabia": "emea", "south africa": "emea", "israel": "emea",
    "united states": "americas", "usa": "americas", "us": "americas", "canada": "americas", "mexico": "americas",
    "brazil": "latam", "argentina": "latam", "colombia": "latam"
}

# START WITH STATIC, can be expanded later
SALES_TAXONOMY = STATIC_SALES_TAXONOMY
SEGMENT_SYNONYMS = STATIC_SEGMENT_SYNONYMS
COMPANY_DETAILS_TAXONOMY = STATIC_COMPANY_DETAILS_TAXONOMY
CULTURE_TAXONOMY = STATIC_CULTURE_TAXONOMY
GEOGRAPHY_COUNTRY_TO_REGION_MAP = STATIC_GEOGRAPHY_MAP

# --- Database Loading ---
PROFILES_BY_ID = {}
ALL_COMPANY_NAMES = []
_PROFILES_CACHE = []
CACHE_INITIALIZED = False

def is_cache_initialized() -> bool:
    global CACHE_INITIALIZED
    return CACHE_INITIALIZED


def load_all_company_names_from_db():
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
            return_db_connection(conn)


def _roles_by_candidate_from_roles_raw(roles_raw: List[Any]) -> Dict[int, List[Dict[str, Any]]]:
    roles_by_candidate: Dict[int, List[Dict[str, Any]]] = {}
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
            "industry": role[8] or "",
        }

        start_date = None
        end_date = None
        details_text = role[3] or ""
        if details_text.strip().startswith('{'):
            try:
                parsed = json.loads(details_text)
                start_date = parsed.get("start_date")
                end_date = parsed.get("end_date")
                details_text = parsed.get("details") or ""
            except Exception:
                pass

        roles_by_candidate[candidate_id].append(
            {
                "company": role[1],
                "title": role[2],
                "details": details_text,
                "duration_years": float(role[4]) if role[4] is not None else 0.0,
                "company_details": company_details,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
    return roles_by_candidate


def _profile_dicts_from_candidates_and_roles(
    candidates_raw: List[Any], roles_by_candidate: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for cand in candidates_raw:
        candidate_id = cand[0]
        fn = cand[7] or ""
        ln = cand[8] or ""
        display_name = cand[1] or ""
        if fn or ln:
            display_name = (f"{fn} {ln}").strip() or display_name
        city_col = cand[4] or ""
        profiles.append(
            {
                "id": candidate_id,
                "name": display_name,
                "linkedin": cand[2],
                "location": cand[3],
                "city": city_col,
                "headline": cand[5],
                "about": cand[6],
                "first_name": fn,
                "last_name": ln,
                "total_experience_years": float(cand[9]) if cand[9] is not None else 0.0,
                "max_people_managed": cand[10] or 0,
                "avg_years_in_company": float(cand[11]) if cand[11] is not None else 0.0,
                "candidate_services": cand[12] or "",
                "extracted_industry": cand[13] or "",
                "raw_fields": json.loads(cand[16]) if isinstance(cand[16], str) and cand[16] else (cand[16] if isinstance(cand[16], dict) else {}),
                "embedding": cand[14],
                "created_by": cand[15] or "System",
                "email": cand[17] or "",
                "phone": cand[18] or "",
                "mobile_phone": cand[18] or "",
                "response": cand[19] or "",
                "notes": cand[20] or "",
                "status": cand[21] or "To be started",
                "heyreach_campaign_id": cand[22],
                "li_status": cand[23],
                "email_campaign_id": cand[24],
                "email_outreach_status": cand[25],
                "li_sent_count": cand[26] or 0,
                "message_sent_count": cand[27] or 0,
                "owner_user_id": cand[28],
                "pool_source": cand[29],
                "normalized_linkedin": cand[30],
                "source_master_candidate_id": cand[31],
                "is_archived": bool(cand[32]),
                "roles": roles_by_candidate.get(candidate_id, []),
            }
        )
    return profiles


_CANDIDATE_SELECT_BODY = """
            SELECT DISTINCT ON (c.id)
                c.id, c.name, c.linkedin, c.location, c.city, c.headline, c.about,
                c.first_name, c.last_name,
                c.total_experience_years, c.max_people_managed, c.avg_years_in_company,
                c.raw_fields->>'services', c.raw_fields->>'extracted_industry',
                c.embedding, c.created_by, c.raw_fields, c.email, COALESCE(c.mobile_phone, c.phone),
                c.response, c.notes, c.status,
                co.heyreach_campaign_id, co.li_status, co.campaign_id, co.status as email_outreach_status,
                co.li_sent_count, co.message_sent_count,
                c.owner_user_id, c.pool_source, c.normalized_linkedin, c.source_master_candidate_id,
                c.is_archived
            FROM candidates c
            LEFT JOIN candidate_outreach co ON c.id = co.candidate_id AND co.recruitment_role_id IS NULL
"""

_ROLES_SELECT_BODY = """
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
"""


def load_all_profiles_from_db():
    logger.info("Loading all profiles from the database into cache...")
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            _CANDIDATE_SELECT_BODY
            + """
            ORDER BY c.id, co.updated_at DESC
        """
        )
        candidates_raw = cur.fetchall()

        cur.execute(_ROLES_SELECT_BODY)
        roles_raw = cur.fetchall()
        roles_by_candidate = _roles_by_candidate_from_roles_raw(roles_raw)
        profiles = _profile_dicts_from_candidates_and_roles(candidates_raw, roles_by_candidate)

        logger.info(f"Successfully loaded and cached {len(profiles)} profiles.")
        return profiles
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)


def refresh_profiles_in_cache(candidate_ids: List[int]) -> int:
    """Reload specific candidates (and their roles) from DB into PROFILES_BY_ID. Returns rows merged."""
    global PROFILES_BY_ID
    ids = sorted({int(i) for i in candidate_ids if i is not None})
    if not ids:
        return 0
    conn = get_db_connection()
    if not conn:
        logger.error("refresh_profiles_in_cache: no database connection")
        return 0
    try:
        cur = conn.cursor()
        cur.execute(
            _CANDIDATE_SELECT_BODY
            + """
            WHERE c.id = ANY(%s)
            ORDER BY c.id, co.updated_at DESC
        """,
            (ids,),
        )
        candidates_raw = cur.fetchall()

        cur.execute(
            _ROLES_SELECT_BODY
            + """
            WHERE r.candidate_id = ANY(%s)
        """,
            (ids,),
        )
        roles_raw = cur.fetchall()
        roles_by_candidate = _roles_by_candidate_from_roles_raw(roles_raw)
        profiles = _profile_dicts_from_candidates_and_roles(candidates_raw, roles_by_candidate)
        for p in profiles:
            PROFILES_BY_ID[p["id"]] = p
        logger.info("Refreshed %s profile(s) in cache (subset of %s id(s)).", len(profiles), len(ids))
        return len(profiles)
    except Exception as e:
        logger.error("refresh_profiles_in_cache failed: %s", e)
        raise
    finally:
        if conn:
            return_db_connection(conn)

def initialize_cache():
    global PROFILES_BY_ID, ALL_COMPANY_NAMES, _PROFILES_CACHE, CACHE_INITIALIZED
    try:
        profiles = load_all_profiles_from_db()
        PROFILES_BY_ID.clear()
        ALL_COMPANY_NAMES.clear()
        PROFILES_BY_ID.update({p['id']: p for p in profiles})
        _PROFILES_CACHE = profiles

        companies = load_all_company_names_from_db()
        ALL_COMPANY_NAMES.extend(companies)

        CACHE_INITIALIZED = True
        logger.info(f"Cache initialized with {len(PROFILES_BY_ID)} profiles and {len(ALL_COMPANY_NAMES)} companies.")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")

def update_profile_cache(candidate_id: int, data: Dict[str, Any]):
    """Update a specific profile in the global cache"""
    global PROFILES_BY_ID
    if candidate_id in PROFILES_BY_ID:
        PROFILES_BY_ID[candidate_id].update(data)
        logger.info(f"Updated cache for candidate {candidate_id}: {data}")
    else:
        logger.warning(f"Attempted to update cache for non-existent candidate {candidate_id}")


def build_candidate_pool(candidate_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Build a lightweight search pool from the in-memory cache.
    We drop embeddings here because the screening API serializes these profiles.
    """
    if candidate_ids is None:
        source_profiles = PROFILES_BY_ID.values()
    else:
        source_profiles = [PROFILES_BY_ID[pid] for pid in candidate_ids if pid in PROFILES_BY_ID]

    pool = []
    for profile in source_profiles:
        profile_copy = dict(profile)
        profile_copy.pop("embedding", None)
        pool.append(profile_copy)
    return pool

# --- Logic Functions ---

def normalize_query_with_llm(query: str) -> str:
    logger.info(f"Normalizing query... Search Query: {query}")
    return query.lower().replace("sme", "smb").replace("mid market", "mid-market")


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
    if isinstance(llm_json_response, list):
        return [str(item) for item in llm_json_response if isinstance(item, str)]
    if isinstance(llm_json_response, dict):
        for value in llm_json_response.values():
            if isinstance(value, list):
                return [str(item) for item in value if isinstance(item, str)]
    return []

# --- CHECK FUNCTIONS ---

def check_company_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    required_companies = criteria.get("required_companies")
    if not required_companies or not isinstance(required_companies, list):
        return True

    required_companies_lower = [c.lower() for c in required_companies]
    found_companies = set()

    for company_name in required_companies_lower:
        req_tokens = set(company_name.split())
        for role in profile.get('roles', []):
            role_company = (role.get('company') or '').lower()
            if not role_company: continue
            
            if company_name in role_company:
                found_companies.add(company_name)
                break
            
            role_tokens = set(role_company.split())
            if req_tokens.issubset(role_tokens):
                 found_companies.add(company_name)
                 break

    is_met = found_companies == set(required_companies_lower)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": "company_presence (AND)",
            "source_text": f"Profile confirms experience in all required companies: {', '.join(required_companies)}."
        })
    return is_met

def check_industry_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_industries")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

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

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"industry_presence ({op})",
            "source_text": f"Found industries: {', '.join(found_values)}."
        })
    return is_met

def check_functional_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_functions")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            if v in role_text:
                found_values.add(v)
                break

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"functional_presence ({op})",
            "source_text": f"Found functions: {', '.join(found_values)}."
        })
    return is_met

def check_customer_segments(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_segments")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

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
             profile.setdefault('evidence_log', []).append({
                "criterion": f"segment_presence ({op})",
                "source_text": f"Found segment match for {original_value}."
            })

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    return is_met

def check_location_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_locations")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

    profile_location = (profile.get('location') or '').lower()
    found_values = set()
    for v in values:
        if v in profile_location:
            found_values.add(v)

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"location_presence ({op})",
            "source_text": f"Location match: {', '.join(found_values)}."
        })
    return is_met

def check_geography_experience(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_geographies")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

    found_values = set()
    for v in values:
        region_for_v = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v)
        profile_location_lower = (profile.get('location') or '').lower()
        
        if v in profile_location_lower:
            found_values.add(v)
            continue
        if region_for_v and region_for_v in profile_location_lower:
            found_values.add(v)
            continue
            
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            company_details = role.get('company_details', {})
            company_office_locations = company_details.get('customer_presence', [])
            company_locations_text = ' '.join([loc.lower() for loc in company_office_locations])
            company_hq_text = (company_details.get('headquarters') or '').lower()
            combined = f"{role_text} {company_locations_text} {company_hq_text}"
            
            if v in combined or (region_for_v and region_for_v in combined):
                found_values.add(v)
                break

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"geography_presence ({op})",
            "source_text": f"Geography match: {', '.join(found_values)}."
        })
    return is_met

def check_company_details(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_company_details")
    if not criteria_obj: return True
    
    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

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

    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"company_details ({op})",
            "source_text": f"Matched company details: {', '.join(found_values)}."
        })
    return is_met

def check_company_culture_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("required_culture_type")
    if not criteria_obj: return True

    op = criteria_obj.get("operator", "OR").upper() if isinstance(criteria_obj, dict) else "OR"
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

    found_values = set()
    for v in values:
        for role in profile.get('roles', []):
            culture_type = (role.get('company_details', {}).get('culture_type') or '').lower()
            if v in culture_type:
                found_values.add(v)
                break
    
    is_met = (found_values == set(values)) if op == "AND" else bool(found_values)
    if is_met:
        profile.setdefault('evidence_log', []).append({
            "criterion": f"culture_type ({op})",
            "source_text": f"Matched culture: {', '.join(found_values)}."
        })
    return is_met

def check_excluded_geography_presence(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    criteria_obj = criteria.get("excluded_geographies")
    if not criteria_obj: return True
    values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not values: return True

    profile_location = (profile.get('location') or '').lower()
    if any(v in profile_location for v in values):
        return False

    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(v in role_text for v in values):
            return False
    return True

def check_tenure_in_latest_role(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    min_tenure = criteria.get("min_tenure_in_latest_role")
    if not min_tenure: return True

    roles = profile.get('roles', [])
    if not roles: return False
    
    latest_role = roles[0]
    duration = latest_role.get('duration_years', 0.0)
    is_met = duration >= min_tenure
    if is_met:
         profile.setdefault('evidence_log', []).append({
            "criterion": "min_tenure_in_latest_role",
            "source_text": f"Latest role tenure {duration} >= {min_tenure}."
        })
    return is_met

def check_avg_tenure_in_last_n_roles(profile: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
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
        profile.setdefault('evidence_log', []).append({
            "criterion": "avg_tenure_in_last_n_roles",
            "source_text": f"Avg tenure {calculated_avg:.1f} >= {avg_years}."
        })
    return is_met


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        clean_str = date_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(clean_str)
        return dt.replace(tzinfo=None)
    except Exception:
        try:
            date_part = date_str.split('T')[0].split()[0]
            dt = datetime.strptime(date_part, "%Y-%m-%d")
            return dt.replace(tzinfo=None)
        except Exception:
            return None

def merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged

def calculate_merged_duration_years(matching_roles: List[Dict[str, Any]]) -> float:
    intervals = []
    undated_duration_years = 0.0
    today = datetime.now()
    
    for role in matching_roles:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        
        start_dt = parse_date(start_str) if start_str else None
        end_dt = parse_date(end_str) if end_str else None
        
        if start_dt:
            if not end_dt:
                end_dt = today
            intervals.append((start_dt, end_dt))
        else:
            undated_duration_years += role.get("duration_years", 0.0)
            
    merged = merge_intervals(intervals)
    
    dated_months = 0
    for start, end in merged:
        if end >= start:
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if end.day >= start.day:
                months += 1
            dated_months += max(months, 0)
            
    dated_years = round(dated_months / 12.0, 2)
    return round(dated_years + undated_duration_years, 2)

# --- DURATION CALCULATIONS ---
def calculate_functional_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    if not criteria_obj or not isinstance(criteria_obj, dict): return 0.0, []
    req_values = [v.lower() for v in criteria_obj.get("values", [])]
    if not req_values: return 0.0, []

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(v in role_text for v in req_values):
            matching_roles.append(role)
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': role.get('duration_years', 0.0) or 0.0
            })
            
    total_duration = calculate_merged_duration_years(matching_roles)
    return total_duration, contributing_roles

def calculate_industry_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    req_values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        role_text = f"{(role.get('company') or '').lower()} {(company_details.get('industry', '') or '').lower()} {(company_details.get('product_service', '') or '').lower()}"
        if any(v in role_text for v in req_values):
            matching_roles.append(role)
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': role.get('duration_years', 0.0) or 0.0
            })
            
    total_duration = calculate_merged_duration_years(matching_roles)
    return total_duration, contributing_roles

def calculate_segment_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    req_values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []

    all_search_terms = {}
    for v in req_values:
        all_search_terms[v] = SEGMENT_SYNONYMS.get(v, [v])

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        company_segments = role.get("company_details", {}).get("customer_segment", [])
        company_segments_lower = ' '.join([cs.lower() for cs in company_segments])
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()} {company_segments_lower}"

        matched = False
        for original_value, synonyms in all_search_terms.items():
            if any(s in role_text for s in synonyms):
                matched = True
                break
        if matched:
            matching_roles.append(role)
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': role.get('duration_years', 0.0) or 0.0
            })
            
    total_duration = calculate_merged_duration_years(matching_roles)
    return total_duration, contributing_roles

def calculate_geography_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    req_values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []

    regions_for_req_values = {v: GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v) for v in req_values}
    regions_to_check = {region for region in regions_for_req_values.values() if region}

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        company_details = role.get('company_details', {})
        combined = f"{role_text} {' '.join([x.lower() for x in company_details.get('customer_presence', [])])} {(company_details.get('headquarters') or '').lower()}"

        if any(v in combined for v in req_values) or any(r in combined for r in regions_to_check):
            if not any(cr['company'] == role.get('company', '') for cr in contributing_roles):
                matching_roles.append(role)
                contributing_roles.append({
                    'company': role.get('company', ''),
                    'title': role.get('title', ''),
                    'duration_years': role.get('duration_years', 0.0) or 0.0
                })
                
    total_duration = calculate_merged_duration_years(matching_roles)
    return total_duration, contributing_roles

def calculate_company_details_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    req_values = [v.lower() for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        details_text = f"{(company_details.get('funding_stage') or '').lower()} {(company_details.get('business_model') or '').lower()} {(company_details.get('product_service') or '').lower()}"
        if any(v in details_text for v in req_values):
            matching_roles.append(role)
            contributing_roles.append({
                'company': role.get('company', ''),
                'title': role.get('title', ''),
                'duration_years': role.get('duration_years', 0.0) or 0.0
            })
            
    total_duration = calculate_merged_duration_years(matching_roles)
    return total_duration, contributing_roles


async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("Applying detailed filters...")
    matching_candidates = []

    sort_criterion = "required_functions"
    if criteria.get("required_segments"): sort_criterion = "required_segments"
    elif criteria.get("required_industries"): sort_criterion = "required_industries"
    elif criteria.get("required_geographies"): sort_criterion = "required_geographies"
    
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

        checks = [
            check_company_presence,
            check_industry_presence,
            check_functional_presence,
            check_customer_segments,
            check_location_presence,
            check_geography_experience,
            check_company_details,
            check_company_culture_presence,
            check_excluded_geography_presence,
            check_tenure_in_latest_role,
            check_avg_tenure_in_last_n_roles
        ]
        
        for check in checks:
            if all_criteria_met and not check(profile, criteria):
                all_criteria_met = False
                break
        
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
                     min_y = crit_obj.get("min_years", 0.0)
                     duration, roles = calc_func(profile, crit_obj)
                     profile['calculated_experience'][key] = {
                        "duration": duration,
                        "roles": roles,
                        "label": ", ".join(get_values_from_criteria(crit_obj)),
                        "required": min_y
                    }
                     if duration < min_y:
                        all_criteria_met = False
                        break
        
        if all_criteria_met:
            if profile['calculated_experience']:
                first_key = next(iter(profile['calculated_experience']))
                profile['contributing_roles_details'] = {'roles': profile['calculated_experience'][first_key]['roles']}
            else:
                # Fallback role details
                profile['contributing_roles_details'] = {'roles': profile.get('roles', [])[:3]}
            
            matching_candidates.append(profile)

    if matching_candidates and sort_criterion:
        matching_candidates.sort(
            key=lambda x: x['calculated_experience'].get(sort_criterion, {}).get('duration', 0.0),
            reverse=True
        )

    top_n = criteria.get("top_n")
    if top_n and top_n > 0:
        matching_candidates = matching_candidates[:top_n]
    
    return matching_candidates

async def generate_reasoning_for_profile(profile: Dict[str, Any], original_criteria: Dict[str, Any], tracker: TokenCostTracker) -> str:
    prompt_template = PromptTemplate(
        input_variables=["original_criteria_json", "matching_profile_json"],
        template="""
        You are an expert recruitment analyst. Synthesize a concise, single-paragraph summary explaining why this candidate is a good match based on the original search criteria.

        **Original Criteria:** {original_criteria_json}
        **Candidate:** {matching_profile_json}

        **Instructions:**
        - Use EVIDENCE provided in the candidate object.
        - Single flowing paragraph. No bullets.
        """
    )
    # Create a safe copy for JSON serialization (remove numpy arrays like 'embedding')
    profile_safe = {k: v for k, v in profile.items() if k != "embedding"}
    formatted_prompt = prompt_template.format(
        original_criteria_json=json.dumps(original_criteria, indent=2),
        matching_profile_json=json.dumps(profile_safe, indent=2)
    )
    response = await specialist_llm.ainvoke(formatted_prompt)
    content = response.content.replace('\n', ' ').replace('|', '')
    tracker.add_usage(specialist_llm.model_name, formatted_prompt, response.content, "Reasoning")
    return content

async def process_query_main(
    query: str,
    session_id: str,
    tracker: TokenCostTracker,
    *,
    screening_user_id: Optional[int] = None,
    screening_role: Optional[str] = None,
) -> AsyncIterator[Any]:
    
    # Ensure cache is initialized
    if not is_cache_initialized():
        logger.info("Cache empty, initializing on demand...")
        initialize_cache()
    screening_r = (screening_role or "").strip().lower()
    normalized_query = normalize_query_with_llm(query)
    normalized_query_lower = normalized_query.lower()

    # 2. Extract Criteria
    yield "Analyzing query requirements..."
    
    criteria_extraction_prompt = PromptTemplate(
        input_variables=["query", "sales_taxonomy_json", "segment_taxonomy_json"],
        template="""
        Extract structured filtering criteria from the user's query: "{query}".
        
        Taxonomies:
        Sales: {sales_taxonomy_json}
        Segments: {segment_taxonomy_json}

        You MUST return a JSON object with the following keys ONLY (omit if not applicable):
        - "required_companies": List[str] (exact company names)
        - "required_industries": {{"operator": "OR", "values": List[str]}}
        - "required_functions": {{"operator": "OR", "values": List[str]}} (Map to Sales Taxonomy if possible)
        - "required_segments": {{"operator": "OR", "values": List[str]}} (Map to Segment Synonyms)
        - "required_locations": {{"operator": "OR", "values": List[str]}} (City/State)
        - "required_geographies": {{"operator": "OR", "values": List[str]}} (Countries/Regions like 'APAC', 'EMEA')
        - "required_company_details": {{"operator": "OR", "values": List[str]}} (e.g. 'SaaS', 'B2B', 'Series A')
        - "min_total_experience": int
        - "min_people_managed": int
        - "top_n": int (default 10 if searching for "top", "best")
        
        Example:
        Query: "Account executives in SaaS companies in Singapore with 5 years exp"
        JSON: {{
            "required_functions": {{"operator": "OR", "values": ["Account Executive"]}},
            "required_company_details": {{"operator": "OR", "values": ["SaaS"]}},
            "required_locations": {{"operator": "OR", "values": ["Singapore"]}},
            "min_total_experience": 5
        }}

        Query: "{query}"
        Answer (JSON):
        """
    )
    
    prompt_text = criteria_extraction_prompt.format(
        query=normalized_query,
        sales_taxonomy_json=json.dumps(SALES_TAXONOMY),
        segment_taxonomy_json=json.dumps(SEGMENT_SYNONYMS)
    )
    
    try:
        criteria_response = await llm.ainvoke(prompt_text)
        criteria = safe_json_loads(criteria_response.content, {})
        tracker.add_usage(llm.model_name, prompt_text, criteria_response.content, "Criteria Extraction")
        
        if not criteria:
            # Fallback
            criteria = {"required_industries": {"operator": "OR", "values": [normalized_query]}}
            
        # Handle "all" or explicit top_n removal
        if "all" in normalized_query_lower:
            criteria["top_n"] = 0
        elif not any(w in normalized_query_lower for w in ["top", "one", "maximum"]):
            if "top_n" in criteria:
                del criteria["top_n"]
                
    except Exception as e:
        logger.error(f"Error extracting criteria: {e}")
        yield f"Error analyzing query: {e}"
        return

    original_criteria = copy.deepcopy(criteria)

    def _pool_scope_ids():
        if screening_r == "recruiter" and screening_user_id is not None:
            return [
                pid
                for pid, p in PROFILES_BY_ID.items()
                if not p.get("is_archived") and p.get("owner_user_id") == screening_user_id
            ]
        return None

    def _scoped_build(ids_override: Optional[List[int]] = None):
        scope = _pool_scope_ids()
        if scope is not None:
            if ids_override is not None:
                allowed = set(scope)
                merged = [i for i in ids_override if i in allowed]
                return build_candidate_pool(merged if merged else scope)
            return build_candidate_pool(scope)
        return build_candidate_pool(ids_override)
    
    # 3. SEMANTIC SEARCH (Vector Retrieval)
    yield "Searching database..."
    
    search_query_text = " ".join(
        (criteria.get("required_companies") or []) + 
        get_values_from_criteria(criteria.get("required_industries")) +
        get_values_from_criteria(criteria.get("required_functions")) +
        get_values_from_criteria(criteria.get("required_segments")) +
        get_values_from_criteria(criteria.get("required_geographies")) +
        get_values_from_criteria(criteria.get("required_locations")) +
        get_values_from_criteria(criteria.get("required_company_details")) +
        get_values_from_criteria(criteria.get("required_culture_type"))
    ).strip()
    
    initial_candidate_pool = []
    used_vector_shortlist = False
    
    if search_query_text:
        try:
            query_embedding = embeddings.embed_query(search_query_text)
            tracker.add_usage(embeddings.model, search_query_text, usage_type="Embedding")
            
            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        if screening_r == "recruiter" and screening_user_id is not None:
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND owner_user_id = %s
                                  AND embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT 500
                                """,
                                (screening_user_id, query_embedding),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT 500
                                """,
                                (query_embedding,),
                            )
                        ids = [row[0] for row in cur.fetchall()]
                        if (
                            not ids
                            and screening_r == "recruiter"
                            and screening_user_id is not None
                        ):
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND owner_user_id = %s
                                LIMIT 2000
                                """,
                                (screening_user_id,),
                            )
                            ids = [row[0] for row in cur.fetchall()]
                finally:
                    return_db_connection(conn)
                initial_candidate_pool = _scoped_build(ids)
                used_vector_shortlist = True
                logger.info(f"Vector search returned {len(initial_candidate_pool)} candidates.")
            else:
                initial_candidate_pool = _scoped_build()
        except Exception as e:
            logger.error(f"Vector search failed: {e}. Falling back to full scan.")
            initial_candidate_pool = _scoped_build()
    else:
        initial_candidate_pool = _scoped_build()
    
    # 4. Filter Candidates
    final_candidates = await filter_candidates_by_criteria(initial_candidate_pool, criteria)

    if not final_candidates and used_vector_shortlist and len(initial_candidate_pool) < len(
        [p for p in PROFILES_BY_ID.values() if not p.get("is_archived")]
    ):
        logger.info("Vector shortlist returned no matches after filtering. Retrying scoped full cache.")
        final_candidates = await filter_candidates_by_criteria(_scoped_build(), criteria)
    
    if not final_candidates:
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    yield {"type": "progress_start", "total": len(final_candidates)}
    
    # 5. Parallel Reasoning Generation
    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def get_reasoning_safe(profile):
        async with semaphore:
            try:
                reasoning = await generate_reasoning_for_profile(profile, original_criteria, tracker)
                profile['reasoning'] = reasoning
                return profile
            except Exception as e:
                logger.error(f"Reasoning Gen Failed for {profile['id']}: {e}")
                return profile # Return without reasoning if failed

    tasks = [get_reasoning_safe(p) for p in final_candidates]
    
    processed_count = 0
    processed_candidates = []
    
    for future in asyncio.as_completed(tasks):
        result = await future
        processed_candidates.append(result)
        processed_count += 1
        yield {
            "type": "profile_chunk",
            "data": result,
            "current": processed_count,
            "total": len(final_candidates)
        }

    # Sort results
    # (Optional: restore sort order from filtering step if needed, currently they come back in completion order)
    # Ideally should sort based on filter score, but for now just yielding as they finish is fine for UX.
    
    yield {"type": "complete", "data": processed_candidates, "summary": tracker.get_summary()}


def profiles_to_excel(profiles_dict: Dict[str, Any]) -> bytes:
    if not profiles_dict: return b""
    flat_data = []
    for p in profiles_dict.values():
        roles_summary = [f"{r.get('title')} at {r.get('company')}" for r in p.get('roles', [])]
        flat_data.append({
            "Name": p.get("name"),
            "LinkedIn": p.get("linkedin"),
            "Location": p.get("location"),
            "Reasoning": p.get("reasoning", "N/A"),
            "Roles": " | ".join(roles_summary)
        })
    df = pd.DataFrame(flat_data)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return out.getvalue()

async def get_analytics_summary(user_email: str = None, role: str = "recruiter", user_id: int = None) -> Dict[str, Any]:
    """
    Generate pipeline and recruiter performance statistics.
    Admins see org-wide (non-archived) candidates; recruiters see only their pool
    (owner_user_id = their user id — uploads and admin-assigned copies).
    """
    role_l = (role or "").strip().lower()
    is_admin = role_l == "admin"

    all_profiles = [p for p in PROFILES_BY_ID.values() if not p.get("is_archived")]

    if user_id is None and user_email:
        conn = get_db_connection(validate=False, register_pgvector=False)
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, name FROM users WHERE email = %s", (user_email,))
                    row = cur.fetchone()
                    if row:
                        user_id = row[0]
            except Exception as e:
                logger.error(f"Error fetching user id: {e}")
            finally:
                return_db_connection(conn)

    if is_admin:
        # Deduplicate to count unique candidates by LinkedIn
        unique_candidates = {}
        for p in all_profiles:
            li = p.get("normalized_linkedin") or p.get("linkedin")
            if not li:
                # If no linkedin, use ID as fallback for uniqueness
                li = f"id_{p.get('id')}"
            
            existing = unique_candidates.get(li)
            if not existing:
                unique_candidates[li] = p
            else:
                # Prefer the row with a more "interesting" status
                # (Non-'To be started' is better for the global dashboard)
                st = (p.get("status") or "").strip().lower()
                est = (existing.get("status") or "").strip().lower()
                
                # Heuristic: If existing is "To be started" or empty, and new one has a real status, swap it.
                if st and st != "to be started" and (not est or est == "to be started"):
                    unique_candidates[li] = p
                # Also prefer rows that are NOT master rows if we already have a status match
                # (since recruiter rows are more likely to have updated notes/activity)
                elif p.get("owner_user_id") is not None and existing.get("owner_user_id") is None:
                    if not est or est == "to be started": # Only if we aren't losing a better status
                        unique_candidates[li] = p

        stats_profiles = list(unique_candidates.values())
    else:
        stats_profiles = [
            p
            for p in all_profiles
            if user_id is not None and p.get("owner_user_id") == user_id
        ]

    team_pipeline_stats = {}
    for p in stats_profiles:
        status = p.get("status") or "To be started"
        team_pipeline_stats[status] = team_pipeline_stats.get(status, 0) + 1

    personal_pipeline_stats = dict(team_pipeline_stats) if not is_admin else {}
    personal_profiles = list(stats_profiles) if not is_admin else []

    # 2. Recruiter Performance (Admin only)
    recruiter_performance = []
    if is_admin:
        conn = get_db_connection(validate=False, register_pgvector=False)
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            COALESCE(NULLIF(u.name, ''), 'Unknown') AS recruiter,
                            COUNT(DISTINCT rc.candidate_id) AS sourced,
                            COUNT(DISTINCT rc.candidate_id) FILTER (
                                WHERE c.status = 'Shortlisted'
                            ) AS shortlisted,
                            COUNT(DISTINCT rc.candidate_id) FILTER (
                                WHERE c.status IN ('Followup / In conversation', 'In Conversation')
                            ) AS in_conversation
                        FROM recruitment_role_candidates rc
                        JOIN recruitment_roles r ON rc.role_id = r.id
                        JOIN users u ON r.user_id = u.id
                        JOIN candidates c ON rc.candidate_id = c.id
                        GROUP BY COALESCE(NULLIF(u.name, ''), 'Unknown')
                        ORDER BY sourced DESC
                    """)
                    for name, sourced, shortlisted, in_conversation in cur.fetchall():
                        recruiter_performance.append({
                            "recruiter": name,
                            "sourced": int(sourced or 0),
                            "shortlisted": int(shortlisted or 0),
                            "in_conversation": int(in_conversation or 0),
                        })
            except Exception as e:
                logger.error(f"Error fetching admin recruiter perf: {e}")
            finally:
                return_db_connection(conn)

    # 3. High-level aggregates (scoped: admin = all active; recruiter = own pool)
    total_sourced = len(stats_profiles)
    
    total_shortlisted = team_pipeline_stats.get("Shortlisted", 0)
    total_in_conversation = team_pipeline_stats.get("Followup / In conversation", 0)

    # 4. Geographic distribution (by Country)
    geo_map = {}
    
    # Simple mapping of common locations/keywords to canonical Country names
    COUNTRY_MAPPING = {
        "india": "India", "bengaluru": "India", "bangalore": "India", "mumbai": "India", "delhi": "India", "hyderabad": "India", "pune": "India", "chennai": "India",
        "united states": "United States", "usa": "United States", "us": "United States", "new york": "United States", "california": "United States", "san francisco": "United States", "texas": "United States", "seattle": "United States", "chicago": "United States", "boston": "United States",
        "united kingdom": "United Kingdom", "uk": "United Kingdom", "london": "United Kingdom", "england": "United Kingdom",
        "canada": "Canada", "toronto": "Canada", "vancouver": "Canada", "montreal": "Canada",
        "australia": "Australia", "sydney": "Australia", "melbourne": "Australia",
        "germany": "Germany", "berlin": "Germany", "munich": "Germany",
        "france": "France", "paris": "France",
        "singapore": "Singapore",
        "uae": "UAE", "dubai": "UAE", "united arab emirates": "UAE",
        "brazil": "Brazil", "são paulo": "Brazil",
        "netherlands": "Netherlands", "amsterdam": "Netherlands",
        "japan": "Japan", "tokyo": "Japan",
        "sweden": "Sweden", "stockholm": "Sweden",
        "spain": "Spain", "madrid": "Spain", "barcelona": "Spain",
        "italy": "Italy", "rome": "Italy", "milan": "Italy"
    }
    
    for p in stats_profiles:
        location = (p.get("location") or "").strip()
        if not location:
            country = "Unknown"
        else:
            loc_lower = location.lower()
            country = "Other"
            
            # Extract standard country based on keyword match in the location string
            for key, canonical_name in COUNTRY_MAPPING.items():
                # We do word boundary matching ideally, but `in loc_lower` works for a fast approximation
                # Specifically checking for full words or substrings separated by commas/spaces
                if key in loc_lower:
                    # To avoid matching "us" inside "australia", we check for word boundaries for short keys
                    if len(key) <= 3:
                        import re
                        if re.search(r'\b' + re.escape(key) + r'\b', loc_lower):
                            country = canonical_name
                            break
                    else:
                        country = canonical_name
                        break
                        
            # If not found in mapping, maybe try extracting the last part of a comma separated location
            if country == "Other" and "," in location:
                parts = [part.strip() for part in location.split(",")]
                if parts:
                    last_part = parts[-1]
                    # if it looks like a country (not a state code), use it
                    if len(last_part) > 2 and not any(char.isdigit() for char in last_part):
                        country = last_part.title()
                        
        geo_map[country] = geo_map.get(country, 0) + 1
    
    def top_distribution_with_other(counts: Dict[str, int], limit: int = 8) -> List[Dict[str, Any]]:
        """Return top buckets while preserving the full total in a rolled-up Other bucket."""
        buckets = [(k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0]
        if len(buckets) <= limit:
            return [{"name": k, "value": v} for k, v in buckets]

        top = buckets[: max(0, limit - 1)]
        omitted_total = sum(v for _, v in buckets[max(0, limit - 1):])
        distribution = [{"name": k, "value": v} for k, v in top]

        for item in distribution:
            if item["name"] == "Other":
                item["value"] += omitted_total
                break
        else:
            distribution.append({"name": "Other", "value": omitted_total})
        return distribution

    geo_distribution = top_distribution_with_other(geo_map)

    # 5. Industry distribution (from extracted_industry or role industries)
    industry_map = {}
    for p in stats_profiles:
        industry = (p.get("extracted_industry") or "").strip()
        if not industry:
            # Fallback: try first role's product_service
            roles = p.get("roles", [])
            if roles:
                industry = (roles[0].get("company_details", {}).get("product_service") or "").strip()
        if not industry:
            industry = "Other"
        # Normalize long names
        if len(industry) > 25:
            industry = industry[:23] + "…"
        industry_map[industry] = industry_map.get(industry, 0) + 1
    
    industry_distribution = top_distribution_with_other(industry_map)

    # 6. Segment distribution (SMB / Enterprise / Mid-Market etc.)
    segment_map = {}
    for p in stats_profiles:
        found_seg = "Unknown"
        for rrow in p.get("roles", []):
            segs = rrow.get("company_details", {}).get("customer_segment", [])
            if isinstance(segs, list) and segs:
                s_lower = str(segs[0]).lower().strip()
                normalized = s_lower
                for canonical, synonyms in SEGMENT_SYNONYMS.items():
                    if s_lower in synonyms or s_lower == canonical:
                        normalized = canonical.upper()
                        break
                found_seg = normalized.capitalize() if normalized == s_lower else normalized
                break
            elif isinstance(segs, str) and segs.strip():
                s_lower = segs.lower().strip()
                normalized = s_lower
                for canonical, synonyms in SEGMENT_SYNONYMS.items():
                    if s_lower in synonyms or s_lower == canonical:
                        normalized = canonical.upper()
                        break
                found_seg = normalized.capitalize() if normalized == s_lower else normalized
                break
                
        segment_map[found_seg] = segment_map.get(found_seg, 0) + 1

    segment_distribution = top_distribution_with_other(segment_map)
    # 7. Functional distribution (using existing data fields)
    functional_map = {}
    for p in stats_profiles:
        found_func = None

        raw = p.get("raw_fields", {})
        if isinstance(raw, str):
            import json
            try: raw = json.loads(raw)
            except Exception: raw = {}
            
        funcs = raw.get("functions") or raw.get("function") or raw.get("department") or raw.get("job_family")
        if isinstance(funcs, list) and funcs:
            found_func = str(funcs[0])
        elif isinstance(funcs, str) and funcs.strip():
            found_func = funcs.strip()
            
        if not found_func:
            found_func = "Other"
            
        # Normalize simple casing
        found_func = found_func.title()
        if len(found_func) > 25:
            found_func = found_func[:23] + "…"

        functional_map[found_func] = functional_map.get(found_func, 0) + 1

    functional_distribution = top_distribution_with_other(functional_map)

    return {
        "summary": {
            "total_sourced": total_sourced,
            "shortlisted": total_shortlisted,
            "in_conversation": total_in_conversation,
            "pipeline_health": team_pipeline_stats
        },
        "distributions": {
            "geo": geo_distribution,
            "industry": industry_distribution,
            "segment": segment_distribution,
            "functional": functional_distribution
        },
        "personal": {
            "total_sourced": len(personal_profiles),
            "shortlisted": personal_pipeline_stats.get("Shortlisted", 0),
            "pipeline_health": personal_pipeline_stats
        } if not is_admin else None,
        "recruiter_performance": recruiter_performance if is_admin else []
    }

async def get_recruiter_list() -> List[str]:
    """Get a unique list of all recruiters who have sourced candidates."""
    recruiters = set()
    for p in PROFILES_BY_ID.values():
        if p.get("created_by"):
            recruiters.add(p.get("created_by"))
    return sorted(list(recruiters))

async def get_semantic_scores(query_text: str) -> Dict[int, float]:
    """
    Calculates semantic similarity scores for all profiles based on a query.
    Uses OpenAI embeddings and in-memory cosine similarity.
    """
    if not query_text:
        return {}
    
    try:
        # 1. Embed the query
        query_vector = await embeddings.aembed_query(query_text)
        
        import numpy as np
        query_vec = np.array(query_vector)
        
        scores = {}
        for pid, profile in PROFILES_BY_ID.items():
            emb = profile.get("embedding")
            if not emb:
                scores[pid] = 0.0
                continue
            
            # 2. Similarity (Cosine)
            cand_vec = np.array(emb)
            similarity = np.dot(query_vec, cand_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(cand_vec))
            scores[pid] = float(similarity)
            
        return scores
    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}")
        return {}

def update_candidate_status(candidate_id: int, status: str) -> bool:
    """Updates the status of a candidate in the database and the in-memory cache."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE candidates SET status = %s WHERE id = %s", (status, candidate_id))
            conn.commit()
            
            # Update cache if present
            if candidate_id in PROFILES_BY_ID:
                PROFILES_BY_ID[candidate_id]["status"] = status
                logger.info(f"Updated status for candidate {candidate_id} to '{status}' in cache.")
                
            return True
    except Exception as e:
        logger.error(f"Failed to update candidate status: {e}")
        return False
    finally:
        return_db_connection(conn)

def update_candidate_notes(candidate_id: int, notes: str) -> bool:
    """Updates candidate notes in the database and cache"""
    conn = get_db_connection()
    if not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE candidates SET notes = %s WHERE id = %s", (notes, candidate_id))
            conn.commit()
            if candidate_id in PROFILES_BY_ID:
                PROFILES_BY_ID[candidate_id]["notes"] = notes
                logger.info(f"Updated notes for candidate {candidate_id} in cache.")
            return True
    except Exception as e:
        logger.error(f"Failed to update candidate notes: {e}")
        return False
    finally:
        return_db_connection(conn)

def update_candidate_contact(
    linkedin_url: str,
    email: Optional[str],
    phone: Optional[str],
    normalized_linkedin: Optional[str] = None,
) -> None:
    """Refresh email/phone in cache for all profiles matching normalized or raw LinkedIn."""
    from backend.services.linkedin_normalize import normalize_linkedin as _norm_li

    target_norm = normalized_linkedin or _norm_li(linkedin_url)
    for candidate_id, profile in PROFILES_BY_ID.items():
        pn = profile.get("normalized_linkedin") or _norm_li(profile.get("linkedin"))
        raw_match = linkedin_url and profile.get("linkedin") == linkedin_url
        norm_match = target_norm and pn and pn == target_norm
        if not (raw_match or norm_match):
            continue
        if email:
            profile["email"] = email
        if phone:
            profile["phone"] = phone
            profile["mobile_phone"] = phone
        profile["enrichment_finished"] = True
        logger.info(
            "Updated contact info for candidate %s in cache (enrichment fan-out).",
            candidate_id,
        )

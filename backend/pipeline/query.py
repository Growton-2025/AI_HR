import asyncio
import os
import json
import logging
import hashlib
import redis
import tiktoken
import copy
import re
import pandas as pd
import io
import psycopg2
from typing import List, Dict, Any, AsyncIterator, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from backend.db.connection import get_db_connection, return_db_connection
from backend.services.ai_columns import (
    build_candidate_context,
    build_candidate_context_pack,
    build_query_plan,
    call_openai_json,
    career_facts_to_text,
    classify_ai_column_prompt,
    compute_career_facts,
    run_candidate_query_tools,
    verify_smart_column_outputs,
)

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Pricing and Token Configuration ---
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
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
SCREENING_CRITERIA_MODEL = os.getenv("SCREENING_CRITERIA_MODEL", "gpt-4o")
SCREENING_REASONING_MODEL = os.getenv("SCREENING_REASONING_MODEL", "gpt-4o")
SCREENING_GENERATION_MODEL = os.getenv("SCREENING_GENERATION_MODEL", SCREENING_REASONING_MODEL)
SCREENING_EMBEDDING_MODEL = os.getenv("SCREENING_EMBEDDING_MODEL", "text-embedding-3-small")
SCREENING_MAX_RESULTS = int(os.getenv("SCREENING_MAX_RESULTS", "25"))
SCREENING_VECTOR_LIMIT = int(os.getenv("SCREENING_VECTOR_LIMIT", "750"))
SCREENING_FULL_SCAN_LIMIT = int(os.getenv("SCREENING_FULL_SCAN_LIMIT", "5000"))
SCREENING_WEB_SEARCH_DEFAULT = os.getenv("SCREENING_WEB_SEARCH_DEFAULT", "true").strip().lower() not in {"0", "false", "no"}
SCREENING_WEB_VERIFY_TOP_K = int(os.getenv("SCREENING_WEB_VERIFY_TOP_K", "30"))
SCREENING_WEB_CONCURRENCY = int(os.getenv("SCREENING_WEB_CONCURRENCY", "3"))
SCREENING_COMPANY_FACT_ENRICH_LIMIT = int(os.getenv("SCREENING_COMPANY_FACT_ENRICH_LIMIT", "80"))
SCREENING_DYNAMIC_VERIFY_LIMIT = int(os.getenv("SCREENING_DYNAMIC_VERIFY_LIMIT", "40"))
SCREENING_WEB_SEARCH_TOOL = os.getenv("SCREENING_WEB_SEARCH_TOOL", os.getenv("AI_COLUMN_WEB_SEARCH_TOOL", "web_search"))
SCREENING_WEB_SEARCH_CONTEXT_SIZE = os.getenv("SCREENING_WEB_SEARCH_CONTEXT_SIZE", os.getenv("AI_COLUMN_WEB_SEARCH_CONTEXT_SIZE", "high"))

llm = ChatOpenAI(model=SCREENING_CRITERIA_MODEL, temperature=0.0)
specialist_llm = ChatOpenAI(model=SCREENING_REASONING_MODEL, temperature=0.1)
generation_llm = ChatOpenAI(model=SCREENING_GENERATION_MODEL, temperature=0.2)
embeddings = OpenAIEmbeddings(model=SCREENING_EMBEDDING_MODEL)

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


def count_active_candidates_from_db() -> Optional[int]:
    """Return active candidate count, or None when the DB cannot be checked."""
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        logger.error("Could not count active candidates: no database connection")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM candidates
                WHERE COALESCE(is_archived, FALSE) = FALSE
                """
            )
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
    except Exception as e:
        logger.error("Failed to count active candidates: %s", e, exc_info=True)
        return None
    finally:
        return_db_connection(conn)


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
                NULL as embedding, c.created_by, c.raw_fields, c.email, COALESCE(c.mobile_phone, c.phone),
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
        logger.error("Failed to load profiles: no database connection")
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

        cur.execute(_ROLES_SELECT_BODY + " ORDER BY r.candidate_id, r.id ASC")
        roles_raw = cur.fetchall()
        roles_by_candidate = _roles_by_candidate_from_roles_raw(roles_raw)
        profiles = _profile_dicts_from_candidates_and_roles(candidates_raw, roles_by_candidate)

        logger.info(f"Successfully loaded and cached {len(profiles)} profiles.")
        return profiles
    except Exception as e:
        logger.error("Failed to load profiles: %s", e, exc_info=True)
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
            ORDER BY r.candidate_id, r.id ASC
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

        if not profiles:
            active_count = count_active_candidates_from_db()
            if active_count is None:
                CACHE_INITIALIZED = bool(PROFILES_BY_ID)
                logger.error(
                    "Profile cache refresh returned zero rows and active DB count is unavailable; "
                    "preserving existing cache profiles=%s initialized=%s",
                    len(PROFILES_BY_ID),
                    CACHE_INITIALIZED,
                )
                return False

            if active_count > 0:
                CACHE_INITIALIZED = bool(PROFILES_BY_ID)
                logger.error(
                    "Profile cache refresh returned zero rows while DB has %s active candidates; "
                    "preserving existing cache profiles=%s initialized=%s",
                    active_count,
                    len(PROFILES_BY_ID),
                    CACHE_INITIALIZED,
                )
                return False

        next_profiles_by_id = {p["id"]: p for p in profiles}
        companies = load_all_company_names_from_db()

        PROFILES_BY_ID.clear()
        PROFILES_BY_ID.update(next_profiles_by_id)
        _PROFILES_CACHE = profiles
        ALL_COMPANY_NAMES.clear()
        ALL_COMPANY_NAMES.extend(companies)

        CACHE_INITIALIZED = True
        logger.info(f"Cache initialized with {len(PROFILES_BY_ID)} profiles and {len(ALL_COMPANY_NAMES)} companies.")
        return True
    except Exception as e:
        CACHE_INITIALIZED = bool(PROFILES_BY_ID)
        logger.error("Failed to initialize cache: %s", e, exc_info=True)
        return False

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
        if "values" in crit_val:
            values = crit_val.get("values", [])
        else:
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target"):
                if crit_val.get(key):
                    values = [crit_val.get(key)]
                    break
    elif isinstance(crit_val, list):
        values = crit_val

    flat_values = []
    for item in values:
        if isinstance(item, str):
            flat_values.append(item)
        elif isinstance(item, dict):
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target"):
                if item.get(key):
                    flat_values.append(str(item.get(key)))
                    break
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, str):
                    flat_values.append(sub_item)
                elif isinstance(sub_item, dict):
                    for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target"):
                        if sub_item.get(key):
                            flat_values.append(str(sub_item.get(key)))
                            break
    return flat_values

def get_list_from_llm_json(llm_json_response: Any) -> List[str]:
    if isinstance(llm_json_response, list):
        return [str(item) for item in llm_json_response if isinstance(item, str)]
    if isinstance(llm_json_response, dict):
        for value in llm_json_response.values():
            if isinstance(value, list):
                return [str(item) for item in value if isinstance(item, str)]
    return []


def _criteria_objects(value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, dict):
        if isinstance(value.get("values"), list):
            objects: List[Dict[str, Any]] = []
            for item in value.get("values") or []:
                if isinstance(item, dict):
                    objects.append(item)
                elif str(item or "").strip():
                    objects.append({"value": str(item).strip()})
            if objects:
                return objects
        return [value]
    if isinstance(value, list):
        objects = []
        for item in value:
            if isinstance(item, dict):
                objects.append(item)
            elif str(item or "").strip():
                objects.append({"value": str(item).strip()})
        return objects
    if str(value or "").strip():
        return [{"value": str(value).strip()}]
    return []


def _criteria_alias_terms(criterion: Any, value: str) -> List[str]:
    value_l = _normalize_search_text(value)
    terms = set()
    for item in _criteria_objects(criterion):
        item_values = {
            _normalize_search_text(item.get(key))
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target")
            if item.get(key)
        }
        aliases: List[Any] = []
        for key in ("aliases", "expanded_terms", "accepted_terms", "countries", "regions", "company_aliases"):
            raw = item.get(key)
            if isinstance(raw, list):
                aliases.extend(raw)
            elif isinstance(raw, str):
                aliases.extend(re.split(r"[,;|]", raw))
        alias_terms = {_normalize_search_text(alias) for alias in aliases if str(alias or "").strip()}
        if value_l and (value_l in item_values or value_l in alias_terms):
            terms.update(item_values)
            terms.update(alias_terms)
    return sorted(term for term in terms if term)


def _criterion_match_terms(value: str, criterion_key: str, criterion: Any) -> List[str]:
    terms = set(_expanded_terms(value, criterion_key))
    terms.update(_criteria_alias_terms(criterion, value))
    return sorted(term for term in terms if term)


def _with_criteria_terms(values: List[str], criterion: Any, criterion_key: str) -> List[str]:
    terms = set()
    for value in values:
        terms.update(_criterion_match_terms(value, criterion_key, criterion))
    return sorted(term for term in terms if term)

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
    req_values = [_normalize_search_text(v) for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []
    req_terms = _with_criteria_terms(req_values, criteria_obj, "required_functions")

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
        if any(_term_matches_text(v, role_text) for v in req_terms):
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
    req_terms = _with_criteria_terms(req_values, criteria_obj, "required_industries")

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        role_text = f"{(role.get('company') or '').lower()} {(company_details.get('industry', '') or '').lower()} {(company_details.get('product_service', '') or '').lower()}"
        if any(_term_matches_text(v, role_text) for v in req_terms):
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

    req_terms = _with_criteria_terms(req_values, criteria_obj, "required_geographies")

    matching_roles = []
    contributing_roles = []
    for role in profile.get('roles', []):
        company_details = role.get('company_details', {})
        allowed_company_geo = {
            key: company_details.get(key)
            for key in (
                "headquarters",
                "office_locations",
                "offices",
                "locations",
                "operations",
                "operating_locations",
                "company_locations",
            )
            if company_details.get(key)
        }
        combined = " ".join(
            _flatten_value_for_evidence(
                {
                    "role_title": role.get("title"),
                    "role_details": role.get("details"),
                    "role_location": role.get("location"),
                    "role_city": role.get("city"),
                    "source_location": role.get("source_location"),
                    "company_location": role.get("company_location"),
                    "company_geography": allowed_company_geo,
                },
                max_items=40,
            )
        ).lower()

        if any(_term_matches_text(v, combined) for v in req_terms):
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


TEXT_CRITERIA_CONFIG = {
    "required_keywords": {"label": "Keywords", "weight": 1.0, "taxonomy": None},
    "required_companies": {"label": "Companies", "weight": 1.35, "taxonomy": None},
    "required_industries": {"label": "Industries", "weight": 1.1, "taxonomy": None},
    "required_functions": {"label": "Functions", "weight": 1.45, "taxonomy": SALES_TAXONOMY},
    "required_segments": {"label": "Customer segments", "weight": 1.2, "taxonomy": SEGMENT_SYNONYMS},
    "required_locations": {"label": "Locations", "weight": 0.9, "taxonomy": None},
    "required_geographies": {"label": "Geographies", "weight": 1.05, "taxonomy": GEOGRAPHY_COUNTRY_TO_REGION_MAP},
    "required_company_details": {"label": "Company details", "weight": 1.1, "taxonomy": COMPANY_DETAILS_TAXONOMY},
    "required_culture_type": {"label": "Culture", "weight": 0.75, "taxonomy": CULTURE_TAXONOMY},
}


def _flatten_value_for_evidence(value: Any, *, max_items: int = 80) -> List[str]:
    parts: List[str] = []

    def walk(item: Any) -> None:
        if len(parts) >= max_items:
            return
        if item is None:
            return
        if isinstance(item, (str, int, float, bool)):
            text = str(item).strip()
            if text:
                parts.append(text)
            return
        if isinstance(item, dict):
            for key, val in item.items():
                key_text = str(key).replace("_", " ").strip()
                if isinstance(val, (dict, list, tuple)):
                    nested = " ".join(_flatten_value_for_evidence(val, max_items=8))
                    if nested:
                        parts.append(f"{key_text}: {nested}")
                else:
                    val_text = str(val or "").strip()
                    if val_text:
                        parts.append(f"{key_text}: {val_text}")
                if len(parts) >= max_items:
                    break
            return
        if isinstance(item, (list, tuple, set)):
            for child in item:
                walk(child)
                if len(parts) >= max_items:
                    break

    walk(value)
    return parts


def _normalize_search_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _term_matches_text(term: str, text: str) -> bool:
    term_l = _normalize_search_text(term)
    if not term_l:
        return False
    if len(term_l) <= 3:
        return re.search(rf"\b{re.escape(term_l)}\b", text) is not None
    return term_l in text


def _evidence_snippet(text: str, term: str, *, max_len: int = 180) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_len:
        return clean
    term_l = _normalize_search_text(term)
    lower = clean.lower()
    idx = lower.find(term_l) if term_l else -1
    if idx < 0:
        return clean[: max_len - 1].rstrip() + "..."
    start = max(0, idx - 60)
    end = min(len(clean), idx + len(term_l) + 90)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix


def build_profile_evidence_chunks(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    def add(source: str, value: Any, *, role: Optional[Dict[str, Any]] = None) -> None:
        if value is None:
            return
        text = " ".join(_flatten_value_for_evidence(value)) if not isinstance(value, str) else value
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if text:
            chunks.append({
                "source": source,
                "text": text[:1200],
                "text_l": _normalize_search_text(text),
                "role": role,
            })

    for key in (
        "name",
        "headline",
        "about",
        "location",
        "city",
        "candidate_services",
        "extracted_industry",
        "response",
        "notes",
    ):
        add(key, profile.get(key))

    raw_fields = profile.get("raw_fields")
    if isinstance(raw_fields, dict):
        add("uploaded fields", raw_fields)

    for idx, role in enumerate(profile.get("roles") or [], start=1):
        company_details = role.get("company_details") or {}
        add(
            f"role {idx}",
            {
                "title": role.get("title"),
                "company": role.get("company"),
                "details": role.get("details"),
                "duration_years": role.get("duration_years"),
                "start_date": role.get("start_date"),
                "end_date": role.get("end_date"),
            },
            role=role,
        )
        add(f"role {idx} company details", company_details, role=role)

    return chunks


def _expanded_terms(value: str, criterion_key: str) -> List[str]:
    value_l = _normalize_search_text(value)
    if not value_l:
        return []

    terms = {value_l}
    if criterion_key == "required_functions":
        for canonical, aliases in SALES_TAXONOMY.items():
            canonical_l = _normalize_search_text(canonical)
            alias_l = {_normalize_search_text(alias) for alias in aliases}
            if value_l == canonical_l or value_l in alias_l:
                terms.add(canonical_l)
                terms.update(alias_l)
    elif criterion_key == "required_segments":
        for canonical, aliases in SEGMENT_SYNONYMS.items():
            canonical_l = _normalize_search_text(canonical)
            alias_l = {_normalize_search_text(alias) for alias in aliases}
            if value_l == canonical_l or value_l in alias_l:
                terms.add(canonical_l)
                terms.update(alias_l)
    elif criterion_key == "required_company_details":
        for canonical, aliases in COMPANY_DETAILS_TAXONOMY.items():
            canonical_l = _normalize_search_text(canonical)
            alias_l = {_normalize_search_text(alias) for alias in aliases}
            if value_l == canonical_l or value_l in alias_l:
                terms.add(canonical_l)
                terms.update(alias_l)
    elif criterion_key == "required_culture_type":
        for canonical, aliases in CULTURE_TAXONOMY.items():
            canonical_l = _normalize_search_text(canonical)
            alias_l = {_normalize_search_text(alias) for alias in aliases}
            if value_l == canonical_l or value_l in alias_l:
                terms.add(canonical_l)
                terms.update(alias_l)
    elif criterion_key == "required_geographies":
        regions = {region for region in GEOGRAPHY_COUNTRY_TO_REGION_MAP.values()}
        if value_l in regions:
            terms.update(country for country, region in GEOGRAPHY_COUNTRY_TO_REGION_MAP.items() if region == value_l)
        mapped_region = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(value_l)
        if mapped_region:
            terms.add(mapped_region)

    return sorted(term for term in terms if term)


def _criterion_operator(criterion: Any) -> str:
    if isinstance(criterion, dict):
        op = str(criterion.get("operator") or "OR").upper()
        return op if op in {"AND", "OR"} else "OR"
    return "OR"


def _score_text_criterion(
    profile: Dict[str, Any],
    criterion_key: str,
    criterion: Any,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    values = get_values_from_criteria(criterion)
    if criterion_key == "required_companies" and isinstance(criterion, list):
        values = [str(item) for item in criterion if str(item).strip()]
    values = [str(value).strip() for value in values if str(value).strip()]
    if not values:
        return {"applicable": False, "score": 1.0, "matched": [], "missing": [], "evidence": [], "met": True}

    matched: List[str] = []
    missing: List[str] = []
    evidence: List[Dict[str, Any]] = []
    matched_roles: List[Dict[str, Any]] = []

    for value in values:
        terms = _criterion_match_terms(value, criterion_key, criterion)
        found = None
        for chunk in chunks:
            if criterion_key == "required_companies":
                if not chunk["source"].startswith("role"):
                    continue
                role_company = chunk.get("role", {}).get("company", "") or ""
                term = next((term for term in terms if _term_matches_text(term, role_company.lower())), None)
            else:
                term = next((term for term in terms if _term_matches_text(term, chunk["text_l"])), None)
            
            if term:
                found = (term, chunk)
                break
        if found:
            term, chunk = found
            matched.append(value)
            evidence.append({
                "criterion": TEXT_CRITERIA_CONFIG.get(criterion_key, {}).get("label", criterion_key),
                "value": value,
                "source": chunk["source"],
                "snippet": _evidence_snippet(chunk["text"], term),
            })
            if chunk.get("role"):
                matched_roles.append(chunk["role"])
        else:
            missing.append(value)

    operator = _criterion_operator(criterion)
    met = len(matched) == len(values) if operator == "AND" else bool(matched)
    return {
        "applicable": True,
        "score": len(matched) / max(1, len(values)),
        "matched": matched,
        "missing": missing,
        "evidence": evidence,
        "matched_roles": matched_roles,
        "met": met,
        "operator": operator,
    }


def _profile_claim_geography_text(profile: Dict[str, Any]) -> str:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    likely_claims: Dict[str, Any] = {}
    for key, value in raw_fields.items():
        key_l = _normalize_search_text(key)
        if any(token in key_l for token in ("geograph", "market", "region", "country", "territory", "location")):
            likely_claims[key] = value
    return " ".join(_flatten_value_for_evidence(likely_claims, max_items=60)).lower()


def _role_geography_text(role: Dict[str, Any]) -> str:
    company_details = role.get("company_details") or {}
    allowed_company_geo = {
        key: company_details.get(key)
        for key in (
            "headquarters",
            "office_locations",
            "offices",
            "locations",
            "operations",
            "operating_locations",
            "company_locations",
        )
        if company_details.get(key)
    }
    return " ".join(
        _flatten_value_for_evidence(
            {
                "title": role.get("title"),
                "details": role.get("details"),
                "location": role.get("location"),
                "city": role.get("city"),
                "source_location": role.get("source_location"),
                "company_location": role.get("company_location"),
                "company_geography": allowed_company_geo,
            },
            max_items=60,
        )
    ).lower()


def _score_geography_criterion(
    profile: Dict[str, Any],
    criterion: Any,
) -> Dict[str, Any]:
    values = [str(value).strip() for value in get_values_from_criteria(criterion) if str(value).strip()]
    if not values:
        return {"applicable": False, "score": 1.0, "matched": [], "missing": [], "evidence": [], "met": True}

    matched: List[str] = []
    missing: List[str] = []
    evidence: List[Dict[str, Any]] = []
    matched_roles: List[Dict[str, Any]] = []
    profile_claim_text = _profile_claim_geography_text(profile)

    for value in values:
        terms = _criterion_match_terms(value, "required_geographies", criterion)
        found = None
        for role in profile.get("roles") or []:
            role_text = _role_geography_text(role)
            term = next((term for term in terms if _term_matches_text(term, role_text)), None)
            if term:
                found = ("role/company geography", role, role_text, term)
                break
        if not found:
            term = next((term for term in terms if _term_matches_text(term, profile_claim_text)), None)
            if term:
                found = ("uploaded geography claims", None, profile_claim_text, term)

        if found:
            source, role, text, term = found
            matched.append(value)
            evidence.append({
                "criterion": "Geographies",
                "value": value,
                "source": source,
                "snippet": _evidence_snippet(text, term),
            })
            if role:
                matched_roles.append(role)
        else:
            missing.append(value)

    operator = _criterion_operator(criterion)
    met = len(matched) == len(values) if operator == "AND" else bool(matched)
    return {
        "applicable": True,
        "score": len(matched) / max(1, len(values)),
        "matched": matched,
        "missing": missing,
        "evidence": evidence,
        "matched_roles": matched_roles,
        "met": met,
        "operator": operator,
    }


def _normalize_company_key(value: Any) -> str:
    text = _normalize_search_text(value)
    text = re.sub(r"\b(inc|inc\.|llc|ltd|limited|pvt|private|corp|corporation|technologies|technology|software)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .,-")


def _company_matches(candidate_company: str, target_company: str) -> bool:
    candidate = _normalize_company_key(candidate_company)
    target = _normalize_company_key(target_company)
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    candidate_pattern = rf"\b{re.escape(candidate)}\b"
    target_pattern = rf"\b{re.escape(target)}\b"
    return (
        re.search(candidate_pattern, target) is not None
        or re.search(target_pattern, candidate) is not None
    )


def _competitor_items(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = _criteria_objects(criteria.get("competitor_of"))
    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    web_competitors = web_facts.get("competitors") if isinstance(web_facts.get("competitors"), list) else []
    for item in items:
        target = str(item.get("target") or item.get("value") or item.get("name") or "").strip()
        if item.get("companies") or item.get("competitors") or not target:
            continue
        for web_item in web_competitors:
            web_target = str(web_item.get("target") or "").strip()
            if web_target and _company_matches(web_target, target):
                item["companies"] = web_item.get("companies") or web_item.get("competitors") or []
                item["sources"] = web_item.get("sources") or []
                break
    return items


def _score_competitor_criteria(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[bool, float, float, List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = _competitor_items(criteria)
    if not items:
        return True, 0.0, 0.0, [], [], [], []

    total_weight = 0.0
    earned_weight = 0.0
    matched_criteria: List[Dict[str, Any]] = []
    missing_criteria: List[str] = []
    evidence: List[Dict[str, Any]] = []
    roles_for_reasoning: List[Dict[str, Any]] = []
    roles = profile.get("roles") or []

    for item in items:
        target = str(item.get("target") or item.get("value") or item.get("name") or "").strip()
        raw_companies = item.get("companies") or item.get("competitors") or item.get("competitor_companies") or []
        if isinstance(raw_companies, str):
            raw_companies = re.split(r"[,;|]", raw_companies)
        competitor_companies = [str(company).strip() for company in raw_companies if str(company or "").strip()]
        scope = _normalize_search_text(item.get("employment_scope") or item.get("scope") or "current_employer")
        current_only = scope not in {"any_employer", "past_or_current", "worked_at", "worked_with", "all_roles"}

        if not competitor_companies:
            missing_criteria.append(f"Competitor relationship for {target or 'target company'} needs web verification")
            evidence.append({
                "criterion": "Competitor company facts",
                "value": target,
                "source": "web required",
                "snippet": "No competitor list was present in DB/profile criteria; verifier must resolve it with web sources.",
            })
            continue

        total_weight += 1.5

        search_roles = roles[:1] if current_only else roles
        matched_role = None
        matched_company = None
        for role in search_roles:
            role_company = role.get("company") or ""
            for competitor in competitor_companies:
                if _company_matches(role_company, competitor):
                    matched_role = role
                    matched_company = competitor
                    break
            if matched_role:
                break

        if matched_role:
            earned_weight += 1.5
            role_scope = "current employer" if current_only else "employer history"
            matched_criteria.append({
                "criterion": "Competitor employer",
                "value": f"{matched_role.get('company')} matched {target or 'target'} competitor {matched_company}",
            })
            evidence.append({
                "criterion": "Competitor employer",
                "value": target,
                "source": role_scope,
                "snippet": f"{matched_role.get('title') or 'Role'} at {matched_role.get('company')} matched the dynamically resolved competitor list.",
            })
            roles_for_reasoning.append(matched_role)
        else:
            missing_criteria.append(
                f"{'Current employer' if current_only else 'Employer history'} at a verified {target or 'target'} competitor"
            )
            return False, total_weight, earned_weight, matched_criteria, missing_criteria, evidence, roles_for_reasoning

    return True, total_weight, earned_weight, matched_criteria, missing_criteria, evidence, roles_for_reasoning


FUNDING_STAGE_RANKS = {
    "pre seed": 0,
    "pre-seed": 0,
    "seed": 1,
    "series a": 2,
    "series b": 3,
    "series c": 4,
    "series d": 5,
    "series e": 6,
    "series f": 7,
    "series g": 8,
    "growth": 9,
    "growth equity": 9,
    "private equity": 10,
    "pe": 10,
    "ipo": 11,
    "public": 11,
    "publicly traded": 11,
}


def _funding_rank(value: Any) -> Optional[int]:
    text = _normalize_search_text(value)
    if not text:
        return None
    if "public" in text or "ipo" in text:
        return FUNDING_STAGE_RANKS["public"]
    if "private equity" in text or re.search(r"\bpe backed\b|\bpe-owned\b", text):
        return FUNDING_STAGE_RANKS["private equity"]
    if "growth" in text:
        return FUNDING_STAGE_RANKS["growth"]
    series_match = re.search(r"\bseries\s*([a-z])\+?\b", text)
    if series_match:
        letter = series_match.group(1).lower()
        return max(2, ord(letter) - ord("a") + 2)
    for key, rank in FUNDING_STAGE_RANKS.items():
        if _term_matches_text(key, text):
            return rank
    return None


def _funding_min_value(criteria: Dict[str, Any]) -> Optional[str]:
    value = criteria.get("funding_stage_min")
    if not value:
        return None
    if isinstance(value, dict):
        for key in ("stage", "value", "name", "min_stage"):
            if value.get(key):
                return str(value.get(key))
    return str(value)


def _web_company_fact_items(criteria: Dict[str, Any], fact_key: str) -> List[Dict[str, Any]]:
    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    items = web_facts.get(fact_key)
    return items if isinstance(items, list) else []


def _web_company_funding_rank(company_name: str, criteria: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
    for item in _web_company_fact_items(criteria, "funding"):
        if not isinstance(item, dict):
            continue
        item_company = str(item.get("company") or item.get("name") or "").strip()
        if not _company_matches(company_name, item_company):
            continue
        rank = _funding_rank(item.get("stage") or item.get("funding_stage") or item.get("status"))
        if rank is not None:
            return rank, item
    return None


def _web_company_office_match(company_name: str, terms: List[str], criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in _web_company_fact_items(criteria, "geography"):
        if not isinstance(item, dict):
            continue
        item_company = str(item.get("company") or item.get("name") or "").strip()
        if not _company_matches(company_name, item_company):
            continue
        text = " ".join(
            _flatten_value_for_evidence(
                {
                    "offices": item.get("offices"),
                    "operations": item.get("operations"),
                    "headquarters": item.get("headquarters"),
                    "geographies": item.get("geographies"),
                },
                max_items=80,
            )
        ).lower()
        if any(_term_matches_text(term, text) for term in terms):
            return item
    return None


def _score_funding_stage(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[bool, float, float, List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    min_stage = _funding_min_value(criteria)
    min_rank = _funding_rank(min_stage)
    if min_rank is None:
        return True, 0.0, 0.0, [], [], [], []

    stage_criterion = criteria.get("funding_stage_min")
    scope = "current_employer"
    if isinstance(stage_criterion, dict):
        scope = _normalize_search_text(stage_criterion.get("employment_scope") or stage_criterion.get("scope") or "current_employer")
    current_only = scope not in {"any_employer", "past_or_current", "worked_at", "worked_with", "all_roles"}
    roles = profile.get("roles") or []
    roles_to_check = roles[:1] if current_only else roles
    known_below: List[str] = []

    for role in roles_to_check:
        company_details = role.get("company_details") or {}
        db_stage_text = " ".join(
            _flatten_value_for_evidence(
                {
                    "funding_stage": company_details.get("funding_stage"),
                    "business_model": company_details.get("business_model"),
                    "company_status": company_details.get("company_status"),
                    "ownership": company_details.get("ownership"),
                },
                max_items=20,
            )
        )
        web_rank_item = _web_company_funding_rank(role.get("company") or "", criteria)
        source = "DB company details"
        sources = []
        stage_text = db_stage_text
        rank = _funding_rank(db_stage_text)
        if web_rank_item:
            rank, web_item = web_rank_item
            stage_text = str(web_item.get("stage") or web_item.get("funding_stage") or web_item.get("status") or "")
            source = "web company facts"
            sources = web_item.get("sources") if isinstance(web_item.get("sources"), list) else []
        if rank is None:
            continue
        if rank >= min_rank:
            return True, 1.2, 1.2, [{
                "criterion": "Funding stage",
                "value": f"{role.get('company')} is {stage_text or 'funding stage matched'}",
            }], [], [{
                "criterion": "Funding stage",
                "value": min_stage,
                "source": source,
                "snippet": f"{role.get('company')}: {stage_text}",
                "sources": sources,
            }], [role]
        known_below.append(f"{role.get('company')}: {stage_text}")

    if known_below:
        return False, 1.2, 0.0, [], [f"Funding stage >= {min_stage}; DB shows below threshold ({'; '.join(known_below[:2])})"], [], []

    has_other_candidate_filters = any(
        criteria.get(key)
        for key in (
            "required_companies",
            "required_industries",
            "required_functions",
            "required_segments",
            "required_locations",
            "required_geographies",
            "required_keywords",
            "required_culture_type",
            "min_total_experience",
            "min_people_managed",
            "min_function_years",
        )
    )
    unknown_missing = [f"Funding stage >= {min_stage} needs web verification"]
    unknown_evidence = [{
        "criterion": "Funding stage",
        "value": min_stage,
        "source": "web required",
        "snippet": "No reliable funding stage was present in DB company details; verifier must resolve it with web sources.",
    }]
    if not has_other_candidate_filters:
        return False, 1.2, 0.0, [], unknown_missing, unknown_evidence, []

    return True, 0.0, 0.0, [], unknown_missing, unknown_evidence, []


def _min_function_year_items(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = _criteria_objects(criteria.get("min_function_years"))
    required_functions = criteria.get("required_functions")
    if isinstance(required_functions, dict) and required_functions.get("min_years"):
        for value in get_values_from_criteria(required_functions):
            items.append({
                "function": value,
                "min_years": required_functions.get("min_years"),
                "aliases": _criteria_alias_terms(required_functions, value),
            })
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        function = str(item.get("function") or item.get("value") or item.get("name") or "").strip()
        min_years = item.get("min_years") or item.get("years") or item.get("minimum_years")
        key = (_normalize_search_text(function), str(min_years))
        if function and min_years and key not in seen:
            next_item = dict(item)
            next_item["function"] = function
            next_item["min_years"] = min_years
            deduped.append(next_item)
            seen.add(key)
    return deduped


def _score_min_function_years(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[bool, float, float, List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    items = _min_function_year_items(criteria)
    if not items:
        return True, 0.0, 0.0, [], [], [], [], {}

    total_weight = 1.35 * len(items)
    earned_weight = 0.0
    matched_criteria: List[Dict[str, Any]] = []
    missing_criteria: List[str] = []
    evidence: List[Dict[str, Any]] = []
    contributing_roles: List[Dict[str, Any]] = []
    calculated: Dict[str, Any] = {}

    for item in items:
        function = item["function"]
        min_years = float(item.get("min_years") or 0)
        criteria_obj = {
            "values": [{
                "function": function,
                "aliases": item.get("aliases") or item.get("expanded_terms") or [],
            }],
        }
        duration, roles = calculate_functional_experience_duration(profile, criteria_obj)
        calculated[f"min_function_years:{function}"] = {
            "duration": duration,
            "roles": roles,
            "label": function,
            "required": min_years,
        }
        if duration < min_years:
            missing_criteria.append(f"{function} experience >= {min_years:g} years")
            return False, total_weight, earned_weight, matched_criteria, missing_criteria, evidence, contributing_roles, calculated
        earned_weight += 1.35
        matched_criteria.append({"criterion": "Function-specific tenure", "value": f"{duration:g} years in {function}"})
        for role in roles[:3]:
            evidence.append({
                "criterion": "Function-specific tenure",
                "value": function,
                "source": "role history",
                "snippet": f"{role.get('title')} at {role.get('company')} for {role.get('duration_years', 0):g} years",
            })
        contributing_roles.extend(roles)

    return True, total_weight, earned_weight, matched_criteria, missing_criteria, evidence, contributing_roles, calculated


def score_candidate_against_criteria(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    profile_copy = copy.deepcopy({k: v for k, v in profile.items() if k != "embedding"})
    chunks = build_profile_evidence_chunks(profile_copy)
    matched_criteria: List[Dict[str, Any]] = []
    missing_criteria: List[str] = []
    evidence_log: List[Dict[str, Any]] = []
    contributing_roles: List[Dict[str, Any]] = []
    calculated_experience: Dict[str, Any] = {}
    total_weight = 0.0
    earned_weight = 0.0

    min_total_exp = criteria.get("min_total_experience")
    if min_total_exp is not None:
        actual = float(profile_copy.get("total_experience_years") or 0)
        total_weight += 1.2
        if actual >= float(min_total_exp):
            earned_weight += 1.2
            matched_criteria.append({"criterion": "Total experience", "value": f"{actual:g} years"})
            evidence_log.append({
                "criterion": "Total experience",
                "value": str(min_total_exp),
                "source": "profile",
                "snippet": f"Total experience {actual:g} years",
            })
        else:
            missing_criteria.append(f"Total experience >= {min_total_exp} years")
            return None

    min_managed = criteria.get("min_people_managed")
    if min_managed is not None:
        actual = int(profile_copy.get("max_people_managed") or 0)
        total_weight += 0.8
        if actual >= int(min_managed):
            earned_weight += 0.8
            matched_criteria.append({"criterion": "People managed", "value": str(actual)})
        else:
            missing_criteria.append(f"Managed team size >= {min_managed}")
            return None

    if not check_excluded_geography_presence(profile_copy, criteria):
        return None
    if not check_tenure_in_latest_role(profile_copy, criteria):
        return None
    if not check_avg_tenure_in_last_n_roles(profile_copy, criteria):
        return None

    for checker in (_score_competitor_criteria, _score_funding_stage):
        met, weight, earned, matched, missing, evidence, roles = checker(profile_copy, criteria)
        total_weight += weight
        earned_weight += earned
        matched_criteria.extend(matched)
        missing_criteria.extend(missing)
        evidence_log.extend(evidence)
        contributing_roles.extend(roles)
        if not met:
            return None

    (
        function_years_met,
        function_years_weight,
        function_years_earned,
        function_years_matched,
        function_years_missing,
        function_years_evidence,
        function_years_roles,
        function_years_calculated,
    ) = _score_min_function_years(profile_copy, criteria)
    total_weight += function_years_weight
    earned_weight += function_years_earned
    matched_criteria.extend(function_years_matched)
    missing_criteria.extend(function_years_missing)
    evidence_log.extend(function_years_evidence)
    contributing_roles.extend(function_years_roles)
    calculated_experience.update(function_years_calculated)
    if not function_years_met:
        return None

    for key, config in TEXT_CRITERIA_CONFIG.items():
        criterion = criteria.get(key)
        if key == "required_geographies":
            result = _score_geography_criterion(profile_copy, criterion)
        else:
            result = _score_text_criterion(profile_copy, key, criterion, chunks)
        if not result["applicable"]:
            continue

        if (
            key == "required_geographies"
            and not result["matched"]
            and SCREENING_WEB_SEARCH_DEFAULT
            and profile_copy.get("roles")
        ):
            web_geo_matched = []
            web_geo_evidence = []
            web_geo_roles = []
            for value in get_values_from_criteria(criterion):
                terms = _criterion_match_terms(str(value), "required_geographies", criterion)
                for role in profile_copy.get("roles") or []:
                    web_geo = _web_company_office_match(role.get("company") or "", terms, criteria)
                    if not web_geo:
                        continue
                    web_geo_matched.append(str(value))
                    web_geo_roles.append(role)
                    web_geo_evidence.append({
                        "criterion": config["label"],
                        "value": str(value),
                        "source": "web company facts",
                        "snippet": f"{role.get('company')}: {web_geo.get('offices') or web_geo.get('operations') or web_geo.get('headquarters') or web_geo.get('geographies')}",
                        "sources": web_geo.get("sources") if isinstance(web_geo.get("sources"), list) else [],
                    })
                    break
            if web_geo_matched:
                result = {
                    **result,
                    "score": len(web_geo_matched) / max(1, len(get_values_from_criteria(criterion))),
                    "matched": web_geo_matched,
                    "missing": [value for value in result["missing"] if str(value) not in set(web_geo_matched)],
                    "evidence": web_geo_evidence,
                    "matched_roles": web_geo_roles,
                    "met": True if _criterion_operator(criterion) == "OR" else len(web_geo_matched) == len(get_values_from_criteria(criterion)),
                }
            else:
                missing_criteria.extend(f"{config['label']}: {value} needs company-office web verification" for value in result["missing"])
                evidence_log.append({
                    "criterion": config["label"],
                    "value": ", ".join(result["missing"]),
                    "source": "web required",
                    "snippet": "No profile/DB geography evidence matched; verifier may use web sources only for employer office/operations facts.",
                })
                continue

        weight = float(config["weight"])
        total_weight += weight
        earned_weight += weight * float(result["score"])

        if result["matched"]:
            matched_criteria.append({
                "criterion": config["label"],
                "value": ", ".join(result["matched"]),
                "operator": result.get("operator", "OR"),
            })
            evidence_log.extend(result["evidence"])
            contributing_roles.extend(result.get("matched_roles") or [])

        if result["missing"]:
            missing_criteria.extend(f"{config['label']}: {value}" for value in result["missing"])

        if result.get("operator") == "AND" and not result["met"]:
            return None
            
        # Strict enforcement for companies: if an employer is requested, it MUST be met
        if key == "required_companies" and not result["met"]:
            return None

        calc_map = {
            "required_functions": calculate_functional_experience_duration,
            "required_industries": calculate_industry_experience_duration,
            "required_segments": calculate_segment_experience_duration,
            "required_geographies": calculate_geography_experience_duration,
            "required_company_details": calculate_company_details_experience_duration,
        }
        calc_func = calc_map.get(key)
        if calc_func and isinstance(criterion, dict):
            if key == "required_functions" and criteria.get("min_function_years"):
                continue
            duration, roles = calc_func(profile_copy, criterion)
            min_years = float(criterion.get("min_years") or 0)
            calculated_experience[key] = {
                "duration": duration,
                "roles": roles,
                "label": ", ".join(get_values_from_criteria(criterion)),
                "required": min_years,
            }
            if min_years and duration < min_years:
                missing_criteria.append(f"{config['label']} experience >= {min_years:g} years")
                return None

    if total_weight <= 0:
        score = 100.0
    else:
        score = round((earned_weight / total_weight) * 100, 1)

    has_text_criteria = (
        any(criteria.get(key) for key in TEXT_CRITERIA_CONFIG)
        or bool(criteria.get("competitor_of"))
        or bool(criteria.get("funding_stage_min"))
        or bool(criteria.get("min_function_years"))
    )
    threshold = float(os.getenv("SCREENING_MATCH_THRESHOLD", "60"))
    if has_text_criteria and (score < threshold or not evidence_log):
        return None

    seen_roles = set()
    role_details = []
    for role in contributing_roles or profile_copy.get("roles", [])[:3]:
        role_key = (role.get("company"), role.get("title"), role.get("start_date"), role.get("end_date"))
        if role_key in seen_roles:
            continue
        seen_roles.add(role_key)
        role_details.append({
            "company": role.get("company", ""),
            "title": role.get("title", ""),
            "duration_years": role.get("duration_years", 0.0) or 0.0,
            "start_date": role.get("start_date"),
            "end_date": role.get("end_date"),
        })
        if len(role_details) >= 5:
            break

    profile_copy["match_score"] = score
    profile_copy["matched_criteria"] = matched_criteria
    profile_copy["missing_criteria"] = missing_criteria[:12]
    profile_copy["evidence_log"] = evidence_log[:12]
    profile_copy["calculated_experience"] = calculated_experience
    profile_copy["contributing_roles_details"] = {"roles": role_details}
    return profile_copy


async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("Applying ranked evidence filters to %s profiles...", len(profiles))
    matching_candidates: List[Dict[str, Any]] = []

    for profile in profiles:
        scored = score_candidate_against_criteria(profile, criteria)
        if scored:
            matching_candidates.append(scored)

    matching_candidates.sort(
        key=lambda x: (
            x.get("match_score") or 0,
            x.get("total_experience_years") or 0,
            len(x.get("evidence_log") or []),
        ),
        reverse=True,
    )

    top_n = criteria.get("top_n")
    if top_n and top_n > 0:
        matching_candidates = matching_candidates[:top_n]
    elif "top_n" not in criteria and SCREENING_MAX_RESULTS > 0:
        if criteria.get("_source_type") != "master":
            matching_candidates = matching_candidates[:SCREENING_MAX_RESULTS]
    
    return matching_candidates

def _has_source_url(sources: Any) -> bool:
    return any(isinstance(source, dict) and str(source.get("url") or "").strip() for source in (sources or []))


def _fallback_reasoning_from_evidence(profile: Dict[str, Any]) -> str:
    evidence = profile.get("evidence_log") or []
    if not evidence:
        return "Matched from structured profile evidence."
    snippets = []
    for item in evidence[:4]:
        source = item.get("source") or "profile"
        snippet = item.get("snippet") or item.get("value") or ""
        if snippet:
            snippets.append(f"{source}: {snippet}")
    return "Matched from profile evidence: " + "; ".join(snippets[:4])


def _strict_decision_is_match(structured: Dict[str, Any]) -> bool:
    decision = _normalize_search_text(structured.get("decision") or structured.get("match") or structured.get("answer"))
    return decision in {"match", "yes", "true", "qualified", "pass"}


def _uses_dynamic_ai_matching(criteria: Dict[str, Any]) -> bool:
    return any(
        criteria.get(key)
        for key in (
            "competitor_of",
            "min_function_years",
            "funding_stage_min",
            "required_industries",
            "required_geographies",
        )
    )


def _candidate_retrieval_text(profile: Dict[str, Any]) -> str:
    values = _flatten_value_for_evidence(
        {
            "name": profile.get("name"),
            "headline": profile.get("headline"),
            "about": profile.get("about"),
            "raw_fields": profile.get("raw_fields"),
            "roles": profile.get("roles"),
            "candidate_services": profile.get("candidate_services"),
            "extracted_industry": profile.get("extracted_industry"),
        },
        max_items=220,
    )
    return _normalize_search_text(" ".join(values))


def _dynamic_retrieval_terms(original_query: str, criteria: Dict[str, Any]) -> List[str]:
    terms = set()
    for source in (
        original_query,
        get_values_from_criteria(criteria.get("required_keywords")),
        get_values_from_criteria(criteria.get("required_industries")),
        get_values_from_criteria(criteria.get("required_functions")),
        get_values_from_criteria(criteria.get("min_function_years")),
        get_values_from_criteria(criteria.get("required_geographies")),
        get_values_from_criteria(criteria.get("required_company_details")),
    ):
        if isinstance(source, str):
            chunks = re.split(r"[^a-zA-Z0-9+\.]+", source)
        else:
            chunks = []
            for item in source or []:
                chunks.extend(re.split(r"[^a-zA-Z0-9+\.]+", str(item)))
        for chunk in chunks:
            normalized = _normalize_search_text(chunk)
            if len(normalized) >= 3 and normalized not in {
                "candidate", "candidates", "with", "who", "have", "has", "and", "for",
                "years", "experience", "worked", "working", "help", "market", "above",
            }:
                terms.add(normalized)
    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    for item in web_facts.get("competitors") or []:
        if isinstance(item, dict):
            for company in item.get("companies") or []:
                if str(company or "").strip():
                    terms.add(_normalize_search_text(company))
    return sorted(term for term in terms if term)


def _dynamic_company_fact_names(criteria: Dict[str, Any], fact_key: str) -> List[str]:
    names: List[str] = []
    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    for item in web_facts.get(fact_key) or []:
        if not isinstance(item, dict):
            continue
        if fact_key == "competitors":
            names.extend(str(company) for company in (item.get("companies") or []) if str(company or "").strip())
        else:
            company = str(item.get("company") or item.get("name") or "").strip()
            if company:
                names.append(company)
    return names


def _dynamic_candidate_candidates(
    profiles: List[Dict[str, Any]],
    original_query: str,
    criteria: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not _uses_dynamic_ai_matching(criteria):
        return profiles

    company_names = []
    if criteria.get("competitor_of"):
        company_names.extend(_dynamic_company_fact_names(criteria, "competitors"))
    if criteria.get("funding_stage_min"):
        company_names.extend(_dynamic_company_fact_names(criteria, "funding"))
    if criteria.get("required_geographies"):
        company_names.extend(_dynamic_company_fact_names(criteria, "geography"))

    company_matched: List[Dict[str, Any]] = []
    if company_names:
        for profile in profiles:
            roles = profile.get("roles") or []
            for role in roles[:5]:
                role_company = role.get("company") or ""
                if any(_company_matches(role_company, company) for company in company_names):
                    profile_copy = copy.deepcopy({k: v for k, v in profile.items() if k != "embedding"})
                    profile_copy.setdefault("evidence_log", []).append({
                        "criterion": "Dynamic retrieval",
                        "value": role_company,
                        "source": "DB employer matched web company fact",
                        "snippet": f"{role.get('title') or 'Role'} at {role_company}",
                    })
                    company_matched.append(profile_copy)
                    break

    terms = _dynamic_retrieval_terms(original_query, criteria)
    ranked: List[Tuple[int, Dict[str, Any]]] = []
    for profile in profiles:
        text = _candidate_retrieval_text(profile)
        score = sum(1 for term in terms if _term_matches_text(term, text))
        if score:
            ranked.append((score, profile))
    ranked.sort(key=lambda item: (item[0], item[1].get("total_experience_years") or 0), reverse=True)

    merged: List[Dict[str, Any]] = []
    seen = set()
    high_tenure: List[Dict[str, Any]] = []
    min_year_reqs = [
        float(item.get("min_years") or item.get("years") or item.get("minimum_years") or 0)
        for item in _criteria_objects(criteria.get("min_function_years"))
        if str(item.get("min_years") or item.get("years") or item.get("minimum_years") or "").strip()
    ]
    if min_year_reqs:
        min_required = min(min_year_reqs)
        high_tenure = sorted(
            [
                profile for profile in profiles
                if float(profile.get("total_experience_years") or 0) >= min_required
            ],
            key=lambda profile: float(profile.get("total_experience_years") or 0),
            reverse=True,
        )[: max(5, SCREENING_DYNAMIC_VERIFY_LIMIT // 2)]

    for profile in company_matched + high_tenure + [profile for _score, profile in ranked]:
        pid = profile.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        merged.append(profile)
        if len(merged) >= SCREENING_DYNAMIC_VERIFY_LIMIT:
            break

    if merged:
        return merged
    return profiles[:SCREENING_DYNAMIC_VERIFY_LIMIT]


def _needs_company_fact_web_enrichment(criteria: Dict[str, Any]) -> bool:
    return bool(criteria.get("competitor_of"))


def _needs_candidate_company_fact_web_enrichment(criteria: Dict[str, Any]) -> bool:
    return bool(criteria.get("funding_stage_min") or criteria.get("required_geographies"))


def _merge_web_company_facts(criteria: Dict[str, Any], structured: Dict[str, Any]) -> Dict[str, Any]:
    enriched = copy.deepcopy(criteria)
    web_facts = enriched.get("_web_company_facts") if isinstance(enriched.get("_web_company_facts"), dict) else {}

    competitors = structured.get("competitors") if isinstance(structured, dict) else None
    if isinstance(competitors, list):
        normalized_competitors = []
        for item in competitors:
            if not isinstance(item, dict):
                continue
            companies = item.get("companies") or item.get("competitors") or []
            if isinstance(companies, str):
                companies = re.split(r"[,;|]", companies)
            companies = [str(company).strip() for company in companies if str(company or "").strip()]
            sources = item.get("sources") if isinstance(item.get("sources"), list) else []
            normalized_competitors.append({
                "target": str(item.get("target") or "").strip(),
                "companies": companies,
                "sources": sources,
            })
        web_facts["competitors"] = normalized_competitors

    for key in ("funding", "geography"):
        values = structured.get(key) if isinstance(structured, dict) else None
        if isinstance(values, list):
            web_facts[key] = [item for item in values if isinstance(item, dict)]

    enriched["_web_company_facts"] = web_facts
    return enriched


async def enrich_criteria_with_company_web_facts(
    original_query: str,
    criteria: Dict[str, Any],
    tracker: TokenCostTracker,
) -> Dict[str, Any]:
    if not _needs_company_fact_web_enrichment(criteria) or not SCREENING_WEB_SEARCH_DEFAULT:
        return criteria

    system_prompt = (
        "You are a company research assistant for recruiting search. Resolve only company-level facts needed by the query. "
        "Use web evidence when available. Do not infer or create candidate career facts. "
        "Return valid JSON only with keys: competitors, funding, geography, notes. "
        "competitors must be a list of objects with target, companies, sources. "
        "funding must be a list of objects with company, stage/status, sources. "
        "geography must be a list of objects with company, offices/operations/headquarters/geographies, sources. "
        "Every source must include a non-empty url, title, and note. Omit facts that do not have reliable source URLs."
    )
    user_prompt = (
        f"Recruiting query:\n{original_query}\n\n"
        f"Extracted structured criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Resolve competitor_of targets dynamically. Prefer direct competitor/category pages or reputable company/research sources. "
        "Do not use a hardcoded taxonomy. Return JSON only."
    )
    try:
        structured = await asyncio.to_thread(
            call_openai_json,
            system_prompt,
            user_prompt,
            model=SCREENING_REASONING_MODEL,
            use_web=True,
            web_search_tool=SCREENING_WEB_SEARCH_TOOL,
            web_search_context_size=SCREENING_WEB_SEARCH_CONTEXT_SIZE,
            temperature=0.0,
            timeout=90.0,
        )
        tracker.add_usage(SCREENING_REASONING_MODEL, f"{system_prompt}\n\n{user_prompt}", json.dumps(structured), "Company Fact Web Enrichment")
    except Exception as e:
        logger.warning("Company fact web enrichment failed: %s", e)
        return criteria

    return _merge_web_company_facts(criteria, structured if isinstance(structured, dict) else {})


def _candidate_company_names_for_web(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[str]:
    counts: Dict[str, int] = {}
    current_only = bool(criteria.get("funding_stage_min") and not criteria.get("required_geographies"))
    for profile in profiles:
        roles = profile.get("roles") or []
        role_iter = roles[:1] if current_only else roles[:5]
        for role in role_iter:
            company = str(role.get("company") or "").strip()
            key = _normalize_company_key(company)
            if not key:
                continue
            counts[company] = counts.get(company, 0) + 1
    return [
        company
        for company, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:SCREENING_COMPANY_FACT_ENRICH_LIMIT]
    ]


async def enrich_criteria_with_candidate_company_web_facts(
    original_query: str,
    criteria: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    tracker: TokenCostTracker,
) -> Dict[str, Any]:
    if not _needs_candidate_company_fact_web_enrichment(criteria) or not SCREENING_WEB_SEARCH_DEFAULT:
        return criteria
    company_names = _candidate_company_names_for_web(candidate_pool, criteria)
    if not company_names:
        return criteria

    system_prompt = (
        "You are a company research assistant for recruiting search. Resolve only company-level facts for the provided company list. "
        "Use live web evidence. Do not infer or create candidate career facts. "
        "Return valid JSON only with keys: funding, geography, notes. "
        "funding: list objects with company, stage/status, sources. "
        "geography: list objects with company, offices/operations/headquarters/geographies, sources. "
        "Every source must include a non-empty url, title, and note. Omit facts that do not have reliable source URLs. "
        "Do not use customer presence, subsidiaries, revenue share, product availability, or customer examples as geography evidence."
    )
    user_prompt = (
        f"Recruiting query:\n{original_query}\n\n"
        f"Extracted structured criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate employer list to verify:\n{json.dumps(company_names, ensure_ascii=False, indent=2)}\n\n"
        "For funding_stage_min, verify whether listed companies meet the threshold. "
        "For geography criteria, verify only offices/operations/headquarters in the requested country/region. "
        "Return only facts for companies from the provided list. Return JSON only."
    )
    try:
        structured = await asyncio.to_thread(
            call_openai_json,
            system_prompt,
            user_prompt,
            model=SCREENING_REASONING_MODEL,
            use_web=True,
            web_search_tool=SCREENING_WEB_SEARCH_TOOL,
            web_search_context_size=SCREENING_WEB_SEARCH_CONTEXT_SIZE,
            temperature=0.0,
            timeout=120.0,
        )
        tracker.add_usage(SCREENING_REASONING_MODEL, f"{system_prompt}\n\n{user_prompt}", json.dumps(structured), "Candidate Company Fact Web Enrichment")
    except Exception as e:
        logger.warning("Candidate company fact web enrichment failed: %s", e)
        return criteria
    return _merge_web_company_facts(criteria, structured if isinstance(structured, dict) else {})


async def evaluate_shortlist_profile(
    profile: Dict[str, Any],
    original_query: str,
    criteria: Dict[str, Any],
    tracker: TokenCostTracker,
    *,
    use_web: bool,
) -> Optional[Dict[str, Any]]:
    profile_safe = copy.deepcopy({k: v for k, v in profile.items() if k != "embedding"})
    context = build_candidate_context(profile_safe)
    career_facts = compute_career_facts(context)
    context_pack = build_candidate_context_pack(context, career_facts)
    routing = classify_ai_column_prompt(original_query)
    query_plan = build_query_plan(
        original_query,
        context,
        [{"key": "decision", "label": "Decision", "type": "text", "primary": True}],
        routing,
    )
    tool_results = run_candidate_query_tools(original_query, context, career_facts, query_plan)
    career_context = career_facts_to_text(career_facts)

    system_prompt = (
        "You are a senior recruitment analyst. Evaluate whether this candidate fits the hiring query. "
        "Return valid JSON only with these exact keys: "
        "decision, match_score, answer, matched_criteria, missing_criteria, evidence, sources, confidence, reasoning. "
        "\ndecision: exactly 'match', 'no_match', or 'unknown'. "
        "\nmatch_score: integer 0-100 reflecting how well the candidate satisfies the query criteria. "
        "\nCRITICAL REJECTION RULES: Interpret relationship intents strictly. If the query asks for candidates working at a specific company (e.g. 'Google'), you MUST reject candidates who only: 1) Have certifications from that company. 2) Sell to that company as a customer. 3) Compete against that company. 4) Use the company's products. The candidate MUST have the company listed as an employer in their role history. If not, return decision='no_match' and match_score below 60."
        "\nDYNAMIC EVIDENCE RULE: Do not use hidden/static taxonomies or your memory alone. Use only the provided DB/profile evidence and live web/company facts in the prompt, plus web search sources you can cite with URLs."
        "\nSTRUCTURED CRITERIA RULES: competitor_of means employer relationship to a competitor of the target company. Use current employer unless employment_scope says any_employer/past_or_current. Verify competitor lists with web sources; do not rely on a hidden/static taxonomy. If a source names a competing product/business unit (for example a marketing cloud or engagement product), employment at the broad parent company counts only when the candidate's role/profile evidence connects them to that competing product/business unit or same business line."
        "\nHARD CRITERIA RULE: Treat every extracted criterion as mandatory. A close match or partial match is no_match unless the user explicitly asks for near matches. If any required criterion is missing or unknown, put it in missing_criteria, set decision='no_match', and keep match_score below 60."
        "\nFor funding_stage_min, use DB company facts first. If DB is missing, use web sources. If reliable sources are unavailable, mark the criterion missing/unknown and do not match by guess. Series C and above includes Series C, later venture rounds, Growth, Private Equity, Public, and IPO."
        "\nFor min_function_years, count only role history/profile evidence for the requested function and its aliases. Do not invent role tenure from web. Show the arithmetic in evidence/reasoning. Do not combine unrelated roles, unrelated functions, geography duration, or total experience to satisfy function-specific years. If the sum is below the requested minimum, reject."
        "\nFor required_industries, require company/product/market category evidence. A generic customer-facing role, CRM usage, certification, customer list, or sales activity does not by itself prove the employer is in that industry."
        "\nFor required_geographies, current candidate location alone does NOT count as market experience. Count role location/details, uploaded focused geography/profile claims, explicit region claims, and verified employer office/operations/headquarters evidence. Do not count subsidiaries, customers, revenue mix, or product/certification mentions."
        "\nanswer: 1-2 natural sentences with your verdict, mentioning actual company names, titles, and durations from the profile. "
        "Example: 'Yes — Sarah has 6 years at Salesforce as an Enterprise AE covering US West, directly meeting the enterprise SaaS requirement.' "
        "\nreasoning: 2-4 sentences explaining the key evidence — walk through what you found in the profile (role history, companies, experience) and how it maps to the query. Be specific; do not use vague phrases like 'the candidate has relevant experience'. "
        "\nmatched_criteria: list of criteria satisfied with the profile evidence that supports each. "
        "\nmissing_criteria: list of criteria not clearly evidenced. "
        "\nevidence: list of evidence items, each with criterion, value, source, and snippet. Label source as DB/profile evidence or web evidence. "
        "\nsources: list of web sources used for company facts if web search was performed, each with url, title, note. "
        "\nconfidence: 'high', 'medium', or 'low' based on the quality of evidence. "
        "Use the full profile, career tool results, and any web evidence available. Do not invent facts not in the data."
    )
    user_prompt = (
        f"Hiring query:\n{original_query}\n\n"
        f"Extracted criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate profile context:\n{json.dumps(context_pack, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Deterministic career tool results:\n{json.dumps(tool_results, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Pre-scored evidence from profile:\n{json.dumps(profile_safe.get('evidence_log') or [], ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Career facts summary:\n{career_context or 'Not available.'}\n\n"
        "Return JSON only. In 'answer', write a human-readable verdict that cites specific roles and companies. "
        "In 'reasoning', walk through the evidence clearly referencing actual data from the profile."
    )

    structured = await asyncio.to_thread(
        call_openai_json,
        system_prompt,
        user_prompt,
        model=SCREENING_REASONING_MODEL,
        use_web=use_web,
        web_search_tool=SCREENING_WEB_SEARCH_TOOL,
        web_search_context_size=SCREENING_WEB_SEARCH_CONTEXT_SIZE,
        temperature=0.0,
        timeout=90.0 if use_web else 40.0,
    )
    tracker.add_usage(SCREENING_REASONING_MODEL, f"{system_prompt}\n\n{user_prompt}", json.dumps(structured), "Shortlist Verification")

    if not structured:
        profile_safe["reasoning"] = _fallback_reasoning_from_evidence(profile_safe)
        profile_safe["confidence"] = profile_safe.get("confidence") or "medium"
        return profile_safe

    sources = structured.get("sources") if isinstance(structured.get("sources"), list) else []
    missing_items = structured.get("missing_criteria") if isinstance(structured.get("missing_criteria"), list) else []
    if missing_items:
        return None
    outputs = {"decision": structured.get("decision", "")}
    verification = verify_smart_column_outputs(
        original_query,
        outputs,
        data_source="web" if use_web else "row",
        sources=sources,
        tool_results=tool_results,
    )

    score = float(structured.get("match_score") or profile_safe.get("match_score") or 0)
    if not _strict_decision_is_match(structured) or score < float(os.getenv("SCREENING_VERIFIED_MATCH_THRESHOLD", "70")):
        return None
    if use_web and verification.get("verification_status") == "failed":
        return None

    profile_safe["match_score"] = round(score, 1)
    profile_safe["matched_criteria"] = structured.get("matched_criteria") if isinstance(structured.get("matched_criteria"), list) else profile_safe.get("matched_criteria", [])
    profile_safe["missing_criteria"] = structured.get("missing_criteria") if isinstance(structured.get("missing_criteria"), list) else profile_safe.get("missing_criteria", [])
    profile_safe["evidence_log"] = structured.get("evidence") if isinstance(structured.get("evidence"), list) else profile_safe.get("evidence_log", [])
    profile_safe["sources"] = sources
    profile_safe["confidence"] = str(structured.get("confidence") or "medium").strip().lower()
    # `answer` is the short natural-language verdict (1-2 sentences, like an AI Column primary_output)
    raw_answer = structured.get("answer") or ""
    raw_reasoning = structured.get("reasoning") or ""
    profile_safe["answer"] = str(raw_answer).replace("|", " ").strip()
    profile_safe["reasoning"] = str(raw_reasoning or raw_answer or _fallback_reasoning_from_evidence(profile_safe)).replace("\n", " ").replace("|", " ")
    profile_safe["verification_status"] = verification.get("verification_status")
    profile_safe["source_verification_status"] = "verified" if _has_source_url(sources) else ("row_context" if not use_web else "not_publicly_verifiable")
    profile_safe["searched_at"] = structured.get("searched_at") or ""
    profile_safe["web_search_tool"] = structured.get("web_search_tool") or ""
    return profile_safe

async def process_query_main(
    query: str,
    session_id: str,
    tracker: TokenCostTracker,
    *,
    screening_user_id: Optional[int] = None,
    screening_role: Optional[str] = None,
    source_type: Optional[str] = None,
    source_role_id: Optional[int] = None,
    pause_event: Optional[asyncio.Event] = None,
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
        CRITICAL: The query may contain typos or grammatical errors. Bravely infer the user's intent and correct misspellings of job titles, skills, and company names when categorizing them.
        CRITICAL: Do not use a hardcoded competitor list. If the user asks for competitors, extract the target company into competitor_of and leave competitor discovery to web enrichment.
        CRITICAL: Candidate career facts must later come from the candidate profile/DB only. Web can verify company facts such as competitors, funding stage, and office/operations presence.
        
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
          If a years/duration phrase is attached to a geography/market phrase (for example "5 years in APAC market" or "5+ years India experience"), put that number as min_years on required_geographies. Do NOT convert that to min_total_experience.
        - "required_company_details": {{"operator": "OR", "values": List[str]}} (e.g. 'SaaS', 'B2B', 'Series A')
        - "competitor_of": List[{{"target": str, "employment_scope": "current_employer"|"any_employer"}}]
          Use current_employer for phrases like "working for competitors"; use any_employer for "worked at/with competitors".
        - "min_function_years": List[{{"function": str, "min_years": number, "aliases": List[str]}}]
          Include aliases from the query intent, e.g. sales development may include SDR/BDR/business development representative; channel sales may include partner/reseller/channel/alliance management; inside sales may include inbound/remote sales; business development may include BD/account development when sales context is present.
        - "funding_stage_min": {{"stage": str, "employment_scope": "current_employer"|"any_employer", "accepted_stages": List[str]}}
          Treat "Series C and above" as stage "Series C" with accepted stages including Series C, Series D/E/F+, Growth, Private Equity, Public, IPO.
        - For "APAC" or other regions in required_geographies, return objects when useful: {{"geography": "APAC", "expanded_terms": List[str]}}. Country searches may include the parent region in expanded_terms so explicit APAC profile claims can match Singapore/India.
        - "required_keywords": {{"operator": "OR", "values": List[str]}} (only when the query has important terms that do not fit the other keys. DO NOT put company names here. Put ALL company names in required_companies, even if there are typos in the query)
        - "min_total_experience": int
          Use only when the query asks for overall candidate experience. Do NOT use it for function-specific years or geography/market years.
        - "min_people_managed": int
        - "top_n": int (default 10 if searching for "top", "best")
        
        Example 1:
        Query: "Account executives in SaaS companies in Singapore with 5 years exp"
        JSON: {{
            "required_functions": {{"operator": "OR", "values": ["Account Executive"]}},
            "required_company_details": {{"operator": "OR", "values": ["SaaS"]}},
            "required_locations": {{"operator": "OR", "values": ["Singapore"]}},
            "min_total_experience": 5
        }}

        Example 2:
        Query: "softweare enginer at mcirosoft" (Note: handle typos bravely and correctly map them)
        JSON: {{
            "required_functions": {{"operator": "OR", "values": ["Software Engineer"]}},
            "required_companies": ["Microsoft"]
        }}

        Example 3:
        Query: "Candidates who are working for CleverTap competitors and has 5 years in channel sales"
        JSON: {{
            "competitor_of": [{{"target": "CleverTap", "employment_scope": "current_employer"}}],
            "min_function_years": [{{"function": "Channel Sales", "min_years": 5, "aliases": ["channel sales", "channel partner", "partner sales", "reseller sales", "alliance management", "alliances"]}}]
        }}

        Example 4:
        Query: "Candidates with 5 years of sales development experience and have worked in APAC market"
        JSON: {{
            "min_function_years": [{{"function": "Sales Development", "min_years": 5, "aliases": ["sales development", "SDR", "BDR", "business development representative"]}}],
            "required_geographies": {{"operator": "OR", "values": [{{"geography": "APAC", "expanded_terms": ["APAC", "Asia Pacific", "India", "Singapore", "Australia", "Japan", "Indonesia", "Malaysia", "Philippines", "Thailand", "Vietnam", "New Zealand"]}}], "min_years": 5}}
        }}

        Example 5:
        Query: "Candidates who are working for CleverTap competitors and has 5 years in APAC market"
        JSON: {{
            "competitor_of": [{{"target": "CleverTap", "employment_scope": "current_employer"}}],
            "required_geographies": {{"operator": "OR", "values": [{{"geography": "APAC", "expanded_terms": ["APAC", "Asia Pacific", "India", "Singapore", "Australia", "Japan", "Indonesia", "Malaysia", "Philippines", "Thailand", "Vietnam", "New Zealand"]}}], "min_years": 5}}
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
        logger.info(f"Extracted Criteria JSON: {json.dumps(criteria)}")
        tracker.add_usage(llm.model_name, prompt_text, criteria_response.content, "Criteria Extraction")
        
        if not criteria:
            criteria = {"required_keywords": {"operator": "OR", "values": [normalized_query]}}
            
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

    if _needs_company_fact_web_enrichment(criteria) and SCREENING_WEB_SEARCH_DEFAULT:
        yield "Resolving company facts with web evidence..."
        criteria = await enrich_criteria_with_company_web_facts(query, criteria, tracker)

    original_criteria = copy.deepcopy(criteria)
    criteria["_screening_query"] = query
    criteria["_source_type"] = source_type

    def _visible_scope_ids() -> Optional[List[int]]:
        if screening_r == "recruiter" and screening_user_id is not None:
            return [
                pid
                for pid, p in PROFILES_BY_ID.items()
                if not p.get("is_archived") and p.get("owner_user_id") == screening_user_id
            ]
        return None

    def _role_scope_ids(role_id: int) -> List[int]:
        conn = get_db_connection()
        if not conn:
            logger.warning("Could not resolve screening role scope: database connection unavailable")
            return []
        try:
            with conn.cursor() as cur:
                if screening_r == "recruiter" and screening_user_id is not None:
                    cur.execute(
                        "SELECT id FROM recruitment_roles WHERE id = %s AND user_id = %s",
                        (role_id, screening_user_id),
                    )
                else:
                    cur.execute("SELECT id FROM recruitment_roles WHERE id = %s", (role_id,))
                if not cur.fetchone():
                    return []
                cur.execute(
                    "SELECT candidate_id FROM recruitment_role_candidates WHERE role_id = %s",
                    (role_id,),
                )
                return [int(row[0]) for row in cur.fetchall() if row and row[0] is not None]
        except Exception as e:
            logger.error("Could not resolve screening role scope %s: %s", role_id, e)
            return []
        finally:
            return_db_connection(conn)

    def _pool_scope_ids() -> Optional[List[int]]:
        visible = _visible_scope_ids()
        normalized_source = (source_type or "master").strip().lower()
        if normalized_source == "role":
            if not source_role_id:
                return []
            role_ids = _role_scope_ids(int(source_role_id))
            if visible is None:
                return role_ids
            visible_set = set(visible)
            return [pid for pid in role_ids if pid in visible_set]
        return visible

    scoped_candidate_ids = _pool_scope_ids()

    def _scoped_build(ids_override: Optional[List[int]] = None):
        scope = scoped_candidate_ids
        if scope is not None:
            if ids_override is not None:
                allowed = set(scope)
                return build_candidate_pool([i for i in ids_override if i in allowed])
            return build_candidate_pool(scope)
        return build_candidate_pool(ids_override)

    scoped_candidate_count = (
        len(scoped_candidate_ids)
        if scoped_candidate_ids is not None
        else len([p for p in PROFILES_BY_ID.values() if not p.get("is_archived")])
    )

    if scoped_candidate_ids is not None and not scoped_candidate_ids:
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return
    
    # 3. SEMANTIC SEARCH (Vector Retrieval)
    yield "Searching database..."

    web_competitor_terms: List[str] = []
    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    for item in web_facts.get("competitors") or []:
        if isinstance(item, dict):
            web_competitor_terms.extend(str(company) for company in (item.get("companies") or []) if str(company or "").strip())
    
    search_query_text = " ".join(
        (criteria.get("required_companies") or []) + 
        get_values_from_criteria(criteria.get("required_keywords")) +
        get_values_from_criteria(criteria.get("required_industries")) +
        get_values_from_criteria(criteria.get("required_functions")) +
        get_values_from_criteria(criteria.get("min_function_years")) +
        get_values_from_criteria(criteria.get("required_segments")) +
        get_values_from_criteria(criteria.get("required_geographies")) +
        get_values_from_criteria(criteria.get("required_locations")) +
        get_values_from_criteria(criteria.get("required_company_details")) +
        get_values_from_criteria(criteria.get("required_culture_type")) +
        get_values_from_criteria(criteria.get("competitor_of")) +
        get_values_from_criteria(criteria.get("funding_stage_min")) +
        web_competitor_terms
    ).strip()
    
    initial_candidate_pool = []
    used_vector_shortlist = False
    
    if scoped_candidate_count <= SCREENING_FULL_SCAN_LIMIT:
        initial_candidate_pool = _scoped_build()
    elif search_query_text:
        try:
            query_embedding = embeddings.embed_query(search_query_text)
            tracker.add_usage(embeddings.model, search_query_text, usage_type="Embedding")
            
            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        if scoped_candidate_ids is not None:
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND id = ANY(%s)
                                  AND embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (scoped_candidate_ids, query_embedding, SCREENING_VECTOR_LIMIT),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (query_embedding, SCREENING_VECTOR_LIMIT),
                            )
                        ids = [row[0] for row in cur.fetchall()]
                        if (
                            not ids
                            and scoped_candidate_ids is not None
                        ):
                            cur.execute(
                                """
                                SELECT id FROM candidates
                                WHERE COALESCE(is_archived, FALSE) = FALSE
                                  AND id = ANY(%s)
                                LIMIT 2000
                                """,
                                (scoped_candidate_ids,),
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

    if _needs_candidate_company_fact_web_enrichment(criteria) and SCREENING_WEB_SEARCH_DEFAULT:
        yield "Updating company facts with web evidence..."
        criteria = await enrich_criteria_with_candidate_company_web_facts(
            query,
            criteria,
            initial_candidate_pool,
            tracker,
        )
    verification_criteria = copy.deepcopy(criteria)
    
    # 4. Soft-rank Candidates
    if _uses_dynamic_ai_matching(criteria):
        yield "Selecting candidates for AI evidence review..."
        final_candidates = _dynamic_candidate_candidates(initial_candidate_pool, query, criteria)
    else:
        final_candidates = await filter_candidates_by_criteria(initial_candidate_pool, criteria)

    if not final_candidates and not _uses_dynamic_ai_matching(criteria) and used_vector_shortlist and len(initial_candidate_pool) < scoped_candidate_count:
        logger.info("Vector shortlist returned no matches after filtering. Retrying scoped full cache.")
        final_candidates = await filter_candidates_by_criteria(_scoped_build(), criteria)
    
    verify_limit = criteria.get("top_n") or SCREENING_WEB_VERIFY_TOP_K
    if verify_limit and verify_limit > 0:
        final_candidates = final_candidates[: int(verify_limit)]

    if not final_candidates:
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    yield {"type": "progress_start", "total": len(final_candidates)}
    
    # 5. Parallel verification and reasoning generation
    CONCURRENCY_LIMIT = max(1, SCREENING_WEB_CONCURRENCY if SCREENING_WEB_SEARCH_DEFAULT else 5)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def verify_profile_safe(profile):
        async with semaphore:
            if pause_event:
                await pause_event.wait()
            try:
                return await evaluate_shortlist_profile(
                    profile,
                    query,
                    verification_criteria,
                    tracker,
                    use_web=SCREENING_WEB_SEARCH_DEFAULT,
                )
            except Exception as e:
                logger.error(f"Shortlist verification failed for {profile.get('id')}: {e}", exc_info=True)
                return None

    tasks = [asyncio.create_task(verify_profile_safe(p)) for p in final_candidates]
    
    processed_count = 0
    processed_candidates = []
    
    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            processed_count += 1
            if result:
                processed_candidates.append(result)
                yield {
                    "type": "profile_chunk",
                    "data": result,
                    "current": processed_count,
                    "total": len(final_candidates)
                }
            else:
                yield {
                    "type": "progress",
                    "current": processed_count,
                    "total": len(final_candidates)
                }
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

    processed_candidates.sort(
        key=lambda x: (
            x.get("match_score") or 0,
            x.get("total_experience_years") or 0,
        ),
        reverse=True,
    )
    
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
    Uses OpenAI embeddings and PostgreSQL pgvector cosine similarity.
    """
    if not query_text:
        return {}
    
    try:
        # 1. Embed the query
        query_vector = await embeddings.aembed_query(query_text)
        
        conn = get_db_connection(validate=False, register_pgvector=True)
        if not conn:
            return {}
            
        try:
            with conn.cursor() as cur:
                # 1 - (cosine distance) = cosine similarity
                cur.execute("""
                    SELECT id, 1 - (embedding <=> %s::vector) as similarity
                    FROM candidates
                    WHERE embedding IS NOT NULL AND COALESCE(is_archived, FALSE) = FALSE
                """, (query_vector,))
                
                scores = {}
                for row in cur.fetchall():
                    scores[row[0]] = float(row[1])
                return scores
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}", exc_info=True)
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

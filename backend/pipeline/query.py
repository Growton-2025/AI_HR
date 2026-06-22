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
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from backend.db.connection import get_db_connection, return_db_connection
from backend.services.ai_columns import (
    build_candidate_context,
    call_openai_json,
    compute_career_facts,
)

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Pricing and Token Configuration ---
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}
tokenizer = tiktoken.get_encoding("cl100k_base")


def _model_pricing(model: str) -> Optional[Dict[str, float]]:
    model_key = str(model or "").strip()
    env_key = re.sub(r"[^A-Z0-9]+", "_", model_key.upper()).strip("_")
    input_env = os.getenv(f"MODEL_PRICE_{env_key}_INPUT")
    output_env = os.getenv(f"MODEL_PRICE_{env_key}_OUTPUT")
    if input_env is not None and output_env is not None:
        try:
            return {"input": float(input_env), "output": float(output_env)}
        except ValueError:
            logger.warning("Invalid pricing env for model %s", model_key)
    return MODEL_PRICING.get(model_key)


class TokenCostTracker:
    """A helper class to track token usage and associated costs."""
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.session_details = []
        self.unknown_pricing_models = set()

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _model_pricing(model)
        if not pricing:
            if model:
                self.unknown_pricing_models.add(str(model))
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
        if self.unknown_pricing_models:
            models = ", ".join(sorted(self.unknown_pricing_models))
            summary_md += f"- **Estimated Cost:** `pricing_missing for {models}`\n"
        else:
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
SCREENING_CRITERIA_MODEL = os.getenv("SCREENING_CRITERIA_MODEL", "gpt-4o-mini")
SCREENING_REASONING_MODEL = os.getenv("SCREENING_REASONING_MODEL", "gpt-4o-mini")
SCREENING_AUDIT_MODEL = os.getenv("SCREENING_AUDIT_MODEL", os.getenv("SCREENING_REASONING_AUDIT_MODEL", "gpt-4o"))
SCREENING_GENERATION_MODEL = os.getenv("SCREENING_GENERATION_MODEL", SCREENING_REASONING_MODEL)
SCREENING_EMBEDDING_MODEL = os.getenv("SCREENING_EMBEDDING_MODEL", "text-embedding-3-small")
SCREENING_MAX_RESULTS = int(os.getenv("SCREENING_MAX_RESULTS", "25"))  # deprecated; shortlist review no longer caps results
SCREENING_VECTOR_LIMIT = int(os.getenv("SCREENING_VECTOR_LIMIT", "750"))  # deprecated for shortlist review completeness
SCREENING_FULL_SCAN_LIMIT = int(os.getenv("SCREENING_FULL_SCAN_LIMIT", "5000"))
SCREENING_WEB_SEARCH_DEFAULT = os.getenv("SCREENING_WEB_SEARCH_DEFAULT", "true").strip().lower() not in {"0", "false", "no"}
SCREENING_COMPANY_FACT_ENRICH_LIMIT = int(os.getenv("SCREENING_COMPANY_FACT_ENRICH_LIMIT", "80"))
SCREENING_LLM_REVIEW_LIMIT = int(os.getenv("SCREENING_LLM_REVIEW_LIMIT", "80"))
SCREENING_LLM_REVIEW_BATCH_SIZE = int(os.getenv("SCREENING_LLM_REVIEW_BATCH_SIZE", "10"))
SCREENING_LLM_REVIEW_MIN_SCORE = float(os.getenv("SCREENING_LLM_REVIEW_MIN_SCORE", "35"))
SCREENING_LOCAL_POTENTIAL_THRESHOLD = float(os.getenv("SCREENING_LOCAL_POTENTIAL_THRESHOLD", "45"))
SCREENING_WEB_SEARCH_TOOL = os.getenv("SCREENING_WEB_SEARCH_TOOL", os.getenv("AI_COLUMN_WEB_SEARCH_TOOL", "web_search"))
SCREENING_WEB_SEARCH_CONTEXT_SIZE = os.getenv("SCREENING_WEB_SEARCH_CONTEXT_SIZE", os.getenv("AI_COLUMN_WEB_SEARCH_CONTEXT_SIZE", "high"))
SHORTLIST_COMPANY_FACT_CACHE_PATH = Path(
    os.getenv(
        "SHORTLIST_COMPANY_FACT_CACHE_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "cache" / "shortlist_company_facts_cache.json"),
    )
)

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

    # 2. Try to find the first outer-most JSON object or array
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')

    if start_idx != -1 and end_idx != -1:
        candidate = json_str[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start_idx = json_str.find('[')
    end_idx = json_str.rfind(']')
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
    'Hunting': ['Hunting', 'new accounts', 'net new', 'New Closures', 'Account Executive', 'new logo', 'hunter'],
    'Farming': ['Account management', 'Account manager', 'Farming', 'Retention'],
    'Sales Development': [
        'Sales Development', 'Business Development', 'inside sales', 'SDR', 'BDR',
        'account development', 'client development', 'outbound', 'outbound sales',
        'outbound prospecting', 'pipeline generation', 'lead generation', 'business development representative',
        'sales development representative'
    ],
    'Partner Sales': ['Partner Sales', 'Partner Development', 'Channel Sales', 'alliance management', 'alliances', 'partner management', 'channel partnerships'],
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
    "series-c": ["series c", "series-c"],
    "series-d": ["series d", "series-d"],
    "series-e": ["series e", "series-e", "series e+", "series f", "series g", "late stage", "growth stage"],
    "public": ["public", "publicly traded", "ipo", "pre-ipo", "listed"],
    "b2b": ["b2b", "business-to-business"],
    "b2c": ["b2c", "business-to-consumer"],
    "saas": ["saas", "software as a service"],
    "customer engagement": ["customer engagement", "customer engagement platform", "clevertap", "marketing automation", "crm", "customer retention platform"]
}

STATIC_CULTURE_TAXONOMY = {
    "startup": ["startup", "fast-paced", "agile environment", "high-growth", "early-stage"],
    "corporate": ["corporate", "mnc", "multinational", "large enterprise", "structured environment", "established company"],
    "remote": ["remote-first", "fully remote", "distributed team"]
}

STATIC_INDUSTRY_DOMAIN_TAXONOMY = {
    "fintech": [
        "fintech", "financial technology", "payment gateway", "payment processing",
        "payment solutions", "cross-border payment", "digital banking", "digital wallet",
        "lending platform", "alternative lending", "peer-to-peer lending", "regtech",
        "insurtech", "wealthtech", "robo-advisory", "financial wellness platform",
    ],
}

STATIC_GEOGRAPHY_MAP = {
    "singapore": "apac", "malaysia": "apac", "indonesia": "apac", "thailand": "apac", "vietnam": "apac",
    "philippines": "apac", "australia": "apac", "new zealand": "apac", "japan": "apac", "south korea": "apac",
    "india": "apac", "hong kong": "apac", "taiwan": "apac", "brunei": "apac", "cambodia": "apac", "laos": "apac",
    "myanmar": "apac", "pakistan": "apac", "bangladesh": "apac", "sri lanka": "apac", "nepal": "apac",
    "asean": "apac",
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
INDUSTRY_DOMAIN_TAXONOMY = STATIC_INDUSTRY_DOMAIN_TAXONOMY
GEOGRAPHY_COUNTRY_TO_REGION_MAP = STATIC_GEOGRAPHY_MAP

# --- Database Loading ---
PROFILES_BY_ID = {}
ALL_COMPANY_NAMES = []
_PROFILES_CACHE = []
CACHE_INITIALIZED = False
_EVIDENCE_CATALOG_CACHE: Optional[Dict[str, Any]] = None

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
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target", "company"):
                if crit_val.get(key):
                    values = [crit_val.get(key)]
                    break
            if not values:
                ignored_keys = {
                    "operator", "scope", "employment_scope", "min_years", "min_function_years",
                    "years", "minimum_years", "field", "dimension", "type", "context",
                    "aliases", "expanded_terms", "accepted_terms", "countries", "regions",
                    "shape", "value_shape", "evidence", "meaning", "comparison",
                    "supports_min_years", "supports_employment_scope",
                }
                for key, value in crit_val.items():
                    if key in ignored_keys or value in (None, "", [], {}):
                        continue
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
    elif isinstance(crit_val, list):
        values = crit_val

    flat_values = []
    for item in values:
        if isinstance(item, str):
            flat_values.append(item)
        elif isinstance(item, dict):
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target", "company"):
                if item.get(key):
                    flat_values.append(str(item.get(key)))
                    break
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, str):
                    flat_values.append(sub_item)
                elif isinstance(sub_item, dict):
                    for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target", "company"):
                        if sub_item.get(key):
                            flat_values.append(str(sub_item.get(key)))
                            break
    return flat_values

def get_list_from_llm_json(llm_json_response: Any) -> List[str]:
    if isinstance(llm_json_response, list):
        values: List[str] = []
        for item in llm_json_response:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                for key in ("company", "name", "value", "target"):
                    if item.get(key):
                        values.append(str(item.get(key)))
                        break
        return values
    if isinstance(llm_json_response, dict):
        for value in llm_json_response.values():
            if isinstance(value, list):
                return [str(item) for item in value if isinstance(item, str)]
    if isinstance(llm_json_response, str):
        return [
            re.sub(r"^[-*\d\.\)\s]+", "", part).strip()
            for part in re.split(r"[\n,;|]", llm_json_response)
            if re.sub(r"^[-*\d\.\)\s]+", "", part).strip()
        ]
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
            for key in ("value", "name", "term", "function", "industry", "geography", "stage", "target", "company")
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

    # Build a broad text corpus from the full profile for market-experience queries.
    # "US market experience" means the candidate sold/operated in the market,
    # not necessarily that they are physically located there.
    profile_headline = (profile.get('headline') or '').lower()
    profile_about = (profile.get('about') or '').lower()
    profile_location = (profile.get('location') or '').lower()
    profile_candidate_services = (profile.get('candidate_services') or '').lower()

    # Broad synonyms for US/NA market experience expressed in profiles
    _GEO_MARKET_SYNONYMS: Dict[str, List[str]] = {
        "united states": ["united states", "us market", "usa market", "north america", "north american", "american clients", "us clients", "us accounts", "us customers", "us region", "us territory", "us sales", "us business", "outbound us", "americas"],
        "us": ["us market", "us clients", "us accounts", "us customers", "us region", "us territory", "us sales", "outbound us", "united states", "north america", "americas"],
        "usa": ["usa", "united states", "us market", "north america", "americas"],
        "north america": ["north america", "north american", "us market", "americas", "united states", "canada"],
        "apac": ["apac", "asia pacific", "asia-pacific", "southeast asia", "sea market"],
        "emea": ["emea", "europe middle east africa", "european market"],
        "uk": ["uk", "united kingdom", "british", "england"],
        "europe": ["europe", "european", "emea"],
        "india": ["india", "indian market", "in market"],
        "singapore": ["singapore", "sg market", "apac"],
    }

    found_values = set()
    for v in values:
        region_for_v = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(v)
        synonyms = _GEO_MARKET_SYNONYMS.get(v, [v])

        # 1. Check candidate location (physical presence)
        if v in profile_location or (region_for_v and region_for_v in profile_location):
            found_values.add(v)
            continue

        # 2. Check headline and about for market-experience signals
        broad_text = f"{profile_headline} {profile_about} {profile_candidate_services}"
        if any(syn in broad_text for syn in synonyms):
            found_values.add(v)
            continue

        # 3. Check role titles, details, and company presence
        for role in profile.get('roles', []):
            role_text = f"{(role.get('title') or '').lower()} {(role.get('details') or '').lower()}"
            company_details = role.get('company_details', {})
            company_hq_text = (company_details.get('headquarters') or '').lower()
            combined = f"{role_text} {company_hq_text}"

            if v in combined or (region_for_v and region_for_v in combined):
                found_values.add(v)
                break
            if any(syn in combined for syn in synonyms):
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
    seen_role_keys = set()
    
    for role in matching_roles:
        role_key = (
            _normalize_company_key(role.get("company")),
            _normalize_search_text(role.get("title")),
            str(role.get("start_date") or ""),
            str(role.get("end_date") or ""),
            round(float(role.get("duration_years") or 0), 2),
        )
        if role_key in seen_role_keys:
            continue
        seen_role_keys.add(role_key)
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


def _coerce_positive_float(value: Any) -> Optional[float]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    return number if number > 0 else None


def _raw_experience_roles(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    indexes = sorted({
        int(match.group(1))
        for key in raw_fields
        for match in [re.match(r"experiences/(\d+)/", str(key))]
        if match
    })
    roles: List[Dict[str, Any]] = []
    today = datetime.now()
    for idx in indexes:
        company = _shortlist_clean_text(raw_fields.get(f"experiences/{idx}/companyName"))
        title_primary = _shortlist_clean_text(raw_fields.get(f"experiences/{idx}/title"))
        title_alt = _shortlist_clean_text(raw_fields.get(f"experiences/{idx}/title.1"))
        # Prefer the canonical title. Some spreadsheets contain a duplicated
        # title.1 header whose value is actually the next company name. Use the
        # alternate only as a fallback or to repair a company-as-title value.
        title = title_primary or title_alt
        if company and _normalize_company_key(title_primary) == _normalize_company_key(company) and title_alt:
            title = title_alt
        details = _shortlist_clean_text(raw_fields.get(f"experiences/{idx}/jobDescription"))
        start_raw = raw_fields.get(f"experiences/{idx}/jobStartedOn")
        end_raw = raw_fields.get(f"experiences/{idx}/jobEndedOn")
        start = parse_date(str(start_raw)) if start_raw else None
        end = parse_date(str(end_raw)) if end_raw else None
        if not start and not _coerce_positive_float(raw_fields.get(f"experiences/{idx}/duration_years")):
            continue
        duration_years = _coerce_positive_float(raw_fields.get(f"experiences/{idx}/duration_years"))
        if duration_years is None and start:
            end_for_calc = end or today
            duration_years = round(max(0, (end_for_calc.year - start.year) * 12 + (end_for_calc.month - start.month)) / 12.0, 2)
        if not (company or title or details):
            continue
        roles.append({
            "company": company,
            "title": title,
            "details": details,
            "duration_years": duration_years or 0.0,
            "start_date": start.date().isoformat() if start else "",
            "end_date": end.date().isoformat() if end else "",
            "location": raw_fields.get(f"experiences/{idx}/jobLocation"),
            "city": raw_fields.get(f"experiences/{idx}/jobLocation"),
            "company_details": {
                "industry": raw_fields.get(f"experiences/{idx}/companyIndustry"),
            },
            "_source": "raw_fields.experiences",
        })
    return roles


def _profile_roles_with_raw_experience(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized_roles = [copy.deepcopy(role) for role in (profile.get("roles") or []) if isinstance(role, dict)]
    raw_roles = _raw_experience_roles(profile)
    if not raw_roles:
        return normalized_roles

    merged = normalized_roles[:]
    for raw_role in raw_roles:
        raw_company = _normalize_company_key(raw_role.get("company"))
        raw_title = _normalize_search_text(raw_role.get("title"))
        matched_existing = None
        for role in merged:
            role_company = _normalize_company_key(role.get("company"))
            role_title = _normalize_search_text(role.get("title"))
            if raw_company and role_company and raw_company != role_company:
                continue
            raw_start = _shortlist_parse_date(raw_role.get("start_date") or raw_role.get("start"))
            role_start = _shortlist_parse_date(role.get("start_date") or role.get("start"))
            same_company_start = bool(
                raw_company
                and role_company
                and raw_company == role_company
                and raw_start
                and role_start
                and raw_start.date() == role_start.date()
            )
            title_matches = bool(raw_title and role_title and (raw_title == role_title or raw_title in role_title or role_title in raw_title))
            company_only = bool(raw_company and role_company and raw_company == role_company and not raw_title and not role_title)
            company_as_title = bool(raw_company and role_company and raw_company == role_company and raw_title == raw_company)
            if same_company_start or title_matches or company_only or company_as_title:
                matched_existing = role
                break
        if matched_existing:
            if not matched_existing.get("start_date") and raw_role.get("start_date"):
                matched_existing["start_date"] = raw_role.get("start_date")
            if not matched_existing.get("end_date") and raw_role.get("end_date"):
                matched_existing["end_date"] = raw_role.get("end_date")
            if not matched_existing.get("duration_years"):
                matched_existing["duration_years"] = raw_role.get("duration_years") or 0.0
            if not matched_existing.get("details") and raw_role.get("details"):
                matched_existing["details"] = raw_role.get("details")
            if not matched_existing.get("title") and raw_role.get("title"):
                matched_existing["title"] = raw_role.get("title")
            if not matched_existing.get("company") and raw_role.get("company"):
                matched_existing["company"] = raw_role.get("company")
            if not matched_existing.get("location") and raw_role.get("location"):
                matched_existing["location"] = raw_role.get("location")
        else:
            merged.append(raw_role)
    return merged


# --- DURATION CALCULATIONS ---
def calculate_functional_experience_duration(profile: Dict[str, Any], criteria_obj: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    if not criteria_obj or not isinstance(criteria_obj, dict): return 0.0, []
    req_values = [_normalize_search_text(v) for v in get_values_from_criteria(criteria_obj)]
    if not req_values: return 0.0, []
    req_terms = _with_criteria_terms(req_values, criteria_obj, "required_functions")

    matching_roles = []
    contributing_roles = []
    for role in _profile_roles_with_raw_experience(profile):
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
    for role in _profile_roles_with_raw_experience(profile):
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
    for role in _profile_roles_with_raw_experience(profile):
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

    req_terms = sorted({term for value in req_values for term in _geography_match_terms(value, criteria_obj)})

    matching_roles = []
    contributing_roles = []
    for role in _profile_roles_with_raw_experience(profile):
        combined = _role_geography_text_for_profile(profile, role)

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
    for role in _profile_roles_with_raw_experience(profile):
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


SCOPED_DURATION_DIMENSION_TO_CRITERION = {
    "function": "required_functions",
    "industry": "required_industries",
    "segment": "required_segments",
    "geography": "required_geographies",
    "company_detail": "required_company_details",
}


def _role_duration_detail(role: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "company": role.get("company", ""),
        "title": role.get("title", ""),
        "duration_years": role.get("duration_years", 0.0) or 0.0,
        "start_date": role.get("start_date"),
        "end_date": role.get("end_date"),
    }


def _scoped_duration_role_text(profile: Dict[str, Any], role: Dict[str, Any], dimension: str) -> str:
    company_details = role.get("company_details") or {}
    if dimension == "function":
        payload = {
            "title": role.get("title"),
            "details": role.get("details"),
        }
    elif dimension == "industry":
        payload = {
            "company": role.get("company"),
            "industry": company_details.get("industry"),
            "product_service": company_details.get("product_service"),
            "business_model": company_details.get("business_model"),
            "details": role.get("details"),
        }
    elif dimension == "segment":
        payload = {
            "customer_segment": company_details.get("customer_segment"),
            "details": role.get("details"),
        }
    elif dimension == "company_detail":
        payload = {
            "funding_stage": company_details.get("funding_stage"),
            "business_model": company_details.get("business_model"),
            "product_service": company_details.get("product_service"),
            "industry": company_details.get("industry"),
            "details": role.get("details"),
        }
    elif dimension == "geography":
        # Market-tenure must come from explicit role/profile market evidence, not
        # from candidate current location or broad employer customer presence.
        payload = {
            "title": role.get("title"),
            "details": role.get("details"),
            "role_location": role.get("location"),
            "source_location": role.get("source_location"),
        }
    else:
        payload = role
    return " ".join(_flatten_value_for_evidence(payload, max_items=80)).lower()


def _has_market_action_text(text: str) -> bool:
    normalized = _normalize_search_text(text)
    return bool(
        re.search(
            r"\b(sold|selling|sell|covered|covering|coverage|owned|owning|managed|handled|generated|prospect(?:ed|ing)?|outreach|pipeline|quota|revenue|territor(?:y|ies)|region(?:al)?|market)\b",
            normalized,
        )
    )


def evaluate_scoped_duration(
    profile: Dict[str, Any],
    *,
    dimension: str,
    criterion: Any,
    min_years: float = 0.0,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    criterion_key = SCOPED_DURATION_DIMENSION_TO_CRITERION.get(dimension, dimension)
    values = [str(value).strip() for value in get_values_from_criteria(criterion) if str(value or "").strip()]
    if not values:
        return {
            "qualified": False,
            "duration": 0.0,
            "required": float(min_years or 0),
            "roles": [],
            "evidence": [],
            "matched_values": [],
            "failure_reason": f"{label or dimension} tenure has no accepted terms",
        }

    terms = sorted({
        term
        for value in values
        for term in (
            _geography_match_terms(value, criterion)
            if dimension == "geography"
            else _criterion_match_terms(value, criterion_key, criterion)
        )
        if term
    })

    matching_roles: List[Dict[str, Any]] = []
    matched_values: List[str] = []
    evidence: List[Dict[str, Any]] = []

    duration_roles = _profile_roles_with_raw_experience(profile)
    if dimension in {"industry", "segment", "company_detail"} and _company_scope_current_only(
        criterion if isinstance(criterion, dict) else {},
        "any_employer",
    ):
        duration_roles = _current_roles({"roles": duration_roles})

    for role in duration_roles:
        role_text = _scoped_duration_role_text(profile, role, dimension)
        if dimension == "geography" and not _has_market_action_text(role_text):
            continue
        matched_term = next((term for term in terms if _term_matches_text(term, role_text)), None)
        if not matched_term:
            continue
        matching_roles.append(role)
        matched_value = next((value for value in values if matched_term in _criterion_match_terms(value, criterion_key, criterion) or matched_term in _geography_match_terms(value, criterion)), matched_term)
        matched_values.append(matched_value)
        evidence.append({
            "criterion": f"{label or TEXT_CRITERIA_CONFIG.get(criterion_key, {}).get('label', dimension)} tenure",
            "value": matched_value,
            "source": "role history",
            "snippet": f"{role.get('title') or 'Role'} at {role.get('company') or 'company'}: {_evidence_snippet(role_text, matched_term)}",
            "source_text": role_text,
            "role": _role_duration_detail(role),
        })

    duration = calculate_merged_duration_years(matching_roles)
    required = float(min_years or 0)
    qualified = duration >= required if required else bool(matching_roles)
    return {
        "qualified": qualified,
        "duration": duration,
        "required": required,
        "roles": [_role_duration_detail(role) for role in matching_roles],
        "evidence": evidence[:6],
        "matched_values": sorted(set(matched_values)),
        "label": label or ", ".join(values),
        "dimension": dimension,
        "failure_reason": "" if qualified else f"{label or ', '.join(values)} tenure >= {required:g} years",
    }


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


def _shortlist_clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _shortlist_parse_date(value: Any) -> Optional[datetime]:
    text = _shortlist_clean_text(value)
    if not text:
        return None
    lower = text.lower()
    if lower in {"present", "current", "now", "ongoing", "till date"}:
        return datetime.utcnow()
    normalized = re.sub(r"\s+", " ", text.replace(",", " ")).strip()
    normalized = normalized.replace("/", "-")
    for pattern in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m-%Y",
        "%Y-%m",
        "%b %Y",
        "%B %Y",
        "%Y",
    ):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        return datetime(int(year_match.group(0)), 1, 1)
    return None


def _shortlist_months_between(start: Optional[datetime], end: Optional[datetime]) -> int:
    if not start:
        return 0
    end = end or datetime.utcnow()
    if end < start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def _shortlist_years(months: int) -> float:
    return round((months or 0) / 12.0, 1)


def _profile_raw_enrichment(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    enrichment = raw_fields.get("enrichment") if isinstance(raw_fields.get("enrichment"), dict) else {}
    return enrichment


def _shortlist_role_windows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    roles = profile.get("roles") or []
    enrichment_roles = _profile_raw_enrichment(profile).get("roles")
    if not roles and isinstance(enrichment_roles, list):
        roles = enrichment_roles
    for role in roles:
        if not isinstance(role, dict):
            continue
        start = _shortlist_parse_date(
            role.get("start_date")
            or role.get("start_raw")
            or role.get("jobStartedOn")
            or role.get("job_started_on")
        )
        end = _shortlist_parse_date(
            role.get("end_date")
            or role.get("end_raw")
            or role.get("jobEndedOn")
            or role.get("job_ended_on")
        )
        if not start:
            continue
        company = _shortlist_clean_text(role.get("company") or role.get("companyName") or role.get("company_name"))
        title = _shortlist_clean_text(role.get("title") or role.get("role"))
        windows.append(
            {
                "company": company,
                "title": title,
                "start": start,
                "end": end,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat() if end else "",
                "duration_months": _shortlist_months_between(start, end),
                "duration_years": _shortlist_years(_shortlist_months_between(start, end)),
            }
        )
    return sorted(windows, key=lambda item: item["start"])


def _shortlist_education_windows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    enrichment = _profile_raw_enrichment(profile)
    education_items = []
    if isinstance(profile.get("education"), list):
        education_items.extend(profile.get("education") or [])
    if isinstance(enrichment.get("education"), list):
        education_items.extend(enrichment.get("education") or [])
    windows: List[Dict[str, Any]] = []
    for item in education_items:
        if not isinstance(item, dict):
            continue
        start = _shortlist_parse_date(item.get("start_date") or item.get("start_raw"))
        end = _shortlist_parse_date(item.get("end_date") or item.get("end_raw"))
        if not start or not end:
            continue
        windows.append(
            {
                "college": _shortlist_clean_text(item.get("college") or item.get("school") or item.get("title")),
                "degree": _shortlist_clean_text(item.get("degree") or item.get("subtitle")),
                "start": start,
                "end": end,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
            }
        )
    return sorted(windows, key=lambda item: item["start"])


def _shortlist_gap_analysis(profile: Dict[str, Any]) -> Dict[str, Any]:
    roles = _shortlist_role_windows(profile)
    education = _shortlist_education_windows(profile)
    gaps: List[Dict[str, Any]] = []
    for previous, current in zip(roles, roles[1:]):
        previous_end = previous.get("end")
        current_start = current.get("start")
        if not previous_end or not current_start:
            continue
        gap_months = _shortlist_months_between(previous_end, current_start)
        if gap_months < 12:
            continue
        overlapping_education = [
            {
                "college": item.get("college"),
                "degree": item.get("degree"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
            }
            for item in education
            if item.get("start") and item.get("end") and item["start"] <= current_start and item["end"] >= previous_end
        ]
        gaps.append(
            {
                "from_company": previous.get("company"),
                "to_company": current.get("company"),
                "start_date": previous_end.date().isoformat(),
                "end_date": current_start.date().isoformat(),
                "gap_months": gap_months,
                "gap_years": _shortlist_years(gap_months),
                "education_overlap": bool(overlapping_education),
                "education": overlapping_education,
            }
        )
    return {
        "has_gap_years": bool(gaps),
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def _shortlist_current_location(profile: Dict[str, Any]) -> Dict[str, str]:
    raw_location = _shortlist_clean_text(profile.get("location") or profile.get("city"))
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    raw_location = raw_location or _shortlist_clean_text(
        raw_fields.get("addressWithCountry")
        or raw_fields.get("address")
        or raw_fields.get("location")
    )
    parts = [part.strip() for part in raw_location.split(",") if part.strip()]
    city = _shortlist_clean_text(profile.get("city")) or (parts[0] if parts else "")
    country = parts[-1] if len(parts) >= 2 else ""
    state = parts[-2] if len(parts) >= 3 else ""
    return {
        "raw": raw_location,
        "city": city,
        "state": state,
        "country": country,
    }


def _shortlist_team_management(profile: Dict[str, Any]) -> Dict[str, Any]:
    enrichment = _profile_raw_enrichment(profile)
    claims = enrichment.get("profile_claims") if isinstance(enrichment.get("profile_claims"), dict) else {}
    max_people = (
        profile.get("max_people_managed")
        or claims.get("max_people_managed")
        or (profile.get("raw_fields") or {}).get("max_people_managed")
        or 0
    )
    years = (
        profile.get("years_team_management")
        or claims.get("years_team_management")
        or (profile.get("raw_fields") or {}).get("years_team_management")
        or 0
    )
    try:
        max_people = int(float(max_people or 0))
    except Exception:
        max_people = 0
    try:
        years = float(years or 0)
    except Exception:
        years = 0
    return {
        "max_people_managed": max_people,
        "years_team_management": years,
        "status": "explicit" if max_people or years else "unknown",
    }


def _shortlist_company_tenure(profile: Dict[str, Any]) -> Dict[str, Any]:
    roles = _shortlist_role_windows(profile)
    by_company: Dict[str, Dict[str, Any]] = {}
    for role in roles:
        company = role.get("company") or "Unknown"
        bucket = by_company.setdefault(
            _normalize_company_key(company) or company,
            {"company": company, "months": 0, "roles": []},
        )
        bucket["months"] += int(role.get("duration_months") or 0)
        bucket["roles"].append(
            {
                "title": role.get("title"),
                "start_date": role.get("start_date"),
                "end_date": role.get("end_date"),
                "duration_years": role.get("duration_years"),
            }
        )
    company_tenures = [
        {
            "company": item["company"],
            "months": item["months"],
            "years": _shortlist_years(item["months"]),
            "roles": item["roles"],
        }
        for item in by_company.values()
    ]
    completed = [item for item in company_tenures[:-1] if item.get("months")]
    average_completed = round(sum(item["months"] for item in completed) / len(completed), 1) if completed else 0
    return {
        "company_tenures": company_tenures,
        "average_completed_company_tenure_months": average_completed,
        "average_completed_company_tenure_years": _shortlist_years(int(average_completed)),
    }


def build_shortlist_intelligence_pack(profile: Dict[str, Any], criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Shortlist-only evidence pack: deterministic calculations plus full profile evidence."""
    profile_safe = {k: v for k, v in (profile or {}).items() if k != "embedding"}
    context = build_candidate_context(profile_safe)
    career_facts = compute_career_facts(context)
    enrichment = _profile_raw_enrichment(profile_safe)
    raw_fields = profile_safe.get("raw_fields") if isinstance(profile_safe.get("raw_fields"), dict) else {}
    profile_claims = enrichment.get("profile_claims") if isinstance(enrichment.get("profile_claims"), dict) else {}
    company_tenure = _shortlist_company_tenure(profile_safe)
    evidence_sections = {
        "headline": profile_safe.get("headline"),
        "about": profile_safe.get("about"),
        "skills": profile_safe.get("skills") or raw_fields.get("Skills"),
        "notes": profile_safe.get("notes"),
        "response": profile_safe.get("response"),
        "roles": profile_safe.get("roles") or enrichment.get("roles") or [],
        "education": profile_safe.get("education") or enrichment.get("education") or [],
        "imported_extra_fields": raw_fields.get("imported_extra_fields"),
        "raw_fields": raw_fields,
    }
    return {
        "candidate_id": profile_safe.get("id"),
        "candidate_name": profile_safe.get("name"),
        "current_location": _shortlist_current_location(profile_safe),
        "career_metrics": {
            "total_experience_years": profile_safe.get("total_experience_years") or career_facts.get("total_experience_years"),
            "total_experience_months": career_facts.get("total_experience_months"),
            "average_tenure_months": career_facts.get("average_tenure_months") or company_tenure.get("average_completed_company_tenure_months"),
            "average_tenure_years": (
                round(float(career_facts.get("average_tenure_months") or 0) / 12.0, 1)
                if career_facts.get("average_tenure_months")
                else company_tenure.get("average_completed_company_tenure_years")
            ),
            "current_job_months": career_facts.get("current_job_months"),
        },
        "company_tenure": company_tenure,
        "gap_analysis": _shortlist_gap_analysis(profile_safe),
        "team_management": _shortlist_team_management(profile_safe),
        "profile_claims": profile_claims,
        "full_profile_evidence": evidence_sections,
        "query_relevant_company_facts": (criteria or {}).get("_web_company_facts", {}),
        "evidence_policy": {
            "candidate_facts": "Use only profile/import/DB evidence. Missing profile facts are unknown.",
            "company_facts": "Use cached or live web facts only when source-backed.",
        },
    }


def _normalize_search_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _term_matches_text(term: str, text: str) -> bool:
    term_l = _normalize_search_text(term)
    if not term_l:
        return False
    if len(term_l) <= 3:
        return re.search(rf"\b{re.escape(term_l)}\b", text) is not None
    return term_l in text


def _iter_evidence_leaf_values(value: Any, prefix: str = "", depth: int = 0, max_depth: int = 5) -> List[Tuple[str, Any]]:
    if value is None or depth > max_depth:
        return []
    if isinstance(value, dict):
        leaves: List[Tuple[str, Any]] = []
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            leaves.extend(_iter_evidence_leaf_values(child, path, depth + 1, max_depth))
        return leaves
    if isinstance(value, list):
        leaves = []
        for idx, child in enumerate(value[:20]):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            leaves.extend(_iter_evidence_leaf_values(child, path, depth + 1, max_depth))
        return leaves
    return [(prefix, value)]


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

    def add(source: str, value: Any, *, role: Optional[Dict[str, Any]] = None, category: str = "candidate_fact", path: str = "") -> None:
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
                "category": category,
                "path": path,
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
        for path, value in _iter_evidence_leaf_values(raw_fields):
            add(f"uploaded fields.{path}", value, path=path)

    for idx, role in enumerate(profile.get("roles") or [], start=1):
        company_details = role.get("company_details") or {}
        add(f"role {idx} title", role.get("title"), role=role)
        add(f"role {idx} details", role.get("details"), role=role)
        add(f"role {idx} company", role.get("company"), role=role, category="relationship_scope")
        add(f"role {idx} dates", {"duration_years": role.get("duration_years"), "start_date": role.get("start_date"), "end_date": role.get("end_date")}, role=role, category="calculated_fact")
        add(f"role {idx} company details", company_details, role=role)

    for row in profile.get("schema_evidence_rows") or []:
        if not isinstance(row, dict):
            continue
        table = str(row.get("table") or "schema").strip()
        category = str(row.get("category") or "candidate_fact").strip()
        add(f"schema {table}", row.get("row") or {}, category=category)

    return chunks


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _classify_evidence_field(table: str, column: str) -> str:
    table_l = _normalize_search_text(table)
    column_l = _normalize_search_text(column)
    if table_l in {"companies"}:
        return "company_fact"
    if table_l in {
        "company_years",
        "experience_gaps",
        "education_gaps",
        "industry_gaps",
        "functional_experiences",
        "functional_experience_roles",
        "industry_experiences",
        "industry_experience_roles",
        "segment_experiences",
        "segment_experience_roles",
        "geography_experiences",
        "geography_experience_regions",
        "titles_held",
    }:
        return "calculated_fact"
    if column_l in {"notes", "feedback", "response", "reason", "rationale"}:
        return "recruiter_note"
    if table_l in {"recruitment_role_candidates", "candidate_uploads"} or column_l in {
        "role_id",
        "candidate_id",
        "source_master_candidate_id",
        "owner_user_id",
        "pool_source",
    }:
        return "relationship_scope"
    if any(token in column_l for token in ("campaign", "status", "sent_count", "created_at", "updated_at", "archived_at", "oauth", "token")):
        return "operational_metadata"
    return "candidate_fact"


def _collect_raw_field_keys(value: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> List[str]:
    if depth > max_depth or value is None:
        return []
    keys: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            keys.append(path)
            keys.extend(_collect_raw_field_keys(child, path, depth + 1, max_depth))
    elif isinstance(value, list):
        for child in value[:5]:
            keys.extend(_collect_raw_field_keys(child, prefix, depth + 1, max_depth))
    return keys


def build_db_evidence_catalog(*, force_refresh: bool = False, profiles: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Discover shortlist evidence fields from live schema plus sampled raw_fields JSON keys."""
    global _EVIDENCE_CATALOG_CACHE
    if _EVIDENCE_CATALOG_CACHE and not force_refresh:
        return _EVIDENCE_CATALOG_CACHE

    tables: Dict[str, Dict[str, Any]] = {}
    raw_field_keys: Dict[str, Dict[str, Any]] = {}
    source = "fallback"

    conn = get_db_connection(validate=False, register_pgvector=False)
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                    """
                )
                for table, column, data_type in cur.fetchall():
                    table_entry = tables.setdefault(str(table), {"columns": [], "candidate_related": False})
                    category = _classify_evidence_field(str(table), str(column))
                    table_entry["columns"].append(
                        {
                            "name": str(column),
                            "type": str(data_type),
                            "category": category,
                        }
                    )
                    if str(column) == "candidate_id":
                        table_entry["candidate_related"] = True

                cur.execute(
                    """
                    SELECT raw_fields
                    FROM candidates
                    WHERE raw_fields IS NOT NULL
                    """
                )
                for (raw_fields,) in cur.fetchall():
                    if isinstance(raw_fields, str):
                        raw_fields = safe_json_loads(raw_fields, {})
                    for key in _collect_raw_field_keys(raw_fields if isinstance(raw_fields, dict) else {}):
                        raw_field_keys.setdefault(
                            key,
                            {
                                "path": key,
                                "category": _classify_evidence_field("candidates.raw_fields", key),
                            },
                        )
                source = "database"
        except Exception:
            logger.warning("Could not build DB evidence catalog from information_schema", exc_info=True)
        finally:
            return_db_connection(conn)

    if not tables:
        fallback_profiles = profiles or list(PROFILES_BY_ID.values())
        core_fields = [
            "id", "name", "linkedin", "location", "city", "headline", "about",
            "total_experience_years", "max_people_managed", "avg_years_in_company",
            "candidate_services", "extracted_industry", "raw_fields", "response",
            "notes", "status", "owner_user_id", "pool_source", "normalized_linkedin",
            "is_archived",
        ]
        tables["candidates"] = {
            "candidate_related": True,
            "columns": [
                {"name": field, "type": "unknown", "category": _classify_evidence_field("candidates", field)}
                for field in core_fields
            ],
        }
        tables["roles"] = {
            "candidate_related": True,
            "columns": [
                {"name": field, "type": "unknown", "category": _classify_evidence_field("roles", field)}
                for field in ("candidate_id", "company", "title", "details", "duration_years", "start_date", "end_date", "company_details")
            ],
        }
        for profile in fallback_profiles:
            raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
            for key in _collect_raw_field_keys(raw_fields):
                raw_field_keys.setdefault(
                    key,
                    {
                        "path": key,
                        "category": _classify_evidence_field("candidates.raw_fields", key),
                    },
                )

    version_seed = json.dumps({"tables": tables, "raw_field_keys": sorted(raw_field_keys)}, sort_keys=True, default=str)
    catalog = {
        "version": hashlib.md5(version_seed.encode("utf-8")).hexdigest(),
        "source": source,
        "tables": tables,
        "raw_field_keys": sorted(raw_field_keys.values(), key=lambda item: item["path"]),
        "candidate_fact_policy": "Candidate facts can come from candidate/profile/import/calculated DB fields only.",
        "company_fact_policy": "Company facts can come from company tables or sourced web company intelligence.",
    }
    _EVIDENCE_CATALOG_CACHE = catalog
    return catalog


def compact_evidence_catalog_for_prompt(
    catalog: Dict[str, Any],
    *,
    max_tables: Optional[int] = None,
    max_raw_keys: Optional[int] = None,
    max_columns_per_table: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the executable schema supplied to the shortlist planner.

    The catalog is cached, so retaining the complete schema here does not add a
    database round-trip per query. Limits remain available for callers that need
    a deliberately reduced diagnostic view.
    """
    tables = []
    table_items = sorted((catalog.get("tables") or {}).items())
    if max_tables is not None:
        table_items = table_items[:max_tables]
    for table, info in table_items:
        columns = info.get("columns") or []
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        tables.append(
            {
                "table": table,
                "candidate_related": bool(info.get("candidate_related")),
                "columns": [
                    {
                        "name": column.get("name"),
                        "category": column.get("category"),
                    }
                    for column in columns
                ],
            }
        )
    raw_field_keys = catalog.get("raw_field_keys", [])
    if max_raw_keys is not None:
        raw_field_keys = raw_field_keys[:max_raw_keys]
    return {
        "version": catalog.get("version"),
        "source": catalog.get("source"),
        "tables": tables,
        "raw_field_keys": raw_field_keys,
        "candidate_fact_policy": catalog.get("candidate_fact_policy"),
        "company_fact_policy": catalog.get("company_fact_policy"),
    }


def attach_schema_evidence_to_profiles(profiles: List[Dict[str, Any]], catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attach rows from candidate-related schema tables so shortlist evidence can use the full DB shape."""
    ids = sorted({int(profile.get("id")) for profile in profiles if profile.get("id") is not None})
    if not ids:
        return profiles

    candidate_rows: Dict[int, List[Dict[str, Any]]] = {pid: [] for pid in ids}
    tables = catalog.get("tables") or {}
    candidate_tables = [
        (table, info)
        for table, info in tables.items()
        if info.get("candidate_related") and table not in {"candidates", "roles", "ai_column_cells", "calls"}
    ]
    if not candidate_tables:
        return profiles

    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        return profiles
    try:
        with conn.cursor() as cur:
            for table, info in candidate_tables:
                columns = [str(column.get("name")) for column in info.get("columns") or [] if column.get("name")]
                if "candidate_id" not in columns:
                    continue
                try:
                    cur.execute(
                        f"SELECT {', '.join(_quote_ident(column) for column in columns)} "
                        f"FROM {_quote_ident(table)} WHERE candidate_id = ANY(%s)",
                        (ids,),
                    )
                    description = [desc[0] for desc in cur.description]
                    for row in cur.fetchall():
                        row_dict = {
                            key: value
                            for key, value in zip(description, row)
                            if value not in (None, "", [], {})
                        }
                        candidate_id = row_dict.get("candidate_id")
                        if candidate_id in candidate_rows and row_dict:
                            candidate_rows[int(candidate_id)].append(
                                {
                                    "table": table,
                                    "category": _classify_evidence_field(table, ""),
                                    "row": row_dict,
                                }
                            )
                except Exception:
                    logger.debug("Skipping schema evidence table %s", table, exc_info=True)
    finally:
        return_db_connection(conn)

    for profile in profiles:
        profile_id = profile.get("id")
        profile["schema_evidence_rows"] = candidate_rows.get(int(profile_id), []) if profile_id is not None else []
    return profiles


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


def _region_to_countries() -> Dict[str, List[str]]:
    regions: Dict[str, List[str]] = {}
    for country, region in GEOGRAPHY_COUNTRY_TO_REGION_MAP.items():
        regions.setdefault(_normalize_search_text(region), []).append(country)
    return {region: sorted(set(countries)) for region, countries in regions.items()}


def _geography_match_terms(value: str, criterion: Any = None) -> List[str]:
    value_l = _normalize_search_text(value)
    if not value_l:
        return []
    terms = set(_criterion_match_terms(value_l, "required_geographies", criterion or {}))
    regions = _region_to_countries()
    if value_l in regions:
        terms.add(value_l)
        terms.update(regions[value_l])
    mapped_region = GEOGRAPHY_COUNTRY_TO_REGION_MAP.get(value_l)
    if mapped_region:
        terms.add(mapped_region)
        terms.add(value_l)
    for item in _criteria_objects(criterion):
        for key in ("expanded_countries", "countries", "regions", "aliases", "expanded_terms"):
            raw = item.get(key)
            if isinstance(raw, list):
                terms.update(_normalize_search_text(term) for term in raw if str(term or "").strip())
    return sorted(term for term in terms if term)


def _role_company_geo_text(role: Dict[str, Any]) -> str:
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
            "customer_presence",
        )
        if company_details.get(key)
    }
    return " ".join(_flatten_value_for_evidence(allowed_company_geo, max_items=60)).lower()


def _role_geography_text_for_profile(profile: Dict[str, Any], role: Dict[str, Any]) -> str:
    base_text = _role_geography_text(role)
    company_geo_text = _role_company_geo_text(role)
    candidate_location_text = _profile_location_text(profile)
    inferred_same_location = []
    if company_geo_text and candidate_location_text:
        for country, region in GEOGRAPHY_COUNTRY_TO_REGION_MAP.items():
            if (
                _term_matches_text(country, candidate_location_text)
                and _term_matches_text(country, company_geo_text)
            ):
                inferred_same_location.extend([country, region, "candidate and employer same location"])
    return " ".join(part for part in [base_text, " ".join(inferred_same_location)] if part).lower()


def _profile_geography_experience_text(profile: Dict[str, Any]) -> str:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    likely_claims: Dict[str, Any] = {}
    for key, value in raw_fields.items():
        key_l = _normalize_search_text(key)
        if any(token in key_l for token in ("address", "current location", "base location", "located", "city")):
            continue
        if any(token in key_l for token in ("geograph", "market", "region", "country", "territory", "coverage", "summary", "notes", "profile")):
            likely_claims[key] = value
    return " ".join(
        _flatten_value_for_evidence(
            {
                "headline": profile.get("headline"),
                "about": profile.get("about"),
                "candidate_services": profile.get("candidate_services"),
                "extracted_industry": profile.get("extracted_industry"),
                "uploaded_geography_claims": likely_claims,
            },
            max_items=120,
        )
    ).lower()


def _company_criteria_items(criterion: Any) -> List[Dict[str, Any]]:
    if not criterion:
        return []
    if isinstance(criterion, dict):
        raw_items = criterion.get("values") if isinstance(criterion.get("values"), list) else [criterion]
    elif isinstance(criterion, list):
        raw_items = criterion
    else:
        raw_items = [criterion]
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        if isinstance(raw, dict):
            company = str(raw.get("company") or raw.get("value") or raw.get("name") or raw.get("target") or "").strip()
            if company:
                item = dict(raw)
                item["company"] = company
                items.append(item)
        elif str(raw or "").strip():
            items.append({"company": str(raw).strip()})
    return items


def _company_scope_current_only(item: Dict[str, Any], default_scope: str = "any_employer") -> bool:
    scope = _normalize_search_text(item.get("employment_scope") or item.get("scope") or default_scope)
    return scope in {"current", "current_employer", "currently_at", "working_at", "working_for"}


def _roles_for_employment_scope(
    profile: Dict[str, Any],
    criterion: Any,
    default_scope: str = "any_employer",
) -> List[Dict[str, Any]]:
    criterion_obj = criterion if isinstance(criterion, dict) else {}
    if _company_scope_current_only(criterion_obj, default_scope):
        return _current_roles(profile)
    return profile.get("roles") or []


def _current_roles(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    roles = profile.get("roles") or []
    if not roles:
        return []

    explicit_current = [
        role for role in roles
        if _normalize_search_text(role.get("end_date") or role.get("end")) in {"present", "current", "now", "ongoing"}
    ]
    if explicit_current:
        return explicit_current

    # An empty end date plus a real start date is the standard representation
    # for an active role in most imported profiles. Prefer the newest open role;
    # older incomplete records must not masquerade as concurrent current jobs.
    open_ended = [
        role for role in roles
        if not str(role.get("end_date") or role.get("end") or "").strip()
        and _shortlist_parse_date(role.get("start_date") or role.get("start"))
    ]
    if open_ended:
        latest_open_start = max(
            _shortlist_parse_date(role.get("start_date") or role.get("start")) or datetime.min
            for role in open_ended
        )
        return [
            role for role in open_ended
            if (_shortlist_parse_date(role.get("start_date") or role.get("start")) or datetime.min) == latest_open_start
        ]

    # Some importers materialize "Present" as the import date. In that shape,
    # newest start date is a safer current-role signal than DB insertion order.
    dated_roles = [
        role for role in roles
        if _shortlist_parse_date(role.get("start_date") or role.get("start"))
    ]
    if dated_roles:
        latest = max(
            dated_roles,
            key=lambda role: _shortlist_parse_date(role.get("start_date") or role.get("start")) or datetime.min,
        )
        return [latest]

    return roles[:1]


def _company_match_terms(item: Dict[str, Any]) -> List[str]:
    company = str(item.get("company") or item.get("value") or "").strip()
    terms = {company}
    for key in ("aliases", "company_aliases", "expanded_terms"):
        raw = item.get(key)
        if isinstance(raw, list):
            terms.update(str(term).strip() for term in raw if str(term or "").strip())
        elif isinstance(raw, str):
            terms.update(term.strip() for term in re.split(r"[,;|]", raw) if term.strip())
    return sorted(term for term in terms if term)


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
        if any(token in key_l for token in ("address", "current location", "base location", "located", "city")):
            continue
        if any(token in key_l for token in ("geograph", "market", "region", "country", "territory", "coverage", "summary", "notes", "profile")):
            likely_claims[key] = value
    return " ".join(_flatten_value_for_evidence(likely_claims, max_items=60)).lower()


def _role_geography_text(role: Dict[str, Any]) -> str:
    # Keep employer headquarters/offices out of direct market-experience
    # evidence. _role_geography_text_for_profile may use company geography only
    # when it corroborates the candidate/role location.
    return " ".join(
        _flatten_value_for_evidence(
            {
                "title": role.get("title"),
                "details": role.get("details"),
                "location": role.get("location"),
                "city": role.get("city"),
                "source_location": role.get("source_location"),
                "company_location": role.get("company_location"),
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


def _validate_company_names_against_db(company_names: List[str], *, exclude: Optional[str] = None) -> List[str]:
    lookup = {_normalize_company_key(name): name for name in ALL_COMPANY_NAMES if str(name or "").strip()}
    exclude_key = _normalize_company_key(exclude)
    validated: List[str] = []
    seen = set()
    for company in company_names:
        company_text = str(company or "").strip()
        if not company_text:
            continue
        company_key = _normalize_company_key(company_text)
        if not company_key or company_key == exclude_key:
            continue
        matched_name = lookup.get(company_key)
        if not matched_name:
            matched_name = next(
                (
                    db_name for db_key, db_name in lookup.items()
                    if db_key != exclude_key and _company_matches(db_name, company_text)
                ),
                "",
            )
        if not matched_name:
            continue
        matched_key = _normalize_company_key(matched_name)
        if matched_key in seen:
            continue
        seen.add(matched_key)
        validated.append(matched_name)
    return validated


def _shortlist_company_cache_key(value: Any) -> str:
    return _normalize_company_key(value)


def _load_shortlist_company_fact_cache() -> Dict[str, Any]:
    try:
        if not SHORTLIST_COMPANY_FACT_CACHE_PATH.exists():
            return {}
        with SHORTLIST_COMPANY_FACT_CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Could not load shortlist company fact cache", exc_info=True)
        return {}


def _save_shortlist_company_fact_cache(cache: Dict[str, Any]) -> None:
    try:
        SHORTLIST_COMPANY_FACT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SHORTLIST_COMPANY_FACT_CACHE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.warning("Could not save shortlist company fact cache", exc_info=True)


def _company_fact_targets(criteria: Dict[str, Any]) -> List[str]:
    targets: List[str] = []
    for item in _criteria_objects(criteria.get("competitor_of")) + _criteria_objects(criteria.get("competitors_of")):
        target = str(item.get("target") or item.get("value") or item.get("name") or "").strip()
        if target:
            targets.append(target)
    for item in _criteria_objects(criteria.get("hiring_company")):
        target = str(item.get("company") or item.get("target") or item.get("value") or item.get("name") or "").strip()
        if target:
            targets.append(target)
    seen = set()
    out = []
    for target in targets:
        key = _shortlist_company_cache_key(target)
        if key and key not in seen:
            seen.add(key)
            out.append(target)
    return out


def _cached_company_facts_for_criteria(criteria: Dict[str, Any]) -> Dict[str, Any]:
    cache = _load_shortlist_company_fact_cache()
    competitors: List[Dict[str, Any]] = []
    company_profiles: List[Dict[str, Any]] = []
    similar_companies: List[Dict[str, Any]] = []
    for target in _company_fact_targets(criteria):
        cached = cache.get(_shortlist_company_cache_key(target))
        if not isinstance(cached, dict):
            continue
        target_name = cached.get("target") or target
        comps = cached.get("competitors") or []
        if comps:
            competitors.append({
                "target": target_name,
                "companies": comps,
                "sources": cached.get("sources") or [],
                "product_service": cached.get("product_service"),
                "customer_segment": cached.get("customer_segment"),
                "customer_presence": cached.get("customer_presence"),
            })
        similar = cached.get("similar_companies") or []
        if similar:
            similar_companies.append({
                "target": target_name,
                "companies": similar,
                "sources": cached.get("sources") or [],
            })
        company_profiles.append(cached)
    out: Dict[str, Any] = {}
    if competitors:
        out["competitors"] = competitors
    if similar_companies:
        out["similar_companies"] = similar_companies
    if company_profiles:
        out["company_profiles"] = company_profiles
    return out


def _cache_company_facts_from_structured(criteria: Dict[str, Any], structured: Dict[str, Any]) -> None:
    if not isinstance(structured, dict):
        return
    cache = _load_shortlist_company_fact_cache()
    touched = False
    profiles_by_key: Dict[str, Dict[str, Any]] = {}
    for item in structured.get("company_profiles") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or item.get("company") or item.get("name") or "").strip()
        key = _shortlist_company_cache_key(target)
        if key:
            profiles_by_key[key] = dict(item)

    for item in structured.get("competitors") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or item.get("company") or item.get("name") or "").strip()
        key = _shortlist_company_cache_key(target)
        if not key:
            continue
        companies = item.get("companies") or item.get("competitors") or []
        if isinstance(companies, str):
            companies = re.split(r"[,;|]", companies)
        companies = [str(company).strip() for company in companies if str(company or "").strip()][:50]
        cached = cache.get(key) if isinstance(cache.get(key), dict) else {}
        profile = profiles_by_key.get(key, {})
        cache[key] = {
            **cached,
            **profile,
            "target": target,
            "competitors": companies,
            "similar_companies": item.get("similar_companies") or cached.get("similar_companies") or [],
            "sources": item.get("sources") or profile.get("sources") or cached.get("sources") or [],
            "last_verified_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        touched = True

    for item in structured.get("similar_companies") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or item.get("company") or item.get("name") or "").strip()
        key = _shortlist_company_cache_key(target)
        if not key:
            continue
        companies = item.get("companies") or item.get("similar") or []
        if isinstance(companies, str):
            companies = re.split(r"[,;|]", companies)
        cached = cache.get(key) if isinstance(cache.get(key), dict) else {"target": target}
        cached["similar_companies"] = [str(company).strip() for company in companies if str(company or "").strip()][:50]
        cached["sources"] = item.get("sources") or cached.get("sources") or []
        cached["last_verified_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cache[key] = cached
        touched = True

    if touched:
        _save_shortlist_company_fact_cache(cache)


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


def _web_company_profile_text(company_name: str, criteria: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    for item in _web_company_fact_items(criteria, "company_profiles"):
        if not isinstance(item, dict):
            continue
        item_company = str(item.get("company") or item.get("target") or item.get("name") or "").strip()
        if not _company_matches(company_name, item_company):
            continue
        text = " ".join(
            _flatten_value_for_evidence(
                {
                    "product_service": item.get("product_service"),
                    "industry": item.get("industry"),
                    "customer_segment": item.get("customer_segment"),
                    "business_model": item.get("business_model"),
                    "culture_type": item.get("culture_type"),
                    "headquarters": item.get("headquarters"),
                    "funding_stage": item.get("funding_stage") or item.get("stage") or item.get("status"),
                },
                max_items=80,
            )
        ).lower()
        return text, item
    return "", None


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


def _score_hiring_company_relevance(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not criteria.get("hiring_company"):
        return 0.0, 0.0, [], [], []

    targets = _company_fact_targets({"hiring_company": criteria.get("hiring_company")})
    if not targets:
        return 0.0, 0.0, [], [], []

    web_facts = criteria.get("_web_company_facts") if isinstance(criteria.get("_web_company_facts"), dict) else {}
    relevant_companies: List[str] = []
    for fact_key in ("competitors", "similar_companies"):
        for item in web_facts.get(fact_key) or []:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip()
            if target and not any(_company_matches(target, wanted) for wanted in targets):
                continue
            relevant_companies.extend(str(company) for company in (item.get("companies") or []) if str(company or "").strip())

    if not relevant_companies:
        return 1.2, 0.0, [], [{
            "criterion": "Hiring company intelligence",
            "value": ", ".join(targets),
            "source": "web required",
            "snippet": "No cached or sourced competitor/similar-company list was available yet.",
        }], []

    roles = profile.get("roles") or []
    for role in roles[:6]:
        role_company = role.get("company") or ""
        matched_company = next((company for company in relevant_companies if _company_matches(role_company, company)), "")
        if not matched_company:
            continue
        return 1.2, 1.2, [{
            "criterion": "Hiring company relevance",
            "value": f"{role_company} matched competitor/similar company {matched_company}",
        }], [{
            "criterion": "Hiring company relevance",
            "value": ", ".join(targets),
            "source": "employer history matched web company facts",
            "snippet": f"{role.get('title') or 'Role'} at {role_company} is relevant to {', '.join(targets)}.",
        }], [role]

    return 1.2, 0.0, [], [], []


def score_candidate_against_criteria(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    profile_copy = copy.deepcopy({k: v for k, v in profile.items() if k != "embedding"})
    shortlist_intelligence = build_shortlist_intelligence_pack(profile_copy, criteria)
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

    relevance_weight, relevance_earned, relevance_matched, relevance_evidence, relevance_roles = _score_hiring_company_relevance(profile_copy, criteria)
    total_weight += relevance_weight
    earned_weight += relevance_earned
    matched_criteria.extend(relevance_matched)
    evidence_log.extend(relevance_evidence)
    contributing_roles.extend(relevance_roles)

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
    profile_copy["shortlist_intelligence"] = shortlist_intelligence
    profile_copy["contributing_roles_details"] = {"roles": role_details}
    return profile_copy


async def filter_candidates_by_criteria(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("Applying strict shortlist filters to %s profiles...", len(profiles))
    matching_candidates: List[Dict[str, Any]] = []

    for profile in profiles:
        scored = _strict_shortlist_score_candidate(profile, criteria)
        if scored:
            matching_candidates.append(scored)

    matching_candidates = _sort_strict_shortlist_candidates(matching_candidates, criteria)
    top_n = criteria.get("top_n")
    if top_n not in (None, 0):
        try:
            matching_candidates = matching_candidates[: int(top_n)]
        except Exception:
            pass

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
        evidence_id = item.get("id") or "evidence"
        snippet = item.get("snippet") or item.get("value") or ""
        if snippet:
            snippets.append(f"{evidence_id} {source}: {snippet}")
    return "Verified from structured evidence: " + "; ".join(snippets[:4])


def _audit_evidence_id_set(profile: Dict[str, Any]) -> set:
    return {str(item.get("id")) for item in (profile.get("evidence_log") or []) if isinstance(item, dict) and item.get("id")}


def _extract_audit_evidence_ids(payload: Dict[str, Any]) -> List[str]:
    raw = (
        payload.get("evidence_ids")
        or payload.get("citations")
        or payload.get("evidence")
        or []
    )
    if isinstance(raw, str):
        raw = re.split(r"[,;|\s]+", raw)
    ids: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            value = item.get("id") or item.get("evidence_id") or item.get("citation")
        else:
            value = item
        text = str(value or "").strip()
        if text:
            ids.append(text)
    return ids


def _audit_output_is_evidence_valid(profile: Dict[str, Any], payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    valid_ids = _audit_evidence_id_set(profile)
    cited_ids = _extract_audit_evidence_ids(payload)
    if not valid_ids or not cited_ids:
        return False
    if any(evidence_id not in valid_ids for evidence_id in cited_ids):
        return False
    text = " ".join(str(payload.get(key) or "") for key in ("answer", "reasoning", "auditor_reasoning"))
    if not text.strip():
        return False
    # Require the visible explanation to carry at least one cited evidence ID.
    return any(evidence_id in text for evidence_id in cited_ids)


def _fallback_audit_payload_from_evidence(profile: Dict[str, Any]) -> Dict[str, Any]:
    evidence_ids = [str(item.get("id")) for item in (profile.get("evidence_log") or []) if isinstance(item, dict) and item.get("id")]
    reasoning = _fallback_reasoning_from_evidence(profile)
    return {
        "final_status": "verified_match" if evidence_ids else "not_verified",
        "answer": reasoning,
        "reasoning": reasoning,
        "evidence_ids": evidence_ids[:6],
        "confidence": "high" if evidence_ids else "low",
        "match_score": profile.get("match_score"),
    }


def _shortlist_candidate_key(profile: Dict[str, Any]) -> str:
    value = profile.get("id") or profile.get("linkedin") or profile.get("name") or id(profile)
    return str(value)


def _prepare_shortlist_visible_candidate(
    profile: Dict[str, Any],
    *,
    status: str,
    verification_pending: bool = False,
    verification_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a frontend-safe candidate row with visible shortlist status metadata."""
    visible = copy.deepcopy({k: v for k, v in (profile or {}).items() if k != "embedding"})
    visible["shortlist_status"] = status
    visible["is_verified_match"] = status == "verified_match"
    visible["verification_pending"] = verification_pending
    if verification_error:
        visible["verification_error"] = str(verification_error)[:500]

    visible.setdefault("matched_criteria", [])
    visible.setdefault("missing_criteria", [])
    visible.setdefault("sources", [])
    visible.setdefault("confidence", "medium" if status in {"verified_match", "potential"} else "low")

    if not visible.get("answer"):
        if status == "verified_match":
            visible["answer"] = visible.get("reasoning") or "Verified match based on AI review."
        elif verification_pending:
            visible["answer"] = ""
        elif status == "verification_error":
            visible["answer"] = "AI review could not complete for this profile. Retry the search for a verified answer."
        else:
            visible["answer"] = "Reviewed by AI; this profile did not fully satisfy the query requirements."

    return visible


def _sort_shortlist_visible_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    status_rank = {
        "verified_match": 0,
        "potential": 1,
        "not_verified": 2,
        "verification_error": 3,
    }
    review_rank = {
        "audited": 0,
        "analyst_reviewed": 1,
        "local_evidence_only": 2,
        "not_reviewed": 3,
        "verification_error": 4,
    }
    return sorted(
        candidates,
        key=lambda x: (
            status_rank.get(x.get("shortlist_status"), 9),
            review_rank.get(x.get("review_stage") or ("audited" if x.get("auditor_status") in {"passed", "downgraded", "needs_review"} else "not_reviewed"), 9),
            -(float(x.get("match_score") or 0)),
            -(float(x.get("total_experience_years") or 0)),
        ),
    )


def _strict_decision_is_match(structured: Dict[str, Any]) -> bool:
    decision = _normalize_search_text(structured.get("decision") or structured.get("match") or structured.get("answer"))
    return decision in {"match", "yes", "true", "qualified", "pass"}


def _normalize_shortlist_status(value: Any) -> str:
    status = _normalize_search_text(value)
    status = status.replace(" ", "_").replace("-", "_")
    aliases = {
        "match": "verified_match",
        "yes": "verified_match",
        "qualified": "verified_match",
        "pass": "verified_match",
        "maybe": "potential",
        "borderline": "potential",
        "manual_review": "potential",
        "review": "potential",
        "unknown": "potential",
        "no": "not_verified",
        "no_match": "not_verified",
        "reject": "not_verified",
        "rejected": "not_verified",
        "fail": "not_verified",
        "failed": "not_verified",
    }
    normalized = aliases.get(status, status)
    if normalized in {"verified_match", "potential", "not_verified", "verification_error"}:
        return normalized
    return ""


def _analyst_status_from_structured(structured: Dict[str, Any], missing_items: List[Any], score: float) -> str:
    explicit = _normalize_shortlist_status(structured.get("shortlist_status") or structured.get("status"))
    if explicit:
        return explicit
    if not missing_items and _strict_decision_is_match(structured) and score >= float(os.getenv("SCREENING_VERIFIED_MATCH_THRESHOLD", "70")):
        return "verified_match"
    if score >= float(os.getenv("SCREENING_POTENTIAL_MATCH_THRESHOLD", "45")) or structured.get("matched_criteria"):
        return "potential"
    return "not_verified"


def _uses_dynamic_ai_matching(criteria: Dict[str, Any]) -> bool:
    return any(
        criteria.get(key)
        for key in (
            "hiring_company",
            "competitor_of",
            "min_function_years",
            "funding_stage_min",
            "required_industries",
            "required_functions",
            "required_segments",
            "required_geographies",
            "required_company_details",
            "required_culture_type",
            "min_people_managed",
            "min_team_management_years",
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
        if fact_key in {"competitors", "similar_companies"}:
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
    if criteria.get("hiring_company"):
        company_names.extend(_dynamic_company_fact_names(criteria, "competitors"))
        company_names.extend(_dynamic_company_fact_names(criteria, "similar_companies"))
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
        )

    for profile in company_matched + high_tenure + [profile for _score, profile in ranked] + profiles:
        pid = profile.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        merged.append(profile)

    return merged


def _needs_company_fact_web_enrichment(criteria: Dict[str, Any]) -> bool:
    return bool(criteria.get("competitor_of") or criteria.get("competitors_of") or criteria.get("hiring_company"))


def _needs_candidate_company_fact_web_enrichment(criteria: Dict[str, Any]) -> bool:
    return bool(
        criteria.get("funding_stage_min")
        or criteria.get("required_geographies")
        or criteria.get("required_company_details")
        or criteria.get("required_industries")
        or criteria.get("required_segments")
        or criteria.get("required_culture_type")
    )


def _merge_web_company_facts(criteria: Dict[str, Any], structured: Dict[str, Any]) -> Dict[str, Any]:
    enriched = copy.deepcopy(criteria)
    web_facts = enriched.get("_web_company_facts") if isinstance(enriched.get("_web_company_facts"), dict) else {}

    competitors = structured.get("competitors") if isinstance(structured, dict) else None
    if isinstance(competitors, list):
        existing_competitors = web_facts.get("competitors") if isinstance(web_facts.get("competitors"), list) else []
        normalized_competitors = list(existing_competitors)
        seen_targets = {_shortlist_company_cache_key(item.get("target")) for item in existing_competitors if isinstance(item, dict)}
        for item in competitors:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip()
            if _shortlist_company_cache_key(target) in seen_targets:
                continue
            companies = item.get("companies") or item.get("competitors") or []
            if isinstance(companies, str):
                companies = re.split(r"[,;|]", companies)
            companies = [str(company).strip() for company in companies if str(company or "").strip()][:50]
            sources = item.get("sources") if isinstance(item.get("sources"), list) else []
            next_item = dict(item)
            next_item.update({
                "target": target,
                "companies": companies,
                "sources": sources,
            })
            normalized_competitors.append(next_item)
            seen_targets.add(_shortlist_company_cache_key(target))
        web_facts["competitors"] = normalized_competitors

    for key in ("funding", "geography", "company_profiles", "similar_companies"):
        values = structured.get(key) if isinstance(structured, dict) else None
        if isinstance(values, list):
            existing = web_facts.get(key) if isinstance(web_facts.get(key), list) else []
            web_facts[key] = [*existing, *[item for item in values if isinstance(item, dict)]]

    enriched["_web_company_facts"] = web_facts
    return enriched


async def enrich_criteria_with_company_web_facts(
    original_query: str,
    criteria: Dict[str, Any],
    tracker: TokenCostTracker,
) -> Dict[str, Any]:
    if not _needs_company_fact_web_enrichment(criteria) or not SCREENING_WEB_SEARCH_DEFAULT:
        return criteria

    cached_structured = _cached_company_facts_for_criteria(criteria)
    if cached_structured:
        criteria = _merge_web_company_facts(criteria, cached_structured)
        cached_targets = {
            _shortlist_company_cache_key(item.get("target"))
            for item in cached_structured.get("competitors", [])
            if isinstance(item, dict)
        }
        requested_targets = {_shortlist_company_cache_key(target) for target in _company_fact_targets(criteria)}
        if requested_targets and requested_targets.issubset(cached_targets):
            return criteria

    system_prompt = (
        "You are a company research assistant for recruiting search. Resolve only company-level facts needed by the query. "
        "Use web evidence when available. Do not infer or create candidate career facts. "
        "Return valid JSON only with keys: competitors, similar_companies, company_profiles, funding, geography, notes. "
        "competitors must be a list of objects with target, companies, sources, product_service, customer_segment, customer_presence. "
        "For each hiring_company or competitor_of target, return up to the top 50 closest competitors with similar product/service and buyer segment. "
        "similar_companies must list non-direct competitors with similar product/service, segment, geography, funding, or culture when sources support it. "
        "company_profiles must include target/company, product_service, customer_segment, customer_presence, funding_stage, revenue, culture_type, headquarters, sources. "
        "funding must be a list of objects with company, stage/status, sources. "
        "geography must be a list of objects with company, offices/operations/headquarters/geographies, sources. "
        "Every source must include a non-empty url, title, and note. Omit facts that do not have reliable source URLs."
    )
    user_prompt = (
        f"Recruiting query:\n{original_query}\n\n"
        f"Extracted structured criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Resolve competitor_of and hiring_company targets dynamically. Prefer direct competitor/category pages or reputable company/research sources. "
        "Do not use a hardcoded taxonomy. Company facts may come from the web; candidate facts must not. Return JSON only."
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

    if isinstance(structured, dict):
        _cache_company_facts_from_structured(criteria, structured)
    return _merge_web_company_facts(criteria, structured if isinstance(structured, dict) else {})


def _candidate_company_names_for_web(profiles: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[str]:
    counts: Dict[str, int] = {}
    scoped_criteria = [
        criteria.get(key)
        for key in EMPLOYMENT_SCOPED_CRITERIA_KEYS
        if criteria.get(key)
    ]
    current_only = bool(scoped_criteria) and all(
        _company_scope_current_only(item if isinstance(item, dict) else {}, "any_employer")
        for item in scoped_criteria
    )
    for profile in profiles:
        roles = profile.get("roles") or []
        role_iter = _current_roles(profile) if current_only else roles[:5]
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
        "Return valid JSON only with keys: funding, geography, company_profiles, notes. "
        "funding: list objects with company, stage/status, sources. "
        "geography: list objects with company, offices/operations/headquarters/geographies, sources. "
        "company_profiles: list objects with company, product_service, industry, customer_segment, business_model, culture_type, headquarters, funding_stage, sources. "
        "Every source must include a non-empty url, title, and note. Omit facts that do not have reliable source URLs. "
        "Do not use customer presence, subsidiaries, revenue share, product availability, or customer examples as geography evidence."
    )
    user_prompt = (
        f"Recruiting query:\n{original_query}\n\n"
        f"Extracted structured criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate employer list to verify:\n{json.dumps(company_names, ensure_ascii=False, indent=2)}\n\n"
        "For funding_stage_min, verify whether listed companies meet the threshold. "
        "For geography criteria, verify only offices/operations/headquarters in the requested country/region. "
        "For industry/company-detail/segment/culture criteria, verify source-backed product, category, segment, business model, culture, and funding facts. "
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


def build_fallback_search_concept_pack(original_query: str, criteria: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Local fallback when the criteria LLM does not return a concept pack."""
    concepts: List[Dict[str, Any]] = []

    def add_concept(kind: str, name: str, terms: List[str], evidence_type: str) -> None:
        clean_terms = []
        seen = set()
        for term in terms:
            term_s = str(term or "").strip()
            key = _normalize_search_text(term_s)
            if term_s and key and key not in seen:
                clean_terms.append(term_s)
                seen.add(key)
        if clean_terms:
            concepts.append(
                {
                    "kind": kind,
                    "name": name,
                    "meaning": f"Evidence for {name} in the recruiter query.",
                    "aliases": clean_terms,
                    "positive_signals": clean_terms,
                    "negative_signals": [],
                    "required_evidence_type": evidence_type,
                }
            )

    for item in _min_function_year_items(criteria):
        function = str(item.get("function") or "").strip()
        add_concept(
            "function",
            function,
            [function, *(item.get("aliases") or item.get("expanded_terms") or [])],
            "role/profile evidence with dated tenure or explicit years claim",
        )

    for key, kind, evidence_type in (
        ("required_functions", "function", "role/profile evidence"),
        ("required_segments", "segment", "customer segment evidence"),
        ("required_industries", "industry", "company/product/market evidence"),
        ("required_company_details", "company_detail", "company fact evidence"),
        ("required_culture_type", "culture", "company culture evidence"),
    ):
        for value in get_values_from_criteria(criteria.get(key)):
            terms = _criterion_match_terms(str(value), key, criteria.get(key))
            add_concept(kind, str(value), terms, evidence_type)

    for value in get_values_from_criteria(criteria.get("required_geographies")):
        terms = _criterion_match_terms(str(value), "required_geographies", criteria.get("required_geographies"))
        add_concept(
            "geography",
            str(value),
            terms,
            "explicit market ownership/sold-into/covered/generated-pipeline evidence; location alone is supporting only",
        )

    for value in get_values_from_criteria(criteria.get("hiring_company")):
        add_concept("hiring_company", str(value), [str(value)], "employer/company relevance evidence")

    return {
        "query": original_query,
        "concepts": concepts,
        "evidence_policy": {
            "candidate_facts": "Use only DB/profile/import evidence.",
            "geography": "Current location, role city, company offices, or customer presence alone cannot verify market experience.",
            "company_facts": "Company facts may use cached/web-backed evidence.",
        },
        "evidence_catalog_version": catalog.get("version"),
    }


def _concept_terms_for_kind(concept_pack: Dict[str, Any], kind: str) -> List[str]:
    terms: List[str] = []
    for concept in concept_pack.get("concepts") or []:
        if not isinstance(concept, dict):
            continue
        if kind and _normalize_search_text(concept.get("kind")) != _normalize_search_text(kind):
            continue
        terms.extend(_flatten_value_for_evidence({
            "name": concept.get("name"),
            "aliases": concept.get("aliases"),
            "positive_signals": concept.get("positive_signals"),
            "meaning": concept.get("meaning"),
        }, max_items=60))
    return terms


def _search_terms_for_requirement(criteria: Dict[str, Any], concept_pack: Dict[str, Any], requirement_key: str, value: Any = None) -> List[str]:
    terms: List[str] = []
    if value is not None:
        terms.append(str(value))
    if requirement_key == "min_function_years":
        for item in _min_function_year_items(criteria):
            terms.append(str(item.get("function") or ""))
            terms.extend(str(alias) for alias in (item.get("aliases") or item.get("expanded_terms") or []))
        terms.extend(_concept_terms_for_kind(concept_pack, "function"))
    elif requirement_key == "required_geographies":
        terms.extend(_criterion_match_terms(str(value or ""), "required_geographies", criteria.get(requirement_key)))
        terms.extend(_concept_terms_for_kind(concept_pack, "geography"))
    elif requirement_key in TEXT_CRITERIA_CONFIG:
        terms.extend(_criterion_match_terms(str(value or ""), requirement_key, criteria.get(requirement_key)))
        kind_map = {
            "required_functions": "function",
            "required_segments": "segment",
            "required_industries": "industry",
            "required_company_details": "company_detail",
            "required_culture_type": "culture",
        }
        if requirement_key in kind_map:
            terms.extend(_concept_terms_for_kind(concept_pack, kind_map[requirement_key]))
    elif requirement_key == "hiring_company":
        terms.extend(_concept_terms_for_kind(concept_pack, "hiring_company"))
        terms.extend(get_values_from_criteria(criteria.get("hiring_company")))

    clean_terms = []
    seen = set()
    stop_terms = {
        "candidate", "candidates", "experience", "years", "market", "sales", "business",
        "required", "evidence", "profile", "company", "role", "function", "customer",
    }
    for term in terms:
        term_l = _normalize_search_text(term)
        if not term_l or term_l in stop_terms or len(term_l) < 2:
            continue
        if term_l not in seen:
            clean_terms.append(term_l)
            seen.add(term_l)
    return clean_terms[:80]


def _is_location_or_identity_source(source: str, path: str = "") -> bool:
    combined = _normalize_search_text(f"{source} {path}")
    location_tokens = (
        "address",
        "addresswithcountry",
        "location",
        "current location",
        "city",
        "state",
        "country",
        "headquarters",
        "hq",
        "office",
        "customer presence",
        "company presence",
        "company details",
        "role company",
        "company geography",
    )
    return any(token in combined for token in location_tokens)


def _geography_text_has_market_action(text: str) -> bool:
    text_l = _normalize_search_text(text)
    action_patterns = (
        r"\b(sold|sell|selling|sales|owned|owning|covered|covering|managed|managing|handled|handling)\b",
        r"\b(generated|generating|built|building|drove|driving|led|leading|expanded|expanding)\b",
        r"\b(pipeline|revenue|quota|territory|region|regional|market|markets|gtm|go to market|go-to-market)\b",
        r"\b(partner|partners|channel|alliances?|reseller|customers?|accounts?|logos?)\b",
        r"\bacross\b",
    )
    return any(re.search(pattern, text_l) for pattern in action_patterns)


def _function_source_supporting_only(source: str, path: str = "") -> bool:
    combined = _normalize_search_text(f"{source} {path}")
    supporting_tokens = (
        "skills",
        "skill",
        "endorsement",
        "role company",
        "role dates",
        "address",
        "location",
        "city",
        "headline company",
    )
    return any(token in combined for token in supporting_tokens)


def _chunk_quality_for_requirement(chunk: Dict[str, Any], requirement_key: str) -> str:
    source = _normalize_search_text(chunk.get("source"))
    category = _normalize_search_text(chunk.get("category"))
    path = _normalize_search_text(chunk.get("path"))
    if requirement_key == "required_geographies":
        if (
            _is_location_or_identity_source(source, path)
            or category in {"company_fact", "relationship_scope", "operational_metadata"}
            or not _geography_text_has_market_action(chunk.get("text", ""))
        ):
            return "supporting_only"
    if requirement_key == "min_function_years":
        if _function_source_supporting_only(source, path) or category in {"relationship_scope", "operational_metadata"}:
            return "supporting_only"
    if category == "operational_metadata":
        return "supporting_only"
    if source in {"about", "headline", "notes", "response"} or source.startswith("uploaded fields") or source.startswith("role") or source.startswith("schema"):
        return "explicit"
    return "partial"


def _evidence_hits_for_terms(
    chunks: List[Dict[str, Any]],
    terms: List[str],
    requirement_key: str,
    label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hits: List[Dict[str, Any]] = []
    matched_roles: List[Dict[str, Any]] = []
    for chunk in chunks:
        term = next((term for term in terms if _term_matches_text(term, chunk.get("text_l", ""))), None)
        if not term:
            continue
        quality = _chunk_quality_for_requirement(chunk, requirement_key)
        hits.append(
            {
                "criterion": label,
                "value": term,
                "source": chunk.get("source"),
                "quality": quality,
                "snippet": _evidence_snippet(chunk.get("text", ""), term),
            }
        )
        if chunk.get("role"):
            matched_roles.append(chunk["role"])
    hits.sort(key=lambda item: _quality_rank(item.get("quality")), reverse=True)
    return hits[:6], matched_roles


def _quality_rank(value: str) -> int:
    return {"missing": 0, "supporting_only": 1, "partial": 2, "explicit": 3, "strong": 4}.get(str(value or ""), 0)


def _best_evidence_quality(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "missing"
    return max((str(hit.get("quality") or "partial") for hit in hits), key=_quality_rank)


def _max_explicit_year_claim(hits: List[Dict[str, Any]]) -> float:
    """Extract explicit duration claims from already-matched evidence snippets."""
    max_years = 0.0
    for hit in hits or []:
        if str(hit.get("quality") or "") != "explicit":
            continue
        text = " ".join(
            str(hit.get(key) or "")
            for key in ("snippet", "value")
            if hit.get(key) is not None
        )
        for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", text, flags=re.IGNORECASE):
            try:
                max_years = max(max_years, float(match.group(1)))
            except Exception:
                continue
    return max_years


def _matched_role_duration_years(roles: List[Dict[str, Any]]) -> float:
    unique_roles: List[Dict[str, Any]] = []
    seen = set()
    for role in roles or []:
        role_key = (role.get("company"), role.get("title"), role.get("start_date"), role.get("end_date"), role.get("duration_years"))
        if role_key in seen:
            continue
        seen.add(role_key)
        unique_roles.append(role)
    return calculate_merged_duration_years(unique_roles) if unique_roles else 0.0


def build_schema_aware_evidence_card(
    profile: Dict[str, Any],
    criteria: Dict[str, Any],
    concept_pack: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    profile_safe = copy.deepcopy({k: v for k, v in (profile or {}).items() if k != "embedding"})
    chunks = build_profile_evidence_chunks(profile_safe)
    intelligence = build_shortlist_intelligence_pack(profile_safe, criteria)
    matched: List[Dict[str, Any]] = []
    missing: List[str] = []
    evidence: List[Dict[str, Any]] = []
    contributing_roles: List[Dict[str, Any]] = []
    calculated: Dict[str, Any] = {}
    score = 0.0
    max_score = 0.0

    min_total = criteria.get("min_total_experience")
    if min_total is not None:
        max_score += 1.0
        actual = float(profile_safe.get("total_experience_years") or 0)
        if actual >= float(min_total):
            score += 1.0
            matched.append({"criterion": "Total experience", "value": f"{actual:g} years"})
            evidence.append({"criterion": "Total experience", "value": min_total, "source": "profile", "quality": "explicit", "snippet": f"Total experience {actual:g} years"})
        else:
            missing.append(f"Total experience >= {min_total} years")

    if criteria.get("min_people_managed") is not None:
        max_score += 1.0
        actual = int(profile_safe.get("max_people_managed") or 0)
        if actual >= int(criteria.get("min_people_managed")):
            score += 1.0
            matched.append({"criterion": "People managed", "value": str(actual)})
        else:
            missing.append(f"Managed team size >= {criteria.get('min_people_managed')}")

    for item in _min_function_year_items(criteria):
        function = str(item.get("function") or "").strip()
        min_years = float(item.get("min_years") or 0)
        max_score += 1.5
        criteria_obj = {"values": [{"function": function, "aliases": item.get("aliases") or item.get("expanded_terms") or []}]}
        duration, roles = calculate_functional_experience_duration(profile_safe, criteria_obj)
        terms = _search_terms_for_requirement(criteria, concept_pack, "min_function_years", function)
        hits, hit_roles = _evidence_hits_for_terms(chunks, terms, "min_function_years", "Function-specific tenure")
        function_quality = _best_evidence_quality(hits)
        explicit_year_claim = _max_explicit_year_claim(hits)
        effective_duration = max(duration, explicit_year_claim)
        calculated[f"min_function_years:{function}"] = {
            "duration": duration,
            "explicit_year_claim": explicit_year_claim or None,
            "effective_duration": effective_duration,
            "required": min_years,
            "roles": roles,
        }
        if effective_duration >= min_years and hits and function_quality == "explicit":
            score += 1.5
            matched.append({"criterion": "Function-specific tenure", "value": f"{effective_duration:g} years in {function}"})
            evidence.extend(hits[:4])
            contributing_roles.extend(hit_roles or roles)
        elif hits:
            score += 0.7
            evidence.extend(hits[:4])
            missing.append(f"{function} experience >= {min_years:g} years")
        else:
            missing.append(f"{function} experience >= {min_years:g} years")

    for key, config in TEXT_CRITERIA_CONFIG.items():
        criterion = criteria.get(key)
        values = get_values_from_criteria(criterion)
        if not values:
            continue
        max_score += float(config.get("weight") or 1.0)
        key_hits: List[Dict[str, Any]] = []
        matched_values: List[str] = []
        for value in values:
            terms = _search_terms_for_requirement(criteria, concept_pack, key, value)
            hits, roles = _evidence_hits_for_terms(chunks, terms, key, config.get("label", key))
            quality = _best_evidence_quality(hits)
            duration_ok = True
            if key == "required_geographies" and isinstance(criterion, dict) and criterion.get("min_years"):
                required_years = float(criterion.get("min_years") or 0)
                evidenced_years = max(_max_explicit_year_claim(hits), _matched_role_duration_years(roles))
                calculated[f"required_geographies:{value}"] = {
                    "duration": evidenced_years,
                    "required": required_years,
                }
                duration_ok = evidenced_years >= required_years
            if quality == "explicit" and duration_ok:
                matched_values.append(str(value))
                key_hits.extend(hits)
                contributing_roles.extend(roles)
            elif hits:
                key_hits.extend(hits)
        if matched_values:
            score += float(config.get("weight") or 1.0)
            matched.append({"criterion": config.get("label", key), "value": ", ".join(matched_values)})
            evidence.extend(key_hits[:5])
        elif key_hits:
            score += float(config.get("weight") or 1.0) * 0.45
            evidence.extend(key_hits[:5])
            missing.extend(f"{config.get('label', key)}: {value}" for value in values)
        else:
            missing.extend(f"{config.get('label', key)}: {value}" for value in values)

    relevance_weight, relevance_earned, relevance_matched, relevance_evidence, relevance_roles = _score_hiring_company_relevance(profile_safe, criteria)
    if relevance_weight:
        max_score += relevance_weight
        score += relevance_earned
        matched.extend(relevance_matched)
        evidence.extend(relevance_evidence)
        contributing_roles.extend(relevance_roles)

    if max_score <= 0:
        keyword_terms = _search_terms_for_requirement(criteria, concept_pack, "required_keywords", None)
        hits, roles = _evidence_hits_for_terms(chunks, keyword_terms, "required_keywords", "Keywords")
        max_score = 1.0
        if hits:
            score = 0.7
            evidence.extend(hits[:5])
            contributing_roles.extend(roles)

    normalized_score = round(min(100.0, (score / max_score) * 100.0), 1) if max_score else 0.0
    best_quality = _best_evidence_quality(evidence)
    status = "potential" if normalized_score >= SCREENING_LOCAL_POTENTIAL_THRESHOLD and best_quality in {"explicit", "partial"} else "not_verified"
    if best_quality == "supporting_only" and status == "potential":
        status = "not_verified"

    role_details = []
    seen_roles = set()
    for role in contributing_roles or profile_safe.get("roles", [])[:3]:
        role_key = (role.get("company"), role.get("title"), role.get("start_date"), role.get("end_date"))
        if role_key in seen_roles:
            continue
        seen_roles.add(role_key)
        role_details.append(
            {
                "company": role.get("company", ""),
                "title": role.get("title", ""),
                "duration_years": role.get("duration_years", 0.0) or 0.0,
                "start_date": role.get("start_date"),
                "end_date": role.get("end_date"),
            }
        )
        if len(role_details) >= 5:
            break

    evidence_card = {
        "candidate_id": profile_safe.get("id"),
        "candidate_name": profile_safe.get("name"),
        "headline": profile_safe.get("headline"),
        "current_location": intelligence.get("current_location"),
        "career_metrics": intelligence.get("career_metrics"),
        "team_management": intelligence.get("team_management"),
        "gap_analysis": intelligence.get("gap_analysis"),
        "matched_criteria": matched,
        "missing_criteria": missing[:12],
        "evidence": evidence[:12],
        "calculated_experience": calculated,
        "contributing_roles": role_details,
        "evidence_catalog_version": catalog.get("version"),
    }

    profile_safe["match_score"] = normalized_score
    profile_safe["matched_criteria"] = matched
    profile_safe["missing_criteria"] = missing[:12]
    profile_safe["evidence_log"] = evidence[:12]
    profile_safe["calculated_experience"] = calculated
    profile_safe["contributing_roles_details"] = {"roles": role_details}
    profile_safe["shortlist_intelligence"] = intelligence
    profile_safe["evidence_card"] = evidence_card
    profile_safe["evidence_quality"] = best_quality
    profile_safe["shortlist_status"] = status
    profile_safe["is_verified_match"] = False
    profile_safe["analyst_status"] = "local_evidence_scan"
    profile_safe["analyst_reasoning"] = "Local schema-aware evidence scan; LLM review not yet run."
    profile_safe["auditor_status"] = "not_run"
    profile_safe["auditor_reasoning"] = ""
    profile_safe["review_stage"] = "local_evidence_only"
    profile_safe["answer"] = (
        "Potential evidence found by local schema-aware scan; not yet reviewed by AI."
        if status == "potential"
        else "No explicit profile evidence found for the required shortlist criteria."
    )
    profile_safe["reasoning"] = profile_safe["analyst_reasoning"]
    return profile_safe


def schema_aware_local_evidence_scan(
    profiles: List[Dict[str, Any]],
    criteria: Dict[str, Any],
    concept_pack: Dict[str, Any],
    catalog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scored = [build_schema_aware_evidence_card(profile, criteria, concept_pack, catalog) for profile in profiles]
    return _sort_shortlist_visible_candidates(scored)


def select_candidates_for_limited_llm_review(scored_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eligible = [
        profile for profile in scored_profiles
        if (
            (
                float(profile.get("match_score") or 0) >= SCREENING_LLM_REVIEW_MIN_SCORE
                or profile.get("shortlist_status") == "potential"
            )
            and profile.get("evidence_quality") in {"strong", "explicit", "partial"}
        )
    ]
    eligible.sort(
        key=lambda profile: (
            float(profile.get("match_score") or 0),
            profile.get("evidence_quality") == "explicit",
            float(profile.get("total_experience_years") or 0),
        ),
        reverse=True,
    )
    if SCREENING_LLM_REVIEW_LIMIT > 0:
        return eligible[:SCREENING_LLM_REVIEW_LIMIT]
    return eligible


def compact_shortlist_evidence_card(profile: Dict[str, Any], *, max_evidence: int = 6) -> Dict[str, Any]:
    evidence_card = profile.get("evidence_card") if isinstance(profile.get("evidence_card"), dict) else {}
    intelligence = profile.get("shortlist_intelligence") if isinstance(profile.get("shortlist_intelligence"), dict) else {}
    return {
        "candidate_id": profile.get("id"),
        "name": profile.get("name"),
        "headline": profile.get("headline"),
        "local_status": profile.get("shortlist_status"),
        "local_score": profile.get("match_score"),
        "evidence_quality": profile.get("evidence_quality"),
        "current_location": (intelligence.get("current_location") or evidence_card.get("current_location")),
        "career_metrics": (intelligence.get("career_metrics") or evidence_card.get("career_metrics")),
        "matched_criteria": profile.get("matched_criteria") or evidence_card.get("matched_criteria") or [],
        "missing_criteria": (profile.get("missing_criteria") or evidence_card.get("missing_criteria") or [])[:10],
        "calculated_experience": profile.get("calculated_experience") or evidence_card.get("calculated_experience") or {},
        "contributing_roles": (profile.get("contributing_roles_details") or {}).get("roles") or evidence_card.get("contributing_roles") or [],
        "evidence": (profile.get("evidence_log") or evidence_card.get("evidence") or [])[:max_evidence],
    }


async def evaluate_shortlist_evidence_batch(
    profiles: List[Dict[str, Any]],
    original_query: str,
    criteria: Dict[str, Any],
    concept_pack: Dict[str, Any],
    catalog: Dict[str, Any],
    tracker: TokenCostTracker,
) -> Dict[str, Dict[str, Any]]:
    if not profiles:
        return {}
    system_prompt = (
        "You are a recruitment shortlist judge. Review compact evidence cards, not full profiles. "
        "Return valid JSON only with key results: list of objects with candidate_id, shortlist_status, match_score, answer, reasoning, matched_criteria, missing_criteria, evidence_quality, confidence. "
        "shortlist_status must be verified_match, potential, or not_verified. "
        "Use the query concept pack to understand terminology and geography context. "
        "Candidate facts must come only from the evidence cards. Missing facts stay missing. "
        "Do not verify geography from location/company presence alone; market experience needs explicit evidence of sold/covered/owned/generated pipeline/revenue or handled partners/customers in that market. "
        "Do not count partner/channel/alliance as direct sales unless the query asks for that motion."
    )
    cards = [compact_shortlist_evidence_card(profile) for profile in profiles]
    user_prompt = (
        f"Hiring query:\n{original_query}\n\n"
        f"Extracted criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Search concept pack:\n{json.dumps(concept_pack, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Evidence catalog summary:\n{json.dumps(compact_evidence_catalog_for_prompt(catalog), ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate evidence cards:\n{json.dumps(cards, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Return JSON only."
    )
    structured = await asyncio.to_thread(
        call_openai_json,
        system_prompt,
        user_prompt,
        model=SCREENING_REASONING_MODEL,
        use_web=False,
        temperature=0.0,
        timeout=60.0,
    )
    tracker.add_usage(SCREENING_REASONING_MODEL, f"{system_prompt}\n\n{user_prompt}", json.dumps(structured), "Shortlist Evidence Batch Review")
    results = structured.get("results") if isinstance(structured, dict) else []
    if not isinstance(results, list):
        return {}
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if candidate_id is not None:
            mapped[str(candidate_id)] = item
    return mapped


async def audit_shortlist_evidence_batch(
    profiles: List[Dict[str, Any]],
    original_query: str,
    criteria: Dict[str, Any],
    concept_pack: Dict[str, Any],
    catalog: Dict[str, Any],
    tracker: TokenCostTracker,
) -> Dict[str, Dict[str, Any]]:
    if not profiles:
        return {}
    system_prompt = (
        "You are a strict evidence auditor for recruiting shortlist results. "
        "Audit each analyst result against its evidence card. Return valid JSON only with key results: list of objects with "
        "candidate_id, auditor_status, final_status, evidence_quality, auditor_reasoning, matched_criteria, missing_criteria, evidence, confidence, match_score, answer. "
        "auditor_status must be passed, downgraded, or needs_review. final_status must be verified_match, potential, or not_verified. "
        "Candidate facts may come only from the candidate evidence cards/profile/import/DB evidence in this prompt. Do not invent facts. "
        "Company facts may be considered only when already present in cached/web-backed company facts in the evidence. "
        "Every required criterion must have explicit candidate evidence to verify. Plausible or inferential evidence cannot be verified_match. "
        "Geography experience requires explicit evidence that the candidate sold into, owned, covered, managed, generated pipeline/revenue in, or handled partners/customers for that market. "
        "Current location, role location, employer HQ/offices, customer presence, or country-region membership are supporting-only and cannot verify geography by themselves. "
        "Function-specific years must be backed by role dates/durations or explicit profile/import year claims. "
        "Partner/channel/alliance experience can satisfy partner/alliance queries, but not direct sales unless the query asks for it."
    )
    cards = [
        {
            **compact_shortlist_evidence_card(profile),
            "analyst_result": {
                "shortlist_status": profile.get("shortlist_status"),
                "match_score": profile.get("match_score"),
                "answer": profile.get("answer"),
                "reasoning": profile.get("reasoning"),
                "matched_criteria": profile.get("matched_criteria"),
                "missing_criteria": profile.get("missing_criteria"),
                "evidence_quality": profile.get("evidence_quality"),
            },
        }
        for profile in profiles
    ]
    user_prompt = (
        f"Hiring query:\n{original_query}\n\n"
        f"Extracted criteria:\n{json.dumps(criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Search concept pack:\n{json.dumps(concept_pack, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Evidence catalog summary:\n{json.dumps(compact_evidence_catalog_for_prompt(catalog), ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Analyst results and evidence cards:\n{json.dumps(cards, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Audit each candidate. Return JSON only."
    )
    structured = await asyncio.to_thread(
        call_openai_json,
        system_prompt,
        user_prompt,
        model=SCREENING_AUDIT_MODEL,
        use_web=False,
        temperature=0.0,
        timeout=60.0,
    )
    tracker.add_usage(SCREENING_AUDIT_MODEL, f"{system_prompt}\n\n{user_prompt}", json.dumps(structured), "Shortlist Evidence Batch Audit")
    results = structured.get("results") if isinstance(structured, dict) else []
    if not isinstance(results, list):
        return {}
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if candidate_id is not None:
            mapped[str(candidate_id)] = item
    return mapped


async def audit_shortlist_evidence_batch_with_retry(
    profiles: List[Dict[str, Any]],
    original_query: str,
    criteria: Dict[str, Any],
    concept_pack: Dict[str, Any],
    catalog: Dict[str, Any],
    tracker: TokenCostTracker,
    *,
    retry_batch_size: int = 5,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    try:
        return (
            await audit_shortlist_evidence_batch(
                profiles,
                original_query,
                criteria,
                concept_pack,
                catalog,
                tracker,
            ),
            None,
        )
    except Exception as first_error:
        logger.warning("Shortlist evidence batch audit failed; retrying in smaller batches: %s", first_error)
        results: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = [str(first_error)]
        for start in range(0, len(profiles), max(1, retry_batch_size)):
            sub_batch = profiles[start : start + max(1, retry_batch_size)]
            try:
                results.update(
                    await audit_shortlist_evidence_batch(
                        sub_batch,
                        original_query,
                        criteria,
                        concept_pack,
                        catalog,
                        tracker,
                    )
                )
            except Exception as sub_error:
                logger.error("Shortlist evidence audit retry batch failed: %s", sub_error, exc_info=True)
                errors.append(str(sub_error))
        return results, "; ".join(errors) if errors else None


def apply_evidence_batch_review(profile: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    if not review:
        return profile
    updated = copy.deepcopy(profile)
    status = _normalize_shortlist_status(review.get("shortlist_status")) or updated.get("shortlist_status") or "not_verified"
    updated["shortlist_status"] = status
    updated["is_verified_match"] = status == "verified_match"
    updated["analyst_status"] = status
    updated["review_stage"] = "analyst_reviewed"
    updated["match_score"] = round(float(review.get("match_score") or updated.get("match_score") or 0), 1)
    updated["answer"] = str(review.get("answer") or updated.get("answer") or "").replace("|", " ").strip()
    updated["reasoning"] = str(review.get("reasoning") or updated.get("reasoning") or "").replace("\n", " ").replace("|", " ").strip()
    updated["analyst_reasoning"] = updated["reasoning"]
    if isinstance(review.get("matched_criteria"), list):
        updated["matched_criteria"] = review["matched_criteria"]
    if isinstance(review.get("missing_criteria"), list):
        updated["missing_criteria"] = review["missing_criteria"]
    if review.get("evidence_quality"):
        updated["evidence_quality"] = str(review.get("evidence_quality")).strip().lower()
    if review.get("confidence"):
        updated["confidence"] = str(review.get("confidence")).strip().lower()
    return updated


def apply_shortlist_audit_result(profile: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    if not audit:
        return profile
    updated = copy.deepcopy(profile)
    analyst_status = _normalize_shortlist_status(updated.get("shortlist_status")) or "not_verified"
    audit_status = _normalize_search_text(audit.get("auditor_status") or "needs_review") or "needs_review"
    final_status = _normalize_shortlist_status(audit.get("final_status")) or analyst_status
    if analyst_status != "verified_match" and final_status == "verified_match":
        final_status = "potential"
    updated["auditor_status"] = audit_status
    updated["review_stage"] = "audited"
    updated["auditor_reasoning"] = str(audit.get("auditor_reasoning") or "").replace("\n", " ").replace("|", " ").strip()
    updated["evidence_quality"] = str(audit.get("evidence_quality") or updated.get("evidence_quality") or "unknown").strip().lower()
    updated["shortlist_status"] = final_status
    updated["is_verified_match"] = final_status == "verified_match"
    if isinstance(audit.get("matched_criteria"), list):
        updated["matched_criteria"] = audit["matched_criteria"]
    if isinstance(audit.get("missing_criteria"), list):
        updated["missing_criteria"] = audit["missing_criteria"]
    if isinstance(audit.get("evidence"), list):
        updated["evidence_log"] = audit["evidence"]
    if audit.get("confidence"):
        updated["confidence"] = str(audit.get("confidence")).strip().lower()
    if audit.get("match_score") is not None:
        try:
            updated["match_score"] = round(float(audit.get("match_score")), 1)
        except Exception:
            pass
    if audit.get("answer"):
        updated["answer"] = str(audit.get("answer")).replace("|", " ").strip()
    if updated.get("auditor_reasoning"):
        updated["reasoning"] = updated["auditor_reasoning"]
    return updated


def _shortlist_review_has_meaningful_evidence(profile: Dict[str, Any]) -> bool:
    return bool(
        profile.get("matched_criteria")
        or profile.get("evidence_log")
        or float(profile.get("match_score") or 0) >= float(os.getenv("SCREENING_AUDIT_BORDERLINE_THRESHOLD", "45"))
    )


def _should_audit_shortlist_result(profile: Dict[str, Any]) -> bool:
    status = _normalize_shortlist_status(profile.get("shortlist_status"))
    if status in {"verified_match", "potential"}:
        return True
    return _shortlist_review_has_meaningful_evidence(profile)


def _set_criteria_values(criteria: Dict[str, Any], key: str, values: List[str]) -> None:
    cleaned = sorted({str(value).strip() for value in values if str(value or "").strip()}, key=str.lower)
    if not cleaned:
        return
    existing = criteria.get(key)
    if isinstance(existing, dict):
        # Values inside one criterion are aliases/alternatives. Independent
        # requirements live in separate criterion keys and are ANDed by the
        # strict scorer; preserving AND here would require every synonym.
        criteria[key] = {**existing, "operator": "OR", "values": cleaned}
    else:
        criteria[key] = {"operator": "OR", "values": cleaned}


def _criteria_values_for_search(criteria: Dict[str, Any], key: str) -> List[str]:
    if key == "required_companies":
        value = criteria.get(key)
        return [item["company"] for item in _company_criteria_items(value)]
    return [str(item).strip() for item in get_values_from_criteria(criteria.get(key)) if str(item or "").strip()]


def _profile_location_text(profile: Dict[str, Any]) -> str:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    location_bits = [
        profile.get("location"),
        profile.get("city"),
        raw_fields.get("addressWithCountry"),
        raw_fields.get("address"),
        raw_fields.get("location"),
    ]
    return " ".join(_flatten_value_for_evidence(location_bits, max_items=20)).lower()


def _profile_general_text(profile: Dict[str, Any]) -> str:
    raw_fields = profile.get("raw_fields") if isinstance(profile.get("raw_fields"), dict) else {}
    role_text = []
    for role in profile.get("roles") or []:
        role_text.extend(
            _flatten_value_for_evidence(
                {
                    "title": role.get("title"),
                    "details": role.get("details"),
                    "company": role.get("company"),
                    "company_details": role.get("company_details"),
                },
                max_items=80,
            )
        )
    return " ".join(
        _flatten_value_for_evidence(
            {
                "headline": profile.get("headline"),
                "about": profile.get("about"),
                "candidate_services": profile.get("candidate_services"),
                "extracted_industry": profile.get("extracted_industry"),
                "raw_fields": raw_fields,
                "roles": role_text,
            },
            max_items=220,
        )
    ).lower()


def _strict_presence_result(
    profile: Dict[str, Any],
    criteria_key: str,
    criterion: Any,
    criteria_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    values = _criteria_values_for_search({"value": criterion}, "value") if criteria_key == "value" else [
        str(item).strip() for item in get_values_from_criteria(criterion) if str(item or "").strip()
    ]
    company_items = _company_criteria_items(criterion) if criteria_key == "required_companies" else []
    if criteria_key == "required_companies":
        values = [item["company"] for item in company_items]
    if not values:
        return {"applicable": False, "met": True, "score": 1.0, "matched": [], "missing": [], "evidence": [], "roles": []}

    operator = _criterion_operator(criterion)
    matched: List[str] = []
    missing: List[str] = []
    evidence: List[Dict[str, Any]] = []
    roles: List[Dict[str, Any]] = []

    iterable_values: List[Any] = company_items if criteria_key == "required_companies" else values
    default_company_scope = "any_employer"
    if isinstance(criterion, dict):
        default_company_scope = str(criterion.get("employment_scope") or criterion.get("scope") or default_company_scope)

    for item_value in iterable_values:
        value = item_value["company"] if isinstance(item_value, dict) else str(item_value)
        terms = _company_match_terms(item_value) if criteria_key == "required_companies" and isinstance(item_value, dict) else _criterion_match_terms(value, criteria_key, criterion)
        found = None

        if criteria_key == "required_companies":
            current_only = _company_scope_current_only(item_value if isinstance(item_value, dict) else {}, default_company_scope)
            role_scope = _current_roles(profile) if current_only else (profile.get("roles") or [])
            for role in role_scope:
                role_company = str(role.get("company") or "")
                matched_company = next((term for term in terms if _company_matches(role_company, term)), None)
                if matched_company:
                    found = ("role company", role_company, role, role_company)
                    break
        elif criteria_key == "required_locations":
            location_text = _profile_location_text(profile)
            term = next((term for term in terms if _term_matches_text(term, location_text)), None)
            if term:
                found = ("candidate location", _evidence_snippet(location_text, term), None, location_text)
        elif criteria_key == "required_geographies":
            for role in profile.get("roles") or []:
                role_text = _role_geography_text_for_profile(profile, role)
                geo_terms = _geography_match_terms(value, criterion)
                term = next((term for term in geo_terms if _term_matches_text(term, role_text)), None)
                if term:
                    found = ("role/company geography", _evidence_snippet(role_text, term), role, role_text)
                    break
            if not found:
                general_text = _profile_geography_experience_text(profile)
                geo_terms = _geography_match_terms(value, criterion)
                term = next((term for term in geo_terms if _term_matches_text(term, general_text)), None)
                if term:
                    found = ("enriched profile geography", _evidence_snippet(general_text, term), None, general_text)
        elif criteria_key in {"required_industries", "required_segments", "required_company_details", "required_culture_type"}:
            # These are employer attributes. Respect employment_scope and avoid
            # satisfying them from unrelated candidate-level or past-role text.
            role_scope = _roles_for_employment_scope(profile, criterion, default_company_scope)
            company_field_map = {
                "required_industries": ("industry", "product_service", "business_model"),
                "required_segments": ("customer_segment",),
                "required_company_details": (
                    "industry", "product_service", "business_model", "funding_stage",
                    "company_status", "ownership", "revenue",
                ),
                "required_culture_type": ("culture_type",),
            }
            allowed_fields = company_field_map[criteria_key]
            for role_idx, role in enumerate(role_scope, start=1):
                details = role.get("company_details") if isinstance(role.get("company_details"), dict) else {}
                company_payload = {field: details.get(field) for field in allowed_fields}
                company_text = " ".join(_flatten_value_for_evidence(company_payload, max_items=80))
                company_text_l = _normalize_search_text(company_text)
                term = next((term for term in terms if _term_matches_text(term, company_text_l)), None)
                if term:
                    found = (
                        f"role {role_idx} company details",
                        _evidence_snippet(company_text, term),
                        role,
                        company_text,
                    )
                    break
            if not found:
                for role in role_scope:
                    web_text, web_item = _web_company_profile_text(role.get("company") or "", criteria_context or {})
                    term = next((term for term in terms if _term_matches_text(term, web_text)), None)
                    if term:
                        found = ("web company profile", _evidence_snippet(web_text, term), role, web_text)
                        break
        else:
            chunks = build_profile_evidence_chunks(profile)
            for chunk in chunks:
                term = next((term for term in terms if _term_matches_text(term, chunk["text_l"])), None)
                if term:
                    found = (chunk["source"], _evidence_snippet(chunk["text"], term), chunk.get("role"), chunk["text"])
                    break

        if found:
            source, snippet, role, source_text = found
            matched.append(value)
            evidence.append(
                {
                    "criterion": TEXT_CRITERIA_CONFIG.get(criteria_key, {}).get("label", criteria_key),
                    "value": value,
                    "source": source,
                    "snippet": snippet,
                    "source_text": source_text,
                }
            )
            if role:
                roles.append(role)
        else:
            missing.append(value)

    met = len(matched) == len(values) if operator == "AND" else bool(matched)
    return {
        "applicable": True,
        "met": met,
        "operator": operator,
        "score": len(matched) / max(1, len(values)),
        "matched": matched,
        "missing": missing,
        "evidence": evidence,
        "roles": roles,
    }


def _strict_funding_stage_result(profile: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
    min_stage = _funding_min_value(criteria)
    min_rank = _funding_rank(min_stage)
    if not min_stage or min_rank is None:
        return {"applicable": False, "met": True, "evidence": [], "roles": [], "matched": [], "missing": []}

    stage_criterion = criteria.get("funding_stage_min")
    if isinstance(stage_criterion, dict):
        scope = _normalize_search_text(stage_criterion.get("employment_scope") or stage_criterion.get("scope") or "current_employer")
    else:
        scope = "current_employer"
    current_only = scope not in {"any_employer", "past_or_current", "worked_at", "worked_with", "all_roles"}
    roles_to_check = _current_roles(profile) if current_only else (profile.get("roles") or [])

    below_threshold: List[str] = []
    for role in roles_to_check:
        company_details = role.get("company_details") or {}
        stage_text = " ".join(
            _flatten_value_for_evidence(
                {
                    "funding_stage": company_details.get("funding_stage"),
                    "company_status": company_details.get("company_status"),
                    "ownership": company_details.get("ownership"),
                },
                max_items=20,
            )
        )
        rank = _funding_rank(stage_text)
        if rank is None:
            continue
        if rank >= min_rank:
            snippet = f"{role.get('company')}: {stage_text}"
            return {
                "applicable": True,
                "met": True,
                "score": 1.0,
                "matched": [min_stage],
                "missing": [],
                "evidence": [{
                    "criterion": "Funding stage",
                    "value": min_stage,
                    "source": "role company details",
                    "snippet": snippet,
                    "source_text": snippet,
                }],
                "roles": [role],
            }
        below_threshold.append(f"{role.get('company')}: {stage_text}")

    missing = f"Funding stage >= {min_stage}"
    if below_threshold:
        missing += f" (known below threshold: {'; '.join(below_threshold[:2])})"
    return {
        "applicable": True,
        "met": False,
        "score": 0.0,
        "matched": [],
        "missing": [missing],
        "evidence": [],
        "roles": [],
    }


def _assign_evidence_ids(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned: List[Dict[str, Any]] = []
    for index, item in enumerate(evidence or [], start=1):
        if not isinstance(item, dict):
            continue
        next_item = dict(item)
        next_item["id"] = str(next_item.get("id") or f"ev{index}")
        assigned.append(next_item)
    return assigned


def _dedupe_evidence_log(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove repeated alias hits while preserving distinct evidence sources."""
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role") if isinstance(item.get("role"), dict) else {}
        key = (
            _normalize_search_text(item.get("criterion")),
            _normalize_search_text(item.get("source")),
            _normalize_search_text(item.get("source_text") or item.get("snippet") or item.get("value")),
            _normalize_search_text(role.get("company")),
            _normalize_search_text(role.get("title")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _prioritize_scoped_tenure_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [item for item in evidence or [] if isinstance(item, dict)],
        key=lambda item: 0 if "tenure" in _normalize_search_text(item.get("criterion")) else 1,
    )


def _link_calculated_experience_evidence_ids(calculated: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not calculated:
        return calculated
    linked = copy.deepcopy(calculated)
    by_criterion: Dict[str, List[str]] = {}
    for item in evidence or []:
        criterion = _normalize_search_text(item.get("criterion"))
        evidence_id = item.get("id")
        if criterion and evidence_id:
            by_criterion.setdefault(criterion, []).append(str(evidence_id))

    for key, value in linked.items():
        if isinstance(value, dict):
            label = _normalize_search_text(value.get("label") or key)
            ids = []
            for criterion, evidence_ids in by_criterion.items():
                if "tenure" in criterion or label and label in criterion:
                    ids.extend(evidence_ids)
            value["evidence_ids"] = sorted(set(ids))
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                label = _normalize_search_text(item.get("label") or key)
                ids = []
                for criterion, evidence_ids in by_criterion.items():
                    if "tenure" in criterion or label and label in criterion:
                        ids.extend(evidence_ids)
                item["evidence_ids"] = sorted(set(ids))
    return linked


def _scoped_tenure_summary(calculated: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for key, value in (calculated or {}).items():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            required = float(item.get("required") or 0)
            duration = float(item.get("duration") or 0)
            if required <= 0:
                continue
            summaries.append({
                "key": key,
                "dimension": item.get("dimension") or key,
                "label": item.get("label") or key,
                "duration": round(duration, 2),
                "required": required,
                "evidence_ids": item.get("evidence_ids") or [],
                "roles": item.get("roles") or [],
            })
    return summaries


def _strict_shortlist_score_candidate(
    profile: Dict[str, Any],
    criteria: Dict[str, Any],
    debug_reasons: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    profile_copy = copy.deepcopy({k: v for k, v in (profile or {}).items() if k != "embedding"})
    # Normalize DB roles with any richer imported experience data before every
    # strict check. This makes dates and current-employer semantics available to
    # all criteria instead of only to duration helpers.
    profile_copy["roles"] = _profile_roles_with_raw_experience(profile_copy)
    matched_criteria: List[Dict[str, Any]] = []
    evidence_log: List[Dict[str, Any]] = []
    contributing_roles: List[Dict[str, Any]] = []
    calculated_experience: Dict[str, Any] = {}
    score_parts: List[float] = []

    def reject(reason: str) -> None:
        if debug_reasons is not None:
            debug_reasons.append(reason)

    min_total_exp = criteria.get("min_total_experience")
    if min_total_exp is not None:
        actual = float(profile_copy.get("total_experience_years") or 0)
        if actual < float(min_total_exp):
            reject("min_total_experience")
            return None
        score_parts.append(1.0)
        matched_criteria.append({"criterion": "Total experience", "value": f"{actual:g} years"})
        evidence_log.append({
            "criterion": "Total experience",
            "value": str(min_total_exp),
            "source": "profile",
            "snippet": f"Total experience {actual:g} years",
            "source_text": f"Total experience {actual:g} years",
        })

    min_managed = criteria.get("min_people_managed")
    if min_managed is not None:
        actual = int(profile_copy.get("max_people_managed") or 0)
        if actual < int(min_managed):
            reject("min_people_managed")
            return None
        score_parts.append(1.0)
        matched_criteria.append({"criterion": "People managed", "value": str(actual)})

    if not check_excluded_geography_presence(profile_copy, criteria):
        reject("excluded_geography")
        return None
    if not check_tenure_in_latest_role(profile_copy, criteria):
        reject("min_tenure_in_latest_role")
        return None
    if criteria.get("min_tenure_in_latest_role"):
        score_parts.append(1.0)
        evidence_log.extend(profile_copy.get("evidence_log") or [])
    if not check_avg_tenure_in_last_n_roles(profile_copy, criteria):
        reject("avg_tenure_in_last_n_roles")
        return None
    if criteria.get("avg_tenure_in_last_n_roles"):
        score_parts.append(1.0)
        evidence_log.extend(profile_copy.get("evidence_log") or [])

    min_function_years = criteria.get("min_function_years")
    if min_function_years:
        items = min_function_years if isinstance(min_function_years, list) else [min_function_years]
        for item in items:
            if not isinstance(item, dict):
                continue
            function = str(item.get("function") or item.get("value") or "").strip()
            aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias or "").strip()]
            min_years = float(item.get("min_years") or 0)
            criterion = {"operator": "OR", "values": [{"function": function, "aliases": aliases}]}
            scoped = evaluate_scoped_duration(
                profile_copy,
                dimension="function",
                criterion=criterion,
                min_years=min_years,
                label=function,
            )
            calculated_experience.setdefault("min_function_years", []).append(
                {
                    "duration": scoped["duration"],
                    "roles": scoped["roles"],
                    "label": function,
                    "required": min_years,
                    "dimension": "function",
                    "evidence_ids": [],
                }
            )
            if not scoped["qualified"]:
                reject(f"min_function_years:{function}")
                return None
            score_parts.append(min(1.0, scoped["duration"] / max(min_years, 0.1)))
            matched_criteria.append({"criterion": "Function-specific tenure", "value": f"{scoped['duration']:g} years in {function}"})
            evidence_log.extend(scoped["evidence"])
            contributing_roles.extend(scoped["roles"])

    funding_result = _strict_funding_stage_result(profile_copy, criteria)
    if funding_result.get("applicable"):
        if not funding_result.get("met"):
            reject("funding_stage_min")
            return None
        score_parts.append(float(funding_result.get("score") or 1.0))
        matched_criteria.append({"criterion": "Funding stage", "value": ", ".join(funding_result.get("matched") or [])})
        evidence_log.extend(funding_result.get("evidence") or [])
        contributing_roles.extend(funding_result.get("roles") or [])

    for key, config in TEXT_CRITERIA_CONFIG.items():
        criterion = criteria.get(key)
        if not criterion:
            continue
        result = _strict_presence_result(profile_copy, key, criterion, criteria)
        if not result["applicable"]:
            continue
        if not result["met"]:
            reject(key)
            return None
        score_parts.append(float(result["score"]))
        matched_criteria.append(
            {
                "criterion": config["label"],
                "value": ", ".join(result["matched"]),
                "operator": result.get("operator", "OR"),
            }
        )
        evidence_log.extend(result["evidence"])
        contributing_roles.extend(result.get("roles") or [])

        calc_map = {
            "required_functions": "function",
            "required_industries": "industry",
            "required_segments": "segment",
            "required_geographies": "geography",
            "required_company_details": "company_detail",
        }
        dimension = calc_map.get(key)
        if dimension and isinstance(criterion, dict):
            min_years = float(criterion.get("min_years") or 0)
            scoped = evaluate_scoped_duration(
                profile_copy,
                dimension=dimension,
                criterion=criterion,
                min_years=min_years,
                label=", ".join(get_values_from_criteria(criterion)),
            )
            calculated_experience[key] = {
                "duration": scoped["duration"],
                "roles": scoped["roles"],
                "label": ", ".join(get_values_from_criteria(criterion)),
                "required": min_years,
                "dimension": dimension,
                "evidence_ids": [],
            }
            if min_years and not scoped["qualified"]:
                reject(f"{key}_duration")
                return None
            if min_years:
                evidence_log.extend(scoped["evidence"])
            contributing_roles.extend(scoped["roles"])

    if not score_parts and not matched_criteria:
        reject("no_applicable_criteria_matched")
        return None

    score = round((sum(score_parts) / max(1, len(score_parts))) * 100, 1)
    seen_roles = set()
    role_details = []
    role_sources = contributing_roles or profile_copy.get("roles", [])[:3]
    for role in role_sources:
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

    evidence_log = _assign_evidence_ids(
        _dedupe_evidence_log(_prioritize_scoped_tenure_evidence(evidence_log))
    )
    calculated_experience = _link_calculated_experience_evidence_ids(calculated_experience, evidence_log)

    profile_copy["shortlist_status"] = "shortlisted"
    profile_copy["is_verified_match"] = True
    profile_copy["verification_pending"] = False
    profile_copy["match_score"] = score
    profile_copy["matched_criteria"] = matched_criteria
    profile_copy["missing_criteria"] = []
    profile_copy["evidence_log"] = evidence_log
    profile_copy["calculated_experience"] = calculated_experience
    profile_copy["scoped_tenure"] = _scoped_tenure_summary(calculated_experience)
    profile_copy["contributing_roles_details"] = {"roles": role_details}
    profile_copy["confidence"] = "high" if score >= 85 else "medium"
    return profile_copy


async def generate_reasoning_for_profile(
    profile: Dict[str, Any],
    original_criteria: Dict[str, Any],
    tracker: TokenCostTracker,
    *,
    use_web_search: bool = False,
) -> Any:
    profile_safe = {k: v for k, v in (profile or {}).items() if k != "embedding"}
    evidence_card = {
        "candidate_id": profile_safe.get("id"),
        "name": profile_safe.get("name"),
        "headline": profile_safe.get("headline"),
        "matched_criteria": profile_safe.get("matched_criteria") or [],
        "missing_criteria": profile_safe.get("missing_criteria") or [],
        "calculated_experience": profile_safe.get("calculated_experience") or {},
        "scoped_tenure": profile_safe.get("scoped_tenure") or [],
        "contributing_roles": (profile_safe.get("contributing_roles_details") or {}).get("roles") or [],
        "evidence_log": profile_safe.get("evidence_log") or [],
    }
    system_prompt = (
        "You are a strict evidence auditor for recruiting shortlist results. "
        "Use only the candidate evidence card. Return valid JSON only with keys: "
        "final_status, match_score, answer, reasoning, matched_criteria, missing_criteria, evidence_ids, confidence. "
        "final_status must be verified_match or not_verified. "
        "Every factual claim in answer/reasoning must be supported by the cited evidence_ids. "
        "Treat every requirement in the original screening query as mandatory AND logic. "
        "Never return verified_match for a partial match or when any stated requirement lacks evidence. "
        "Mention evidence IDs inline, e.g. ev1. Do not invent missing candidate facts. "
        "If evidence is insufficient, return not_verified."
    )
    user_prompt = (
        f"Original filtering criteria:\n{json.dumps(original_criteria, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate evidence card:\n{json.dumps(evidence_card, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Return JSON only. Keep answer and reasoning to one concise paragraph each."
    )
    try:
        structured = await asyncio.to_thread(
            call_openai_json,
            system_prompt,
            user_prompt,
            model=SCREENING_AUDIT_MODEL,
            use_web=False,
            temperature=0.0,
            timeout=90.0,
        )
        tracker.add_usage(
            SCREENING_AUDIT_MODEL,
            f"{system_prompt}\n\n{user_prompt}",
            json.dumps(structured, ensure_ascii=False, default=str),
            "Shortlist Evidence-Cited Audit",
        )
        if _audit_output_is_evidence_valid(profile_safe, structured):
            structured["answer"] = str(structured.get("answer") or structured.get("reasoning") or "").replace("\n", " ").replace("|", " ").strip()
            structured["reasoning"] = str(structured.get("reasoning") or structured.get("answer") or "").replace("\n", " ").replace("|", " ").strip()
            return structured
        logger.warning("Shortlist audit returned unsupported output for candidate %s", profile_safe.get("id"))
    except Exception as e:
        logger.warning("Evidence-cited audit failed for candidate %s: %s", profile_safe.get("id"), e)
    return _fallback_audit_payload_from_evidence(profile_safe)


def _observed_company_terms_for_expansion(category: str, *, limit: int = 300) -> List[str]:
    category_l = _normalize_search_text(category)
    field_map = {
        "industry": ("industry", "product_service", "business_model"),
        "company attributes": ("industry", "product_service", "business_model", "funding_stage"),
        "customer segments": ("customer_segment",),
        "company culture": ("culture_type",),
    }
    fields = field_map.get(category_l)
    if not fields:
        return []

    counts: Counter = Counter()
    for profile in PROFILES_BY_ID.values():
        for role in profile.get("roles") or []:
            details = role.get("company_details") if isinstance(role.get("company_details"), dict) else {}
            for field in fields:
                raw = details.get(field)
                values = raw if isinstance(raw, list) else [raw]
                for value in values:
                    clean = re.sub(r"\s+", " ", str(value or "")).strip()
                    if clean and clean.lower() not in {"unknown", "none", "n/a"} and len(clean) <= 160:
                        counts[clean] += 1
    return [term for term, _count in counts.most_common(limit)]


async def _expand_keywords_with_llm(values: List[str], category: str, tracker: TokenCostTracker) -> List[str]:
    values = [str(value).strip() for value in values if str(value or "").strip()]
    if not values:
        return []
    observed_terms = _observed_company_terms_for_expansion(category)
    prompt = PromptTemplate(
        input_variables=["keywords", "category", "observed_terms"],
        template="""
        You are an expert business analyst translating recruiter language into evidence terms. Generate a JSON list of semantically equivalent aliases and, for industry/company categories, product or service subcategories that are genuinely entailed by the initial concept.
        Category: {category}
        Prefer exact phrases that actually occur in Observed Database Terms. A broad domain may be evidenced by its characteristic product categories, even when the broad label itself is absent. Include every observed term that is a direct subtype or characteristic product of the initial concept, copying its spelling exactly. Preserve category boundaries: do not broaden a technology category to all of its customers or parent industry (for example, fintech is not equivalent to every bank, insurer, or generic financial-services company). Exclude merely adjacent industries and generic words.
        Return up to 60 precise terms. Preserve the initial keywords. Do not include explanatory text.

        Observed Database Terms: {observed_terms}

        Initial Keywords: {keywords}
        JSON List:
        """
    )
    prompt_text = prompt.format(
        keywords=json.dumps(values),
        category=category,
        observed_terms=json.dumps(observed_terms, ensure_ascii=False),
    )
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt_text), timeout=30.0)
        tracker.add_usage(llm.model_name, prompt_text, response.content, "Keyword Expansion")
        return get_list_from_llm_json(safe_json_loads(response.content, []))
    except asyncio.TimeoutError:
        logger.warning("Shortlist %s keyword expansion timed out; using original terms", category)
        return []


async def _expand_locations_with_llm(values: List[str], tracker: TokenCostTracker) -> List[str]:
    values = [str(value).strip() for value in values if str(value or "").strip()]
    if not values:
        return []
    prompt = PromptTemplate(
        input_variables=["locations"],
        template="""
        You are a geography expert. For the given countries, states, cities, or regions, generate a JSON list containing original names, common abbreviations, and up to 5 major cities or business hubs.

        Initial Locations: {locations}
        JSON List:
        """
    )
    prompt_text = prompt.format(locations=json.dumps(values))
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt_text), timeout=30.0)
        tracker.add_usage(llm.model_name, prompt_text, response.content, "Location Expansion")
        return get_list_from_llm_json(safe_json_loads(response.content, []))
    except asyncio.TimeoutError:
        logger.warning("Shortlist location expansion timed out; using original locations")
        return []


async def _expand_geographies_with_llm(values: List[str], tracker: TokenCostTracker) -> List[str]:
    values = [str(value).strip() for value in values if str(value or "").strip()]
    if not values:
        return []
    prompt = PromptTemplate(
        input_variables=["geographies"],
        template="""
        You are a geography and business market expert. Expand the given market regions or countries into a JSON object with key "geographies" and a list of constituent countries, abbreviations, and major business hubs.
        When expanding APAC, exclude China unless China is explicitly requested.

        Initial Geographies: {geographies}
        JSON Output:
        """
    )
    prompt_text = prompt.format(geographies=json.dumps(values))
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt_text), timeout=30.0)
        tracker.add_usage(llm.model_name, prompt_text, response.content, "Geography Expansion")
        return get_list_from_llm_json(safe_json_loads(response.content, {}))
    except asyncio.TimeoutError:
        logger.warning("Shortlist geography expansion timed out; using original geographies")
        return []


def _sort_strict_shortlist_candidates(candidates: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    sort_criterion = None
    for key in (
        "required_segments",
        "required_functions",
        "required_industries",
        "required_geographies",
        "required_company_details",
        "required_culture_type",
    ):
        if criteria.get(key):
            sort_criterion = key
            break

    def duration_for(candidate: Dict[str, Any]) -> float:
        if not sort_criterion:
            return 0.0
        calculated = candidate.get("calculated_experience") or {}
        value = calculated.get(sort_criterion)
        if isinstance(value, dict):
            return float(value.get("duration") or 0)
        return 0.0

    return sorted(
        candidates,
        key=lambda item: (
            duration_for(item),
            float(item.get("match_score") or 0),
            float(item.get("total_experience_years") or 0),
        ),
        reverse=True,
    )


def _strict_search_should_scan_full_scope(criteria: Dict[str, Any]) -> bool:
    if not isinstance(criteria, dict):
        return False
    if criteria.get("min_total_experience") is not None:
        return True
    if criteria.get("min_people_managed") is not None:
        return True
    if criteria.get("min_tenure_in_latest_role") is not None:
        return True
    if criteria.get("avg_tenure_in_last_n_roles") is not None:
        return True
    if criteria.get("min_function_years"):
        return True
    for key in (
        "required_functions",
        "required_industries",
        "required_segments",
        "required_geographies",
        "required_company_details",
    ):
        criterion = criteria.get(key)
        if isinstance(criterion, dict) and _coerce_positive_float(criterion.get("min_years")):
            return True
    return False


FILTER_PLAN_CRITERIA_KEYS = {
    "required_companies",
    "required_industries",
    "required_functions",
    "required_segments",
    "required_locations",
    "required_geographies",
    "excluded_geographies",
    "required_company_details",
    "required_culture_type",
    "required_keywords",
    "competitors_of",
    "competitor_of",
    "funding_stage_min",
    "min_total_experience",
    "min_people_managed",
    "min_tenure_in_latest_role",
    "avg_tenure_in_last_n_roles",
    "min_function_years",
    "top_n",
}

EMPLOYMENT_SCOPED_CRITERIA_KEYS = {
    "required_companies",
    "required_industries",
    "required_segments",
    "required_company_details",
    "required_culture_type",
    "funding_stage_min",
}


def _executable_criteria_contract() -> Dict[str, Any]:
    common_text_shape = {
        "shape": {"operator": "AND|OR", "values": ["string or typed object"]},
        "evidence": "strict evidence match; missing evidence does not pass",
    }
    return {
        "required_companies": {
            **common_text_shape,
            "value_shape": {"company": "name", "aliases": ["alias"], "employment_scope": "current_employer|any_employer"},
        },
        "competitors_of": {
            "shape": [{"target": "company", "employment_scope": "current_employer|any_employer"}],
            "evidence": "competitors are dynamically resolved then validated against known DB companies",
        },
        "required_functions": {**common_text_shape, "supports_min_years": True},
        "min_function_years": {
            "shape": [{"function": "function", "aliases": ["alias"], "min_years": "number"}],
        },
        "required_industries": {**common_text_shape, "supports_min_years": True, "supports_employment_scope": True},
        "required_segments": {**common_text_shape, "supports_min_years": True, "supports_employment_scope": True},
        "required_company_details": {**common_text_shape, "supports_min_years": True, "supports_employment_scope": True},
        "required_culture_type": {**common_text_shape, "supports_employment_scope": True},
        "required_geographies": {**common_text_shape, "supports_min_years": True, "meaning": "market/territory experience"},
        "required_locations": {**common_text_shape, "meaning": "candidate current/base location"},
        "excluded_geographies": common_text_shape,
        "funding_stage_min": {
            "shape": {"stage": "funding stage", "employment_scope": "current_employer|any_employer"},
            "comparison": "ordered minimum",
        },
        "min_total_experience": {"shape": "number"},
        "min_people_managed": {"shape": "integer"},
        "min_tenure_in_latest_role": {"shape": "number"},
        "avg_tenure_in_last_n_roles": {"shape": {"min_years": "number", "last_n": "integer"}},
        "required_keywords": common_text_shape,
        "top_n": {"shape": "integer or null"},
    }


def _build_schema_manifest(catalog: Dict[str, Any], *, scoped_candidate_count: int = 0) -> Dict[str, Any]:
    compact = compact_evidence_catalog_for_prompt(catalog)
    company_detail_fields = sorted({
        field
        for profile in PROFILES_BY_ID.values()
        for role in (profile.get("roles") or [])
        if isinstance(role.get("company_details"), dict)
        for field in role["company_details"].keys()
    })
    funding_stages = sorted({
        str((role.get("company_details") or {}).get("funding_stage")).strip()
        for profile in PROFILES_BY_ID.values()
        for role in (profile.get("roles") or [])
        if str((role.get("company_details") or {}).get("funding_stage") or "").strip()
    }, key=str.lower)
    geography_fields = [
        "roles.details",
        "roles.location",
        "roles.city",
        "roles.company_details.headquarters",
        "roles.company_details.office_locations",
        "roles.company_details.offices",
        "roles.company_details.locations",
        "roles.company_details.operations",
        "candidate.headline/about/profile market claims",
        "raw_fields keys containing geography/market/region/country/territory/coverage",
    ]
    return {
        "candidate_count_in_scope": scoped_candidate_count,
        "core_candidate_fields": [
            "headline", "about", "location", "city", "candidate_services",
            "extracted_industry", "total_experience_years", "max_people_managed", "raw_fields",
        ],
        "role_fields": ["title", "details", "company", "duration_years", "start_date", "end_date", "location", "city", "company_details"],
        "company_detail_fields": company_detail_fields[:80],
        "funding_stages_seen": funding_stages[:40],
        "geography_evidence_fields": geography_fields,
        "executable_criteria_contract": _executable_criteria_contract(),
        "employment_scoped_criteria": sorted(EMPLOYMENT_SCOPED_CRITERIA_KEYS),
        "known_company_names_sample": sorted({str(name) for name in ALL_COMPANY_NAMES if str(name or "").strip()}, key=str.lower)[:250],
        "db_catalog": compact,
        "policies": {
            "current_location_only_for": ["candidates in X", "based in X", "located in X", "living in X"],
            "market_experience_for": ["X experience", "X market", "worked in X", "sold into X", "covered X"],
            "allowed_company_geography_inference": "Only headquarters/offices/operations/location fields for companies the candidate worked at during that role.",
            "disallowed_geography_inference": "Do not infer from subsidiaries, customer presence, revenue, or generic large-company assumptions.",
        },
    }


def _build_terminology_pack() -> Dict[str, Any]:
    return {
        "base_sales_taxonomy": SALES_TAXONOMY,
        "base_segment_synonyms": SEGMENT_SYNONYMS,
        "base_company_details_taxonomy": COMPANY_DETAILS_TAXONOMY,
        "base_industry_domain_taxonomy": INDUSTRY_DOMAIN_TAXONOMY,
        "base_culture_taxonomy": CULTURE_TAXONOMY,
        "base_geography_country_to_region": GEOGRAPHY_COUNTRY_TO_REGION_MAP,
        "region_to_countries": _region_to_countries(),
        "funding_stage_order": FUNDING_STAGE_RANKS,
        "product_semantics": {
            "outbound_exp": "Sales Development/BDR/SDR/outbound prospecting unless the query explicitly says AE, hunter, new-logo, or net-new closing.",
            "working_for_company": "current employer",
            "worked_at_company": "any past/current employer",
            "competitors": "LLM brainstorm then validate names against known DB company names before filtering.",
            "series_c_and_above": "Series C, Series D, Series E+, growth, pre-IPO, public/acquired/listed when present in schema.",
        },
    }


def _query_company_scope(query: str) -> str:
    query_l = _normalize_search_text(query)
    if re.search(r"\b(?:current|present)\s+(?:company|employer|organisation|organization)\b", query_l):
        return "current_employer"
    if re.search(r"\b(?:working|employed)\s+(?:currently\s+)?(?:for|at|in|with)\b", query_l):
        return "current_employer"
    if re.search(r"\bcurrently\s+(?:working|employed)(?:\s+(?:for|at|in|with))?\b", query_l):
        return "current_employer"
    if re.search(r"\bcurrently\s+(?:for|at|with)\b", query_l):
        return "current_employer"
    if re.search(r"\b(ex[-\s]?|worked|from|previously|past)\s*(?:for|at|in|with)?\b", query_l):
        return "any_employer"
    return "any_employer"


def _query_uses_current_location(query: str) -> bool:
    query_l = _normalize_search_text(query)
    return re.search(r"\b(candidates?|people|person)\s+(?:who\s+are\s+)?(?:in|based in|located in|living in)\b", query_l) is not None


def _query_uses_market_geography(query: str) -> bool:
    query_l = _normalize_search_text(query)
    return any(token in query_l for token in (" market", " experience", "worked in", "sold into", "covered ", "coverage", "territory"))


def _normalize_companies_with_scope(value: Any, query: str) -> Any:
    scope = _query_company_scope(query)
    if not value:
        return value
    if isinstance(value, dict):
        normalized = copy.deepcopy(value)
        normalized.setdefault("employment_scope", scope)
        values = normalized.get("values")
        if isinstance(values, list):
            normalized["values"] = [
                {**item, "employment_scope": item.get("employment_scope") or scope}
                if isinstance(item, dict)
                else {"company": str(item), "employment_scope": scope}
                for item in values
                if str(item or "").strip()
            ]
        return normalized
    items = value if isinstance(value, list) else [value]
    return {
        "operator": "OR",
        "employment_scope": scope,
        "values": [
            {**item, "employment_scope": item.get("employment_scope") or scope}
            if isinstance(item, dict)
            else {"company": str(item), "employment_scope": scope}
            for item in items
            if str(item or "").strip()
        ],
    }


def _append_min_function_items(criteria: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    existing = criteria.get("min_function_years")
    if isinstance(existing, list):
        merged = list(existing)
    elif isinstance(existing, dict):
        merged = [existing]
    else:
        merged = []
    merged.extend(item for item in items if isinstance(item, dict))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in merged:
        function = str(item.get("function") or item.get("value") or item.get("name") or "").strip()
        years = _coerce_positive_float(item.get("min_years") or item.get("years") or item.get("minimum_years"))
        key = (_normalize_search_text(function), years)
        if not function or not years or key in seen:
            continue
        next_item = dict(item)
        next_item["function"] = function
        next_item["min_years"] = years
        deduped.append(next_item)
        seen.add(key)
    criteria["min_function_years"] = deduped


def _function_values_from_required_functions(required_functions: Any) -> List[str]:
    values = get_values_from_criteria(required_functions)
    return [str(value).strip() for value in values if str(value or "").strip()]


def _grouped_min_function_item(required_functions: Any, years: float) -> Optional[Dict[str, Any]]:
    functions = _function_values_from_required_functions(required_functions)
    if not functions or not years:
        return None
    primary = functions[0]
    aliases: List[str] = []
    for function in functions:
        if function != primary:
            aliases.append(function)
        aliases.extend(_criteria_alias_terms(required_functions, function))
    return {
        "function": primary,
        "min_years": years,
        "aliases": sorted({alias for alias in aliases if str(alias or "").strip()}),
    }


def _extract_embedded_function_years(criteria: Dict[str, Any]) -> None:
    required_functions = criteria.get("required_functions")
    if not isinstance(required_functions, dict):
        return
    embedded_items: List[Dict[str, Any]] = []
    values = required_functions.get("values")
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, dict):
                continue
            years = (
                _coerce_positive_float(item.get("min_function_years"))
                or _coerce_positive_float(item.get("min_years"))
                or _coerce_positive_float(item.get("years"))
                or _coerce_positive_float(item.get("minimum_years"))
            )
            function = str(item.get("function") or item.get("value") or item.get("name") or item.get("term") or "").strip()
            if function and years:
                embedded_items.append({
                    "function": function,
                    "min_years": years,
                    "aliases": item.get("aliases") or item.get("expanded_terms") or [],
                })
    top_level_years = (
        _coerce_positive_float(required_functions.get("min_years"))
        or _coerce_positive_float(required_functions.get("min_function_years"))
    )
    if top_level_years:
        grouped = _grouped_min_function_item(required_functions, top_level_years)
        if grouped:
            embedded_items.append(grouped)
    _append_min_function_items(criteria, embedded_items)


def _set_geography_min_years_from_context(criteria: Dict[str, Any], context: Any, years: float) -> bool:
    required_geographies = criteria.get("required_geographies")
    if not required_geographies or not years:
        return False
    context_text = _normalize_search_text(context)
    values = _function_values_from_required_functions(required_geographies)
    if not values:
        return False
    if context_text and not any(
        _term_matches_text(term, context_text)
        for value in values
        for term in _geography_match_terms(value, required_geographies)
    ):
        return False
    if isinstance(required_geographies, dict):
        current = _coerce_positive_float(required_geographies.get("min_years"))
        required_geographies["min_years"] = max(current or 0, years)
    else:
        criteria["required_geographies"] = {"operator": "OR", "values": values, "min_years": years}
    return True


def _duration_rule_field(item: Dict[str, Any]) -> str:
    return _normalize_search_text(
        item.get("field")
        or item.get("key")
        or item.get("dimension")
        or item.get("type")
        or item.get("name")
        or ""
    )


def _duration_rule_years(item: Dict[str, Any]) -> Optional[float]:
    return (
        _coerce_positive_float(item.get("value"))
        or _coerce_positive_float(item.get("min_years"))
        or _coerce_positive_float(item.get("min_function_years"))
        or _coerce_positive_float(item.get("years"))
        or _coerce_positive_float(item.get("minimum_years"))
    )


def _ensure_criterion_with_min_years(criteria: Dict[str, Any], key: str, value: Any, years: float) -> None:
    value_text = str(value or "").strip()
    existing = criteria.get(key)
    if isinstance(existing, dict):
        values = get_values_from_criteria(existing)
        if value_text and value_text not in values:
            values.append(value_text)
        existing["operator"] = existing.get("operator") or "OR"
        existing["values"] = values
        current = _coerce_positive_float(existing.get("min_years"))
        existing["min_years"] = max(current or 0, years)
    elif isinstance(existing, list):
        values = [str(item).strip() for item in existing if str(item or "").strip()]
        if value_text and value_text not in values:
            values.append(value_text)
        criteria[key] = {"operator": "OR", "values": values, "min_years": years}
    else:
        values = [value_text] if value_text else []
        criteria[key] = {"operator": "OR", "values": values, "min_years": years}


def _normalize_numeric_keyed_criterion_shape(criteria: Dict[str, Any], key: str) -> None:
    criterion = criteria.get(key)
    if not isinstance(criterion, dict) or isinstance(criterion.get("values"), list):
        return
    ignored_keys = {
        "operator", "scope", "employment_scope", "min_years", "min_function_years",
        "years", "minimum_years", "field", "dimension", "type", "context",
        "aliases", "expanded_terms", "accepted_terms", "countries", "regions",
        "shape", "value_shape", "evidence", "meaning", "comparison",
        "supports_min_years", "supports_employment_scope",
    }
    keyed_values: List[str] = []
    value_terms: List[str] = []
    max_years = _coerce_positive_float(criterion.get("min_years")) or 0
    for item_key, item_value in criterion.items():
        if item_key in ignored_keys or item_value in (None, "", [], {}):
            continue
        item_years = _coerce_positive_float(item_value)
        if item_years:
            keyed_values.append(str(item_key))
            max_years = max(max_years, item_years)
        elif isinstance(item_value, list):
            for entry in item_value:
                if isinstance(entry, str) and entry.strip():
                    value_terms.append(entry.strip())
                elif isinstance(entry, dict):
                    value_terms.extend(get_values_from_criteria(entry))
        elif isinstance(item_value, str):
            value_terms.append(item_value)
    if not keyed_values:
        return
    values = sorted({str(value).strip() for value in value_terms + keyed_values if str(value or "").strip()}, key=str.lower)
    criterion["operator"] = criterion.get("operator") or "OR"
    criterion["values"] = values
    if max_years and not _coerce_positive_float(criterion.get("min_years")):
        criterion["min_years"] = max_years


def _sanitize_planner_criterion(value: Any) -> Any:
    """Strip schema-reference metadata accidentally echoed by the planner."""
    if not isinstance(value, dict):
        return value

    metadata_keys = {
        "shape", "value_shape", "evidence", "meaning", "comparison",
        "supports_min_years", "supports_employment_scope",
    }
    shape = value.get("shape")
    if isinstance(shape, dict) and any(
        key in shape for key in ("values", "value", "stage", "company", "target", "min_years")
    ):
        sanitized = copy.deepcopy(shape)
        for key in ("scope", "employment_scope", "min_years"):
            if value.get(key) not in (None, "", [], {}):
                sanitized.setdefault(key, value.get(key))
        return sanitized

    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in metadata_keys
    }


def _query_years_near_terms(query: str, terms: List[str], context_terms: List[str]) -> Optional[float]:
    query_l = _normalize_search_text(query)
    if not query_l:
        return None
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", query_l):
        start = max(0, match.start() - 90)
        end = min(len(query_l), match.end() + 120)
        window = query_l[start:end]
        if terms and not any(_term_matches_text(term, window) for term in terms):
            continue
        if context_terms and not any(_term_matches_text(term, window) for term in context_terms):
            continue
        return float(match.group(1))
    return None


def _query_years_followed_by_terms(
    query: str,
    terms: List[str],
    context_terms: List[str],
    years: Optional[float] = None,
) -> bool:
    query_l = _normalize_search_text(query)
    if not query_l:
        return False
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", query_l):
        matched_years = _coerce_positive_float(match.group(1))
        if years and matched_years and abs(matched_years - years) > 0.01:
            continue
        window = query_l[match.end(): min(len(query_l), match.end() + 140)]
        if terms and not any(_term_matches_text(term, window) for term in terms):
            continue
        if context_terms and not any(_term_matches_text(term, window) for term in context_terms):
            continue
        return True
    return False


def _apply_query_scoped_duration_hints(criteria: Dict[str, Any], query: str) -> None:
    hint_configs = [
        ("required_industries", ["industry", "industries", "vertical", "domain"]),
        ("required_company_details", ["saas", "software", "product", "platform", "company", "business model"]),
        ("required_segments", ["segment", "customer", "customers", "enterprise", "mid market", "smb", "mm"]),
        ("required_geographies", ["market", "geography", "geo", "region", "territory", "selling into", "covered", "covering"]),
    ]
    for key, context_terms in hint_configs:
        criterion = criteria.get(key)
        if not isinstance(criterion, dict) or _coerce_positive_float(criterion.get("min_years")):
            continue
        values = [str(value).strip() for value in get_values_from_criteria(criterion) if str(value or "").strip()]
        if not values:
            continue
        terms = sorted({term for value in values for term in _criterion_match_terms(value, key, criterion)})
        if key == "required_geographies":
            terms = sorted({term for value in values for term in _geography_match_terms(value, criterion)})
        years = _query_years_near_terms(query, terms, context_terms)
        if years:
            criterion["min_years"] = years


def _prune_function_years_shadowed_by_scoped_tenure(criteria: Dict[str, Any], query: str) -> None:
    items = criteria.get("min_function_years")
    if not isinstance(items, list) or not items:
        return

    non_function_configs = [
        ("required_industries", ["industry", "industries", "vertical", "domain"]),
        ("required_company_details", ["saas", "software", "product", "platform", "company", "business model", "industry"]),
        ("required_segments", ["segment", "customer", "customers", "enterprise", "mid market", "smb", "mm", "selling to"]),
        ("required_geographies", ["market", "geography", "geo", "region", "territory", "selling into", "covered", "covering"]),
    ]
    scoped_year_terms: List[Tuple[float, str, List[str], List[str]]] = []
    for key, context_terms in non_function_configs:
        criterion = criteria.get(key)
        if not isinstance(criterion, dict):
            continue
        years = _coerce_positive_float(criterion.get("min_years"))
        if not years:
            continue
        values = [str(value).strip() for value in get_values_from_criteria(criterion) if str(value or "").strip()]
        if key == "required_geographies":
            terms = sorted({term for value in values for term in _geography_match_terms(value, criterion)})
        else:
            terms = sorted({term for value in values for term in _criterion_match_terms(value, key, criterion)})
        if terms:
            scoped_year_terms.append((years, key, terms, context_terms))

    if not scoped_year_terms:
        return

    kept: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        years = _coerce_positive_float(item.get("min_years") or item.get("min_function_years"))
        if not years:
            kept.append(item)
            continue
        function_terms = set(_criterion_match_terms(str(item.get("function") or ""), "required_functions", criteria.get("required_functions")))
        aliases = item.get("aliases") or item.get("expanded_terms") or []
        if isinstance(aliases, str):
            aliases = re.split(r"[,;|]", aliases)
        function_terms.update(_normalize_search_text(alias) for alias in aliases if str(alias or "").strip())
        function_terms = {term for term in function_terms if term}
        direct_function_scope = _query_years_followed_by_terms(
            query,
            sorted(function_terms),
            ["role", "roles", "function", "sales development", "business development", "bdr", "sdr", "outbound", "prospecting"],
            years,
        )
        shadowed_by_non_function_scope = any(
            abs(scoped_years - years) <= 0.01
            and _query_years_followed_by_terms(query, terms, context_terms, years)
            for scoped_years, _key, terms, context_terms in scoped_year_terms
        )
        if shadowed_by_non_function_scope and not direct_function_scope:
            continue
        kept.append(item)

    if kept:
        criteria["min_function_years"] = kept
    else:
        criteria.pop("min_function_years", None)


def _enforce_explicit_query_requirements(criteria: Dict[str, Any], query: str) -> None:
    """Add requirements stated plainly in the query even if the LLM omits them.

    Criteria categories are evaluated with AND semantics by the strict scorer;
    values inside one category remain aliases/alternatives (OR).
    """
    query_l = _normalize_search_text(query)
    if not query_l:
        return

    # Explicit role acronyms must remain executable function requirements.
    function_terms = []
    if re.search(r"\bbdrs?\b", query_l):
        function_terms.append("BDR")
    if re.search(r"\bsdrs?\b", query_l):
        function_terms.append("SDR")
    if function_terms:
        existing = criteria.get("required_functions")
        existing_values = _criteria_values_for_search(criteria, "required_functions")
        merged_function = copy.deepcopy(existing) if isinstance(existing, dict) else {}
        merged_function["operator"] = "OR"
        merged_function["values"] = list(dict.fromkeys(existing_values + function_terms))
        criteria["required_functions"] = merged_function

    # "US experience" and equivalent wording means market experience, never
    # merely an employer HQ or the candidate's current location.
    if _query_uses_market_geography(query_l):
        explicit_geographies = []
        geography_patterns = (
            (r"\b(?:u\.?s\.?a?|united states)\b", "US"),
            (r"\bnorth america(?:n)?\b", "North America"),
            (r"\bapac\b|\basia[ -]pacific\b", "APAC"),
            (r"\bemea\b", "EMEA"),
            (r"\blatam\b|\blatin america\b", "LATAM"),
        )
        for pattern, canonical in geography_patterns:
            if re.search(pattern, query_l):
                explicit_geographies.append(canonical)
        if explicit_geographies:
            existing = criteria.get("required_geographies")
            existing_values = _criteria_values_for_search(criteria, "required_geographies")
            merged_geography = copy.deepcopy(existing) if isinstance(existing, dict) else {}
            merged_geography["operator"] = "OR"
            merged_geography["values"] = list(dict.fromkeys(existing_values + explicit_geographies))
            criteria["required_geographies"] = merged_geography

    # Bind years to an explicitly named company type/domain. This prevents
    # "15+ years in SaaS" from becoming 15 years in the BDR function.
    scoped_duration_patterns = (
        (r"\bsaas\b|\bsoftware as a service\b", "SaaS"),
        (r"\bfintech\b", "fintech"),
    )
    scoped_years: List[Tuple[float, str]] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", query_l):
        years = float(match.group(1))
        after = query_l[match.end(): min(len(query_l), match.end() + 80)]
        if not re.match(r"\s*(?:of\s+experience\s+)?(?:in|with|selling\s+(?:in|to))\b", after):
            continue
        for pattern, canonical in scoped_duration_patterns:
            if re.search(pattern, after):
                scoped_years.append((years, canonical))
                break

    for years, value in scoped_years:
        target_key = "required_company_details"
        for key in ("required_company_details", "required_industries"):
            criterion = criteria.get(key)
            terms = [
                term
                for existing_value in get_values_from_criteria(criterion)
                for term in _criterion_match_terms(str(existing_value), key, criterion)
            ]
            if any(_term_matches_text(term, value) or _term_matches_text(value, term) for term in terms):
                target_key = key
                break
        _ensure_criterion_with_min_years(criteria, target_key, value, years)

        min_function_items = criteria.get("min_function_years")
        if isinstance(min_function_items, list):
            kept = [
                item
                for item in min_function_items
                if not isinstance(item, dict)
                or abs(float(_coerce_positive_float(item.get("min_years") or item.get("min_function_years")) or 0) - years) > 0.01
            ]
            if kept:
                criteria["min_function_years"] = kept
            else:
                criteria.pop("min_function_years", None)

    # Keep tenure attached to outbound work instead of broadening it to all BDR
    # experience. Inbound qualification must not satisfy this requirement.
    outbound_years: Optional[float] = None
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", query_l):
        after = query_l[match.end(): min(len(query_l), match.end() + 100)]
        if not re.match(r"\s*(?:of\s+(?:exp|experience)\s+)?(?:in|with|doing)\b", after):
            continue
        if re.search(r"\boutbound\b", after) and re.search(
            r"\b(?:lead\s+qualification|qualif(?:y|ying|ication)|prospect(?:ing)?|lead\s+generation)\b",
            after,
        ):
            outbound_years = float(match.group(1))
            break

    if outbound_years:
        existing_items = criteria.get("min_function_years")
        if not isinstance(existing_items, list):
            existing_items = [existing_items] if isinstance(existing_items, dict) else []
        kept_items = []
        for item in existing_items:
            item_years = _coerce_positive_float(item.get("min_years") or item.get("min_function_years"))
            item_text = _normalize_search_text(
                " ".join(
                    [
                        str(item.get("function") or item.get("value") or ""),
                        *[str(alias) for alias in (item.get("aliases") or [])],
                    ]
                )
            )
            if item_years and abs(item_years - outbound_years) <= 0.01 and "outbound" not in item_text:
                continue
            kept_items.append(item)
        criteria["min_function_years"] = kept_items
        _append_min_function_items(
            criteria,
            [
                {
                    "function": "Outbound lead qualification",
                    "min_years": outbound_years,
                    "aliases": [
                        "outbound lead qualification",
                        "outbound prospecting",
                        "outbound lead generation",
                        "cold outreach",
                        "cold calling",
                    ],
                }
            ],
        )


def _coerce_filter_plan_to_criteria(plan: Dict[str, Any], query: str) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}

    source = plan.get("filter_plan") if isinstance(plan.get("filter_plan"), dict) else plan
    criteria: Dict[str, Any] = {}

    hard_filters = source.get("hard_filters") if isinstance(source.get("hard_filters"), dict) else {}
    for container in (source, hard_filters):
        for key in FILTER_PLAN_CRITERIA_KEYS:
            if key in container and container.get(key) not in (None, "", [], {}):
                criteria[key] = _sanitize_planner_criterion(container.get(key))

    for key in (
        "required_industries",
        "required_functions",
        "required_segments",
        "required_locations",
        "required_geographies",
        "excluded_geographies",
        "required_company_details",
        "required_culture_type",
        "required_keywords",
    ):
        value = criteria.get(key)
        if isinstance(value, list):
            criteria[key] = {"operator": "OR", "values": value}
        elif isinstance(value, str):
            criteria[key] = {"operator": "OR", "values": [value]}
        elif isinstance(value, dict):
            _normalize_numeric_keyed_criterion_shape(criteria, key)
            criteria[key]["operator"] = "OR"

    duration_rules = source.get("duration_rules")
    if isinstance(duration_rules, dict):
        for key in ("min_total_experience", "min_tenure_in_latest_role", "avg_tenure_in_last_n_roles", "min_function_years"):
            if key in duration_rules and duration_rules.get(key) not in (None, "", [], {}):
                criteria[key] = copy.deepcopy(duration_rules[key])
    elif isinstance(duration_rules, list):
        min_function_items: List[Dict[str, Any]] = []
        for item in duration_rules:
            if not isinstance(item, dict):
                continue
            dimension = _duration_rule_field(item)
            years = _duration_rule_years(item)
            if not years:
                continue
            if item.get("geography"):
                geography_value = str(item.get("geography") or "").strip()
                existing_geo = criteria.get("required_geographies")
                if not existing_geo:
                    criteria["required_geographies"] = {"operator": "OR", "values": [geography_value], "min_years": years}
                else:
                    if isinstance(existing_geo, dict):
                        values = _function_values_from_required_functions(existing_geo)
                        if geography_value and geography_value not in values:
                            values.append(geography_value)
                        existing_geo["values"] = values
                    _set_geography_min_years_from_context(criteria, geography_value, years)
            elif "function" in dimension or item.get("function"):
                function = item.get("function") or item.get("target") or item.get("role")
                if function:
                    function_values = [str(function).strip()]
                    for function_value in function_values:
                        min_function_items.append({
                            "function": function_value,
                            "min_years": years,
                            "aliases": item.get("aliases") or item.get("expanded_terms") or _criteria_alias_terms(criteria.get("required_functions"), function_value),
                        })
                else:
                    grouped = _grouped_min_function_item(criteria.get("required_functions"), years)
                    if grouped:
                        min_function_items.append(grouped)
            elif "total" in dimension:
                criteria["min_total_experience"] = years
            elif "geograph" in dimension or "market" in dimension or "territor" in dimension:
                _set_geography_min_years_from_context(criteria, item.get("context") or item.get("value_name") or "", years)
            elif "industr" in dimension or item.get("industry"):
                value = item.get("industry") or item.get("target") or item.get("value_name") or item.get("value")
                _ensure_criterion_with_min_years(criteria, "required_industries", value, years)
            elif "segment" in dimension or "customer" in dimension or item.get("segment"):
                value = item.get("segment") or item.get("customer_segment") or item.get("target") or item.get("value_name") or item.get("value")
                _ensure_criterion_with_min_years(criteria, "required_segments", value, years)
            elif "company detail" in dimension or "company_detail" in dimension or "product" in dimension or "saas" in dimension or item.get("company_detail") or item.get("product_service"):
                value = item.get("company_detail") or item.get("product_service") or item.get("target") or item.get("value_name") or item.get("value")
                _ensure_criterion_with_min_years(criteria, "required_company_details", value, years)
            elif "latest" in dimension and "role" in dimension:
                if not _set_geography_min_years_from_context(criteria, item.get("context"), years):
                    criteria["min_tenure_in_latest_role"] = years
        if min_function_items:
            _append_min_function_items(criteria, min_function_items)

    _extract_embedded_function_years(criteria)
    _apply_query_scoped_duration_hints(criteria, query)

    if criteria.get("min_function_years") and not isinstance(criteria.get("min_function_years"), list):
        raw_min_function = criteria.get("min_function_years")
        if isinstance(raw_min_function, dict):
            criteria["min_function_years"] = [raw_min_function]
        elif criteria.get("required_functions") and isinstance(criteria.get("required_functions"), dict):
            years = _coerce_positive_float(raw_min_function)
            if years:
                grouped = _grouped_min_function_item(criteria.get("required_functions"), years)
                if grouped:
                    _append_min_function_items(criteria, [grouped])
            else:
                criteria.pop("min_function_years", None)
        else:
            criteria.pop("min_function_years", None)

    _prune_function_years_shadowed_by_scoped_tenure(criteria, query)

    company_scope = source.get("company_scope")
    inferred_company_scope = _query_company_scope(query)
    explicit_company_scope = None
    if isinstance(company_scope, dict):
        explicit_company_scope = company_scope.get("employment_scope") or company_scope.get("scope")

    effective_company_scope = explicit_company_scope or inferred_company_scope
    for scoped_key in EMPLOYMENT_SCOPED_CRITERIA_KEYS:
        criterion = criteria.get(scoped_key)
        if not criterion:
            continue
        if scoped_key == "required_companies":
            criterion = _normalize_companies_with_scope(criterion, query)
            criteria[scoped_key] = criterion
        if isinstance(criterion, dict):
            criterion["employment_scope"] = criterion.get("employment_scope") or effective_company_scope
            for item in criterion.get("values") or []:
                if isinstance(item, dict):
                    item["employment_scope"] = item.get("employment_scope") or criterion["employment_scope"]

    if criteria.get("required_companies"):
        criteria["required_companies"] = _normalize_companies_with_scope(criteria["required_companies"], query)

    competitor_key = "competitors_of" if criteria.get("competitors_of") else "competitor_of"
    if criteria.get(competitor_key):
        competitor_items = _criteria_objects(criteria.get(competitor_key))
        scope = _query_company_scope(query)
        normalized_items = []
        for item in competitor_items:
            item = dict(item)
            item["target"] = item.get("target") or item.get("company") or item.get("value") or item.get("name")
            item["employment_scope"] = item.get("employment_scope") or scope
            normalized_items.append(item)
        criteria["competitors_of"] = normalized_items
        criteria.pop("competitor_of", None)

    if criteria.get("funding_stage_min") and isinstance(criteria["funding_stage_min"], dict):
        criteria["funding_stage_min"].setdefault("employment_scope", _query_company_scope(query))

    if (
        criteria.get("required_locations")
        and _query_uses_market_geography(query)
        and not _query_uses_current_location(query)
    ):
        location_values = _criteria_values_for_search(criteria, "required_locations")
        existing_geo = _criteria_values_for_search(criteria, "required_geographies")
        criteria["required_geographies"] = {"operator": "OR", "values": existing_geo + location_values}
        criteria.pop("required_locations", None)

    semantic_terms = source.get("semantic_terms")
    if semantic_terms and not any(criteria.get(key) for key in FILTER_PLAN_CRITERIA_KEYS if key != "top_n"):
        terms = semantic_terms if isinstance(semantic_terms, list) else [semantic_terms]
        criteria["required_keywords"] = {"operator": "OR", "values": [str(term) for term in terms if str(term or "").strip()]}

    if re.search(r"\ball\b", _normalize_search_text(query)):
        criteria["top_n"] = 0
    elif not any(word in _normalize_search_text(query) for word in ["top", "one", "maximum", "best"]):
        criteria.pop("top_n", None)

    _enforce_explicit_query_requirements(criteria, query)

    criteria["_filter_plan_debug"] = {
        "debug_reasoning": source.get("debug_reasoning"),
        "geography_policy": source.get("geography_policy"),
        "sort_policy": source.get("sort_policy"),
    }
    return {key: value for key, value in criteria.items() if value not in (None, "", [], {})}


async def _generate_schema_aware_filter_plan(
    query: str,
    schema_manifest: Dict[str, Any],
    terminology_pack: Dict[str, Any],
    tracker: TokenCostTracker,
) -> Dict[str, Any]:
    prompt = PromptTemplate(
        input_variables=["query", "schema_manifest_json", "terminology_pack_json"],
        template="""
You are a senior product-minded recruiting search planner. Generate an executable filter plan for a shortlist engine.

You must be brave enough to produce filters for recruiter queries, but you must only use the schema and evidence policies below. Static mappings are only the base terminology; use product judgment to map recruiter language dynamically.
The schema manifest is the complete discoverable database/raw-field schema and includes the exact executable criteria contract. Inspect it before planning. Map novel recruiter language to the closest evidence-backed executable dimensions instead of giving up merely because wording is unfamiliar. Never invent a field, criterion, candidate fact, or company fact.

Schema manifest:
{schema_manifest_json}

Terminology pack:
{terminology_pack_json}

Rules:
- Return JSON only.
- Use only these executable criteria keys when possible: required_companies, competitors_of, required_functions, min_function_years, required_industries, required_segments, required_company_details, required_culture_type, required_geographies, required_locations, excluded_geographies, funding_stage_min, min_total_experience, min_people_managed, min_tenure_in_latest_role, avg_tenure_in_last_n_roles, required_keywords, top_n.
- In hard_filters, emit only executable values such as operator, values, min_years, stage, and employment_scope. Never copy schema-reference metadata keys such as shape, value_shape, evidence, meaning, comparison, or supports_* into hard_filters.
- Current/base location filters only for phrases like "candidates in X", "based in X", "located in X".
- Market/geography experience filters for phrases like "X experience", "X market", "worked in X", "sold into X", "covered X".
- APAC/EMEA/etc. are market regions. Expand them through geography policy; do not treat them as candidate current location.
- A country query can match explicit region evidence when the country belongs to that region.
- Company geography can be inferred only from headquarters/offices/operations/location fields for companies the candidate worked at. Never infer from subsidiaries, customer presence, revenue, or broad company assumptions.
- "working for/at/in COMPANY" means current_employer. "worked at/from/ex COMPANY" means any_employer.
- "current company/employer", "present company/employer", and company attributes attached to "currently working" mean current_employer.
- employment_scope applies to required companies, industries, customer segments, company details/business model/product, culture, and funding stage. Preserve that scope on every affected criterion.
- "working for COMPANY competitors" means current_employer at a validated competitor.
- "Series C and above" means funding_stage_min Series C with ordered funding comparison.
- "outbound exp" should map to Sales Development/BDR/SDR/outbound prospecting unless the query explicitly asks AE/hunting/new-logo closing.
- Function-specific years must become min_function_years or min_years on required_functions.
- Years attached to an industry, domain, company type, product, service, or business model must become min_years on required_industries or required_company_details. Example: "5 years in SaaS/software/fintech" is not min_function_years.
- Years attached to a customer segment must become min_years on required_segments. Example: "3 years selling enterprise/MM/SMB" is segment tenure.
- Years attached to a market, territory, or geography must become min_years on required_geographies. Example: "2 years selling into APAC" is market tenure; current location alone cannot satisfy it.
- Total experience can satisfy only min_total_experience; it cannot satisfy function, industry, segment, or geography tenure.

Return shape:
{{
  "filter_plan": {{
    "hard_filters": {{}},
    "semantic_terms": [],
    "duration_rules": [],
    "company_scope": {{"employment_scope": "current_employer|any_employer"}},
    "geography_policy": {{"use_current_location": false, "expand_regions": true, "allow_country_region_reverse_match": true}},
    "sort_policy": {{"primary": "relevant_duration", "secondary": ["match_score", "total_experience"]}},
    "top_n": null,
    "debug_reasoning": "short explanation"
  }}
}}

User query: {query}
JSON:
        """
    )
    prompt_text = prompt.format(
        query=query,
        schema_manifest_json=json.dumps(schema_manifest, ensure_ascii=False, indent=2, default=str),
        terminology_pack_json=json.dumps(terminology_pack, ensure_ascii=False, indent=2, default=str),
    )
    response = await llm.ainvoke(prompt_text)
    tracker.add_usage(llm.model_name, prompt_text, response.content, "Schema-aware Filter Plan Generation")
    structured = safe_json_loads(response.content, {})
    return structured if isinstance(structured, dict) else {}


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
    use_web_search: bool = False,
) -> AsyncIterator[Any]:
    if not is_cache_initialized():
        logger.info("Cache empty, initializing on demand...")
        initialize_cache()

    screening_r = (screening_role or "").strip().lower()
    web_enabled = bool(use_web_search) and SCREENING_WEB_SEARCH_DEFAULT
    normalized_query = normalize_query_with_llm(query)
    normalized_query_lower = normalized_query.lower()

    yield "Generating schema-aware filter plan..."
    try:
        active_candidate_count = len([p for p in PROFILES_BY_ID.values() if not p.get("is_archived")])
        evidence_catalog = build_db_evidence_catalog(profiles=list(PROFILES_BY_ID.values()))
        schema_manifest = _build_schema_manifest(evidence_catalog, scoped_candidate_count=active_candidate_count)
        terminology_pack = _build_terminology_pack()
        raw_filter_plan = await _generate_schema_aware_filter_plan(normalized_query, schema_manifest, terminology_pack, tracker)
        criteria = _coerce_filter_plan_to_criteria(raw_filter_plan, normalized_query)
        logger.info("SHORTLIST schema_manifest_summary=%s", json.dumps({
            "source": schema_manifest.get("db_catalog", {}).get("source"),
            "candidate_count_in_scope": schema_manifest.get("candidate_count_in_scope"),
            "company_detail_fields": schema_manifest.get("company_detail_fields", [])[:20],
            "raw_field_count": len(schema_manifest.get("db_catalog", {}).get("raw_field_keys", []) or []),
        }, default=str))
        logger.info("SHORTLIST terminology_pack_base=%s", json.dumps({
            "sales_taxonomy": terminology_pack.get("base_sales_taxonomy"),
            "region_to_countries": terminology_pack.get("region_to_countries"),
            "funding_stage_order": terminology_pack.get("funding_stage_order"),
        }, default=str))
        logger.info("SHORTLIST generated_filter_plan=%s", json.dumps(raw_filter_plan, ensure_ascii=False, default=str))
        logger.info("SHORTLIST executable_criteria=%s", json.dumps({k: v for k, v in criteria.items() if not k.startswith("_")}, ensure_ascii=False, default=str))
        logger.info("SHORTLIST web_search_enabled=%s requested=%s", web_enabled, bool(use_web_search))
        if not criteria:
            if any(token in normalized_query_lower for token in ("worked at", "worked in", "from ")):
                company_name = re.split(r"\b(?:worked at|worked in|from|at)\b", normalized_query, flags=re.IGNORECASE)[-1].strip()
                criteria = {"required_companies": _normalize_companies_with_scope([company_name], normalized_query)} if company_name else {}
            else:
                criteria = {"required_keywords": {"operator": "OR", "values": [normalized_query]}}
        if not criteria:
            raise ValueError("Failed to generate filter plan from query")
    except Exception as e:
        logger.error(f"Error generating filter plan: {e}", exc_info=True)
        yield f"Error analyzing query: {e}"
        return

    final_competitors: List[str] = []
    competitor_values = criteria.get("competitors_of") or criteria.get("competitor_of")
    if competitor_values:
        competitor_items = _criteria_objects(competitor_values)
        target = ""
        competitor_scope = _query_company_scope(normalized_query)
        first_item = competitor_items[0] if competitor_items else None
        if isinstance(first_item, dict):
            target = str(first_item.get("target") or first_item.get("company") or first_item.get("value") or "").strip()
            competitor_scope = str(first_item.get("employment_scope") or first_item.get("scope") or competitor_scope)
        else:
            target = str(first_item or "").strip()

        if target:
            task = "identify all direct competitors for the given company"
            if "top" in normalized_query_lower and criteria.get("top_n"):
                task = f"identify the top {criteria['top_n']} direct competitors for the given company"
            yield f"Identifying competitors for {target}..."

            try:
                if web_enabled:
                    yield "Researching company facts..."
                    company_fact_criteria = copy.deepcopy(criteria)
                    company_fact_criteria["competitor_of"] = [
                        {
                            "target": target,
                            "employment_scope": competitor_scope,
                        }
                    ]
                    company_fact_criteria.pop("competitors_of", None)
                    web_enriched = await enrich_criteria_with_company_web_facts(
                        normalized_query,
                        company_fact_criteria,
                        tracker,
                    )
                    web_facts = web_enriched.get("_web_company_facts") if isinstance(web_enriched.get("_web_company_facts"), dict) else {}
                    web_competitor_names: List[str] = []
                    for item in web_facts.get("competitors") or []:
                        if not isinstance(item, dict):
                            continue
                        web_target = str(item.get("target") or "").strip()
                        if web_target and not _company_matches(web_target, target):
                            continue
                        raw_names = item.get("companies") or item.get("competitors") or []
                        if isinstance(raw_names, str):
                            raw_names = re.split(r"[,;|]", raw_names)
                        web_competitor_names.extend(str(name).strip() for name in raw_names if str(name or "").strip())
                    final_competitors = _validate_company_names_against_db(web_competitor_names, exclude=target)

                if web_enabled and not final_competitors:
                    fallback_system_prompt = (
                        "You identify current direct competitors for recruiting search. Use live web knowledge. "
                        "Return JSON only with key competitors. Competitors may be strings or objects with name. "
                        "Prefer close category competitors over generic software companies."
                    )
                    fallback_user_prompt = (
                        f"Find direct competitors of {target} for this recruiting query: {normalized_query}\n"
                        "Return companies in the same product category and buyer segment. "
                        "For CleverTap-like companies, include customer engagement, retention, marketing automation, mobile engagement, and cross-channel messaging platforms."
                    )
                    fallback_structured = await asyncio.to_thread(
                        call_openai_json,
                        fallback_system_prompt,
                        fallback_user_prompt,
                        model=SCREENING_REASONING_MODEL,
                        use_web=True,
                        web_search_tool=SCREENING_WEB_SEARCH_TOOL,
                        web_search_context_size=SCREENING_WEB_SEARCH_CONTEXT_SIZE,
                        temperature=0.0,
                        timeout=90.0,
                    )
                    tracker.add_usage(
                        SCREENING_REASONING_MODEL,
                        f"{fallback_system_prompt}\n\n{fallback_user_prompt}",
                        json.dumps(fallback_structured),
                        "Competitor Web Identification",
                    )
                    fallback_names = []
                    if isinstance(fallback_structured, dict):
                        fallback_names = get_list_from_llm_json(fallback_structured.get("competitors") or fallback_structured.get("companies") or [])
                    final_competitors = _validate_company_names_against_db(fallback_names, exclude=target)

                if not final_competitors:
                    competitor_prompt = PromptTemplate(
                        input_variables=["company_name", "competitor_task"],
                        template="""
                        You are an expert business analyst with current market knowledge. Your task is to {competitor_task}.
                        Return a JSON list of company names only. Include close category competitors, not generic large software vendors.

                        Target Company: {company_name}
                        JSON List:
                        """
                    )
                    competitor_prompt_text = competitor_prompt.format(company_name=target, competitor_task=task)
                    response = await specialist_llm.ainvoke(competitor_prompt_text)
                    tracker.add_usage(specialist_llm.model_name, competitor_prompt_text, response.content, "Competitor Identification")
                    llm_competitors = get_list_from_llm_json(safe_json_loads(response.content, response.content))
                    final_competitors = _validate_company_names_against_db(llm_competitors, exclude=target)
            except Exception as e:
                logger.error("Competitor identification failed: %s", e)
                yield "There was an issue identifying competitors."

            if not final_competitors:
                yield "Could not identify any valid competitors from the database. Halting search."
                yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
                return

            yield f"Found competitors: {', '.join(final_competitors)}"
            criteria["required_companies"] = {
                "operator": "OR",
                "employment_scope": competitor_scope,
                "values": [
                    {"company": company, "employment_scope": competitor_scope, "source": f"competitor_of:{target}"}
                    for company in final_competitors
                ],
            }
            criteria["_competitor_resolution"] = {
                "target": target,
                "validated_companies": final_competitors,
                "employment_scope": competitor_scope,
            }
            logger.info("SHORTLIST competitor_validation=%s", json.dumps(criteria["_competitor_resolution"], ensure_ascii=False, default=str))
            criteria.pop("competitors_of", None)
            criteria.pop("competitor_of", None)

    try:
        company_keywords = criteria.pop("required_companies", [])
        competitor_search_was_run = bool(final_competitors)

        if not competitor_search_was_run and criteria.get("required_industries"):
            yield "Expanding keywords..."
            values = _criteria_values_for_search(criteria, "required_industries")
            base_expanded: List[str] = []
            unknown_industries: List[str] = []
            for value in values:
                canonical = _normalize_search_text(value)
                if canonical in INDUSTRY_DOMAIN_TAXONOMY:
                    base_expanded.extend(INDUSTRY_DOMAIN_TAXONOMY[canonical])
                else:
                    unknown_industries.append(value)
            dynamic_expanded = (
                await _expand_keywords_with_llm(unknown_industries, "Industry", tracker)
                if unknown_industries
                else []
            )
            _set_criteria_values(
                criteria,
                "required_industries",
                values + base_expanded + dynamic_expanded + unknown_industries,
            )

        if criteria.get("required_functions"):
            values = _criteria_values_for_search(criteria, "required_functions")
            expanded: List[str] = []
            unknown: List[str] = []
            for value in values:
                if value in SALES_TAXONOMY:
                    expanded.extend(SALES_TAXONOMY.get(value, [value]))
                else:
                    unknown.append(value)
            if unknown:
                expanded.extend(unknown)
                expanded.extend(await _expand_keywords_with_llm(unknown, "Sales Job Titles", tracker))
            _set_criteria_values(criteria, "required_functions", expanded or values)

        if criteria.get("required_segments"):
            values = _criteria_values_for_search(criteria, "required_segments")
            expanded = []
            unknown = []
            for value in values:
                if value in SEGMENT_SYNONYMS:
                    expanded.extend(SEGMENT_SYNONYMS.get(value, [value]))
                else:
                    unknown.append(value)
            if unknown:
                expanded.extend(unknown)
                expanded.extend(await _expand_keywords_with_llm(unknown, "Customer Segments", tracker))
            _set_criteria_values(criteria, "required_segments", expanded or values)

        if criteria.get("required_company_details"):
            values = _criteria_values_for_search(criteria, "required_company_details")
            expanded = []
            unknown = []
            for value in values:
                if value in COMPANY_DETAILS_TAXONOMY:
                    expanded.extend(COMPANY_DETAILS_TAXONOMY.get(value, [value]))
                else:
                    unknown.append(value)
            if unknown:
                expanded.extend(unknown)
                expanded.extend(await _expand_keywords_with_llm(unknown, "Company Attributes", tracker))
            _set_criteria_values(criteria, "required_company_details", expanded or values)

        if criteria.get("required_culture_type"):
            values = _criteria_values_for_search(criteria, "required_culture_type")
            expanded = []
            unknown = []
            for value in values:
                if value in CULTURE_TAXONOMY:
                    expanded.extend(CULTURE_TAXONOMY.get(value, [value]))
                else:
                    unknown.append(value)
            if unknown:
                expanded.extend(unknown)
                expanded.extend(await _expand_keywords_with_llm(unknown, "Company Culture", tracker))
            _set_criteria_values(criteria, "required_culture_type", expanded or values)

        if criteria.get("required_geographies"):
            yield "Expanding geographies..."
            values = _criteria_values_for_search(criteria, "required_geographies")
            _set_criteria_values(criteria, "required_geographies", values + await _expand_geographies_with_llm(values, tracker))

        if criteria.get("required_locations"):
            yield "Expanding locations..."
            values = _criteria_values_for_search(criteria, "required_locations")
            _set_criteria_values(criteria, "required_locations", values + await _expand_locations_with_llm(values, tracker))

        for key in (
            "required_industries",
            "required_functions",
            "required_geographies",
            "required_segments",
            "required_company_details",
            "required_culture_type",
            "excluded_geographies",
            "required_locations",
            "required_keywords",
        ):
            if key in criteria:
                _set_criteria_values(criteria, key, _criteria_values_for_search(criteria, key))

        if company_keywords:
            criteria["required_companies"] = company_keywords
    except Exception as e:
        logger.error("Error expanding shortlist criteria: %s", e, exc_info=True)

    original_criteria = copy.deepcopy(criteria)
    original_criteria["_use_web_search"] = web_enabled
    original_criteria["_screening_query"] = query
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

    active_candidate_ids = [pid for pid, p in PROFILES_BY_ID.items() if not p.get("is_archived")]
    scoped_candidate_count = len(scoped_candidate_ids) if scoped_candidate_ids is not None else len(active_candidate_ids)

    if scoped_candidate_ids is not None and not scoped_candidate_ids:
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    selected_candidate_count = len(scoped_candidate_ids) if scoped_candidate_ids is not None else len(active_candidate_ids)

    async def _wait_if_paused() -> None:
        if pause_event:
            await pause_event.wait()

    await _wait_if_paused()
    yield "Loading the complete selected candidate scope..."
    search_query_parts: List[str] = []
    for key in (
        "required_companies",
        "required_industries",
        "required_functions",
        "required_segments",
        "required_geographies",
        "required_company_details",
        "required_culture_type",
        "required_keywords",
    ):
        search_query_parts.extend(_criteria_values_for_search(criteria, key))
    for item in criteria.get("min_function_years") or []:
        if isinstance(item, dict):
            search_query_parts.append(str(item.get("function") or ""))
            search_query_parts.extend(str(alias) for alias in (item.get("aliases") or []))
    search_query_text = " ".join(part for part in search_query_parts if str(part or "").strip())

    hard_filters_present = (
        criteria.get("required_locations")
        or criteria.get("min_people_managed") is not None
        or criteria.get("min_total_experience") is not None
        or criteria.get("required_companies")
        or criteria.get("funding_stage_min")
        or criteria.get("min_tenure_in_latest_role")
        or criteria.get("avg_tenure_in_last_n_roles")
    )
    if not search_query_text and not hard_filters_present:
        yield "Your query is too broad. Please specify industries, functions, segments, geographies, or locations."
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    # Shortlisting must evaluate the complete user-selected source. A semantic
    # top-N prefilter made equivalent phrasings inspect different candidates and
    # could therefore return false zero-result sets.
    initial_candidate_pool: List[Dict[str, Any]] = _scoped_build(
        scoped_candidate_ids if scoped_candidate_ids is not None else active_candidate_ids
    )
    scan_full_scope_for_strict_filters = True

    if not initial_candidate_pool:
        yield {"type": "complete", "data": [], "summary": tracker.get_summary()}
        return

    await _wait_if_paused()
    if web_enabled and _needs_candidate_company_fact_web_enrichment(criteria):
        yield "Researching company facts..."
        criteria = await enrich_criteria_with_candidate_company_web_facts(
            normalized_query,
            criteria,
            initial_candidate_pool,
            tracker,
        )
        logger.info(
            "SHORTLIST company_web_facts=%s",
            json.dumps(
                {
                    key: len(value) if isinstance(value, list) else 0
                    for key, value in (criteria.get("_web_company_facts") or {}).items()
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        original_criteria["_web_company_facts"] = criteria.get("_web_company_facts")

    try:
        initial_candidate_pool = await asyncio.to_thread(
            attach_schema_evidence_to_profiles, initial_candidate_pool, evidence_catalog
        )
    except Exception:
        logger.debug("Could not attach schema evidence to shortlist pool", exc_info=True)

    logger.info(
        "SHORTLIST complete_scope query=%s scoped_count=%s evaluated_count=%s full_scope_strict=%s search_text=%s",
        query,
        scoped_candidate_count,
        len(initial_candidate_pool),
        scan_full_scope_for_strict_filters,
        search_query_text,
    )
    # --- Batched strict scoring — keeps the event loop free between batches ---
    SCORING_BATCH_SIZE = 150

    if web_enabled and criteria.get("_web_company_facts"):
        yield "Scoring profiles with web-backed company data..."
    else:
        yield "Scoring profiles against enriched candidate data..."
    yield {
        "type": "progress",
        "phase": "filtering",
        "current": 0,
        "total": len(initial_candidate_pool),
        "passed": 0,
        "message": f"Evaluating candidate pool 0/{len(initial_candidate_pool)}...",
    }

    final_candidates: List[Dict[str, Any]] = []
    reject_reasons: Counter = Counter()
    reviewed_count = 0

    def _score_batch(batch: List[Dict[str, Any]]) -> tuple:
        passed: List[Dict[str, Any]] = []
        batch_reasons: List[str] = []
        for profile in batch:
            reasons: List[str] = []
            scored = _strict_shortlist_score_candidate(profile, criteria, reasons)
            if scored:
                passed.append(scored)
            else:
                batch_reasons.extend(reasons or ["unknown"])
        return passed, batch_reasons

    for batch_start in range(0, len(initial_candidate_pool), SCORING_BATCH_SIZE):
        await _wait_if_paused()
        batch = initial_candidate_pool[batch_start: batch_start + SCORING_BATCH_SIZE]

        passed_batch, batch_reasons = await asyncio.to_thread(_score_batch, batch)

        final_candidates.extend(passed_batch)
        reject_reasons.update(batch_reasons)
        reviewed_count += len(batch)

        # Stream all passers from this batch as ONE profile_chunk event
        # (avoids flooding the WebSocket with hundreds of frames on large pools)
        if passed_batch:
            safe_batch = []
            for passing in passed_batch:
                safe = {k: v for k, v in passing.items() if k != "embedding"}
                safe["shortlist_status"] = "pending_reasoning"
                safe["is_verified_match"] = False
                safe_batch.append(safe)
            yield {
                "type": "candidate_batch",
                "phase": "scoring",
                "data": safe_batch,
                "reviewed": reviewed_count,
                "passed": len(final_candidates),
                "total_pool": len(initial_candidate_pool),
            }

        yield {
            "type": "progress",
            "phase": "filtering",
            "current": reviewed_count,
            "total": len(initial_candidate_pool),
            "passed": len(final_candidates),
            "message": f"Evaluating candidate pool {reviewed_count}/{len(initial_candidate_pool)}...",
        }

    # -----------------------------------------------------------------

    logger.info(
        "SHORTLIST strict_filter_counts seen=%s passed=%s failed=%s reject_reason_counts=%s",
        len(initial_candidate_pool),
        len(final_candidates),
        len(initial_candidate_pool) - len(final_candidates),
        dict(reject_reasons.most_common(12)),
    )

    final_candidates = _sort_strict_shortlist_candidates(final_candidates, criteria)
    strict_passed_count = len(final_candidates)
    top_n = criteria.get("top_n")
    if top_n not in (None, 0):
        try:
            final_candidates = final_candidates[: int(top_n)]
        except Exception:
            pass

    if not final_candidates:
        yield {
            "type": "complete",
            "data": [],
            "summary": tracker.get_summary(),
            "verified_count": 0,
            "total_reviewed": len(initial_candidate_pool),
            "filter_debug": {
                "criteria": {k: v for k, v in criteria.items() if not k.startswith("_")},
                "semantic_pool_count": len(initial_candidate_pool),
                "full_scope_strict_scan": scan_full_scope_for_strict_filters,
                "passed": 0,
                "failed": len(initial_candidate_pool),
                "reject_reason_counts": dict(reject_reasons.most_common(12)),
            },
        }
        return

    await _wait_if_paused()
    yield {"type": "progress_start", "total": len(final_candidates), "total_considered": selected_candidate_count}
    await _wait_if_paused()
    yield "Generating match reasoning..."

    original_order = {str(profile.get("id")): index for index, profile in enumerate(final_candidates)}

    async def _with_reasoning(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await _wait_if_paused()
        review = await generate_reasoning_for_profile(
            profile,
            original_criteria,
            tracker,
            use_web_search=web_enabled,
        )
        await _wait_if_paused()
        updated = copy.deepcopy(profile)
        if isinstance(review, dict):
            final_status = _normalize_shortlist_status(review.get("final_status")) or "not_verified"
            if final_status != "verified_match":
                updated["shortlist_status"] = final_status
                updated["is_verified_match"] = False
                updated["missing_criteria"] = review.get("missing_criteria") if isinstance(review.get("missing_criteria"), list) else updated.get("missing_criteria", [])
                return None
            updated["shortlist_status"] = "verified_match"
            updated["is_verified_match"] = True
            updated["review_stage"] = "evidence_audited"
            updated["auditor_status"] = "passed"
            updated["answer"] = str(review.get("answer") or review.get("reasoning") or _fallback_reasoning_from_evidence(updated)).strip()
            updated["reasoning"] = str(review.get("reasoning") or updated["answer"]).strip()
            updated["evidence_ids"] = _extract_audit_evidence_ids(review)
            updated["confidence"] = str(review.get("confidence") or updated.get("confidence") or "high").strip().lower()
            if isinstance(review.get("matched_criteria"), list):
                updated["matched_criteria"] = review["matched_criteria"]
            if isinstance(review.get("missing_criteria"), list):
                updated["missing_criteria"] = review["missing_criteria"]
            if review.get("match_score") is not None:
                try:
                    updated["match_score"] = round(float(review.get("match_score")), 1)
                except Exception:
                    pass
        else:
            updated["shortlist_status"] = "verified_match"
            updated["is_verified_match"] = True
            updated["review_stage"] = "evidence_fallback"
            updated["reasoning"] = str(review or _fallback_reasoning_from_evidence(updated)).strip()
            updated["answer"] = updated["reasoning"]
        return updated

    processed_candidates: List[Dict[str, Any]] = []
    running_verified = 0

    for profile in final_candidates:
        await _wait_if_paused()
        try:
            visible = await _with_reasoning(profile)
        except asyncio.CancelledError:
            continue
        except Exception as e:
            logger.warning("Shortlist reasoning task failed: %s", e)
            continue
        if not visible:
            continue
        await _wait_if_paused()
        processed_candidates.append(visible)
        running_verified += 1
        yield {
            "type": "profile_chunk",
            "data": visible,
            "current": len(processed_candidates),
            "total": len(final_candidates),
            "reviewed": len(processed_candidates),
            "verified": running_verified,
            "potential": 0,
            "llm_reviewed": 0,
            "audited": 0,
        }

    processed_candidates.sort(key=lambda profile: original_order.get(str(profile.get("id")), 10**9))
    verified_count = len(processed_candidates)

    await _wait_if_paused()
    yield {
        "type": "complete",
        "data": processed_candidates,
        "summary": tracker.get_summary(),
        "verified_count": verified_count,
        "total_reviewed": len(initial_candidate_pool),
        "total_considered": selected_candidate_count,
        "selected_candidate_count": selected_candidate_count,
        "evidence_scored": len(initial_candidate_pool),
        "llm_reviewed": 0,
        "audited_count": 0,
        "potential_count": 0,
        "filter_debug": {
            "criteria": {k: v for k, v in criteria.items() if not k.startswith("_")},
            "semantic_pool_count": len(initial_candidate_pool),
            "full_scope_strict_scan": scan_full_scope_for_strict_filters,
            "passed": strict_passed_count,
            "returned": len(processed_candidates),
            "failed": len(initial_candidate_pool) - strict_passed_count,
            "reject_reason_counts": dict(reject_reasons.most_common(12)),
        },
    }


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

import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- OpenAI Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not found. Please set it in the .env file.")
    exit()

# Use the powerful model for generation
generation_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# --- Static Seed Taxonomies (Copied from your script) ---
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

# --- Helper Functions (Copied from your script) ---

def safe_json_loads(json_str: str, default_val: any = None) -> any:
    """Safely loads a JSON string, stripping markdown and handling errors."""
    if default_val is None:
        default_val = {}
    try:
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

def generate_dynamic_taxonomy(seed_taxonomy: dict, category: str) -> dict:
    """
    Uses an LLM to expand a seed taxonomy with more synonyms and related terms.
    """
    logger.info(f"Generating dynamic taxonomy for category: {category}...")

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

        logger.info(f"Successfully generated dynamic taxonomy for {category}.")
        return expanded_taxonomy
    except Exception as e:
        logger.error(f"An error occurred during taxonomy generation for {category}: {e}")
        return seed_taxonomy

def generate_dynamic_geography_map(seed_map: dict) -> dict:
    """
    Uses an LLM to expand a seed geography map with more countries and variations.
    """
    logger.info("Generating dynamic geography map...")

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

        logger.info("Successfully generated dynamic geography map.")
        return expanded_map
    except Exception as e:
        logger.error(f"An error occurred during geography map generation: {e}")
        return seed_map

def save_to_json(data: dict, filename: str):
    """Saves a dictionary to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to {filename}")
    except IOError as e:
        logger.error(f"Failed to write to {filename}: {e}")

def main():
    """Main function to generate all taxonomies and save them to files."""
    logger.info("--- Starting Taxonomy Generation ---")

    # Generate Sales Taxonomy
    sales_taxonomy = generate_dynamic_taxonomy(
        seed_taxonomy=STATIC_SALES_TAXONOMY,
        category="Sales Functions"
    )
    save_to_json(sales_taxonomy, "sales_taxonomy.json")

    # Generate Segment Synonyms
    segment_synonyms = generate_dynamic_taxonomy(
        seed_taxonomy=STATIC_SEGMENT_SYNONYMS,
        category="Customer Segments"
    )
    save_to_json(segment_synonyms, "segment_synonyms.json")

    # Generate Company Details Taxonomy
    company_details_taxonomy = generate_dynamic_taxonomy(
        seed_taxonomy=STATIC_COMPANY_DETAILS_TAXONOMY,
        category="Company Attributes (Funding, Business Model)"
    )
    save_to_json(company_details_taxonomy, "company_details_taxonomy.json")

    # Generate Culture Taxonomy
    culture_taxonomy = generate_dynamic_taxonomy(
        seed_taxonomy=STATIC_CULTURE_TAXONOMY,
        category="Company Culture Types"
    )
    save_to_json(culture_taxonomy, "culture_taxonomy.json")

    # Generate Geography Map
    geography_map = generate_dynamic_geography_map(STATIC_GEOGRAPHY_MAP)
    save_to_json(geography_map, "geography_map.json")

    logger.info("--- Taxonomy Generation Complete ---")

if __name__ == "__main__":
    main()
import pandas as pd
import json
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "data/raw/Account Executive - Dataset - Apify.xlsx" # CHANGE THIS to your new file
OUTPUT_FILE = "data/processed/ready_to_ingest.xlsx"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- The Strict Schema ingest_data.py Expects ---
# We define the "Target" columns and a brief description for the AI to understand them.
TARGET_SCHEMA_DEF = {
    # Personal Info
    "First Name": "Candidate's first name",
    "Last Name": "Candidate's last name",
    "Person Linkedin Url": "Unique identifier, full LinkedIn profile URL",
    "headline": "Current professional headline",
    "addressWithCountry": "Full location string (e.g., 'London, UK')",
    "about": "Summary or About section text",
    
    # We will generate the role/education keys dynamically below to save space, 
    # but the AI needs to know the pattern.
}

def generate_target_columns():
    """Generates the full list of 100+ columns expected by ingest_data.py"""
    cols = list(TARGET_SCHEMA_DEF.keys())
    
    # Roles 1 to 10
    for i in range(1, 11):
        suffix = "" if i == 1 else f".{i-1}" # Role 1 has no suffix, Role 2 is .1, etc.
        cols.append(f"Company {i} Name")
        cols.append(f"Title{suffix}")
        cols.append(f"Start date{suffix}")
        cols.append(f"End Date{suffix}")
        cols.append(f"Details {suffix}") # Note the space in "Details "
        
    # Education 1 to 3
    # Note: ingest_data.py uses a weird index offset for dates (Start date.10 is Edu 1)
    for i in range(1, 4):
        edu_date_idx = 9 + i
        deg_suffix = "" if i == 1 else f".{i-1}"
        
        cols.append(f"Education {i} - College Name")
        cols.append(f"Degree Name{deg_suffix}")
        cols.append(f"Start date.{edu_date_idx}")
        cols.append(f"End Date.{edu_date_idx}")
        
    return cols

TARGET_COLUMNS = generate_target_columns()

def get_smart_mapping(input_columns, sample_row):
    """
    Uses GPT-4o to map Input Columns -> Target Columns based on name and sample data.
    """
    logger.info("Asking AI to map columns...")
    
    prompt = f"""
    You are a data engineering assistant. Your task is to map the columns from a user's dataset to a strict Target Schema.
    
    --- TARGET SCHEMA (The columns I NEED) ---
    {json.dumps(TARGET_COLUMNS[:30])} ... (and so on following the pattern: Company X Name, Title.X-1, Start date.X-1, End Date.X-1)
    
    Key Rules for Target Schema:
    1. "Person Linkedin Url" is the unique ID.
    2. Roles are numbered 1 to 10. Role 1 cols have NO suffix (e.g. "Title"). Role 2 cols have ".1" (e.g. "Title.1").
    3. Education is numbered 1 to 3. Dates for Education 1 start at "Start date.10".
    
    --- USER INPUT DATA ---
    User Columns: {json.dumps(input_columns)}
    Sample Row Data: {json.dumps(sample_row, default=str)}
    
    --- INSTRUCTIONS ---
    1. Analyze the User Columns and Sample Data to understand what each column contains.
    2. Map each "Target Column" to the best matching "User Column".
    3. If a User Column matches a Target Column exactly, map it.
    4. If the user has "Current Company", map it to "Company 1 Name". "Previous Company" -> "Company 2 Name", etc.
    5. If the user has a list of roles (e.g. "Job 1 Title", "Job 2 Title"), map them to the corresponding numbered Target columns.
    6. **CRITICAL**: Do not hallucinate. If there is no data for a Target Column (e.g., the user only has 3 roles, but Target has 10), map it to null (None).
    
    Return ONLY a valid JSON object: {{ "Target Column Name": "User Column Name" or null }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise data mapper. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    return extract_json_from_text(content)

def extract_json_from_text(text):
    """
    Extracts JSON object from a string that might contain Markdown or other text.
    "```json ... ```" or "``` ... ```"
    """
    try:
        # A simple pass first
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first '{' and the last '}'
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Check for markdown blocks if simple extraction failed
    import re
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Default fallback - return empty dict or raise based on preference. 
    # Raising allows us to see the error.
    logger.error(f"Failed to extract JSON from: {text}")
    return {}

def normalize_data():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data
    logger.info(f"Loading {INPUT_FILE}...")
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
        
    df = df.fillna("") # Fill NaNs to avoid JSON errors
    
    # 2. Get AI Mapping
    input_cols = df.columns.tolist()
    sample_row = df.iloc[0].to_dict()
    
    mapping = get_smart_mapping(input_cols, sample_row)
    
    logger.info("Generated Mapping:")
    print(json.dumps(mapping, indent=2))
    
    # 3. Apply Mapping & Transform
    logger.info("Transforming data...")
    new_df = pd.DataFrame()
    
    for target_col, input_col in mapping.items():
        if input_col and input_col in df.columns:
            new_df[target_col] = df[input_col]
        else:
            # If mapped to null or column missing, create empty column
            new_df[target_col] = ""
            
    # 4. Save
    logger.info(f"Saving normalized data to {OUTPUT_FILE}...")
    new_df.to_excel(OUTPUT_FILE, index=False)
    logger.info("Done! You can now run 'ingest_data.py' pointing to this new file.")

if __name__ == "__main__":
    normalize_data()

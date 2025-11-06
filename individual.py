import pandas as pd
import openai
import json
import logging
import os
import asyncio  # <-- NEW: Import for asynchronous operations
import time     # <-- NEW: To time our script
from datetime import datetime
from dateutil import parser
from openai import AsyncOpenAI  # <-- NEW: Use the Async client

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Configuration ===
EXCEL_PATH = "/home/nethranand-ps/AI_HR/individual/Sales Leader - Sample Dataset.xlsx"
OUTPUT_JSON = "enriched_candidate_profiles_async.json"
COMPANY_CACHE_JSON = "company_cache.json"

# --- NEW: Model & Concurrency Settings ---
# We keep gpt-4o for high-accuracy analysis
ACCURATE_MODEL = "gpt-4o"
# We use a *cheaper, faster* model for simple company fact-retrieval
FAST_MODEL = "gpt-4o-mini"
# Your OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set how many requests to run at the same time.
# Start with 10 or 20. If you get rate-limit errors, lower it.
# If you have a high-tier plan, you can increase this.
CONCURRENCY_LIMIT = 20

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    exit(1)

# === Helper Functions (Unchanged) ===
# All helper functions (clean, get_datetime, merge_periods, etc.)
# are fast, local code, so they don't need to be async.

def clean(x):
    result = str(x).strip() if pd.notnull(x) else ""
    return result

def get_datetime(val):
    if pd.isnull(val) or str(val).strip().lower() in ['na', '', 'present']:
        return pd.NaT
    if isinstance(val, datetime):
        return pd.Timestamp(val)
    try:
        num = float(val)
        if 1900 <= num <= 2100:
            return pd.Timestamp(f"{int(num)}-01-01")
        else:
            return pd.Timestamp(num, unit='d', origin='1899-12-30')
    except (ValueError, TypeError):
        return pd.to_datetime(val, errors='coerce')

def merge_periods(periods):
    if not periods: return []
    periods = sorted(periods, key=lambda x: x[0])
    merged = [periods[0]]
    for curr in periods[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)
    return merged

def extract_gap_years(row, current_date):
    periods = []
    for i in range(1, 11):
        idx = '' if i == 1 else f'.{i-1}'
        start = get_datetime(row.get(f"Start date{idx}"))
        end = get_datetime(row.get(f"End Date{idx}"))
        if pd.isnull(end): end = current_date
        if pd.notnull(start) and end > start:
            periods.append((start, end))
    if not periods: return False, []
    
    merged = merge_periods(periods)
    gaps = []
    for j in range(len(merged) - 1):
        gap_start, gap_end = merged[j][1], merged[j+1][0]
        days = (gap_end - gap_start).days
        if days > 180:
            gaps.append({
                "from": gap_start.strftime("%Y-%m"),
                "to": gap_end.strftime("%Y-%m"),
                "duration_months": round(days / 30),
                "reason": "unknown"
            })
    return len(gaps) > 0, gaps

def extract_education_gaps(row, current_date):
    periods = []
    for i in range(1, 4):
        s = get_datetime(row.get(f"Start date.{9+i}"))
        e = get_datetime(row.get(f"End Date.{9+i}")) or current_date
        if pd.notnull(s) and e > s and clean(row.get(f"Education {i} - College Name")):
            periods.append((s, e))
    
    merged = merge_periods(periods)
    gaps = []
    for a, b in zip(merged, merged[1:]):
        days = (b[0] - a[1]).days
        if days > 180:
            gaps.append({
                "from": a[1].strftime("%Y-%m"), "to": b[0].strftime("%Y-%m"),
                "duration_months": round(days / 30), "reason": "education"
            })
    return bool(gaps), gaps

def extract_industry_gaps(roles, ind_roles):
    if not ind_roles or len(roles) != len(ind_roles):
        return False, []
    
    stints = sorted(
        [(r["start_dt"], r["end_dt"], ir.get("industry"))
         for r, ir in zip(roles, ind_roles) if pd.notnull(r["start_dt"]) and r.get("end_dt") > r["start_dt"] and ir.get("industry")],
        key=lambda x: x[0]
    )

    gaps = []
    for (s1, e1, ind1), (s2, e2, ind2) in zip(stints, stints[1:]):
        days = (s2 - e1).days
        if days > 180 and ind1 != ind2:
            gaps.append({
                "from": e1.strftime("%Y-%m"), "to": s2.strftime("%Y-%m"),
                "duration_months": round(days / 30),
                "from_industry": ind1, "to_industry": ind2,
                "reason": "industry"
            })
    return len(gaps) > 0, gaps

# --- NEW: Asynchronous GPT Call Function ---
async def call_gpt_async(client, model, data, prompt, retries=3, delay=5):
    """
    Asynchronously call OpenAI and robustly parse JSON response.
    Includes exponential backoff for rate limits.
    """
    logger.info(f"Calling {model} for prompt: {prompt[:50]}...")
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You MUST return *only* a single valid JSON object—no extra text, no markdown, no explanations."},
                    {"role": "user", "content": f"{prompt}\n\nData:\n{json.dumps(data, indent=2)}"}
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Error calling OpenAI or parsing JSON: {e}")
            return None  # Return None on non-retryable error
    logger.error(f"Failed to call OpenAI after all retries (RateLimit) for prompt: {prompt[:50]}.")
    return None

# --- NEW: Async Function to get/fetch company data (thread-safe) ---
async def get_or_fetch_company_details(client, companies_needed, company_cache, lock):
    """
    Asynchronously checks cache for company details, fetches missing ones,
    and updates the *global* cache. This is lock-protected to prevent race conditions.
    """
    new_companies_to_fetch = []
    
    # --- CRITICAL SECTION (READ) ---
    # Check cache to see what we're missing
    async with lock:
        for comp in companies_needed:
            if comp not in company_cache:
                new_companies_to_fetch.append(comp)
    # --- END CRITICAL SECTION ---

    # If we have everything we need, we're done.
    if not new_companies_to_fetch:
        return

    logger.info(f"Fetching details for {len(new_companies_to_fetch)} new companies using {FAST_MODEL}...")
    comp_prompt = (
        'For each company in the list, provide firmographic details. Your response must be a single JSON object where keys are the company names.'
        'Each value should be an object with keys: "product_service", "customer_segment" (list), "customer_presence" (list), "funding_stage", "revenue", "culture_type", "headquarters", "business_model".'
    )
    
    # --- API CALL (Outside lock) ---
    # We make the API call *outside* the lock so other tasks aren't blocked from
    # reading the cache while we wait for the network.
    comp_res = await call_gpt_async(client, FAST_MODEL, {"companies": new_companies_to_fetch}, comp_prompt)
    
    if comp_res:
        # --- CRITICAL SECTION (WRITE) ---
        # Now we lock again to safely update the global cache and save to disk
        async with lock:
            logger.info(f"Updating company cache with {len(comp_res)} new entries.")
            company_cache.update(comp_res)
            # Save the updated cache immediately for robustness
            try:
                with open(COMPANY_CACHE_JSON, "w") as f:
                    json.dump(company_cache, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save company cache: {e}")
        # --- END CRITICAL SECTION ---
    else:
        logger.error(f"Failed to fetch any new company details for: {new_companies_to_fetch}")

# --- NEW: Async Function to save profile to JSON (thread-safe) ---
async def save_profile_incrementally(profile, lock):
    """
    Asynchronously appends a single processed profile to the
    output JSON file, protected by a lock to prevent file corruption.
    """
    # --- CRITICAL SECTION (FILE I/O) ---
    async with lock:
        try:
            # Read the existing data
            try:
                with open(OUTPUT_JSON, "r") as f:
                    profiles = json.load(f)
            except FileNotFoundError:
                profiles = []
            
            # Append the new profile
            profiles.append(profile)
            
            # Write the updated data back
            with open(OUTPUT_JSON, "w") as f:
                json.dump(profiles, f, indent=2, default=str)
                
            logger.info(f"Successfully saved profile for: {profile['name']}")
            
        except Exception as e:
            logger.error(f"Failed to save profile for {profile['name']}: {e}")
    # --- END CRITICAL SECTION ---

# --- UPDATED: Asynchronous Candidate Processing Function ---
async def process_candidate_async(row, client, semaphore, current_date, company_cache, company_lock, file_lock):
    """
    Processes a single candidate row asynchronously, fetches company data,
    assembles the full profile, and saves it to the JSON file.
    """
    # Use the semaphore to limit concurrency of GPT-4o calls
    async with semaphore:
        try:
            profile_name = f"{clean(row.get('First Name'))} {clean(row.get('Last Name'))}".strip()
            logger.info(f"--- Processing profile: {profile_name} ---")
            raw_fields = {col: clean(row[col]) for col in row.index}

            # --- UPDATED: Role Parsing Logic ---
            # Build roles list (local, fast)
            roles, company_years, total_days = [], {}, 0
            unique_companies = set()
            last_valid_company = ""  # <-- NEW: Track the last company

            for i in range(1, 11):
                idx_str = '' if i == 1 else f'.{i-1}'
                
                comp = clean(row.get(f"Company {i} Name"))
                title = clean(row.get(f"Title{idx_str}")) # <-- Get title *before* checking company
                
                # If company is blank but title is present, use the last known company
                if not comp and title:
                    comp = last_valid_company
                elif comp:
                    last_valid_company = comp # <-- Update the last known company
                
                # If we still don't have a company (e.g., first role is blank)
                # or if we don't have a title, skip.
                if not comp or not title:
                    continue

                unique_companies.add(comp)
                
                start = get_datetime(row.get(f"Start date{idx_str}"))
                end = get_datetime(row.get(f"End Date{idx_str}")) or current_date
                details_key = f"Details .{i-1}" if i > 1 else "Details "

                if pd.notnull(start) and end > start:
                    days = (end - start).days
                    yrs = round(days / 365.25, 2)
                    roles.append({
                        "company": comp,
                        "title": title, # <-- Use the title we got earlier
                        "details": clean(row.get(details_key)),
                        "duration_years": yrs,
                        "start_dt": start, "end_dt": end,
                        "start": str(row.get(f"Start date{idx_str}")),
                        "end": str(row.get(f"End Date{idx_str}"))
                    })
                    company_years[comp] = company_years.get(comp, 0) + yrs
                    total_days += days
            # --- END UPDATED: Role Parsing Logic ---
            
            total_exp = round(total_days / 365.25, 2)
            avg_tenure = round(total_exp / len(company_years), 2) if company_years else 0

            # Gaps (Pre-AI)
            has_gap, gaps = extract_gap_years(row, current_date)
            has_edu_gap, edu_gaps = extract_education_gaps(row, current_date)
            
            # Combined 'about' text
            about_sections = [
                raw_fields.get("about"),
                raw_fields.get("Details"),
                raw_fields.get("Details.1")
            ]
            combined_about_text = " ".join(filter(None, about_sections))

            # --- ASYNC API CALL 1: Candidate Analysis (Accurate Model) ---
            # This call is limited by the semaphore
            master_prompt = """
            Based on the candidate's profile data, provide a comprehensive evaluation as a single JSON object.
            The JSON object must have these top-level keys: "functional_experience", "industry_experience", "segment_experience", "geography_experience", "team_management".
            Each key must contain an object with a score (int), rationale (str), and other specific fields as described below:
            - functional_experience: roles list [{"company": str, "activity_type": str, "reason": str, "duration_years": float}]
            - industry_experience: roles list [{"company": str, "industry": str, "reason": str, "duration_years": float}]
            - segment_experience: roles list [{"company": str, "segment": str, "reason": str, "duration_years": float}]
            - geography_experience: regions list [str]
            - team_management: {"max_people_managed": int, "years_team_management": float}
            """
            gpt_data = {
                "headline": clean(row.get("headline")),
                "about": combined_about_text,
                "roles": [{"company": r["company"], "title": r["title"], "details": r["details"], "duration_years": r["duration_years"]} for r in roles],
                "location": clean(row.get("addressWithCountry")),
            }
            
            master_res = await call_gpt_async(client, ACCURATE_MODEL, gpt_data, master_prompt)
            if not master_res:
                logger.error(f"Skipping profile for {profile_name} due to GPT API failure.")
                return # This candidate failed, stop processing it

            # Extract results (local, fast)
            func_res = master_res.get("functional_experience", {})
            ind_res = master_res.get("industry_experience", {})
            seg_res = master_res.get("segment_experience", {})
            geo_res = master_res.get("geography_experience", {})
            tm_res = master_res.get("team_management", {})
            
            has_ind_gap, ind_gaps = extract_industry_gaps(roles, ind_res.get("roles", []))
            
            # --- NEW: ASYNC API CALL 2 (Conditional) ---
            # Fetch company details *for this candidate*
            # This uses its own lock, not the semaphore
            await get_or_fetch_company_details(client, unique_companies, company_cache, company_lock)
            
            # Education
            education = []
            for j in range(1, 4):
                if college := clean(row.get(f"Education {j} - College Name")):
                    deg_idx = '' if j == 1 else f'.{j-1}'
                    education.append({
                        "college": college,
                        "degree": clean(row.get(f"Degree Name{deg_idx}")),
                        "start": str(row.get(f"Start date.{9 + j}")),
                        "end": str(row.get(f"End Date.{9 + j}"))
                    })
            
            # Final Profile Assembly
            location_str = clean(row.get("addressWithCountry"))
            city = location_str.split(",")[0].strip() if location_str else ""

            # Add company details (which are now in the global cache)
            for r in roles:
                r["company_details"] = company_cache.get(r["company"], {})

            profile = {
                "name": profile_name,
                "linkedin": clean(row.get("Person Linkedin Url")),
                "location": location_str,
                "city": city,
                "headline": clean(row.get("headline")),
                "about": combined_about_text,
                "roles": [{k: v for k, v in r.items() if k not in ["start_dt", "end_dt"]} for r in roles],
                "raw_fields": raw_fields,
                "total_experience_years": total_exp,
                "avg_years_in_company": avg_tenure,
                "company_years": company_years,
                "has_gap_years": has_gap, "gaps": gaps,
                "has_education_gaps": has_edu_gap, "education_gaps": edu_gaps,
                "has_industry_gaps": has_ind_gap, "industry_gaps": ind_gaps,
                "functional_experience": func_res, "industry_experience": ind_res,
                "segment_experience": seg_res, "geography_experience": geo_res,
                "team_management": tm_res,
                "education": education,
                "titles_held": [{"title": r["title"], "company": r["company"], "start": r["start"], "end": r["end"]} for r in roles],
                "full_row": json.dumps(raw_fields, default=str),
                "embedding": []
            }
            
            # --- NEW: Asynchronous Save ---
            # Save this completed profile to the main JSON file
            await save_profile_incrementally(profile, file_lock)
        
        except Exception as e:
            profile_name = f"{clean(row.get('First Name'))} {clean(row.get('Last Name'))}".strip()
            logger.error(f"CRITICAL error in process_candidate_async for {profile_name}: {e}")
            return # Ensure task exits on unexpected error

# --- UPDATED: Rewritten Main Function (Orchestrator) ---
async def main():
    logger.info("🚀 Asynchronous processing starting…")
    start_time = time.time()
    
    # Load data and state
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        return
        
    logger.info(f"Loaded Excel file with {len(df)} rows.")
    
    # We load profiles just to find out where to start
    try:
        with open(OUTPUT_JSON, "r") as f:
            enriched_profiles = json.load(f)
    except FileNotFoundError:
        enriched_profiles = []
    
    # We load the company cache to be shared by all tasks
    try:
        with open(COMPANY_CACHE_JSON, "r") as f:
            # This cache will be modified in-place by our async functions
            company_cache = json.load(f) 
    except FileNotFoundError:
        company_cache = {}

    start_idx = len(enriched_profiles)
    num_to_process = len(df.iloc[start_idx:])
    
    if num_to_process == 0:
        logger.info("✅ All profiles are already processed. Exiting.")
        return

    logger.info(f"Resuming at row {start_idx + 1}. Need to process {num_to_process} new profiles.")
    current_date = pd.to_datetime("now")

    # --- NEW: Async Setup ---
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    # Limits how many GPT-4o calls run at once
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT) 
    # Prevents race conditions when updating company_cache
    company_lock = asyncio.Lock()
    # Prevents race conditions when writing to OUTPUT_JSON
    file_lock = asyncio.Lock()
    
    tasks = []

    # --- PHASE 1: Create all candidate analysis tasks ---
    logger.info(f"Creating {num_to_process} candidate processing tasks...")
    for _, row in df.iloc[start_idx:].iterrows():
        tasks.append(process_candidate_async(
            row, client, semaphore, current_date, 
            company_cache, company_lock, file_lock
        ))
    
    # --- PHASE 2: Run all tasks concurrently ---
    # return_exceptions=True ensures one failed task doesn't stop the whole batch
    logger.info("Running all tasks concurrently...")
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # --- PHASE 3: Final Tally ---
    failed_tasks = 0
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            row_num = start_idx + i + 1
            logger.error(f"Task for row {row_num} failed with exception: {result}")
            failed_tasks += 1

    logger.info(f"--- Task execution complete. {len(tasks) - failed_tasks} successful, {failed_tasks} failed. ---")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"✅ All processing complete in {total_time / 60:.2f} minutes.")
    logger.info(f"Final output saved to {OUTPUT_JSON}")

# --- NEW: Run the main asynchronous function ---
if __name__ == "__main__":
    asyncio.run(main())



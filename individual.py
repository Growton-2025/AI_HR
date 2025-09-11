


import pandas as pd
import openai
import json
import logging
from datetime import datetime
from dateutil import parser
import re

# === Logging Configuration ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Configuration ===
EXCEL_PATH = "/home/nethranand-ps/AI_HR/Sales Leader - Dataset.xlsx"
OUTPUT_JSON = "enriched_candidate_profiles.json"
OPENAI_MODEL = "gpt-4o"  # or another model you have access to
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def clean(x):
    """Convert input to cleaned string, handling nulls."""
    result = str(x).strip() if pd.notnull(x) else ""
    logger.debug(f"clean() input: {x!r} -> {result!r}")
    return result

def get_datetime(val):
    """Convert value to pd.Timestamp, handling Excel serial, years, strings, and datetime objects."""
    if pd.isnull(val) or str(val).strip().lower() in ['na', '', 'present']:
        return pd.NaT
    if isinstance(val, datetime):
        return pd.Timestamp(val)
    try:
        num = float(val)
        if 1900 <= num <= 2100:  # Likely a year
            return pd.Timestamp(f"{int(num)}-01-01")
        else:  # Excel serial date
            return pd.Timestamp(num, unit='d', origin='1899-12-30')
    except (ValueError, TypeError):
        # String date or other
        return pd.to_datetime(val, errors='coerce')

def calculate_experience_and_company_years(row, current_date):
    """Calculate total experience, avg tenure, and per-company years."""
    logger.debug("Entering calculate_experience_and_company_years")
    company_years = {}
    total_days = 0
    for i in range(1, 11):
        idx = '' if i == 1 else f'.{i-1}'
        start = get_datetime(row.get(f"Start date{idx}"))
        end = get_datetime(row.get(f"End Date{idx}"))
        if pd.isnull(end):
            end = current_date
        if pd.notnull(start) and end > start:
            days = (end - start).days
            yrs = round(days / 365.25, 2)
            comp = clean(row.get(f"Company {i} Name"))
            if comp:
                company_years[comp] = company_years.get(comp, 0) + yrs
                total_days += days
                logger.debug(f"Company {comp}: {yrs} years ({days} days)")
    total_years = round(total_days / 365.25, 2)
    avg_tenure = round(total_years / len(company_years), 2) if company_years else 0
    logger.debug(f"Total years: {total_years}, Avg tenure: {avg_tenure}")
    return total_years, avg_tenure, company_years

def merge_periods(periods):
    """Merge overlapping date periods."""
    logger.debug(f"Merging periods: {periods}")
    if not periods:
        return []
    periods = sorted(periods, key=lambda x: x[0])
    merged = [periods[0]]
    for curr in periods[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)
    logger.debug(f"Merged periods: {merged}")
    return merged

def extract_gap_years(row, current_date):
    """Extract gaps >6 months, with from-to and reason."""
    logger.debug("Entering extract_gap_years")
    periods = []
    for i in range(1, 11):
        idx = '' if i == 1 else f'.{i-1}'
        start = get_datetime(row.get(f"Start date{idx}"))
        end = get_datetime(row.get(f"End Date{idx}"))
        if pd.isnull(end):
            end = current_date
        if pd.notnull(start) and end > start:
            periods.append((start, end))
    if not periods:
        return False, []
    merged = merge_periods(periods)
    gaps = []
    for j in range(len(merged) - 1):
        gap_start, gap_end = merged[j][1], merged[j+1][0]
        days = (gap_end - gap_start).days
        if days > 180:
            gap_info = {
                "from": gap_start.strftime("%Y-%m"),
                "to": gap_end.strftime("%Y-%m"),
                "duration_months": round(days / 30),
                "reason": "unknown"
            }
            gaps.append(gap_info)
            logger.debug(f"Detected gap: {gap_info}")
    return len(gaps) > 0, gaps

def extract_education_gaps(row, current_date):
    """Extract education gaps >6 months."""
    edu_idxs = [10, 11, 12]
    periods = []
    for i in edu_idxs:
        s = get_datetime(row.get(f"Start date.{i}"))
        e = get_datetime(row.get(f"End Date.{i}")) or current_date
        college = clean(row.get(f"Education {i-9} - College Name"))
        if pd.notnull(s) and e > s and college:
            periods.append((s, e))
    merged = merge_periods(periods)
    gaps = []
    for a, b in zip(merged, merged[1:]):
        days = (b[0] - a[1]).days
        if days > 180:
            gaps.append({
                "from": a[1].strftime("%Y-%m"),
                "to": b[0].strftime("%Y-%m"),
                "duration_months": round(days / 30),
                "reason": "education"
            })
    return bool(gaps), gaps

def extract_industry_gaps(roles, ind_roles, current_date):
    """Extract industry-change gaps >6 months."""
    if len(roles) != len(ind_roles):
        logger.warning("Mismatch in roles length for industry gaps.")
        return False, []
    stints = []
    for orig, ind in zip(roles, ind_roles):
        start = orig.get("start_dt")
        end = orig.get("end_dt")
        industry = ind.get("industry")
        if pd.notnull(start) and end > start and industry:
            stints.append((start, end, industry))
    stints.sort(key=lambda x: x[0])
    gaps = []
    for (s1, e1, ind1), (s2, e2, ind2) in zip(stints, stints[1:]):
        days = (s2 - e1).days
        if days > 180 and ind1 != ind2:
            gaps.append({
                "from": e1.strftime("%Y-%m"),
                "to": s2.strftime("%Y-%m"),
                "duration_months": round(days / 30),
                "from_industry": ind1,
                "to_industry": ind2,
                "reason": "industry"
            })
    return len(gaps) > 0, gaps

def call_score(data, prompt):
    """Call OpenAI to classify/score; return parsed JSON with robust extraction."""
    logger.info(f"Calling GPT for prompt: {prompt}")
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You MUST return *only* a single valid JSON object—no extra text, no markdown."},
            {"role": "user", "content": prompt + "\n\nData:\n" + json.dumps(data)}
        ],
        temperature=0.0,
        max_tokens=1500
    )
    content = resp.choices[0].message.content.strip()
    logger.debug(f"Raw GPT response: {content}")
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    snippet = content[start:end+1]
    opens = snippet.count("{")
    closes = snippet.count("}")
    snippet += "}" * max(0, opens - closes)
    snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as e:
        logger.error(f"Final JSON parse failed. Snippet was:\n{snippet}")
        raise

def verify_profile(profile):
    """Cross-verify key fields between raw_fields and enriched profile."""
    raw = profile.get('raw_fields', {})
    expected_name = f"{raw.get('First Name', '').strip()} {raw.get('Last Name', '').strip()}".strip()
    if profile.get('name') != expected_name:
        logger.warning(f"Name mismatch: enriched '{profile.get('name')}' vs raw '{expected_name}'")
    if profile.get('headline') != raw.get('headline'):
        logger.warning(f"Headline mismatch: enriched '{profile.get('headline')}' vs raw '{raw.get('headline')}'")
    if profile.get('location') != raw.get('addressWithCountry'):
        logger.warning(f"Location mismatch: enriched '{profile.get('location')}' vs raw '{raw.get('addressWithCountry')}'")
    raw_about = raw.get('about', '') or raw.get('Details', '')
    if profile.get('about') != raw_about:
        logger.warning(f"About mismatch: enriched '{profile.get('about')}' vs raw '{raw_about}'")

# === Main Processing ===
def main():
    logger.info("🚀 main starting…")
    df = pd.read_excel(EXCEL_PATH)
    logger.info(f"Loaded Excel file with {len(df)} rows.")
    try:
        with open(OUTPUT_JSON, "r") as f:
            enriched_profiles = json.load(f)
    except FileNotFoundError:
        enriched_profiles = []
    start_idx = len(enriched_profiles)
    logger.info(f"Resuming at row {start_idx + 1}")
    current_date = pd.to_datetime("2025-09-11")

    for idx, row in df.iloc[start_idx:].iterrows():
        logger.info(f"Processing row {idx + 1}/{len(df)}")
        raw_fields = {col: clean(row[col]) for col in df.columns}

        # Build roles list with accurate durations
        roles = []
        company_years = {}
        total_days = 0
        for i in range(1, 11):
            idx_str = '' if i == 1 else f'.{i-1}'
            comp = clean(row.get(f"Company {i} Name"))
            if not comp:
                continue
            title = clean(row.get(f"Title{idx_str}"))
            details = clean(row.get(f"Details {idx_str}"))
            start_val = row.get(f"Start date{idx_str}")
            end_val = row.get(f"End Date{idx_str}")
            start = get_datetime(start_val)
            end = get_datetime(end_val)
            if pd.isnull(end):
                end = current_date
            if pd.notnull(start) and end > start:
                days = (end - start).days
                yrs = round(days / 365.25, 2)
                roles.append({
                    "company": comp,
                    "title": title,
                    "details": details,
                    "duration_years": yrs,
                    "start_dt": start,
                    "end_dt": end,
                    "start": str(start_val),
                    "end": str(end_val) if end_val not in ['NA', '', 'Present'] else ""
                })
                company_years[comp] = company_years.get(comp, 0) + yrs
                total_days += days
        total_exp = round(total_days / 365.25, 2)
        avg_tenure = round(total_exp / len(company_years), 2) if company_years else 0

        # Gaps
        has_gap, gaps = extract_gap_years(row, current_date)
        has_edu_gap, edu_gaps = extract_education_gaps(row, current_date)

        # Prepare gpt_data with roles including duration
        gpt_data = {
            "headline": clean(row.get("headline")),
            "about": clean(row.get("about")),
            "roles": [{"company": r["company"], "title": r["title"], "details": r["details"], "duration_years": r["duration_years"]} for r in roles],
            "location": clean(row.get("addressWithCountry")),
            "raw_fields": raw_fields
        }

        # Prompts with specified output structure
        func_prompt = (
            "Evaluate the candidate's functional sales experience using headline, about, and roles.\n"
            "Provide JSON: {\"functional_experience_score\": int, \"rationale\": str, \"roles\": list of {\"company\": str, \"activity_type\": str, \"reason\": str, \"duration_years\": float}}.\n"
        )
        ind_prompt = (
            "Evaluate the candidate's industry experience using headline, about, and roles.\n"
            "Provide JSON: {\"industry_experience_score\": int, \"rationale\": str, \"roles\": list of {\"company\": str, \"industry\": str, \"reason\": str, \"duration_years\": float}}.\n"
        )
        seg_prompt = (
            "Evaluate the candidate's customer segment exposure (e.g., Enterprise, SMB) using headline, about, and roles.\n"
            "Provide JSON: {\"segment_experience_score\": int, \"rationale\": str, \"roles\": list of {\"company\": str, \"segment\": str, \"reason\": str, \"duration_years\": float}}.\n"
        )
        geo_prompt = (
            "Assess the candidate's geographic experience using location, headline, about, and roles.\n"
            "Provide JSON: {\"geography_experience_score\": int, \"rationale\": str, \"regions\": list of str}.\n"
        )
        tm_prompt = (
            "Assess the candidate's team management experience using headline, about, and roles.\n"
            "Provide JSON: {\"team_management_score\": int, \"rationale\": str, \"max_people_managed\": int, \"years_team_management\": float}.\n"
        )

        # Call GPT
        func_res = call_score(gpt_data, func_prompt)
        ind_res = call_score(gpt_data, ind_prompt)
        seg_res = call_score(gpt_data, seg_prompt)
        geo_res = call_score(gpt_data, geo_prompt)
        tm_res = call_score(gpt_data, tm_prompt)

        # Industry gaps after ind_res
        has_ind_gap, ind_gaps = extract_industry_gaps(roles, ind_res["roles"], current_date)

        # Company details
        unique_comps = list(set(r["company"] for r in roles))
        comp_prompt = (
            "For each company, provide details based on your knowledge.\n"
            "Output JSON: {\"company_details\": {company: {\"company\": str, \"product_service\": str, \"customer_segment\": list, \"customer_presence\": list, \"funding_stage\": str, \"revenue\": str, \"culture_type\": str, \"headquarters\": str, \"business_model\": str} for each company}}.\n"
            "Companies: " + json.dumps(unique_comps)
        )
        comp_res = call_score({"companies": unique_comps}, comp_prompt)
        comp_details_dict = comp_res.get("company_details", {})
        for r in roles:
            r["company_details"] = comp_details_dict.get(r["company"], {})

        # Education
        education = []
        for j in range(1, 4):
            college = clean(row.get(f"Education {j} - College Name"))
            if not college:
                continue
            deg_idx = '' if j == 1 else f'.{j-1}'
            degree = clean(row.get(f"Degree Name{deg_idx}"))
            start_val = row.get(f"Start date.{9 + j}")
            end_val = row.get(f"End Date.{9 + j}")
            education.append({
                "college": college,
                "degree": degree,
                "start": str(start_val),
                "end": str(end_val) if end_val not in ['NA', '', 'Present'] else ""
            })

        # Titles held
        titles_held = [{"title": r["title"], "company": r["company"], "start": r["start"], "end": r["end"]} for r in roles]

        # Build profile
        profile = {
            "name": f"{clean(row.get('First Name'))} {clean(row.get('Last Name'))}".strip(),
            "linkedin": clean(row.get("Person Linkedin Url")),
            "location": clean(row.get("addressWithCountry")),
            "city": clean(row.get("addressWithCountry")).split(",")[0].strip() if clean(row.get("addressWithCountry")) else "",
            "headline": clean(row.get("headline")),
            "about": clean(row.get("about")),
            "roles": [{k: v for k, v in r.items() if k not in ["start_dt", "end_dt"]} for r in roles],  # Exclude internal dt
            "raw_fields": raw_fields,
            "total_experience_years": total_exp,
            "avg_years_in_company": avg_tenure,
            "company_years": company_years,
            "has_gap_years": has_gap,
            "gaps": gaps,
            "has_education_gaps": has_edu_gap,
            "education_gaps": edu_gaps,
            "has_industry_gaps": has_ind_gap,
            "industry_gaps": ind_gaps,
            "functional_experience": func_res,
            "industry_experience": ind_res,
            "segment_experience": seg_res,
            "geography_experience": geo_res,
            "team_management": tm_res,
            "education": education,
            "titles_held": titles_held,
            "full_row": json.dumps(raw_fields),
            "embedding": []
        }

        verify_profile(profile)
        enriched_profiles.append(profile)

        # Persist
        with open(OUTPUT_JSON, "w") as f:
            json.dump(enriched_profiles, f, indent=2)

    logger.info(f"✅ Enriched profiles saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    logger.info("🚀 vector.py starting…")
    main()
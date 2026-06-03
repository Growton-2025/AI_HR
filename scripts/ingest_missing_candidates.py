import sys
import os
import json
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from backend.db.connection import get_db_connection_context
from scripts.ingest_and_verify import (
    _setup_recruiter_and_role, _push_to_db,
    parse_roles_from_raw, parse_education_from_raw, calculate_tenure_metrics,
    _company_contexts_by_norm, _load_db_company_details, classify_company,
    extract_profile_claims, build_enrichment_payload
)

def normalize_url(url):
    if not isinstance(url, str):
        return ""
    url = url.strip().lower()
    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")
    # Remove www.
    url = url.replace("www.", "")
    # Remove trailing slash
    if url.endswith("/"):
        url = url[:-1]
    return url

def _push_to_db_truncated(candidate_data, roles, education, payload, metrics, user_id, role_id):
    def trunc(val, limit=255):
        if val is None:
            return ""
        s = str(val).strip()
        return s[:limit]

    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO candidates (
                    name, first_name, last_name, linkedin, headline, about, email, mobile_phone, phone,
                    location, city, skills, licenses_and_certifications, total_experience_years, avg_years_in_company, has_gap_years, 
                    has_education_gaps, has_industry_gaps, functional_experience_score, 
                    functional_experience_rationale, industry_experience_score, 
                    industry_experience_rationale, segment_experience_score, 
                    segment_experience_rationale, geography_experience_score, 
                    geography_experience_rationale, team_management_score, team_management_rationale,
                    max_people_managed, years_team_management, raw_fields, pool_source, created_at, updated_at, 
                    created_by, owner_user_id, normalized_linkedin, assigned_by_user_id
                ) VALUES (
                    %(name)s, %(first_name)s, %(last_name)s, %(linkedin)s, %(headline)s, %(about)s, %(email)s, %(mobile_phone)s, %(phone)s,
                    %(location)s, %(city)s, %(skills)s, %(licenses_and_certifications)s, %(total_experience_years)s, %(avg_years_in_company)s, %(has_gap_years)s, 
                    %(has_education_gaps)s, %(has_industry_gaps)s, %(functional_experience_score)s, 
                    %(functional_experience_rationale)s, %(industry_experience_score)s, 
                    %(industry_experience_rationale)s, %(segment_experience_score)s, 
                    %(segment_experience_rationale)s, %(geography_experience_score)s, 
                    %(geography_experience_rationale)s, %(team_management_score)s, %(team_management_rationale)s,
                    %(max_people_managed)s, %(years_team_management)s, %(raw_fields)s, 'catalog_from_upload', NOW(), NOW(), 
                    'catalog_upload', %(owner_user_id)s, %(normalized_linkedin)s, %(owner_user_id)s
                ) RETURNING id;
            """, {
                "name": trunc(candidate_data.get("fullName"), 255),
                "first_name": trunc(candidate_data.get("firstName"), 255),
                "last_name": trunc(candidate_data.get("lastName"), 255),
                "linkedin": trunc(candidate_data.get("linkedinPublicUrl"), 1024),
                "normalized_linkedin": trunc(candidate_data.get("linkedinProfileUrl") or candidate_data.get("linkedinPublicUrl"), 255),
                "headline": str(candidate_data.get("headline") or ''),
                "about": str(candidate_data.get("about") or ''),
                "email": trunc(candidate_data.get("Email"), 255),
                "mobile_phone": trunc(candidate_data.get("Mobile"), 50),
                "phone": trunc(candidate_data.get("Phone"), 50),
                "location": str(candidate_data.get("addressWithCountry") or ''),
                "city": trunc(candidate_data.get("city"), 100),
                "skills": str(candidate_data.get("skills") or ''),
                "licenses_and_certifications": str(candidate_data.get("certifications") or ''),
                "total_experience_years": metrics.get("total_experience_years") or 0,
                "avg_years_in_company": metrics.get("avg_tenure_years") or 0,
                "has_gap_years": False,
                "has_education_gaps": False,
                "has_industry_gaps": False,
                "functional_experience_score": payload.get("functional_experience_score"),
                "functional_experience_rationale": payload.get("functional_experience_rationale"),
                "industry_experience_score": payload.get("industry_experience_score"),
                "industry_experience_rationale": payload.get("industry_experience_rationale"),
                "segment_experience_score": payload.get("segment_experience_score"),
                "segment_experience_rationale": payload.get("segment_experience_rationale"),
                "geography_experience_score": payload.get("geography_experience_score"),
                "geography_experience_rationale": payload.get("geography_experience_rationale"),
                "team_management_score": payload.get("team_management_score"),
                "team_management_rationale": payload.get("team_management_rationale"),
                "max_people_managed": payload.get("profile_claims", {}).get("max_people_managed", 0),
                "years_team_management": payload.get("profile_claims", {}).get("years_team_management", 0),
                "raw_fields": json.dumps(candidate_data, default=str),
                "owner_user_id": user_id
            })
            candidate_id = cur.fetchone()[0]
            
            for role in roles:
                company_name = role.company.strip()
                company_id = None
                if company_name:
                    cur.execute("SELECT id FROM companies WHERE LOWER(name) = LOWER(%s) LIMIT 1", (company_name,))
                    c_row = cur.fetchone()
                    if c_row:
                        company_id = c_row[0]
                    else:
                        cur.execute("""
                            INSERT INTO companies (name, created_at, updated_at) 
                            VALUES (%s, NOW(), NOW()) RETURNING id;
                        """, (company_name,))
                        company_id = cur.fetchone()[0]
                
                cur.execute("""
                    INSERT INTO roles (candidate_id, company_id, title, details, duration_years)
                    VALUES (%s, %s, %s, %s, %s);
                """, (
                    candidate_id,
                    company_id,
                    trunc(role.title, 255),
                    json.dumps({
                        "company": role.company,
                        "start_date": role.start.isoformat() if role.start else None,
                        "end_date": role.end.isoformat() if role.end else None,
                        "details": role.details,
                        "duration_months": role.duration_months
                    }, default=str),
                    None if role.duration_unknown else round((role.duration_months or 0) / 12.0, 2)
                ))
            
            for ed in education:
                cur.execute("""
                    INSERT INTO education (candidate_id, college, degree, start_date, end_date, details)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    candidate_id,
                    trunc(ed.college, 255),
                    trunc(ed.degree, 255),
                    ed.start.isoformat() if ed.start else None,
                    ed.end.isoformat() if ed.end else None,
                    json.dumps({"details": ed.details}, default=str)
                ))
                
            cur.execute("""
                INSERT INTO recruitment_role_candidates (role_id, candidate_id, created_at)
                VALUES (%s, %s, NOW());
            """, (role_id, candidate_id))
            
            conn.commit()
            return candidate_id

def process_single_row_lenient(idx, row, user_id, role_id):
    start_time = time.perf_counter()
    try:
        raw_data = {str(k): str(v) for k, v in row.items() if not pd.isna(v)}
        candidate_dict = {"raw_fields": raw_data, "id": 0}
        
        # 1. Parse roles and education
        roles = parse_roles_from_raw(raw_data, None)
        education = parse_education_from_raw(raw_data)
        
        # 2. Calculate tenure metrics (with overlapping logic)
        tenure = calculate_tenure_metrics(roles, raw_total_exp_years=float(raw_data.get("totalExperienceYears") or 0))
        
        # 3. Classify companies
        company_contexts = _company_contexts_by_norm([candidate_dict], {0: roles})
        unique_companies = {r.company for r in roles if r.company}
        db_companies = _load_db_company_details(unique_companies)
        
        company_classification = {}
        for company in unique_companies:
            details = classify_company(company, role_texts=company_contexts.get(company.lower().strip(), []), db_details=db_companies.get(company, {}), allow_web=True)
            company_classification[company] = details
            
        for role in roles:
            details = company_classification.get(role.company, {})
            role.product_service = details.get("product_service") or "Unknown"
            role.industry = details.get("industry") or "Unknown"
            role.customer_segment = list(details.get("customer_segment") or [])
            role.business_model = details.get("business_model") or "Unknown"
            role.function = details.get("function") or "Unknown"

        candidate_dict["company_contexts"] = company_classification
        
        # 4. Extract profile claims and build payload
        profile_claims = extract_profile_claims(candidate_dict)
        
        payload = build_enrichment_payload(
            roles=roles, education=education, metrics=tenure,
            profile_claims=profile_claims, errors=[], contact_from_excel=False
        )
        
        # 5. Bypass the LLM Verifier Block and push to database directly
        candidate_id = _push_to_db_truncated(raw_data, roles, education, payload, tenure, user_id, role_id)
        duration = round(time.perf_counter() - start_time, 2)
        
        return True, f"[{idx}] [LENIENT_IMPORT] Inserted ID {candidate_id} | Time: {duration}s"
    except Exception as e:
        duration = round(time.perf_counter() - start_time, 2)
        return False, f"[{idx}] [ERROR] {e} | Time: {duration}s"

def main():
    file_path = "For Hayasa Product - BDR - Master List.xlsx"
    user_id, role_id = _setup_recruiter_and_role()
    
    print("Fetching existing candidates from database...")
    existing_normalized_urls = set()
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT normalized_linkedin FROM candidates WHERE owner_user_id = %s", (user_id,))
            for row in cur.fetchall():
                if row[0]:
                    existing_normalized_urls.add(normalize_url(row[0]))
                    
    print(f"Loaded {len(existing_normalized_urls)} existing candidates in DB.")
    
    print(f"Reading sheet '{file_path}'...")
    df = pd.read_excel(file_path)
    
    rows_to_process = []
    for idx, row in df.iterrows():
        url = row.get("linkedinPublicUrl")
        norm_url = normalize_url(url)
        if norm_url and norm_url not in existing_normalized_urls:
            rows_to_process.append((idx + 1, row))
            # Avoid duplicate runs for duplicate entries in sheet
            existing_normalized_urls.add(norm_url)
            
    print(f"Found {len(rows_to_process)} missing unique candidates to load.")
    
    if not rows_to_process:
        print("No missing candidates to load. All are already imported.")
        return
        
    success_count = 0
    error_count = 0
    start_time = time.perf_counter()
    
    print(f"Starting ingestion of {len(rows_to_process)} candidates using ThreadPool...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, row in rows_to_process:
            futures.append(executor.submit(process_single_row_lenient, idx, row, user_id, role_id))
            
        for future in as_completed(futures):
            success, msg = future.result()
            print(msg)
            sys.stdout.flush()
            if success:
                success_count += 1
            else:
                error_count += 1
                
    total_duration = round(time.perf_counter() - start_time, 2)
    print(f"\n--- COMPLETED IN {total_duration}s ---")
    print(f"Successfully processed and pushed: {success_count}/{len(rows_to_process)}")
    print(f"Failed due to runtime errors: {error_count}")
    
    # Touch main.py to reload server cache
    try:
        main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/main.py"))
        os.utime(main_py_path, None)
        print("Touched backend/main.py to reload server cache.")
    except Exception as e:
        print(f"Could not touch backend/main.py: {e}")

if __name__ == "__main__":
    main()

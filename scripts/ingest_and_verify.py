import sys
import os
import json
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from backend.db.connection import get_db_connection_context
from backend.services.import_enrichment import (
    parse_roles_from_raw, parse_education_from_raw, extract_profile_claims,
    _company_contexts_by_norm, _load_db_company_details, calculate_tenure_metrics,
    classify_company, build_enrichment_payload, clean_text
)
from backend.core.security import get_password_hash
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _setup_recruiter_and_role():
    email = "ashwin@growton.co"
    password = "Ashwin@123"
    name = "Ashwin"
    role_name = "BDR"
    
    with get_db_connection_context() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            if not row:
                hashed_pw = get_password_hash(password)
                cur.execute("""
                    INSERT INTO users (name, email, hashed_password, role, is_verified, created_at, is_active)
                    VALUES (%s, %s, %s, 'recruiter', true, NOW(), true)
                    RETURNING id;
                """, (name, email, hashed_pw))
                user_id = cur.fetchone()[0]
                print(f"Created recruiter {email} with ID {user_id}")
            else:
                user_id = row[0]
                print(f"Recruiter {email} already exists with ID {user_id}")
                
            cur.execute("SELECT id FROM recruitment_roles WHERE user_id = %s AND name = %s", (user_id, role_name))
            r_row = cur.fetchone()
            if not r_row:
                cur.execute("""
                    INSERT INTO recruitment_roles (user_id, name, created_at)
                    VALUES (%s, %s, NOW())
                    RETURNING id;
                """, (user_id, role_name))
                role_id = cur.fetchone()[0]
                print(f"Created recruitment role {role_name} with ID {role_id}")
            else:
                role_id = r_row[0]
                print(f"Recruitment role {role_name} already exists with ID {role_id}")
                
            conn.commit()
            return user_id, role_id

def _run_verifier(raw_data, enriched_claims):
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""You are a strict anti-hallucination verifying node.
Your job is to compare the raw data of a candidate against the extracted/enriched claims.

CRITICAL INSTRUCTIONS:
1. Candidate facts (tenure, education, titles) MUST be strictly justifiable by the raw data.
2. However, 'industry', 'segment', and 'functional' claims are often derived from the *companies* the candidate worked for. Do NOT flag these as hallucinations if they logically match the companies listed in the candidate's raw experience, even if the specific words (like 'Enterprise' or 'Mid-Market') are missing from the candidate's raw text.
3. If a job has no end date in the raw text, it is assumed the candidate still works there, so the backend sets the end date to '{today_str}' (today). Do NOT flag this as a hallucination.
4. Only output 'HALLUCINATION' if a claim explicitly contradicts the raw data or is completely unrelated to the candidate's known companies/roles.
Otherwise, output 'ACCURATE'.

RAW DATA:
{json.dumps(raw_data, indent=2, default=str)}

ENRICHED CLAIMS:
{json.dumps(enriched_claims, indent=2, default=str)}

Respond with a JSON object:
{{
  "status": "ACCURATE" or "HALLUCINATION",
  "reason": "Detailed explanation"
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result["tokens"] = response.usage.total_tokens
        result["cost"] = (response.usage.prompt_tokens * 0.005 / 1000) + (response.usage.completion_tokens * 0.015 / 1000)
        return result
    except Exception as e:
        return {"status": "ERROR", "reason": str(e), "cost": 0}

def _push_to_db(candidate_data, roles, education, payload, metrics, user_id, role_id):
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
                "name": str(candidate_data.get("fullName") or ''),
                "first_name": str(candidate_data.get("firstName") or ''),
                "last_name": str(candidate_data.get("lastName") or ''),
                "linkedin": str(candidate_data.get("linkedinPublicUrl") or ''),
                "normalized_linkedin": str(candidate_data.get("linkedinProfileUrl") or candidate_data.get("linkedinPublicUrl") or ''),
                "headline": str(candidate_data.get("headline") or ''),
                "about": str(candidate_data.get("about") or ''),
                "email": str(candidate_data.get("Email") or ''),
                "mobile_phone": str(candidate_data.get("Mobile") or ''),
                "phone": str(candidate_data.get("Phone") or ''),
                "location": str(candidate_data.get("addressWithCountry") or ''),
                "city": str(candidate_data.get("city") or ''),
                "skills": str(candidate_data.get("skills") or ''),
                "licenses_and_certifications": str(candidate_data.get("certifications") or ''),
                "total_experience_years": metrics.get("total_experience_years") or 0,
                "avg_years_in_company": metrics.get("avg_tenure_years") or 0,
                "has_gap_years": False, # TODO: implement gap year logic if needed
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
                    role.title,
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
                """, (candidate_id, ed.college, ed.degree, ed.start.isoformat() if ed.start else None, ed.end.isoformat() if ed.end else None, json.dumps({"details": ed.details}, default=str)))
                
            cur.execute("""
                INSERT INTO recruitment_role_candidates (role_id, candidate_id, created_at)
                VALUES (%s, %s, NOW());
            """, (role_id, candidate_id))
            
            conn.commit()
            return candidate_id

def process_single_row(idx, row, user_id, role_id):
    start_time = time.perf_counter()
    try:
        raw_data = {str(k): str(v) for k, v in row.items() if not pd.isna(v)}
        candidate_dict = {"raw_fields": raw_data, "id": 0}
        
        roles = parse_roles_from_raw(raw_data, None)
        education = parse_education_from_raw(raw_data)
        
        tenure = calculate_tenure_metrics(roles, raw_total_exp_years=float(raw_data.get("totalExperienceYears") or 0))
        
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
        profile_claims = extract_profile_claims(candidate_dict)
        
        payload = build_enrichment_payload(
            roles=roles, education=education, metrics=tenure,
            profile_claims=profile_claims, errors=[], contact_from_excel=False
        )
        
        verification = _run_verifier(raw_data, payload)
        duration = round(time.perf_counter() - start_time, 2)
        cost = verification.get('cost', 0)
        
        if verification['status'] == "ACCURATE":
            candidate_id = _push_to_db(raw_data, roles, education, payload, tenure, user_id, role_id)
            return True, f"[{idx}] [ACCURATE] Inserted ID {candidate_id} | Time: {duration}s | Cost: ${cost:.4f}", cost
        else:
            return False, f"[{idx}] [HALLUCINATION] {verification['reason']} | Time: {duration}s | Cost: ${cost:.4f}", cost
    except Exception as e:
        duration = round(time.perf_counter() - start_time, 2)
        return False, f"[{idx}] [ERROR] {e} | Time: {duration}s", 0

def process_file(file_path, limit=None):
    user_id, role_id = _setup_recruiter_and_role()
    
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path, nrows=limit)
    df = df.iloc[500:] # Skip already processed rows
    
    success_count = 0
    hallucination_count = 0
    total_cost = 0
    
    # Fix import deadlock by forcing openai module to load fully
    try:
        client.models.list()
    except Exception:
        pass
        
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(process_single_row, idx + 1, row, user_id, role_id))
            
        for future in as_completed(futures):
            success, msg, cost = future.result()
            print(msg)
            total_cost += cost
            if success:
                success_count += 1
            else:
                hallucination_count += 1
            
    total_duration = round(time.perf_counter() - start_time, 2)
    print(f"\n--- COMPLETED IN {total_duration}s ---")
    print(f"Successfully processed and pushed: {success_count}/{len(df) if limit is None else limit}")
    print(f"Failed/Skipped: {hallucination_count}")
    print(f"Total API Cost for processed rows: ${total_cost:.4f}")
    
    if len(df) > 0:
        print(f"Average Speed: {round(total_duration/len(df), 2)}s per profile")
        print(f"Average Cost: ${total_cost/len(df):.4f} per profile")
 
if __name__ == "__main__":
    process_file("For Hayasa Product - BDR - Master List.xlsx", limit=None)

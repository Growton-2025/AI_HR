from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from backend.pipeline.query import update_candidate_status, PROFILES_BY_ID

router = APIRouter()

class StatusUpdate(BaseModel):
    status: str

class NotesUpdate(BaseModel):
    notes: str

RECRUITMENT_STAGES = [
    'To be started', 'Shortlisted', 'Rejected', 'For Future', 
    'Reached out - Linkedin', 'Reached out - Phone', 'Not Interested', 
    'Followup / In conversation', 'Shortlist - Rejected', 'High CTC', 
    'Duplicate', 'Not responding', 'Internal Review', 'Shared with customer'
]


# --- Industry Keywords for Extraction ---
INDUSTRY_KEYWORDS = [
    "SaaS", "Cloud", "Fintech", "HRTech", "MarTech", "AdTech", "EdTech", "PropTech", "HealthTech", "CyberSecurity",
    "AI", "Machine Learning", "Data Science", "Big Data", "Blockchain", "Crypto", "E-commerce", "Retail",
    "FMCG", "Banking", "Insurance", "Payments", "Logistics", "Supply Chain", "Manufacturing", "Automotive",
    "Telecom", "Real Estate", "Healthcare", "Pharma", "Biotech", "Energy", "Gaming", "IT Services"
]


@router.get("/candidates/browse")
async def browse_candidates(
    # Pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100),
    # Search
    q: Optional[str] = None,
    # Filters
    title: Optional[str] = None,
    company: Optional[str] = None,
    city: Optional[str] = None,
    location_type: Optional[str] = None,   # On-site | Remote | Hybrid
    product_service: Optional[str] = None,
    status: Optional[str] = None,          # shortlisted | rejected | etc.
    min_exp: Optional[float] = None,
    max_exp: Optional[float] = None,
    min_avg_tenure: Optional[float] = None, # stability
    # Sort
    sort_by: Optional[str] = "name",
    sort_dir: Optional[str] = "asc",
):
    """Browse all candidates with filtering, search, and pagination"""
    all_profiles = list(PROFILES_BY_ID.values())

    # --- Semantic Search for Expertise ---
    semantic_scores = {}
    if product_service and product_service.strip():
        from backend.pipeline.query import get_semantic_scores
        semantic_scores = await get_semantic_scores(product_service)

    results = []
    for p in all_profiles:
        profile_id = p.get("id")
        score = semantic_scores.get(profile_id, 0.0) if semantic_scores else None

        primary_role = (p.get("roles") or [{}])[0]
        city_val = (p.get("location") or "").split(",")[0].strip()
        loc_val = p.get("work_preference") or p.get("location_type") or ""
        company_product = (primary_role.get("company_details") or {}).get("product_service") or ""
        extracted_prod = p.get("extracted_industry") or ""
        cand_services = p.get("candidate_services") or ""
        
        product_val = extracted_prod or cand_services or company_product or primary_role.get("industry") or ""
        
        if not product_val:
            search_text = f"{p.get('headline') or ''} {p.get('about') or ''}"
            found = []
            for kw in INDUSTRY_KEYWORDS:
                if kw.lower() in search_text.lower():
                    found.append(kw)
            if found:
                product_val = ", ".join(list(set(found))[:3])
        
        exp_val = p.get("total_experience_years") or 0
        stability_val = p.get("avg_years_in_company") or 0
        status_val = p.get("status") or ""
        name_val = p.get("name") or ""
        title_val = primary_role.get("title") or p.get("headline") or ""
        company_val = primary_role.get("company") or ""

        # --- Filter logic ---
        if q:
            ql = q.lower()
            searchable = f"{name_val} {title_val} {company_val} {city_val}".lower()
            if ql not in searchable:
                continue

        # Multi-value filter helper
        def matches_filter(filter_str, target_val, field_name=""):
            if not filter_str: return True
            filter_vals = [v.strip().lower() for v in filter_str.split(",") if v.strip()]
            if not filter_vals: return True
            if not target_val: return False
            tv_lower = str(target_val).lower()
            return any(v in tv_lower for v in filter_vals)

        if not matches_filter(title, title_val, "title"):
            continue
        if not matches_filter(company, company_val, "company"):
            continue
        if not matches_filter(city, city_val, "city"):
            continue
        if not matches_filter(location_type, loc_val):
            continue
        
        # If semantic scores are NOT present, fallback to traditional substring match
        if not semantic_scores and product_service and not matches_filter(product_service, product_val):
            continue
        
        # If semantic scores ARE present, only keep those with a decent score (e.g., > 0.4)
        if semantic_scores and score is not None and score < 0.45:
            continue

        results.append({
            "id": profile_id,
            "name": name_val,
            "first_name": (name_val.split() or [""])[0],
            "last_name": (" ".join(name_val.split()[1:]) or ""),
            "linkedin": p.get("linkedin"),
            "email": p.get("email") or "",
            "phone": p.get("phone") or p.get("mobile_phone") or "",
            "response": p.get("response", ""),
            "notes": p.get("notes", ""),
            "title": title_val,
            "company": company_val,
            "product_service": product_val,
            "city": city_val,
            "location_type": loc_val,
            "total_experience_years": round(float(exp_val), 1) if exp_val else 0,
            "avg_tenure_years": round(float(stability_val), 1) if stability_val else 0,
            "status": status_val,
            "li_status": p.get("li_status") or "",
            "heyreach_campaign_id": p.get("heyreach_campaign_id") or "",
            "email_campaign_id": p.get("email_campaign_id") or "",
            "message_sent_count": p.get("message_sent_count") or 0,
            "li_sent_count": p.get("li_sent_count") or 0,
            "headline": p.get("headline") or "",
            "match_score": round(score * 100, 1) if score is not None else None,
        })

    # Filter by experience
    if min_exp is not None:
        results = [r for r in results if r["total_experience_years"] >= min_exp]
    if max_exp is not None:
        results = [r for r in results if r["total_experience_years"] <= max_exp]
    if min_avg_tenure is not None:
        results = [r for r in results if r["avg_tenure_years"] >= min_avg_tenure]

    # --- Status counts for tabs (calculate BEFORE status filter) ---
    status_counts = {}
    for r in results:
        s = (r.get("status") or "").strip()
        if s:
            status_counts[s] = status_counts.get(s, 0) + 1
        else:
            status_counts["To be started"] = status_counts.get("To be started", 0) + 1

    # Filter by status
    if status and status.strip():
        status_vals = [s.strip().lower() for s in status.split(",") if s.strip()]
        if status_vals:
            results = [r for r in results if (r.get("status") or "To be started").strip().lower() in status_vals]

    # --- Sorting ---
    if semantic_scores:
        # Default to match score if semantic search is active
        results.sort(key=lambda x: x.get("match_score") or 0, reverse=True)
    else:
        sort_key_map = {
            "name": "name",
            "title": "title",
            "company": "company",
            "city": "city",
            "exp": "total_experience_years",
            "tenure": "avg_tenure_years",
        }
        key = sort_key_map.get(sort_by, "name")
        results.sort(key=lambda x: (x.get(key) or ""), reverse=(sort_dir == "desc"))

    # --- Pagination ---
    total = len(results)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_results = results[offset: offset + page_size]

    return {
        "candidates": page_results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "status_counts": status_counts,
        "is_semantic_search": bool(semantic_scores),
    }


@router.get("/candidates/browse/meta")
async def browse_metadata():
    """Return unique filter values (for dropdowns)"""
    all_profiles = list(PROFILES_BY_ID.values())

    companies, cities, titles, products, locations, statuses = set(), set(), set(), set(), set(), set()
    for p in all_profiles:
        primary_role = (p.get("roles") or [{}])[0]
        if c := primary_role.get("company"): companies.add(c)
        if ci := (p.get("location") or "").split(",")[0].strip(): cities.add(ci)
        if t := (primary_role.get("title") or p.get("headline") or ""): titles.add(t)
        
        # Combine local services and company product/service
        if cp := (primary_role.get("company_details") or {}).get("product_service"): products.add(cp)
        if cs := p.get("candidate_services"): 
            for s in cs.split(","):
                s_strip = s.strip()
                if s_strip and len(s_strip) < 50: products.add(s_strip)
        
        if ei := p.get("extracted_industry"): products.add(ei)
        
        # Add fallback terms to metadata
        search_text = f"{p.get('headline') or ''} {p.get('about') or ''}"
        for kw in INDUSTRY_KEYWORDS:
            if kw.lower() in search_text.lower():
                products.add(kw)
        
        if lc := (p.get("work_preference") or p.get("location_type") or ""): locations.add(lc)
    
    # Ensure all professional stages are in the filter
    for s in RECRUITMENT_STAGES:
        statuses.add(s)

    return {
        "companies": sorted(companies)[:100],
        "cities": sorted(cities)[:100],
        "titles": sorted(titles)[:100],
        "products": sorted(products)[:100],
        "location_types": sorted(locations),
        "statuses": sorted(statuses),
    }

@router.post("/candidates/{candidate_id}/status")
async def update_status(candidate_id: int, update: StatusUpdate):
    success = update_candidate_status(candidate_id, update.status)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update candidate status")
    return {"message": "Status updated successfully"}

@router.patch("/candidates/{candidate_id}/notes")
async def update_notes(candidate_id: int, update: NotesUpdate):
    from backend.pipeline.query import update_candidate_notes
    success = update_candidate_notes(candidate_id, update.notes)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update notes")
    return {"message": "Notes updated successfully"}

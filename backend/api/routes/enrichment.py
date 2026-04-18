
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
import logging

from backend.db.connection import get_db_connection, return_db_connection
from backend.services.clay import trigger_clay

router = APIRouter()
logger = logging.getLogger(__name__)

def clean_val(val):
    """Cleaning logic from user's script."""
    if not val or str(val).lower() in ['none found', 'not found', 'undefined', 'null', '', 'n/a']:
        return None
    return val

@router.post("/enrich/{candidate_id}")
async def enrich_candidate(candidate_id: int, background_tasks: BackgroundTasks):
    """Trigger enrichment for a candidate - checks cache first to save money!"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cur:
            # Fetch candidate info INCLUDING existing email/phone
            cur.execute("""
                SELECT first_name, last_name, name, linkedin, email, mobile_phone 
                FROM candidates WHERE id = %s
            """, (candidate_id,))
            row = cur.fetchone()
            
        if not row:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        first_name = row[0]
        last_name = row[1]
        full_name = row[2]
        linkedin_url = row[3]
        existing_email = row[4]
        existing_phone = row[5]
        
        # If we already have BOTH email and phone, skip Clay!
        if existing_email and existing_phone:
            logger.info(f"💰 CACHE HIT: {full_name} already has email & phone - skipping Clay!")
            return {
                "status": "cached", 
                "message": f"Already enriched: {full_name}",
                "email": existing_email,
                "phone": existing_phone
            }
        
        if not first_name or not last_name:
            parts = full_name.split(" ", 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else ""

        logger.info(f"🔍 No cached data for {full_name} - calling Clay...")
        background_tasks.add_task(trigger_clay, first_name, last_name, linkedin_url)
        
        return {"status": "processing", "message": f"Enrichment started for {first_name} {last_name}"}
        
    finally:
        return_db_connection(conn)

@router.post("/results")
async def receive_results(request: Request):
    """
    Listens for Clay results - exact same logic as user's script.
    Endpoint: /api/results
    """
    data = await request.json()
    
    first = data.get('first_name', 'N/A')
    last = data.get('last_name', 'N/A')
    email = clean_val(data.get('result_email'))
    phone = clean_val(data.get('mobile_phone'))
    li_url = data.get('linkedin_url', 'N/A')

    logger.info("─" * 50)
    logger.info("  ENRICHMENT RESULT")
    logger.info("─" * 50)
    logger.info(f"  Name:      {first} {last}")
    logger.info(f"  LinkedIn:  {li_url}")
    logger.info(f"  Email:     {email if email else 'Not Found'}")
    logger.info(f"  Mobile:    {phone if phone else 'Not Found'}")
    logger.info("─" * 50)

    # Update DB
    if li_url and li_url != 'N/A':
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE candidates 
                        SET email = COALESCE(%s, email), 
                            mobile_phone = COALESCE(%s, mobile_phone)
                        WHERE linkedin = %s
                    """, (email, phone, li_url))
                    conn.commit()
                    logger.info(f"Updated candidate in DB for LinkedIn: {li_url}")
            finally:
                return_db_connection(conn)

            # Update the in-memory cache so frontend polling works
            from backend.pipeline.query import update_candidate_contact
            update_candidate_contact(li_url, email, phone)

    return {"status": "success"}

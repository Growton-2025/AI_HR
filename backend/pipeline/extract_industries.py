import os
import json
import logging
import psycopg2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.db.connection import get_db_connection, return_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
BATCH_SIZE = 50
MODEL_NAME = "gpt-4o-mini"

# System Prompt
SYSTEM_PROMPT = """You are an expert recruiter. 
Your task is to identify the specific 'Product' or 'Service' NAME the candidate works with.

Rules:
1. Prioritize exact BRAND NAMES or PRODUCT LINES mentioned (e.g., "Informatica", "Salesforce Marketing Cloud", "Tally ERP", "Talent Solutions", "Google Cloud").
2. Only use categories (like "SaaS", "Fintech") if no specific product name is found.
3. Return ONLY the name(s). Max 3 words. 
4. Do NOT use full sentences or descriptions like "Customer relationship management...".
5. Use comma-separated values if multiple products are found (e.g., "Azure, Office 365").
"""

def extract_industries():
    conn = get_db_connection()
    if not conn:
        return

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Headline: {headline}\nAbout: {about}")
    ])
    chain = prompt | llm

    try:
        with conn.cursor() as cur:
            # Identifies candidates - we re-run all that don't have a high-quality product/service assigned
            cur.execute("""
                SELECT c.id, c.name, c.headline, c.about, r.title
                FROM candidates c
                LEFT JOIN roles r ON r.candidate_id = c.id AND r.is_primary = true
            """)
            candidates = cur.fetchall()
            
            logger.info(f"Found {len(candidates)} candidates for product/service extraction.")
            
            count = 0
            for cand_id, name, headline, about, title in candidates:
                try:
                    if not headline and not about and not title:
                        industry = "IT Services"
                    else:
                        response = chain.invoke({
                            "headline": f"{title or ''} - {headline or ''}", 
                            "about": about or ""
                        })
                        industry = response.content.strip()
                    
                    # Update DB
                    cur.execute("""
                        UPDATE candidates 
                        SET raw_fields = jsonb_set(
                            COALESCE(raw_fields, '{}'::jsonb), 
                            '{extracted_industry}', 
                            %s::jsonb
                        )
                        WHERE id = %s
                    """, (json.dumps(industry), cand_id))
                    
                    count += 1
                    if count % 10 == 0:
                        conn.commit()
                        logger.info(f"Processed {count}/{len(candidates)} candidates...")
                except Exception as e:
                    logger.error(f"Error processing candidate {name} ({cand_id}): {e}")
                    continue
            
            conn.commit()
            logger.info("Extraction complete!")

    except Exception as e:
        logger.error(f"Failed to extract industries: {e}")
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    extract_industries()

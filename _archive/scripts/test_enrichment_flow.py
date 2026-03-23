
import requests
import json
import time
from backend.db.connection import get_db_connection

BASE_URL = "http://localhost:8000/api"

def get_test_candidate():
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Get a candidate who has a LinkedIn URL
        cur.execute("SELECT id, name, linkedin FROM candidates WHERE linkedin IS NOT NULL LIMIT 1")
        return cur.fetchone()
    conn.close()

def verify_db_update(candidate_id):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT email, mobile_phone FROM candidates WHERE id = %s", (candidate_id,))
        return cur.fetchone()
    conn.close()

def run_test():
    print("--- Starting Enrichment Flow Test ---")
    
    # 1. Get Candidate
    candidate = get_test_candidate()
    if not candidate:
        print("FAIL: No candidates with LinkedIn found in DB.")
        return
    
    cid, name, linkedin = candidate
    print(f"Target Candidate: {name} (ID: {cid})")
    
    # 2. Simulate Frontend Trigger
    print(f"\n[1] Simulating User Selection (POST /enrich/{cid})...")
    try:
        res = requests.post(f"{BASE_URL}/enrich/{cid}")
        print(f"Response: {res.status_code} - {res.json()}")
    except Exception as e:
        print(f"FAIL: Could not connect to backend: {e}")
        return

    # 3. Simulate Clay Webhook Callback
    # We fake what Clay would send back after finding data
    fake_clay_payload = {
        "candidate_id": cid,
        "linkedin_url": linkedin,
        "result_email": "test.user@example.com",
        "mobile_phone": "+1-555-0199"
    }
    
    print(f"\n[2] Simulating Clay Webhook (POST /webhooks/clay)...")
    try:
        res = requests.post(f"{BASE_URL}/webhooks/clay", json=fake_clay_payload)
        print(f"Response: {res.status_code} - {res.json()}")
    except Exception as e:
        print(f"FAIL: Webhook connection failed: {e}")
        return

    # 4. Verify DB
    print(f"\n[3] Verifying Database Update...")
    email, phone = verify_db_update(cid)
    print(f"DB Email: {email}")
    print(f"DB Phone: {phone}")
    
    if email == "test.user@example.com" and phone == "+1-555-0199":
        print("\nSUCCESS: Candidate enriched successfully! 🚀")
    else:
        print("\nFAIL: Database was not updated correctly.")

if __name__ == "__main__":
    # Wait a bit for server to come up if just restarted
    time.sleep(2) 
    run_test()

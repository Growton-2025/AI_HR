
import requests
import time
import os

BASE_URL = "http://localhost:8001/api"
LOG_FILE = "backend_log.txt"

def find_unenriched_candidate():
    print("🔍 Looking for a candidate without email/phone...")
    try:
        # Fetch first batch of candidates
        res = requests.get(f"{BASE_URL}/candidates?limit=50")
        if res.status_code != 200:
            print(f"❌ Failed to fetch candidates: {res.status_code}")
            return None
            
        candidates = res.json().get("candidates", [])
        
        for c in candidates:
            cid = c.get("id")
            # Get full details to check email/phone
            detail_res = requests.get(f"{BASE_URL}/candidates/{cid}")
            if detail_res.status_code == 200:
                detail = detail_res.json()
                email = detail.get("email")
                phone = detail.get("mobile_phone")
                
                # Check if missing either
                if not email or not phone:
                    print(f"✅ Found target: {detail.get('name')} (ID: {cid})")
                    print(f"   Missing: {'Email' if not email else ''} {'Phone' if not phone else ''}")
                    return cid
                    
        print("⚠️ All 50 checked candidates already have info. Trying ID 100 as fallback.")
        return 100
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return None

def monitor_logs(start_time, duration=45):
    print(f"👀 Monitoring {LOG_FILE} for {duration} seconds...")
    end_time = time.time() + duration
    
    # Read current file size to skip old logs
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(0, 2)
            last_pos = f.tell()
    except:
        last_pos = 0
        
    while time.time() < end_time:
        try:
            with open(LOG_FILE, 'r') as f:
                f.seek(last_pos)
                new_data = f.read()
                if new_data:
                    last_pos = f.tell()
                    lines = new_data.split('\n')
                    for line in lines:
                        if "ENRICHMENT RESULT" in line:
                            print("\n🎉 SUCCESS! Enrichment Result Received from Clay!")
                            print(line)
                            # Print confirmation of data
                            return True
                        if "Clay Webhook Status: " in line:
                            print(f"   -> Outbound webhook sent: {line.strip()}")
        except Exception as e:
            pass
            
        time.sleep(1)
        
    print("\n⏰ Timed out waiting for return data.")
    return False

def main():
    candidate_id = find_unenriched_candidate()
    if not candidate_id:
        return
        
    print(f"🚀 Triggering enrichment for ID: {candidate_id}")
    start_time = time.time()
    
    try:
        res = requests.post(f"{BASE_URL}/enrich/{candidate_id}")
        print(f"   API Response: {res.json()}")
        
        if res.json().get("status") == "cached":
            print("   ⚠️ This candidate was already cached. Picking another might be better, but let's see.")
        
        success = monitor_logs(start_time)
        
        if success: 
            print("\n✅ End-to-End Test PASSED: App -> Clay -> App")
        else:
            print("\n❌ Test FAILED: Data did not return within 45 seconds.")
            print("   Please check your Clay table run history.")
            
    except Exception as e:
        print(f"❌ Error triggering: {e}")

if __name__ == "__main__":
    main()

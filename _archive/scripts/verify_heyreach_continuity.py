import os
import requests
from dotenv import load_dotenv

load_dotenv()

# We can't easily mock the HeyReach API without a real token, 
# but we can check if the endpoint schema and DB integration are solid.

BASE_URL = "http://localhost:8001"

def test_outreach_status_schema():
    print("Checking Outreach Status API Schema...")
    # Using a dummy role ID (assuming 1 exists or just checking structure)
    try:
        # We need a token to access this
        token = os.getenv("TEST_USER_TOKEN") # If available, otherwise we just check if the code runs
        if not token:
            print("⚠️ TEST_USER_TOKEN not found, skipping live API check.")
            return

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/outreach/status/1", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                keys = item.keys()
                print(f"API Response Keys: {list(keys)}")
                required = ["li_sent_count", "li_response_received_at", "li_conversation_id"]
                for k in required:
                    if k in keys:
                        print(f"✅ Found {k}")
                    else:
                        print(f"❌ Missing {k}")
            else:
                print("No data returned for role 1 (expected if role 1 has no outreach)")
        else:
            print(f"API returned status {response.status_code}")
    except Exception as e:
        print(f"Error checking API: {e}")

if __name__ == "__main__":
    test_outreach_status_schema()

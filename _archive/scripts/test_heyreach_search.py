
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
PROFILE_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_get_campaigns():
    url = "https://api.heyreach.io/api/public/campaign/GetCampaignsForLead"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "profileUrl": PROFILE_URL,
        "offset": 0,
        "limit": 100
    }
    
    print(f"Searching for campaigns for lead: {PROFILE_URL}")
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print("Response Data:")
        import json
        print(json.dumps(data, indent=2))
        
        if isinstance(data, list) and len(data) > 0:
            print(f"\n✅ Found {len(data)} campaigns for this lead.")
        else:
            print("\n❌ No campaigns found for this lead.")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_get_campaigns()

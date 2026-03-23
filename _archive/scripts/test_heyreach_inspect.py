
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_inspect_leads():
    url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "campaignId": CAMPAIGN_ID,
        "limit": 100
    }
    
    print(f"Fetching leads for campaign: {CAMPAIGN_ID}")
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        
        if isinstance(data, list):
            print(f"Found {len(data)} leads.")
            for i, entry in enumerate(data):
                lead = entry.get('lead', {})
                p_url = lead.get('profileUrl') or lead.get('profile_url')
                print(f"Lead {i}: {p_url}")
                
                if p_url and TARGET_URL.lower().rstrip('/') in p_url.lower().rstrip('/'):
                    print("✅ MATCH FOUND!")
                    print(json.dumps(entry, indent=2))
        else:
            print("Response is not a list:")
            print(json.dumps(data, indent=2))
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_inspect_leads()


import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_inspect_keys():
    url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "campaignId": CAMPAIGN_ID,
        "limit": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        items = data.get('items', [])
        
        for entry in items:
            lead = entry.get('lead', {})
            p_url = lead.get('profileUrl') or lead.get('profile_url')
            if p_url and TARGET_URL.lower().rstrip('/') in p_url.lower().rstrip('/'):
                print("✅ Found target lead. Entry keys:")
                print(list(entry.keys()))
                print("\nLead keys:")
                print(list(lead.keys()))
                
                # Check for messages
                print(f"\nRecent messages: {entry.get('recentMessages') or entry.get('recent_messages')}")
                print(f"\nStatus: {entry.get('status')} ({entry.get('leadCampaignStatus')})")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_inspect_keys()

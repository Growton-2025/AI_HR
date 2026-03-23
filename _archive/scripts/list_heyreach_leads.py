
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428

def list_all_leads():
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
        print(f"Top level keys: {list(data.keys())}")
        items = data.get('items', [])
        print(f"Total items: {len(items)}")
        
        for i, entry in enumerate(items):
            lead = entry.get('lead', {})
            p_url = lead.get('profileUrl') or lead.get('profile_url')
            print(f"[{i}] URL: {p_url}")
            if i == 0:
                print("Sample Entry Keys:")
                print(list(entry.keys()))
                print("Most recent message keys:")
                recent = entry.get('recentMessages') or entry.get('recent_messages', [])
                if recent:
                    print(list(recent[0].keys()))
                    print(f"Message Text: {recent[0].get('message') or recent[0].get('text')}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    list_all_leads()

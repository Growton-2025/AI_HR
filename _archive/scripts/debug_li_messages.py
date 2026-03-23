
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def debug_messages():
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
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        items = data.get('items', [])
        
        for entry in items:
            profile = entry.get('linkedInUserProfile') or {}
            raw_url = (profile.get('profileUrl') or profile.get('profile_url') or "").lower().rstrip('/')
            
            if TARGET_URL.lower().rstrip('/') in raw_url:
                print(f"Found match: {raw_url}")
                print(f"Lead Message Status: {entry.get('leadMessageStatus')}")
                print(f"Lead Connection Status: {entry.get('leadConnectionStatus')}")
                
                recent = entry.get('recentMessages') or entry.get('recent_messages', [])
                print(f"Recent Messages Count: {len(recent)}")
                print("Messages JSON:")
                print(json.dumps(recent, indent=2))
                
                # Check for other message fields
                for key in entry.keys():
                    if "message" in key.lower() or "text" in key.lower():
                        print(f"Field '{key}': {entry[key]}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_messages()

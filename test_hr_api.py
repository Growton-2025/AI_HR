import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HEYREACH_API_KEY")

headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json",
    "accept": "application/json"
}

def test_endpoint(name, method, url, json_payload=None):
    print(f"\n--- Testing {name} ---")
    try:
        res = requests.request(method, url, headers=headers, json=json_payload, timeout=10)
        print(f"Status: {res.status_code}")
        try:
            print("Response:", res.json())
        except:
            print("Response text:", res.text[:200])
    except Exception as e:
        print("Error:", e)

# 1. GetAll campaigns
test_endpoint("GetAll Campaigns", "POST", "https://api.heyreach.io/api/public/campaign/GetAll", {"offset": 0, "limit": 1})

# 2. GetLeadsFromCampaign
test_endpoint("GetLeadsFromCampaign", "POST", "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign", {"campaignId": 379659, "limit": 1})

# 3. GetConversationsV2
test_endpoint("GetConversationsV2", "POST", "https://api.heyreach.io/api/public/inbox/GetConversationsV2", {"filters": {"campaignIds": [379659]}, "offset": 0, "limit": 10})

# 4. GetConversationsV2 (no filter)
test_endpoint("GetConversationsV2 (No filters)", "POST", "https://api.heyreach.io/api/public/inbox/GetConversationsV2", {"offset": 0, "limit": 10})


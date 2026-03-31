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

payload = {
    "filters": {"campaignIds": [379659]},
    "offset": 0,
    "limit": 50
}

url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"

try:
    print("Testing HeyReach API...")
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    print("Status Code:", res.status_code)
    
    if res.status_code == 200:
        data = res.json()
        print("Total Items:", data.get('totalItems'))
        print("First few items:", data.get('items', [])[:2])
    else:
        print("Response:", res.text)
        
except Exception as e:
    print("Error:", e)

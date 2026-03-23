
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_conv_v2_optimized():
    url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    # Try with leadProfileUrl in filters
    payload = {
        "filters": {
            "leadProfileUrl": TARGET_URL
        },
        "offset": 0,
        "limit": 5
    }
    
    print(f"Requesting conversation for: {TARGET_URL}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Status: {response.status_code}")
        data = response.json()
        
        items = data.get('items', [])
        print(f"Found {len(items)} items.")
        
        for item in items:
            print(f"Conversation ID: {item.get('id')}")
            print(f"Last Message: {item.get('lastMessageText')}")
            print(f"Sender: {item.get('lastMessageSender')}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    test_conv_v2_optimized()

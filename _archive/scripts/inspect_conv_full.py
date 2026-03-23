
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def inspect_conversation():
    url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "filters": {"leadProfileUrl": TARGET_URL},
        "limit": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        items = data.get('items', [])
        if items:
            print(json.dumps(items[0], indent=2))
        else:
            print("No conversation found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_conversation()

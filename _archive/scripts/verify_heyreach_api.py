
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
headers = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json",
    "accept": "application/json"
}

def check_endpoint(name, url, method="POST", payload=None):
    print(f"Checking {name}: {url}...")
    try:
        if method == "POST":
            res = requests.post(url, headers=headers, json=payload or {}, timeout=10)
        else:
            res = requests.get(url, headers=headers, timeout=10)
            
        print(f"Result for {name}: {res.status_code}")
        if res.status_code != 404:
            print(f"Response: {res.text[:200]}")
        return res.status_code
    except Exception as e:
        print(f"Error checking {name}: {e}")
        return None

if __name__ == "__main__":
    # Test common variations
    check_endpoint("SendMessage", "https://api.heyreach.io/api/public/inbox/SendMessage")
    check_endpoint("sendMessage", "https://api.heyreach.io/api/public/inbox/sendMessage")
    check_endpoint("SendMessage (no public)", "https://api.heyreach.io/api/inbox/SendMessage")
    check_endpoint("Send (InboxV2)", "https://api.heyreach.io/api/public/inbox/Send")
    
    # Try to find a conversation to test with real data if possible
    conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    res = requests.post(conv_url, headers=headers, json={"limit": 1})
    if res.status_code == 200:
        items = res.json().get('items', [])
        if items:
            conv = items[0]
            conv_id = conv.get('id')
            acc_id = conv.get('linkedInAccountId') or conv.get('linkedInAccount', {}).get('id')
            print(f"Found conversation {conv_id} on account {acc_id}")
            
            # Now try SendMessage with these IDs
            payload = {
                "linkedInAccountId": acc_id,
                "conversationId": conv_id,
                "message": "API Test"
            }
            check_endpoint("SendMessage with Real IDs", "https://api.heyreach.io/api/public/inbox/SendMessage", payload=payload)

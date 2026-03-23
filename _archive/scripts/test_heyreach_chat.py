
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_fetch_messages():
    # Step 1: Get Conversation ID
    conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    conv_payload = {
        "filters": {
            "campaignIds": [CAMPAIGN_ID],
            "leadProfileUrl": TARGET_URL
        },
        "offset": 0,
        "limit": 1
    }
    
    print(f"Step 1: Finding conversation for {TARGET_URL}...")
    try:
        response = requests.post(conv_url, headers=headers, json=conv_payload)
        response.raise_for_status()
        data = response.json()
        items = data.get('items', [])
        
        if not items:
            print("❌ No conversation found for this lead/campaign.")
            return
            
        conv = items[0]
        conversation_id = conv.get('id')
        account_id = conv.get('linkedInAccountId')
        
        print(f"✅ Found Conversation! ID: {conversation_id}, Account: {account_id}")
        
        # Step 2: Get Chatroom Messages
        chat_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{account_id}/{conversation_id}"
        print(f"Step 2: Fetching messages from {chat_url}...")
        
        chat_response = requests.get(chat_url, headers=headers)
        chat_response.raise_for_status()
        chat_data = chat_response.json()
        
        messages = chat_data.get('messages', [])
        print(f"✅ Retrieved {len(messages)} messages.")
        
        for msg in messages:
            sender = msg.get('sender')
            body = msg.get('body')
            time = msg.get('createdAt')
            print(f"[{time}] {sender}: {body}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    test_fetch_messages()

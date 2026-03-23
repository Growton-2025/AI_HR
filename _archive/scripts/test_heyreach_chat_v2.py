
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
CAMPAIGN_ID = 332428
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

def test_fetch_messages_v2():
    # Step 1: Get ALL Conversations for the campaign
    conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    conv_payload = {
        "filters": {
            "campaignIds": [CAMPAIGN_ID]
        },
        "offset": 0,
        "limit": 50
    }
    
    print(f"Step 1: Fetching conversations for campaign {CAMPAIGN_ID}...")
    try:
        response = requests.post(conv_url, headers=headers, json=conv_payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get('items', [])
        
        print(f"✅ Received {len(items)} conversations.")
        
        target_conv = None
        for conv in items:
            participants = conv.get('participants', [])
            for p in participants:
                p_url = p.get('profileUrl') or p.get('profile_url')
                if p_url and TARGET_URL.lower().rstrip('/') in p_url.lower().rstrip('/'):
                    target_conv = conv
                    break
            if target_conv: break
            
        if not target_conv:
            print("❌ No matching conversation found for the lead's profile URL.")
            # Let's print the URLs of the first few participants to see what we're getting
            for i, conv in enumerate(items[:5]):
                ps = [p.get('profileUrl') for p in conv.get('participants', [])]
                print(f"[{i}] Participants: {ps}")
            return
            
        conversation_id = target_conv.get('id')
        account_id = target_conv.get('linkedInAccountId')
        print(f"✅ Found Match! ID: {conversation_id}, Account: {account_id}")
        
        # Step 2: Get Chatroom Messages
        chat_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{account_id}/{conversation_id}"
        print(f"Step 2: Fetching messages from {chat_url}...")
        
        chat_response = requests.get(chat_url, headers=headers, timeout=15)
        chat_response.raise_for_status()
        chat_data = chat_response.json()
        
        messages = chat_data.get('messages', [])
        print(f"✅ Retrieved {len(messages)} messages.")
        
        for msg in messages:
            sender = msg.get('sender')
            body = msg.get('body')
            print(f"[{msg.get('createdAt')}] {sender}: {body}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_fetch_messages_v2()

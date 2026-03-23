
import os
import json
from backend.integrations.heyreach import HeyReachBot
from dotenv import load_dotenv

load_dotenv()

def debug_li_messages(profile_url):
    bot = HeyReachBot()
    print(f"Fetching chat history for: {profile_url}")
    
    # 1. Get raw conversations
    headers = {
        "X-API-KEY": bot.api_key,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    import requests
    conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    conv_payload = {
        "filters": {"leadProfileUrl": profile_url},
        "limit": 5
    }
    
    res = requests.post(conv_url, headers=headers, json=conv_payload)
    if res.status_code != 200:
        print(f"Error fetching conversations: {res.text}")
        return
        
    convs = res.json().get('items', [])
    print(f"Found {len(convs)} conversations.")
    
    for i, conv in enumerate(convs):
        print(f"\n--- Conversation {i+1} ---")
        print(f"ID: {conv.get('id')}")
        print(f"Last Message: {conv.get('lastMessageText')}")
        print(f"Updated At: {conv.get('updatedAt')}")
        
        account_id = conv.get('linkedInAccountId') or conv.get('linkedInAccount', {}).get('id')
        conv_id = conv.get('id')
        
        if account_id and conv_id:
            msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{account_id}/{conv_id}"
            msg_res = requests.get(msg_url, headers=headers)
            if msg_res.status_code == 200:
                msg_data = msg_res.json()
                messages = msg_data.get('messages', []) or msg_data.get('items', [])
                print(f"Found {len(messages)} messages.")
                for j, msg in enumerate(messages):
                    sender = msg.get('sender')
                    body = msg.get('body') or msg.get('text')
                    print(f"  [{j}] {msg.get('createdAt')} | Caller: {sender} | Slice: {str(body)[:50]}")
            else:
                print(f"Error fetching chatroom: {msg_res.text}")

if __name__ == "__main__":
    profile_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"
    debug_li_messages(profile_url)

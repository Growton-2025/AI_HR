
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HEYREACH_API_KEY")
TARGET_URL = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

headers = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json",
    "accept": "application/json"
}

def debug_messages():
    # 1. Get conversation
    conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
    payload = {"filters": {"leadProfileUrl": TARGET_URL}, "limit": 1}
    
    res = requests.post(conv_url, headers=headers, json=payload)
    items = res.json().get('items', [])
    if not items:
        print("No conversation found.")
        return
        
    conv = items[0]
    conv_id = conv.get('id')
    acc_id = conv.get('linkedInAccountId') or conv.get('linkedInAccount', {}).get('id')
    
    print(f"Conversation ID: {conv_id}")
    print(f"Account ID: {acc_id}")
    
    # 2. Get Chatroom
    msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{acc_id}/{conv_id}"
    msg_res = requests.get(msg_url, headers=headers)
    chat_data = msg_res.json()
    
    messages = chat_data.get('messages', []) or chat_data.get('items', [])
    print(f"Found {len(messages)} messages.")
    
    for i, msg in enumerate(messages):
        print(f"\n--- Message {i} ---")
        print(json.dumps(msg, indent=2))
        sender = msg.get('sender')
        sender_type = None
        if isinstance(sender, dict):
            sender_type = sender.get('senderType')
        elif isinstance(sender, str):
            sender_type = "ME" if sender.upper() == "ME" else sender
            
        final_sender_type = sender_type or msg.get('senderType')
        print(f"Detected Sender Type: {final_sender_type}")

if __name__ == "__main__":
    debug_messages()

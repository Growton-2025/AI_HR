import requests
import os
from typing import List, Dict, Optional

class HeyReachBot:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HEYREACH_API_KEY")
        self.push_url = "https://api.heyreach.io/api/public/campaign/AddLeadsToCampaign"
    def push_lead(self, campaign_id: int, account_id: int, first_name: str, last_name: str, profile_url: str):
        """
        Push a single lead to a HeyReach campaign.
        """
        headers = {
            "X-API-KEY": self.api_key, 
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        payload = {
            "campaignId": campaign_id,
            "accountLeadPairs": [{
                "accountId": account_id,
                "lead": {
                    "firstName": first_name,
                    "lastName": last_name,
                    "profileUrl": profile_url
                }
            }],
            "resumePausedCampaign": True
        }
        
        try:
            response = requests.post(self.push_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ HeyReach push failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def find_campaign_by_name(self, name: str) -> Optional[int]:
        """
        Find a campaign ID by its name using the GetAll endpoint with a keyword filter.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        url = "https://api.heyreach.io/api/public/campaign/GetAll"
        payload = {
            "offset": 0,
            "limit": 50,
            "keyword": name
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])
            
            for campaign in items:
                # Prioritize exact matches
                if campaign.get('name') == name:
                    return campaign.get('id')
            
            # If no exact match, return the first partial match if any
            if items:
                return items[0].get('id')
                
            return None
        except Exception as e:
            print(f"❌ Error finding HeyReach campaign by name: {e}")
            return None

    def get_campaign_leads(self, campaign_id: int):

        """
        Fetch all leads in a campaign.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
        payload = {
            "campaignId": campaign_id,
            "limit": 100
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('items') if isinstance(data, dict) else data
        except Exception as e:
            print(f"❌ Error fetching campaign leads: {e}")
            return None


    def get_lead_status(self, campaign_id: int, profile_url: str, leads_list: Optional[List[Dict]] = None):
        """
        Fetch lead status and messages for a profile in a campaign.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        # 1. Get leads in the campaign (unless list is already provided for optimization)
        items = leads_list
        if items is None:
            url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
            payload = {
                "campaignId": campaign_id,
                "limit": 100 # API maximum is 100
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                items = data.get('items') if isinstance(data, dict) else data
            except requests.exceptions.RequestException as e:
                print(f"❌ HeyReach lead status fetch failed: {e}")
                return None
        
        try:
            # 2. Find the lead by profile URL
            if isinstance(items, list):

                target_url = profile_url.lower().rstrip('/')
                
                for entry in items:
                    # New discovered mapping: linkedInUserProfile.profileUrl
                    profile = entry.get('linkedInUserProfile') or {}
                    raw_url = (profile.get('profileUrl') or profile.get('profile_url') or "").lower().rstrip('/')
                    
                    if raw_url == target_url or target_url in raw_url:
                        # Extract status mapping
                        msg_status = entry.get('leadMessageStatus')
                        li_status = "in_campaign"
                        
                        if msg_status == "MessageReply":
                            li_status = "replied"
                        elif msg_status == "MessageSent":
                            li_status = "message_sent"
                        elif entry.get('leadConnectionStatus') == "Connected":
                            li_status = "connection_accepted"
                        
                        # 3. SECOND STEP: Get actual message text from Conversations API
                        # This is necessary because GetLeadsFromCampaign doesn't return text
                        last_reply_text = None
                        try:
                            conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
                            conv_payload = {
                                "filters": {"leadProfileUrl": profile_url},
                                "limit": 1
                            }
                            conv_res = requests.post(conv_url, headers=headers, json=conv_payload, timeout=10)
                            if conv_res.status_code == 200:
                                conv_data = conv_res.json()
                                conv_items = conv_data.get('items', [])
                                if conv_items:
                                    last_reply_text = conv_items[0].get('lastMessageText')
                        except Exception as e:
                            print(f"⚠️ Failed to fetch conversation text: {e}")

                        return {
                            "status": li_status,
                            "response_text": last_reply_text,
                            "last_action_at": entry.get('lastActionTime') or entry.get('updatedAt')
                        }
            return None
        except requests.exceptions.RequestException as e:


            print(f"❌ HeyReach lead status fetch failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None


    def get_lead_activity(self, profile_url: str):
        """
        Fetch detailed activity for sync: Sent messages, Replies, Timestamps, and ConversationID.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        # 1. Get conversation to find IDs and meta
        url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
        payload = {
            "filters": {"leadProfileUrl": profile_url},
            "limit": 1
        }
        
        try:
            print(f"DEBUG: HeyReach GetConversationsV2 for {profile_url}")
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            items = data.get('items', [])
            print(f"DEBUG: HeyReach found {len(items)} conversations")
            
            if not items:
                print(f"DEBUG: No HeyReach conversation found for {profile_url}")
                return None
                
            conv = items[0]
            conv_id = conv.get('id')
            # Extract accountId from conversation object
            account_id = conv.get('linkedInAccountId') or conv.get('linkedInAccount', {}).get('id')
            
            if not account_id:
                print(f"DEBUG: Could not find accountId in conversation {conv_id}")
                return None

            # 2. Get full messages history via GetChatroom
            msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{account_id}/{conv_id}"
            msg_res = requests.get(msg_url, headers=headers, timeout=10)
            msg_res.raise_for_status()
            
            # GetChatroom response usually contains a 'messages' array or similar
            chat_data = msg_res.json()
            messages = chat_data.get('messages', []) or chat_data.get('items', [])
            
            stats = {
                "sent_count": 0,
                "last_sent_at": None,
                "reply_text": None,
                "reply_at": None,
                "is_replied": False,
                "conversation_id": conv_id
            }
            
            if messages:
                import re
                # Messages are usually returned in descending order (newest first)
                for msg in messages:
                    # HeyReach structure has 'sender' which can be dict or 'ME' string
                    sender_val = msg.get('sender')
                    sender_type = None
                    if isinstance(sender_val, dict):
                        sender_type = sender_val.get('senderType')
                    elif isinstance(sender_val, str):
                        if sender_val.upper() == 'ME':
                            sender_type = 'User'
                        else:
                            sender_type = 'Lead'
                    
                    # Fallback to top-level if not found
                    sender_type = sender_type or msg.get('senderType')
                    timestamp = msg.get('createdAt')
                    
                    if sender_type and sender_type.lower() != 'lead': # Sent by us
                        stats["sent_count"] += 1
                        if not stats["last_sent_at"] or timestamp > stats["last_sent_at"]:
                            stats["last_sent_at"] = timestamp
                    else: # Sent by lead
                        # Chatroom uses 'body' for text
                        # HeyReach uses 'body' or 'text'
                        text = msg.get('body') or msg.get('text', '')
                        # Basic cleanup
                        clean_text = re.sub('<[^<]+?>', '', str(text or '')).strip()
                        stats["reply_text"] = clean_text
                        stats["reply_at"] = timestamp
                        stats["is_replied"] = True
                            
            return stats
            
        except Exception as e:
            print(f"❌ HeyReach get_lead_activity failed for {profile_url}: {e}")
            return None


    def get_li_chat_history(self, profile_url: str) -> List[Dict]:
        """
        Fetch conversation history for a LinkedIn profile.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
        payload = {
            "filters": {"leadProfileUrl": profile_url},
            "limit": 1
        }
        
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            items = data.get('items', [])
            
            if not items:
                return []
                
            # Get the conversation ID and account ID
            conversation_id = items[0].get('id')
            account_id = items[0].get('linkedInAccountId') or items[0].get('linkedInAccount', {}).get('id')
            
            if not conversation_id or not account_id:
                return []
                
            # Now fetch messages for this conversation via GetChatroom
            msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{account_id}/{conversation_id}"
            msg_res = requests.get(msg_url, headers=headers, timeout=10)
            msg_res.raise_for_status()
            msg_data = msg_res.json()
            
            raw_messages = msg_data.get('messages', []) or msg_data.get('items', [])
            print(f"DEBUG: HeyReach found {len(raw_messages)} messages in chatroom")
            if raw_messages:
                print(f"DEBUG: First message raw keys: {list(raw_messages[0].keys())}")
                print(f"DEBUG: First message sender object: {raw_messages[0].get('sender')}")
                print(f"DEBUG: First message body snippet: {str(raw_messages[0].get('body'))[:50]}")

            # Map to consistent format
            formatted = []
            for m in raw_messages:
                # 'sender' can be dict (Lead) or 'ME' string (User)
                sender_val = m.get('sender')
                s_type = None
                s_name = None
                
                if isinstance(sender_val, dict):
                    s_type = sender_val.get('senderType')
                    s_name = sender_val.get('name')
                elif isinstance(sender_val, str):
                    if sender_val.upper() == 'ME':
                        s_type = 'USER'
                        s_name = 'Me'
                    else:
                        s_type = 'Lead'
                        s_name = sender_val

                # Fallbacks
                s_type = s_type or m.get('senderType')
                s_name = s_name or m.get('senderName')
                
                # If name is generic, clear it so frontend uses candidate name
                if s_name and s_name.upper() in ['CORRESPONDENT', 'LEAD', 'CONTACT']:
                    s_name = None
                
                # Check for incoming (from Lead)
                is_incoming = s_type in ['Lead', 'LEAD', 'lead', 'inbox', 'INBOX']
                
                # Content in 'body' or 'text'
                text_content = m.get('body') or m.get('text') or m.get('message') or m.get('content') or ''
                
                formatted.append({
                    "type": "REPLY" if is_incoming else "SENT",
                    "time": m.get('createdAt') or m.get('time'),
                    "email_body": text_content, 
                    "message_id": m.get('id'),
                    "sender_name": s_name
                })
            
            # Return chronological (oldest first)
            formatted.sort(key=lambda x: str(x['time'] or ''))
            return formatted
            
        except Exception as e:
            print(f"❌ Error fetching LI chat history: {e}")
            return []

    def send_li_message(self, profile_url: str, message: str, conversation_id: Optional[str] = None, account_id: Optional[int] = None) -> bool:
        """
        Send a LinkedIn message to a profile via the PublicInbox/SendMessage endpoint.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        final_conv_id = conversation_id
        final_acc_id = account_id
        
        # 1. If conversation_id or account_id not provided, find them
        if not final_conv_id or not final_acc_id:
            url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
            payload = {
                "filters": {"leadProfileUrl": profile_url},
                "limit": 1
            }
            
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=10)
                res.raise_for_status()
                items = res.json().get('items', [])
                if not items:
                    print(f"❌ No LinkedIn conversation found for profile: {profile_url}")
                    return False
                
                conv = items[0]
                if not final_conv_id:
                    final_conv_id = conv.get('id')
                if not final_acc_id:
                    # Get accountId from the conversation object
                    final_acc_id = conv.get('linkedInAccountId') or conv.get('linkedInAccount', {}).get('id')
            except Exception as e:
                print(f"❌ Failed to lookup conversation for message: {e}")
                return False
                
        if not final_conv_id or not final_acc_id:
            print(f"❌ Missing identifiers for message: conv_id={final_conv_id}, acc_id={final_acc_id}")
            return False

        # 2. Send the message
        try:
            send_url = "https://api.heyreach.io/api/public/inbox/SendMessage"
            send_payload = {
                "linkedInAccountId": final_acc_id,
                "conversationId": final_conv_id,
                "message": message
            }
            
            print(f"DEBUG: Sending HeyReach message to {final_conv_id} via account {final_acc_id}")
            send_res = requests.post(send_url, headers=headers, json=send_payload, timeout=10)
            
            if send_res.status_code != 200:
                print(f"❌ HeyReach SendMessage failed with status {send_res.status_code}: {send_res.text}")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Failed to send LI message: {e}")
            return False

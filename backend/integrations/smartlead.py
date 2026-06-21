import requests
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

class SmartleadBot:
    def __init__(self, api_key=None, base_url="https://server.smartlead.ai"):
        self.api_key = api_key or os.getenv("SMARTLEAD_API_KEY")
        self.base_url = base_url
        self.campaign_id = None

    def _get_headers(self):
        return {"Content-Type": "application/json"}

    def _handle_response(self, response, context):
        """Helper to handle API responses and print status"""
        if response.status_code in [200, 201]:
            print(f"✅ {context} successful!")
            if not response.text or not response.text.strip():
                return {"success": True, "status": "empty_body"}
            try:
                return response.json()
            except Exception:
                return {"success": True, "status": "non_json_body", "raw": response.text}
        else:
            print(f"❌ {context} failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    def create_campaign(self, campaign_name):
        """Step 1: Create the empty campaign container"""
        print(f"\n🏗️ Creating Campaign: '{campaign_name}'...")
        url = f"{self.base_url}/api/v1/campaigns/create?api_key={self.api_key}"
        payload = {"name": campaign_name}
        
        data = self._handle_response(requests.post(url, json=payload), "Campaign Creation")
        if data:
            self.campaign_id = data.get('id')
            print(f"   Campaign ID: {self.campaign_id}")
            return self.campaign_id
        return None

    def get_mailbox_id_by_email(self, email_address):
        """Helper to find mailbox ID from email address"""
        print(f"\n🔍 Looking up ID for email: {email_address}...")
        url = f"{self.base_url}/api/v1/email-accounts?api_key={self.api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            accounts = response.json()
            for acc in accounts:
                if acc.get('from_email') == email_address:
                    print(f"   Found ID: {acc['id']}")
                    return acc['id']
            print(f"❌ Email {email_address} not found in your Smartlead account!")
        else:
            print(f"❌ Failed to fetch email accounts: {response.text}")
        return None

    def list_email_accounts(self):
        """Return the Smartlead workspace's connected sender accounts."""
        url = f"{self.base_url}/api/v1/email-accounts?api_key={self.api_key}&limit=100"
        response = requests.get(url, timeout=15)
        data = self._handle_response(response, "Email Account Fetch")
        return data if isinstance(data, list) else []

    def add_email_account(self, mailbox_id_or_email):
        """Step 2: Attach the sender email (Mailbox) - Accepts ID or Email Address"""
        if not self.campaign_id: return
        
        mailbox_id = mailbox_id_or_email
        if isinstance(mailbox_id_or_email, str) and "@" in mailbox_id_or_email:
             mailbox_id = self.get_mailbox_id_by_email(mailbox_id_or_email)
             if not mailbox_id: return

        print(f"\n📧 Attaching Mailbox ID {mailbox_id}...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/email-accounts?api_key={self.api_key}"
        payload = {"email_account_ids": [mailbox_id]}
        
        return self._handle_response(requests.post(url, json=payload, timeout=15), "Mailbox Attachment")

    def remove_email_account(self, mailbox_id):
        """Remove one sender from this campaign's rotation pool."""
        if not self.campaign_id:
            return None
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/email-accounts?api_key={self.api_key}"
        payload = {"email_account_ids": [int(mailbox_id)]}
        return self._handle_response(
            requests.delete(url, json=payload, timeout=15),
            "Mailbox Removal",
        )

    def set_email_sequence(self, subject, body):
        """Step 3: Set the email content"""
        if not self.campaign_id: return
        print(f"\n✍️ Setting Email Content...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/sequences?api_key={self.api_key}"
        payload = {
            "sequences": [{
                "subject": subject,
                "email_body": body,
                "seq_number": 1,
                "seq_delay_details": {"delay_in_days": 0}
            }]
        }
        
        return self._handle_response(requests.post(url, json=payload, timeout=15), "Email Sequence")

    def set_schedule(self, tz="Asia/Kolkata", start_hour="10:00", end_hour="18:00", start_time=None, days_of_the_week=[1, 2, 3, 4, 5]):
        """Step 4: Configure the sending schedule"""
        if not self.campaign_id: return
        print(f"\n⏰ Setting Schedule ({start_hour}-{end_hour} {tz})...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/schedule?api_key={self.api_key}"
        
        schedule_start_time = None
        if start_time:
             schedule_start_time = start_time.isoformat()
        else:
            # Force start time to TODAY at the specified hour
            from datetime import datetime, timezone as tz_module
            import pytz
            
            # Get current IST time
            ist_tz = pytz.timezone(tz)
            now_ist = datetime.now(ist_tz)
            
            # Parse start hour (e.g., "23:25" -> hour=23, minute=25)
            hour, minute = map(int, start_hour.split(':'))
            
            # Create start time for TODAY at the specified hour
            start_datetime = now_ist.replace(hour=hour, minute=minute, second=0, microsecond=0)
            schedule_start_time = start_datetime.astimezone(tz_module.utc).isoformat()
        
        payload = {
            "timezone": tz,
            "days_of_the_week": days_of_the_week,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "min_time_btw_emails": 3,
            "max_new_leads_per_day": 100,
            "schedule_start_time": schedule_start_time
        }
        
        return self._handle_response(requests.post(url, json=payload, timeout=15), "Schedule Configuration")

    def update_campaign_settings(self, follow_up_percentage=50):
        """Step 4.5: Update campaign settings (Critical for New Leads)"""
        if not self.campaign_id: return
        print(f"\n⚙️ Updating Settings (Follow-up %: {follow_up_percentage})...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/settings?api_key={self.api_key}"
        payload = {"follow_up_percentage": follow_up_percentage}
        
        return self._handle_response(requests.post(url, json=payload, timeout=15), "Settings Update")

    def add_leads(self, leads_list):
        """Step 5: Add leads to the campaign"""
        if not self.campaign_id: return
        print(f"\n🚀 Adding {len(leads_list)} Leads...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/leads?api_key={self.api_key}"
        payload = {"lead_list": leads_list}
        
        return self._handle_response(requests.post(url, json=payload, timeout=30), "Lead Addition")

    def get_campaign_analytics(self):
        """Get campaign stats (Sent, Replied, etc)"""
        if not self.campaign_id: return None
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/analytics?api_key={self.api_key}"
        return self._handle_response(requests.get(url), "Analytics Fetch")

        return stats

    def get_chat_history(self, lead_email, campaign_id=None):
        """Fetch raw message history for a lead in a campaign."""
        active_campaign_id = campaign_id or self.campaign_id
        if not active_campaign_id:
            return None

        # 1. Find Lead ID
        url = f"{self.base_url}/api/v1/leads/?api_key={self.api_key}&email={lead_email}"
        data = self._handle_response(requests.get(url), "Lead Lookup")
        
        lead_id = None
        if data:
            if isinstance(data, list) and len(data) > 0:
                lead_id = data[0].get('id')
            elif isinstance(data, dict):
                lead_id = data.get('id')
        
        if not lead_id:
            return None

        # 2. Use ONLY the provided campaign_id for a "clean" history
        # (Ignore lead_campaign_data to avoid pulling old "trash" messages)
        potential_campaigns = [active_campaign_id] if active_campaign_id else []
        
        all_messages = []
        seen_message_ids = set()

        for cid in potential_campaigns:
            if not cid: continue
            url = f"{self.base_url}/api/v1/campaigns/{cid}/leads/{lead_id}/message-history?api_key={self.api_key}"
            res = requests.get(url)
            if res.status_code == 200:
                raw = res.json()
                new_msgs = []
                if raw and isinstance(raw, dict):
                    new_msgs = raw.get('data') or raw.get('history') or raw.get('messages') or []
                elif raw and isinstance(raw, list):
                    new_msgs = raw
                
                if new_msgs:
                    print(f"✅ Found {len(new_msgs)} messages in campaign {cid}")
                    for msg in new_msgs:
                        mid = msg.get('message_id') or msg.get('stats_id')
                        if mid not in seen_message_ids:
                            all_messages.append(msg)
                            seen_message_ids.add(mid)
        
        # Sort messages by time (ascending)
        if all_messages:
            all_messages.sort(key=lambda x: x.get('time') or x.get('created_at') or x.get('timestamp') or '', reverse=False)

        # Ensure each message has an 'email_body' field for frontend consistency
        if all_messages:
            for msg in all_messages:
                if 'email_body' not in msg:
                    msg['email_body'] = msg.get('html_body') or msg.get('body') or msg.get('message') or ""
        
        return all_messages

    def get_lead_activity(self, lead_email):
        """Fetch detailed activity for sync: Sent messages, Replies, Timestamps"""
        messages = self.get_chat_history(lead_email)
        
        stats = {
            "sent_count": 0,
            "last_sent_at": None,
            "reply_text": None,
            "reply_at": None,
            "is_replied": False
        }
        
        if messages and isinstance(messages, list):
            import re
            print(f"DEBUG: Processing {len(messages)} messages for activity...")
            # Messages are usually descending
            for msg in messages:
                msg_type = str(msg.get('type', '')).upper()
                timestamp = msg.get('time') or msg.get('created_at')
                print(f"DEBUG: Found message type: {msg_type}")
                
                if msg_type == 'SENT':
                    stats["sent_count"] += 1
                    if timestamp:
                        if not stats["last_sent_at"] or timestamp > stats["last_sent_at"]:
                            stats["last_sent_at"] = timestamp
                
                # Broaden reply detection: anything not SENT/OUTBOX/DRAFT/SEQUENCE
                elif msg_type not in ['SENT', 'OUTBOX', 'DRAFT', 'SEQUENCE', 'INITIAL']:
                    print(f"DEBUG: Potential reply found! Type: {msg_type}")
                    if not stats["is_replied"]: # Keep latest reply
                        body = msg.get('email_body', '') or msg.get('html_body') or msg.get('body') or ""
                        # Simple cleanup
                        clean_body = re.sub('<[^<]+?>', '', body)
                        clean_body = re.split(r'On\s+.*(?:wrote|sent):', clean_body, flags=re.IGNORECASE | re.DOTALL)[0]
                        
                        stats["reply_text"] = clean_body.strip()
                        stats["reply_at"] = timestamp
                        stats["is_replied"] = True
                        print(f"DEBUG: Reply extracted: {stats['reply_text'][:50]}...")
        else:
            print(f"DEBUG: No messages list found for {lead_email}")
            
        return stats

    def reply_to_email_thread(self, campaign_id, email_stats_id, message, reply_message_id, reply_email_time, reply_email_body):
        """Reply to a specific email thread."""
        url = f"{self.base_url}/api/v1/campaigns/{campaign_id}/reply-email-thread?api_key={self.api_key}"
        payload = {
            "email_stats_id": email_stats_id,
            "email_body": message,
            "reply_message_id": reply_message_id,
            "reply_email_time": reply_email_time,
            "reply_email_body": reply_email_body
        }
        return self._handle_response(requests.post(url, json=payload), "Reply to Thread")

    def get_lead_reply(self, lead_email):
        """Legacy wrapper for get_lead_activity"""
        activity = self.get_lead_activity(lead_email)
        if activity and activity["is_replied"]:
            return [activity["reply_text"]]
        return []

    def start_campaign(self):
        """Step 6: Activate the campaign"""
        if not self.campaign_id: return
        print(f"\n🔥 Starting Campaign...")
        
        url = f"{self.base_url}/api/v1/campaigns/{self.campaign_id}/status?api_key={self.api_key}"
        payload = {"status": "START"}
        
        return self._handle_response(requests.post(url, json=payload, timeout=15), "Campaign Start")

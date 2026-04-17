import requests
import os
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Connection pooling for better performance
_session = None


def _get_session():
    """Get or create a requests session with connection pooling and retries"""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure retries: 3 retries on connection errors
        retry_strategy = Retry(
            total=4,  # Increased retries for rate limiting
            backoff_factor=1.0,  # Increased backoff factor
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=20, pool_maxsize=20
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


class HeyReachBot:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HEYREACH_API_KEY")
        self.push_url = "https://api.heyreach.io/api/public/campaign/AddLeadsToCampaign"
        self._session = _get_session()

    @staticmethod
    def _normalize_linkedin_url(profile_url: Optional[str]) -> str:
        if not profile_url:
            return ""

        raw = str(profile_url).strip()
        if not raw:
            return ""

        if not raw.startswith("http"):
            raw = f"https://{raw.lstrip('/')}"

        parsed = urlparse(raw)
        path = re.sub(r"/+$", "", (parsed.path or "").strip().lower())
        return path

    def _conversation_matches_profile(
        self, conversation: Dict, profile_url: str
    ) -> bool:
        target = self._normalize_linkedin_url(profile_url)
        if not target:
            return False

        candidate_urls = []
        linked_profile = conversation.get("linkedInUserProfile") or {}
        corr_profile = conversation.get("correspondentProfile") or {}
        participants = conversation.get("participants") or []

        candidate_urls.extend(
            [
                linked_profile.get("profileUrl"),
                linked_profile.get("profile_url"),
                corr_profile.get("profileUrl"),
                corr_profile.get("profile_url"),
                conversation.get("leadProfileUrl"),
                conversation.get("profileUrl"),
            ]
        )

        for participant in participants:
            candidate_urls.extend(
                [
                    participant.get("profileUrl"),
                    participant.get("profile_url"),
                    participant.get("linkedinProfileUrl"),
                    participant.get("linkedInProfileUrl"),
                ]
            )

        normalized_candidates = [
            self._normalize_linkedin_url(candidate_url)
            for candidate_url in candidate_urls
            if candidate_url
        ]

        return any(
            candidate == target or target in candidate or candidate in target
            for candidate in normalized_candidates
            if candidate
        )

    def _find_conversation(
        self,
        profile_url: str,
        campaign_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Find a LinkedIn conversation by profile URL, campaign ID, or conversation ID.
        Uses HeyReach API filters for efficiency.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"

        # 1. If we have a conversation_id, try fetching it directly first (though GetConversations doesn't support specific ID filter easily,
        # we can still look it up in the filtered result).

        # 2. Prefer using the official leadProfileUrl filter
        payload = {"filters": {"leadProfileUrl": profile_url}, "offset": 0, "limit": 50}

        # Add campaign filter if provided
        if campaign_id:
            payload["filters"]["campaignId"] = campaign_id

        try:
            print(f"DEBUG: HeyReach lookup for {profile_url} (campaign: {campaign_id})")
            res = self._session.post(url, headers=headers, json=payload, timeout=8)

            items = []
            if res.status_code == 200:
                items = res.json().get("items", [])
            else:
                print(f"DEBUG API ERROR ({res.status_code}): {res.text}")
                # Don't raise_for_status yet, let it try the fallback or return empty

            if campaign_id and items:
                has_match = any(
                    self._conversation_matches_profile(c, profile_url) for c in items
                )
                if not has_match:
                    # Fallback: Maybe it's in a different campaign? Try without campaign filter.
                    print(
                        f"DEBUG: No matching conversation in campaign {campaign_id}, trying global lookup."
                    )
                    payload["filters"].pop("campaignId")
                    res = self._session.post(
                        url, headers=headers, json=payload, timeout=8
                    )
                    if res.status_code == 200:
                        items = res.json().get("items", [])
                    else:
                        print(
                            f"DEBUG API ERROR ({res.status_code}) on fallback: {res.text}"
                        )

            for conversation in items:
                conv_id = conversation.get("id")
                # If we have a specific ID target, check it
                if conversation_id and str(conv_id) == str(conversation_id):
                    return conversation

                # Otherwise, verify the profile URL matches
                if self._conversation_matches_profile(conversation, profile_url):
                    return conversation

            # 3. Final fallback: If filter failed (API quirk), try a small manual search of recent items
            if not items:
                print(
                    f"DEBUG: Filtered lookup returned nothing for {profile_url}, trying 20 most recent."
                )
                fallback_payload = {"offset": 0, "limit": 20}
                res = self._session.post(
                    url, headers=headers, json=fallback_payload, timeout=8
                )
                if res.status_code == 200:
                    for conversation in res.json().get("items", []):
                        if self._conversation_matches_profile(
                            conversation, profile_url
                        ):
                            return conversation

        except Exception as e:
            print(f"⚠️ HeyReach conversation lookup failed: {e}")

        return None

    def push_lead(
        self,
        campaign_id: int,
        account_id: int,
        first_name: str,
        last_name: str,
        profile_url: str,
    ):
        """
        Push a single lead to a HeyReach campaign.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        payload = {
            "campaignId": campaign_id,
            "accountLeadPairs": [
                {
                    "accountId": account_id,
                    "lead": {
                        "firstName": first_name,
                        "lastName": last_name,
                        "profileUrl": profile_url,
                    },
                }
            ],
            "resumePausedCampaign": True,
        }

        try:
            response = self._session.post(self.push_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ HeyReach push failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def find_campaign_by_name(self, name: str) -> Optional[int]:
        """
        Find a campaign ID by its name using the GetAll endpoint with a keyword filter.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        url = "https://api.heyreach.io/api/public/campaign/GetAll"
        payload = {"offset": 0, "limit": 50, "keyword": name}
        try:
            response = self._session.post(url, headers=headers, json=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            for campaign in items:
                # Prioritize exact matches
                if campaign.get("name") == name:
                    return campaign.get("id")

            # If no exact match, return the first partial match if any
            if items:
                return items[0].get("id")

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
            "accept": "application/json",
        }
        url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
        payload = {"campaignId": campaign_id, "limit": 100}
        try:
            response = self._session.post(url, headers=headers, json=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
            return data.get("items") if isinstance(data, dict) else data
        except Exception as e:
            print(f"❌ Error fetching campaign leads: {e}")
            return None

    def get_lead_status(
        self,
        campaign_id: int,
        profile_url: str,
        leads_list: Optional[List[Dict]] = None,
    ):
        """
        Fetch lead status and messages for a profile in a campaign.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        # 1. Get leads in the campaign (unless list is already provided for optimization)
        items = leads_list
        if items is None:
            url = "https://api.heyreach.io/api/public/campaign/GetLeadsFromCampaign"
            payload = {
                "campaignId": campaign_id,
                "limit": 100,  # API maximum is 100
            }

            try:
                response = self._session.post(
                    url, headers=headers, json=payload, timeout=8
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("items") if isinstance(data, dict) else data
            except requests.exceptions.RequestException as e:
                print(f"❌ HeyReach lead status fetch failed: {e}")
                return None

        try:
            # 2. Find the lead by profile URL
            if isinstance(items, list):
                target_url = profile_url.lower().rstrip("/")

                for entry in items:
                    # New discovered mapping: linkedInUserProfile.profileUrl
                    profile = entry.get("linkedInUserProfile") or {}
                    raw_url = (
                        (profile.get("profileUrl") or profile.get("profile_url") or "")
                        .lower()
                        .rstrip("/")
                    )

                    if raw_url == target_url or target_url in raw_url:
                        # Extract status mapping
                        msg_status = entry.get("leadMessageStatus")
                        li_status = "in_campaign"

                        if msg_status == "MessageReply":
                            li_status = "replied"
                        elif msg_status == "MessageSent":
                            li_status = "message_sent"
                        elif entry.get("leadConnectionStatus") == "Connected":
                            li_status = "connection_accepted"

                        # 3. SECOND STEP: Get actual message text from Conversations API
                        # This is necessary because GetLeadsFromCampaign doesn't return text
                        last_reply_text = None
                        try:
                            conv_url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"
                            conv_payload = {
                                "filters": {"leadProfileUrl": profile_url},
                                "limit": 1,
                            }
                            conv_res = self._session.post(
                                conv_url, headers=headers, json=conv_payload, timeout=8
                            )
                            if conv_res.status_code == 200:
                                conv_data = conv_res.json()
                                conv_items = conv_data.get("items", [])
                                if conv_items:
                                    last_reply_text = conv_items[0].get(
                                        "lastMessageText"
                                    )
                        except Exception as e:
                            print(f"⚠️ Failed to fetch conversation text: {e}")

                        return {
                            "li_status": li_status,
                            "li_response_text": last_reply_text,
                            "li_last_action_at": entry.get("lastActionTime")
                            or entry.get("updatedAt"),
                        }
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ HeyReach lead status fetch failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def get_lead_activity(
        self,
        profile_url: str,
        campaign_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        account_id: Optional[int] = None,
    ):
        """
        Fetch detailed activity for sync: Sent messages, Replies, Timestamps, and ConversationID.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        try:
            conv_id = conversation_id
            final_acc_id = account_id

            if not (conv_id and final_acc_id):
                conv = self._find_conversation(profile_url, campaign_id=campaign_id)
                if not conv:
                    print(f"DEBUG: No HeyReach conversation found for {profile_url}")
                    return None

                conv_id = conv.get("id")
                final_acc_id = conv.get("linkedInAccountId") or conv.get(
                    "linkedInAccount", {}
                ).get("id")

            if not final_acc_id:
                print(f"DEBUG: Could not find accountId for conversation {conv_id}")
                return None

            # 2. Try fetching messages via GetChatroom
            msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{final_acc_id}/{conv_id}"
            msg_res = self._session.get(msg_url, headers=headers, timeout=8)
            
            # 3. Handle specific errors with Auto-Recovery
            if msg_res.status_code in [400, 401, 404]:
                print(f"DEBUG: HeyReach get_lead_activity direct fetch failed ({msg_res.status_code}). Attempting recovery for {profile_url}...")
                conversation = self._find_conversation(profile_url, campaign_id=campaign_id)
                if conversation:
                    conv_id = conversation.get("id")
                    final_acc_id = conversation.get("linkedInAccountId") or conversation.get(
                        "linkedInAccount", {}
                    ).get("id")
                    print(f"DEBUG: Recovered new IDs for activity: conv={conv_id}, acc={final_acc_id}")
                    # Final attempt with new IDs
                    msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{final_acc_id}/{conv_id}"
                    msg_res = self._session.get(msg_url, headers=headers, timeout=8)

            msg_res.raise_for_status()

            # GetChatroom response usually contains a 'messages' array or similar
            chat_data = msg_res.json()
            messages = chat_data.get("messages", []) or chat_data.get("items", [])

            stats = {
                "sent_count": 0,
                "last_sent_at": None,
                "reply_text": None,
                "reply_at": None,
                "is_replied": False,
                "conversation_id": conv_id,
                "account_id": final_acc_id,
            }

            if messages:
                import re

                # Messages are usually returned in descending order (newest first)
                for msg in messages:
                    # HeyReach structure has 'sender' which can be dict or 'ME' string
                    sender_val = msg.get("sender")
                    sender_type = None
                    if isinstance(sender_val, dict):
                        sender_type = sender_val.get("senderType")
                    elif isinstance(sender_val, str):
                        if sender_val.upper() == "ME":
                            sender_type = "User"
                        else:
                            sender_type = "Lead"

                    # Fallback to top-level if not found
                    sender_type = sender_type or msg.get("senderType")
                    timestamp = msg.get("createdAt")

                    # Determine "incoming" (from Candidate)
                    is_incoming = any(
                        x in (sender_type or "").lower()
                        for x in ["lead", "contact", "correspondent", "inbox"]
                    )

                    if not is_incoming:  # Sent by us
                        stats["sent_count"] += 1
                        if not stats["last_sent_at"] or timestamp > (
                            stats["last_sent_at"] or ""
                        ):
                            stats["last_sent_at"] = timestamp
                    else:  # Sent by lead
                        # Chatroom/Conversations API can use different fields
                        text = (
                            msg.get("body")
                            or msg.get("text")
                            or msg.get("messageText")
                            or ""
                        )
                        # Basic cleanup
                        clean_text = re.sub("<[^<]+?>", "", str(text or "")).strip()

                        # Only update if this is the newest reply we've seen
                        # (since messages are usually descending, but we want to be robust)
                        if not stats["reply_at"] or timestamp > stats["reply_at"]:
                            stats["reply_text"] = clean_text
                            stats["reply_at"] = timestamp
                            stats["is_replied"] = True

            return stats

        except Exception as e:
            print(f"❌ HeyReach get_lead_activity failed for {profile_url}: {e}")
            return None

    def get_campaign_activities(self, campaign_id: int) -> Dict[str, Dict]:
        """
        Fetch all conversation activities for a specific campaign in batch.
        Returns a mapping of normalized linkedin profile URLs to activity stats.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        url = "https://api.heyreach.io/api/public/inbox/GetConversationsV2"

        offset = 0
        limit = 100
        activities = {}

        try:
            while True:
                payload = {
                    "filters": {"campaignId": campaign_id},
                    "offset": offset,
                    "limit": limit,
                }
                res = self._session.post(url, headers=headers, json=payload, timeout=12)
                res.raise_for_status()
                data = res.json()

                items = data.get("items", [])
                if not items:
                    break

                for conv in items:
                    corr_profile = conv.get("correspondentProfile") or {}
                    profile_url = corr_profile.get("profileUrl") or corr_profile.get(
                        "publicProfileUrl"
                    )
                    if not profile_url:
                        continue

                    norm_url = self._normalize_linkedin_url(profile_url)
                    if not norm_url:
                        continue

                    messages = conv.get("messages", [])

                    stats = {
                        "sent_count": 0,
                        "last_sent_at": None,
                        "reply_text": None,
                        "reply_at": None,
                        "is_replied": False,
                        "conversation_id": conv.get("id"),
                    }

                    import re

                    for msg in messages:
                        sender_val = msg.get("sender")
                        sender_type = None
                        if isinstance(sender_val, dict):
                            sender_type = sender_val.get("senderType")
                        elif isinstance(sender_val, str):
                            if sender_val.upper() == "ME":
                                sender_type = "User"
                            else:
                                sender_type = "Lead"

                        sender_type = sender_type or msg.get("senderType")
                        timestamp = msg.get("createdAt")

                        is_incoming = any(
                            x in (sender_type or "").lower()
                            for x in ["lead", "contact", "correspondent", "inbox"]
                        )

                        if not is_incoming:
                            stats["sent_count"] += 1
                            if not stats["last_sent_at"] or timestamp > (
                                stats["last_sent_at"] or ""
                            ):
                                stats["last_sent_at"] = timestamp
                        else:
                            text = (
                                msg.get("body")
                                or msg.get("text")
                                or msg.get("messageText")
                                or ""
                            )
                            clean_text = re.sub("<[^<]+?>", "", str(text or "")).strip()

                            if not stats["reply_at"] or timestamp > stats["reply_at"]:
                                stats["reply_text"] = clean_text
                                stats["reply_at"] = timestamp
                                stats["is_replied"] = True

                    activities[norm_url] = stats

                if len(items) < limit:
                    break
                offset += limit

            return activities
        except Exception as e:
            print(f"❌ Failed to fetch campaign activities for {campaign_id}: {e}")
            return activities

    def get_li_chat_history(
        self,
        profile_url: str,
        campaign_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        account_id: Optional[int] = None,
    ) -> Dict:
        """
        Fetch conversation history for a LinkedIn profile.
        If conversation_id and account_id are provided, it fetches directly from the chatroom (fast).
        Automatically falls back to searching if direct ID lookup fails.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        final_conv_id = conversation_id
        final_acc_id = account_id

        try:
            # 1. Ensure we have IDs (lookup if missing)
            if not (final_conv_id and final_acc_id):
                print(f"DEBUG: Ids missing, searching for conversation for {profile_url}")
                conversation = self._find_conversation(
                    profile_url, campaign_id=campaign_id, conversation_id=final_conv_id
                )
                if not conversation:
                    return {"messages": [], "conversation_id": None, "account_id": None}

                final_conv_id = conversation.get("id")
                final_acc_id = conversation.get("linkedInAccountId") or conversation.get(
                    "linkedInAccount", {}
                ).get("id")

            # 2. Try fetching messages via GetChatroom
            msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{final_acc_id}/{final_conv_id}"
            msg_res = self._session.get(msg_url, headers=headers, timeout=8)
            
            # 3. Handle specific errors with Auto-Recovery
            if msg_res.status_code in [400, 401, 404]:
                print(f"DEBUG: HeyReach direct fetch failed ({msg_res.status_code}). Attempting recovery for {profile_url}...")
                conversation = self._find_conversation(profile_url, campaign_id=campaign_id)
                if conversation:
                    final_conv_id = conversation.get("id")
                    final_acc_id = conversation.get("linkedInAccountId") or conversation.get(
                        "linkedInAccount", {}
                    ).get("id")
                    print(f"DEBUG: Recovered new IDs: conv={final_conv_id}, acc={final_acc_id}")
                    # Final attempt with new IDs
                    msg_url = f"https://api.heyreach.io/api/public/inbox/GetChatroom/{final_acc_id}/{final_conv_id}"
                    msg_res = self._session.get(msg_url, headers=headers, timeout=8)

            msg_res.raise_for_status()
            msg_data = msg_res.json()
            raw_messages = msg_data.get("messages", []) or msg_data.get("items", [])

            print(f"DEBUG: HeyReach found {len(raw_messages)} messages in chatroom")
            if raw_messages:
                print(f"DEBUG: First message raw keys: {list(raw_messages[0].keys())}")
                print(
                    f"DEBUG: First message sender object: {raw_messages[0].get('sender')}"
                )
            # Map to consistent format
            formatted = []
            print(f"DEBUG: Processing {len(raw_messages)} messages for chatroom.")
            for m in raw_messages:
                # Log the raw text of the newest messages to diagnose missing replies
                raw_body = str(
                    m.get("body") or m.get("text") or m.get("messageText") or ""
                )
                if len(raw_messages) > 0 and raw_messages.index(m) < 5:
                    print(
                        f"DEBUG: Msg Sample - Sender: {m.get('sender')} | SenderType: {m.get('senderType')} | Body: {raw_body[:30]}"
                    )
                # 'sender' can be dict (Lead) or 'ME' string (User)
                sender_val = m.get("sender")
                s_type = m.get("senderType") or ""
                s_name = m.get("senderName") or ""

                if isinstance(sender_val, dict):
                    s_type = s_type or sender_val.get("senderType") or ""
                    s_name = s_name or sender_val.get("name") or ""
                elif isinstance(sender_val, str):
                    if sender_val.upper() == "ME":
                        s_type = "USER"
                        s_name = "Me"
                    else:
                        s_name = s_name or sender_val

                # Determine "direction" (inbound = from candidate, outbound = from us)
                # HeyReach uses various labels: 'LEAD', 'CORRESPONDENT', 'PROSPECT' for candidates.
                # It uses 'ME', 'USER', 'ACCOUNT' for us.
                is_incoming = True  # Assume incoming unless proven otherwise

                # The 'sender' field directly is often 'ME' or 'CORRESPONDENT'
                sender_val_lower = (
                    str(sender_val).lower() if isinstance(sender_val, str) else ""
                )

                if sender_val_lower == "me":
                    is_incoming = False
                elif sender_val_lower == "correspondent":
                    is_incoming = True
                else:
                    # Fallback to type/name checking
                    s_type_lower = (s_type or "").lower()
                    s_name_lower = (s_name or "").lower()

                    if any(
                        x in s_type_lower for x in ["user", "account", "self", "me"]
                    ):
                        is_incoming = False
                    elif any(x in s_name_lower for x in ["me", "you", "ashwin"]):
                        is_incoming = False
                    elif not s_type and not s_name:
                        pass

                # Content extraction
                text_content = (
                    m.get("body")
                    or m.get("text")
                    or m.get("message")
                    or m.get("content")
                    or m.get("messageText")
                    or ""
                )

                # Time parsing
                msg_time = m.get("createdAt") or m.get("time") or m.get("updatedAt")

                formatted.append(
                    {
                        "id": str(m.get("id") or hash(text_content)),
                        "type": "REPLY" if is_incoming else "SENT",
                        "time": msg_time,
                        "email_body": text_content,
                        "sender_name": s_name if is_incoming else "You",
                        "direction": "inbound" if is_incoming else "outbound",
                    }
                )

            # Robust chronological sort (oldest first)
            def get_timestamp(msg):
                t = msg.get("time")
                if not t:
                    return 0
                try:
                    from dateutil.parser import parse

                    return parse(t).timestamp()
                except:
                    return 0

            formatted.sort(key=get_timestamp)
            return {
                "messages": formatted,
                "conversation_id": final_conv_id,
                "account_id": final_acc_id,
            }

        except Exception as e:
            print(f"❌ Error fetching LI chat history: {e}")
            return {"messages": [], "conversation_id": None, "account_id": None}

    def send_li_message(
        self,
        profile_url: str,
        message: str,
        conversation_id: Optional[str] = None,
        account_id: Optional[int] = None,
        campaign_id: Optional[int] = None,
    ) -> bool:
        """
        Send a LinkedIn message to a profile via the PublicInbox/SendMessage endpoint.
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        final_conv_id = conversation_id
        final_acc_id = account_id

        # 1. If conversation_id or account_id not provided, find them
        if not final_conv_id or not final_acc_id:
            try:
                conversation = self._find_conversation(
                    profile_url,
                    campaign_id=campaign_id,
                    conversation_id=conversation_id,
                )
                if not conversation:
                    print(
                        f"❌ No LinkedIn conversation found for profile: {profile_url}"
                    )
                    return False

                if not final_conv_id:
                    final_conv_id = conversation.get("id")
                if not final_acc_id:
                    final_acc_id = conversation.get(
                        "linkedInAccountId"
                    ) or conversation.get("linkedInAccount", {}).get("id")
            except Exception as e:
                print(f"❌ Failed to lookup conversation for message: {e}")
                return False

        if not final_conv_id or not final_acc_id:
            print(
                f"❌ Missing identifiers for message: conv_id={final_conv_id}, acc_id={final_acc_id}"
            )
            return False

        # 2. Send the message
        try:
            send_url = "https://api.heyreach.io/api/public/inbox/SendMessage"
            send_payload = {
                "linkedInAccountId": final_acc_id,
                "conversationId": final_conv_id,
                "message": message,
            }

            print(
                f"DEBUG: Sending HeyReach message to {final_conv_id} via account {final_acc_id}"
            )
            send_res = self._session.post(
                send_url, headers=headers, json=send_payload, timeout=8
            )

            if send_res.status_code != 200:
                print(
                    f"❌ HeyReach SendMessage failed with status {send_res.status_code}: {send_res.text}"
                )
                return False

            print(f"✅ HeyReach SendMessage success: {send_res.text}")
            return True
        except Exception as e:
            print(f"❌ Failed to send LI message: {e}")
            return False

import requests
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class FreJunManager:
    def __init__(self):
        self.base_url = "https://api.frejun.com/api/v2"
        self.client_id = os.getenv("FREJUN_CLIENT_ID")
        self.client_secret = os.getenv("FREJUN_CLIENT_SECRET")
        self.user_email = os.getenv("FREJUN_USER_EMAIL")
        self.virtual_number = os.getenv("FREJUN_VIRTUAL_NUMBER")
        self._token = None

    def get_access_token(self) -> Optional[str]:
        """
        Retrieves a fresh access token using the Client Credentials.
        NOTE: FreJun documentation typically favors API Key for server-side.
        If Client Credentials flow is used, it follows standard OAuth2.
        """
        if self._token:
            return self._token
            
        # If the user provided an API Key instead of an OAuth Secret, 
        # we might just return that. 
        # But here we handle OAuth token exchange if needed.
        # For now, we'll try to use the secret directly as a Bearer token 
        # if it's an API Key, or perform the exchange.
        
        # Placeholder for OAuth token exchange logic
        # For Many providers, Client ID + Secret are exchanged at /oauth/token
        return self.client_secret # Temporary fallback to using secret as token

    def initiate_call(self, candidate_phone: str, recruiter_email: Optional[str] = None, candidate_name: Optional[str] = None) -> Dict:
        """
        Initiates a 2-legged call via FreJun.
        Rings the recruiter first, then the candidate.
        """
        email = recruiter_email or self.user_email
        virtual_number = self.virtual_number

        # Validation
        if not email:
            return {"success": False, "error": "Recruiter email missing (Set FREJUN_USER_EMAIL in .env)"}
        if not virtual_number:
            return {"success": False, "error": "Virtual number missing (Set FREJUN_VIRTUAL_NUMBER in .env)"}
        if not candidate_phone:
            return {"success": False, "error": "Candidate phone missing"}
        if not self.client_secret:
            return {"success": False, "error": "FreJun API Key/Secret missing (Set FREJUN_CLIENT_SECRET in .env)"}

        # Use the confirmed working v1 endpoint for call initiation
        url = "https://api.frejun.com/api/v1/integrations/create-call/"
        
        # Determine prefix. If it has a dot, it's likely an Access Token (Bearer).
        # Otherwise, we trial Bearer as it's the standard for many new accounts.
        auth_prefix = "Api-Key"
        headers = {
            "Authorization": f"{auth_prefix} {self.client_secret}",
            "Content-Type": "application/json"
        }
        
        # Email as query param is required by FreJun for tracking the initiating user
        params = {"email": email}
        
        payload = {
            "user_email": email,
            "candidate_number": candidate_phone,
            "virtual_number": virtual_number,
            "candidate_name": candidate_name or "Candidate"
        }
        
        try:
            logger.info(f"Initiating FreJun call to {candidate_phone} via {virtual_number}")
            response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
            
            if response.status_code in [200, 201]:
                return {"success": True, "data": response.json()}
            elif response.status_code == 401:
                return {
                    "success": False, 
                    "error": "FreJun authentication failed. The token may be expired or invalid.", 
                    "status_code": 401,
                    "raw_response": response.text
                }
            elif response.status_code == 403:
                return {
                    "success": False, 
                    "error": "FreJun access forbidden. Check if your API key has permissions for this action.", 
                    "status_code": 403,
                    "raw_response": response.text
                }
            else:
                logger.error(f"FreJun call failed: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "FreJun API request timed out"}
        except Exception as e:
            logger.exception("Error calling FreJun API")
            return {"success": False, "error": str(e)}

    def handle_webhook(self, payload: Dict) -> Dict:
        """
        Processes incoming webhook events from FreJun.
        """
        event_type = payload.get("event_type")
        call_id = payload.get("call_id")
        status = payload.get("status")
        
        logger.info(f"Received FreJun webhook: {event_type} for call {call_id} (Status: {status})")
        
        # Logic to update database goes here (handled in the route)
        return {"processed": True}

import logging
import os
import re
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

class FreJunManager:
    def __init__(self):
        self.base_url = "https://api.frejun.com/api/v2"
        self.create_call_url = "https://api.frejun.com/api/v1/integrations/create-call/"
        self.client_id = (os.getenv("FREJUN_CLIENT_ID") or "").strip()
        self.client_secret = (os.getenv("FREJUN_CLIENT_SECRET") or "").strip()
        self.api_key = (os.getenv("FREJUN_API_KEY") or self.client_secret).strip()
        self.user_email = (os.getenv("FREJUN_USER_EMAIL") or "").strip()
        self.virtual_number = (os.getenv("FREJUN_VIRTUAL_NUMBER") or "").strip()
        self._token = None

    @staticmethod
    def _normalize_phone(value: Optional[str]) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""

        if raw.startswith("+"):
            return "+" + re.sub(r"\D", "", raw)

        digits = re.sub(r"\D", "", raw)
        if digits.startswith("00"):
            return "+" + digits[2:]
        return digits

    @staticmethod
    def _parse_response_body(response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {"message": response.text}

        if isinstance(payload, dict):
            return payload
        return {"message": str(payload)}

    @staticmethod
    def _extract_error_message(body: Dict[str, Any], fallback: str) -> str:
        message = body.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        if isinstance(message, dict):
            nested_detail = message.get("detail")
            if isinstance(nested_detail, str) and nested_detail.strip():
                return nested_detail.strip()
            return str(message)

        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, dict):
            return str(detail)
        return fallback

    def _api_key_env_name(self) -> str:
        if (os.getenv("FREJUN_API_KEY") or "").strip():
            return "FREJUN_API_KEY"
        return "FREJUN_CLIENT_SECRET"

    def _api_key_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

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

    def list_virtual_numbers(self, recruiter_email: Optional[str] = None) -> Dict[str, Any]:
        email = recruiter_email or self.user_email

        if not self.api_key:
            return {
                "success": False,
                "error": f"FreJun API key missing (Set {self._api_key_env_name()} in the environment)",
            }

        params = {"email": email} if email else None
        url = f"{self.base_url}/integrations/retrieve-virtual-numbers/"

        try:
            response = requests.get(url, headers=self._api_key_headers(), params=params, timeout=15)
            body = self._parse_response_body(response)

            if response.status_code != 200:
                message = self._extract_error_message(body, "Failed to retrieve FreJun virtual numbers")
                logger.error(
                    "FreJun virtual-number lookup failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
                return {
                    "success": False,
                    "error": f"FreJun virtual-number lookup failed ({response.status_code}): {message}",
                    "status_code": response.status_code,
                    "raw_response": response.text,
                }

            numbers = body.get("data")
            if not isinstance(numbers, list):
                numbers = []
            return {"success": True, "data": numbers}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "FreJun virtual-number lookup timed out"}
        except Exception as e:
            logger.exception("Error retrieving FreJun virtual numbers")
            return {"success": False, "error": str(e)}

    def _resolve_virtual_number(self, recruiter_email: Optional[str] = None) -> Dict[str, Any]:
        email = recruiter_email or self.user_email
        configured_virtual = self._normalize_phone(self.virtual_number)
        lookup = self.list_virtual_numbers(recruiter_email=email)

        if not lookup.get("success"):
            if configured_virtual:
                logger.warning(
                    "Proceeding with configured FreJun virtual number because lookup failed: %s",
                    lookup.get("error"),
                )
                return {
                    "success": True,
                    "virtual_number": configured_virtual,
                    "source": "configured",
                    "warning": lookup.get("error"),
                }
            return lookup

        numbers = lookup.get("data") or []
        normalized_numbers = {}
        for item in numbers:
            normalized = self._normalize_phone(item.get("number"))
            if normalized:
                normalized_numbers[normalized] = item

        if configured_virtual:
            if not normalized_numbers:
                return {
                    "success": False,
                    "error": (
                        f"No virtual numbers are configured in FreJun for {email}. "
                        "Add and assign a virtual number in FreJun before placing calls."
                    ),
                }
            if configured_virtual not in normalized_numbers:
                available = sorted(normalized_numbers.keys())
                available_text = ", ".join(available) if available else "none"
                return {
                    "success": False,
                    "error": (
                        f"Configured FreJun virtual number {self.virtual_number} is not available "
                        f"for {email}. Available FreJun virtual numbers: {available_text}"
                    ),
                }
            selected = normalized_numbers[configured_virtual]
            return {
                "success": True,
                "virtual_number": selected.get("number") or configured_virtual,
                "source": "configured",
            }

        if not numbers:
            return {
                "success": False,
                "error": (
                    f"No virtual numbers are configured in FreJun for {email}. "
                    "Add and assign a virtual number in FreJun before placing calls."
                ),
            }

        selected = next(
            (item for item in numbers if item.get("default_calling_number")),
            numbers[0],
        )
        selected_number = self._normalize_phone(selected.get("number"))
        if not selected_number:
            return {
                "success": False,
                "error": (
                    f"FreJun returned virtual numbers for {email}, but none included a dialable number."
                ),
            }

        return {
            "success": True,
            "virtual_number": selected.get("number") or selected_number,
            "source": "default" if selected.get("default_calling_number") else "first_available",
        }

    def initiate_call(self, candidate_phone: str, recruiter_email: Optional[str] = None, candidate_name: Optional[str] = None) -> Dict:
        """
        Initiates a 2-legged call via FreJun.
        Rings the recruiter first, then the candidate.
        """
        email = recruiter_email or self.user_email
        candidate_phone = self._normalize_phone(candidate_phone)

        # Validation
        if not email:
            return {"success": False, "error": "Recruiter email missing (Set FREJUN_USER_EMAIL in .env)"}
        if not candidate_phone:
            return {"success": False, "error": "Candidate phone missing"}
        if not self.api_key:
            return {
                "success": False,
                "error": f"FreJun API key missing (Set {self._api_key_env_name()} in the environment)",
            }

        virtual_result = self._resolve_virtual_number(recruiter_email=email)
        if not virtual_result.get("success"):
            return {"success": False, "error": virtual_result.get("error")}

        virtual_number = virtual_result.get("virtual_number")
        if not virtual_number:
            return {"success": False, "error": "Unable to determine a FreJun virtual number for this call"}

        # Use the confirmed working v1 endpoint for call initiation
        url = self.create_call_url
        headers = self._api_key_headers()

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
            logger.debug(f"DEBUG: FreJun Params: {params}")
            logger.debug(f"DEBUG: FreJun Payload: {payload}")

            if virtual_result.get("warning"):
                logger.warning("FreJun virtual-number lookup warning: %s", virtual_result.get("warning"))

            response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)

            body = self._parse_response_body(response)
            if response.status_code in [200, 201]:
                return {"success": True, "data": body}
            else:
                message = self._extract_error_message(body, response.text or "FreJun request failed")
                if response.status_code >= 500 and "contact frejun support" in message.lower():
                    message = (
                        f"{message} Verify that the FreJun user has an assigned phone number "
                        "and that the selected virtual number is active for outbound calling."
                    )
                logger.error(f"FreJun call failed with status {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"FreJun Error ({response.status_code}): {message}",
                    "status_code": response.status_code,
                    "raw_response": response.text
                }

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

import base64
import hmac
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from backend.db.connection import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)
load_dotenv()

class FreJunManager:
    def __init__(self):
        self.base_url = "https://api.frejun.com/api/v2"
        self.create_call_url = "https://api.frejun.com/api/v1/integrations/create-call/"
        self.call_to_voip_url = "https://api.frejun.com/api/v1/integrations/call-to-voip/"
        self.calls_url = f"{self.base_url}/integrations/calls/"
        self.webhooks_url = f"{self.base_url}/integrations/webhooks/"
        self.create_webhook_url = f"{self.base_url}/integrations/create-webhook/"
        self.client_id = (os.getenv("FREJUN_CLIENT_ID") or "").strip()
        self.oauth_client_id = (os.getenv("FREJUN_OAUTH_CLIENT_ID") or self.client_id).strip()
        self.client_secret = (os.getenv("FREJUN_CLIENT_SECRET") or "").strip()
        self.api_key = (os.getenv("FREJUN_API_KEY") or self.client_secret).strip()
        self.access_token = (os.getenv("FREJUN_ACCESS_TOKEN") or "").strip()
        self.refresh_token = (os.getenv("FREJUN_REFRESH_TOKEN") or "").strip()
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

    def _bearer_headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _client_credentials_headers(self) -> Dict[str, str]:
        credentials = f"{self.oauth_client_id}:{self.client_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("utf-8")
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_nested_value(body: Dict[str, Any], key: str) -> Any:
        if key in body and body.get(key) not in (None, ""):
            return body.get(key)

        nested = body.get("data")
        if isinstance(nested, dict):
            return nested.get(key)
        return None

    def _load_managed_token(self) -> Optional[Dict[str, Any]]:
        """Load the latest managed OAuth credentials from the database."""
        conn = get_db_connection()
        if not conn:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT access_token, refresh_token, expires_at, frejun_user_email, token_type
                    FROM frejun_oauth_credentials
                    ORDER BY updated_at DESC LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    return {
                        "access_token": row[0],
                        "refresh_token": row[1],
                        "expires_at": row[2],
                        "email": row[3],
                        "token_type": row[4]
                    }
        except Exception as e:
            logger.error(f"Error loading FreJun managed token: {e}")
        finally:
            return_db_connection(conn)
        return None

    def _save_managed_token(self, access_token: str, refresh_token: str, expires_in: int, email: Optional[str] = None):
        """Persist rotated OAuth tokens and expiry to the database."""
        conn = get_db_connection()
        if not conn:
            return
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            with conn.cursor() as cur:
                # We use a single global row for now as per assumptions
                cur.execute("""
                    INSERT INTO frejun_oauth_credentials 
                    (access_token, refresh_token, expires_at, frejun_user_email, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (access_token, refresh_token, expires_at, email))
                conn.commit()
                logger.info(f"FreJun managed tokens persisted/rotated for {email}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving FreJun managed token: {e}")
        finally:
            return_db_connection(conn)

    def get_access_token(self) -> Optional[str]:
        """
        Returns a pre-configured access token when available.

        FreJun's documented token exchange uses an authorization code flow, so we do
        not attempt to mint bearer tokens server-side without an explicit token in the
        environment.
        """
        if self._token:
            return self._token

        if self.access_token:
            self._token = self.access_token
            return self._token

        return None

    def get_voip_access_token(self, recruiter_email: Optional[str] = None) -> Dict[str, Any]:
        email = (recruiter_email or self.user_email or "").strip()

        # 1. Try managed DB store first (The "Forever Online" bridge)
        managed = self._load_managed_token()
        if managed:
            access_token = managed["access_token"]
            refresh_token = managed["refresh_token"]
            expires_at = managed["expires_at"]
            
            # Check if token is still valid (with 5 min safety buffer)
            if expires_at > datetime.utcnow() + timedelta(minutes=5):
                logger.info(f"Using cached FreJun access token from DB (expires at {expires_at})")
                self._token = access_token
                return {
                    "success": True,
                    "access_token": self._token,
                    "expires_in": int((expires_at - datetime.utcnow()).total_seconds()),
                    "agent_email": email or managed["email"],
                    "source": "database_cache",
                }
            
            # Token expired - trigger rotation using stored refresh token
            logger.info("FreJun access token expired in DB. Attempting rotation...")
            rotate_result = self._refresh_oauth_token(refresh_token, email or managed["email"])
            if rotate_result.get("success"):
                return rotate_result
            
            # If refresh fails, we fall back to env or return failure
            logger.warning(f"FreJun background rotation failed: {rotate_result.get('error')}. Checking fallbacks...")

        # 2. Fallback: Check for hardcoded Access Token in .env (Manual bypass)
        if self.access_token:
            self._token = self.access_token
            return {
                "success": True,
                "access_token": self._token,
                "expires_in": None,
                "agent_email": email or self.user_email,
                "source": "configured_access_token",
            }

        # 3. Fallback: Check for Refresh Token in .env (Legacy/Bootstrap)
        if self.refresh_token:
            rotate_result = self._refresh_oauth_token(self.refresh_token, email)
            if rotate_result.get("success"):
                return rotate_result

        return {
            "success": False,
            "error": "FreJun browser VoIP is not connected. Please go to Settings and click 'Connect FreJun VoIP'.",
            "status_code": 401,
        }

    def _refresh_oauth_token(self, refresh_token: str, email: Optional[str]) -> Dict[str, Any]:
        """Internal helper to execute the OAuth refresh grant and persist results."""
        if not self.oauth_client_id or not self.client_secret:
            return {
                "success": False,
                "error": "FreJun OAuth client credentials missing.",
                "status_code": 500,
            }

        url = "https://api.frejun.com/api/v1/oauth/token/"
        credentials = f"{self.oauth_client_id}:{self.client_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("utf-8")
        
        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        form_data = {
            "grant_type": "refresh_token",
            "refresh": refresh_token
        }

        try:
            logger.info(f"Requesting fresh FreJun token pair for {email}...")
            response = requests.post(url, headers=headers, data=form_data, timeout=20)
            body = self._parse_response_body(response)

            if response.status_code in (200, 201):
                new_access = self._extract_nested_value(body, "access_token")
                new_refresh = self._extract_nested_value(body, "refresh_token") or refresh_token
                expires_in = self._extract_nested_value(body, "expires_in") or 21600

                if new_access:
                    # PERSIST THE ROTATION
                    self._save_managed_token(new_access, new_refresh, expires_in, email)
                    self._token = new_access
                    return {
                        "success": True,
                        "access_token": new_access,
                        "expires_in": expires_in,
                        "agent_email": email,
                        "source": "oauth_rotation",
                    }

            error_msg = self._extract_error_message(body, "FreJun refresh failed")
            return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code,
                "raw_response": response.text
            }
        except Exception as e:
            logger.exception("FreJun token refresh exception")
            return {"success": False, "error": str(e), "status_code": 500}

    def get_voip_agent(self, recruiter_email: Optional[str] = None) -> Dict[str, Any]:
        email = (recruiter_email or self.user_email or "").strip()
        if not email:
            return {
                "success": False,
                "error": "Recruiter email missing (Set FREJUN_USER_EMAIL in .env)",
                "status_code": 400,
            }

        token_result = self.get_voip_access_token(recruiter_email=email)
        if not token_result.get("success"):
            return token_result

        # [ZERO-TOUCH] Use pre-configured Agent ID if available to bypass lookup
        agent_id = os.getenv("FREJUN_AGENT_ID", "").strip()
        if agent_id:
            return {
                "success": True,
                "agent_id": agent_id,
                "agent_email": email,
                "token": token_result["access_token"]
            }

        try:
            response = requests.get(
                f"{self.base_url}/integrations/users/",
                headers=self._bearer_headers(token_result["access_token"]),
                params={"email": email},
                timeout=15,
            )
            body = self._parse_response_body(response)
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": self._extract_error_message(body, "Failed to retrieve FreJun VoIP agent"),
                    "status_code": response.status_code,
                    "raw_response": response.text,
                }

            data = body.get("data")
            if isinstance(data, dict):
                users = [data]
            elif isinstance(data, list):
                users = data
            else:
                users = []

            matched_user = None
            for user in users:
                if isinstance(user, dict) and (user.get("email") or "").strip().lower() == email.lower():
                    matched_user = user
                    break
            if matched_user is None and users:
                matched_user = users[0]

            if not isinstance(matched_user, dict):
                return {
                    "success": False,
                    "error": (
                        f"FreJun did not return a browser-calling user for {email}. "
                        "Verify SDK/browser-calling access is enabled for this user."
                    ),
                    "status_code": 424,
                }

            agent_id = matched_user.get("user_id") or matched_user.get("id")
            if not agent_id:
                return {
                    "success": False,
                    "error": (
                        f"FreJun returned user details for {email}, but no agent identifier was present."
                    ),
                    "status_code": 424,
                }

            return {
                "success": True,
                "agent_id": str(agent_id),
                "agent_email": (matched_user.get("email") or email).strip(),
                "access_token": token_result["access_token"],
                "expires_in": token_result.get("expires_in"),
                "source": token_result.get("source"),
                "user": matched_user,
            }
        except requests.exceptions.Timeout:
            return {"success": False, "error": "FreJun VoIP agent lookup timed out", "status_code": 504}
        except Exception as exc:
            logger.exception("Error getting FreJun VoIP agent")
            return {"success": False, "error": str(exc), "status_code": 500}

    def _iter_call_auth_attempts(self):
        seen = set()
        access_token = self.get_access_token()
        if access_token:
            header = tuple(sorted(self._bearer_headers(access_token).items()))
            if header not in seen:
                seen.add(header)
                yield "bearer", dict(header)

        if self.api_key:
            header = tuple(sorted(self._api_key_headers().items()))
            if header not in seen:
                seen.add(header)
                yield "api_key", dict(header)

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
                # Fallback: match by last 10 digits
                configured_digits = re.sub(r"\D", "", configured_virtual)
                matched_key = None
                if len(configured_digits) >= 10:
                    last_ten = configured_digits[-10:]
                    for k in normalized_numbers.keys():
                        k_digits = re.sub(r"\D", "", k)
                        if len(k_digits) >= 10 and k_digits[-10:] == last_ten:
                            matched_key = k
                            break
                            
                if matched_key:
                    selected = normalized_numbers[matched_key]
                    return {
                        "success": True,
                        "virtual_number": selected.get("number") or matched_key,
                        "source": "configured",
                    }
                    
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

    def initiate_call(
        self,
        candidate_phone: str,
        recruiter_email: Optional[str] = None,
        candidate_name: Optional[str] = None,
        candidate_id: Optional[str] = None,
        job_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        dial_mode: str = "voip",
    ) -> Dict:
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
        if len(candidate_phone) == 10 and not candidate_phone.startswith("+"):
            candidate_phone = f"+91{candidate_phone}"
            
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

        # Format virtual number with country code for VoIP endpoint requirements
        formatted_virtual_number = virtual_number
        if len(str(formatted_virtual_number)) == 10 and not str(formatted_virtual_number).startswith("+"):
            formatted_virtual_number = f"+91{formatted_virtual_number}"

        # Browser VoIP initiation (Strictly enforced per user request)
        agent_result = self.get_voip_agent(email)
        if not agent_result.get("success"):
            return {
                "success": False,
                "error": f"FreJun browser VoIP is unavailable: {agent_result.get('error')}",
                "status_code": agent_result.get("status_code", 424),
                "dial_mode": "voip",
            }

        url = self.call_to_voip_url
        params = {}
        payload = {
            "agent_id": agent_result["agent_id"],
            "dstn_number": candidate_phone,
            "virtual_number": formatted_virtual_number,
            "candidate_name": candidate_name or "Candidate",
        }
        if transaction_id:
            payload["transaction_id"] = str(transaction_id)
        
        dial_mode = "voip" # Enforce dial mode for logging below

        try:
            headers = self._bearer_headers(agent_result["token"])
            logger.info(
                "Initiating FreJun %s call to %s via %s",
                dial_mode,
                candidate_phone,
                formatted_virtual_number,
            )
            logger.debug(f"DEBUG: FreJun Params: {params}")
            logger.debug(f"DEBUG: FreJun Payload: {payload}")

            if virtual_result.get("warning"):
                logger.warning("FreJun virtual-number lookup warning: %s", virtual_result.get("warning"))

            response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)

            body = self._parse_response_body(response)
            if response.status_code in [200, 201]:
                response_data = body.get("data") if isinstance(body, dict) else None
                if not isinstance(response_data, dict):
                    response_data = {}
                return {
                    "success": True,
                    "data": body,
                    "call_data": response_data,
                    "dial_mode": dial_mode,
                }
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
                    "raw_response": response.text,
                    "dial_mode": dial_mode,
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "FreJun API request timed out",
                "status_code": 504,
                "dial_mode": dial_mode,
            }
        except Exception as e:
            logger.exception("Error calling FreJun API")
            return {
                "success": False,
                "error": str(e),
                "status_code": 500,
                "dial_mode": dial_mode,
            }

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

    def get_call_logs(
        self,
        recruiter_email: Optional[str] = None,
        call_id: Optional[str] = None,
        event_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        candidate_number: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        email = (recruiter_email or self.user_email or "").strip()
        params: Dict[str, str] = {}
        if email:
            params["email"] = email
            params["recruiter_email"] = email
        if call_id:
            params["call_id"] = str(call_id)
        if event_id:
            params["event_id"] = str(event_id)
        if transaction_id:
            params["transaction_id"] = str(transaction_id)
        if candidate_id:
            params["candidate_id"] = str(candidate_id)
        normalized_candidate_number = self._normalize_phone(candidate_number)
        if normalized_candidate_number:
            params["candidate_number"] = normalized_candidate_number

        attempts = list(self._iter_call_auth_attempts())
        if not attempts:
            return {
                "success": False,
                "error": (
                    "FreJun call-log lookup requires authentication. Configure FREJUN_API_KEY "
                    "or FREJUN_ACCESS_TOKEN."
                ),
            }

        last_error = None
        for auth_mode, headers in attempts:
            try:
                response = requests.get(self.calls_url, headers=headers, params=params, timeout=30)
                body = self._parse_response_body(response)
                if response.status_code == 200:
                    payload = body.get("data") if isinstance(body, dict) else {}
                    if not isinstance(payload, dict):
                        payload = {}
                    results = payload.get("results")
                    if not isinstance(results, list):
                        results = []
                    return {
                        "success": True,
                        "data": payload,
                        "results": results,
                        "auth_mode": auth_mode,
                    }

                last_error = {
                    "status_code": response.status_code,
                    "message": self._extract_error_message(body, response.text or "FreJun request failed"),
                    "auth_mode": auth_mode,
                    "raw_response": response.text,
                }

                if response.status_code not in (401, 403):
                    break
            except requests.exceptions.Timeout:
                last_error = {"message": "FreJun call-log lookup timed out", "auth_mode": auth_mode}
            except Exception as exc:
                logger.exception("Error retrieving FreJun call logs")
                last_error = {"message": str(exc), "auth_mode": auth_mode}

        if last_error is None:
            last_error = {"message": "FreJun call-log lookup failed"}

        status_code = last_error.get("status_code")
        detail = last_error.get("message") or "FreJun call-log lookup failed"
        if last_error.get("auth_mode") == "api_key" and status_code in (401, 403):
            detail = (
                f"{detail}. Configure FREJUN_ACCESS_TOKEN if this FreJun endpoint is restricted "
                "to bearer-token authentication in your account."
            )
        return {
            "success": False,
            "error": detail,
            "status_code": status_code,
            "raw_response": last_error.get("raw_response"),
        }

    def list_webhooks(self) -> Dict[str, Any]:
        if not self.client_id or not self.client_secret:
            return {"success": False, "error": "FreJun client credentials are required to list webhooks"}

        try:
            response = requests.get(self.webhooks_url, headers=self._client_credentials_headers(), timeout=30)
            body = self._parse_response_body(response)
            if response.status_code == 200:
                data = body.get("data")
                if not isinstance(data, list):
                    data = []
                return {"success": True, "data": data}

            return {
                "success": False,
                "error": self._extract_error_message(body, "Failed to list FreJun webhooks"),
                "status_code": response.status_code,
                "raw_response": response.text,
            }
        except Exception as exc:
            logger.exception("Error listing FreJun webhooks")
            return {"success": False, "error": str(exc)}

    def create_webhook(self, event: str, callback_url: str, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not self.client_id or not self.client_secret:
            return {"success": False, "error": "FreJun client credentials are required to create webhooks"}

        payload: Dict[str, Any] = {
            "event": event,
            "callback_url": callback_url,
        }
        if custom_headers:
            payload["custom_headers"] = custom_headers

        try:
            response = requests.post(
                self.create_webhook_url,
                headers=self._client_credentials_headers(),
                json=payload,
                timeout=30,
            )
            body = self._parse_response_body(response)
            if response.status_code in (200, 201):
                return {"success": True, "data": body.get("data") if isinstance(body, dict) else body}

            return {
                "success": False,
                "error": self._extract_error_message(body, "Failed to create FreJun webhook"),
                "status_code": response.status_code,
                "raw_response": response.text,
            }
        except Exception as exc:
            logger.exception("Error creating FreJun webhook")
            return {"success": False, "error": str(exc)}

    def ensure_webhooks(self, callback_url: str, events: list[str]) -> Dict[str, Any]:
        existing = self.list_webhooks()
        if not existing.get("success"):
            return existing

        current_hooks = existing.get("data") or []
        created = []
        skipped = []
        for event in events:
            event_exists = any(
                hook.get("event") == event and hook.get("callback_url") == callback_url
                for hook in current_hooks
            )
            if event_exists:
                skipped.append(event)
                continue

            created_hook = self.create_webhook(event=event, callback_url=callback_url)
            if not created_hook.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to create webhook for {event}: {created_hook.get('error')}",
                    "created": created,
                    "skipped": skipped,
                }
            created.append(event)

        return {"success": True, "created": created, "skipped": skipped, "existing": current_hooks}

    def validate_webhook_signature(
        self,
        *,
        method: str,
        request_uri: str,
        raw_body: bytes,
        signature: Optional[str],
        signature_slim: Optional[str],
        call_id: Optional[str],
    ) -> Dict[str, Any]:
        if not self.client_secret:
            return {"valid": False, "error": "FreJun client secret missing"}

        method = (method or "POST").upper()
        body_text = raw_body.decode("utf-8")
        signature = (signature or "").strip()
        signature_slim = (signature_slim or "").strip()

        # Check for debug bypass
        if os.getenv("SKIP_FREJUN_WEBHOOK_SIGNATURE") == "true":
            return {"valid": True, "mode": "bypass"}

        if signature:
            payload = f"{method}{request_uri}{body_text}".encode("utf-8")
            expected = base64.b64encode(
                hmac.new(self.client_secret.encode("utf-8"), payload, "sha256").digest()
            ).decode("utf-8")
            if hmac.compare_digest(signature, expected):
                return {"valid": True, "mode": "frejun-signature"}

        if signature_slim and call_id:
            payload = f"{method}{request_uri}{call_id}".encode("utf-8")
            expected = base64.b64encode(
                hmac.new(self.client_secret.encode("utf-8"), payload, "sha256").digest()
            ).decode("utf-8")
            if hmac.compare_digest(signature_slim, expected):
                return {"valid": True, "mode": "frejun-signature-slim"}

        return {"valid": False, "error": "Invalid FreJun webhook signature"}

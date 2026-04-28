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
        self.user_url = f"{self.base_url}/integrations/user/"
        # FreJun's Browser VoIP initiation endpoint is still served on v1.
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

    def _resolve_voip_identity(self, recruiter_email: Optional[str] = None) -> tuple[str, list[str]]:
        requested_email = self._normalize_email(recruiter_email)
        configured_email = self._normalize_email(self.user_email)
        canonical_email = configured_email or requested_email

        aliases: list[str] = []
        for candidate in (canonical_email, requested_email):
            if candidate and candidate not in aliases:
                aliases.append(candidate)

        return canonical_email, aliases

    @staticmethod
    def _frejun_settings_url() -> str:
        return "https://product.frejun.com/settings"

    @staticmethod
    def _frejun_virtual_numbers_url() -> str:
        return "https://product.frejun.com/virtual-numbers"

    @staticmethod
    def _frejun_browser_calling_url() -> str:
        return "https://product.frejun.com/billing"

    @staticmethod
    def _normalize_email(value: Optional[str]) -> str:
        return (value or "").strip().lower()

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

    def _build_voip_error(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 424,
        action_label: Optional[str] = None,
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        detail = {
            "code": code,
            "message": message,
            "action_label": action_label,
            "action_url": action_url,
            "metadata": metadata or {},
        }
        return {
            "success": False,
            "error": message,
            "code": code,
            "status_code": status_code,
            "action_label": action_label,
            "action_url": action_url,
            "metadata": metadata or {},
            "detail": detail,
            "raw_response": raw_response,
        }

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
            "Authorization": f"Bearer {encoded}",
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

    def _load_managed_token(
        self,
        email: Optional[str] = None,
        *,
        candidate_emails: Optional[list[str]] = None,
        allow_legacy_unmapped: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load managed OAuth credentials, preferring recruiter-specific rows."""
        conn = get_db_connection()
        if not conn:
            return None
        normalized_email = self._normalize_email(email)
        normalized_candidates = []
        for candidate in candidate_emails or []:
            normalized_candidate = self._normalize_email(candidate)
            if normalized_candidate and normalized_candidate not in normalized_candidates:
                normalized_candidates.append(normalized_candidate)
        if normalized_email and normalized_email not in normalized_candidates:
            normalized_candidates.insert(0, normalized_email)
        try:
            with conn.cursor() as cur:
                row = None
                if normalized_candidates:
                    select_sql = """
                        SELECT access_token, refresh_token, expires_at, frejun_user_email, token_type, agent_id, virtual_number
                        FROM frejun_oauth_credentials
                        WHERE LOWER(COALESCE(frejun_user_email, '')) = %s
                        ORDER BY updated_at DESC LIMIT 1
                    """
                    for candidate in normalized_candidates:
                        cur.execute(
                            select_sql,
                            (candidate,),
                        )
                        row = cur.fetchone()
                        if row:
                            break
                    if not row and allow_legacy_unmapped:
                        # Older OAuth callbacks stored tokens without frejun_user_email.
                        # Prefer that legacy bridge before forcing recruiters to reconnect.
                        cur.execute(
                            """
                            SELECT access_token, refresh_token, expires_at, frejun_user_email, token_type, agent_id, virtual_number
                            FROM frejun_oauth_credentials
                            WHERE COALESCE(NULLIF(TRIM(frejun_user_email), ''), '') = ''
                            ORDER BY updated_at DESC LIMIT 1
                            """
                        )
                        row = cur.fetchone()
                        if row:
                            logger.info(
                                "Using legacy unmapped FreJun OAuth token for %s until it is reconnected.",
                                normalized_email or ",".join(normalized_candidates),
                            )
                else:
                    cur.execute(
                        """
                        SELECT access_token, refresh_token, expires_at, frejun_user_email, token_type, agent_id, virtual_number
                        FROM frejun_oauth_credentials
                        ORDER BY updated_at DESC LIMIT 1
                        """
                    )
                    row = cur.fetchone()
                if row:
                    return {
                        "access_token": row[0],
                        "refresh_token": row[1],
                        "expires_at": row[2],
                        "email": row[3],
                        "token_type": row[4],
                        "agent_id": row[5],
                        "virtual_number": row[6]
                    }
        except Exception as e:
            logger.error(f"Error loading FreJun managed token: {e}")
        finally:
            return_db_connection(conn)
        return None

    def _promote_managed_token(
        self,
        access_token: str,
        refresh_token: Optional[str],
        expires_at: datetime,
        email: Optional[str],
    ) -> None:
        canonical_email = self._normalize_email(email)
        if not canonical_email or not refresh_token:
            return

        remaining_seconds = int((expires_at - datetime.utcnow()).total_seconds())
        if remaining_seconds <= 0:
            return

        self._save_managed_token(access_token, refresh_token, remaining_seconds, canonical_email)

    def _save_managed_token(
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        email: Optional[str] = None,
        agent_id: Optional[str] = None,
        virtual_number: Optional[str] = None,
    ):
        """Persist rotated OAuth tokens and metadata to the database."""
        conn = get_db_connection()
        if not conn:
            return
        normalized_email = self._normalize_email(email)
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            with conn.cursor() as cur:
                if normalized_email:
                    cur.execute(
                        """
                        UPDATE frejun_oauth_credentials
                        SET access_token = %s,
                            refresh_token = %s,
                            expires_at = %s,
                            agent_id = COALESCE(%s, agent_id),
                            virtual_number = COALESCE(%s, virtual_number),
                            token_type = 'Bearer',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = (
                            SELECT id FROM frejun_oauth_credentials
                            WHERE LOWER(COALESCE(frejun_user_email, '')) = %s
                            ORDER BY updated_at DESC
                            LIMIT 1
                        )
                        """,
                        (access_token, refresh_token, expires_at, agent_id, virtual_number, normalized_email),
                    )
                    if cur.rowcount == 0:
                        cur.execute(
                            """
                            INSERT INTO frejun_oauth_credentials
                            (access_token, refresh_token, expires_at, frejun_user_email, agent_id, virtual_number, token_type, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, 'Bearer', CURRENT_TIMESTAMP)
                            """,
                            (access_token, refresh_token, expires_at, normalized_email, agent_id, virtual_number),
                        )
                else:
                    cur.execute("""
                        INSERT INTO frejun_oauth_credentials
                        (access_token, refresh_token, expires_at, frejun_user_email, agent_id, virtual_number, token_type, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, 'Bearer', CURRENT_TIMESTAMP)
                    """, (access_token, refresh_token, expires_at, None, agent_id, virtual_number))
                conn.commit()
                logger.info(f"FreJun managed tokens persisted/rotated for {normalized_email or 'default account'}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving FreJun managed token: {e}")
        finally:
            return_db_connection(conn)

    def _cache_voip_meta(self, email: str, agent_id: str, virtual_number: str):
        """Update only the VOIP metadata (agent_id, number) in the database bridge."""
        conn = get_db_connection()
        if not conn:
            return
        normalized_email = self._normalize_email(email)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE frejun_oauth_credentials
                    SET agent_id = %s,
                        virtual_number = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE LOWER(COALESCE(frejun_user_email, '')) = %s
                    """,
                    (agent_id, virtual_number, normalized_email),
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error caching FreJun VOIP metadata: {e}")
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

    def get_voip_access_token(self, recruiter_email: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        email, email_aliases = self._resolve_voip_identity(recruiter_email)

        # 1. Try managed DB store first (The "Forever Online" bridge)
        managed = self._load_managed_token(email=email, candidate_emails=email_aliases)
        if managed and not force_refresh:
            access_token = managed["access_token"]
            refresh_token = managed["refresh_token"]
            expires_at = managed["expires_at"]
            managed_email = self._normalize_email(managed.get("email"))
            
            # Check if token is still valid (with 5 min safety buffer)
            # PROD COMBO: Use a 60-minute safety buffer.
            # This ensures the browser always gets a "Fresh" token valid for at least an hour,
            # but avoids refreshing for every single call to prevent rate limiting.
            if expires_at > datetime.utcnow() + timedelta(minutes=60):
                logger.info(f"Using cached FreJun access token from DB (expires at {expires_at})")
                self._token = access_token
                if email and managed_email != email:
                    self._promote_managed_token(access_token, refresh_token, expires_at, email)
                return {
                    "success": True,
                    "access_token": self._token,
                    "expires_in": int((expires_at - datetime.utcnow()).total_seconds()),
                    "agent_email": email or managed_email,
                    "agent_id": managed.get("agent_id"),
                    "virtual_number": managed.get("virtual_number"),
                    "source": "database_cache",
                }
            
            # Token expired - trigger rotation using stored refresh token
            logger.info("FreJun access token expired in DB. Attempting rotation...")
            rotate_result = self._refresh_oauth_token(refresh_token, email or managed_email)
            if rotate_result.get("success"):
                # PROD FIX: Preserve cached metadata across rotations to avoid redundant lookups
                rotate_result["agent_id"] = managed.get("agent_id")
                rotate_result["virtual_number"] = managed.get("virtual_number")
                return rotate_result
            
            # If refresh fails, we fall back to env or return failure
            logger.warning(f"FreJun background rotation failed: {rotate_result.get('error')}. Checking fallbacks...")

        # 2. Fallback: refresh from configured refresh token (hands-off rotation path)
        if self.refresh_token:
            rotate_result = self._refresh_oauth_token(self.refresh_token, email)
            if rotate_result.get("success"):
                return rotate_result
            logger.warning(
                "FreJun configured refresh-token rotation failed: %s",
                rotate_result.get("error"),
            )

        # 3. Bootstrap fallback for legacy env-only setups that do not yet have
        # a durable refresh token bridge configured.
        if self.access_token and not self.refresh_token:
            self._token = self.access_token
            return {
                "success": True,
                "access_token": self._token,
                "expires_in": None,
                "agent_email": email or self.user_email,
                "source": "configured_access_token_bootstrap",
            }

        return self._build_voip_error(
            "oauth_not_connected",
            "FreJun browser VoIP is not connected yet. Connect FreJun VoIP once to establish production OAuth access.",
            status_code=401,
            action_label="Connect FreJun VoIP",
            action_url="/api/auth/frejun-login",
            metadata={"agent_email": email},
        )

    def _refresh_oauth_token(self, refresh_token: str, email: Optional[str]) -> Dict[str, Any]:
        """Internal helper to execute the OAuth refresh grant and persist results."""
        if not self.oauth_client_id or not self.client_secret:
            return self._build_voip_error(
                "token_refresh_failed",
                "FreJun OAuth client credentials are missing.",
                status_code=500,
                action_label="Open FreJun Settings",
                action_url=self._frejun_settings_url(),
                metadata={"agent_email": self._normalize_email(email)},
            )

        url = "https://api.frejun.com/api/v2/oauth/token/refresh/"
        credentials = f"{self.oauth_client_id}:{self.client_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("utf-8")
        
        headers = {
            "Authorization": f"Bearer {encoded}",
            "Content-Type": "application/json",
        }
        payload = {"refresh": refresh_token}

        try:
            logger.info(f"Requesting fresh FreJun token pair for {email}...")
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            body = self._parse_response_body(response)

            if response.status_code in (200, 201):
                new_access = (
                    self._extract_nested_value(body, "access")
                    or self._extract_nested_value(body, "access_token")
                )
                new_refresh = (
                    self._extract_nested_value(body, "refresh")
                    or self._extract_nested_value(body, "refresh_token")
                    or refresh_token
                )
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
            return self._build_voip_error(
                "token_refresh_failed",
                error_msg,
                status_code=response.status_code or 502,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
                metadata={"agent_email": self._normalize_email(email)},
                raw_response=response.text,
            )
        except Exception as e:
            logger.exception("FreJun token refresh exception")
            return self._build_voip_error(
                "token_refresh_failed",
                str(e),
                status_code=500,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
                metadata={"agent_email": self._normalize_email(email)},
            )

    def _build_user_metadata(
        self,
        email: str,
        user: Optional[Dict[str, Any]] = None,
        *,
        virtual_number: Optional[str] = None,
        virtual_number_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = user if isinstance(user, dict) else {}
        numbers = payload.get("virtual_numbers")
        if not isinstance(numbers, list):
            numbers = []
        return {
            "agent_email": (payload.get("email") or email or "").strip().lower(),
            "agent_id": str(payload.get("user_id") or payload.get("id") or "").strip() or None,
            "bb_calling": bool(payload.get("bb_calling")),
            "license": payload.get("license"),
            "edge_domain": payload.get("edge_domain"),
            "virtual_number_count": len(numbers),
            "virtual_number": virtual_number,
            "virtual_number_source": virtual_number_source,
        }

    def _iter_voip_auth_attempts(self, access_token: Optional[str] = None):
        """Iterate through available authentication methods for VoIP user lookup."""
        seen = set()
        
        # 1. Try provided access token (usually from OAuth flow or DB)
        if access_token:
            header = tuple(sorted(self._bearer_headers(access_token).items()))
            if header not in seen:
                seen.add(header)
                yield "bearer", dict(header)

        # 2. Try configured API Key (The "Root" fallback)
        if self.api_key:
            header = tuple(sorted(self._api_key_headers().items()))
            if header not in seen:
                seen.add(header)
                yield "api_key", dict(header)

    def _retrieve_voip_user(self, email: str, access_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves user details from FreJun, attempting multiple auth modes if necessary.
        """
        email = self._normalize_email(email)
        last_error = None
        
        attempts = list(self._iter_voip_auth_attempts(access_token))
        if not attempts:
            return self._build_voip_error(
                "oauth_not_connected",
                "No authentication methods available for FreJun user lookup.",
                status_code=401,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
                metadata={"agent_email": email},
            )

        for auth_mode, headers in attempts:
            try:
                # 1. Try the v2 retrieve-user endpoint.
                response = requests.get(
                    self.user_url,
                    headers=headers,
                    params={"email": email},
                    timeout=15,
                )
                body = self._parse_response_body(response)
                users = []

                if response.status_code == 200:
                    data = body.get("data")
                    if isinstance(data, dict):
                        users = [data]
                else:
                    # 2. Try V2 users endpoint as fallback for this auth mode
                    fallback_response = requests.get(
                        f"{self.base_url}/integrations/users/",
                        headers=headers,
                        params={"email": email},
                        timeout=15,
                    )
                    fallback_body = self._parse_response_body(fallback_response)
                    
                    if fallback_response.status_code == 200:
                        data = fallback_body.get("data")
                        if isinstance(data, dict):
                            users = [data]
                        elif isinstance(data, list):
                            users = data
                    else:
                        error_msg = self._extract_error_message(fallback_body, "User not found")
                        # If this auth mode explicitly says "not found", we keep looking via other modes
                        last_error = self._build_voip_error(
                            "frejun_user_not_found",
                            error_msg,
                            status_code=fallback_response.status_code,
                            action_label="Connect FreJun VoIP",
                            action_url="/api/auth/frejun-login",
                            metadata={"agent_email": email, "auth_mode": auth_mode},
                            raw_response=fallback_response.text,
                        )
                        continue

                # Process matched users
                matched_user = None
                for user in users:
                    if isinstance(user, dict) and (user.get("email") or "").strip().lower() == email.lower():
                        matched_user = user
                        break
                
                if matched_user is None and users:
                    matched_user = users[0]

                if isinstance(matched_user, dict):
                    logger.info(f"Successfully retrieved FreJun user {email} via {auth_mode}")
                    return {"success": True, "user": matched_user, "auth_mode": auth_mode}

            except requests.exceptions.Timeout:
                last_error = self._build_voip_error(
                    "frejun_user_not_found",
                    f"FreJun lookup timed out via {auth_mode}.",
                    status_code=504,
                    metadata={"agent_email": email},
                )
            except Exception as exc:
                logger.exception(f"Error retrieving FreJun VoIP user via {auth_mode}")
                last_error = self._build_voip_error(
                    "frejun_user_not_found",
                    str(exc),
                    status_code=500,
                    metadata={"agent_email": email},
                )

        return last_error or self._build_voip_error(
            "frejun_user_not_found",
            f"FreJun did not return a Browser VoIP user for {email} after trying all auth modes.",
            status_code=424,
            action_label="Connect FreJun VoIP",
            action_url="/api/auth/frejun-login",
            metadata={"agent_email": email},
        )

    def _enable_browser_calling(self, access_token: str, email: str) -> Dict[str, Any]:
        try:
            response = requests.patch(
                self.user_url,
                headers=self._bearer_headers(access_token),
                params={"email": email},
                json={"browser_calls": True},
                timeout=15,
            )
            body = self._parse_response_body(response)
            if response.status_code in (200, 201):
                return {
                    "success": True,
                    "data": body.get("data") if isinstance(body, dict) else None,
                    "version": "v2",
                }

            return self._build_voip_error(
                "browser_calling_enable_failed",
                self._extract_error_message(body, "FreJun could not enable browser calling for this user."),
                status_code=response.status_code or 424,
                action_label="Open FreJun Browser Calling",
                action_url=self._frejun_browser_calling_url(),
                metadata={"agent_email": email},
                raw_response=response.text,
            )
        except requests.exceptions.Timeout:
            return self._build_voip_error(
                "browser_calling_enable_failed",
                "FreJun timed out while enabling browser calling for this user.",
                status_code=504,
                action_label="Open FreJun Browser Calling",
                action_url=self._frejun_browser_calling_url(),
                metadata={"agent_email": email},
            )
        except Exception as exc:
            logger.exception("Error enabling FreJun browser calling")
            return self._build_voip_error(
                "browser_calling_enable_failed",
                str(exc),
                status_code=500,
                action_label="Open FreJun Browser Calling",
                action_url=self._frejun_browser_calling_url(),
                metadata={"agent_email": email},
            )

    def ensure_browser_voip_ready(self, recruiter_email: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        email = self._normalize_email(recruiter_email or self.user_email)
        if not email:
            return self._build_voip_error(
                "frejun_user_not_found",
                "Recruiter email is missing for FreJun Browser VoIP.",
                status_code=400,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
            )

        token_result = self.get_voip_access_token(recruiter_email=email, force_refresh=force_refresh)
        if not token_result.get("success"):
            return token_result

        access_token = token_result["access_token"]
        cached_agent_id = token_result.get("agent_id")
        cached_virtual_number = token_result.get("virtual_number")
        
        # PROD OPTIMIZATION: If we have cached metadata, skip identity lookup
        # This is now more aggressive: it works for BOTH cache hits and fresh rotations
        if not force_refresh and cached_agent_id and cached_virtual_number:
            logger.info(f"Using cached VOIP metadata for {email} (Agent: {cached_agent_id})")
            return {
                "success": True,
                "access_token": access_token,
                "agent_email": email,
                "agent_id": cached_agent_id,
                "virtual_number": cached_virtual_number,
                "source": token_result.get("source", "database_cache"),
                "metadata": {
                    "last_resolved_at": datetime.utcnow().isoformat(),
                    "cached": True
                }
            }

        user_result = self._retrieve_voip_user(email, access_token=access_token)
        
        if not user_result.get("success"):
            return user_result

        # PROD TWEAK: If lookup only worked via api_key fallback, we still proceed
        # but log a warning. This avoids blocking users whose OAuth tokens are 
        # valid for calling but restricted for identity lookups.
        if user_result.get("auth_mode") == "api_key" and access_token:
            logger.warning(f"FreJun OAuth token for {email} failed identity lookup, but API Key worked. Proceeding with hybrid auth.")

        user = user_result["user"]
        agent_id = user.get("user_id") or user.get("id")
        
        # Step 3: Resolve Virtual Number
        number_result = self._resolve_virtual_number(recruiter_email=email, user_payload=user)
        if not number_result.get("success"):
            return number_result

        virtual_number = number_result["virtual_number"]
        
        # PROD OPTIMIZATION: Persist the resolved metadata back to the DB bridge
        # Cache on every successful resolution to ensure follow-up calls are lightning fast
        if token_result.get("source") in ("database_cache", "oauth_rotation"):
            self._cache_voip_meta(email, agent_id, virtual_number)

        return {
            "success": True,
            "access_token": access_token,
            "agent_email": email,
            "agent_id": agent_id,
            "virtual_number": virtual_number,
            "source": user_result.get("auth_mode", "bearer"),
            "metadata": user_result.get("metadata", {}),
        }

    def get_voip_agent(self, recruiter_email: Optional[str] = None) -> Dict[str, Any]:
        return self.ensure_browser_voip_ready(recruiter_email=recruiter_email)

    def _resolve_virtual_number_from_user(self, user_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        configured_virtual = self._normalize_phone(self.virtual_number)
        if not isinstance(user_payload, dict):
            return {"success": False, "error": "FreJun user payload missing for virtual-number resolution"}

        numbers = user_payload.get("virtual_numbers")
        if not isinstance(numbers, list):
            numbers = []

        normalized_numbers = {}
        for item in numbers:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_phone(item.get("number"))
            if normalized:
                normalized_numbers[normalized] = item

        if configured_virtual:
            if configured_virtual in normalized_numbers:
                selected = normalized_numbers[configured_virtual]
                return {
                    "success": True,
                    "virtual_number": selected.get("number") or configured_virtual,
                    "source": "user_payload_configured",
                }

            configured_digits = re.sub(r"\D", "", configured_virtual)
            if len(configured_digits) >= 10:
                last_ten = configured_digits[-10:]
                for normalized, item in normalized_numbers.items():
                    digits = re.sub(r"\D", "", normalized)
                    if len(digits) >= 10 and digits[-10:] == last_ten:
                        return {
                            "success": True,
                            "virtual_number": item.get("number") or normalized,
                            "source": "user_payload_configured_last10",
                        }

        if numbers:
            selected = next(
                (item for item in numbers if isinstance(item, dict) and item.get("default_calling_number")),
                numbers[0],
            )
            normalized = self._normalize_phone(selected.get("number"))
            if normalized:
                return {
                    "success": True,
                    "virtual_number": selected.get("number") or normalized,
                    "source": "user_payload_default" if selected.get("default_calling_number") else "user_payload_first_available",
                }

        if configured_virtual:
            return {
                "success": True,
                "virtual_number": configured_virtual,
                "source": "configured_fallback",
            }

        return {"success": False, "error": "No virtual numbers available in FreJun user payload"}

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

    def _resolve_virtual_number(self, recruiter_email: Optional[str] = None, user_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        email = recruiter_email or self.user_email
        user_resolution = self._resolve_virtual_number_from_user(user_payload)
        if user_resolution.get("success"):
            return user_resolution
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
    ) -> Dict:
        """
        Initiates a Browser VoIP call via FreJun.
        FreJun calls the candidate first, then routes the recruiter to the softphone.
        """
        email = recruiter_email or self.user_email
        candidate_phone = self._normalize_phone(candidate_phone)

        # Validation
        if not email:
            return self._build_voip_error(
                "frejun_user_not_found",
                "Recruiter email missing (Set FREJUN_USER_EMAIL in .env)",
                status_code=400,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
            )
        if not candidate_phone:
            return self._build_voip_error(
                "voip_call_start_failed",
                "Candidate phone missing",
                status_code=400,
                metadata={"agent_email": self._normalize_email(email)},
            )
        if len(candidate_phone) == 10 and not candidate_phone.startswith("+"):
            candidate_phone = f"+91{candidate_phone}"

        readiness = self.ensure_browser_voip_ready(email)
        if not readiness.get("success"):
            return readiness

        virtual_number = readiness.get("virtual_number")
        if not virtual_number:
            return self._build_voip_error(
                "virtual_number_missing",
                "Unable to determine a FreJun virtual number for this call.",
                status_code=424,
                action_label="Open FreJun Virtual Numbers",
                action_url=self._frejun_virtual_numbers_url(),
                metadata=readiness.get("metadata"),
            )

        # Format virtual number with country code for VoIP endpoint requirements
        formatted_virtual_number = virtual_number
        if len(str(formatted_virtual_number)) == 10 and not str(formatted_virtual_number).startswith("+"):
            formatted_virtual_number = f"+91{formatted_virtual_number}"

        url = self.call_to_voip_url
        params = {"email": email}
        payload = {
            "agent_id": readiness["agent_id"],
            "dstn_number": candidate_phone,
            "virtual_number": formatted_virtual_number,
            "candidate_name": candidate_name or "Candidate",
        }
        if candidate_id:
            payload["candidate_id"] = str(candidate_id)
        if job_id:
            payload["job_id"] = str(job_id)
        if transaction_id:
            payload["transaction_id"] = str(transaction_id)

        attempts = list(self._iter_call_auth_attempts())
        if not attempts:
            return self._build_voip_error(
                "voip_call_start_failed",
                "No authentication methods available for FreJun call initiation.",
                status_code=401,
                action_label="Connect FreJun VoIP",
                action_url="/api/auth/frejun-login",
                metadata=readiness.get("metadata"),
            )

        last_error = None
        for auth_mode, headers in attempts:
            try:
                logger.info(
                    "Initiating FreJun Browser VoIP call to %s via %s (Mode: %s)",
                    candidate_phone,
                    formatted_virtual_number,
                    auth_mode
                )
                
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
                        "dial_mode": "voip",
                        "auth_mode": auth_mode
                    }
                else:
                    message = self._extract_error_message(body, response.text or "FreJun request failed")
                    last_error = self._build_voip_error(
                        "voip_call_start_failed",
                        f"FreJun Error ({response.status_code}): {message}",
                        status_code=response.status_code or 502,
                        action_label="Retry VoIP",
                        metadata=readiness.get("metadata"),
                        raw_response=response.text,
                    )
                    # If it's an auth error, continue to next attempt. Otherwise, stop.
                    # PROD BRIDGE: If Bearer fails (even with 404), we MUST continue to try API Key
                    if response.status_code in (401, 403, 404):
                        continue
                    break
            except requests.exceptions.Timeout:
                last_error = self._build_voip_error(
                    "voip_call_start_failed",
                    f"FreJun API request timed out via {auth_mode}",
                    status_code=504,
                    action_label="Retry VoIP",
                    metadata=readiness.get("metadata"),
                )
            except Exception as e:
                logger.exception(f"Error calling FreJun API via {auth_mode}")
                last_error = self._build_voip_error(
                    "voip_call_start_failed",
                    str(e),
                    status_code=500,
                    action_label="Retry VoIP",
                    metadata=readiness.get("metadata"),
                )

        return last_error

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

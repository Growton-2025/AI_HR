"""Per-user Google connection used to read the user's own Google Sheets.

The browser gets an authorization code from Google Identity Services (popup,
``spreadsheets.readonly``). ``connect`` exchanges it for a refresh token, which
is stored encrypted per user; ``access_token_for_user`` trades it for a
short-lived access token whenever a sheet is read.

The DB helpers are module-level so tests can replace them without a database.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

import httpx
from cryptography.fernet import Fernet, InvalidToken
from jose import jwt

from backend.db.connection import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
# ``email`` lets the UI show which Google account is connected.
REQUESTED_SCOPES = f"openid email {SHEETS_SCOPE}"
HTTP_TIMEOUT_SECONDS = 20.0

_tables_ready = False
_http_lock = threading.Lock()
_http_client: Optional[httpx.Client] = None


def http() -> httpx.Client:
    """One shared client for every Google call, bound to IPv4.

    Google's hosts resolve to several IPv6 addresses first; on a network where IPv6 is
    announced but does not route, each one is tried for the full timeout before IPv4
    is reached, so a single read took minutes. Azure's outbound traffic is IPv4 too."""
    global _http_client
    if _http_client is None:
        with _http_lock:
            if _http_client is None:
                _http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=1),
                                            timeout=HTTP_TIMEOUT_SECONDS)
    return _http_client
_tables_lock = threading.Lock()


class GoogleAuthError(RuntimeError):
    """``code`` is one of: not_configured, not_connected, reconnect, exchange_failed."""

    def __init__(self, message: str, code: str = "exchange_failed"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- config
def _client_config() -> tuple[str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise GoogleAuthError(
            "Google Sheets import is not set up on the server (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET).",
            code="not_configured",
        )
    return client_id, client_secret


def _fernet() -> Fernet:
    key = os.getenv("GOOGLE_TOKEN_ENC_KEY", "").strip()
    if not key:
        raise GoogleAuthError(
            "Google Sheets import is not set up on the server (GOOGLE_TOKEN_ENC_KEY).", code="not_configured"
        )
    try:
        return Fernet(key.encode())
    except Exception as exc:
        raise GoogleAuthError(f"GOOGLE_TOKEN_ENC_KEY is not a valid Fernet key: {exc}", code="not_configured")


def is_configured() -> bool:
    try:
        _client_config()
        _fernet()
        return True
    except GoogleAuthError:
        return False


def encrypt_token(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def decrypt_token(blob: str) -> str:
    try:
        return _fernet().decrypt(blob.encode()).decode()
    except InvalidToken:
        raise GoogleAuthError("Stored Google connection can no longer be read. Connect Google again.",
                              code="reconnect")


# ----------------------------------------------------------------------- DB helpers
def _with_conn(fn):
    conn = get_db_connection(validate=False, register_pgvector=False)
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        with conn.cursor() as cur:
            result = fn(cur)
        conn.commit()
        return result
    except Exception:
        try:
            conn.rollback()
        except Exception:      # a dropped connection cannot roll back; keep the real error
            pass
        raise
    finally:
        return_db_connection(conn)


def ensure_tables() -> None:
    """Create the Google Sheets tables once per process (idempotent)."""
    global _tables_ready
    if _tables_ready:
        return
    with _tables_lock:
        if _tables_ready:
            return

        def run(cur):
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_google_tokens (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    refresh_token_enc TEXT NOT NULL,
                    scope TEXT,
                    google_email VARCHAR(320),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sheet_connections (
                    id SERIAL PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    connected_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    role_id INTEGER,
                    spreadsheet_id VARCHAR(128) NOT NULL,
                    tab_title VARCHAR(512) NOT NULL,
                    sheet_title VARCHAR(512),
                    mapping JSONB DEFAULT '{}'::jsonb,
                    enrich_linkedin BOOLEAN DEFAULT TRUE,
                    enrichment_mode VARCHAR(50) DEFAULT 'none',
                    last_synced_at TIMESTAMP,
                    last_upload_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS sheet_connections_unique_target
                ON sheet_connections (owner_user_id, COALESCE(role_id, 0), spreadsheet_id, tab_title);
                """
            )

            # Automatic sync (services/sheet_auto_sync.py): which rows were already
            # imported, which are waiting to settle or for LinkedIn data, and when the
            # sheet was last read.
            for column in ("seen_row_keys JSONB", "pending_rows JSONB", "last_checked_at TIMESTAMP",
                           "last_error TEXT", "retry_after TIMESTAMP", "content_digest VARCHAR(64)",
                           "last_changed_at TIMESTAMP", "held_rows JSONB"):
                cur.execute(f"ALTER TABLE sheet_connections ADD COLUMN IF NOT EXISTS {column};")

        _with_conn(run)
        _tables_ready = True


def save_tokens(user_id: int, refresh_token: str, scope: str, google_email: Optional[str]) -> None:
    ensure_tables()
    blob = encrypt_token(refresh_token)

    def run(cur):
        cur.execute(
            """
            INSERT INTO user_google_tokens (user_id, refresh_token_enc, scope, google_email, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET refresh_token_enc = EXCLUDED.refresh_token_enc,
                  scope = EXCLUDED.scope,
                  google_email = COALESCE(EXCLUDED.google_email, user_google_tokens.google_email),
                  updated_at = NOW()
            """,
            (user_id, blob, scope, google_email),
        )

    _with_conn(run)


def load_tokens(user_id: int) -> Optional[Dict[str, Any]]:
    ensure_tables()

    def run(cur):
        cur.execute(
            "SELECT refresh_token_enc, scope, google_email, updated_at FROM user_google_tokens WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"refresh_token_enc": row[0], "scope": row[1] or "", "google_email": row[2], "updated_at": row[3]}

    return _with_conn(run)


def delete_tokens(user_id: int) -> None:
    ensure_tables()
    _with_conn(lambda cur: cur.execute("DELETE FROM user_google_tokens WHERE user_id = %s", (user_id,)))


# ------------------------------------------------------------------------- Google
def _post_token(data: Dict[str, str]) -> httpx.Response:
    return http().post(TOKEN_URL, data=data, timeout=HTTP_TIMEOUT_SECONDS)


def _email_from_id_token(id_token: Optional[str]) -> Optional[str]:
    if not id_token:
        return None
    try:
        # Received directly from Google over TLS in the code exchange, so the
        # claims are read without re-verifying the signature.
        return jwt.get_unverified_claims(id_token).get("email")
    except Exception:
        return None


def connect(user_id: int, code: str) -> Dict[str, Any]:
    """Exchange a browser authorization code and store the refresh token."""
    client_id, client_secret = _client_config()
    _fernet()  # fail before talking to Google if the token cannot be stored
    if not (code or "").strip():
        raise GoogleAuthError("Missing Google authorization code.")
    try:
        resp = _post_token({
            "code": code.strip(),
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": "postmessage",  # required value for the popup code flow
            "grant_type": "authorization_code",
        })
    except httpx.HTTPError as exc:
        raise GoogleAuthError(f"Could not reach Google: {exc}")
    payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code != 200:
        detail = payload.get("error_description") or payload.get("error") or f"HTTP {resp.status_code}"
        raise GoogleAuthError(f"Google rejected the sign-in: {detail}")

    scope = str(payload.get("scope") or "")
    if SHEETS_SCOPE not in scope.split():
        raise GoogleAuthError(
            "Google Sheets access was not granted. Connect again and tick the spreadsheet permission.",
            code="reconnect",
        )
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        # Google only returns a refresh token on a fresh consent; keep an existing one.
        existing = load_tokens(user_id)
        if not existing:
            raise GoogleAuthError(
                "Google did not return a long-lived token. Remove Hayasa under your Google Account's "
                "third-party access, then connect again.",
                code="reconnect",
            )
        return {"connected": True, "google_email": existing.get("google_email")}

    email = _email_from_id_token(payload.get("id_token"))
    save_tokens(user_id, refresh_token, scope, email)
    return {"connected": True, "google_email": email}


def access_token_for_user(user_id: int) -> str:
    """A fresh access token for the user's stored Google connection."""
    client_id, client_secret = _client_config()
    stored = load_tokens(user_id)
    if not stored:
        raise GoogleAuthError("Connect your Google account first.", code="not_connected")
    refresh_token = decrypt_token(stored["refresh_token_enc"])
    try:
        resp = _post_token({
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        })
    except httpx.HTTPError as exc:
        raise GoogleAuthError(f"Could not reach Google: {exc}")
    payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code != 200:
        if payload.get("error") == "invalid_grant":
            # Revoked, or expired (unverified apps: 7 days). The stored token is dead.
            delete_tokens(user_id)
            raise GoogleAuthError("Your Google connection expired. Connect Google again.", code="reconnect")
        detail = payload.get("error_description") or payload.get("error") or f"HTTP {resp.status_code}"
        raise GoogleAuthError(f"Google token refresh failed: {detail}")
    token = payload.get("access_token")
    if not token:
        raise GoogleAuthError("Google token refresh returned no access token.")
    return str(token)


def status(user_id: int) -> Dict[str, Any]:
    """Connection state for the UI. ``client_id`` is public and lets the browser open
    the consent popup without its own copy of the setting."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip() or None
    if not is_configured():
        return {"configured": False, "connected": False, "google_email": None, "client_id": client_id}
    stored = load_tokens(user_id)
    return {
        "configured": True,
        "connected": bool(stored),
        "google_email": stored.get("google_email") if stored else None,
        "client_id": client_id,
    }


def disconnect(user_id: int) -> None:
    stored = load_tokens(user_id)
    if not stored:
        return
    try:
        http().post(REVOKE_URL, params={"token": decrypt_token(stored["refresh_token_enc"])},
                   timeout=HTTP_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("google token revoke failed for user %s; removing the stored token anyway", user_id)
    delete_tokens(user_id)

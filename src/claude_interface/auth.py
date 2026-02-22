"""
Claude Interface - OAuth Authentication

Handles OAuth login flow and token refresh for Claude Pro/Max subscriptions.
Uses the same OAuth flow as Claude Code CLI.
"""

import hashlib
import base64
import secrets
import time
from typing import Callable, Awaitable
import httpx

from .types import OAuthCredentials


# ─────────────────────────────────────────────────────────────────────────────
# Constants (matching Claude Code)
# ─────────────────────────────────────────────────────────────────────────────

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"

# Token refresh buffer (refresh 5 minutes before expiry)
REFRESH_BUFFER_MS = 5 * 60 * 1000


# ─────────────────────────────────────────────────────────────────────────────
# PKCE Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base64url_encode(data: bytes) -> str:
    """Base64 URL encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    # Generate random verifier (43-128 characters)
    verifier = _base64url_encode(secrets.token_bytes(32))
    
    # Generate challenge (SHA-256 hash of verifier)
    challenge_bytes = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url_encode(challenge_bytes)
    
    return verifier, challenge


# ─────────────────────────────────────────────────────────────────────────────
# OAuth Flow
# ─────────────────────────────────────────────────────────────────────────────

async def login(
    on_auth_url: Callable[[str], None],
    on_prompt_code: Callable[[], Awaitable[str]],
) -> OAuthCredentials:
    """
    Start the OAuth login flow for Claude Pro/Max.
    
    Args:
        on_auth_url: Callback with the authorization URL to open in browser
        on_prompt_code: Async callback to get the authorization code from user
        
    Returns:
        OAuth credentials
        
    Example:
        ```python
        async def get_code():
            return input("Paste code: ")
            
        credentials = await login(
            on_auth_url=lambda url: print(f"Open: {url}"),
            on_prompt_code=get_code,
        )
        ```
    """
    verifier, challenge = _generate_pkce()
    
    # Build authorization URL
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    
    query = "&".join(f"{k}={v}" for k, v in params.items())
    auth_url = f"{AUTHORIZE_URL}?{query}"
    
    # Notify caller to open browser
    on_auth_url(auth_url)
    
    # Wait for authorization code
    auth_code = (await on_prompt_code()).strip()
    
    # Validate format: code#state
    if "#" not in auth_code:
        raise ValueError(
            "Invalid authorization code format. Expected: code#state\n"
            "Copy the full code including the # and everything after it."
        )
    
    code, state = auth_code.split("#", 1)
    if not code or not state:
        raise ValueError(
            "Invalid authorization code format. Expected: code#state\n"
            "Both code and state parts are required."
        )
    
    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code != 200:
            raise ValueError(f"OAuth token exchange failed: {response.text}")
        
        data = response.json()
    
    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=int(time.time() * 1000) + data["expires_in"] * 1000 - REFRESH_BUFFER_MS,
    )


async def refresh_token(refresh_token_str: str) -> OAuthCredentials:
    """
    Refresh an OAuth token using the refresh token.
    
    Args:
        refresh_token_str: The refresh token from previous login/refresh
        
    Returns:
        Fresh OAuth credentials
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token_str,
            },
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code != 200:
            raise ValueError(f"OAuth token refresh failed: {response.text}")
        
        data = response.json()
    
    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=int(time.time() * 1000) + data["expires_in"] * 1000 - REFRESH_BUFFER_MS,
    )


def is_expired(credentials: OAuthCredentials) -> bool:
    """Check if credentials are expired or about to expire."""
    return int(time.time() * 1000) >= credentials.expires_at


def is_oauth_token(token: str) -> bool:
    """Check if the token is an OAuth token (vs API key)."""
    return token.startswith("sk-ant-oat")

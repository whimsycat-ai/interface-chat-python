"""
Claude Interface - Lightweight Claude API interface with OAuth support.

Example:
    ```python
    from claude_interface import ClaudeClient, AuthConfig, login
    
    # Login with OAuth (Claude Pro/Max)
    credentials = await login(
        on_auth_url=lambda url: print(f"Open: {url}"),
        on_prompt_code=lambda: input("Code: "),
    )
    
    # Create client
    client = ClaudeClient(auth=AuthConfig(oauth=credentials))
    
    # Create session and chat
    client.create_session(name="Code Review")
    result = await client.send("Review this code...")
    print(result.content)
    ```
"""

from .client import ClaudeClient
from .auth import login, refresh_token, is_expired, is_oauth_token
from .session import SessionManager
from .memory import MemoryManager
from .logger import Logger, LogStats
from .types import (
    # Auth
    OAuthCredentials,
    AuthConfig,
    # Messages
    Message,
    TextContent,
    ImageContent,
    ContentBlock,
    # Session
    Session,
    SessionSummary,
    # Memory
    MemoryEntry,
    MemoryType,
    # Logging
    LogEntry,
    RequestPayload,
    ResponsePayload,
    Usage,
    # Client
    SendResult,
    SpinOutOptions,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "ClaudeClient",
    # Auth
    "login",
    "refresh_token",
    "is_expired",
    "is_oauth_token",
    "OAuthCredentials",
    "AuthConfig",
    # Session
    "SessionManager",
    "Session",
    "SessionSummary",
    # Memory
    "MemoryManager",
    "MemoryEntry",
    "MemoryType",
    # Logging
    "Logger",
    "LogStats",
    "LogEntry",
    "RequestPayload",
    "ResponsePayload",
    "Usage",
    # Messages
    "Message",
    "TextContent",
    "ImageContent",
    "ContentBlock",
    # Client
    "SendResult",
    "SpinOutOptions",
]

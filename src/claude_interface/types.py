"""
Claude Interface - Type Definitions
"""

from dataclasses import dataclass, field
from typing import Literal, Any
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OAuthCredentials:
    """OAuth credentials for Claude Pro/Max."""
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp in milliseconds


@dataclass
class AuthConfig:
    """Authentication configuration."""
    oauth: OAuthCredentials | None = None
    api_key: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

Role = Literal["user", "assistant"]


@dataclass
class TextContent:
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content block."""
    type: Literal["image"] = "image"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = "image/png"
    data: str = ""  # Base64 encoded


ContentBlock = TextContent | ImageContent


@dataclass
class Message:
    """A conversation message."""
    role: Role
    content: str | list[ContentBlock]
    timestamp: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Memory Types
# ─────────────────────────────────────────────────────────────────────────────

MemoryType = Literal["fact", "preference", "context", "summary", "custom"]


@dataclass
class MemoryEntry:
    """A memory entry associated with a session."""
    id: str
    content: str
    type: MemoryType = "fact"
    tags: list[str] = field(default_factory=list)
    created_at: int = 0
    priority: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Session Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """A conversation session."""
    id: str
    name: str | None = None
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory: list[MemoryEntry] = field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0


@dataclass
class SessionSummary:
    """Summary of a session for listing."""
    id: str
    name: str | None
    message_count: int
    created_at: int
    updated_at: int


# ─────────────────────────────────────────────────────────────────────────────
# Logging Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Usage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass
class RequestPayload:
    """Logged request payload."""
    model: str
    messages: list[Message]
    system_prompt: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass
class ResponsePayload:
    """Logged response payload."""
    content: str
    stop_reason: str
    usage: Usage
    model: str


@dataclass
class LogEntry:
    """A log entry for request/response."""
    id: str
    session_id: str
    timestamp: int
    direction: Literal["request", "response"]
    payload: RequestPayload | ResponsePayload
    duration_ms: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Client Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SendResult:
    """Result from sending a message."""
    content: str
    stop_reason: Literal["stop", "length", "tool_use", "error"]
    usage: Usage
    duration_ms: int
    model: str


@dataclass
class SpinOutOptions:
    """Options for spinning out a conversation thread."""
    topic: str | None = None
    name: str | None = None
    system_prompt: str | None = None
    include_last_n: int | None = None
    message_ids: list[int] | None = None
    include_all: bool = False
    generate_summary: bool = False
    copy_memories: bool = False
    memory_tags: list[str] | None = None
    initial_prompt: str | None = None
    switch_to: bool = True

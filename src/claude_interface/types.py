"""
Claude Interface - Type Definitions
"""

from dataclasses import dataclass, field
from typing import Literal, Any, Callable, Awaitable
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


# Supported image media types
ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
SUPPORTED_IMAGE_TYPES: set[ImageMediaType] = {"image/jpeg", "image/png", "image/gif", "image/webp"}


@dataclass
class TextContent:
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content block."""
    type: Literal["image"] = "image"
    media_type: ImageMediaType = "image/png"
    data: str = ""  # Base64 encoded


ContentBlock = TextContent | ImageContent


@dataclass
class Message:
    """A conversation message."""
    role: Role
    content: str | list[ContentBlock]
    timestamp: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Tool Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class Tool:
    """A tool that Claude can use."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Awaitable[str]] | Callable[..., str] | None = None


@dataclass
class ToolCall:
    """A tool call from Claude."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_call_id: str
    content: str
    is_error: bool = False


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
    tools: list[Tool] = field(default_factory=list)
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
    tool_calls: list[ToolCall] = field(default_factory=list)


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


# ─────────────────────────────────────────────────────────────────────────────
# Image Helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageInput:
    """Helper for creating image content."""
    data: str  # Base64 encoded or file path
    media_type: ImageMediaType = "image/png"
    
    @classmethod
    def from_file(cls, path: str) -> "ImageInput":
        """Create ImageInput from a file path."""
        import base64
        from pathlib import Path
        
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        
        media_type_map: dict[str, ImageMediaType] = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        
        media_type = media_type_map.get(suffix, "image/png")
        
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return cls(data=data, media_type=media_type)
    
    @classmethod
    def from_base64(cls, data: str, media_type: ImageMediaType = "image/png") -> "ImageInput":
        """Create ImageInput from base64 data."""
        return cls(data=data, media_type=media_type)
    
    @classmethod
    def from_url(cls, url: str) -> "ImageInput":
        """Create ImageInput from a URL (downloads the image)."""
        import base64
        import httpx
        
        response = httpx.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "image/png")
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()
        
        # Validate content-type is a supported image type
        if content_type not in SUPPORTED_IMAGE_TYPES:
            content_type = "image/png"  # Default to PNG for unsupported types
        
        # Cast to ImageMediaType after validation
        media_type: ImageMediaType = content_type  # type: ignore[assignment]
        
        data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=data, media_type=media_type)

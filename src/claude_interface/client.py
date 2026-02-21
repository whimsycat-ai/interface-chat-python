"""
Claude Interface - Main Client

The primary interface for interacting with Claude via OAuth or API key.
Handles authentication, sessions, logging, and memory.
"""

import os
import time
from pathlib import Path
from typing import AsyncIterator, Any

import anthropic

from .types import (
    AuthConfig,
    OAuthCredentials,
    Session,
    Message,
    SendResult,
    SpinOutOptions,
    Usage,
    RequestPayload,
    ResponsePayload,
)
from .auth import is_expired, is_oauth_token, refresh_token
from .session import SessionManager
from .memory import MemoryManager
from .logger import Logger


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_STORAGE_DIR = Path.home() / ".claude-interface"

# Claude Code stealth headers (for OAuth)
CLAUDE_CODE_VERSION = "2.1.2"
OAUTH_HEADERS = {
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
    "x-app": "cli",
}

# Standard API headers
API_HEADERS = {
    "anthropic-beta": "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
}


class ClaudeClient:
    """
    Main client for interacting with Claude.
    
    Example:
        ```python
        client = ClaudeClient(auth=AuthConfig(api_key="sk-ant-..."))
        client.create_session(name="Code Review")
        result = await client.send("Review this code...")
        print(result.content)
        ```
    """
    
    def __init__(
        self,
        auth: AuthConfig,
        model: str = DEFAULT_MODEL,
        storage_dir: str | Path | None = None,
        enable_logging: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
        base_url: str | None = None,
    ):
        self._auth = auth
        self._model = model
        self._storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self._enable_logging = enable_logging
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking = thinking
        self._thinking_budget = thinking_budget
        self._base_url = base_url
        
        self._client: anthropic.AsyncAnthropic | None = None
        self._oauth_credentials = auth.oauth
        self._current_session_id: str | None = None
        
        self._session_manager = SessionManager(str(self._storage_dir))
        self._logger = Logger(str(self._storage_dir))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Authentication
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client, refreshing OAuth if needed."""
        if self._client and not self._needs_refresh():
            return self._client
        
        # Refresh OAuth token if needed
        if self._oauth_credentials and is_expired(self._oauth_credentials):
            await self._refresh_auth()
        
        # Create new client
        self._client = self._create_client()
        return self._client
    
    def _needs_refresh(self) -> bool:
        """Check if OAuth needs refresh."""
        return self._oauth_credentials is not None and is_expired(self._oauth_credentials)
    
    async def _refresh_auth(self) -> None:
        """Refresh OAuth credentials."""
        if not self._oauth_credentials:
            raise ValueError("No OAuth credentials to refresh")
        
        self._oauth_credentials = await refresh_token(self._oauth_credentials.refresh_token)
    
    def _create_client(self) -> anthropic.AsyncAnthropic:
        """Create a new Anthropic client."""
        token = self._get_auth_token()
        is_oauth = is_oauth_token(token)
        
        if is_oauth:
            return anthropic.AsyncAnthropic(
                api_key="",  # Not used for OAuth
                auth_token=token,
                base_url=self._base_url,
                default_headers=OAUTH_HEADERS,
            )
        
        return anthropic.AsyncAnthropic(
            api_key=token,
            base_url=self._base_url,
            default_headers=API_HEADERS,
        )
    
    def _get_auth_token(self) -> str:
        """Get the current auth token."""
        if self._oauth_credentials:
            return self._oauth_credentials.access_token
        if self._auth.api_key:
            return self._auth.api_key
        raise ValueError("No authentication configured")
    
    def get_oauth_credentials(self) -> OAuthCredentials | None:
        """Get current OAuth credentials (for saving)."""
        return self._oauth_credentials
    
    def set_oauth_credentials(self, credentials: OAuthCredentials) -> None:
        """Update OAuth credentials."""
        self._oauth_credentials = credentials
        self._client = None  # Force client recreation
    
    # ─────────────────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_session(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
    ) -> Session:
        """Create a new session."""
        session = self._session_manager.create(name=name, system_prompt=system_prompt)
        self._current_session_id = session.id
        return session
    
    def load_session(self, session_id: str) -> Session | None:
        """Load an existing session."""
        session = self._session_manager.get(session_id)
        if session:
            self._current_session_id = session.id
        return session
    
    def get_current_session(self) -> Session | None:
        """Get the current session."""
        if not self._current_session_id:
            return None
        return self._session_manager.get(self._current_session_id)
    
    def list_sessions(self):
        """List all sessions."""
        return self._session_manager.list()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        self._logger.clear_logs(session_id)
        return self._session_manager.delete(session_id)
    
    @property
    def sessions(self) -> SessionManager:
        """Get the session manager for advanced operations."""
        return self._session_manager
    
    # ─────────────────────────────────────────────────────────────────────────
    # Spin Out
    # ─────────────────────────────────────────────────────────────────────────
    
    async def spin_out(self, options: SpinOutOptions) -> Session:
        """
        Spin out a thought/topic into a new session.
        
        Creates a new session with context from the current conversation.
        
        Example:
            ```python
            new_session = await client.spin_out(SpinOutOptions(
                topic="TypeScript best practices",
                include_last_n=4,
                system_prompt="Focus on TypeScript patterns discussed earlier.",
            ))
            ```
        """
        current_session = self.get_current_session()
        if not current_session:
            raise ValueError("No active session to spin out from")
        
        # Create new session
        new_session = self._session_manager.create(
            name=options.name or f"Spinout: {options.topic or 'New Thread'}",
            system_prompt=options.system_prompt,
            metadata={
                "spun_out_from": current_session.id,
                "spun_out_at": int(time.time() * 1000),
                "topic": options.topic,
            },
        )
        
        # Determine which messages to include
        context_messages: list[Message] = []
        
        if options.message_ids:
            context_messages = [
                m for i, m in enumerate(current_session.messages)
                if i in options.message_ids
            ]
        elif options.include_last_n and options.include_last_n > 0:
            context_messages = current_session.messages[-options.include_last_n:]
        elif options.include_all:
            context_messages = list(current_session.messages)
        
        # Copy memories if requested
        if options.copy_memories:
            memory = self.get_memory()
            if memory:
                memories = memory.get_all()
                if options.memory_tags:
                    memories = [
                        m for m in memories
                        if any(t in m.tags for t in options.memory_tags)
                    ]
                new_session.memory = list(memories)
        
        # Generate context summary if requested
        if options.generate_summary and context_messages:
            summary_prompt = (
                f'Summarize the following conversation context for a new focused '
                f'discussion about "{options.topic or "the topic"}". Be concise but '
                f'preserve key details:\n\n'
            )
            for m in context_messages:
                content = m.content if isinstance(m.content, str) else "[complex content]"
                summary_prompt += f"{m.role}: {content}\n\n"
            
            # Use the API to generate a summary
            client = await self._get_client()
            response = await client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            
            summary_text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            
            new_session.messages.append(Message(
                role="user",
                content=f"[Context from previous conversation]\n\n{summary_text}",
                timestamp=int(time.time() * 1000),
            ))
        elif context_messages and not options.generate_summary:
            new_session.messages = [
                Message(role=m.role, content=m.content, timestamp=int(time.time() * 1000))
                for m in context_messages
            ]
        
        # Add initial prompt if provided
        if options.initial_prompt:
            new_session.messages.append(Message(
                role="user",
                content=options.initial_prompt,
                timestamp=int(time.time() * 1000),
            ))
        
        self._session_manager.save(new_session.id)
        
        # Optionally switch to new session
        if options.switch_to:
            self._current_session_id = new_session.id
        
        return new_session
    
    # ─────────────────────────────────────────────────────────────────────────
    # Memory Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_memory(self) -> MemoryManager | None:
        """Get memory manager for the current session."""
        session = self.get_current_session()
        if not session:
            return None
        return MemoryManager(
            session,
            lambda: self._session_manager.save(session.id),
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_logs(self, session_id: str | None = None):
        """Get logs for a session."""
        id = session_id or self._current_session_id
        if not id:
            raise ValueError("No session ID provided")
        return self._logger.get_logs(id)
    
    def get_log_stats(self, session_id: str | None = None):
        """Get log statistics for a session."""
        id = session_id or self._current_session_id
        if not id:
            raise ValueError("No session ID provided")
        return self._logger.get_stats(id)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────────
    
    async def send(
        self,
        content: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> SendResult:
        """Send a message and get a response."""
        session = self.get_current_session()
        if not session:
            raise ValueError("No active session. Create or load a session first.")
        
        client = await self._get_client()
        model = model or self._model
        max_tokens = max_tokens or self._max_tokens
        is_oauth = is_oauth_token(self._get_auth_token())
        
        # Add user message to session
        user_message = Message(
            role="user",
            content=content,
            timestamp=int(time.time() * 1000),
        )
        self._session_manager.add_message(session.id, user_message)
        
        # Build system prompt
        sys_prompt = system_prompt or session.system_prompt or ""
        
        # For OAuth, prepend required Claude Code identity
        if is_oauth and "Claude Code" not in sys_prompt:
            sys_prompt = "You are Claude Code, Anthropic's official CLI for Claude.\n\n" + sys_prompt
        
        # Add memory context
        memory = self.get_memory()
        if memory and memory.count() > 0:
            sys_prompt += "\n\n" + memory.format_as_context()
        
        # Build messages for API
        messages = [
            {"role": m.role, "content": m.content}
            for m in session.messages
        ]
        
        # Log request
        request_id = ""
        if self._enable_logging:
            request_id = self._logger.log_request(
                session.id,
                RequestPayload(
                    model=model,
                    messages=session.messages,
                    system_prompt=sys_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        
        start_time = time.time()
        
        try:
            # Build request params
            params: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            
            if sys_prompt:
                params["system"] = [{
                    "type": "text",
                    "text": sys_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            
            if temperature is not None:
                params["temperature"] = temperature
            elif self._temperature is not None:
                params["temperature"] = self._temperature
            
            # Make API call
            response = await client.messages.create(**params)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract text content
            response_text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            
            # Add assistant message to session
            assistant_message = Message(
                role="assistant",
                content=response_text,
                timestamp=int(time.time() * 1000),
            )
            self._session_manager.add_message(session.id, assistant_message)
            
            # Build result
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", None),
                cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
            )
            
            result = SendResult(
                content=response_text,
                stop_reason=self._map_stop_reason(response.stop_reason),
                usage=usage,
                duration_ms=duration_ms,
                model=response.model,
            )
            
            # Update session metadata
            self._session_manager.update_metadata(session.id, {
                "model": response.model,
                "total_tokens": (session.metadata.get("total_tokens", 0) +
                                usage.input_tokens + usage.output_tokens),
            })
            
            # Log response
            if self._enable_logging:
                self._logger.log_response(
                    session.id,
                    request_id,
                    ResponsePayload(
                        content=response_text,
                        stop_reason=result.stop_reason,
                        usage=usage,
                        model=response.model,
                    ),
                    duration_ms,
                )
            
            return result
            
        except Exception as e:
            # Log failed request
            if self._enable_logging:
                self._logger.log_response(
                    session.id,
                    request_id,
                    ResponsePayload(
                        content="",
                        stop_reason="error",
                        usage=Usage(),
                        model=model,
                    ),
                    int((time.time() - start_time) * 1000),
                )
            raise
    
    async def stream(
        self,
        content: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message with streaming response."""
        session = self.get_current_session()
        if not session:
            raise ValueError("No active session. Create or load a session first.")
        
        client = await self._get_client()
        model = model or self._model
        max_tokens = max_tokens or self._max_tokens
        is_oauth = is_oauth_token(self._get_auth_token())
        
        # Add user message
        user_message = Message(
            role="user",
            content=content,
            timestamp=int(time.time() * 1000),
        )
        self._session_manager.add_message(session.id, user_message)
        
        # Build system prompt
        sys_prompt = system_prompt or session.system_prompt or ""
        if is_oauth and "Claude Code" not in sys_prompt:
            sys_prompt = "You are Claude Code, Anthropic's official CLI for Claude.\n\n" + sys_prompt
        
        messages = [
            {"role": m.role, "content": m.content}
            for m in session.messages
        ]
        
        start_time = time.time()
        full_content = ""
        
        try:
            yield {"type": "start"}
            
            params: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            
            if sys_prompt:
                params["system"] = [{
                    "type": "text",
                    "text": sys_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                full_content += event.delta.text
                                yield {"type": "text", "text": event.delta.text}
                
                final_message = await stream.get_final_message()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Add assistant message
            self._session_manager.add_message(session.id, Message(
                role="assistant",
                content=full_content,
                timestamp=int(time.time() * 1000),
            ))
            
            result = SendResult(
                content=full_content,
                stop_reason=self._map_stop_reason(final_message.stop_reason),
                usage=Usage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=final_message.usage.output_tokens,
                ),
                duration_ms=duration_ms,
                model=final_message.model,
            )
            
            yield {"type": "done", "result": result}
            
        except Exception as e:
            yield {"type": "error", "error": e}
    
    def _map_stop_reason(self, reason: str | None) -> str:
        """Map Anthropic stop reason to our format."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_use",
        }
        return mapping.get(reason or "", "error")

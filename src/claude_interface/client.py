"""
Claude Interface - Main Client

The primary interface for interacting with Claude via OAuth or API key.
Handles authentication, sessions, logging, and memory.
"""

import asyncio
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
    ImageInput,
    ImageContent,
    TextContent,
    Tool,
    ToolCall,
    ToolParameter,
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
        
        # Lock for thread-safe session operations
        self._session_lock = asyncio.Lock()
        
        self._session_manager = SessionManager(str(self._storage_dir))
        self._logger = Logger(str(self._storage_dir))
        
        # Tool registry
        self._tools: dict[str, Tool] = {}
    
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
    # Tool Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool that Claude can use.
        
        Example:
            ```python
            async def get_weather(location: str) -> str:
                return f"Weather in {location}: Sunny, 72°F"
            
            client.register_tool(Tool(
                name="get_weather",
                description="Get the current weather for a location",
                parameters=[
                    ToolParameter(name="location", type="string", description="City name"),
                ],
                handler=get_weather,
            ))
            ```
        """
        self._tools[tool.name] = tool
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def _get_tools_for_api(self) -> list[dict[str, Any]] | None:
        """Convert registered tools to Anthropic API format."""
        if not self._tools:
            return None
        
        tools = []
        for tool in self._tools.values():
            properties = {}
            required = []
            
            for param in tool.parameters:
                prop = {"type": param.type, "description": param.description}
                if param.enum:
                    prop["enum"] = param.enum
                properties[param.name] = prop
                
                if param.required:
                    required.append(param.name)
            
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        
        return tools
    
    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool and return the result."""
        tool = self._tools.get(tool_call.name)
        if not tool or not tool.handler:
            return f"Error: Tool '{tool_call.name}' not found or has no handler"
        
        try:
            # Check if handler is async
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**tool_call.input)
            else:
                result = tool.handler(**tool_call.input)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {e}"
    
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
    # Helper: Build System Prompt
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_system_prompt(
        self,
        session: Session,
        override: str | None,
        is_oauth: bool,
    ) -> str:
        """Build system prompt with memory context."""
        sys_prompt = override or session.system_prompt or ""
        
        # For OAuth, prepend required Claude Code identity
        if is_oauth and "Claude Code" not in sys_prompt:
            sys_prompt = "You are Claude Code, Anthropic's official CLI for Claude.\n\n" + sys_prompt
        
        # Add memory context
        memory = self.get_memory()
        if memory and memory.count() > 0:
            sys_prompt += "\n\n" + memory.format_as_context()
        
        return sys_prompt
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helper: Build Message Content
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_content(
        self,
        text: str,
        images: list[ImageInput] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Build message content, optionally with images."""
        if not images:
            return text
        
        # Build content blocks
        content: list[dict[str, Any]] = []
        
        # Add text first
        if text:
            content.append({"type": "text", "text": text})
        
        # Add images
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.data,
                },
            })
        
        return content
    
    def _convert_messages_for_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to API format, handling images."""
        result = []
        for m in messages:
            if isinstance(m.content, str):
                result.append({"role": m.role, "content": m.content})
            else:
                # Convert content blocks
                blocks = []
                for block in m.content:
                    if isinstance(block, TextContent):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.media_type,
                                "data": block.data,
                            },
                        })
                result.append({"role": m.role, "content": blocks})
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────────
    
    async def send(
        self,
        content: str,
        images: list[ImageInput] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        auto_execute_tools: bool = True,
    ) -> SendResult:
        """
        Send a message and get a response.
        
        Args:
            content: Text content to send
            images: Optional list of images to include
            model: Override model for this request
            max_tokens: Override max tokens
            temperature: Override temperature
            system_prompt: Override system prompt
            auto_execute_tools: Automatically execute tool calls and continue
        """
        session = self.get_current_session()
        if not session:
            raise ValueError("No active session. Create or load a session first.")
        
        client = await self._get_client()
        model = model or self._model
        max_tokens = max_tokens or self._max_tokens
        is_oauth = is_oauth_token(self._get_auth_token())
        
        # Build message content
        message_content = self._build_content(content, images)
        
        # Add user message to session (with lock for thread safety)
        async with self._session_lock:
            # Store as proper content blocks if we have images
            if images:
                blocks = [TextContent(text=content)] if content else []
                for img in images:
                    blocks.append(ImageContent(
                        media_type=img.media_type,
                        data=img.data,
                    ))
                user_message = Message(
                    role="user",
                    content=blocks,
                    timestamp=int(time.time() * 1000),
                )
            else:
                user_message = Message(
                    role="user",
                    content=content,
                    timestamp=int(time.time() * 1000),
                )
            self._session_manager.add_message(session.id, user_message)
        
        # Build system prompt with memory
        sys_prompt = self._build_system_prompt(session, system_prompt, is_oauth)
        
        # Build messages for API
        messages = self._convert_messages_for_api(session.messages)
        
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
            
            # Add tools if registered
            tools = self._get_tools_for_api()
            if tools:
                params["tools"] = tools
            
            # Make API call
            response = await client.messages.create(**params)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract content and tool calls
            response_text = ""
            tool_calls = []
            
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    ))
            
            # Add assistant message to session
            async with self._session_lock:
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
                tool_calls=tool_calls,
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
            
            # Auto-execute tools if requested
            if auto_execute_tools and tool_calls and result.stop_reason == "tool_use":
                return await self._execute_tools_and_continue(
                    tool_calls, model, max_tokens, temperature, sys_prompt
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
    
    async def _execute_tools_and_continue(
        self,
        tool_calls: list[ToolCall],
        model: str,
        max_tokens: int,
        temperature: float | None,
        system_prompt: str,
    ) -> SendResult:
        """Execute tool calls and continue the conversation."""
        session = self.get_current_session()
        if not session:
            raise ValueError("No active session")
        
        client = await self._get_client()
        
        # Execute each tool
        tool_results = []
        for tc in tool_calls:
            result = await self._execute_tool(tc)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        
        # Add tool results as user message
        messages = self._convert_messages_for_api(session.messages)
        
        # Add the assistant's tool use message
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                for tc in tool_calls
            ],
        })
        
        # Add tool results
        messages.append({
            "role": "user",
            "content": tool_results,
        })
        
        # Continue conversation
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        if system_prompt:
            params["system"] = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        
        if temperature is not None:
            params["temperature"] = temperature
        
        tools = self._get_tools_for_api()
        if tools:
            params["tools"] = tools
        
        start_time = time.time()
        response = await client.messages.create(**params)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract response
        response_text = ""
        new_tool_calls = []
        
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text
            elif block.type == "tool_use":
                new_tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))
        
        # Add assistant message
        async with self._session_lock:
            self._session_manager.add_message(session.id, Message(
                role="assistant",
                content=response_text,
                timestamp=int(time.time() * 1000),
            ))
        
        result = SendResult(
            content=response_text,
            stop_reason=self._map_stop_reason(response.stop_reason),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            duration_ms=duration_ms,
            model=response.model,
            tool_calls=new_tool_calls,
        )
        
        # Recursively handle more tool calls
        if new_tool_calls and result.stop_reason == "tool_use":
            return await self._execute_tools_and_continue(
                new_tool_calls, model, max_tokens, temperature, system_prompt
            )
        
        return result
    
    async def stream(
        self,
        content: str,
        images: list[ImageInput] | None = None,
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
        
        # Build message content
        message_content = self._build_content(content, images)
        
        # Add user message (with lock)
        async with self._session_lock:
            if images:
                blocks = [TextContent(text=content)] if content else []
                for img in images:
                    blocks.append(ImageContent(media_type=img.media_type, data=img.data))
                user_message = Message(role="user", content=blocks, timestamp=int(time.time() * 1000))
            else:
                user_message = Message(role="user", content=content, timestamp=int(time.time() * 1000))
            self._session_manager.add_message(session.id, user_message)
        
        # Build system prompt WITH MEMORY (fix for Copilot issue)
        sys_prompt = self._build_system_prompt(session, system_prompt, is_oauth)
        
        messages = self._convert_messages_for_api(session.messages)
        
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
            
            if temperature is not None:
                params["temperature"] = temperature
            elif self._temperature is not None:
                params["temperature"] = self._temperature
            
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                full_content += event.delta.text
                                yield {"type": "text", "text": event.delta.text}
                
                final_message = await stream.get_final_message()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Add assistant message (with lock)
            async with self._session_lock:
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

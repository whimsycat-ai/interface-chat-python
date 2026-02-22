"""
Claude Interface - Session Management

Handles saving, loading, and managing conversation sessions.
"""

import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Any

from .types import Session, Message, MemoryEntry, SessionSummary, TextContent, ImageContent

logger = logging.getLogger(__name__)

# Maximum sessions to keep in memory cache
MAX_CACHED_SESSIONS = 100

# Required fields for session schema validation
REQUIRED_SESSION_FIELDS = {"id"}
VALID_MESSAGE_ROLES = {"user", "assistant", "system"}


class SessionLoadError(Exception):
    """Raised when a session fails to load."""
    
    def __init__(self, session_id: str, reason: str, original_error: Exception | None = None):
        self.session_id = session_id
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Failed to load session '{session_id}': {reason}")


class SessionValidationError(Exception):
    """Raised when session data fails validation."""
    
    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message)


class PathSecurityError(Exception):
    """Raised when a path fails security validation."""
    pass


class SessionManager:
    """Manages conversation sessions with file persistence."""
    
    def __init__(self, storage_dir: str, allowed_export_dirs: list[str] | None = None):
        self.session_dir = Path(storage_dir) / "sessions"
        self._sessions: dict[str, Session] = {}
        self._access_order: list[str] = []  # LRU tracking
        # Directories where export/import is allowed (defaults to session_dir parent)
        self._allowed_dirs = [Path(storage_dir).resolve()]
        if allowed_export_dirs:
            self._allowed_dirs.extend(Path(d).resolve() for d in allowed_export_dirs)
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.session_dir / f"{session_id}.json"
    
    def _validate_path_security(self, path: str, operation: str) -> Path:
        """
        Validate that a path is safe for file operations.
        
        Prevents path traversal attacks by ensuring the resolved path
        is within allowed directories.
        
        Args:
            path: The path to validate
            operation: Description of operation (for error messages)
            
        Returns:
            Resolved Path object
            
        Raises:
            PathSecurityError: If path is outside allowed directories
        """
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError) as e:
            raise PathSecurityError(f"Invalid path for {operation}: {e}")
        
        # Check if path is within any allowed directory
        for allowed_dir in self._allowed_dirs:
            try:
                resolved.relative_to(allowed_dir)
                return resolved
            except ValueError:
                continue
        
        raise PathSecurityError(
            f"Path '{path}' is outside allowed directories for {operation}. "
            f"Allowed: {[str(d) for d in self._allowed_dirs]}"
        )
    
    def _validate_session_schema(self, data: dict, source: str = "data") -> None:
        """
        Validate imported session data against expected schema.
        
        Args:
            data: The session data dict to validate
            source: Description of data source (for error messages)
            
        Raises:
            SessionValidationError: If data fails validation
        """
        if not isinstance(data, dict):
            raise SessionValidationError(f"Session {source} must be a JSON object")
        
        # Check required fields
        for field in REQUIRED_SESSION_FIELDS:
            if field not in data:
                raise SessionValidationError(
                    f"Missing required field '{field}' in session {source}",
                    field=field
                )
        
        # Validate id is a non-empty string
        if not isinstance(data.get("id"), str) or not data["id"].strip():
            raise SessionValidationError(
                "Session 'id' must be a non-empty string",
                field="id"
            )
        
        # Validate messages array if present
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise SessionValidationError(
                "'messages' must be an array",
                field="messages"
            )
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise SessionValidationError(
                    f"Message at index {i} must be an object",
                    field=f"messages[{i}]"
                )
            if "role" not in msg:
                raise SessionValidationError(
                    f"Message at index {i} missing 'role' field",
                    field=f"messages[{i}].role"
                )
            if msg["role"] not in VALID_MESSAGE_ROLES:
                raise SessionValidationError(
                    f"Message at index {i} has invalid role '{msg['role']}'. "
                    f"Valid roles: {VALID_MESSAGE_ROLES}",
                    field=f"messages[{i}].role"
                )
            if "content" not in msg:
                raise SessionValidationError(
                    f"Message at index {i} missing 'content' field",
                    field=f"messages[{i}].content"
                )
        
        # Validate memory array if present
        memory = data.get("memory", [])
        if not isinstance(memory, list):
            raise SessionValidationError(
                "'memory' must be an array",
                field="memory"
            )
        
        for i, mem in enumerate(memory):
            if not isinstance(mem, dict):
                raise SessionValidationError(
                    f"Memory entry at index {i} must be an object",
                    field=f"memory[{i}]"
                )
            if "id" not in mem or "content" not in mem:
                raise SessionValidationError(
                    f"Memory entry at index {i} missing required 'id' or 'content' field",
                    field=f"memory[{i}]"
                )
        
        # Validate metadata is a dict if present
        if "metadata" in data and not isinstance(data["metadata"], dict):
            raise SessionValidationError(
                "'metadata' must be an object",
                field="metadata"
            )
    
    def _touch_cache(self, session_id: str) -> None:
        """Update LRU order for a session."""
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        self._access_order.append(session_id)
    
    def _evict_if_needed(self) -> None:
        """Evict oldest sessions from cache if over limit."""
        while len(self._sessions) > MAX_CACHED_SESSIONS:
            if self._access_order:
                oldest = self._access_order.pop(0)
                self._sessions.pop(oldest, None)
    
    def clear_cache(self) -> None:
        """Clear the in-memory session cache."""
        self._sessions.clear()
        self._access_order.clear()
    
    def generate_id(self) -> str:
        """Generate a unique session ID."""
        import secrets
        timestamp = hex(int(time.time()))[2:]
        random_part = secrets.token_hex(3)
        return f"session_{timestamp}_{random_part}"
    
    def _serialize_content(self, content: str | list) -> str | list[dict]:
        """Serialize message content to JSON-compatible format."""
        if isinstance(content, str):
            return content
        
        # Handle list of ContentBlocks
        result = []
        for block in content:
            if isinstance(block, TextContent):
                result.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                result.append({
                    "type": "image",
                    "media_type": block.media_type,
                    "data": block.data,
                })
            elif isinstance(block, dict):
                result.append(block)
            else:
                # Fallback for unknown types
                result.append({"type": "unknown", "data": str(block)})
        return result
    
    def _deserialize_content(self, content: str | list) -> str | list:
        """Deserialize message content from JSON."""
        if isinstance(content, str):
            return content
        
        # Handle list of dicts -> ContentBlocks
        result = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    result.append(TextContent(text=block.get("text", "")))
                elif block.get("type") == "image":
                    result.append(ImageContent(
                        media_type=block.get("media_type", "image/png"),
                        data=block.get("data", ""),
                    ))
                else:
                    result.append(block)
            else:
                result.append(block)
        return result
    
    def create(
        self,
        id: str | None = None,
        name: str | None = None,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        now = int(time.time() * 1000)
        session = Session(
            id=id or self.generate_id(),
            name=name,
            system_prompt=system_prompt,
            messages=[],
            metadata=metadata or {},
            memory=[],
            created_at=now,
            updated_at=now,
        )
        
        self._sessions[session.id] = session
        self._touch_cache(session.id)
        self._evict_if_needed()
        self.save(session.id)
        return session
    
    def get(self, session_id: str) -> Session | None:
        """Get a session by ID, loading from disk if necessary."""
        if session_id in self._sessions:
            self._touch_cache(session_id)
            return self._sessions[session_id]
        return self.load(session_id)
    
    def load(self, session_id: str) -> Session | None:
        """
        Load a session from disk.
        
        Args:
            session_id: The session ID to load
            
        Returns:
            The loaded Session, or None if not found
            
        Raises:
            SessionLoadError: If the session file exists but cannot be loaded
        """
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SessionLoadError(
                session_id,
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}",
                original_error=e
            )
        except PermissionError as e:
            raise SessionLoadError(
                session_id,
                f"Permission denied reading session file",
                original_error=e
            )
        except OSError as e:
            raise SessionLoadError(
                session_id,
                f"I/O error reading session file: {e.strerror}",
                original_error=e
            )
        
        try:
            # Validate required fields
            if "id" not in data:
                raise SessionLoadError(
                    session_id,
                    "Session file missing required 'id' field"
                )
            
            # Convert dicts back to dataclass instances
            messages = [
                Message(
                    role=m["role"],
                    content=self._deserialize_content(m["content"]),
                    timestamp=m.get("timestamp"),
                )
                for m in data.get("messages", [])
            ]
            
            memory = [
                MemoryEntry(
                    id=m["id"],
                    content=m["content"],
                    type=m.get("type", "fact"),
                    tags=m.get("tags", []),
                    created_at=m.get("created_at", 0),
                    priority=m.get("priority", 0),
                )
                for m in data.get("memory", [])
            ]
            
            session = Session(
                id=data["id"],
                name=data.get("name"),
                system_prompt=data.get("system_prompt"),
                messages=messages,
                metadata=data.get("metadata", {}),
                memory=memory,
                created_at=data.get("created_at", 0),
                updated_at=data.get("updated_at", 0),
            )
            
            self._sessions[session_id] = session
            self._touch_cache(session_id)
            self._evict_if_needed()
            return session
            
        except KeyError as e:
            raise SessionLoadError(
                session_id,
                f"Missing required field in message or memory: {e}",
                original_error=e
            )
        except TypeError as e:
            raise SessionLoadError(
                session_id,
                f"Invalid data type in session structure: {e}",
                original_error=e
            )
    
    def save(self, session_id: str) -> None:
        """
        Save a session to disk atomically.
        
        Uses write-to-temp-then-rename pattern to prevent corruption
        on disk full or permission errors.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.updated_at = int(time.time() * 1000)
        session_path = self._get_session_path(session_id)
        
        # Convert to dict for JSON serialization
        data = {
            "id": session.id,
            "name": session.name,
            "system_prompt": session.system_prompt,
            "messages": [
                {
                    "role": m.role,
                    "content": self._serialize_content(m.content),
                    "timestamp": m.timestamp,
                }
                for m in session.messages
            ],
            "metadata": session.metadata,
            "memory": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.type,
                    "tags": m.tags,
                    "created_at": m.created_at,
                    "priority": m.priority,
                }
                for m in session.memory
            ],
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
        
        # Atomic write: write to temp file, then rename
        # This prevents corruption if disk is full or write is interrupted
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.session_dir,
                prefix=f".{session_id}_",
                suffix=".tmp"
            )
            with os.fdopen(temp_fd, "w") as f:
                temp_fd = None  # os.fdopen takes ownership
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, session_path)
            temp_path = None  # Successfully moved, don't delete
            
        finally:
            # Clean up temp file if something went wrong
            if temp_fd is not None:
                os.close(temp_fd)
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session."""
        session = self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if message.timestamp is None:
            message.timestamp = int(time.time() * 1000)
        
        session.messages.append(message)
        self.save(session_id)
    
    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Get messages from a session with optional limit."""
        session = self.get(session_id)
        if not session:
            return []
        
        if limit and limit > 0:
            return session.messages[-limit:]
        return session.messages
    
    def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Update session metadata."""
        session = self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.metadata.update(metadata)
        self.save(session_id)
    
    def list(self) -> list[SessionSummary]:
        """List all sessions."""
        summaries = []
        
        for file in self.session_dir.glob("*.json"):
            session_id = file.stem
            session = self.get(session_id)
            if session:
                summaries.append(SessionSummary(
                    id=session.id,
                    name=session.name,
                    message_count=len(session.messages),
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                ))
        
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        self._sessions.pop(session_id, None)
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            return True
        return False
    
    def fork(self, session_id: str, new_name: str | None = None) -> Session | None:
        """Fork a session (create a copy with new ID)."""
        original = self.get(session_id)
        if not original:
            return None
        
        forked = self.create(
            name=new_name or f"{original.name or 'Session'} (fork)",
            system_prompt=original.system_prompt,
            metadata={**original.metadata, "forked_from": session_id},
        )
        
        # Copy messages and memory
        forked.messages = [
            Message(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in original.messages
        ]
        forked.memory = [
            MemoryEntry(
                id=m.id,
                content=m.content,
                type=m.type,
                tags=list(m.tags),
                created_at=m.created_at,
                priority=m.priority,
            )
            for m in original.memory
        ]
        self.save(forked.id)
        
        return forked
    
    def clear(self, session_id: str) -> None:
        """Clear all messages from a session (keep metadata)."""
        session = self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.messages = []
        self.save(session_id)
    
    def export(self, session_id: str, output_path: str) -> None:
        """
        Export a session to a file.
        
        Args:
            session_id: The session ID to export
            output_path: Path to write the exported JSON
            
        Raises:
            ValueError: If session not found
            PathSecurityError: If output_path is outside allowed directories
        """
        # Validate path security before proceeding
        validated_path = self._validate_path_security(output_path, "export")
        
        session = self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Build explicit JSON-serializable representation
        data = {
            "id": session.id,
            "name": session.name,
            "system_prompt": session.system_prompt,
            "metadata": session.metadata,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [
                {
                    "role": m.role,
                    "content": self._serialize_content(m.content),
                    "timestamp": m.timestamp,
                }
                for m in session.messages
            ],
            "memory": [
                {
                    "id": mem.id,
                    "content": mem.content,
                    "type": mem.type,
                    "tags": list(mem.tags),
                    "created_at": mem.created_at,
                    "priority": mem.priority,
                }
                for mem in session.memory
            ],
        }
        
        # Use atomic write for export as well
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=validated_path.parent,
                prefix=".export_",
                suffix=".tmp"
            )
            with os.fdopen(temp_fd, "w") as f:
                temp_fd = None
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(temp_path, validated_path)
            temp_path = None
            
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def import_session(self, input_path: str, new_id: str | None = None) -> Session:
        """
        Import a session from a file.
        
        Args:
            input_path: Path to the JSON file to import
            new_id: Optional new ID for the imported session
            
        Returns:
            The imported Session
            
        Raises:
            PathSecurityError: If input_path is outside allowed directories
            SessionValidationError: If the imported data fails schema validation
            FileNotFoundError: If the input file doesn't exist
        """
        # Validate path security before proceeding
        validated_path = self._validate_path_security(input_path, "import")
        
        if not validated_path.exists():
            raise FileNotFoundError(f"Import file not found: {input_path}")
        
        try:
            with open(validated_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SessionValidationError(
                f"Invalid JSON in import file at line {e.lineno}: {e.msg}"
            )
        
        # Validate the imported data schema
        self._validate_session_schema(data, source=f"file '{input_path}'")
        
        if new_id:
            data["id"] = new_id
        
        session = self.create(
            id=data["id"],
            name=data.get("name"),
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
        )
        
        # Load messages and memory from imported data
        session.messages = [
            Message(
                role=m["role"],
                content=self._deserialize_content(m["content"]),
                timestamp=m.get("timestamp"),
            )
            for m in data.get("messages", [])
        ]
        session.memory = [
            MemoryEntry(
                id=m["id"],
                content=m["content"],
                type=m.get("type", "fact"),
                tags=m.get("tags", []),
                created_at=m.get("created_at", 0),
                priority=m.get("priority", 0),
            )
            for m in data.get("memory", [])
        ]
        
        self.save(session.id)
        return session

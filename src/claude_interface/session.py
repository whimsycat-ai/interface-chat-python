"""
Claude Interface - Session Management

Handles saving, loading, and managing conversation sessions.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any

from .types import Session, Message, MemoryEntry, SessionSummary, TextContent, ImageContent

logger = logging.getLogger(__name__)

# Maximum sessions to keep in memory cache
MAX_CACHED_SESSIONS = 100


class SessionManager:
    """Manages conversation sessions with file persistence."""
    
    def __init__(self, storage_dir: str):
        self.session_dir = Path(storage_dir) / "sessions"
        self._sessions: dict[str, Session] = {}
        self._access_order: list[str] = []  # LRU tracking
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.session_dir / f"{session_id}.json"
    
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
        """Load a session from disk."""
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
            
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
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def save(self, session_id: str) -> None:
        """Save a session to disk."""
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
        
        with open(session_path, "w") as f:
            json.dump(data, f, indent=2)
    
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
        """Export a session to a file."""
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
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def import_session(self, input_path: str, new_id: str | None = None) -> Session:
        """Import a session from a file."""
        with open(input_path, "r") as f:
            data = json.load(f)
        
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

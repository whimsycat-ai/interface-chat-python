"""
Claude Interface - Session Management

Handles saving, loading, and managing conversation sessions.
"""

import json
import os
import time
from pathlib import Path
from dataclasses import asdict, field
from typing import Any

from .types import Session, Message, MemoryEntry, SessionSummary


class SessionManager:
    """Manages conversation sessions with file persistence."""
    
    def __init__(self, storage_dir: str):
        self.session_dir = Path(storage_dir) / "sessions"
        self._sessions: dict[str, Session] = {}
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.session_dir / f"{session_id}.json"
    
    def generate_id(self) -> str:
        """Generate a unique session ID."""
        import secrets
        timestamp = hex(int(time.time()))[2:]
        random_part = secrets.token_hex(3)
        return f"session_{timestamp}_{random_part}"
    
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
        self.save(session.id)
        return session
    
    def get(self, session_id: str) -> Session | None:
        """Get a session by ID, loading from disk if necessary."""
        if session_id in self._sessions:
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
                    content=m["content"],
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
            return session
        except Exception as e:
            print(f"Failed to load session {session_id}: {e}")
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
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
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
        
        with open(output_path, "w") as f:
            json.dump(asdict(session), f, indent=2)
    
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
            Message(role=m["role"], content=m["content"], timestamp=m.get("timestamp"))
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

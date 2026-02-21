"""
Claude Interface - Memory Management

Manages persistent memory entries associated with sessions.
Supports tagging, priority, and simple retrieval.
"""

import time
import secrets
from typing import Callable

from .types import Session, MemoryEntry, MemoryType


class MemoryManager:
    """Manages memory entries for a session."""
    
    def __init__(self, session: Session, on_update: Callable[[], None]):
        self._session = session
        self._on_update = on_update
    
    def _generate_id(self) -> str:
        """Generate a unique memory ID."""
        timestamp = hex(int(time.time()))[2:]
        random_part = secrets.token_hex(3)
        return f"mem_{timestamp}_{random_part}"
    
    def add(
        self,
        content: str,
        type: MemoryType = "fact",
        tags: list[str] | None = None,
        priority: int = 0,
    ) -> MemoryEntry:
        """Add a new memory entry."""
        entry = MemoryEntry(
            id=self._generate_id(),
            content=content,
            type=type,
            tags=tags or [],
            created_at=int(time.time() * 1000),
            priority=priority,
        )
        
        self._session.memory.append(entry)
        self._on_update()
        return entry
    
    def get(self, id: str) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        for entry in self._session.memory:
            if entry.id == id:
                return entry
        return None
    
    def get_all(self) -> list[MemoryEntry]:
        """Get all memory entries."""
        return list(self._session.memory)
    
    def get_by_type(self, type: MemoryType) -> list[MemoryEntry]:
        """Get memory entries by type."""
        return [m for m in self._session.memory if m.type == type]
    
    def get_by_tag(self, tag: str) -> list[MemoryEntry]:
        """Get memory entries by tag."""
        return [m for m in self._session.memory if tag in m.tags]
    
    def search(self, query: str) -> list[MemoryEntry]:
        """Search memory entries by content."""
        query_lower = query.lower()
        return [
            m for m in self._session.memory
            if query_lower in m.content.lower()
            or any(query_lower in t.lower() for t in m.tags)
        ]
    
    def get_top_priority(self, limit: int = 10) -> list[MemoryEntry]:
        """Get top memories by priority."""
        sorted_memories = sorted(
            self._session.memory,
            key=lambda m: m.priority,
            reverse=True,
        )
        return sorted_memories[:limit]
    
    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get recent memories."""
        sorted_memories = sorted(
            self._session.memory,
            key=lambda m: m.created_at,
            reverse=True,
        )
        return sorted_memories[:limit]
    
    def update(
        self,
        id: str,
        content: str | None = None,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        priority: int | None = None,
    ) -> bool:
        """Update a memory entry."""
        for i, entry in enumerate(self._session.memory):
            if entry.id == id:
                if content is not None:
                    entry.content = content
                if type is not None:
                    entry.type = type
                if tags is not None:
                    entry.tags = tags
                if priority is not None:
                    entry.priority = priority
                self._on_update()
                return True
        return False
    
    def remove(self, id: str) -> bool:
        """Remove a memory entry."""
        for i, entry in enumerate(self._session.memory):
            if entry.id == id:
                self._session.memory.pop(i)
                self._on_update()
                return True
        return False
    
    def clear(self) -> None:
        """Clear all memories."""
        self._session.memory = []
        self._on_update()
    
    def count(self) -> int:
        """Get count of memories."""
        return len(self._session.memory)
    
    def format_as_context(
        self,
        max_entries: int = 20,
        types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        min_priority: int | None = None,
    ) -> str:
        """
        Format memories as context for the model.
        Returns a string that can be injected into the system prompt.
        """
        entries = list(self._session.memory)
        
        # Filter by type
        if types:
            entries = [m for m in entries if m.type in types]
        
        # Filter by tags
        if tags:
            entries = [m for m in entries if any(t in m.tags for t in tags)]
        
        # Filter by priority
        if min_priority is not None:
            entries = [m for m in entries if m.priority >= min_priority]
        
        # Sort by priority, then by recency
        entries.sort(key=lambda m: (m.priority, m.created_at), reverse=True)
        
        # Limit
        entries = entries[:max_entries]
        
        if not entries:
            return ""
        
        # Format as markdown
        lines = ["## Session Memory", ""]
        for entry in entries:
            tag_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            lines.append(f"- **{entry.type}**{tag_str}: {entry.content}")
        
        return "\n".join(lines)

"""
Claude Interface - Request/Response Logger

Logs all API interactions to disk for debugging and analysis.
"""

import json
import time
import secrets
from pathlib import Path
from dataclasses import dataclass

from .types import LogEntry, RequestPayload, ResponsePayload, Message, Usage, TextContent, ImageContent


@dataclass
class LogStats:
    """Statistics for logged requests."""
    request_count: int
    response_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_duration_ms: int
    average_duration_ms: float


class Logger:
    """Logs requests and responses to JSONL files."""
    
    # Maximum number of sessions to keep in memory cache
    MAX_CACHED_SESSIONS = 50
    
    def __init__(self, storage_dir: str):
        self.log_dir = Path(storage_dir) / "logs"
        self._logs: dict[str, list[LogEntry]] = {}
        self._access_order: list[str] = []  # LRU tracking
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_id(self) -> str:
        """Generate a unique log ID."""
        timestamp = hex(int(time.time()))[2:]
        random_part = secrets.token_hex(3)
        return f"log_{timestamp}_{random_part}"
    
    def _get_log_path(self, session_id: str) -> Path:
        """Get file path for session logs."""
        return self.log_dir / f"{session_id}.jsonl"
    
    def log_request(self, session_id: str, payload: RequestPayload) -> str:
        """Log a request being sent."""
        entry = LogEntry(
            id=self._generate_id(),
            session_id=session_id,
            timestamp=int(time.time() * 1000),
            direction="request",
            payload=payload,
        )
        
        self._append_log(session_id, entry)
        return entry.id
    
    def log_response(
        self,
        session_id: str,
        request_id: str,
        payload: ResponsePayload,
        duration_ms: int,
    ) -> None:
        """Log a response received."""
        entry = LogEntry(
            id=self._generate_id(),
            session_id=session_id,
            timestamp=int(time.time() * 1000),
            direction="response",
            payload=payload,
            duration_ms=duration_ms,
        )
        
        self._append_log(session_id, entry)
    
    def _append_log(self, session_id: str, entry: LogEntry) -> None:
        """Append a log entry to cache and file."""
        # Append to cache with LRU eviction
        if session_id not in self._logs:
            self._logs[session_id] = []
            self._access_order.append(session_id)
            self._evict_if_needed()
        else:
            # Move to end (most recently used)
            if session_id in self._access_order:
                self._access_order.remove(session_id)
            self._access_order.append(session_id)
        
        self._logs[session_id].append(entry)
        
        # Append to file (JSONL format)
        log_path = self._get_log_path(session_id)
        
        # Serialize entry to JSON
        entry_dict = {
            "id": entry.id,
            "session_id": entry.session_id,
            "timestamp": entry.timestamp,
            "direction": entry.direction,
            "duration_ms": entry.duration_ms,
        }
        
        if isinstance(entry.payload, RequestPayload):
            entry_dict["payload"] = {
                "model": entry.payload.model,
                "messages": [
                    {"role": m.role, "content": self._serialize_content(m.content)}
                    for m in entry.payload.messages
                ],
                "system_prompt": entry.payload.system_prompt,
                "max_tokens": entry.payload.max_tokens,
                "temperature": entry.payload.temperature,
            }
        else:
            entry_dict["payload"] = {
                "content": entry.payload.content,
                "stop_reason": entry.payload.stop_reason,
                "usage": {
                    "input_tokens": entry.payload.usage.input_tokens,
                    "output_tokens": entry.payload.usage.output_tokens,
                    "cache_read_tokens": entry.payload.usage.cache_read_tokens,
                    "cache_write_tokens": entry.payload.usage.cache_write_tokens,
                },
                "model": entry.payload.model,
            }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(entry_dict) + "\n")
    
    def get_logs(self, session_id: str) -> list[LogEntry]:
        """Get all logs for a session."""
        # Check cache first
        if session_id in self._logs:
            return self._logs[session_id]
        
        # Load from disk
        log_path = self._get_log_path(session_id)
        if not log_path.exists():
            return []
        
        entries = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                payload_data = data["payload"]
                
                if data["direction"] == "request":
                    payload = RequestPayload(
                        model=payload_data["model"],
                        messages=[
                            Message(role=m["role"], content=m["content"])
                            for m in payload_data.get("messages", [])
                        ],
                        system_prompt=payload_data.get("system_prompt"),
                        max_tokens=payload_data.get("max_tokens"),
                        temperature=payload_data.get("temperature"),
                    )
                else:
                    usage_data = payload_data.get("usage", {})
                    payload = ResponsePayload(
                        content=payload_data["content"],
                        stop_reason=payload_data["stop_reason"],
                        usage=Usage(
                            input_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0),
                            cache_read_tokens=usage_data.get("cache_read_tokens"),
                            cache_write_tokens=usage_data.get("cache_write_tokens"),
                        ),
                        model=payload_data["model"],
                    )
                
                entries.append(LogEntry(
                    id=data["id"],
                    session_id=data["session_id"],
                    timestamp=data["timestamp"],
                    direction=data["direction"],
                    payload=payload,
                    duration_ms=data.get("duration_ms"),
                ))
        
        self._logs[session_id] = entries
        return entries
    
    def get_stats(self, session_id: str) -> LogStats:
        """Get summary statistics for a session."""
        logs = self.get_logs(session_id)
        requests = [l for l in logs if l.direction == "request"]
        responses = [l for l in logs if l.direction == "response"]
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_duration_ms = 0
        
        for log in responses:
            if isinstance(log.payload, ResponsePayload):
                total_input_tokens += log.payload.usage.input_tokens
                total_output_tokens += log.payload.usage.output_tokens
            total_duration_ms += log.duration_ms or 0
        
        return LogStats(
            request_count=len(requests),
            response_count=len(responses),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            total_duration_ms=total_duration_ms,
            average_duration_ms=total_duration_ms / len(responses) if responses else 0,
        )
    
    def clear_logs(self, session_id: str) -> None:
        """Clear logs for a session."""
        self._logs.pop(session_id, None)
        log_path = self._get_log_path(session_id)
        if log_path.exists():
            log_path.unlink()
    
    def _serialize_content(self, content: str | list) -> str | list[dict]:
        """Serialize message content to JSON-compatible format."""
        if isinstance(content, str):
            return content
        
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
                result.append({"type": "unknown", "data": str(block)})
        return result
    
    def export_logs(self, session_id: str, output_path: str) -> None:
        """Export logs to a single JSON file."""
        logs = self.get_logs(session_id)
        
        # Properly serialize log entries
        serialized = []
        for log in logs:
            entry = {
                "id": log.id,
                "session_id": log.session_id,
                "timestamp": log.timestamp,
                "direction": log.direction,
                "duration_ms": log.duration_ms,
            }
            
            if isinstance(log.payload, RequestPayload):
                entry["payload"] = {
                    "type": "request",
                    "model": log.payload.model,
                    "messages": [
                        {"role": m.role, "content": self._serialize_content(m.content)}
                        for m in log.payload.messages
                    ],
                    "system_prompt": log.payload.system_prompt,
                    "max_tokens": log.payload.max_tokens,
                    "temperature": log.payload.temperature,
                }
            else:
                entry["payload"] = {
                    "type": "response",
                    "content": log.payload.content,
                    "stop_reason": log.payload.stop_reason,
                    "usage": {
                        "input_tokens": log.payload.usage.input_tokens,
                        "output_tokens": log.payload.usage.output_tokens,
                        "cache_read_tokens": log.payload.usage.cache_read_tokens,
                        "cache_write_tokens": log.payload.usage.cache_write_tokens,
                    },
                    "model": log.payload.model,
                }
            serialized.append(entry)
        
        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)
    
    def _evict_if_needed(self) -> None:
        """Evict oldest sessions from cache if over limit."""
        while len(self._logs) > self.MAX_CACHED_SESSIONS:
            if self._access_order:
                oldest = self._access_order.pop(0)
                self._logs.pop(oldest, None)
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache (logs remain on disk)."""
        self._logs.clear()
        self._access_order.clear()

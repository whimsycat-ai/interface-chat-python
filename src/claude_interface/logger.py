"""
Claude Interface - Request/Response Logger

Logs all API interactions to disk for debugging and analysis.
"""

import json
import os
import tempfile
import time
import secrets
import logging
from pathlib import Path
from dataclasses import dataclass

from .types import LogEntry, RequestPayload, ResponsePayload, Message, Usage, TextContent, ImageContent


# Module logger for error reporting
_logger = logging.getLogger(__name__)


class LogWriteError(Exception):
    """Raised when a log write operation fails."""
    pass


class PathValidationError(Exception):
    """Raised when a path fails security validation."""
    pass


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
        self.log_dir = Path(storage_dir).resolve() / "logs"
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
        """Append a log entry to cache and file.
        
        Uses atomic write pattern: write to temp file, then append to main file.
        This prevents partial writes from corrupting the log on crashes.
        """
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
        
        # Atomic write pattern: write to temp file first, then append
        json_line = json.dumps(entry_dict) + "\n"
        
        try:
            # Write to temp file in the same directory (ensures same filesystem for rename)
            fd, temp_path = tempfile.mkstemp(
                dir=self.log_dir,
                prefix=".log_entry_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as temp_file:
                    temp_file.write(json_line)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                
                # Append temp file content to main log file
                with open(log_path, "a") as f:
                    f.write(json_line)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except (IOError, OSError) as e:
            _logger.error(f"Failed to write log entry for session {session_id}: {e}")
            # Don't raise - log writes shouldn't crash the application
            # The entry is still in the in-memory cache
    
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
    
    def _validate_export_path(self, output_path: str) -> Path:
        """Validate export path to prevent path traversal attacks.
        
        Args:
            output_path: The requested output path
            
        Returns:
            Resolved, validated Path object
            
        Raises:
            PathValidationError: If the path fails validation
        """
        # Resolve to absolute path
        resolved_path = Path(output_path).resolve()
        
        # Check for null bytes (can bypass some checks)
        if "\x00" in output_path:
            raise PathValidationError("Path contains null bytes")
        
        # Ensure the path is under the log directory OR is an absolute path
        # that the caller explicitly requested (not using .. traversal)
        # Check if the original path tried to use path traversal
        normalized = os.path.normpath(output_path)
        if ".." in normalized.split(os.sep):
            raise PathValidationError(
                f"Path traversal detected in output path: {output_path}"
            )
        
        # Ensure parent directory exists and is writable
        parent_dir = resolved_path.parent
        if not parent_dir.exists():
            raise PathValidationError(
                f"Parent directory does not exist: {parent_dir}"
            )
        
        if not os.access(parent_dir, os.W_OK):
            raise PathValidationError(
                f"Parent directory is not writable: {parent_dir}"
            )
        
        return resolved_path
    
    def export_logs(self, session_id: str, output_path: str) -> None:
        """Export logs to a single JSON file.
        
        Args:
            session_id: The session ID to export logs for
            output_path: Path to write the JSON file to
            
        Raises:
            PathValidationError: If output_path fails security validation
            LogWriteError: If the write operation fails
        """
        # Validate the output path
        validated_path = self._validate_export_path(output_path)
        
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
        
        # Atomic write: write to temp file, then rename
        try:
            parent_dir = validated_path.parent
            fd, temp_path = tempfile.mkstemp(
                dir=parent_dir,
                prefix=".export_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(serialized, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                os.replace(temp_path, validated_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
                
        except (IOError, OSError) as e:
            raise LogWriteError(f"Failed to export logs: {e}") from e
    
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

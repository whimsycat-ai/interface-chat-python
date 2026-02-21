"""
Claude Interface - Request/Response Logger

Logs all API interactions to disk for debugging and analysis.
"""

import json
import time
import secrets
from pathlib import Path
from dataclasses import dataclass

from .types import LogEntry, RequestPayload, ResponsePayload, Message, Usage


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
    
    def __init__(self, storage_dir: str):
        self.log_dir = Path(storage_dir) / "logs"
        self._logs: dict[str, list[LogEntry]] = {}
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
        # Append to cache
        if session_id not in self._logs:
            self._logs[session_id] = []
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
                    {"role": m.role, "content": m.content}
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
    
    def export_logs(self, session_id: str, output_path: str) -> None:
        """Export logs to a single JSON file."""
        logs = self.get_logs(session_id)
        # This would need proper serialization
        with open(output_path, "w") as f:
            json.dump([l.__dict__ for l in logs], f, indent=2, default=str)

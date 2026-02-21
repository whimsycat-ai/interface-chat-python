"""Tests for logger module."""

import tempfile
import pytest
from pathlib import Path
from claude_interface import Logger, Message, Usage
from claude_interface.types import RequestPayload, ResponsePayload


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def logger(temp_dir):
    return Logger(temp_dir)


class TestLogRequest:
    def test_logs_request_returns_id(self, logger):
        request_id = logger.log_request(
            "session-1",
            RequestPayload(
                model="claude-sonnet-4-20250514",
                messages=[Message(role="user", content="Hello")],
                max_tokens=1024,
            ),
        )
        assert request_id.startswith("log_")

    def test_persists_log_to_file(self, logger, temp_dir):
        logger.log_request(
            "session-1",
            RequestPayload(model="test", messages=[]),
        )
        log_path = Path(temp_dir) / "logs" / "session-1.jsonl"
        assert log_path.exists()


class TestLogResponse:
    def test_logs_response_with_duration(self, logger):
        request_id = logger.log_request(
            "session-1",
            RequestPayload(model="test", messages=[]),
        )
        
        logger.log_response(
            "session-1",
            request_id,
            ResponsePayload(
                content="Hello!",
                stop_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5),
                model="test",
            ),
            150,
        )
        
        logs = logger.get_logs("session-1")
        assert len(logs) == 2
        assert logs[1].duration_ms == 150


class TestGetLogs:
    def test_returns_logs_for_session(self, logger):
        logger.log_request("session-1", RequestPayload(model="test", messages=[]))
        logger.log_request("session-1", RequestPayload(model="test", messages=[]))
        logger.log_request("session-2", RequestPayload(model="test", messages=[]))

        assert len(logger.get_logs("session-1")) == 2
        assert len(logger.get_logs("session-2")) == 1

    def test_loads_from_disk_on_fresh_instance(self, temp_dir):
        logger1 = Logger(temp_dir)
        logger1.log_request("session-1", RequestPayload(model="test", messages=[]))

        logger2 = Logger(temp_dir)
        assert len(logger2.get_logs("session-1")) == 1

    def test_returns_empty_for_nonexistent(self, logger):
        assert logger.get_logs("non-existent") == []


class TestGetStats:
    def test_calculates_statistics(self, logger):
        request_id = logger.log_request(
            "session-1",
            RequestPayload(model="test", messages=[]),
        )
        
        logger.log_response(
            "session-1",
            request_id,
            ResponsePayload(
                content="Response",
                stop_reason="stop",
                usage=Usage(input_tokens=100, output_tokens=50),
                model="test",
            ),
            200,
        )
        
        stats = logger.get_stats("session-1")
        assert stats.request_count == 1
        assert stats.response_count == 1
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_tokens == 150
        assert stats.average_duration_ms == 200


class TestClearLogs:
    def test_clears_logs_for_session(self, logger, temp_dir):
        logger.log_request("session-1", RequestPayload(model="test", messages=[]))
        logger.clear_logs("session-1")
        
        assert logger.get_logs("session-1") == []
        log_path = Path(temp_dir) / "logs" / "session-1.jsonl"
        assert not log_path.exists()

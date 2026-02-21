"""Tests for session module."""

import tempfile
import pytest
from pathlib import Path
from claude_interface import SessionManager, Message


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def manager(temp_dir):
    return SessionManager(temp_dir)


class TestCreate:
    def test_creates_session_with_generated_id(self, manager):
        session = manager.create()
        assert session.id.startswith("session_")
        assert session.messages == []
        assert session.memory == []

    def test_creates_session_with_custom_id(self, manager):
        session = manager.create(id="custom-id")
        assert session.id == "custom-id"

    def test_creates_session_with_name_and_prompt(self, manager):
        session = manager.create(
            name="Test Session",
            system_prompt="You are helpful.",
        )
        assert session.name == "Test Session"
        assert session.system_prompt == "You are helpful."

    def test_persists_session_to_disk(self, manager, temp_dir):
        session = manager.create(id="persist-test")
        session_path = Path(temp_dir) / "sessions" / "persist-test.json"
        assert session_path.exists()


class TestGetLoad:
    def test_gets_session_from_cache(self, manager):
        created = manager.create(id="cache-test")
        retrieved = manager.get("cache-test")
        assert retrieved == created

    def test_loads_session_from_disk(self, temp_dir):
        manager1 = SessionManager(temp_dir)
        created = manager1.create(id="disk-test", name="Disk Test")
        
        # Create new manager to clear cache
        manager2 = SessionManager(temp_dir)
        loaded = manager2.load("disk-test")
        
        assert loaded is not None
        assert loaded.id == "disk-test"
        assert loaded.name == "Disk Test"

    def test_returns_none_for_nonexistent(self, manager):
        assert manager.get("non-existent") is None


class TestAddMessage:
    def test_adds_message_to_session(self, manager):
        session = manager.create(id="msg-test")
        manager.add_message("msg-test", Message(role="user", content="Hello"))
        
        updated = manager.get("msg-test")
        assert updated is not None
        assert len(updated.messages) == 1
        assert updated.messages[0].content == "Hello"
        assert updated.messages[0].timestamp is not None

    def test_raises_for_nonexistent_session(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.add_message("non-existent", Message(role="user", content="Hello"))


class TestGetMessages:
    def test_returns_all_messages(self, manager):
        manager.create(id="msgs-test")
        manager.add_message("msgs-test", Message(role="user", content="One"))
        manager.add_message("msgs-test", Message(role="assistant", content="Two"))
        manager.add_message("msgs-test", Message(role="user", content="Three"))

        messages = manager.get_messages("msgs-test")
        assert len(messages) == 3

    def test_returns_limited_messages(self, manager):
        manager.create(id="limit-test")
        manager.add_message("limit-test", Message(role="user", content="One"))
        manager.add_message("limit-test", Message(role="assistant", content="Two"))
        manager.add_message("limit-test", Message(role="user", content="Three"))

        messages = manager.get_messages("limit-test", limit=2)
        assert len(messages) == 2
        assert messages[0].content == "Two"
        assert messages[1].content == "Three"


class TestList:
    def test_lists_sessions_sorted_by_updated(self, manager):
        manager.create(id="list-1", name="First")
        import time; time.sleep(0.01)
        manager.create(id="list-2", name="Second")

        sessions = manager.list()
        assert len(sessions) == 2
        assert sessions[0].name == "Second"  # Most recent first


class TestDelete:
    def test_deletes_session(self, manager):
        manager.create(id="delete-test")
        assert manager.delete("delete-test") is True
        assert manager.get("delete-test") is None

    def test_returns_false_for_nonexistent(self, manager):
        assert manager.delete("non-existent") is False


class TestFork:
    def test_forks_session_with_new_id(self, manager):
        original = manager.create(id="original", name="Original")
        manager.add_message("original", Message(role="user", content="Test"))

        forked = manager.fork("original", "Forked")
        assert forked is not None
        assert forked.id != "original"
        assert forked.name == "Forked"
        assert len(forked.messages) == 1
        assert forked.metadata.get("forked_from") == "original"

    def test_returns_none_for_nonexistent(self, manager):
        assert manager.fork("non-existent") is None


class TestClear:
    def test_clears_messages_keeps_metadata(self, manager):
        manager.create(id="clear-test", name="Clear Test")
        manager.add_message("clear-test", Message(role="user", content="Test"))
        manager.clear("clear-test")

        cleared = manager.get("clear-test")
        assert cleared is not None
        assert len(cleared.messages) == 0
        assert cleared.name == "Clear Test"

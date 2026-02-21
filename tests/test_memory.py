"""Tests for memory module."""

import pytest
from claude_interface import MemoryManager, Session


@pytest.fixture
def session():
    return Session(
        id="test-session",
        messages=[],
        metadata={},
        memory=[],
        created_at=0,
        updated_at=0,
    )


@pytest.fixture
def memory(session):
    return MemoryManager(session, lambda: None)


class TestAdd:
    def test_adds_memory_with_defaults(self, memory):
        entry = memory.add(content="Test memory")
        
        assert entry.id.startswith("mem_")
        assert entry.content == "Test memory"
        assert entry.type == "fact"
        assert entry.tags == []
        assert entry.priority == 0

    def test_adds_memory_with_custom_options(self, memory):
        entry = memory.add(
            content="User likes TypeScript",
            type="preference",
            tags=["language", "typescript"],
            priority=5,
        )
        
        assert entry.type == "preference"
        assert entry.tags == ["language", "typescript"]
        assert entry.priority == 5


class TestGet:
    def test_gets_memory_by_id(self, memory):
        entry = memory.add(content="Test")
        assert memory.get(entry.id) == entry

    def test_returns_none_for_nonexistent(self, memory):
        assert memory.get("non-existent") is None


class TestGetByType:
    def test_filters_by_type(self, memory):
        memory.add(content="Fact 1", type="fact")
        memory.add(content="Pref 1", type="preference")
        memory.add(content="Fact 2", type="fact")

        facts = memory.get_by_type("fact")
        assert len(facts) == 2
        assert all(m.type == "fact" for m in facts)


class TestGetByTag:
    def test_filters_by_tag(self, memory):
        memory.add(content="A", tags=["a", "b"])
        memory.add(content="B", tags=["b", "c"])
        memory.add(content="C", tags=["c"])

        tagged = memory.get_by_tag("b")
        assert len(tagged) == 2


class TestSearch:
    def test_searches_content(self, memory):
        memory.add(content="TypeScript is great")
        memory.add(content="JavaScript is good")
        memory.add(content="Python is nice")

        results = memory.search("script")
        assert len(results) == 2

    def test_searches_tags(self, memory):
        memory.add(content="A", tags=["typescript"])
        memory.add(content="B", tags=["javascript"])

        results = memory.search("typescript")
        assert len(results) == 1


class TestGetTopPriority:
    def test_returns_sorted_by_priority(self, memory):
        memory.add(content="Low", priority=1)
        memory.add(content="High", priority=10)
        memory.add(content="Medium", priority=5)

        top = memory.get_top_priority(2)
        assert len(top) == 2
        assert top[0].content == "High"
        assert top[1].content == "Medium"


class TestUpdate:
    def test_updates_memory(self, memory):
        entry = memory.add(content="Original", priority=0)
        result = memory.update(entry.id, content="Updated", priority=5)
        
        assert result is True
        updated = memory.get(entry.id)
        assert updated is not None
        assert updated.content == "Updated"
        assert updated.priority == 5

    def test_returns_false_for_nonexistent(self, memory):
        assert memory.update("non-existent", content="Test") is False


class TestRemove:
    def test_removes_memory(self, memory):
        entry = memory.add(content="Test")
        assert memory.remove(entry.id) is True
        assert memory.get(entry.id) is None

    def test_returns_false_for_nonexistent(self, memory):
        assert memory.remove("non-existent") is False


class TestFormatAsContext:
    def test_formats_as_markdown(self, memory):
        memory.add(content="User likes TypeScript", type="preference", tags=["lang"])
        memory.add(content="Project uses React", type="context")

        context = memory.format_as_context()
        assert "## Session Memory" in context
        assert "**preference** [lang]: User likes TypeScript" in context
        assert "**context**: Project uses React" in context

    def test_returns_empty_for_no_memories(self, memory):
        assert memory.format_as_context() == ""

    def test_filters_by_options(self, memory):
        memory.add(content="A", type="fact", priority=1)
        memory.add(content="B", type="preference", priority=10)

        context = memory.format_as_context(types=["preference"])
        assert "B" in context
        assert "A" not in context

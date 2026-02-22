"""
Comprehensive tests for the ClaudeClient class.
"""

import tempfile
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from claude_interface import (
    ClaudeClient,
    AuthConfig,
    OAuthCredentials,
    Tool,
    ToolParameter,
    SpinOutOptions,
    ImageInput,
    Message,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def api_key_auth():
    return AuthConfig(api_key="sk-ant-api-test-key")


@pytest.fixture
def oauth_auth():
    return AuthConfig(oauth=OAuthCredentials(
        access_token="sk-ant-oat-test-token",
        refresh_token="refresh-token",
        expires_at=9999999999999,  # Far future
    ))


@pytest.fixture
def mock_response():
    """Create a mock Anthropic response."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Hello! I'm Claude.")]
    response.content[0].text = "Hello! I'm Claude."
    response.stop_reason = "end_turn"
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return response


@pytest.fixture
def mock_anthropic(mock_response):
    """Create a mock Anthropic client."""
    with patch("claude_interface.client.anthropic.AsyncAnthropic") as mock:
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=mock_response)
        mock.return_value = client
        yield mock


class TestSessionManagement:
    """Tests for session management methods."""
    
    def test_create_session(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(name="Test Session")
        
        assert session.id.startswith("session_")
        assert session.name == "Test Session"
        assert client.get_current_session() == session
    
    def test_create_session_with_system_prompt(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(
            name="Code Review",
            system_prompt="You are a code reviewer.",
        )
        
        assert session.system_prompt == "You are a code reviewer."
    
    def test_load_session(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        created = client.create_session(name="Original")
        session_id = created.id
        
        # Create new client to clear memory cache
        client2 = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        loaded = client2.load_session(session_id)
        
        assert loaded is not None
        assert loaded.name == "Original"
        assert client2.get_current_session() == loaded
    
    def test_load_nonexistent_session(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        loaded = client.load_session("nonexistent")
        assert loaded is None
    
    def test_list_sessions(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Session 1")
        client.create_session(name="Session 2")
        
        sessions = client.list_sessions()
        assert len(sessions) == 2
        names = [s.name for s in sessions]
        assert "Session 1" in names
        assert "Session 2" in names
    
    def test_delete_session(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(name="To Delete")
        session_id = session.id
        
        result = client.delete_session(session_id)
        assert result is True
        assert client.load_session(session_id) is None
    
    def test_get_current_session_none(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        assert client.get_current_session() is None


class TestSendMethod:
    """Tests for the send() method."""
    
    @pytest.mark.asyncio
    async def test_send_basic(self, temp_dir, api_key_auth, mock_anthropic, mock_response):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        result = await client.send("Hello!")
        
        assert result.content == "Hello! I'm Claude."
        assert result.stop_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
    
    @pytest.mark.asyncio
    async def test_send_no_session_raises(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        
        with pytest.raises(ValueError, match="No active session"):
            await client.send("Hello!")
    
    @pytest.mark.asyncio
    async def test_send_adds_messages_to_session(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(name="Test")
        
        await client.send("Hello!")
        
        messages = client.sessions.get_messages(session.id)
        assert len(messages) == 2  # User + Assistant
        assert messages[0].role == "user"
        assert messages[0].content == "Hello!"
        assert messages[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_send_with_temperature(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        await client.send("Hello!", temperature=0.5)
        
        # Verify temperature was passed to API
        call_kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5
    
    @pytest.mark.asyncio
    async def test_send_with_system_prompt_override(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test", system_prompt="Original prompt")
        
        await client.send("Hello!", system_prompt="Override prompt")
        
        call_kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
        system = call_kwargs.get("system", [])
        assert any("Override prompt" in s.get("text", "") for s in system)


class TestToolManagement:
    """Tests for tool registration and management."""
    
    def test_register_tool(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        
        tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            parameters=[
                ToolParameter(name="location", type="string", description="City name"),
            ],
        )
        
        client.register_tool(tool)
        tools = client.list_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
    
    def test_unregister_tool(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        
        tool = Tool(name="test_tool", description="Test")
        client.register_tool(tool)
        
        result = client.unregister_tool("test_tool")
        assert result is True
        assert len(client.list_tools()) == 0
    
    def test_unregister_nonexistent_tool(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        result = client.unregister_tool("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_tools_sent_to_api(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        tool = Tool(
            name="calculator",
            description="Do math",
            parameters=[
                ToolParameter(name="expression", type="string", description="Math expression"),
            ],
        )
        client.register_tool(tool)
        
        await client.send("Calculate 2+2")
        
        call_kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "calculator"


class TestMemoryIntegration:
    """Tests for memory management integration."""
    
    def test_get_memory_returns_manager(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        memory = client.get_memory()
        assert memory is not None
    
    def test_get_memory_no_session(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        memory = client.get_memory()
        assert memory is None
    
    def test_memory_persists(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(name="Test")
        
        memory = client.get_memory()
        memory.add(content="User likes Python", type="preference")
        
        # Reload
        client2 = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client2.load_session(session.id)
        memory2 = client2.get_memory()
        
        assert memory2.count() == 1
        assert memory2.get_all()[0].content == "User likes Python"


class TestSpinOut:
    """Tests for the spin_out() method."""
    
    @pytest.mark.asyncio
    async def test_spin_out_basic(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        original = client.create_session(name="Original")
        client.sessions.add_message(original.id, Message(role="user", content="Hello"))
        client.sessions.add_message(original.id, Message(role="assistant", content="Hi!"))
        
        new_session = await client.spin_out(SpinOutOptions(
            topic="Greeting",
            include_last_n=2,
        ))
        
        assert new_session.id != original.id
        assert "Greeting" in new_session.name
        assert len(new_session.messages) == 2
        assert new_session.metadata.get("spun_out_from") == original.id
    
    @pytest.mark.asyncio
    async def test_spin_out_no_session_raises(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        
        with pytest.raises(ValueError, match="No active session"):
            await client.spin_out(SpinOutOptions(topic="Test"))
    
    @pytest.mark.asyncio
    async def test_spin_out_switches_session(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        original = client.create_session(name="Original")
        
        new_session = await client.spin_out(SpinOutOptions(topic="New"))
        
        assert client.get_current_session().id == new_session.id
    
    @pytest.mark.asyncio
    async def test_spin_out_switch_to_false(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        original = client.create_session(name="Original")
        
        await client.spin_out(SpinOutOptions(topic="New", switch_to=False))
        
        assert client.get_current_session().id == original.id
    
    @pytest.mark.asyncio
    async def test_spin_out_copy_memories(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Original")
        
        memory = client.get_memory()
        memory.add(content="Important fact", tags=["important"])
        memory.add(content="Skip this", tags=["skip"])
        
        new_session = await client.spin_out(SpinOutOptions(
            topic="With Memory",
            copy_memories=True,
            memory_tags=["important"],
        ))
        
        assert len(new_session.memory) == 1
        assert new_session.memory[0].content == "Important fact"


class TestOAuthCredentials:
    """Tests for OAuth credential handling."""
    
    def test_get_oauth_credentials(self, temp_dir, oauth_auth):
        client = ClaudeClient(auth=oauth_auth, storage_dir=temp_dir)
        creds = client.get_oauth_credentials()
        
        assert creds is not None
        assert creds.access_token == "sk-ant-oat-test-token"
    
    def test_get_oauth_credentials_none_for_api_key(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        creds = client.get_oauth_credentials()
        assert creds is None
    
    def test_set_oauth_credentials(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        
        new_creds = OAuthCredentials(
            access_token="sk-ant-oat-new-token",
            refresh_token="new-refresh",
            expires_at=9999999999999,
        )
        client.set_oauth_credentials(new_creds)
        
        assert client.get_oauth_credentials() == new_creds


class TestAuthTokenSelection:
    """Tests for auth token selection logic."""
    
    @pytest.mark.asyncio
    async def test_prefers_oauth_over_api_key(self, temp_dir, mock_anthropic):
        # Both OAuth and API key provided
        auth = AuthConfig(
            api_key="sk-ant-api-key",
            oauth=OAuthCredentials(
                access_token="sk-ant-oat-oauth-token",
                refresh_token="refresh",
                expires_at=9999999999999,
            ),
        )
        client = ClaudeClient(auth=auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        await client.send("Hello!")
        
        # Check that OAuth token was used (auth_token parameter)
        call_kwargs = mock_anthropic.call_args.kwargs
        assert call_kwargs.get("auth_token") == "sk-ant-oat-oauth-token"
    
    @pytest.mark.asyncio
    async def test_uses_api_key_when_no_oauth(self, temp_dir, mock_anthropic):
        auth = AuthConfig(api_key="sk-ant-api-key")
        client = ClaudeClient(auth=auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        await client.send("Hello!")
        
        # Check that API key was used
        call_kwargs = mock_anthropic.call_args.kwargs
        assert call_kwargs.get("api_key") == "sk-ant-api-key"


class TestLogging:
    """Tests for request/response logging."""
    
    @pytest.mark.asyncio
    async def test_logs_created_when_enabled(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(
            auth=api_key_auth,
            storage_dir=temp_dir,
            enable_logging=True,
        )
        session = client.create_session(name="Test")
        
        await client.send("Hello!")
        
        logs = client.get_logs(session.id)
        assert len(logs) == 2  # Request + Response
        assert logs[0].direction == "request"
        assert logs[1].direction == "response"
    
    @pytest.mark.asyncio
    async def test_no_logs_when_disabled(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(
            auth=api_key_auth,
            storage_dir=temp_dir,
            enable_logging=False,
        )
        session = client.create_session(name="Test")
        
        await client.send("Hello!")
        
        logs = client.get_logs(session.id)
        assert len(logs) == 0
    
    def test_get_log_stats(self, temp_dir, api_key_auth):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        session = client.create_session(name="Test")
        
        stats = client.get_log_stats(session.id)
        assert stats.request_count == 0
        assert stats.response_count == 0


class TestImageSupport:
    """Tests for image handling."""
    
    def test_image_input_from_base64(self):
        img = ImageInput.from_base64("dGVzdA==", "image/png")
        assert img.data == "dGVzdA=="
        assert img.media_type == "image/png"
    
    @pytest.mark.asyncio
    async def test_send_with_images(self, temp_dir, api_key_auth, mock_anthropic):
        client = ClaudeClient(auth=api_key_auth, storage_dir=temp_dir)
        client.create_session(name="Test")
        
        img = ImageInput.from_base64("dGVzdA==", "image/png")
        await client.send("What's in this image?", images=[img])
        
        # Verify image was included in message
        session = client.get_current_session()
        user_msg = session.messages[0]
        assert isinstance(user_msg.content, list)
        assert len(user_msg.content) == 2  # Text + Image

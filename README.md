# Claude Interface (Python)

A lightweight Python library for interfacing with Claude via OAuth or API key. Designed for building chat clients and code assistants.

## Features

- 🔐 **OAuth Support** - Use your Claude Pro/Max subscription (same as Claude Code CLI)
- 💬 **Session Management** - Save, load, and fork conversation sessions
- 📝 **Request/Response Logging** - Full logging of all API interactions (JSONL format)
- 🧠 **Memory System** - Attach persistent memory to sessions
- 🌊 **Streaming** - Real-time response streaming
- 🔀 **Spin Out** - Branch conversations into focused threads
- 📦 **Async/Await** - Fully async API

## Installation

```bash
pip install claude-interface
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

### With OAuth (Claude Pro/Max)

```python
import asyncio
from claude_interface import ClaudeClient, AuthConfig, login

async def main():
    # Interactive login
    credentials = await login(
        on_auth_url=lambda url: print(f"\nOpen this URL:\n{url}\n"),
        on_prompt_code=lambda: asyncio.get_event_loop().run_in_executor(
            None, input, "Paste authorization code: "
        ),
    )
    
    # Create client with OAuth
    client = ClaudeClient(auth=AuthConfig(oauth=credentials))
    
    # Save credentials for next time
    print(f"Save these credentials: {credentials}")

asyncio.run(main())
```

### With API Key

```python
import os
from claude_interface import ClaudeClient, AuthConfig

client = ClaudeClient(
    auth=AuthConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
)
```

## Usage

### Basic Chat

```python
import asyncio
from claude_interface import ClaudeClient, AuthConfig

async def main():
    client = ClaudeClient(auth=AuthConfig(api_key="sk-ant-..."))
    
    # Create a new session
    session = client.create_session(
        name="Code Review",
        system_prompt="You are a helpful code reviewer.",
    )
    
    # Send a message
    result = await client.send("Review this code:\n```js\nconst x = 1;\n```")
    print(result.content)
    
    # Continue the conversation
    follow_up = await client.send("What improvements would you suggest?")
    print(follow_up.content)

asyncio.run(main())
```

### Session Management

```python
# List all sessions
sessions = client.list_sessions()
print(sessions)
# [SessionSummary(id="session_abc123", name="Code Review", message_count=4, ...)]

# Load an existing session
client.load_session("session_abc123")

# Fork a session (create a copy)
forked = client.sessions.fork("session_abc123", "Code Review - Branch A")

# Delete a session
client.delete_session("session_abc123")
```

### Streaming Responses

```python
async for event in client.stream("Explain async/await"):
    match event["type"]:
        case "start":
            print("Starting...")
        case "text":
            print(event["text"], end="", flush=True)
        case "done":
            print(f"\n\nTokens: {event['result'].usage.input_tokens + event['result'].usage.output_tokens}")
        case "error":
            print(f"Error: {event['error']}")
```

### Spin Out Thoughts

Create a new focused session from part of an existing conversation:

```python
from claude_interface import SpinOutOptions

# Spin out the last 4 messages into a new thread
new_session = await client.spin_out(SpinOutOptions(
    topic="TypeScript Best Practices",
    include_last_n=4,
    system_prompt="Focus on the TypeScript patterns we discussed.",
))

# Spin out with AI-generated summary
await client.spin_out(SpinOutOptions(
    topic="Performance Optimization",
    generate_summary=True,
    initial_prompt="Let's dive deeper into the performance issues.",
))

# Spin out with memories
await client.spin_out(SpinOutOptions(
    topic="Code Review",
    copy_memories=True,
    memory_tags=["preferences", "project"],
))
```

### Memory System

```python
# Get memory manager for current session
memory = client.get_memory()

# Add memories
memory.add(
    content="User prefers TypeScript over JavaScript",
    type="preference",
    tags=["language", "typescript"],
    priority=5,
)

memory.add(
    content="Project uses React 18 with Next.js",
    type="context",
    tags=["framework", "react"],
)

# Search memories
results = memory.search("typescript")

# Get memories by tag
prefs = memory.get_by_tag("language")

# Memories are automatically included in system prompt
result = await client.send("What framework should I use?")
```

### Request Logging

```python
# Enable logging (on by default)
client = ClaudeClient(
    auth=AuthConfig(api_key="..."),
    enable_logging=True,
    storage_dir="./my-logs",  # default: ~/.claude-interface
)

# Get logs for current session
logs = client.get_logs()

# Get statistics
stats = client.get_log_stats()
print(f"Requests: {stats.request_count}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Avg response time: {stats.average_duration_ms}ms")
```

## API Reference

### `ClaudeClient`

Main client class for interacting with Claude.

```python
ClaudeClient(
    auth: AuthConfig,           # Required: OAuth or API key
    model: str = "claude-sonnet-4-20250514",
    storage_dir: str | Path = "~/.claude-interface",
    enable_logging: bool = True,
    max_tokens: int = 8192,
    temperature: float | None = None,
    thinking: bool = False,
    thinking_budget: int = 1024,
    base_url: str | None = None,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `send(content, ...)` | Send a message and get response |
| `stream(content, ...)` | Send with streaming response |
| `create_session(...)` | Create a new session |
| `load_session(id)` | Load an existing session |
| `get_current_session()` | Get current session |
| `list_sessions()` | List all sessions |
| `delete_session(id)` | Delete a session |
| `spin_out(options)` | Branch into new session |
| `get_memory()` | Get memory manager |
| `get_logs(session_id?)` | Get logs |
| `get_log_stats(session_id?)` | Get log statistics |

### `login(on_auth_url, on_prompt_code)`

Start OAuth login flow for Claude Pro/Max.

### `SpinOutOptions`

Options for `spin_out()`:

| Option | Type | Description |
|--------|------|-------------|
| `topic` | `str` | Topic/title for new session |
| `name` | `str` | Custom session name |
| `system_prompt` | `str` | System prompt for new session |
| `include_last_n` | `int` | Include last N messages |
| `message_ids` | `list[int]` | Include specific message indices |
| `include_all` | `bool` | Include all messages |
| `generate_summary` | `bool` | AI-summarize context |
| `copy_memories` | `bool` | Copy memories to new session |
| `memory_tags` | `list[str]` | Only copy memories with these tags |
| `initial_prompt` | `str` | Initial prompt in new session |
| `switch_to` | `bool` | Switch to new session (default: True) |

## Storage Structure

```
~/.claude-interface/
├── sessions/
│   ├── session_abc123.json
│   └── session_def456.json
└── logs/
    ├── session_abc123.jsonl
    └── session_def456.jsonl
```

## License

MIT

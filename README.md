# Microsoft AutoGen (v0.4+) — Tools, Teams & Multi-Agent Collaboration

A comprehensive engineering repository tracking advanced multi-agent patterns in Microsoft AutoGen v0.4+. This repository details tool calling schemas, web search integration, conversational memory, collaborative multi-agent teams (`RoundRobinGroupChat`, `SelectorGroupChat`), team operation lifecycles, and termination conditions.

---

## Architecture Overview

```
                                      +-------------------------------+
                                      |   OpenAIChatCompletionClient   |
                                      +---------------+---------------+
                                                      |
                                                      v
                                        +-------------+---------------+
                                        |    Termination Condition    |
                                        |  (TextMention / MaxMessage) |
                                        +-------------+---------------+
                                                      |
                                                      v
+------------------+                    +-------------+---------------+
|   User Prompt    | -----------------> |       Multi-Agent Team      |
+------------------+                    |  (RoundRobin / Selector)    |
                                        +-------+-------------+-------+
                                                |             |
                                                v             v
                                     +----------+----+   +----+----------+
                                     | Weather Agent |   | Editor Agent  |
                                     |  (DDGS Tool)  |   | (Synthesis)   |
                                     +---------------+   +---------------+
                                                |
                                                v
                                        +-------+---------------------+
                                        |      Console UI Stream      |
                                        +-----------------------------+
```

AutoGen v0.4 introduces declarative team orchestration and isolated agent boundaries:

- **Tool Execution & Reflection**: Agents call typed Python functions via structured JSON schema generation. When `reflect_on_tool_use=True`, the agent runs a synthesis step on raw tool results before returning output.
- **Conversational Memory**: Teams maintain an internal context thread across turns, passing message histories (`TextMessage`, `ToolCallMessage`) between participating agents.
- **Multi-Agent Teams**: Pre-built orchestration topologies like `RoundRobinGroupChat` (sequential turn-taking) and `SelectorGroupChat` (dynamic model-selected speaker routing).
- **Explicit Termination**: Teams stop executing through composable termination conditions (`TextMentionTermination`, `MaxMessageTermination`, or logical operators `|` / `&`).

---

## Repository Structure

```
.
├── 01_web_search_tool/
│   └── weather_news.py             # Day 2 Lab: DDGS tool integration & reflect_on_tool_use
├── 02_agent_memory/
│   └── context_memory.py           # Multi-turn context preservation and memory retention
├── 03_teams_and_collaboration/
│   ├── round_robin_team.py         # Sequential collaboration (Researcher + Writer)
│   └── selector_team.py            # Dynamic model-directed speaker selection
├── 04_termination_conditions/
│   └── custom_termination.py       # TextMention, MaxMessage, and combined termination rules
├── 05_multi_agent_operations/
│   └── news_desk_pipeline.py       # End-to-end team workflow (Search -> Draft -> Review -> TERMINATE)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1. Prerequisites

- Python 3.10 – 3.12 (or 3.13 with compatible wheels)
- OpenAI API key configured in an active `.env` file

### 2. Virtual Environment Configuration

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Secret Configuration

```bash
cp .env.example .env
```

Set your API key in `.env`:
```env
OPENAI_API_KEY=your_actual_api_key_here
```

---

## Core Concepts & Implementation Patterns

### 1. Web Search Integration & Tool Reflection
Tools are native Python functions equipped with type hints and explicit docstrings. Enabling `reflect_on_tool_use=True` ensures the LLM formulates a synthesized natural language answer from raw tool output.

```python
from ddgs import DDGS

def search_weather_news(city: str) -> str:
    """Searches the web for today's weather news and forecasts for a given city."""
    results = DDGS().text(f"today weather forecast {city}", max_results=3)
    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
```

### 2. Multi-Agent Teams (`RoundRobinGroupChat`)
`RoundRobinGroupChat` sequences conversation turns deterministically through a designated list of agents.

```python
from autogen_agentchat.teams import RoundRobinGroupChat

team = RoundRobinGroupChat(
    participants=[researcher_agent, editor_agent],
    termination_condition=termination,
)
```

### 3. Termination Conditions
Multi-agent discussions run until a termination rule evaluates to `True`. You can compose rules using bitwise operators (`|` for OR, `&` for AND):

```python
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Stop if an agent says 'TERMINATE' OR if 10 messages have elapsed
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
```

### 4. Team Operations & Lifecycle Management
Executing team workflows requires streaming or running the orchestrator asynchronously, followed by client teardown:

```python
# Stream team execution
await Console(team.run_stream(task="Provide a weather bulletin for Mumbai."))

# Always close the underlying model client
await model_client.close()
```

---

## Lab Execution: Weather News Agent (Day 2)

Run the Day 2 single-agent search integration:

```bash
python 01_web_search_tool/weather_news.py
```

Run the multi-agent editorial desk workflow:

```bash
python 05_multi_agent_operations/news_desk_pipeline.py
```

---

## Troubleshooting Guide

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| Agent outputs raw JSON/unformatted search snippets | `reflect_on_tool_use` was left as `False`. | Set `reflect_on_tool_use=True` during `AssistantAgent` instantiation. |
| Group chat runs in an infinite loop | Missing or unachieved termination condition. | Attach `TextMentionTermination("TERMINATE") | MaxMessageTermination(N)`. |
| `Tool execution error: ddgs` | Network timeout or rate limit on DuckDuckGo. | Add retry/exception handling inside the tool function. |
| Terminal process hangs on exit | Unclosed HTTP connection pool in model client. | Ensure `await model_client.close()` is placed in a `finally` block. |

# Building Agents in LangGraph: A Practical Guide

## Introduction to Agentic Workflow Patterns

When building AI agents, there are five key workflow patterns to consider:

1. **Planning** — Create an outline of the workflow before execution
2. **Tool Use** — Define what tools are available and how to use them
3. **Reflection** — Iteratively improve results, possibly with multiple LLMs critiquing and suggesting edits
4. **Multi-Agent Communication** — Each agent can have a distinct persona
5. **Memory** — Track progress and results of each step

### Additional Useful Capabilities

- **Human-in-the-Loop** — Allows you to guide the agent at critical decision points
- **Persistence** — Store the current state of information so you can return to it later (useful for debugging and production)

---

## Types of Cyclic Graph Architectures

| Architecture | Description |
|--------------|-------------|
| **ReAct** | Reasoning and Acting in Language Models |
| **Self-Refine** | Iterative Refinement with Self-Feedback |
| **AlphaCodium** | Code Generation via Flow Engineering |

---

## Building a ReAct Agent

The ReAct pattern follows a simple loop:

1. The LLM **thinks** about what to do, then decides what **action** to take
2. The action is **executed** in an environment, and an **observation** is returned
3. With the observation, the LLM repeats steps 1–2 until it decides it is done

> **Key Insight:** Pay attention to what falls to the LLM versus what is managed by the code around the LLM (the "runtime").

---

## LangGraph Core Concepts

LangGraph uses a graph-based structure with the following components:

| Component | Purpose |
|-----------|---------|
| **Nodes** | Agents or functions |
| **Edges** | Connect nodes together |
| **Conditional Edges** | Route based on decisions |
| **Entry Point** | The starting node |
| **End Node** | Where to exit the cycle |

### Agent State

The agent state is:
- Accessible to all parts of the graph
- Local to the graph instance
- Persistable to a storage layer
- In a simple implementation, it stores the list of appended messages from the agent's internal thought process

---

## Required Libraries and Setup

### Environment Configuration

```python
from dotenv import load_dotenv
_ = load_dotenv()
```

The `python-dotenv` library loads environment variables from a `.env` file into your application. This is where you'll store API keys for services like OpenAI and Tavily. The underscore `_` is a Python convention for discarding the return value (we don't need it).

### Core Imports

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
```

| Import | Purpose |
|--------|---------|
| `StateGraph` | The main class for building LangGraph workflows |
| `END` | A special constant that marks the termination point of the graph |
| `TypedDict` | Creates dictionaries with a fixed set of typed keys (used for state) |
| `Annotated` | Adds metadata to type hints (used here to specify how state updates work) |
| `operator.add` | The addition operator as a function—used to specify that messages should be appended |
| `AnyMessage` | A union type representing any message type in LangChain |
| `SystemMessage` | A message representing system instructions to the LLM |
| `HumanMessage` | A message representing user input |
| `ToolMessage` | A message containing the result of a tool call |
| `ChatOpenAI` | LangChain's wrapper for OpenAI chat models |
| `TavilySearchResults` | A search tool that uses the Tavily API for web searches |

### Setting Up the Search Tool

```python
tool = TavilySearchResults(max_results=4)

print(type(tool))  # <class 'langchain_community.tools.tavily_search.tool.TavilySearchResults'>
print(tool.name)   # tavily_search_results_json
```

Tavily is a search API designed specifically for LLM applications. Setting `max_results=4` returns up to 4 search results per query, giving the agent more information to work with.

> **Note:** You'll need a Tavily API key in your `.env` file: `TAVILY_API_KEY=your_key_here`

---

## Defining the Agent State

The `AgentState` is the shared data structure that all nodes in your graph can read from and write to.

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

### Understanding This Definition

| Component | Explanation |
|-----------|-------------|
| `TypedDict` | Makes `AgentState` a dictionary with predefined keys and types |
| `messages` | The only key in our state—stores the conversation history |
| `list[AnyMessage]` | The type: a list that can contain any LangChain message type |
| `Annotated[..., operator.add]` | Tells LangGraph *how* to update this field |

### Why `operator.add`?

The `Annotated` wrapper with `operator.add` is crucial. It tells LangGraph that when a node returns `{'messages': [new_message]}`, it should **append** to the existing list rather than **replace** it.

```python
# Without operator.add: replacement behavior
state['messages'] = [new_message]  # Old messages lost!

# With operator.add: append behavior  
state['messages'] = state['messages'] + [new_message]  # History preserved!
```

This is what enables the agent to maintain conversation history across the reasoning loop.

---

## Implementation: ReAct Agent Class

### Complete Class Definition

```python
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
```

---

## Class Breakdown

### Constructor: `__init__`

The `Agent` class requires three parameters:

| Parameter | Description |
|-----------|-------------|
| `model` | The LLM to use (e.g., `ChatOpenAI`) |
| `tools` | List of available tools for the agent |
| `system` | System prompt instructions (optional, defaults to empty string) |

**What happens in `__init__`:**

1. **Save the system prompt** as an instance attribute
2. **Build the graph structure:**
   - Add an `"llm"` node that calls the language model
   - Add an `"action"` node that executes tools
   - Add a conditional edge from `"llm"` that routes to `"action"` if tools are called, or `END` if not
   - Add an edge from `"action"` back to `"llm"` (creates the loop)
   - Set `"llm"` as the entry point
3. **Compile the graph** into an executable workflow
4. **Create a tools dictionary** mapping tool names to tool objects for easy lookup
5. **Bind tools to the model** so the LLM knows what tools are available

### Method: `exists_action`

```python
def exists_action(self, state: AgentState):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0
```

**Purpose:** Conditional edge function that determines whether to take an action or end.

**Logic:** Checks the most recent message from the LLM. If it contains any `tool_calls`, returns `True` (route to action node); otherwise returns `False` (end the loop).

### Method: `call_openai`

```python
def call_openai(self, state: AgentState):
    messages = state['messages']
    if self.system:
        messages = [SystemMessage(content=self.system)] + messages
    message = self.model.invoke(messages)
    return {'messages': [message]}
```

**Purpose:** The LLM node—invokes the language model with the current conversation history.

**Logic:** 
1. Gets all messages from state
2. Prepends the system prompt if one exists
3. Calls the model and gets a response
4. Returns the new message (which gets *appended* to state due to `operator.add`)

### Method: `take_action`

```python
def take_action(self, state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling: {t}")
        if not t['name'] in self.tools:
            print("\n ....bad tool name....")
            result = "bad tool name, retry"
        else:
            result = self.tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Back to the model!")
    return {'messages': results}
```

**Purpose:** The action node—executes the tool calls requested by the LLM.

**Logic:**
1. Extracts tool calls from the last message
2. Loops through each tool call (supports parallel tool calling)
3. Looks up the tool by name and invokes it with the provided arguments
4. Handles bad tool names gracefully by returning an error message
5. Wraps each result in a `ToolMessage` and returns them all

> **Note:** The results are returned as `{'messages': results}`, which appends them to the conversation history thanks to `operator.add`.

---

## Putting It All Together

### Define the Agent

```python
from langchain_openai import ChatOpenAI

prompt = """You are a smart research assistant. Use the search engine to look up information.
You are allowed to make multiple calls (either together or in sequence).
Only look up information when you are sure of what you want.
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt)
```

### Visualize the Graph

```python
from IPython.display import Image

Image(abot.graph.get_graph().draw_png())
```

### Run the Agent

```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})
```

### Access the Results

```python
# View the full result (includes all messages and tool calls)
result

# Get just the final response
result['messages'][-1].content
```

---

## Agentic Search vs. Regular Search

### What's the Difference?

| Regular Search | Agentic Search (Tavily) |
|----------------|-------------------------|
| Returns raw website URLs | Returns structured JSON data |
| Requires web scraping to extract text | Information is pre-extracted and cleaned |
| Results are messy with non-relevant content | Results are succinct and relevant |
| Needs additional processing for LLM use | Ready for LLM consumption out of the box |

> **Key Insight:** Tavily is designed specifically for LLM applications. It returns information in a structured JSON format that language models can process directly to generate quality responses.

---

## Setting Up Agentic Search with Tavily

### Basic Setup

```python
from dotenv import load_dotenv
import os
from tavily import TavilyClient

# Load environment variables from .env file
_ = load_dotenv()

# Connect to Tavily
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
```

### Example 1: Simple Query with Direct Answer

```python
# Run search with answer generation
result = client.search(
    "What is in Nvidia's new Blackwell GPU?",
    include_answer=True
)

# Print the generated answer
print(result["answer"])
```

The `include_answer=True` parameter tells Tavily to generate a concise answer based on the search results—perfect for quick factual queries.

### Example 2: Location-Based Query

```python
# Set your location
city = "San Francisco"

query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""

# Run search (limiting to 1 result)
result = client.search(query, max_results=1)

# Extract the content from the first result
data = result["results"][0]["content"]
print(data)
```

### Pretty-Printing JSON Results

For better readability when debugging or exploring results:

```python
import json
from pygments import highlight, lexers, formatters

# Parse JSON (handle single quotes if present)
parsed_json = json.loads(data.replace("'", '"'))

# Pretty print with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(
    formatted_json,
    lexers.JsonLexer(),
    formatters.TerminalFormatter()
)

print(colorful_json)
```

### Example Output

```json
{
    "location": {
        "name": "San Francisco",
        "region": "California",
        "country": "United States of America",
        "lat": 37.775,
        "lon": -122.4183,
        "tz_id": "America/Los_Angeles",
        "localtime": "2026-01-22 01:59"
    },
    "current": {
        "temp_c": 10.6,
        "temp_f": 51.1,
        "is_day": 0,
        "condition": {
            "text": "Overcast",
            "icon": "//cdn.weatherapi.com/weather/64x64/night/122.png",
            "code": 1009
        },
        "wind_mph": 2.2,
        "wind_dir": "SW",
        "humidity": 86,
        "feelslike_f": 51.6,
        "vis_miles": 9.0,
        "uv": 0.0
    }
}
```

> **Note:** This clean, structured output is exactly what makes agentic search powerful—the LLM can easily parse and reason about this data without additional preprocessing.

---
## Persistence and Streaming

### Why These Matter

| Capability | Purpose |
|------------|---------|
| **Persistence** | Save agent state at any point in time, allowing you to resume from that exact state later |
| **Streaming** | Emit real-time signals showing what the agent is doing during long-running operations |

---

## Adding Persistence

Persistence is implemented through a **checkpointer** that saves the state after and between every node in the graph.

### Setting Up the Checkpointer

We'll use `SqliteSaver` with an in-memory database. You can easily swap this for an external database, Redis, etc.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
```

### Integrating with the Agent Class

Add `checkpointer` as a parameter in the `Agent` class:

```python
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        # ... (node and edge setup)
        self.graph = graph.compile(checkpointer=checkpointer)  # Pass checkpointer here
        # ... (rest of init)
```

Then instantiate with the memory object:

```python
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
```

---

## Streaming Messages

There are two levels of streaming to consider:

1. **Messages** — AI messages (actions to take) and observation messages (results of actions)
2. **Tokens** — Individual tokens as the LLM generates output

Let's start with message streaming.

### Setting Up Thread Configuration

Threads allow you to maintain separate conversations simultaneously—essential for production applications. A thread is simply a dictionary with a configurable ID:

```python
thread = {"configurable": {"thread_id": "1"}}
```

### Streaming with `.stream()`

Instead of `.invoke()`, use `.stream()` to get events as they happen:

```python
messages = [HumanMessage(content="What is the weather in sf?")]
thread = {"configurable": {"thread_id": "1"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

Events represent updates to the state over time. Since our state only has a `messages` key, we loop through the values and print them.

### Conversation Memory Across Messages

When you keep the **same thread ID**, the agent remembers previous context:

```python
# Follow-up question on the same thread
messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}  # Same thread ID

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

The agent will remember you were asking about weather and continue that line of questioning.

> **Note:** If you change to `{"thread_id": "2"}`, the agent won't have context from the previous conversation.

---

## Streaming Tokens

For real-time token-by-token output, use the `astream_events` method. This is an **asynchronous** method, so we need an async checkpointer.

### Async Setup

```python
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
```

### Streaming Token Events

The `astream_events` method emits various event types. We filter for `"on_chat_model_stream"` events, which represent new tokens:

```python
messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}

async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content means the model is asking for a tool to be invoked
            # So we only print non-empty content
            print(content, end="|")
```

The pipe delimiter `|` visually shows tokens streaming in real-time:

```
The|weather|in|San|Francisco|is|currently|55|°F|with|partly|cloudy|skies|.|
```

> **Note:** Empty content chunks typically indicate the model is requesting a tool invocation, so we skip those.


---
## Human-in-the-Loop

Human-in-the-loop allows you to pause agent execution for manual approval, modify agent state, and even "time travel" to previous states. This is essential for debugging, testing, and building agents that require human oversight.

---

### Custom Message Reducer for Editing

When humans edit messages, we need to handle replacements (not just appends). The `reduce_messages` function replaces messages with the same ID and appends new ones:

```python
from uuid import uuid4
from langchain_core.messages import AnyMessage

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # Assign IDs to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    
    # Merge new messages with existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # Replace any existing messages with the same ID
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # Append any new messages to the end
            merged.append(message)
    return merged
```

Update the `AgentState` to use this reducer:

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
```

> **Note:** If no edits are made, messages append normally. The reducer only replaces when IDs match.

---

### Adding Manual Approval (Interrupt Before Action)

To require human approval before tool execution, add `interrupt_before` to the graph compilation:

```python
self.graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["action"]  # Pause before the action node
)
```

> **Tip:** You can also interrupt only for specific tools—see the [LangGraph documentation](https://langchain-ai.github.io/langgraph/) for details.

### Checking Interrupt State

```python
# Get the current state
abot.graph.get_state(thread)

# Check what node is next (if paused)
abot.graph.get_state(thread).next  # Returns ("action",) if paused before action
```

### Continuing After an Interrupt

Pass `None` to `.stream()` to continue from where it paused:

```python
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
```

After completion, `.next` will be empty:

```python
abot.graph.get_state(thread).next  # Returns () when complete
```

### Interactive Yes/No Approval Loop

```python
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}

# Initial run (will pause before action)
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# Approval loop
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread), "\n")
    _input = input("proceed? (y/n): ")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
```

---

### State Memory and Snapshots

As the graph executes, a snapshot of each state is stored in memory.

#### What's in a Snapshot?

| Field | Description |
|-------|-------------|
| `values` | The `AgentState` (messages, etc.) |
| `config` | Contains `thread_id` and `thread_ts` |
| `thread_ts` | Unique identifier for this specific snapshot |

#### Accessing State

```python
# Get current state (most recent snapshot)
abot.graph.get_state(thread)

# Get all snapshots (returns iterator)
abot.graph.get_state_history(thread)
```

#### Using `thread_ts` for Time Travel

| Command | Behavior |
|---------|----------|
| `g.invoke(None, thread)` | Uses current state as starting point |
| `g.invoke(None, {**thread, "thread_ts": ts})` | Uses specific snapshot as starting point |
| `g.stream(None, {**thread, "thread_ts": ts})` | Streams from specific snapshot |

---

### Modifying State

You can modify the agent's state mid-execution to change its behavior.

#### Example: Changing a Search Query

**Step 1:** Ask a question (pauses before action due to `interrupt_before`)

```python
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "3"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

**Step 2:** Save the current state

```python
current_values = abot.graph.get_state(thread)

# View the AI message (shows it wants to search for LA weather)
current_values.values['messages'][-1]

# View the tool calls
current_values.values['messages'][-1].tool_calls
```

**Step 3:** Modify the tool call (change "LA" to "Louisiana")

```python
# Get the tool call ID (must preserve this)
_id = current_values.values['messages'][-1].tool_calls[0]['id']

# Update the tool call with new query
current_values.values['messages'][-1].tool_calls = [
    {
        'name': 'tavily_search_results_json',
        'args': {'query': 'current weather in Louisiana'},
        'id': _id  # Keep the same ID
    }
]
```

**Step 4:** Apply the update

```python
abot.graph.update_state(thread, current_values.values)

# Verify the change
abot.graph.get_state(thread)  # Should show "Louisiana" in the query
```

**Step 5:** Continue execution

```python
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
```

> **Important:** Modifying state creates a *new* snapshot rather than overwriting. This preserves history and enables time travel.

---

### Time Travel

Access and replay from any previous state in the execution history.

#### Collecting State History

```python
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)
```

The list is in reverse chronological order, so `states[-1]` is the earliest state.

#### Replaying from a Previous State

```python
to_replay = states[-1]  # Get the earliest state

for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)
```

---

### Go Back in Time and Edit

You can travel to a previous state, modify it, and create a new branch of execution.

```python
# Get an old state
to_replay = states[-1]

# Get the tool call ID
_id = to_replay.values['messages'][-1].tool_calls[0]['id']

# Modify the tool call (e.g., add "accuweather" to the query)
to_replay.values['messages'][-1].tool_calls = [
    {
        'name': 'tavily_search_results_json',
        'args': {'query': 'current weather in LA, accuweather'},
        'id': _id
    }
]

# Create a new branch with the modified state
branch_state = abot.graph.update_state(to_replay.config, to_replay.values)

# Execute from the new branch
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

> **How this works:** `update_state()` returns a new config pointing to a branched state. The original history remains unchanged—you've created an alternate timeline.

---

### Inject a Message at a Specific Point

You can add a synthetic message (like a fake tool response) at a previous state, effectively bypassing the actual tool call.

```python
# Get an old state
to_replay = states[-1]

# Get the tool call ID that we're "responding" to
_id = to_replay.values['messages'][-1].tool_calls[0]['id']

# Create a fake tool response
state_update = {
    "messages": [
        ToolMessage(
            tool_call_id=_id,
            name="tavily_search_results_json",
            content="54 degree celsius",  # Fake/mocked response
        )
    ]
}

# Apply update AS IF it came from the "action" node
branch_and_add = abot.graph.update_state(
    to_replay.config,
    state_update,
    as_node="action"  # This is key!
)

# Continue execution (LLM will respond to the fake tool result)
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)
```

> **Why use `as_node="action"`?** This tells LangGraph to treat the update as if it came from the action node. The graph then knows to proceed to the next node in the flow (back to the LLM) rather than staying at the current position.

**Use cases for message injection:**
- **Testing:** Mock tool responses without making actual API calls
- **Debugging:** See how the agent responds to specific tool outputs
- **Correction:** Override a bad tool response with accurate information

---
## Example: Essay Writer Agent

This example demonstrates a more complex multi-node agent that plans, researches, writes, and iteratively refines an essay based on self-critique.

### Architecture Overview

```
┌─────────┐    ┌───────────────┐    ┌──────────┐
│ Planner │───▶│ Research Plan │───▶│ Generate │
└─────────┘    └───────────────┘    └────┬─────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                   [max revisions                  ┌─────────┐
                    reached?]                      │ Reflect │
                         │                         └────┬────┘
                         ▼                              │
                       [END]                            ▼
                                         ┌──────────────────────┐
                                         │ Research Critique    │
                                         └──────────┬───────────┘
                                                    │
                                                    └───▶ [back to Generate]
```

---

### Imports and Setup

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")
```

Standard imports plus persistence setup. Note that we import `List` from typing for the content field.

---

### Agent State Definition

```python
class AgentState(TypedDict):
    task: str              # The essay topic/assignment
    plan: str              # High-level outline of the essay
    draft: str             # Current version of the essay
    critique: str          # Feedback on the current draft
    content: List[str]     # Research content collected from searches
    revision_number: int   # Current revision count
    max_revisions: int     # Maximum allowed revisions before stopping
```

Unlike the simple ReAct agent that only tracks messages, this agent has **structured state fields** for each stage of the writing process. This makes it easy to pass specific information between nodes.

---

### Model Setup

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

Using `temperature=0` for more deterministic, consistent outputs—important for structured tasks like outlining and critique.

---

### Prompt Templates

Each node has a specialized prompt that defines its role:

```python
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
```

| Prompt | Purpose |
|--------|---------|
| `PLAN_PROMPT` | Creates the essay outline |
| `WRITER_PROMPT` | Generates/revises the essay draft |
| `REFLECTION_PROMPT` | Acts as a teacher providing critique |
| `RESEARCH_PLAN_PROMPT` | Generates search queries for initial research |
| `RESEARCH_CRITIQUE_PROMPT` | Generates search queries to address critique |

---

### Structured Output for Queries

```python
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]
```

This Pydantic model ensures the LLM returns a structured list of search queries. Using `with_structured_output()` forces the model to conform to this schema.

---

### Tavily Client Setup

```python
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
```

Direct Tavily client for making search requests within the nodes.

---

### Node Functions

#### 1. Plan Node

```python
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}
```

**Purpose:** Creates a high-level outline for the essay.

**Input:** The task (essay topic) from state  
**Output:** Updates the `plan` field with the outline

---

#### 2. Research Plan Node

```python
def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
```

**Purpose:** Generates search queries and collects research material.

**How it works:**
1. Uses `with_structured_output(Queries)` to get a list of search queries
2. Iterates through each query and searches Tavily
3. Appends all search result content to the `content` list

**Input:** The task from state  
**Output:** Updates `content` with research findings

---

#### 3. Generation Node

```python
def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }
```

**Purpose:** Writes or revises the essay draft.

**How it works:**
1. Joins all research content into a single string
2. Combines the task and plan into the user message
3. Injects research content into the system prompt
4. Increments the revision number

**Input:** Task, plan, and content from state  
**Output:** Updates `draft` and increments `revision_number`

---

#### 4. Reflection Node

```python
def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}
```

**Purpose:** Acts as a teacher grading the essay, providing detailed feedback.

**Input:** The current draft  
**Output:** Updates `critique` with improvement suggestions

---

#### 5. Research Critique Node

```python
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}
```

**Purpose:** Researches additional information to address the critique.

**How it works:** Similar to `research_plan_node`, but generates queries based on the critique rather than the original task.

**Input:** The critique from state  
**Output:** Appends additional research to `content`

---

### Conditional Edge Function

```python
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
```

**Purpose:** Determines whether to continue revising or finish.

**Logic:**
- If `revision_number` exceeds `max_revisions` → End the graph
- Otherwise → Continue to the reflection node for another revision cycle

---

### Building the Graph

```python
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

# Set entry point
builder.set_entry_point("planner")

# Add conditional edge from generate (continue or end?)
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

# Add linear edges
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

# Compile with persistence
graph = builder.compile(checkpointer=memory)
```

**Flow summary:**
1. `planner` → `research_plan` → `generate`
2. From `generate`: check if max revisions reached
   - If yes → `END`
   - If no → `reflect` → `research_critique` → `generate` (loop)

---

### Visualizing the Graph

```python
from IPython.display import Image

Image(graph.get_graph().draw_png())
```

---

### Running the Agent

```python
thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
```

**Initial state values:**
- `task`: The essay topic
- `max_revisions`: Stop after 2 revision cycles
- `revision_number`: Start at 1

The agent will:
1. Plan the essay
2. Research the topic
3. Write a draft
4. Reflect and critique
5. Research to address critique
6. Revise (repeat steps 4-6 until `max_revisions` is reached)

---

### Running with a GUI (Optional)

```python
import warnings
warnings.filterwarnings("ignore")

from helper import ewriter, writer_gui

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()
```

> **Note:** This requires a `helper` module with `ewriter` and `writer_gui` functions that wrap the agent in a Gradio or similar web interface.

---

### Key Patterns Demonstrated

| Pattern | Implementation |
|---------|----------------|
| **Multi-node workflow** | 5 specialized nodes instead of a single LLM + action loop |
| **Structured state** | Typed fields for each piece of data (plan, draft, critique, etc.) |
| **Iterative refinement** | Generate → Reflect → Research → Generate loop |
| **Conditional termination** | `should_continue` checks revision count |
| **Research integration** | Tavily searches at multiple points in the workflow |
| **Structured outputs** | Pydantic model ensures queries are properly formatted |

---

## Summary

Building a ReAct agent in LangGraph involves:

1. Defining an **AgentState** to track messages
2. Creating **nodes** for the LLM and tool execution
3. Using **conditional edges** to decide when to act vs. when to stop
4. **Compiling** the graph and invoking it with user messages

This architecture provides a clean separation between reasoning (LLM) and acting (tools), while the graph structure handles the control flow.

For search-enabled agents, **Tavily's agentic search** provides LLM-optimized results that integrate seamlessly with the ReAct pattern.

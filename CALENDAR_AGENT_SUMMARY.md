# Calendar Agent - Specialized Natural Language Parser

## 🧠 Overview

The Calendar Agent is a specialized natural language processing agent within a multi-agent system that parses user queries about calendar events and generates structured output for downstream agents.

## 🎯 Core Capabilities

### Supported Intent Types

1. **`create_event`** - User wants to add/schedule something
2. **`read_events`** - User wants to see/find existing events
3. **`query_memory`** - User asks about past events (memory-based queries)
4. **`delete_event`** - User wants to cancel/remove something
5. **`unknown`** - Cannot determine intent

### Structured Output Fields

- **`query`** - Original user query
- **`intent`** - One of the 5 intent types above
- **`target_info`** - Key nouns like "dentist", "barber", "Sarah", "soccer"
- **`timeframe`** - Specific date (YYYY-MM-DD) or relative timeframe
- **`ambiguities`** - List of unclear elements needing clarification
- **`test_passed`** - Whether the extracted meaning passes validation

## 📋 Example: "When last did I go to the barber?"

```json
{
  "query": "When last did I go to the barber?",
  "intent": "query_memory",
  "target_info": "barber",
  "timeframe": "past",
  "ambiguities": [],
  "test_passed": true
}
```

## 🔄 Processing Pipeline

1. **Input Validation** - Security checks and PII sanitization
2. **LLM Parsing** - Primary method using OpenAI function calling
3. **Regex Fallback** - Pattern matching when LLM fails
4. **Validation** - Ensures extracted data makes sense
5. **Structured Output** - JSON format for downstream agents

## 🤖 Integration with Multi-Agent System

### Downstream Agent Routing

Based on the extracted `intent`, the structured output is routed to specialized agents:

| Intent         | Downstream Agent       | Action                             |
| -------------- | ---------------------- | ---------------------------------- |
| `query_memory` | `MemoryLookupAgent`    | Search past events for target_info |
| `create_event` | `SchedulerAgent`       | Create new calendar event          |
| `read_events`  | `CalendarReaderAgent`  | Search existing events             |
| `delete_event` | `CalendarManagerAgent` | Remove calendar event              |
| `unknown`      | `ClarificationAgent`   | Ask user for more details          |

### Example Flow

```
User: "When last did I go to the barber?"
↓
Calendar Agent: {intent: "query_memory", target_info: "barber", ...}
↓
MemoryLookupAgent: Search past events containing "barber"
↓
Response: "You last went to the barber on March 15th, 2025"
```

## 🛠️ Technical Implementation

### Architecture

- Inherits from `BaseAgent` for consistent messaging
- Uses `AgentMessage`/`AgentResponse` for communication
- Supports both LLM and regex parsing methods
- Includes comprehensive error handling and logging

### Key Features

- **Dual Parsing Strategy**: LLM function calling with regex fallback
- **Security Integration**: PII sanitization and input validation
- **Flexible Patterns**: Extensible regex patterns for different query types
- **Validation Logic**: Ensures extracted data is actionable
- **Confidence Scoring**: Provides confidence levels for downstream decisions

### Regex Patterns

```python
"query_memory": [
    r"(when\s+last|last\s+time)\s+(did\s+I|I\s+went|I\s+had)",
    r"when\s+last",
    # ... more patterns
]
```

## 📊 Test Results

The calendar agent successfully processes various query types:

| Query Type                            | Intent       | Target Info | Timeframe     | Test Passed |
| ------------------------------------- | ------------ | ----------- | ------------- | ----------- |
| "When last did I go to the barber?"   | query_memory | barber      | past          | ✅          |
| "Add meeting with Alex next Thursday" | create_event | meeting     | next thursday | ✅          |
| "Show my dentist appointments"        | read_events  | appointment | unspecified   | ✅          |
| "Cancel the soccer game"              | delete_event | soccer      | unspecified   | ✅          |

## 🚀 Usage

### Basic Usage

```python
from agents.calendar_agent import CalendarAgent
from agents.base import AgentMessage

# Initialize agent
agent = CalendarAgent(openai_client, settings)

# Process query
message = AgentMessage(
    recipient="calendar_agent",
    sender="user",
    body={"text": "When last did I go to the barber?"}
)

response = await agent.process_message(message)
result = response.data  # CalendarQueryResult object
```

### Integration with Orchestrator

```python
# The orchestrator can route based on intent
if result.intent == "query_memory":
    await memory_agent.process_message(memory_message)
elif result.intent == "create_event":
    await scheduler_agent.process_message(scheduler_message)
```

## 🎯 Benefits

1. **Specialized Focus**: Dedicated to calendar query parsing
2. **Structured Output**: Consistent JSON format for downstream agents
3. **Robust Parsing**: Dual LLM + regex approach ensures reliability
4. **Clear Intent Classification**: 5 distinct intent types for precise routing
5. **Validation**: Built-in validation ensures quality output
6. **Extensible**: Easy to add new patterns and capabilities

## 🔮 Future Enhancements

- **Temporal Reasoning**: Better handling of complex time expressions
- **Context Awareness**: Consider user's calendar context
- **Multi-language Support**: Extend to other languages
- **Learning**: Improve patterns based on usage data
- **Integration**: Connect with actual calendar APIs

---

The Calendar Agent serves as the intelligent front-end parser for the multi-agent calendar system, ensuring that natural language queries are properly understood and routed to the appropriate specialized agents for execution.

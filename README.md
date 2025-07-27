# Calendar Agent

A smart calendar assistant using AI to help manage your Google Calendar events.

## 🚀 Recent Updates - OpenAI v1.x SDK + OpenRouter Integration

**REFACTORED**: The entire codebase has been updated to use OpenAI SDK v1.x with OpenRouter support:

### ✅ **Key Changes Made:**

1. **Environment Variables**:

   - All files now properly load `OPENAI_API_KEY` and `OPENAI_API_BASE` via `load_dotenv()`
   - Added default fallback to OpenRouter: `https://openrouter.ai/api/v1`

2. **OpenAI Client Initialization**:

   ```python
   # OLD (deprecated):
   openai.api_key = os.getenv("OPENAI_API_KEY")

   # NEW (v1.x SDK with OpenRouter):
   client = OpenAI(
       api_key=os.getenv("OPENAI_API_KEY"),
       base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
   )
   ```

3. **Model IDs Updated for OpenRouter**:

   - `"gpt-4o"` → `"openai/gpt-4o"`
   - `"gpt-3.5-turbo"` → `"openai/gpt-3.5-turbo"`

4. **API Calls Updated**:
   - All `openai.ChatCompletion.create()` calls replaced with `client.chat.completions.create()`
   - Preserved all existing logic, error handling, and functionality

### 📁 **Files Updated:**

- ✅ `app_minimal.py` - Complete refactor from deprecated to v1.x syntax
- ✅ `app.py` - Updated client initialization and model IDs
- ✅ `test_agent.py` - Updated client initialization and model IDs
- ✅ `agents/nlu_parser.py` - Updated default model ID
- ✅ `test_openai.py` - Already used correct v1.x syntax
- ✅ `main.py` - Already used correct v1.x syntax

### 🧪 **Testing Status:**

- ✅ `python test_openai.py` - **PASSES** - OpenRouter API connection verified
- ✅ `python test_agent.py` - **PASSES** - Calendar agent OpenAI integration verified
- ✅ All OpenAI calls now use v1.x SDK with OpenRouter compatibility

### 🔧 **Setup Requirements:**

# Secrets removed for security.  See `.env.example` in the repo for the list of required environment variables and configure them locally.  Do NOT include API keys or tokens in this document.

---

## Config & Secrets

Load secrets from a `.env` file using `python‑dotenv`. The spec should never embed raw API keys or OAuth tokens.  Provide a `.env.example` template in your repo listing the required variables (e.g., `OPENAI_API_KEY`, `GOOGLE_OAUTH_TOKEN`, `APPLE_CALENDAR_SECRET`), and instruct developers to populate their own values locally.

## Machine‑Learning Logic

* Training occurs when F1 ≥ 0.70.  By default we use a TF‑IDF + RandomForest baseline, but this module can be swapped out for improved models.  The ML suggestion agent should surface candidates but not bypass validation or user approval.

## Roadmap & Assumptions

* Use environment‑specific settings to control resource usage.  For example, limit calls to paid LLMs in free tiers, but allow scaling in production.  Budgets and quotas belong in config, not hard‑coded spec text.

## Building Block Principles & Best Practices

The calendar agent should adhere to proven agentic design principles.  Key takeaways from the AI‑cookbook's seven foundational building blocks and broader industry research:

* **Intelligence (LLM reasoning)** – Only call the LLM when deterministic logic isn't enough.  Separate reasoning (LLM) from execution (code) and include all relevant context (task details, examples, user preferences) so the model can produce accurate, structured outputs:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
* **Memory (context persistence)** – Maintain short‑term scratchpads, a mid‑term task board, and a long‑term vector store to provide the LLM with necessary history.  Pass prior interactions into prompts so conversations stay coherent:contentReference[oaicite:2]{index=2}.
* **Tools (function calling)** – Expose scheduling, database and calendar operations as explicit functions with JSON schemas.  The LLM decides which tool to call; your code handles the call and returns results:contentReference[oaicite:3]{index=3}.
* **Validation** – Validate every structured output from the LLM against a schema using Pydantic or similar:contentReference[oaicite:4]{index=4}.  If validation fails, ask the model to fix its output or fall back to a deterministic parser.
* **Control** – Centralize routing logic in an Orchestrator.  Use deterministic heuristics and conditions to decide which sub‑agent to invoke:contentReference[oaicite:5]{index=5}.
* **Recovery** – Implement retries with exponential backoff and safe fallbacks for API failures and model errors:contentReference[oaicite:6]{index=6}.
* **Feedback (human‑in‑the‑loop)** – For high‑risk tasks (sending invites, deleting events), require human approval before execution:contentReference[oaicite:7]{index=7}.  After tasks finish, collect feedback on success/failure to improve future suggestions.
* **Context engineering & scope** – Fill the context window intelligently; include relevant data and avoid fluff.  Don't build agents for trivial or high‑stakes deterministic workflows:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}.
* **Privacy & fairness** – Don't store personally identifiable information in prompts or logs.  Implement tiered personalization so users control how much is remembered.  Respect regulatory requirements:contentReference[oaicite:10]{index=10}.
* **Observability & testing** – Instrument your agent with structured logging, metrics and health checks.  Provide unit tests (e.g., `test_openai.py`, `test_agent.py`, `test_google_auth.py`) and an evaluation harness to guard against regressions.
* **Modularity & scalability** – Keep agents small and focused (e.g., NLUParsingAgent, SchedulerAgent, MLSuggestionAgent, MemoryManager, FeedbackAgent, ErrorRetryManager) and orchestrate them via a central planner.  Plug in new calendar adapters without changing core logic.

## Validation & Error Handling

Add a `Validator` module that checks LLM‑generated task objects against the defined schema.  On failure, the Orchestrator retries parsing twice with additional context; if it still fails, it falls back to a regex parser and logs a warning.  This ensures that downstream code always receives well‑formed data.

## Human Approval Policies

Define a list of actions that always require human confirmation — such as sending calendar invites, modifying events on external calendars, or deleting tasks.  The FeedbackAgent should present a concise summary of the pending action and wait for a clear yes/no before proceeding.

## Privacy & Secrets

Do not store secrets or tokens in the doc.  Load keys like `OPENAI_API_KEY` and `GOOGLE_OAUTH_TOKEN` from `.env` using `python‑dotenv`.  Always sanitize user input before logging and avoid including PII in prompts or logs.

## 🧠 Architecture (TL;DR)

**Core agents**

- **Orchestrator** – plans, spawns sub‑agents, routes tool calls
- **NLU / Parsing** – LLM function calling + regex fallback
- **Scheduler** – finds/places time slots, resolves conflicts
- **ML Suggestion** – predicts success probabilities & durations
- **Memory Manager** – 3‑tier memory: Redis (scratchpad), Postgres (task board), pgvector/Chroma (long‑term)
- **Feedback** – asks post‑event questions, feeds data back to ML
- **Error / Retry** – exponential backoff, circuit breakers, safe fallbacks

**Tool adapters**

- Google Calendar CRUD adapter
- Plug‑in pattern for Apple/Outlook/etc.

**Infra**

- Redis cache, Postgres (+ pgvector), Streamlit UI
- OpenTelemetry/Sentry + structured JSON logs

---

## 🚀 Quick Start

### 1. Prereqs

- Python 3.9+
- OpenAI API key (and/or Anthropic if you toggle)
- Google Calendar OAuth creds
- (Optional) Postgres + pgvector, Redis

### 2. Install

```bash
git clone <repo-url>
cd calendar-agent
python -m venv env
# Windows PowerShell
.\env\Scripts\Activate.ps1
# macOS / Linux
source env/bin/activate
pip install -r requirements.txt

```

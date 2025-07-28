Calendar Agent

An AI-powered multi-agent calendar assistant that understands natural language, learns your preferences over time, and manages your schedule across platforms like Google Calendar and Apple Calendar.

🚀 Features

✅ Natural language understanding (e.g., “Add dentist next Tuesday at 2 PM”)

✅ Google Calendar integration (Apple Calendar support coming)

✅ Task metadata tracking (importance, duration, status, blockers)

✅ Machine learning-based time suggestions (based on your past behavior)

✅ Streamlit-based desktop UI + voice-ready pipeline

✅ Auto-updating checklist in Google Docs

✅ Modular multi-agent architecture with memory and error handling

🏗️ Architecture Overview

┌──────────────────────┐        ┌──────────────────────┐
│  User Input  │───────▶│  NLU Agent  │
└──────────────────────┘        └──────────────┘
                               ▼
                        ┌──────────────────┐
                        │  Scheduler   │
                        └───────────────┘
                               ▼
                        ┌────────────────────┐
                        │ Memory Agent  │
                        └───────────────┘
                               ▼
                        ┌──────────────────┐
                        │ Calendar API  │
                        └───────────────┘

Bonus Agents: ML Suggestions, Error Retry, Feedback Tracker

🧠 Tech Stack

Python 3.x

Streamlit for UI

Google Calendar API

OpenAI API for LLM-driven parsing

LangChain-style multi-agent setup (custom-built)

BFG for Git history cleanup

🛠️ Setup Instructions

1. Clone the repo

git clone https://github.com/Patrick784/calendar-agent.git
cd calendar-agent

2. Create & activate a virtual environment

python -m venv env
# Windows
.\env\Scripts\Activate.ps1
# macOS/Linux
source env/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Setup Google Credentials

Create a credentials.json file from Google Cloud Console. On first run, it will open a browser window to authenticate.

💬 Example Prompts

“Schedule a team sync every Monday at 9am”

“Add dentist appointment next Tuesday, 1 hour, importance 4”

“Find a better slot for my deep work session”

🤖 Roadmap

Google Calendar integration

ML-based time suggestions

Apple Calendar support

Telegram voice input agent

Full AI agent memory and planning cycle

🛹 Repo Status

✅ Clean history (no .venv or DLLs)

✅ Optimized for GitHub + Codex

✅ Ready for review, forks, or AI-powered extension

📄 License

MIT — feel free to fork, extend, or integrate into your AI stack.


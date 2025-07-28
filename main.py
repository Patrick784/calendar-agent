"""
Multi-Agent Calendar System - Unified Main Application

Consolidated entry point that routes all functionality through the OrchestratorAgent.
Provides multiple UI modes: Production, Minimal, ML-Enhanced, and Debug.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
import streamlit as st
from datetime import datetime, timezone
import json
import openai
from dotenv import load_dotenv
import re

# Import multi-agent system components
from agents.orchestrator import OrchestratorAgent
from agents.nlu_parser import NLUParsingAgent
from agents.scheduler import SchedulerAgent
from agents.memory_manager import MemoryManager
from agents.error_manager import ErrorRetryManager
from agents.ml_suggestions import MLSuggestionAgent
from agents.feedback import FeedbackAgent
from agents.base import AgentMessage

from adapters.google_calendar import GoogleCalendarAdapter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedCalendarSystem:
    def __init__(self):
        self.initialized = False
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", "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"),
            "openai_api_base": os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
            "openai_model": "openai/gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.2,
            "user_timezone": os.getenv("USER_TIMEZONE", "UTC")
        }
        self.ui_mode = "production"

    async def initialize(self, ui_mode: str = "production"):
        if self.initialized and self.ui_mode == ui_mode:
            return

        logger.info(f"Initializing Unified Calendar System in {ui_mode} mode...")

        try:
            self.ui_mode = ui_mode

            if not self.config["openai_api_key"]:
                raise ValueError("OpenAI API key not found in environment variables")

            openai.api_key = self.config["openai_api_key"]
            openai.base_url = self.config["openai_api_base"]
            self.openai_client = openai

            self.error_manager = ErrorRetryManager(self.config)

            memory_config = self.config.copy()
            memory_config.update({
                "redis_url": os.getenv("REDIS_URL"),
                "postgres_url": os.getenv("POSTGRES_URL"),
                "chroma_path": os.getenv("CHROMA_PATH", "./chroma_db"),
                "use_pgvector": os.getenv("USE_PGVECTOR", "true").lower() == "true"
            })
            self.memory_manager = MemoryManager(memory_config)

            await self._initialize_agents()
            await self._initialize_adapters()
            await self._initialize_orchestrator()

            self.initialized = True
            logger.info(f"Unified Calendar System initialized successfully in {ui_mode} mode")

        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise

    async def _initialize_agents(self):
        nlu_settings = {
            "model": self.config["openai_model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"]
        }
        self.nlu_agent = NLUParsingAgent(self.openai_client, nlu_settings)

        scheduler_settings = {"timezone": self.config["user_timezone"]}
        self.scheduler_agent = SchedulerAgent(scheduler_settings)

        if self.ui_mode in ["production", "ml_enhanced", "debug"]:
            ml_settings = {"model_dir": "./models"}
            self.ml_agent = MLSuggestionAgent(ml_settings)

            feedback_settings = {"feedback_delay_hours": 1}
            self.feedback_agent = FeedbackAgent(feedback_settings)
        else:
            self.ml_agent = None
            self.feedback_agent = None

    async def _initialize_adapters(self):
        google_config = {
            "client_secrets_file": os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
            "token_file": os.getenv("GOOGLE_TOKEN_FILE", "token.json")
        }
        self.google_adapter = GoogleCalendarAdapter(google_config)

    async def _initialize_orchestrator(self):
        orchestrator_settings = {
            "ui_mode": self.ui_mode,
            "enable_memory": self.ui_mode in ["production", "debug"],
            "enable_ml": self.ui_mode in ["production", "ml_enhanced", "debug"],
            "enable_feedback": self.ui_mode in ["production", "ml_enhanced", "debug"]
        }

        self.orchestrator = OrchestratorAgent(orchestrator_settings)
        self.orchestrator.register_agent(self.nlu_agent)
        self.orchestrator.register_agent(self.scheduler_agent)
        if self.ml_agent:
            self.orchestrator.register_agent(self.ml_agent)
        if self.feedback_agent:
            self.orchestrator.register_agent(self.feedback_agent)
        self.orchestrator.register_tool_adapter(self.google_adapter)

        if self.ui_mode in ["production", "debug"]:
            self.orchestrator.set_memory_manager(self.memory_manager)

    async def process_user_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("System not initialized")

        message = AgentMessage(
            recipient="orchestrator",
            sender="user",
            message_type="request",
            body={
                "request": request,
                "ui_mode": self.ui_mode,
                "context": context or {}
            }
        )

        try:
            response = await self.orchestrator.process_message(message)
            return {
                "success": response.success,
                "data": response.data,
                "error": response.error,
                "confidence": response.confidence,
                "suggestions": response.suggestions,
                "metadata": response.metadata
            }
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "confidence": 0.0,
                "suggestions": [],
                "metadata": {"error_type": type(e).__name__}
            }

# Export global instance so test files can import it
system = UnifiedCalendarSystem()

# Optional Streamlit interface
if __name__ == "__main__":
    import streamlit as st

    async def main():
        st.title("🧠 Pat's Calendar Agent")
        st.markdown("Enter a request to schedule or view calendar info")

        request = st.text_input("What do you want me to do?")
        if request:
            with st.spinner("Thinking..."):
                await system.initialize("debug")
                result = await system.process_user_request(request)
                if result["success"]:
                    st.success(result["data"])
                else:
                    st.error(result["error"])

    asyncio.run(main())

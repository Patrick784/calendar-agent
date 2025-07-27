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
    """
    Unified Calendar System that routes all requests through OrchestratorAgent.
    
    Supports multiple UI modes:
    - Production: Full multi-agent system with all features
    - Minimal: Simple calendar operations with basic AI
    - ML-Enhanced: Advanced ML suggestions and learning
    - Debug: Full system with extensive debugging information
    """
    
    def __init__(self):
        self.initialized = False
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_api_base": os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
            "openai_model": "openai/gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.2,
            "user_timezone": os.getenv("USER_TIMEZONE", "UTC")
        }
        self.ui_mode = "production"  # Default mode
    
    async def initialize(self, ui_mode: str = "production"):
        """Initialize the system with specified UI mode"""
        
        if self.initialized and self.ui_mode == ui_mode:
            return
            
        logger.info(f"Initializing Unified Calendar System in {ui_mode} mode...")
        
        try:
            self.ui_mode = ui_mode
            
            # Initialize OpenAI client
            if not self.config["openai_api_key"]:
                raise ValueError("OpenAI API key not found in environment variables")
            
            self.openai_client = openai.OpenAI(
                api_key=self.config["openai_api_key"],
                base_url=self.config["openai_api_base"]
            )
            
            # Initialize error manager
            self.error_manager = ErrorRetryManager(self.config)
            
            # Initialize memory manager (optional dependencies handled gracefully)
            memory_config = self.config.copy()
            memory_config.update({
                "redis_url": os.getenv("REDIS_URL"),
                "postgres_url": os.getenv("POSTGRES_URL"), 
                "chroma_path": os.getenv("CHROMA_PATH", "./chroma_db"),
                "use_pgvector": os.getenv("USE_PGVECTOR", "true").lower() == "true"
            })
            self.memory_manager = MemoryManager(memory_config)
            
            # Initialize specialized agents based on UI mode
            await self._initialize_agents()
            
            # Initialize tool adapters
            await self._initialize_adapters()
            
            # Initialize orchestrator and connect everything
            await self._initialize_orchestrator()
            
            self.initialized = True
            logger.info(f"Unified Calendar System initialized successfully in {ui_mode} mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise
    
    async def _initialize_agents(self):
        """Initialize agents based on UI mode"""
        
        # Core agents (always needed)
        nlu_settings = {
            "model": self.config["openai_model"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"]
        }
        self.nlu_agent = NLUParsingAgent(self.openai_client, nlu_settings)
        
        # Scheduler agent
        scheduler_settings = {"timezone": self.config["user_timezone"]}
        self.scheduler_agent = SchedulerAgent(scheduler_settings)
        
        # ML and Feedback agents (for ML-enhanced and production modes)
        if self.ui_mode in ["production", "ml_enhanced", "debug"]:
            ml_settings = {"model_dir": "./models"}
            self.ml_agent = MLSuggestionAgent(ml_settings)
            
            feedback_settings = {"feedback_delay_hours": 1}
            self.feedback_agent = FeedbackAgent(feedback_settings)
        else:
            self.ml_agent = None
            self.feedback_agent = None
    
    async def _initialize_adapters(self):
        """Initialize tool adapters"""
        
        # Google Calendar adapter
        google_config = {
            "client_secrets_file": os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
            "token_file": os.getenv("GOOGLE_TOKEN_FILE", "token.json")
        }
        self.google_adapter = GoogleCalendarAdapter(google_config)
    
    async def _initialize_orchestrator(self):
        """Initialize orchestrator and wire up all components"""
        
        orchestrator_settings = {
            "ui_mode": self.ui_mode,
            "enable_memory": self.ui_mode in ["production", "debug"],
            "enable_ml": self.ui_mode in ["production", "ml_enhanced", "debug"],
            "enable_feedback": self.ui_mode in ["production", "ml_enhanced", "debug"]
        }
        
        self.orchestrator = OrchestratorAgent(orchestrator_settings)
        
        # Register agents
        self.orchestrator.register_agent(self.nlu_agent)
        self.orchestrator.register_agent(self.scheduler_agent)
        
        if self.ml_agent:
            self.orchestrator.register_agent(self.ml_agent)
        if self.feedback_agent:
            self.orchestrator.register_agent(self.feedback_agent)
        
        # Register tool adapters
        self.orchestrator.register_tool_adapter(self.google_adapter)
        
        # Set memory manager
        if self.ui_mode in ["production", "debug"]:
            self.orchestrator.set_memory_manager(self.memory_manager)
    
    async def process_user_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user request through the orchestrator"""
        
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        # Create agent message
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
            # Process through orchestrator
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


# Global system instance
system = UnifiedCalendarSystem()

def main():
    """Main Streamlit application with unified interface"""
    
    st.set_page_config(
        page_title="Unified Calendar Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    # Mode selection in sidebar
    with st.sidebar:
        st.header("🎛️ System Configuration")
        
        ui_mode = st.selectbox(
            "Select UI Mode:",
            ["production", "minimal", "ml_enhanced", "debug"],
            index=0,
            help="""
            • Production: Full multi-agent system
            • Minimal: Simple calendar operations  
            • ML-Enhanced: Advanced ML suggestions
            • Debug: Full system with debug info
            """
        )
        
        st.markdown("---")
        
        # Mode descriptions
        mode_descriptions = {
            "production": "🏢 Full multi-agent system with memory, ML, and feedback",
            "minimal": "⚡ Simple calendar operations with basic AI",
            "ml_enhanced": "🧠 Advanced ML suggestions and learning",
            "debug": "🔍 Full system with extensive debugging"
        }
        
        st.info(mode_descriptions[ui_mode])
        
        # Show debug options for debug mode
        if ui_mode == "debug":
            st.markdown("### Debug Options")
            show_agent_communication = st.checkbox("Show Agent Communication", False)
            show_memory_state = st.checkbox("Show Memory State", False)
            show_ml_predictions = st.checkbox("Show ML Predictions", False)
        else:
            show_agent_communication = False
            show_memory_state = False
            show_ml_predictions = False
    
    # Main interface
    st.title("🤖 Unified Calendar Agent")
    st.markdown(f"*Running in {ui_mode.title()} mode*")
    st.markdown("---")
    
    # Initialize session state
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
        st.session_state.current_mode = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Check if mode changed
    if st.session_state.current_mode != ui_mode:
        st.session_state.system_initialized = False
        st.session_state.current_mode = ui_mode
        st.session_state.chat_history = []  # Clear history on mode change
    
    # System initialization
    if not st.session_state.system_initialized:
        with st.spinner(f"Initializing system in {ui_mode} mode..."):
            try:
                asyncio.run(system.initialize(ui_mode))
                st.session_state.system_initialized = True
                st.success(f"✅ System initialized in {ui_mode} mode!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {str(e)}")
                st.stop()
    
    # System status in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        if st.button("Check System Health"):
            with st.spinner("Checking system health..."):
                try:
                    health_status = asyncio.run(system.orchestrator.health_check())
                    if health_status["status"] == "healthy":
                        st.success("✅ System is healthy")
                    else:
                        st.warning(f"⚠️ System status: {health_status['status']}")
                    
                    if ui_mode == "debug":
                        st.json(health_status)
                        
                except Exception as e:
                    st.error(f"❌ Health check failed: {str(e)}")
    
    # Main chat interface
    st.subheader("💬 Chat with your calendar agent")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Enter your request:",
            placeholder="e.g., 'Schedule a meeting tomorrow at 2pm' or 'Show my events this week'",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    # Process user input
    if send_button and user_input:
        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Process through orchestrator
        with st.spinner("Processing your request..."):
            try:
                context = {
                    "ui_mode": ui_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": st.session_state.get("session_id", "default")
                }
                
                response = asyncio.run(system.process_user_request(user_input, context))
                
                # Add assistant response to history
                assistant_message = {
                    "role": "assistant",
                    "content": response["data"] if response["success"] else response["error"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": response["success"],
                    "confidence": response["confidence"],
                    "suggestions": response.get("suggestions", [])
                }
                
                if ui_mode == "debug":
                    assistant_message["debug_info"] = response.get("metadata", {})
                
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                
                # Add error to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "confidence": 0.0
                })
        
        # Clear input
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        
        # Show last 10 messages
        recent_messages = st.session_state.chat_history[-10:]
        
        for message in reversed(recent_messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    if ui_mode == "debug":
                        st.caption(f"Sent at: {message['timestamp']}")
            
            else:  # assistant
                with st.chat_message("assistant"):
                    if message["success"]:
                        st.write(message["content"])
                        
                        # Show confidence if available
                        if message.get("confidence", 0) > 0:
                            confidence_pct = int(message["confidence"] * 100)
                            st.progress(message["confidence"], text=f"Confidence: {confidence_pct}%")
                        
                        # Show suggestions if available
                        if message.get("suggestions"):
                            with st.expander("💡 Suggestions"):
                                for suggestion in message["suggestions"]:
                                    st.write(f"• {suggestion}")
                    
                    else:
                        st.error(message["content"])
                    
                    if ui_mode == "debug" and message.get("debug_info"):
                        with st.expander("🔍 Debug Info"):
                            st.json(message["debug_info"])
                    
                    if ui_mode == "debug":
                        st.caption(f"Responded at: {message['timestamp']}")
    
    # Mode-specific features
    if ui_mode == "ml_enhanced" and st.session_state.chat_history:
        with st.expander("🧠 ML Insights"):
            st.info("ML suggestions and learning insights would appear here")
    
    if ui_mode == "debug":
        # Debug information
        with st.expander("🔍 System Debug Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Agent Status")
                if hasattr(system, 'orchestrator') and system.orchestrator:
                    st.write(f"Orchestrator: ✅ Active")
                    st.write(f"Registered Agents: {len(system.orchestrator.agents)}")
                    st.write(f"Tool Adapters: {len(system.orchestrator.tool_adapters)}")
            
            with col2:
                st.subheader("Configuration")
                st.json({
                    "UI Mode": ui_mode,
                    "OpenAI Model": system.config["openai_model"],
                    "Temperature": system.config["temperature"],
                    "Max Tokens": system.config["max_tokens"]
                })

if __name__ == "__main__":
    main() 
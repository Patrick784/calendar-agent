"""
Base Agent Architecture

Defines the core interfaces and abstract classes for the multi-agent system.
All agents inherit from BaseAgent and communicate via AgentMessage/AgentResponse.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import uuid
import logging

@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    recipient: str
    sender: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_type: str = "request"  # request, response, notification
    body: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=high

@dataclass 
class AgentResponse:
    """Standardized response format from agents"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Provides common functionality:
    - Message handling and routing
    - Logging and observability 
    - Error handling
    - Settings management
    """
    
    def __init__(self, agent_id: str, settings: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.settings = settings or {}
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self._message_history: List[AgentMessage] = []
        self._metrics = {
            "requests_handled": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process an incoming message and return a response.
        Each agent implements this method based on their specialization.
        """
        pass
    
    def update_metrics(self, response_time: float, success: bool):
        """Update agent performance metrics"""
        self._metrics["requests_handled"] += 1
        if not success:
            self._metrics["errors"] += 1
        
        # Update rolling average response time
        total_requests = self._metrics["requests_handled"]
        current_avg = self._metrics["avg_response_time"]
        self._metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current agent metrics"""
        return self._metrics.copy()
    
    def log_message(self, message: AgentMessage, level: str = "info"):
        """Log an agent message with structured format"""
        log_data = {
            "agent_id": self.agent_id,
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "type": message.message_type
        }
        getattr(self.logger, level)(f"Message processed: {log_data}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "metrics": self.get_metrics(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }

class ToolAdapter(ABC):
    """
    Abstract base class for external tool adapters (Calendar APIs, etc.).
    Provides standardized interface for external service integration.
    """
    
    def __init__(self, adapter_id: str, config: Dict[str, Any] = None):
        self.adapter_id = adapter_id
        self.config = config or {}
        self.logger = logging.getLogger(f"adapter.{adapter_id}")
    
    @abstractmethod
    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool action with given parameters"""
        pass
    
    @abstractmethod 
    async def health_check(self) -> bool:
        """Check if the external service is available"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's action schema for the orchestrator"""
        return {
            "adapter_id": self.adapter_id,
            "actions": self._get_available_actions(),
            "description": self._get_description()
        }
    
    @abstractmethod
    def _get_available_actions(self) -> List[str]:
        """Return list of available actions this adapter supports"""
        pass
    
    @abstractmethod
    def _get_description(self) -> str:
        """Return description of what this adapter does"""
        pass 
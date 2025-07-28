"""
Calendar Agent

Specialized natural language calendar agent that parses user queries and generates 
structured output for downstream agents. Supports both event management and 
memory-based contextual queries.

Intent types: create_event, read_events, query_memory, delete_event, unknown
"""

import re
import json
import openai
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import dateutil.parser
from dataclasses import dataclass

from .base import BaseAgent, AgentMessage, AgentResponse

# Import security module for PII sanitization
try:
    from src.security import sanitize_input, validate_input_safety, setup_secure_logging
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

@dataclass
class CalendarQueryResult:
    """Structured result from calendar query parsing"""
    query: str
    intent: str  # create_event, read_events, query_memory, delete_event, unknown
    target_info: str  # Key nouns like dentist, barber, Sarah, soccer
    timeframe: str  # Specific date or relative timeframe
    ambiguities: List[str]  # Unclear elements that need clarification
    test_passed: bool  # Whether the extracted meaning passes validation

class CalendarAgent(BaseAgent):
    """
    Specialized calendar agent for parsing natural language queries.
    
    Capabilities:
    - Parse event creation/update/delete requests
    - Handle memory-based contextual queries
    - Extract target_info, timeframe, and identify ambiguities
    - Generate structured output for downstream agents
    """
    
    def __init__(self, openai_client: openai.OpenAI, settings: Dict[str, Any] = None):
        super().__init__("calendar_agent", settings)
        self.openai_client = openai_client
        self.model = settings.get("model", "openai/gpt-4o")
        self.max_tokens = settings.get("max_tokens", 500)
        self.temperature = settings.get("temperature", 0.1)
        
        # Set up secure logging if security module is available
        if SECURITY_AVAILABLE:
            self.logger = setup_secure_logging(f"agent.{self.agent_id}")
            self.logger.info("Calendar Agent initialized with secure logging")
        
        # Regex patterns for fallback parsing
        self._intent_patterns = {
            "create_event": [
                r"(add|create|schedule|book|set up)\s+(meeting|appointment|event)",
                r"(meeting|appointment|event)\s+with",
                r"(add|create|schedule)\s+(.+?)\s+(next|this|on)",
                r"(book|schedule)\s+(.+?)\s+(appointment|meeting)",
            ],
            "read_events": [
                r"(show|find|get|list|what)\s+(my|the)\s+(meetings|appointments|events)",
                r"(when|what)\s+(do\s+I\s+have|is\s+my)",
                r"(check|see)\s+(my|the)\s+(schedule|calendar)",
                r"(show|find|list)\s+(.+?)\s+(appointments|meetings)",
            ],
            "query_memory": [
                r"(when\s+last|last\s+time)\s+(did\s+I|I\s+went|I\s+had)",
                r"(remember|recall|when)\s+(did\s+I)\s+(.+?)\s+(last|before)",
                r"(how\s+long\s+ago)\s+(did\s+I)",
                r"(when\s+did\s+I\s+last)",
                r"(when\s+was\s+my\s+last)",
                r"(when\s+last\s+did\s+I)",
                r"when\s+last\s+did\s+I\s+go",
                r"when\s+last\s+did\s+I",
                r"last\s+time\s+I\s+went",
                r"when\s+last",
            ],
            "delete_event": [
                r"(cancel|delete|remove)\s+(meeting|appointment|event)",
                r"(cancel|delete|remove)\s+(.+?)\s+(meeting|appointment)",
                r"(cancel|delete|remove)\s+(.+?)\s+(game|event)",
            ]
        }
        
        self._timeframe_patterns = {
            "specific_date": [
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # MM/DD/YYYY
                r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
            ],
            "relative": [
                r"(today|tomorrow|yesterday)",
                r"(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                r"(next|this|last)\s+(week|month|year)",
                r"(in\s+\d+\s+(days?|weeks?|months?))",
                r"(past|last\s+\d+\s+(days?|weeks?|months?))",
            ]
        }
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Parse natural language calendar query into structured output.
        
        Returns structured data with:
        - intent: create_event, read_events, query_memory, delete_event, unknown
        - target_info: key nouns extracted
        - timeframe: specific or relative time reference
        - ambiguities: unclear elements
        - test_passed: validation result
        """
        start_time = datetime.now(timezone.utc)
        
        # Handle specific actions from orchestrator
        if "action" in message.body:
            return await self._handle_action_request(message)
        
        text = message.body.get("text", "")
        
        if not text.strip():
            return AgentResponse(
                success=False,
                error="Empty query text"
            )
        
        # Security validation and PII sanitization
        if SECURITY_AVAILABLE:
            # Validate input safety
            is_safe, safety_issues = validate_input_safety(text)
            if not is_safe:
                self.logger.warning(f"Unsafe input detected: {safety_issues}")
                return AgentResponse(
                    success=False,
                    error=f"Input safety validation failed: {safety_issues}"
                )
            
            # Sanitize PII
            text = sanitize_input(text)
        
        try:
            # Try LLM parsing first
            result = await self._parse_with_llm(text, message.context)
            if result.success:
                return result
            
            # Fallback to regex parsing
            result = await self._parse_with_regex(text, message.context)
            if result.success:
                return result
            
            # If both fail, return unknown intent
            return self._create_unknown_result(text)
            
        except Exception as e:
            self.logger.error(f"Error parsing calendar query: {e}")
            return AgentResponse(
                success=False,
                error=f"Failed to parse calendar query: {str(e)}"
            )
        finally:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_metrics(response_time, True)
    
    async def _parse_with_llm(self, text: str, context: Dict[str, Any]) -> AgentResponse:
        """Use OpenAI function calling to parse the calendar query"""
        
        # Check if we have a valid OpenAI client
        if not hasattr(self.openai_client, 'chat') or not hasattr(self.openai_client.chat, 'completions'):
            self.logger.warning("Invalid OpenAI client - falling back to regex parsing")
            return AgentResponse(success=False, error="Invalid OpenAI client")
        
        # Define the function schema for structured extraction
        function_schema = {
            "name": "parse_calendar_query",
            "description": "Parse natural language calendar queries into structured data",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["create_event", "read_events", "query_memory", "delete_event", "unknown"],
                        "description": "The primary intent of the query"
                    },
                    "target_info": {
                        "type": "string",
                        "description": "Key nouns like dentist, barber, Sarah, soccer, meeting, appointment"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Specific date (YYYY-MM-DD) or relative timeframe (next Friday, last week, past)"
                    },
                    "ambiguities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of unclear elements that need clarification"
                    },
                    "test_passed": {
                        "type": "boolean",
                        "description": "Whether the extracted meaning passes validation"
                    }
                },
                "required": ["intent", "target_info", "timeframe", "ambiguities", "test_passed"]
            }
        }
        
        current_time = datetime.now(timezone.utc).isoformat()
        
        system_prompt = f"""You are a specialized calendar agent that parses natural language queries about calendar events and generates structured output.

Current time: {current_time}

Your role is to determine:
1. **intent**: One of [create_event, read_events, query_memory, delete_event, unknown]
   - create_event: User wants to add/schedule something
   - read_events: User wants to see/find existing events
   - query_memory: User asks about past events (when last did I...)
   - delete_event: User wants to cancel/remove something
   - unknown: Cannot determine intent

2. **target_info**: Extract key nouns like "dentist", "barber", "Sarah", "soccer", "meeting", "appointment"

3. **timeframe**: 
   - Specific dates: "2025-07-25"
   - Relative: "next Friday", "last week", "past", "tomorrow"

4. **ambiguities**: List anything unclear that needs clarification

5. **test_passed**: True if the extracted meaning makes sense and is actionable

Examples:
- "When last did I go to the barber?" → intent: query_memory, target_info: barber, timeframe: past, ambiguities: [], test_passed: true
- "Add meeting with Alex next Thursday at 1pm" → intent: create_event, target_info: meeting, timeframe: next Thursday, ambiguities: ["exact time"], test_passed: true
- "Show my dentist appointments" → intent: read_events, target_info: dentist, timeframe: future, ambiguities: [], test_passed: true
- "Cancel the soccer game" → intent: delete_event, target_info: soccer, timeframe: unspecified, ambiguities: ["which soccer game"], test_passed: false

Be precise and conservative. If unsure, set test_passed to false and note ambiguities."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                functions=[function_schema],
                function_call={"name": "parse_calendar_query"},
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "parse_calendar_query":
                parsed_data = json.loads(function_call.arguments)
                
                # Create structured result
                result = CalendarQueryResult(
                    query=text,
                    intent=parsed_data.get("intent", "unknown"),
                    target_info=parsed_data.get("target_info", ""),
                    timeframe=parsed_data.get("timeframe", ""),
                    ambiguities=parsed_data.get("ambiguities", []),
                    test_passed=parsed_data.get("test_passed", False)
                )
                
                return AgentResponse(
                    success=True,
                    data=result,
                    confidence=0.9 if result.test_passed else 0.5
                )
            
            return AgentResponse(
                success=False,
                error="LLM did not return valid function call"
            )
            
        except Exception as e:
            self.logger.error(f"LLM parsing failed: {e}")
            return AgentResponse(
                success=False,
                error=f"LLM parsing error: {str(e)}"
            )
    
    async def _parse_with_regex(self, text: str, context: Dict[str, Any]) -> AgentResponse:
        """Fallback regex-based parsing when LLM fails"""
        
        text_lower = text.lower()
        
        # Determine intent
        intent = "unknown"
        for intent_type, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intent = intent_type
                    self.logger.info(f"Matched pattern '{pattern}' for intent '{intent_type}'")
                    break
            if intent != "unknown":
                break
        
        # Extract target_info (key nouns)
        target_info = self._extract_target_info(text)
        
        # Extract timeframe
        timeframe = self._extract_timeframe(text)
        
        # Identify ambiguities
        ambiguities = self._identify_ambiguities(text, intent, target_info, timeframe)
        
        # Determine if test passes
        test_passed = self._validate_parsing(intent, target_info, timeframe, ambiguities)
        
        result = CalendarQueryResult(
            query=text,
            intent=intent,
            target_info=target_info,
            timeframe=timeframe,
            ambiguities=ambiguities,
            test_passed=test_passed
        )
        
        return AgentResponse(
            success=True,
            data=result,
            confidence=0.6 if test_passed else 0.3
        )
    
    def _extract_target_info(self, text: str) -> str:
        """Extract key nouns from the text"""
        # Common calendar-related nouns
        calendar_nouns = [
            "meeting", "appointment", "event", "call", "lunch", "dinner",
            "dentist", "doctor", "barber", "haircut", "soccer", "football",
            "gym", "workout", "class", "lesson", "interview", "presentation",
            "game"
        ]
        
        text_lower = text.lower()
        
        # Look for specific nouns
        for noun in calendar_nouns:
            if noun in text_lower:
                return noun
        
        # Look for names (capitalized words that might be names)
        name_pattern = r'\b[A-Z][a-z]+\b'
        names = re.findall(name_pattern, text)
        if names:
            return names[0]
        
        # Look for any noun-like words
        noun_pattern = r'\b\w+(?:ing|ment|tion|sion)\b'
        nouns = re.findall(noun_pattern, text_lower)
        if nouns:
            return nouns[0]
        
        return "unspecified"
    
    def _extract_timeframe(self, text: str) -> str:
        """Extract timeframe from text"""
        text_lower = text.lower()
        
        # Check for specific dates
        for pattern in self._timeframe_patterns["specific_date"]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Check for relative timeframes
        for pattern in self._timeframe_patterns["relative"]:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(0)
        
        # Check for past queries (memory-based)
        if any(word in text_lower for word in ["last", "past", "when", "ago"]):
            return "past"
        
        # Check for future queries
        if any(word in text_lower for word in ["next", "tomorrow", "upcoming"]):
            return "future"
        
        return "unspecified"
    
    def _identify_ambiguities(self, text: str, intent: str, target_info: str, timeframe: str) -> List[str]:
        """Identify unclear elements in the query"""
        ambiguities = []
        
        if target_info == "unspecified":
            ambiguities.append("what event/person to look for")
        
        if timeframe == "unspecified":
            if intent in ["create_event", "read_events"]:
                ambiguities.append("when (date/time)")
            elif intent == "query_memory":
                ambiguities.append("time period to search")
        
        if intent == "unknown":
            ambiguities.append("intent unclear")
        
        # Check for multiple possible interpretations
        if text.count("with") > 1:
            ambiguities.append("multiple people/entities mentioned")
        
        if text.count("at") > 1:
            ambiguities.append("multiple time references")
        
        return ambiguities
    
    def _validate_parsing(self, intent: str, target_info: str, timeframe: str, ambiguities: List[str]) -> bool:
        """Validate if the parsed data makes sense"""
        
        # Basic validation
        if intent == "unknown":
            return False
        
        if target_info == "unspecified" and intent != "read_events":
            return False
        
        # Too many ambiguities
        if len(ambiguities) > 2:
            return False
        
        # Specific validation for each intent
        if intent == "query_memory":
            if timeframe == "unspecified":
                return False
        
        if intent == "create_event":
            if target_info == "unspecified":
                return False
        
        return True
    
    def _create_unknown_result(self, text: str) -> AgentResponse:
        """Create result for completely unparseable queries"""
        result = CalendarQueryResult(
            query=text,
            intent="unknown",
            target_info="unspecified",
            timeframe="unspecified",
            ambiguities=["cannot parse query"],
            test_passed=False
        )
        
        return AgentResponse(
            success=True,
            data=result,
            confidence=0.1
        )
    
    async def _handle_action_request(self, message: AgentMessage) -> AgentResponse:
        """Handle specific action requests from orchestrator"""
        action = message.body.get("action")
        
        if action == "health_check":
            return AgentResponse(
                success=True,
                data=await self.health_check()
            )
        
        elif action == "get_schema":
            return AgentResponse(
                success=True,
                data={
                    "agent_id": self.agent_id,
                    "capabilities": [
                        "parse_calendar_queries",
                        "extract_intent",
                        "identify_target_info",
                        "extract_timeframe",
                        "detect_ambiguities"
                    ],
                    "supported_intents": [
                        "create_event",
                        "read_events", 
                        "query_memory",
                        "delete_event",
                        "unknown"
                    ]
                }
            )
        
        return AgentResponse(
            success=False,
            error=f"Unknown action: {action}"
        ) 
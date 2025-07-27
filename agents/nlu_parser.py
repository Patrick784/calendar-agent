"""
NLU / Parsing Agent

Specialized agent that parses natural language task requests into structured 
objects with fields such as title, datetime, duration, importance, recurrence.
Primary method: LLM function-calling; fallback: regex parser.
"""

import re
import json
import openai
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import dateutil.parser

from .base import BaseAgent, AgentMessage, AgentResponse

# Import security module for PII sanitization
try:
    from src.security import sanitize_input, validate_input_safety, setup_secure_logging
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

class NLUParsingAgent(BaseAgent):
    """
    Natural Language Understanding agent for parsing user requests.
    
    Capabilities:
    - Extract structured data from natural language
    - Identify intent (create, find, update, delete events)
    - Parse dates, times, durations, recurrence patterns
    - Handle ambiguous requests with clarifying questions
    - Fallback to regex parsing when LLM fails
    """
    
    def __init__(self, openai_client: openai.OpenAI, settings: Dict[str, Any] = None):
        super().__init__("nlu_parser", settings)
        self.openai_client = openai_client
        self.model = settings.get("model", "openai/gpt-4o")  # Updated for OpenRouter compatibility
        self.max_tokens = settings.get("max_tokens", 500)
        self.temperature = settings.get("temperature", 0.2)
        
        # Set up secure logging if security module is available
        if SECURITY_AVAILABLE:
            self.logger = setup_secure_logging(f"agent.{self.agent_id}")
            self.logger.info("NLU Parser initialized with secure logging")
        
        # Regex patterns for fallback parsing
        self._date_patterns = [
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # MM/DD/YYYY or MM-DD-YYYY
            r"(today|tomorrow|yesterday)",
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(next|this|last)\s+(week|month|year)",
            r"(in\s+\d+\s+(days?|weeks?|months?))"
        ]
        
        self._time_patterns = [
            r"(\d{1,2}:\d{2}\s*(?:am|pm)?)",  # 3:30pm, 15:30
            r"(\d{1,2}\s*(?:am|pm))",        # 3pm, 3am
            r"(noon|midnight)"
        ]
        
        self._duration_patterns = [
            r"(\d+\s*(?:hours?|hrs?|h))",
            r"(\d+\s*(?:minutes?|mins?|m))",
            r"(all\s+day|full\s+day)"
        ]
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Parse natural language request into structured data.
        
        Flow:
        1. Check if this is a specific action request from orchestrator
        2. Validate input safety and sanitize PII
        3. Try LLM function calling first
        4. If that fails, use regex fallback
        5. If still unclear, ask clarifying questions
        """
        start_time = datetime.now(timezone.utc)
        
        # Handle specific actions from orchestrator
        if "action" in message.body:
            return await self._handle_action_request(message)
        
        text = message.body.get("text", "")
        
        if not text.strip():
            return AgentResponse(
                success=False,
                error="Empty request text"
            )
        
        # Security validation and PII sanitization
        if SECURITY_AVAILABLE:
            # Validate input safety
            is_safe, safety_issues = validate_input_safety(text)
            if not is_safe:
                self.logger.warning(f"Unsafe input detected: {safety_issues}")
                return AgentResponse(
                    success=False,
                    error="Input validation failed for security reasons",
                    suggestions=["Please review your input for potential security issues"]
                )
            
            # Sanitize PII for logging and processing (but preserve original for LLM)
            sanitized_text = sanitize_input(text)
            self.logger.info(f"Processing sanitized request: {sanitized_text}")
        else:
            self.logger.info(f"Processing request: {text}")
        
        try:
            # First attempt: LLM function calling
            llm_result = await self._parse_with_llm(text, message.context)
            
            if llm_result.success and llm_result.confidence > 0.7:
                response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.update_metrics(response_time, True)
                return llm_result
            
            # Fallback: Regex parsing
            self.logger.info("LLM parsing failed or low confidence, trying regex fallback")
            regex_result = await self._parse_with_regex(text, message.context)
            
            if regex_result.success:
                response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.update_metrics(response_time, True)
                return regex_result
            
            # Last resort: Ask for clarification
            clarification_result = self._generate_clarification_request(text)
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return clarification_result
            
        except Exception as e:
            self.logger.error(f"NLU parsing error: {str(e)}")
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return AgentResponse(
                success=False,
                error=f"Parsing failed: {str(e)}"
            )
    
    async def _parse_with_llm(self, text: str, context: Dict[str, Any]) -> AgentResponse:
        """Use OpenAI function calling to parse the request"""
        
        # Define the function schema for structured extraction
        function_schema = {
            "name": "extract_calendar_request",
            "description": "Extract structured information from a calendar-related request",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["create_event", "find_events", "update_event", "delete_event", 
                               "check_availability", "schedule_optimal_time", "unclear"],
                        "description": "The primary intent of the request"
                    },
                    "title": {
                        "type": "string",
                        "description": "Event title or subject"
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": "Start date and time in ISO format (e.g., 2024-03-15T14:30:00)"
                    },
                    "end_datetime": {
                        "type": "string", 
                        "description": "End date and time in ISO format"
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes if specified"
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location"
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional event details"
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Event importance level"
                    },
                    "recurrence": {
                        "type": "object",
                        "properties": {
                            "frequency": {"type": "string", "enum": ["daily", "weekly", "monthly", "yearly"]},
                            "interval": {"type": "integer", "description": "Repeat every N intervals"},
                            "until": {"type": "string", "description": "End date for recurrence"}
                        }
                    },
                    "search_criteria": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string"},
                                    "end": {"type": "string"}
                                }
                            },
                            "time_filter": {"type": "string"}
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence level in the parsing (0-1)"
                    },
                    "missing_info": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of missing required information"
                    }
                },
                "required": ["intent", "confidence"]
            }
        }
        
        current_time = datetime.now(timezone.utc).isoformat()
        user_context = context.get("user_timezone", "UTC")
        
        system_prompt = f"""You are a calendar assistant specialized in parsing natural language requests into structured data.

Current context:
- Current time: {current_time}
- User timezone: {user_context}

When parsing requests:
1. Be precise about dates and times
2. Use ISO format for all datetimes
3. If information is missing or ambiguous, note it in missing_info
4. Set confidence based on how clear the request is
5. For recurring events, extract the pattern details
6. For searches, identify keywords and date ranges

Examples:
"Meeting tomorrow at 3pm" → intent: create_event, start_datetime: [tomorrow 3pm in ISO], confidence: 0.8, missing_info: ["duration", "title"]
"Find my doctor appointments this week" → intent: find_events, search_criteria: {{keywords: ["doctor"], date_range: {{start: [this week start], end: [this week end]}}}}, confidence: 0.9
"""
        
        try:
            # Convert function schema to tools format
            tool_schema = {
                "type": "function",
                "function": function_schema
            }
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "extract_calendar_request"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls[0].function.name == "extract_calendar_request":
                try:
                    parsed_data = json.loads(tool_calls[0].function.arguments)
                    
                    # Validate and clean the parsed data
                    cleaned_data = self._validate_and_clean_parsed_data(parsed_data)
                    
                    return AgentResponse(
                        success=True,
                        data=cleaned_data,
                        confidence=parsed_data.get("confidence", 0.5),
                        metadata={"method": "llm_function_calling", "model": self.model}
                    )
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parsing error in LLM response: {str(e)}")
                    self.logger.error(f"Raw response: {tool_calls[0].function.arguments}")
                    return AgentResponse(
                        success=False,
                        error=f"Failed to parse LLM response as JSON: {str(e)}",
                        confidence=0.0
                    )
            else:
                return AgentResponse(
                    success=False,
                    error="LLM did not use function calling properly",
                    confidence=0.0
                )
                
        except Exception as e:
            self.logger.error(f"LLM parsing error: {str(e)}")
            return AgentResponse(
                success=False,
                error=f"LLM parsing failed: {str(e)}",
                confidence=0.0
            )
    
    async def _parse_with_regex(self, text: str, context: Dict[str, Any]) -> AgentResponse:
        """Fallback regex-based parsing for common patterns"""
        
        result = {
            "intent": "unclear",
            "confidence": 0.3,
            "missing_info": []
        }
        
        text_lower = text.lower()
        
        # Detect intent from keywords
        if any(word in text_lower for word in ["create", "add", "schedule", "book", "new"]):
            result["intent"] = "create_event"
        elif any(word in text_lower for word in ["find", "search", "show", "list", "get"]):
            result["intent"] = "find_events"
        elif any(word in text_lower for word in ["update", "change", "modify", "edit"]):
            result["intent"] = "update_event"
        elif any(word in text_lower for word in ["delete", "remove", "cancel"]):
            result["intent"] = "delete_event"
        elif any(word in text_lower for word in ["available", "free", "busy"]):
            result["intent"] = "check_availability"
        
        # Extract dates
        dates_found = []
        for pattern in self._date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates_found.extend(matches)
        
        # Extract times
        times_found = []
        for pattern in self._time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times_found.extend(matches)
        
        # Extract durations
        durations_found = []
        for pattern in self._duration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            durations_found.extend(matches)
        
        # Try to parse and combine date/time information
        if dates_found or times_found:
            try:
                # Use dateutil for fuzzy date/time parsing
                combined_datetime = self._combine_date_time(dates_found, times_found, context)
                if combined_datetime:
                    result["start_datetime"] = combined_datetime.isoformat()
                    result["confidence"] = 0.6
            except Exception as e:
                self.logger.warning(f"Date/time parsing failed: {str(e)}")
        
        # Extract potential event title (simple heuristic)
        if result["intent"] == "create_event":
            # Look for quoted text or text after "for" or "about"
            title_patterns = [
                r'"([^"]+)"',  # Quoted text
                r'for\s+([^,\n]+)',  # Text after "for"
                r'about\s+([^,\n]+)',  # Text after "about"
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["title"] = match.group(1).strip()
                    result["confidence"] = min(result["confidence"] + 0.2, 1.0)
                    break
        
        # If we found some useful information, return success
        if result["confidence"] > 0.5:
            return AgentResponse(
                success=True,
                data=result,
                confidence=result["confidence"],
                metadata={"method": "regex_parsing"}
            )
        else:
            return AgentResponse(
                success=False,
                error="Could not extract meaningful information with regex",
                confidence=result["confidence"],
                data=result
            )
    
    def _combine_date_time(self, dates: List[str], times: List[str], 
                          context: Dict[str, Any]) -> Optional[datetime]:
        """Combine date and time strings into a datetime object"""
        
        if not dates and not times:
            return None
        
        # Start with current time as base
        base_time = datetime.utcnow()
        
        # Parse date component
        target_date = base_time.date()
        if dates:
            try:
                # Use dateutil for flexible date parsing
                parsed_date = dateutil.parser.parse(dates[0], default=base_time)
                target_date = parsed_date.date()
            except Exception as e:
                self.logger.warning(f"Could not parse date '{dates[0]}': {str(e)}")
        
        # Parse time component
        target_time = base_time.time()
        if times:
            try:
                time_str = times[0]
                # Handle special cases
                if time_str.lower() == "noon":
                    target_time = datetime.strptime("12:00", "%H:%M").time()
                elif time_str.lower() == "midnight":
                    target_time = datetime.strptime("00:00", "%H:%M").time()
                else:
                    parsed_time = dateutil.parser.parse(time_str, default=base_time)
                    target_time = parsed_time.time()
            except Exception as e:
                self.logger.warning(f"Could not parse time '{times[0]}': {str(e)}")
        
        # Combine date and time
        try:
            combined = datetime.combine(target_date, target_time)
            return combined
        except Exception as e:
            self.logger.warning(f"Could not combine date and time: {str(e)}")
            return None
    
    def _validate_and_clean_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the parsed data from LLM"""
        
        cleaned = data.copy()
        
        # Validate datetime formats
        for field in ["start_datetime", "end_datetime"]:
            if field in cleaned and cleaned[field]:
                try:
                    # Ensure it's a valid ISO datetime
                    dt = dateutil.parser.isoparse(cleaned[field])
                    cleaned[field] = dt.isoformat()
                except Exception as e:
                    self.logger.warning(f"Invalid datetime format for {field}: {cleaned[field]}")
                    del cleaned[field]
                    if "missing_info" not in cleaned:
                        cleaned["missing_info"] = []
                    cleaned["missing_info"].append(field)
        
        # Validate duration
        if "duration_minutes" in cleaned:
            try:
                duration = int(cleaned["duration_minutes"])
                if duration <= 0 or duration > 24 * 60:  # Max 24 hours
                    raise ValueError("Duration out of reasonable range")
                cleaned["duration_minutes"] = duration
            except (ValueError, TypeError):
                del cleaned["duration_minutes"]
        
        # Ensure confidence is in valid range
        confidence = cleaned.get("confidence", 0.5)
        cleaned["confidence"] = max(0.0, min(1.0, confidence))
        
        # Clean up empty fields
        fields_to_remove = []
        for key, value in cleaned.items():
            if value is None or value == "" or (isinstance(value, list) and not value):
                fields_to_remove.append(key)
        
        for key in fields_to_remove:
            del cleaned[key]
        
        return cleaned
    
    def _generate_clarification_request(self, text: str) -> AgentResponse:
        """Generate a clarifying question when parsing fails"""
        
        suggestions = []
        
        # Analyze what might be missing based on the text
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["create", "add", "schedule"]):
            suggestions.extend([
                "What is the title of the event?",
                "When would you like to schedule it?",
                "How long should it be?"
            ])
        elif any(word in text_lower for word in ["find", "search", "show"]):
            suggestions.extend([
                "What kind of events are you looking for?",
                "What time period should I search?",
                "Any specific keywords to search for?"
            ])
        else:
            suggestions.extend([
                "Are you trying to create, find, or modify an event?",
                "Could you provide more details about what you need?"
            ])
        
        clarification_msg = "I couldn't fully understand your request. Could you please provide more details?"
        
        return AgentResponse(
            success=True,
            data={
                "intent": "clarification_needed",
                "message": clarification_msg,
                "original_text": text
            },
            suggestions=suggestions,
            confidence=0.0
        ) 

    async def _handle_action_request(self, message: AgentMessage) -> AgentResponse:
        """Handle specific action requests from orchestrator"""
        action = message.body.get("action")
        context = message.body.get("context", {})
        
        if action == "parse_search_criteria":
            # Return the search criteria that was already parsed
            search_data = context.get("search_criteria", {})
            return AgentResponse(
                success=True,
                data=search_data,
                confidence=0.9,
                metadata={"action": action, "method": "context_extraction"}
            )
        
        elif action == "validate_event_data":
            # Return the event data that was already parsed
            event_data = {
                "title": context.get("title", "Event"),
                "start_datetime": context.get("start_datetime"),
                "end_datetime": context.get("end_datetime"),
                "location": context.get("location"),
                "description": context.get("description")
            }
            return AgentResponse(
                success=True,
                data=event_data,
                confidence=0.9,
                metadata={"action": action, "method": "context_validation"}
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown action: {action}",
                confidence=0.0
            ) 
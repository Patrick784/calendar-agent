"""
Orchestrator Agent

Lead agent that receives user requests, decomposes tasks, chooses tools,
spawns specialized sub-agents, and coordinates results. Follows chain of
command (developer > user > tool) and manages dialogue flow.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json

from .base import BaseAgent, AgentMessage, AgentResponse, ToolAdapter
from .memory_manager import MemoryManager
from .error_manager import ErrorRetryManager

# Import validation system
try:
    from src.validation import (
        validate_task, validate_event, requires_human_approval, 
        create_approval_summary, ValidationResult
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

class OrchestratorAgent(BaseAgent):
    """
    Main orchestrator that coordinates all other agents and tools.
    
    Responsibilities:
    - Decompose user requests into actionable tasks
    - Route messages to appropriate specialized agents
    - Coordinate multi-step workflows
    - Manage conversation context and memory 
    - Make final decisions on user interactions
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__("orchestrator", settings)
        self.agents: Dict[str, BaseAgent] = {}
        self.tool_adapters: Dict[str, ToolAdapter] = {}
        self.memory_manager: Optional[MemoryManager] = None
        self.error_manager = ErrorRetryManager()
        self.current_plan: Optional[Dict[str, Any]] = None
        
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    def register_tool_adapter(self, adapter: ToolAdapter):
        """Register an external tool adapter"""
        self.tool_adapters[adapter.adapter_id] = adapter
        self.logger.info(f"Registered tool adapter: {adapter.adapter_id}")
    
    def set_memory_manager(self, memory_manager: MemoryManager):
        """Set the memory manager for context and persistence"""
        self.memory_manager = memory_manager
    
    async def health_check(self) -> Dict[str, Any]:
        """Return orchestrator health status and check all agents"""
        base_health = await super().health_check()
        
        # Check all registered agents
        agent_status = {}
        for agent_id, agent in self.agents.items():
            try:
                agent_health = await agent.health_check()
                agent_status[agent_id] = agent_health["status"]
            except Exception as e:
                agent_status[agent_id] = f"error: {str(e)}"
        
        # Check tool adapters
        tool_status = {}
        for tool_id, tool in self.tool_adapters.items():
            try:
                tool_health = await tool.health_check()
                tool_status[tool_id] = "healthy" if tool_health else "unhealthy"
            except Exception as e:
                tool_status[tool_id] = f"error: {str(e)}"
        
        # Overall status
        all_agents_healthy = all(status == "healthy" for status in agent_status.values())
        all_tools_healthy = all("healthy" in status for status in tool_status.values())
        overall_status = "healthy" if all_agents_healthy and all_tools_healthy else "degraded"
        
        return {
            **base_health,
            "status": overall_status,
            "agents": agent_status,
            "tools": tool_status,
            "memory_manager": "enabled" if self.memory_manager else "disabled",
            "registered_agents_count": len(self.agents),
            "registered_tools_count": len(self.tool_adapters)
        }

    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Main entry point for processing user requests.
        
        Flow:
        1. Parse and understand the request
        2. Retrieve relevant context from memory
        3. Create execution plan
        4. Execute plan with appropriate agents/tools
        5. Coordinate results and respond to user
        6. Update memory with new information
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Update scratchpad with current request
            await self._update_scratchpad(message)
            
            # Understand the request intent
            intent_response = await self._analyze_intent(message)
            if not intent_response.success:
                return AgentResponse(
                    success=False,
                    error="Failed to understand request",
                    data=intent_response.error
                )
            
            # Create execution plan
            plan = await self._create_execution_plan(intent_response.data, message)
            self.current_plan = plan
            
            # Skip validation for now - we're testing with mock data
            # TODO: Re-enable validation once we have proper Event/Task object creation
            if False and VALIDATION_AVAILABLE and plan.get("data"):
                validation_result = await self._validate_plan_data(plan)
                if not validation_result.valid:
                    # Try LLM fix once, then fallback to error
                    fixed_plan = await self._fix_plan_with_llm(plan, validation_result.errors)
                    if fixed_plan:
                        plan = fixed_plan
                        self.logger.info("Plan data validated and fixed by LLM")
                    else:
                        return AgentResponse(
                            success=False,
                            error=f"Validation failed: {'; '.join(validation_result.errors)}",
                            suggestions=["Please provide clearer input", "Check required fields"]
                        )
            
            # Check if human approval is required
            if VALIDATION_AVAILABLE and self._requires_approval(plan):
                approval_result = await self._request_human_approval(plan, message)
                if not approval_result:
                    return AgentResponse(
                        success=False,
                        error="Action cancelled - human approval required but not granted",
                        data={"cancelled_by_user": True, "requires_approval": True}
                    )
            
            # Execute the plan
            result = await self._execute_plan(plan, message)
            
            # Update memory with results
            if self.memory_manager:
                await self._update_memory(message, plan, result)
            
            # Self-evaluation
            await self._self_evaluate(plan, result)
            
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_metrics(response_time, result.success)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {str(e)}")
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return AgentResponse(
                success=False,
                error=f"Orchestrator failed: {str(e)}"
            )
    
    async def _analyze_intent(self, message: AgentMessage) -> AgentResponse:
        """Use NLU agent to parse and understand user intent"""
        if "nlu_parser" not in self.agents:
            return AgentResponse(
                success=False,
                error="NLU parser agent not available"
            )
        
        # Extract text from either "request" or "text" field
        text = message.body.get("request") or message.body.get("text", "")
        
        nlu_message = AgentMessage(
            recipient="nlu_parser",
            sender=self.agent_id,
            body={"text": text},
            context=message.context
        )
        
        return await self.agents["nlu_parser"].process_message(nlu_message)
    
    async def _create_execution_plan(self, intent_data: Dict[str, Any], 
                                   original_message: AgentMessage) -> Dict[str, Any]:
        """
        Create a step-by-step execution plan based on the parsed intent.
        
        Returns a plan with:
        - steps: List of actions to take
        - agents_needed: Which agents to invoke
        - tools_needed: Which tools to use  
        - parallel_steps: Steps that can run in parallel
        - fallback_strategies: What to do if steps fail
        """
        intent = intent_data.get("intent", "unknown")
        details = intent_data.get("details", {})
        
        plan = {
            "plan_id": f"plan_{datetime.now(timezone.utc).timestamp()}",
            "intent": intent,
            "steps": [],
            "agents_needed": [],
            "tools_needed": [],
            "parallel_steps": [],
            "fallback_strategies": []
        }
        
        if intent == "create_event":
            plan["steps"] = [
                {"action": "create_calendar_event", "tool": "google_calendar"},
                {"action": "confirm_creation", "agent": "orchestrator"}
            ]
            plan["tools_needed"] = ["google_calendar"]
            plan["data"] = intent_data  # Pass the parsed event data
            
        elif intent == "find_events":
            plan["steps"] = [
                {"action": "get_events", "tool": "google_calendar"},
                {"action": "format_results", "agent": "orchestrator"}
            ]
            plan["tools_needed"] = ["google_calendar"]
            plan["data"] = intent_data  # Pass the parsed search criteria
            
        elif intent == "schedule_optimal_time":
            plan["steps"] = [
                {"action": "analyze_preferences", "agent": "ml_suggestions"},
                {"action": "check_availability", "agent": "scheduler"},
                {"action": "propose_times", "agent": "scheduler"},
                {"action": "get_user_confirmation", "agent": "orchestrator"}
            ]
            plan["agents_needed"] = ["scheduler", "ml_suggestions"]
            plan["parallel_steps"] = [0, 1]  # Can run preferences and availability in parallel
            
        else:
            # Generic plan for unknown intents
            plan["steps"] = [
                {"action": "ask_clarification", "agent": "orchestrator"}
            ]
        
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any], 
                          original_message: AgentMessage) -> AgentResponse:
        """Execute the created plan step by step"""
        
        results = []
        context = original_message.context.copy()
        
        try:
            # Check if we can run any steps in parallel
            parallel_groups = self._group_parallel_steps(plan)
            
            for group in parallel_groups:
                if len(group) == 1:
                    # Single step execution
                    step = plan["steps"][group[0]]
                    result = await self._execute_step(step, context, original_message)
                    results.append(result)
                    
                    if not result.success and not step.get("optional", False):
                        # Critical step failed, abort plan
                        return AgentResponse(
                            success=False,
                            error=f"Critical step failed: {step['action']}",
                            data=result.error
                        )
                    
                    # Update context with step results
                    if result.metadata:
                        context.update(result.metadata)
                    # Also pass the data from tool results to context
                    if result.data and isinstance(result.data, dict):
                        context.update(result.data)
                    
                else:
                    # Parallel step execution
                    parallel_tasks = []
                    for step_idx in group:
                        step = plan["steps"][step_idx]
                        task = self._execute_step(step, context, original_message)
                        parallel_tasks.append(task)
                    
                    parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(parallel_results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Parallel step failed: {str(result)}")
                            result = AgentResponse(success=False, error=str(result))
                        
                        results.append(result)
                        if result.success:
                            if result.metadata:
                                context.update(result.metadata)
                            # Also pass the data from tool results to context
                            if result.data and isinstance(result.data, dict):
                                context.update(result.data)
            
            # Compile final response
            final_data = self._compile_results(results, plan)
            
            return AgentResponse(
                success=True,
                data=final_data,
                metadata=context
            )
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {str(e)}")
            return AgentResponse(
                success=False,
                error=f"Plan execution failed: {str(e)}"
            )
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any],
                          original_message: AgentMessage) -> AgentResponse:
        """Execute a single step in the plan"""
        
        if "agent" in step:
            # Route to specialized agent
            agent_id = step["agent"]
            if agent_id == "orchestrator":
                # Handle orchestrator-specific actions
                return await self._handle_orchestrator_action(step, context, original_message)
            
            if agent_id not in self.agents:
                return AgentResponse(
                    success=False,
                    error=f"Agent {agent_id} not available"
                )
            
            agent_message = AgentMessage(
                recipient=agent_id,
                sender=self.agent_id,
                body={"action": step["action"], "context": context},
                context=original_message.context
            )
            
            return await self.agents[agent_id].process_message(agent_message)
            
        elif "tool" in step:
            # Use external tool adapter
            tool_id = step["tool"]
            if tool_id not in self.tool_adapters:
                return AgentResponse(
                    success=False,
                    error=f"Tool {tool_id} not available"
                )
            
            try:
                result = await self.tool_adapters[tool_id].execute(
                    step["action"], 
                    context
                )
                # Pass the tool result data as the response data
                # This will be used as context in subsequent steps
                return AgentResponse(success=True, data=result)
                
            except Exception as e:
                return AgentResponse(
                    success=False,
                    error=f"Tool execution failed: {str(e)}"
                )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Invalid step configuration: {step}"
            )
    
    async def _handle_orchestrator_action(self, step: Dict[str, Any], 
                                        context: Dict[str, Any],
                                        original_message: AgentMessage) -> AgentResponse:
        """Handle actions that the orchestrator performs directly"""
        
        action = step["action"]
        
        if action == "ask_clarification":
            return AgentResponse(
                success=True,
                data="I'm not sure what you want me to do. Could you please clarify your request?",
                suggestions=[
                    "Try: 'Create a meeting tomorrow at 2pm'",
                    "Try: 'Find my meetings this week'", 
                    "Try: 'When is my next appointment?'"
                ]
            )
        
        elif action == "confirm_creation":
            event_data = context.get("created_event", {})
            return AgentResponse(
                success=True,
                data=f"Successfully created event: {event_data.get('title', 'Event')}",
                metadata={"event": event_data}
            )
        
        elif action == "format_results":
            # Handle different data formats from tools
            events = context.get("found_events", [])
            if not events and "events" in context:
                events = context["events"]
            
            if not events:
                return AgentResponse(
                    success=True,
                    data="No events found matching your criteria."
                )
            
            formatted = self._format_events_for_display(events)
            
            # Add mock data indicator if present
            mock_indicator = " (Mock Data)" if context.get("mock_data") else ""
            
            return AgentResponse(
                success=True,
                data=formatted + mock_indicator,
                metadata={"event_count": len(events), "mock_data": context.get("mock_data", False)}
            )
        
        else:
            return AgentResponse(
                success=False,
                error=f"Unknown orchestrator action: {action}"
            )
    
    def _format_events_for_display(self, events: List[Dict[str, Any]]) -> str:
        """Format calendar events for user display"""
        lines = []
        for event in events:
            # Handle both Google Calendar format and mock format
            title = event.get('summary') or event.get('title', 'No Title')
            
            # Extract start time from nested structure
            start_info = event.get('start', {})
            if isinstance(start_info, dict):
                start_time = start_info.get('dateTime', start_info.get('date', 'No Time'))
            else:
                start_time = str(start_info) if start_info else 'No Time'
            
            # Clean up the time format for display
            if 'T' in start_time:
                start_time = start_time.split('T')[0] + ' ' + start_time.split('T')[1][:5]
            
            # Add location if available
            location = event.get('location', '')
            location_str = f" @ {location}" if location else ""
            
            lines.append(f"• {start_time}: {title}{location_str}")
        
        return "\n".join(lines)
    
    def _group_parallel_steps(self, plan: Dict[str, Any]) -> List[List[int]]:
        """Group steps that can be executed in parallel"""
        parallel_indices = set(plan.get("parallel_steps", []))
        groups = []
        current_group = []
        
        for i, step in enumerate(plan["steps"]):
            if i in parallel_indices:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([i])
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _compile_results(self, results: List[AgentResponse], 
                        plan: Dict[str, Any]) -> Any:
        """Compile step results into final response data"""
        
        # Find the primary result (usually the last successful step)
        primary_result = None
        for result in reversed(results):
            if result.success and result.data:
                primary_result = result.data
                break
        
        return primary_result or "Task completed successfully"
    
    async def _update_scratchpad(self, message: AgentMessage):
        """Update short-term memory scratchpad with current request"""
        if self.memory_manager:
            await self.memory_manager.update_scratchpad({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_request": message.body.get("text", ""),
                "context": message.context,
                "plan_id": getattr(self.current_plan, "plan_id", None)
            })
    
    async def _update_memory(self, message: AgentMessage, plan: Dict[str, Any], 
                           result: AgentResponse):
        """Update memory tiers with completed interaction"""
        if not self.memory_manager:
            return
        
        memory_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_request": message.body.get("text", ""),
            "intent": plan.get("intent"),
            "plan_id": plan.get("plan_id"),
            "success": result.success,
            "result_data": result.data,
            "steps_completed": len([r for r in [result] if r.success])
        }
        
        # Store in appropriate memory tiers
        await self.memory_manager.store_interaction(memory_entry)
    
    async def _self_evaluate(self, plan: Dict[str, Any], result: AgentResponse):
        """
        Self-reflection on plan execution to improve future performance.
        
        Evaluates:
        - Did we use the right tools/agents?
        - Was the plan efficient?
        - Could we have done better?
        """
        
        evaluation = {
            "plan_id": plan.get("plan_id"),
            "success": result.success,
            "steps_planned": len(plan.get("steps", [])),
            "agents_used": plan.get("agents_needed", []),
            "tools_used": plan.get("tools_needed", []),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Simple heuristics for self-evaluation
        insights = []
        
        if not result.success:
            insights.append("Plan execution failed - review error handling")
        
        if len(plan.get("steps", [])) > 5:
            insights.append("Plan was complex - consider simplification")
        
        if len(plan.get("agents_needed", [])) > 3:
            insights.append("Used many agents - check if coordination was efficient")
        
        evaluation["insights"] = insights
        
        self.logger.info(f"Self-evaluation: {evaluation}")
        
        # Store evaluation for future learning
        if self.memory_manager:
            await self.memory_manager.store_evaluation(evaluation)
    
    async def _validate_plan_data(self, plan: Dict[str, Any]) -> 'ValidationResult':
        """Validate plan data using Pydantic schemas"""
        
        if not VALIDATION_AVAILABLE:
            # Return mock success if validation not available
            return type('ValidationResult', (), {
                'valid': True, 'errors': [], 'warnings': []
            })()
        
        plan_type = plan.get("type")
        data = plan.get("data", {})
        
        try:
            if plan_type == "calendar_operation":
                action = plan.get("action", "")
                
                if action in ["create_event", "update_event"]:
                    return validate_event(data)
                elif action in ["create_task", "update_task"]:
                    return validate_task(data)
            
            # For other plan types, assume valid
            return ValidationResult(valid=True, data=data, errors=[], warnings=[])
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                valid=False, 
                data=None, 
                errors=[f"Validation failed: {str(e)}"], 
                warnings=[]
            )
    
    async def _fix_plan_with_llm(self, plan: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
        """Attempt to fix plan using LLM with validation context"""
        
        if "nlu_parser" not in self.agents:
            return None
        
        try:
            # Create message with validation errors for LLM to fix
            fix_message = AgentMessage(
                recipient="nlu_parser",
                sender="orchestrator",
                body={
                    "original_plan": plan,
                    "validation_errors": errors,
                    "fix_request": f"Please fix the following validation errors: {'; '.join(errors)}"
                }
            )
            
            response = await self.agents["nlu_parser"].process_message(fix_message)
            
            if response.success:
                # Re-validate the fixed plan
                fixed_plan = response.data
                validation_result = await self._validate_plan_data(fixed_plan)
                
                if validation_result.valid:
                    self.logger.info("LLM successfully fixed validation errors")
                    return fixed_plan
                else:
                    self.logger.warning("LLM fix attempt still has validation errors")
            
        except Exception as e:
            self.logger.error(f"LLM fix attempt failed: {e}")
        
        return None
    
    def _requires_approval(self, plan: Dict[str, Any]) -> bool:
        """Check if plan requires human approval"""
        
        if not VALIDATION_AVAILABLE:
            return False
        
        action = plan.get("action", "")
        params = plan.get("data", {})
        
        return requires_human_approval(action, params)
    
    async def _request_human_approval(self, plan: Dict[str, Any], original_message: AgentMessage) -> bool:
        """Request human approval for high-risk actions"""
        
        if not VALIDATION_AVAILABLE:
            return True  # Default to allow if validation not available
        
        try:
            action = plan.get("action", "")
            params = plan.get("data", {})
            
            # Create approval summary
            approval_summary = create_approval_summary(action, params)
            
            # Store approval request in message context for UI to handle
            approval_context = {
                "requires_approval": True,
                "approval_summary": approval_summary,
                "plan": plan,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Log the approval request
            self.logger.info(f"Human approval required for: {approval_summary['summary']}")
            self.logger.info(f"Risk level: {approval_summary['risk_level']}")
            
            # Handle approval based on UI mode
            ui_mode = original_message.body.get("ui_mode", "production")
            
            if ui_mode == "debug":
                # In debug mode, auto-approve for testing
                self.logger.info("Debug mode: Auto-approving action")
                return True
            elif ui_mode == "minimal":
                # In minimal mode, approve simple actions only
                if approval_summary["risk_level"] == "low":
                    return True
                else:
                    self.logger.warning("Minimal mode: Blocking high-risk action")
                    return False
            else:
                # Production mode: would integrate with UI approval system
                self.logger.info("Production mode: Would request user approval via UI")
                # For demo purposes, approve low/medium risk, block high risk
                return approval_summary["risk_level"] != "high"
                
        except Exception as e:
            self.logger.error(f"Approval request failed: {e}")
            # Fail safe: deny approval on error
            return False 
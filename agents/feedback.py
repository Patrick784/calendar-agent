"""
Feedback Agent

After a task's scheduled end, prompts the user for completion status,
actual duration and blockers. Feeds labelled data back to the ML agent
for retraining and system improvement.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from .base import BaseAgent, AgentMessage, AgentResponse

class FeedbackAgent(BaseAgent):
    """
    Feedback collection and learning agent.
    
    Responsibilities:
    - Monitor scheduled events for completion
    - Collect user feedback on event success/failure
    - Track actual vs planned duration
    - Identify common blockers and issues
    - Feed labeled data back to ML agent for learning
    - Generate insights for system improvement
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__("feedback", settings)
        
        # Feedback collection settings
        self.feedback_delay_hours = settings.get("feedback_delay_hours", 1)  # Wait 1 hour after event
        self.max_feedback_attempts = settings.get("max_feedback_attempts", 3)
        self.feedback_timeout_days = settings.get("feedback_timeout_days", 7)
        
        # Event tracking
        self.pending_feedback: Dict[str, Dict[str, Any]] = {}
        self.collected_feedback: List[Dict[str, Any]] = []
        
        # Connected components
        self.ml_agent = None
        self.memory_manager = None
        self.calendar_adapters: Dict[str, Any] = {}
    
    def set_ml_agent(self, ml_agent):
        """Set ML agent for feeding back training data"""
        self.ml_agent = ml_agent
    
    def set_memory_manager(self, memory_manager):
        """Set memory manager for storing feedback data"""
        self.memory_manager = memory_manager
    
    def set_calendar_adapters(self, adapters: Dict[str, Any]):
        """Set calendar adapters for event monitoring"""
        self.calendar_adapters = adapters
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process feedback requests.
        
        Supported actions:
        - track_event: Start tracking an event for feedback
        - collect_feedback: Collect feedback for a specific event
        - process_user_feedback: Process user-provided feedback
        - get_feedback_insights: Get insights from collected feedback
        - trigger_ml_retraining: Send feedback data to ML agent
        """
        
        start_time = datetime.utcnow()
        action = message.body.get("action", "")
        context = message.body.get("context", {})
        
        try:
            if action == "track_event":
                result = await self._track_event(context)
            elif action == "collect_feedback":
                result = await self._collect_feedback(context)
            elif action == "process_user_feedback":
                result = await self._process_user_feedback(context)
            elif action == "get_feedback_insights":
                result = await self._get_feedback_insights(context)
            elif action == "trigger_ml_retraining":
                result = await self._trigger_ml_retraining(context)
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown feedback action: {action}"
                )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, True)
            
            return AgentResponse(
                success=True,
                data=result,
                metadata={"action": action, "response_time": response_time}
            )
            
        except Exception as e:
            self.logger.error(f"Feedback agent error for action '{action}': {str(e)}")
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return AgentResponse(
                success=False,
                error=f"Feedback collection failed: {str(e)}"
            )
    
    async def _track_event(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Start tracking an event for feedback collection"""
        
        event_data = context.get("event_data", {})
        event_id = context.get("event_id")
        user_id = context.get("user_id", "default")
        
        if not event_id:
            raise ValueError("Missing event_id for tracking")
        
        if not event_data:
            raise ValueError("Missing event_data for tracking")
        
        # Calculate feedback collection time
        end_datetime_str = event_data.get("end_datetime")
        if end_datetime_str:
            try:
                end_time = datetime.fromisoformat(end_datetime_str.replace('Z', ''))
                feedback_time = end_time + timedelta(hours=self.feedback_delay_hours)
            except Exception:
                feedback_time = datetime.utcnow() + timedelta(hours=self.feedback_delay_hours)
        else:
            feedback_time = datetime.utcnow() + timedelta(hours=self.feedback_delay_hours)
        
        # Store tracking information
        tracking_info = {
            "event_id": event_id,
            "event_data": event_data,
            "user_id": user_id,
            "scheduled_start": event_data.get("start_datetime"),
            "scheduled_end": event_data.get("end_datetime"),
            "scheduled_duration": event_data.get("duration_minutes", 60),
            "feedback_due_time": feedback_time.isoformat(),
            "feedback_attempts": 0,
            "tracking_started": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        self.pending_feedback[event_id] = tracking_info
        
        self.logger.info(f"Started tracking event {event_id} for feedback collection")
        
        return {
            "tracking_started": True,
            "event_id": event_id,
            "feedback_due_time": feedback_time.isoformat(),
            "tracking_info": tracking_info
        }
    
    async def _collect_feedback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect feedback for a specific event (typically called by scheduler)"""
        
        event_id = context.get("event_id")
        
        if not event_id or event_id not in self.pending_feedback:
            return {
                "feedback_collected": False,
                "reason": "Event not found in pending feedback list"
            }
        
        tracking_info = self.pending_feedback[event_id]
        
        # Check if it's time to collect feedback
        feedback_due_time = datetime.fromisoformat(tracking_info["feedback_due_time"])
        current_time = datetime.utcnow()
        
        if current_time < feedback_due_time:
            return {
                "feedback_collected": False,
                "reason": "Too early to collect feedback",
                "due_time": feedback_due_time.isoformat()
            }
        
        # Check if we've exceeded max attempts
        if tracking_info["feedback_attempts"] >= self.max_feedback_attempts:
            # Move to timeout status
            tracking_info["status"] = "timeout"
            self._archive_feedback(event_id, "timeout")
            
            return {
                "feedback_collected": False,
                "reason": "Maximum feedback attempts exceeded",
                "attempts": tracking_info["feedback_attempts"]
            }
        
        # Generate feedback request
        feedback_request = self._generate_feedback_request(tracking_info)
        
        # Update attempt count
        tracking_info["feedback_attempts"] += 1
        tracking_info["last_feedback_request"] = current_time.isoformat()
        
        return {
            "feedback_collected": False,
            "feedback_request_generated": True,
            "feedback_request": feedback_request,
            "attempts": tracking_info["feedback_attempts"],
            "event_id": event_id
        }
    
    async def _process_user_feedback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user-provided feedback for an event"""
        
        event_id = context.get("event_id")
        user_feedback = context.get("feedback", {})
        user_id = context.get("user_id", "default")
        
        if not event_id:
            raise ValueError("Missing event_id for feedback processing")
        
        if not user_feedback:
            raise ValueError("Missing feedback data")
        
        # Validate and structure feedback
        structured_feedback = self._structure_feedback(event_id, user_feedback, user_id)
        
        # Store feedback
        self.collected_feedback.append(structured_feedback)
        
        # Remove from pending feedback
        if event_id in self.pending_feedback:
            del self.pending_feedback[event_id]
        
        # Store in memory manager if available
        if self.memory_manager:
            try:
                await self.memory_manager.store_interaction({
                    "type": "feedback",
                    "event_id": event_id,
                    "user_id": user_id,
                    "feedback_data": structured_feedback,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Failed to store feedback in memory: {str(e)}")
        
        self.logger.info(f"Processed feedback for event {event_id}")
        
        # Check if we should trigger ML retraining
        if len(self.collected_feedback) % 10 == 0:  # Retrain every 10 feedback items
            await self._trigger_ml_retraining({"feedback_data": [structured_feedback]})
        
        return {
            "feedback_processed": True,
            "event_id": event_id,
            "structured_feedback": structured_feedback,
            "total_feedback_collected": len(self.collected_feedback)
        }
    
    async def _get_feedback_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from collected feedback data"""
        
        user_id = context.get("user_id", "default")
        time_period = context.get("time_period", "month")  # week, month, quarter
        
        # Filter feedback by user and time period
        cutoff_date = self._get_cutoff_date(time_period)
        relevant_feedback = []
        
        for feedback in self.collected_feedback:
            if feedback.get("user_id") == user_id:
                feedback_date = datetime.fromisoformat(feedback.get("timestamp", ""))
                if feedback_date >= cutoff_date:
                    relevant_feedback.append(feedback)
        
        if not relevant_feedback:
            return {
                "insights": [],
                "metrics": {},
                "recommendations": [],
                "feedback_count": 0
            }
        
        # Generate insights
        insights = self._analyze_feedback_patterns(relevant_feedback)
        
        return {
            "insights": insights["insights"],
            "metrics": insights["metrics"],
            "recommendations": insights["recommendations"],
            "feedback_count": len(relevant_feedback),
            "time_period": time_period,
            "user_id": user_id
        }
    
    async def _trigger_ml_retraining(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send feedback data to ML agent for retraining"""
        
        if not self.ml_agent:
            return {
                "retraining_triggered": False,
                "reason": "ML agent not available"
            }
        
        # Prepare feedback data for ML training
        training_data = context.get("feedback_data", self.collected_feedback[-50:])  # Last 50 items
        
        if not training_data:
            return {
                "retraining_triggered": False,
                "reason": "No feedback data available"
            }
        
        # Convert feedback to ML training format
        ml_training_data = []
        for feedback in training_data:
            try:
                ml_sample = self._convert_feedback_to_ml_format(feedback)
                if ml_sample:
                    ml_training_data.append(ml_sample)
            except Exception as e:
                self.logger.warning(f"Failed to convert feedback to ML format: {str(e)}")
                continue
        
        if not ml_training_data:
            return {
                "retraining_triggered": False,
                "reason": "No valid ML training data after conversion"
            }
        
        try:
            # Send to ML agent for retraining
            ml_message = AgentMessage(
                recipient="ml_suggestions",
                sender=self.agent_id,
                body={
                    "action": "retrain_models",
                    "context": {
                        "feedback_data": ml_training_data,
                        "source": "feedback_agent"
                    }
                }
            )
            
            ml_response = await self.ml_agent.process_message(ml_message)
            
            return {
                "retraining_triggered": True,
                "ml_response": ml_response.data if ml_response.success else ml_response.error,
                "training_samples_sent": len(ml_training_data),
                "success": ml_response.success
            }
            
        except Exception as e:
            self.logger.error(f"Failed to trigger ML retraining: {str(e)}")
            return {
                "retraining_triggered": False,
                "reason": f"ML retraining failed: {str(e)}"
            }
    
    def _generate_feedback_request(self, tracking_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a feedback request for the user"""
        
        event_data = tracking_info["event_data"]
        event_title = event_data.get("title", "Untitled Event")
        scheduled_duration = tracking_info.get("scheduled_duration", 60)
        
        return {
            "event_id": tracking_info["event_id"],
            "event_title": event_title,
            "scheduled_start": tracking_info["scheduled_start"],  
            "scheduled_end": tracking_info["scheduled_end"],
            "scheduled_duration_minutes": scheduled_duration,
            "questions": [
                {
                    "id": "completion_status",
                    "question": "Did this event happen as scheduled?",
                    "type": "multiple_choice",
                    "options": ["Yes, as planned", "Yes, but modified", "Partially completed", "Cancelled", "Rescheduled"]
                },
                {
                    "id": "actual_duration",
                    "question": f"How long did this event actually take? (Scheduled: {scheduled_duration} minutes)",
                    "type": "number",
                    "unit": "minutes"
                },
                {
                    "id": "success_rating",
                    "question": "How successful was this event?",
                    "type": "rating",
                    "scale": 5,
                    "labels": ["Very unsuccessful", "Unsuccessful", "Neutral", "Successful", "Very successful"]
                },
                {
                    "id": "blockers",
                    "question": "What, if anything, prevented this event from going as planned?",
                    "type": "multiple_choice",
                    "options": ["Nothing - went as planned", "Technical issues", "Scheduling conflict", "Attendee unavailable", "Ran out of time", "Other"],
                    "allow_multiple": True
                },
                {
                    "id": "improvements",
                    "question": "What could be improved for similar events in the future?",
                    "type": "text",
                    "optional": True
                }
            ],
            "feedback_due": tracking_info["feedback_due_time"],
            "attempt_number": tracking_info["feedback_attempts"] + 1
        }
    
    def _structure_feedback(self, event_id: str, user_feedback: Dict[str, Any], 
                          user_id: str) -> Dict[str, Any]:
        """Structure raw user feedback into standardized format"""
        
        # Get original tracking info if available
        tracking_info = self.pending_feedback.get(event_id, {})
        
        # Extract structured data
        completion_status = user_feedback.get("completion_status", "unknown")
        actual_duration = user_feedback.get("actual_duration")
        success_rating = user_feedback.get("success_rating", 3)
        blockers = user_feedback.get("blockers", [])
        improvements = user_feedback.get("improvements", "")
        
        # Determine success boolean
        success = (
            completion_status in ["Yes, as planned", "Yes, but modified"] and
            success_rating >= 4
        )
        
        structured = {
            "event_id": event_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            
            # Event details
            "event_data": tracking_info.get("event_data", {}),
            "scheduled_duration": tracking_info.get("scheduled_duration"),
            "actual_duration": actual_duration,
            
            # Feedback details
            "completion_status": completion_status,
            "success_rating": success_rating,
            "success": success,
            "blockers": blockers if isinstance(blockers, list) else [blockers],
            "improvements": improvements,
            
            # Derived metrics
            "duration_accuracy": self._calculate_duration_accuracy(
                tracking_info.get("scheduled_duration"),
                actual_duration
            ),
            "was_completed": completion_status not in ["Cancelled"],
            "had_issues": len(blockers) > 0 if isinstance(blockers, list) else bool(blockers)
        }
        
        return structured
    
    def _calculate_duration_accuracy(self, scheduled: Optional[int], 
                                   actual: Optional[int]) -> Optional[float]:
        """Calculate how accurate the duration estimate was"""
        
        if not scheduled or not actual or scheduled <= 0:
            return None
        
        # Calculate percentage accuracy (1.0 = perfect, 0.0 = completely wrong)
        accuracy = 1.0 - abs(scheduled - actual) / scheduled
        return max(0.0, accuracy)  # Don't go below 0
    
    def _convert_feedback_to_ml_format(self, feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert feedback to format suitable for ML training"""
        
        try:
            event_data = feedback.get("event_data", {})
            
            # Create ML training sample
            ml_sample = {
                "event_data": event_data,
                "time_slot": {
                    "start_datetime": event_data.get("start_datetime"),
                    "end_datetime": event_data.get("end_datetime"),
                    "duration_minutes": feedback.get("scheduled_duration", 60)
                },
                "success": feedback.get("success", False),
                "completion_status": feedback.get("completion_status"),
                "success_rating": feedback.get("success_rating", 3),
                "had_issues": feedback.get("had_issues", False),
                "duration_accuracy": feedback.get("duration_accuracy"),
                "user_id": feedback.get("user_id", "default")
            }
            
            return ml_sample
            
        except Exception as e:
            self.logger.warning(f"Failed to convert feedback to ML format: {str(e)}")
            return None
    
    def _analyze_feedback_patterns(self, feedback_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in collected feedback"""
        
        if not feedback_list:
            return {"insights": [], "metrics": {}, "recommendations": []}
        
        # Calculate metrics
        total_events = len(feedback_list)
        successful_events = sum(1 for f in feedback_list if f.get("success", False))
        completed_events = sum(1 for f in feedback_list if f.get("was_completed", False))
        
        success_rate = successful_events / total_events if total_events > 0 else 0
        completion_rate = completed_events / total_events if total_events > 0 else 0
        
        # Analyze duration accuracy
        duration_accuracies = [
            f.get("duration_accuracy") for f in feedback_list 
            if f.get("duration_accuracy") is not None
        ]
        avg_duration_accuracy = sum(duration_accuracies) / len(duration_accuracies) if duration_accuracies else 0
        
        # Analyze common blockers
        all_blockers = []
        for feedback in feedback_list:
            blockers = feedback.get("blockers", [])
            if isinstance(blockers, list):
                all_blockers.extend(blockers)
            elif blockers:
                all_blockers.append(blockers)
        
        blocker_counts = {}
        for blocker in all_blockers:
            if blocker != "Nothing - went as planned":
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        
        common_blockers = sorted(blocker_counts.keys(), 
                               key=lambda b: blocker_counts[b], reverse=True)[:3]
        
        # Generate insights
        insights = []
        
        if success_rate < 0.7:
            insights.append(f"Success rate is {success_rate:.1%} - consider reviewing scheduling patterns")
        
        if avg_duration_accuracy < 0.8:
            insights.append(f"Duration estimates are {avg_duration_accuracy:.1%} accurate - consider adjusting time estimates")
        
        if common_blockers:
            insights.append(f"Most common issues: {', '.join(common_blockers[:2])}")
        
        # Generate recommendations
        recommendations = []
        
        if "Technical issues" in common_blockers:
            recommendations.append("Consider adding buffer time for events that might have technical components")
        
        if "Scheduling conflict" in common_blockers:
            recommendations.append("Improve conflict detection when scheduling new events")
        
        if "Ran out of time" in common_blockers:
            recommendations.append("Consider increasing default duration estimates")
        
        metrics = {
            "total_events": total_events,
            "success_rate": success_rate,
            "completion_rate": completion_rate,
            "avg_duration_accuracy": avg_duration_accuracy,
            "common_blockers": dict(list(blocker_counts.items())[:5])
        }
        
        return {
            "insights": insights,
            "metrics": metrics,
            "recommendations": recommendations
        }
    
    def _get_cutoff_date(self, time_period: str) -> datetime:
        """Get cutoff date for filtering feedback by time period"""
        
        now = datetime.utcnow()
        
        if time_period == "week":
            return now - timedelta(days=7)
        elif time_period == "month":
            return now - timedelta(days=30)
        elif time_period == "quarter":
            return now - timedelta(days=90)
        else:
            return now - timedelta(days=30)  # Default to month
    
    def _archive_feedback(self, event_id: str, reason: str):
        """Archive feedback that couldn't be collected"""
        
        if event_id in self.pending_feedback:
            tracking_info = self.pending_feedback[event_id]
            tracking_info["archived_reason"] = reason
            tracking_info["archived_at"] = datetime.utcnow().isoformat()
            
            # Could store archived items for analysis
            self.logger.info(f"Archived feedback for event {event_id}: {reason}")
            
            del self.pending_feedback[event_id]
    
    async def check_pending_feedback(self) -> List[Dict[str, Any]]:
        """Check for events that need feedback collection (called by scheduler)"""
        
        current_time = datetime.utcnow()
        ready_for_feedback = []
        
        for event_id, tracking_info in list(self.pending_feedback.items()):
            feedback_due_time = datetime.fromisoformat(tracking_info["feedback_due_time"])
            
            # Check if feedback is due
            if current_time >= feedback_due_time:
                # Check if we haven't exceeded timeout
                tracking_started = datetime.fromisoformat(tracking_info["tracking_started"])
                if (current_time - tracking_started).days <= self.feedback_timeout_days:
                    ready_for_feedback.append({
                        "event_id": event_id,
                        "tracking_info": tracking_info
                    })
                else:
                    # Timeout - archive this feedback
                    self._archive_feedback(event_id, "timeout")
        
        return ready_for_feedback
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about feedback collection"""
        
        current_time = datetime.utcnow()
        
        # Count pending feedback by status
        pending_count = len(self.pending_feedback)
        overdue_count = 0
        
        for tracking_info in self.pending_feedback.values():
            feedback_due_time = datetime.fromisoformat(tracking_info["feedback_due_time"])
            if current_time > feedback_due_time:
                overdue_count += 1
        
        return {
            "pending_feedback": pending_count,
            "overdue_feedback": overdue_count,
            "collected_feedback": len(self.collected_feedback),
            "collection_rate": len(self.collected_feedback) / (len(self.collected_feedback) + pending_count) if (len(self.collected_feedback) + pending_count) > 0 else 0
        } 
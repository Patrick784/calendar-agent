"""
Scheduler Agent

Determines if/where to place tasks. Consults the task database, external calendars,
user preferences and ML predictions to suggest optimal time slots. Uses heuristics
to match tool usage to intent; starts broad then narrows search.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from .base import BaseAgent, AgentMessage, AgentResponse

class SchedulerAgent(BaseAgent):
    """
    Scheduling agent that finds optimal times for events and tasks.
    
    Capabilities:
    - Analyze calendar availability across multiple calendars
    - Consider user preferences and patterns
    - Integrate with ML suggestions for optimal scheduling
    - Handle recurring events and complex scheduling constraints
    - Suggest alternative times when conflicts exist
    - Optimize for productivity and preferences
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__("scheduler", settings)
        
        # Scheduling preferences and heuristics
        self.default_preferences = {
            "preferred_hours": {"start": 9, "end": 17},  # 9 AM to 5 PM
            "preferred_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "meeting_buffer_minutes": 15,
            "max_consecutive_meetings": 3,
            "lunch_break": {"start": 12, "end": 13},
            "default_duration_minutes": 60,
            "timezone": "UTC"
        }
        
        # Calendar adapters and ML agent references
        self.calendar_adapters: Dict[str, Any] = {}
        self.ml_agent = None
        self.memory_manager = None
    
    def set_calendar_adapters(self, adapters: Dict[str, Any]):
        """Set calendar adapters for availability checking"""
        self.calendar_adapters = adapters
        self.logger.info(f"Registered {len(adapters)} calendar adapters")
    
    def set_ml_agent(self, ml_agent):
        """Set ML suggestion agent for predictive scheduling"""
        self.ml_agent = ml_agent
    
    def set_memory_manager(self, memory_manager):
        """Set memory manager for storing preferences and patterns"""
        self.memory_manager = memory_manager
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process scheduling requests.
        
        Supported actions:
        - find_optimal_time: Find best time slot for an event
        - check_availability: Check if specific time is available
        - suggest_alternatives: Suggest alternative times for conflicts
        - analyze_schedule: Analyze schedule patterns and optimization
        """
        
        start_time = datetime.utcnow()
        action = message.body.get("action", "")
        context = message.body.get("context", {})
        
        try:
            if action == "find_optimal_time":
                result = await self._find_optimal_time(context)
            elif action == "check_availability":
                result = await self._check_availability(context)
            elif action == "suggest_alternatives":
                result = await self._suggest_alternatives(context)
            elif action == "analyze_schedule":
                result = await self._analyze_schedule(context)
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown scheduler action: {action}"
                )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, True)
            
            return AgentResponse(
                success=True,
                data=result,
                metadata={"action": action, "response_time": response_time}
            )
            
        except Exception as e:
            self.logger.error(f"Scheduler error for action '{action}': {str(e)}")
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return AgentResponse(
                success=False,
                error=f"Scheduler failed: {str(e)}"
            )
    
    async def _find_optimal_time(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the optimal time slot for an event based on multiple factors.
        
        Considers:
        - Calendar availability
        - User preferences
        - ML predictions
        - Existing schedule patterns
        - Meeting fatigue and productivity
        """
        
        # Extract scheduling requirements
        duration_minutes = context.get("duration_minutes", self.default_preferences["default_duration_minutes"])
        preferred_date = context.get("preferred_date")
        time_constraints = context.get("time_constraints", {})
        priority = context.get("priority", "medium")
        attendees = context.get("attendees", [])
        
        # Get user preferences (from memory or defaults)
        user_preferences = await self._get_user_preferences(context.get("user_id", "default"))
        
        # Define search window
        search_start, search_end = self._define_search_window(preferred_date, time_constraints)
        
        # Generate candidate time slots
        candidate_slots = await self._generate_candidate_slots(
            search_start, search_end, duration_minutes, user_preferences
        )
        
        # Check availability for each candidate
        available_slots = []
        for slot in candidate_slots:
            availability = await self._check_slot_availability(slot, attendees)
            if availability["available"]:
                available_slots.append({
                    **slot,
                    "conflicts": availability["conflicts"],
                    "availability_score": availability["score"]
                })
        
        if not available_slots:
            # No available slots found, suggest alternatives
            alternatives = await self._suggest_alternatives({
                "duration_minutes": duration_minutes,
                "search_start": search_start,
                "search_end": search_end,
                "attendees": attendees
            })
            
            return {
                "optimal_time": None,
                "available_slots": [],
                "alternatives": alternatives["alternatives"],
                "reason": "No available slots found in the requested time range"
            }
        
        # Score each available slot
        scored_slots = []
        for slot in available_slots:
            score = await self._score_time_slot(slot, context, user_preferences)
            scored_slots.append({
                **slot,
                "optimization_score": score["total_score"],
                "score_breakdown": score["breakdown"]
            })
        
        # Sort by score (highest first)
        scored_slots.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        optimal_slot = scored_slots[0]
        
        return {
            "optimal_time": {
                "start_datetime": optimal_slot["start_datetime"],
                "end_datetime": optimal_slot["end_datetime"],
                "duration_minutes": duration_minutes,
                "confidence": optimal_slot["optimization_score"]
            },
            "available_slots": scored_slots[:5],  # Top 5 alternatives
            "optimization_factors": optimal_slot["score_breakdown"],
            "user_preferences_applied": user_preferences
        }
    
    async def _check_availability(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check availability for a specific time slot"""
        
        start_datetime = context.get("start_datetime")
        end_datetime = context.get("end_datetime")
        attendees = context.get("attendees", [])
        
        if not start_datetime or not end_datetime:
            raise ValueError("Missing start_datetime or end_datetime")
        
        slot = {
            "start_datetime": start_datetime,
            "end_datetime": end_datetime
        }
        
        availability = await self._check_slot_availability(slot, attendees)
        
        return {
            "available": availability["available"],
            "conflicts": availability["conflicts"],
            "availability_score": availability["score"],
            "alternative_suggestions": await self._get_nearby_alternatives(slot) if not availability["available"] else []
        }
    
    async def _suggest_alternatives(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest alternative time slots when the preferred time is not available"""
        
        duration_minutes = context.get("duration_minutes", 60)
        search_start = context.get("search_start")
        search_end = context.get("search_end")
        attendees = context.get("attendees", [])
        
        # If no search window provided, use default (next 2 weeks)
        if not search_start:
            search_start = datetime.utcnow().isoformat()
        if not search_end:
            search_end = (datetime.utcnow() + timedelta(days=14)).isoformat()
        
        # Generate alternative slots with different strategies
        alternatives = []
        
        # Strategy 1: Earlier in the day
        early_slots = await self._find_slots_in_time_range(
            search_start, search_end, duration_minutes, 
            time_range=(8, 12), attendees=attendees
        )
        alternatives.extend(early_slots)
        
        # Strategy 2: Later in the day
        late_slots = await self._find_slots_in_time_range(
            search_start, search_end, duration_minutes,
            time_range=(14, 18), attendees=attendees
        )
        alternatives.extend(late_slots)
        
        # Strategy 3: Different days
        extended_end = (datetime.fromisoformat(search_end.replace('Z', '')) + timedelta(days=7)).isoformat()
        extended_slots = await self._find_slots_in_time_range(
            search_end, extended_end, duration_minutes,
            attendees=attendees
        )
        alternatives.extend(extended_slots)
        
        # Remove duplicates and sort by score
        unique_alternatives = self._deduplicate_slots(alternatives)
        
        return {
            "alternatives": unique_alternatives[:10],  # Top 10 alternatives
            "total_found": len(unique_alternatives),
            "search_strategies": ["early_day", "late_day", "extended_days"]
        }
    
    async def _analyze_schedule(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schedule patterns and provide optimization insights"""
        
        user_id = context.get("user_id", "default")
        analysis_period = context.get("analysis_period", "week")  # week, month, quarter
        
        # Get historical schedule data
        if self.memory_manager:
            historical_data = await self.memory_manager.search_memory(
                f"user schedule patterns {user_id}", limit=50
            )
        else:
            historical_data = []
        
        # Get current calendar data
        current_events = await self._get_current_schedule_data(user_id, analysis_period)
        
        analysis = {
            "schedule_density": self._calculate_schedule_density(current_events),
            "meeting_patterns": self._analyze_meeting_patterns(current_events),
            "productivity_windows": self._identify_productivity_windows(current_events),
            "optimization_suggestions": [],
            "workload_balance": self._analyze_workload_balance(current_events)
        }
        
        # Generate optimization suggestions
        suggestions = []
        
        if analysis["schedule_density"] > 0.8:
            suggestions.append({
                "type": "schedule_density",
                "message": "Your schedule is very dense. Consider blocking time for deep work.",
                "priority": "high"
            })
        
        if analysis["meeting_patterns"]["consecutive_meetings"] > 3:
            suggestions.append({
                "type": "meeting_fatigue",
                "message": "You have many consecutive meetings. Consider adding buffers between meetings.",
                "priority": "medium"
            })
        
        if len(analysis["productivity_windows"]) < 2:
            suggestions.append({
                "type": "productivity_time",
                "message": "Limited productivity windows detected. Consider protecting morning hours for focused work.",
                "priority": "medium"
            })
        
        analysis["optimization_suggestions"] = suggestions
        
        return analysis
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user scheduling preferences from memory or defaults"""
        
        if self.memory_manager:
            try:
                # Search for user preferences in memory
                preference_results = await self.memory_manager.search_memory(
                    f"user preferences scheduling {user_id}", limit=1
                )
                
                if preference_results:
                    stored_prefs = preference_results[0].get("metadata", {})
                    # Merge with defaults
                    preferences = self.default_preferences.copy()
                    preferences.update(stored_prefs)
                    return preferences
            except Exception as e:
                self.logger.warning(f"Failed to retrieve user preferences: {str(e)}")
        
        return self.default_preferences.copy()
    
    def _define_search_window(self, preferred_date: Optional[str], 
                            time_constraints: Dict[str, Any]) -> Tuple[str, str]:
        """Define the time window for searching available slots"""
        
        if preferred_date:
            base_date = datetime.fromisoformat(preferred_date.replace('Z', ''))
        else:
            base_date = datetime.utcnow()
        
        # Default search window: start of preferred day to 7 days later
        search_start = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        search_end = search_start + timedelta(days=7)
        
        # Apply time constraints
        if "earliest_start" in time_constraints:
            earliest = datetime.fromisoformat(time_constraints["earliest_start"].replace('Z', ''))
            search_start = max(search_start, earliest)
        
        if "latest_end" in time_constraints:
            latest = datetime.fromisoformat(time_constraints["latest_end"].replace('Z', ''))
            search_end = min(search_end, latest)
        
        return search_start.isoformat(), search_end.isoformat()
    
    async def _generate_candidate_slots(self, search_start: str, search_end: str,
                                      duration_minutes: int, 
                                      user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate time slots based on preferences"""
        
        slots = []
        start_dt = datetime.fromisoformat(search_start.replace('Z', ''))
        end_dt = datetime.fromisoformat(search_end.replace('Z', ''))
        
        current_dt = start_dt
        slot_interval = timedelta(minutes=30)  # Check every 30 minutes
        
        while current_dt + timedelta(minutes=duration_minutes) <= end_dt:
            # Check if this time fits user preferences
            if self._fits_user_preferences(current_dt, duration_minutes, user_preferences):
                slot_end = current_dt + timedelta(minutes=duration_minutes)
                slots.append({
                    "start_datetime": current_dt.isoformat(),
                    "end_datetime": slot_end.isoformat(),
                    "duration_minutes": duration_minutes
                })
            
            current_dt += slot_interval
        
        return slots
    
    def _fits_user_preferences(self, start_time: datetime, duration_minutes: int,
                             preferences: Dict[str, Any]) -> bool:
        """Check if a time slot fits user preferences"""
        
        # Check day of week
        day_name = start_time.strftime("%A").lower()
        if day_name not in preferences.get("preferred_days", []):
            return False
        
        # Check hours
        start_hour = start_time.hour
        end_hour = (start_time + timedelta(minutes=duration_minutes)).hour
        
        pref_start = preferences.get("preferred_hours", {}).get("start", 0)
        pref_end = preferences.get("preferred_hours", {}).get("end", 24)
        
        if start_hour < pref_start or end_hour > pref_end:
            return False
        
        # Check lunch break
        lunch = preferences.get("lunch_break")
        if lunch:
            lunch_start = lunch.get("start", 12)
            lunch_end = lunch.get("end", 13)
            
            # Don't schedule during lunch
            if lunch_start <= start_hour < lunch_end or lunch_start < end_hour <= lunch_end:
                return False
        
        return True
    
    async def _check_slot_availability(self, slot: Dict[str, Any], 
                                     attendees: List[str] = None) -> Dict[str, Any]:
        """Check if a time slot is available across all relevant calendars"""
        
        conflicts = []
        availability_score = 1.0
        
        # Check primary calendar adapter (usually Google Calendar)
        primary_adapter = self.calendar_adapters.get("google_calendar")
        if primary_adapter:
            try:
                availability_result = await primary_adapter.execute("check_availability", {
                    "start_datetime": slot["start_datetime"],
                    "end_datetime": slot["end_datetime"],
                    "calendar_ids": ["primary"]
                })
                
                if not availability_result["available"]:
                    conflicts.extend(availability_result["conflicting_events"])
                    availability_score *= 0.5  # Reduce score for conflicts
                    
            except Exception as e:
                self.logger.warning(f"Failed to check calendar availability: {str(e)}")
                availability_score *= 0.8  # Slight penalty for unavailable data
        
        # TODO: Check attendee availability if provided
        # This would require integration with multiple calendar systems
        
        return {
            "available": len(conflicts) == 0,
            "conflicts": conflicts,
            "score": availability_score
        }
    
    async def _score_time_slot(self, slot: Dict[str, Any], context: Dict[str, Any],
                             user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Score a time slot based on multiple optimization factors"""
        
        scores = {}
        
        # Time of day preference score
        start_time = datetime.fromisoformat(slot["start_datetime"].replace('Z', ''))
        hour = start_time.hour
        
        pref_start = user_preferences.get("preferred_hours", {}).get("start", 9)
        pref_end = user_preferences.get("preferred_hours", {}).get("end", 17)
        
        if pref_start <= hour < pref_end:
            scores["time_preference"] = 1.0
        else:
            scores["time_preference"] = 0.3
        
        # Day of week preference score
        day_name = start_time.strftime("%A").lower()
        preferred_days = user_preferences.get("preferred_days", [])
        
        if day_name in preferred_days:
            scores["day_preference"] = 1.0
        else:
            scores["day_preference"] = 0.5
        
        # Urgency/priority score
        priority = context.get("priority", "medium")
        priority_scores = {"low": 0.3, "medium": 0.6, "high": 1.0, "urgent": 1.2}
        scores["priority"] = priority_scores.get(priority, 0.6)
        
        # ML prediction score (if ML agent is available)
        if self.ml_agent:
            try:
                ml_message = {
                    "action": "predict_success",
                    "context": {
                        "time_slot": slot,
                        "user_context": context
                    }
                }
                # This would be implemented when ML agent is ready
                scores["ml_prediction"] = 0.7  # Placeholder
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {str(e)}")
                scores["ml_prediction"] = 0.5
        else:
            scores["ml_prediction"] = 0.5
        
        # Calculate weighted total score
        weights = {
            "time_preference": 0.3,
            "day_preference": 0.2, 
            "priority": 0.3,
            "ml_prediction": 0.2
        }
        
        total_score = sum(scores[factor] * weights[factor] for factor in scores)
        
        return {
            "total_score": total_score,
            "breakdown": scores
        }
    
    async def _find_slots_in_time_range(self, search_start: str, search_end: str,
                                      duration_minutes: int, time_range: Tuple[int, int] = None,
                                      attendees: List[str] = None) -> List[Dict[str, Any]]:
        """Find available slots within a specific time range"""
        
        slots = []
        start_dt = datetime.fromisoformat(search_start.replace('Z', ''))
        end_dt = datetime.fromisoformat(search_end.replace('Z', ''))
        
        current_dt = start_dt
        
        while current_dt + timedelta(minutes=duration_minutes) <= end_dt:
            # Apply time range filter if specified
            if time_range:
                if not (time_range[0] <= current_dt.hour < time_range[1]):
                    current_dt += timedelta(minutes=30)
                    continue
            
            slot = {
                "start_datetime": current_dt.isoformat(),
                "end_datetime": (current_dt + timedelta(minutes=duration_minutes)).isoformat(),
                "duration_minutes": duration_minutes
            }
            
            # Check availability
            availability = await self._check_slot_availability(slot, attendees)
            if availability["available"]:
                slots.append({
                    **slot,
                    "availability_score": availability["score"]
                })
            
            current_dt += timedelta(minutes=30)
        
        return slots
    
    def _deduplicate_slots(self, slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate time slots and sort by score"""
        
        seen = set()
        unique_slots = []
        
        for slot in slots:
            key = (slot["start_datetime"], slot["end_datetime"])
            if key not in seen:
                seen.add(key)
                unique_slots.append(slot)
        
        # Sort by availability score if present
        unique_slots.sort(key=lambda x: x.get("availability_score", 0), reverse=True)
        
        return unique_slots
    
    async def _get_current_schedule_data(self, user_id: str, period: str) -> List[Dict[str, Any]]:
        """Get current schedule data for analysis"""
        
        # Define time range based on period
        now = datetime.utcnow()
        if period == "week":
            start_date = now - timedelta(days=7)
        elif period == "month":
            start_date = now - timedelta(days=30)
        else:  # quarter
            start_date = now - timedelta(days=90)
        
        events = []
        
        # Get events from primary calendar
        primary_adapter = self.calendar_adapters.get("google_calendar")
        if primary_adapter:
            try:
                result = await primary_adapter.execute("get_events", {
                    "start_date": start_date.isoformat(),
                    "end_date": now.isoformat()
                })
                events = result.get("events", [])
            except Exception as e:
                self.logger.warning(f"Failed to get schedule data: {str(e)}")
        
        return events
    
    def _calculate_schedule_density(self, events: List[Dict[str, Any]]) -> float:
        """Calculate how densely packed the schedule is"""
        
        if not events:
            return 0.0
        
        total_time = 0
        meeting_time = 0
        
        for event in events:
            try:
                start = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                end = datetime.fromisoformat(event["end_datetime"].replace('Z', ''))
                duration = (end - start).total_seconds() / 3600  # hours
                meeting_time += duration
            except Exception:
                continue
        
        # Assume 8 working hours per day
        working_days = len(set(
            datetime.fromisoformat(event["start_datetime"].replace('Z', '')).date()
            for event in events
        ))
        
        total_time = working_days * 8  # hours
        
        return min(meeting_time / total_time if total_time > 0 else 0, 1.0)
    
    def _analyze_meeting_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in meeting scheduling"""
        
        if not events:
            return {"consecutive_meetings": 0, "average_duration": 0, "peak_hours": []}
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda x: x["start_datetime"])
        
        # Find consecutive meetings
        consecutive_count = 0
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(sorted_events)):
            prev_end = datetime.fromisoformat(sorted_events[i-1]["end_datetime"].replace('Z', ''))
            curr_start = datetime.fromisoformat(sorted_events[i]["start_datetime"].replace('Z', ''))
            
            # If less than 15 minutes between meetings, consider consecutive
            if (curr_start - prev_end).total_seconds() < 15 * 60:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        max_consecutive = max(max_consecutive, current_consecutive)
        
        # Calculate average duration
        total_duration = 0
        for event in events:
            try:
                start = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                end = datetime.fromisoformat(event["end_datetime"].replace('Z', ''))
                total_duration += (end - start).total_seconds() / 60  # minutes
            except Exception:
                continue
        
        avg_duration = total_duration / len(events) if events else 0
        
        # Find peak hours
        hour_counts = {}
        for event in events:
            try:
                hour = datetime.fromisoformat(event["start_datetime"].replace('Z', '')).hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except Exception:
                continue
        
        peak_hours = sorted(hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True)[:3]
        
        return {
            "consecutive_meetings": max_consecutive,
            "average_duration": avg_duration,
            "peak_hours": peak_hours,
            "total_meetings": len(events)
        }
    
    def _identify_productivity_windows(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify free time windows that could be used for focused work"""
        
        # This is a simplified implementation
        # You could make this much more sophisticated
        
        windows = []
        
        # Look for gaps of 2+ hours between meetings
        sorted_events = sorted(events, key=lambda x: x["start_datetime"])
        
        for i in range(len(sorted_events) - 1):
            curr_end = datetime.fromisoformat(sorted_events[i]["end_datetime"].replace('Z', ''))
            next_start = datetime.fromisoformat(sorted_events[i+1]["start_datetime"].replace('Z', ''))
            
            gap_duration = (next_start - curr_end).total_seconds() / 3600  # hours
            
            if gap_duration >= 2:
                windows.append({
                    "start": curr_end.isoformat(),
                    "end": next_start.isoformat(),
                    "duration_hours": gap_duration
                })
        
        return windows
    
    def _analyze_workload_balance(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workload distribution across days and weeks"""
        
        daily_loads = {}
        
        for event in events:
            try:
                start = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                end = datetime.fromisoformat(event["end_datetime"].replace('Z', ''))
                duration = (end - start).total_seconds() / 3600  # hours
                
                date_key = start.date().isoformat()
                daily_loads[date_key] = daily_loads.get(date_key, 0) + duration
            except Exception:
                continue
        
        if not daily_loads:
            return {"average_daily_load": 0, "max_load_day": None, "load_variance": 0}
        
        loads = list(daily_loads.values())
        avg_load = sum(loads) / len(loads)
        max_load = max(loads)
        max_load_day = max(daily_loads.keys(), key=lambda k: daily_loads[k])
        
        # Calculate variance
        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        
        return {
            "average_daily_load": avg_load,
            "max_load_day": max_load_day,
            "max_load_hours": max_load,
            "load_variance": variance,
            "balanced": variance < 2.0  # Low variance indicates good balance
        } 
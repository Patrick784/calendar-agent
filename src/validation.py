"""
Validation Module

Pydantic schemas and validation functions for calendar agent data structures.
Ensures all LLM-generated objects conform to expected formats before processing.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date, time
from enum import Enum
import re
from pydantic import BaseModel, Field, validator, ValidationError

class Priority(str, Enum):
    """Priority levels for tasks and events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class EventStatus(str, Enum):
    """Status of calendar events"""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"

class RecurrenceFrequency(str, Enum):
    """Recurrence frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

class Task(BaseModel):
    """
    Pydantic schema for Task objects.
    
    Used for validating task-related data from LLM parsing and user input.
    """
    
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=1000, description="Task description")
    
    # Time-related fields
    due_date: Optional[datetime] = Field(None, description="When the task is due")
    estimated_duration_minutes: Optional[int] = Field(
        None, 
        ge=1, 
        le=1440,  # Max 24 hours
        description="Estimated time to complete task in minutes"
    )
    
    # Metadata
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority level")
    tags: List[str] = Field(default_factory=list, description="Task tags/categories")
    
    # Context
    project: Optional[str] = Field(None, max_length=100, description="Associated project")
    location: Optional[str] = Field(None, max_length=200, description="Where task should be done")
    
    # State
    completed: bool = Field(default=False, description="Task completion status")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate task title"""
        if not v or not v.strip():
            raise ValueError("Task title cannot be empty")
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for obvious placeholder text
        placeholders = ['todo', 'task', 'thing to do', 'something']
        if v.lower().strip() in placeholders:
            raise ValueError(f"Task title appears to be placeholder text: {v}")
        
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate task tags"""
        if v:
            # Clean and deduplicate tags
            cleaned_tags = []
            for tag in v:
                if isinstance(tag, str) and tag.strip():
                    clean_tag = tag.strip().lower()
                    if clean_tag not in cleaned_tags:
                        cleaned_tags.append(clean_tag)
            return cleaned_tags
        return []
    
    @validator('due_date')
    def validate_due_date(cls, v):
        """Validate due date"""
        if v and v < datetime.now():
            # Allow past dates but warn
            pass  # Could add warning here
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Event(BaseModel):
    """
    Pydantic schema for Event objects.
    
    Used for validating calendar event data from LLM parsing and user input.
    """
    
    title: str = Field(..., min_length=1, max_length=200, description="Event title")
    description: Optional[str] = Field(None, max_length=2000, description="Event description")
    
    # Time fields (required)
    start_datetime: datetime = Field(..., description="Event start time")
    end_datetime: datetime = Field(..., description="Event end time")
    
    # Location and logistics
    location: Optional[str] = Field(None, max_length=300, description="Event location")
    virtual_meeting_url: Optional[str] = Field(None, description="Online meeting URL")
    
    # Attendees
    attendees: List[str] = Field(default_factory=list, description="List of attendee email addresses")
    organizer: Optional[str] = Field(None, description="Event organizer email")
    
    # Metadata
    priority: Priority = Field(default=Priority.MEDIUM, description="Event priority")
    status: EventStatus = Field(default=EventStatus.CONFIRMED, description="Event status")
    all_day: bool = Field(default=False, description="All-day event flag")
    
    # Calendar info
    calendar_id: str = Field(default="primary", description="Target calendar ID")
    
    # Recurrence
    recurrence_frequency: Optional[RecurrenceFrequency] = Field(None, description="Recurrence pattern")
    recurrence_end_date: Optional[date] = Field(None, description="When recurrence stops")
    recurrence_count: Optional[int] = Field(None, ge=1, le=365, description="Number of recurrences")
    
    # Notifications
    reminder_minutes: List[int] = Field(
        default_factory=lambda: [15], 
        description="Reminder times in minutes before event"
    )
    
    @validator('title')
    def validate_title(cls, v):
        """Validate event title"""
        if not v or not v.strip():
            raise ValueError("Event title cannot be empty")
        
        # Clean title
        v = ' '.join(v.split())
        
        # Check for placeholder text
        placeholders = ['meeting', 'event', 'appointment', 'call', 'untitled']
        if v.lower().strip() in placeholders:
            raise ValueError(f"Event title appears to be placeholder text: {v}")
        
        return v
    
    @validator('end_datetime')
    def validate_end_after_start(cls, v, values):
        """Ensure end time is after start time"""
        if 'start_datetime' in values and v <= values['start_datetime']:
            raise ValueError("Event end time must be after start time")
        return v
    
    @validator('start_datetime', 'end_datetime')
    def validate_reasonable_times(cls, v):
        """Validate that times are reasonable (not too far in past/future)"""
        now = datetime.now()
        
        # Don't allow events more than 5 years in the future
        max_future = now.replace(year=now.year + 5)
        if v > max_future:
            raise ValueError(f"Event time too far in future: {v}")
        
        # Allow past events (for logging/tracking) but not more than 2 years
        min_past = now.replace(year=now.year - 2)
        if v < min_past:
            raise ValueError(f"Event time too far in past: {v}")
        
        return v
    
    @validator('attendees')
    def validate_attendees(cls, v):
        """Validate attendee email addresses"""
        if not v:
            return []
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        validated_attendees = []
        
        for attendee in v:
            if isinstance(attendee, str) and attendee.strip():
                email = attendee.strip().lower()
                if email_pattern.match(email):
                    validated_attendees.append(email)
                else:
                    raise ValueError(f"Invalid email address: {attendee}")
        
        return validated_attendees
    
    @validator('virtual_meeting_url')
    def validate_meeting_url(cls, v):
        """Validate virtual meeting URL"""
        if v:
            url_pattern = re.compile(
                r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
            )
            if not url_pattern.match(v):
                raise ValueError(f"Invalid meeting URL format: {v}")
        return v
    
    @validator('reminder_minutes')
    def validate_reminders(cls, v):
        """Validate reminder times"""
        if v:
            # Ensure all reminders are positive and reasonable
            validated_reminders = []
            for reminder in v:
                if isinstance(reminder, int) and 0 <= reminder <= 10080:  # Max 1 week
                    validated_reminders.append(reminder)
                else:
                    raise ValueError(f"Invalid reminder time: {reminder} minutes")
            return validated_reminders
        return []
    
    @validator('recurrence_end_date')
    def validate_recurrence_end(cls, v, values):
        """Validate recurrence end date"""
        if v and 'start_datetime' in values:
            if v < values['start_datetime'].date():
                raise ValueError("Recurrence end date must be after event start date")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class ValidationResult(BaseModel):
    """Result of validation operations"""
    valid: bool
    data: Optional[Union[Task, Event]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# High-risk actions that require human approval
HIGH_RISK_ACTIONS = {
    "send_invite",
    "delete_event", 
    "modify_external_calendar",
    "cancel_recurring_series",
    "share_calendar",
    "change_permissions",
    "bulk_delete",
    "mass_invite"
}

def validate_task(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate task data against Task schema.
    
    Args:
        data: Dictionary containing task data
        
    Returns:
        ValidationResult with validation status and cleaned data
    """
    
    try:
        # Create and validate Task object
        task = Task(**data)
        
        return ValidationResult(
            valid=True,
            data=task,
            errors=[],
            warnings=[]
        )
        
    except ValidationError as e:
        errors = []
        warnings = []
        
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            error_type = error['type']
            
            error_text = f"{field}: {message}"
            
            # Some validation errors are warnings rather than hard failures
            if error_type in ['value_error.past_date']:
                warnings.append(error_text)
            else:
                errors.append(error_text)
        
        return ValidationResult(
            valid=len(errors) == 0,
            data=None,
            errors=errors,
            warnings=warnings
        )
    
    except Exception as e:
        return ValidationResult(
            valid=False,
            data=None,
            errors=[f"Unexpected validation error: {str(e)}"],
            warnings=[]
        )

def validate_event(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate event data against Event schema.
    
    Args:
        data: Dictionary containing event data
        
    Returns:
        ValidationResult with validation status and cleaned data
    """
    
    try:
        # Create and validate Event object
        event = Event(**data)
        
        # Additional business logic validation
        warnings = []
        
        # Check for very short events (might be mistakes)
        duration = event.end_datetime - event.start_datetime
        if duration.total_seconds() < 300:  # Less than 5 minutes
            warnings.append("Event duration is very short (less than 5 minutes)")
        
        # Check for very long events
        if duration.total_seconds() > 86400:  # More than 24 hours
            warnings.append("Event duration is very long (more than 24 hours)")
        
        # Check for events outside business hours (as info only)
        start_hour = event.start_datetime.hour
        if start_hour < 6 or start_hour > 22:
            warnings.append("Event is scheduled outside typical business hours")
        
        return ValidationResult(
            valid=True,
            data=event,
            errors=[],
            warnings=warnings
        )
        
    except ValidationError as e:
        errors = []
        warnings = []
        
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            error_type = error['type']
            
            error_text = f"{field}: {message}"
            
            # Some validation errors are warnings
            if error_type in ['value_error.past_date', 'value_error.business_hours']:
                warnings.append(error_text)
            else:
                errors.append(error_text)
        
        return ValidationResult(
            valid=len(errors) == 0,
            data=None,
            errors=errors,
            warnings=warnings
        )
    
    except Exception as e:
        return ValidationResult(
            valid=False,
            data=None,
            errors=[f"Unexpected validation error: {str(e)}"],
            warnings=[]
        )

def requires_human_approval(action: str, params: Dict[str, Any] = None) -> bool:
    """
    Determine if an action requires human approval.
    
    Args:
        action: The action being performed
        params: Action parameters for context-sensitive decisions
        
    Returns:
        True if human approval is required
    """
    
    # Check against high-risk actions list
    if action.lower() in HIGH_RISK_ACTIONS:
        return True
    
    # Context-sensitive approval requirements
    if params:
        # Large number of attendees
        if action == "create_event" and "attendees" in params:
            if len(params["attendees"]) > 10:
                return True
        
        # Events with external attendees
        if action in ["create_event", "update_event"] and "attendees" in params:
            for attendee in params.get("attendees", []):
                # Simple heuristic: external if not from common internal domains
                if "@" in attendee and not any(domain in attendee for domain in [
                    "@company.com", "@internal.com"  # Add your internal domains
                ]):
                    return True
        
        # Very expensive or long events
        if action == "create_event":
            start_dt = params.get("start_datetime")
            end_dt = params.get("end_datetime")
            if start_dt and end_dt:
                try:
                    start = datetime.fromisoformat(start_dt) if isinstance(start_dt, str) else start_dt
                    end = datetime.fromisoformat(end_dt) if isinstance(end_dt, str) else end_dt
                    duration = end - start
                    
                    # Events longer than 4 hours
                    if duration.total_seconds() > 14400:
                        return True
                except:
                    pass
        
        # Recurring events with many occurrences
        if action == "create_event" and params.get("recurrence_count", 0) > 20:
            return True
    
    return False

def create_approval_summary(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary for human approval.
    
    Args:
        action: The action being performed
        params: Action parameters
        
    Returns:
        Dictionary with approval summary information
    """
    
    summary = {
        "action": action,
        "requires_approval": requires_human_approval(action, params),
        "risk_level": "low",
        "summary": "",
        "details": [],
        "warnings": []
    }
    
    # Determine risk level
    if action in HIGH_RISK_ACTIONS:
        summary["risk_level"] = "high"
    elif requires_human_approval(action, params):
        summary["risk_level"] = "medium"
    
    # Create human-readable summary
    if action == "create_event":
        title = params.get("title", "Untitled Event")
        start = params.get("start_datetime", "Unknown time")
        attendees = params.get("attendees", [])
        
        summary["summary"] = f"Create event '{title}' starting {start}"
        summary["details"] = [
            f"Title: {title}",
            f"Start: {start}",
            f"End: {params.get('end_datetime', 'Unknown')}",
            f"Attendees: {len(attendees)} people" if attendees else "No attendees",
            f"Location: {params.get('location', 'Not specified')}"
        ]
        
        if len(attendees) > 5:
            summary["warnings"].append(f"Large number of attendees ({len(attendees)})")
        
    elif action == "delete_event":
        summary["summary"] = f"Delete event '{params.get('title', 'Unknown event')}'"
        summary["details"] = [
            f"Event ID: {params.get('event_id', 'Unknown')}",
            f"Title: {params.get('title', 'Unknown')}"
        ]
        summary["warnings"].append("This action cannot be undone")
    
    elif action == "send_invite":
        attendees = params.get("attendees", [])
        summary["summary"] = f"Send calendar invite to {len(attendees)} people"
        summary["details"] = [
            f"Recipients: {', '.join(attendees[:3])}" + ("..." if len(attendees) > 3 else ""),
            f"Event: {params.get('title', 'Unknown event')}"
        ]
        
        if len(attendees) > 10:
            summary["warnings"].append("Sending to large number of recipients")
    
    else:
        summary["summary"] = f"Perform action: {action}"
        summary["details"] = [f"Parameters: {params}"]
    
    return summary 
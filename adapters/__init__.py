"""
Calendar Adapters Module

Consolidated module for all calendar service adapters.
Provides unified imports and factory functions for creating calendar adapters.
"""

from .base_calendar import BaseCalendarAdapter
from .google_calendar import GoogleCalendarAdapter

# Import future adapters when implemented
try:
    from .apple_calendar import AppleCalendarAdapter
except ImportError:
    AppleCalendarAdapter = None

# Adapter registry for dynamic creation
AVAILABLE_ADAPTERS = {
    "google": GoogleCalendarAdapter,
    "google_calendar": GoogleCalendarAdapter,
}

# Add Apple Calendar if available
if AppleCalendarAdapter:
    AVAILABLE_ADAPTERS.update({
        "apple": AppleCalendarAdapter,
        "apple_calendar": AppleCalendarAdapter,
    })

def create_calendar_adapter(adapter_type: str, config: dict = None) -> BaseCalendarAdapter:
    """
    Factory function to create calendar adapters.
    
    Args:
        adapter_type: Type of adapter ("google", "apple", etc.)
        config: Configuration dictionary for the adapter
        
    Returns:
        BaseCalendarAdapter instance
        
    Raises:
        ValueError: If adapter type is not supported
    """
    
    adapter_type = adapter_type.lower()
    
    if adapter_type not in AVAILABLE_ADAPTERS:
        available = ", ".join(AVAILABLE_ADAPTERS.keys())
        raise ValueError(f"Unsupported adapter type: {adapter_type}. Available: {available}")
    
    adapter_class = AVAILABLE_ADAPTERS[adapter_type]
    return adapter_class(config or {})

def get_available_adapters() -> list:
    """Return list of available adapter types"""
    return list(AVAILABLE_ADAPTERS.keys())

# Make key classes available at module level
__all__ = [
    "BaseCalendarAdapter",
    "GoogleCalendarAdapter", 
    "AppleCalendarAdapter",
    "create_calendar_adapter",
    "get_available_adapters",
    "AVAILABLE_ADAPTERS"
] 
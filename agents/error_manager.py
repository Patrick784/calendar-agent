"""
Error/Retry Manager

Implements robust error handling: exponential backoff for API errors,
rate-limit detection, and fallbacks such as the regex parser to avoid
blocking the user. Logs all tool calls with structured JSON.
"""

import asyncio
import time
import logging
import random
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

class ErrorType(Enum):
    """Classification of error types for appropriate handling"""
    NETWORK_ERROR = "network_error"
    API_RATE_LIMIT = "api_rate_limit"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    AUTHENTICATION_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT_ERROR = "timeout_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.0

@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    attempt: int
    error_type: ErrorType
    error_message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ErrorRetryManager:
    """
    Centralized error handling and retry logic for the multi-agent system.
    
    Features:
    - Exponential backoff with jitter
    - Rate limit detection and handling
    - Circuit breaker pattern
    - Fallback strategy management
    - Structured error logging
    - Performance metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("error_manager")
        
        # Error tracking
        self._error_counts: Dict[str, int] = {}
        self._last_error_times: Dict[str, datetime] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Default retry policies for different operations
        self._default_policies = {
            "llm_call": RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0),
            "api_call": RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=60.0),
            "database_operation": RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=5.0),
            "file_operation": RetryPolicy(max_attempts=2, base_delay=0.1, max_delay=1.0)
        }
        
        # Fallback strategies
        self._fallback_strategies: Dict[str, List[Callable]] = {}
        
        # Performance metrics
        self._metrics = {
            "total_operations": 0,
            "total_errors": 0,
            "total_retries": 0,
            "avg_retry_delay": 0.0,
            "circuit_breaker_activations": 0
        }
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        operation_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        fallback_strategies: Optional[List[Callable]] = None
    ) -> Any:
        """
        Execute an operation with retry logic and fallback strategies.
        
        Args:
            operation: Async function to execute
            operation_name: Name for logging and metrics
            retry_policy: Custom retry policy (uses default if None)
            fallback_strategies: List of fallback functions to try if all retries fail
        
        Returns:
            Result of the operation or fallback
        
        Raises:
            Exception: If all retries and fallbacks fail
        """
        
        self._metrics["total_operations"] += 1
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(operation_name):
            self.logger.warning(f"Circuit breaker open for {operation_name}, trying fallback")
            return await self._execute_fallback(operation_name, fallback_strategies)
        
        # Use default policy if none provided
        if retry_policy is None:
            retry_policy = self._get_default_policy(operation_name)
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(retry_policy.max_attempts):
            try:
                # Log attempt
                self.logger.debug(f"Executing {operation_name}, attempt {attempt + 1}/{retry_policy.max_attempts}")
                
                # Execute the operation
                result = await operation()
                
                # Success - reset error tracking
                self._reset_error_tracking(operation_name)
                
                # Log success metrics
                execution_time = time.time() - start_time
                self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}, took {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)
                error_context = ErrorContext(
                    operation=operation_name,
                    attempt=attempt + 1,
                    error_type=error_type,
                    error_message=str(e),
                    metadata={"execution_time": time.time() - start_time}
                )
                
                # Log the error
                await self._log_error(error_context)
                
                # Update error tracking
                self._update_error_tracking(operation_name, error_type)
                
                # Check if we should retry
                if attempt + 1 >= retry_policy.max_attempts:
                    self.logger.error(f"Operation {operation_name} failed after {retry_policy.max_attempts} attempts")
                    break
                
                if not self._should_retry(error_type):
                    self.logger.error(f"Operation {operation_name} failed with non-retryable error: {error_type}")
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, retry_policy, error_type)
                
                self.logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {str(e)}")
                
                # Wait before retry
                await asyncio.sleep(delay)
                self._metrics["total_retries"] += 1
        
        # All retries failed, try fallback strategies
        self._metrics["total_errors"] += 1
        
        if fallback_strategies or operation_name in self._fallback_strategies:
            try:
                return await self._execute_fallback(operation_name, fallback_strategies)
            except Exception as fallback_error:
                self.logger.error(f"All fallbacks failed for {operation_name}: {str(fallback_error)}")
        
        # Check if we should activate circuit breaker
        self._check_circuit_breaker(operation_name)
        
        # No fallback available or fallback failed
        raise last_exception or Exception(f"Operation {operation_name} failed with unknown error")
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error to determine appropriate handling strategy"""
        
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Network and connection errors
        if any(keyword in error_str for keyword in ["connection", "network", "dns", "resolve"]):
            return ErrorType.NETWORK_ERROR
        
        # Rate limiting
        if any(keyword in error_str for keyword in ["rate limit", "too many requests", "429"]):
            return ErrorType.API_RATE_LIMIT
        
        # Quota exceeded
        if any(keyword in error_str for keyword in ["quota", "usage limit", "billing"]):
            return ErrorType.API_QUOTA_EXCEEDED
        
        # Authentication errors
        if any(keyword in error_str for keyword in ["auth", "unauthorized", "401", "403", "token", "key"]):
            return ErrorType.AUTHENTICATION_ERROR
        
        # Timeout errors
        if any(keyword in error_str for keyword in ["timeout", "timed out"]) or "timeout" in error_type_name:
            return ErrorType.TIMEOUT_ERROR
        
        # Service unavailable
        if any(keyword in error_str for keyword in ["503", "service unavailable", "server error", "502", "504"]):
            return ErrorType.SERVICE_UNAVAILABLE
        
        # Invalid request
        if any(keyword in error_str for keyword in ["400", "bad request", "invalid", "malformed"]):
            return ErrorType.INVALID_REQUEST
        
        return ErrorType.UNKNOWN_ERROR
    
    def _should_retry(self, error_type: ErrorType) -> bool:
        """Determine if an error type should be retried"""
        
        # Don't retry these error types
        non_retryable = {
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.INVALID_REQUEST,
            ErrorType.API_QUOTA_EXCEEDED
        }
        
        return error_type not in non_retryable
    
    def _calculate_delay(self, attempt: int, policy: RetryPolicy, error_type: ErrorType) -> float:
        """Calculate delay before next retry attempt"""
        
        # Base exponential backoff
        delay = policy.base_delay * (policy.exponential_base ** attempt) * policy.backoff_multiplier
        
        # Apply special handling for specific error types
        if error_type == ErrorType.API_RATE_LIMIT:
            # Longer delays for rate limiting
            delay *= 2.0
        elif error_type == ErrorType.SERVICE_UNAVAILABLE:
            # Even longer delays for service issues
            delay *= 3.0
        
        # Cap at max delay
        delay = min(delay, policy.max_delay)
        
        # Add jitter to avoid thundering herd
        if policy.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return delay
    
    def _get_default_policy(self, operation_name: str) -> RetryPolicy:
        """Get default retry policy for an operation"""
        
        # Match operation name to policy
        for policy_key, policy in self._default_policies.items():
            if policy_key in operation_name.lower():
                return policy
        
        # Default fallback policy
        return RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0)
    
    async def _log_error(self, error_context: ErrorContext):
        """Log error with structured format for observability"""
        
        log_data = {
            "timestamp": error_context.timestamp.isoformat(),
            "operation": error_context.operation,
            "attempt": error_context.attempt,
            "error_type": error_context.error_type.value,
            "error_message": error_context.error_message,
            "metadata": error_context.metadata
        }
        
        self.logger.error(f"Operation error: {log_data}")
        
        # Also log to structured logging system if configured
        if self.config.get("structured_logging_enabled"):
            await self._send_to_observability_system(log_data)
    
    async def _send_to_observability_system(self, log_data: Dict[str, Any]):
        """Send error data to observability system (OpenTelemetry, Sentry, etc.)"""
        
        # Placeholder for integration with observability systems
        # You would implement actual integration here
        pass
    
    def _update_error_tracking(self, operation_name: str, error_type: ErrorType):
        """Update error tracking for circuit breaker logic"""
        
        current_time = datetime.utcnow()
        
        # Update error count
        self._error_counts[operation_name] = self._error_counts.get(operation_name, 0) + 1
        self._last_error_times[operation_name] = current_time
    
    def _reset_error_tracking(self, operation_name: str):
        """Reset error tracking after successful operation"""
        
        if operation_name in self._error_counts:
            del self._error_counts[operation_name]
        if operation_name in self._last_error_times:
            del self._last_error_times[operation_name]
        
        # Reset circuit breaker if it was open
        if operation_name in self._circuit_breakers:
            self._circuit_breakers[operation_name]["state"] = "closed"
    
    def _check_circuit_breaker(self, operation_name: str):
        """Check if circuit breaker should be activated"""
        
        error_count = self._error_counts.get(operation_name, 0)
        error_threshold = self.config.get("circuit_breaker_threshold", 5)
        
        if error_count >= error_threshold:
            self._activate_circuit_breaker(operation_name)
    
    def _activate_circuit_breaker(self, operation_name: str):
        """Activate circuit breaker for an operation"""
        
        timeout_minutes = self.config.get("circuit_breaker_timeout_minutes", 5)
        timeout_time = datetime.utcnow() + timedelta(minutes=timeout_minutes)
        
        self._circuit_breakers[operation_name] = {
            "state": "open",
            "activated_at": datetime.utcnow(),
            "timeout_at": timeout_time
        }
        
        self._metrics["circuit_breaker_activations"] += 1
        
        self.logger.warning(f"Circuit breaker activated for {operation_name}, timeout at {timeout_time}")
    
    def _is_circuit_breaker_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for an operation"""
        
        if operation_name not in self._circuit_breakers:
            return False
        
        breaker = self._circuit_breakers[operation_name]
        
        if breaker["state"] == "closed":
            return False
        
        # Check if timeout has passed
        if datetime.utcnow() >= breaker["timeout_at"]:
            # Move to half-open state
            breaker["state"] = "half-open"
            self.logger.info(f"Circuit breaker for {operation_name} moved to half-open state")
            return False
        
        return True
    
    async def _execute_fallback(
        self,
        operation_name: str,
        custom_strategies: Optional[List[Callable]] = None
    ) -> Any:
        """Execute fallback strategies when primary operation fails"""
        
        strategies = custom_strategies or self._fallback_strategies.get(operation_name, [])
        
        if not strategies:
            raise Exception(f"No fallback strategies available for {operation_name}")
        
        last_exception = None
        
        for i, strategy in enumerate(strategies):
            try:
                self.logger.info(f"Trying fallback strategy {i + 1} for {operation_name}")
                
                if asyncio.iscoroutinefunction(strategy):
                    result = await strategy()
                else:
                    result = strategy()
                
                self.logger.info(f"Fallback strategy {i + 1} succeeded for {operation_name}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Fallback strategy {i + 1} failed for {operation_name}: {str(e)}")
        
        raise last_exception or Exception(f"All fallback strategies failed for {operation_name}")
    
    def register_fallback_strategy(self, operation_name: str, strategy: Callable):
        """Register a fallback strategy for an operation"""
        
        if operation_name not in self._fallback_strategies:
            self._fallback_strategies[operation_name] = []
        
        self._fallback_strategies[operation_name].append(strategy)
        self.logger.info(f"Registered fallback strategy for {operation_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error handling and retry metrics"""
        
        current_time = datetime.utcnow()
        
        # Calculate error rates
        recent_errors = sum(
            1 for error_time in self._last_error_times.values()
            if current_time - error_time < timedelta(hours=1)
        )
        
        metrics = self._metrics.copy()
        metrics.update({
            "recent_error_rate": recent_errors,
            "active_circuit_breakers": len([
                name for name, breaker in self._circuit_breakers.items()
                if breaker["state"] in ["open", "half-open"]
            ]),
            "error_counts_by_operation": self._error_counts.copy()
        })
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of error manager"""
        
        return {
            "status": "healthy",
            "metrics": self.get_metrics(),
            "circuit_breakers": {
                name: {
                    "state": breaker["state"],
                    "activated_at": breaker["activated_at"].isoformat() if breaker.get("activated_at") else None
                }
                for name, breaker in self._circuit_breakers.items()
            }
        } 
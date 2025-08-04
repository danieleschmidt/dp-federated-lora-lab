"""
Error handling and recovery system for DP-Federated LoRA.

This module provides comprehensive error handling, recovery mechanisms,
and fault tolerance for distributed federated learning.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .exceptions import (
    DPFederatedLoRAError,
    NetworkError,
    AuthenticationError,
    PrivacyBudgetError,
    ModelError,
    DataError,
    AggregationError,
    ClientError,
    ServerError,
    TrainingError,
    SecurityError,
    ResourceError,
    TimeoutError,
    ValidationError,
    ByzantineError,
    CommunicationError,
    RegistrationError,
    SynchronizationError,
    MonitoringError,
    ErrorContext,
    ErrorSeverity,
    create_error_with_context
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    half_open_max_calls: int = 3


class CircuitBreakerState:
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: str = field(default=CircuitBreakerState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_calls: int = field(default=0)
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def execute_call(self) -> None:
        """Mark call execution for half-open state."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.error_callbacks: List[Callable] = []
        self.monitoring_enabled = True
    
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable
    ) -> None:
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback for monitoring."""
        self.error_callbacks.append(callback)
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(config or CircuitBreakerConfig())
        return self.circuit_breakers[name]
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        context: ErrorContext,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with comprehensive error handling and recovery.
        
        Args:
            operation: Function to execute
            context: Error context information
            retry_config: Retry configuration
            circuit_breaker_name: Circuit breaker name
            *args, **kwargs: Arguments for operation
            
        Returns:
            Operation result
            
        Raises:
            DPFederatedLoRAError: If all recovery attempts fail
        """
        retry_config = retry_config or RetryConfig()
        
        # Check circuit breaker
        if circuit_breaker_name:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
            if not circuit_breaker.can_execute():
                raise create_error_with_context(
                    ServerError,
                    f"Circuit breaker {circuit_breaker_name} is open",
                    context
                )
            circuit_breaker.execute_call()
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                context.retry_count = attempt
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Record success
                if circuit_breaker_name:
                    circuit_breaker.record_success()
                
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                context.increment_retry()
                
                # Notify callbacks about error
                await self._notify_error_callbacks(e, context)
                
                # Check if we should retry
                if not self._should_retry(e, context, attempt, retry_config):
                    break
                
                # Apply recovery strategy if available
                recovery_applied = await self._apply_recovery_strategy(e, context)
                
                # Calculate delay and wait
                if attempt < retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, retry_config)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
        
        # All attempts failed
        if circuit_breaker_name:
            circuit_breaker.record_failure()
        
        # Create final error with context
        final_error = create_error_with_context(
            type(last_exception) if isinstance(last_exception, DPFederatedLoRAError) else ServerError,
            f"Operation failed after {retry_config.max_attempts} attempts",
            context,
            last_exception
        )
        
        logger.error(f"Operation permanently failed: {final_error}")
        raise final_error
    
    def _should_retry(
        self,
        exception: Exception,
        context: ErrorContext,
        attempt: int,
        retry_config: RetryConfig
    ) -> bool:
        """Determine if operation should be retried."""
        # Don't retry if max attempts reached
        if attempt >= retry_config.max_attempts - 1:
            return False
        
        # Don't retry if context says not recoverable
        if not context.can_retry():
            return False
        
        # Don't retry certain error types
        non_retryable_errors = (
            AuthenticationError,
            ValidationError,
            SecurityError,
            PrivacyBudgetError
        )
        
        if isinstance(exception, non_retryable_errors):
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * (attempt + 1)
        else:  # fixed
            delay = config.base_delay
        
        # Apply maximum delay
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    async def _apply_recovery_strategy(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> bool:
        """Apply recovery strategy for the exception type."""
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(exception, error_type):
                try:
                    if asyncio.iscoroutinefunction(strategy):
                        await strategy(exception, context)
                    else:
                        strategy(exception, context)
                    logger.info(f"Applied recovery strategy for {error_type.__name__}")
                    return True
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
        return False
    
    async def _notify_error_callbacks(
        self,
        exception: Exception,
        context: ErrorContext
    ) -> None:
        """Notify error callbacks about the error."""
        if not self.monitoring_enabled:
            return
        
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(exception, context)
                else:
                    callback(exception, context)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    component: str,
    operation: str,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    severity: str = ErrorSeverity.ERROR,
    recoverable: bool = True
):
    """
    Decorator for automatic error handling and recovery.
    
    Args:
        component: Component name
        operation: Operation name
        retry_config: Retry configuration
        circuit_breaker_name: Circuit breaker name
        severity: Error severity
        recoverable: Whether error is recoverable
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                severity=severity,
                recoverable=recoverable
            )
            
            # Extract client_id and round_num from args/kwargs if available
            if args and hasattr(args[0], 'client_id'):
                context.client_id = args[0].client_id
            if args and hasattr(args[0], 'current_round'):
                context.round_num = args[0].current_round
            
            return await error_handler.execute_with_recovery(
                func,
                context,
                retry_config,
                circuit_breaker_name,
                *args,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@asynccontextmanager
async def error_boundary(
    component: str,
    operation: str,
    client_id: Optional[str] = None,
    round_num: Optional[int] = None,
    severity: str = ErrorSeverity.ERROR
):
    """
    Async context manager for error boundary.
    
    Args:
        component: Component name
        operation: Operation name
        client_id: Client identifier
        round_num: Round number
        severity: Error severity
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        client_id=client_id,
        round_num=round_num,
        severity=severity
    )
    
    try:
        yield context
    except Exception as e:
        # Notify error callbacks
        await error_handler._notify_error_callbacks(e, context)
        
        # Re-raise with context if it's not already a DP-Federated LoRA error
        if not isinstance(e, DPFederatedLoRAError):
            raise create_error_with_context(
                ServerError if "server" in component.lower() else ClientError,
                f"Error in {component}.{operation}",
                context,
                e
            )
        raise


def setup_default_recovery_strategies():
    """Setup default recovery strategies for common errors."""
    
    def network_recovery(exception: NetworkError, context: ErrorContext):
        """Recovery strategy for network errors."""
        logger.info(f"Applying network recovery for {context.component}")
        # Could implement connection reset, DNS refresh, etc.
    
    def resource_recovery(exception: ResourceError, context: ErrorContext):
        """Recovery strategy for resource errors."""
        logger.info(f"Applying resource recovery for {context.component}")
        # Could implement garbage collection, memory cleanup, etc.
        import gc
        gc.collect()
    
    def model_recovery(exception: ModelError, context: ErrorContext):
        """Recovery strategy for model errors."""
        logger.info(f"Applying model recovery for {context.component}")
        # Could implement model reloading, checkpoint restoration, etc.
    
    error_handler.register_recovery_strategy(NetworkError, network_recovery)
    error_handler.register_recovery_strategy(ResourceError, resource_recovery)
    error_handler.register_recovery_strategy(ModelError, model_recovery)


# Initialize default recovery strategies
setup_default_recovery_strategies()
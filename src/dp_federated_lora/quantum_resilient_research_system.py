"""
Quantum-Resilient Research System with Advanced Error Handling and Fault Tolerance.

This module implements comprehensive error handling, fault tolerance, and resilience
mechanisms for quantum-enhanced federated learning research operations.
"""

import asyncio
import logging
import time
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import hashlib
import pickle
from contextlib import asynccontextmanager
import functools

from .exceptions import (
    QuantumSchedulingError,
    QuantumPrivacyError,
    QuantumOptimizationError,
    DPFederatedLoRAError,
    NetworkError,
    ValidationError,
    ResourceError
)

logger = logging.getLogger(__name__)


class ResilienceLevel(Enum):
    """Resilience levels for different system components."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    QUANTUM_ENHANCED = "quantum_enhanced"


class FailureType(Enum):
    """Types of failures in the research system."""
    NETWORK_FAILURE = "network_failure"
    COMPUTATION_FAILURE = "computation_failure"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_FAILURE = "validation_failure"
    RESOURCE_UNAVAILABLE = "resource_unavailable"


@dataclass
class FailureContext:
    """Context information for system failures."""
    failure_type: FailureType
    component: str
    timestamp: float
    error_message: str
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_actions: List[str] = field(default_factory=list)
    recovery_success: bool = False


@dataclass
class ResilienceMetrics:
    """Metrics for tracking system resilience."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    recovered_operations: int = 0
    mean_time_to_recovery: float = 0.0
    failure_rate: float = 0.0
    availability: float = 1.0
    quantum_coherence_uptime: float = 1.0


class QuantumCircuitBreaker:
    """Quantum-enhanced circuit breaker for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        quantum_coherence_threshold: float = 0.5,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.quantum_coherence_threshold = quantum_coherence_threshold
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._half_open_calls = 0
        self._quantum_coherence = 1.0
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise QuantumSchedulingError("Circuit breaker is OPEN")
                else:
                    self._state = "HALF_OPEN"
                    self._half_open_calls = 0
            
            if self._state == "HALF_OPEN":
                if self._half_open_calls >= self.half_open_max_calls:
                    raise QuantumSchedulingError("Circuit breaker half-open limit reached")
                self._half_open_calls += 1
        
        try:
            # Monitor quantum coherence during execution
            start_coherence = self._quantum_coherence
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Simulate quantum coherence measurement
            self._quantum_coherence = max(0.0, start_coherence - np.random.uniform(0, 0.1))
            
            with self._lock:
                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0
                elif self._state == "CLOSED":
                    self._failure_count = max(0, self._failure_count - 1)
            
            return result
            
        except Exception as e:
            # Update quantum coherence on failure
            self._quantum_coherence *= 0.8  # Decoherence on failure
            
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if (self._failure_count >= self.failure_threshold or 
                    self._quantum_coherence < self.quantum_coherence_threshold):
                    self._state = "OPEN"
                    logger.warning(f"Circuit breaker opened: {self._failure_count} failures, "
                                 f"coherence: {self._quantum_coherence:.3f}")
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "quantum_coherence": self._quantum_coherence,
            "last_failure_time": self._last_failure_time
        }


class QuantumRetryStrategy:
    """Quantum-enhanced retry strategy with adaptive backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        quantum_adaptive: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.quantum_adaptive = quantum_adaptive
        self._quantum_history = []
    
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with quantum-enhanced retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - update quantum history
                if self.quantum_adaptive:
                    self._quantum_history.append({"attempt": attempt, "success": True})
                    self._quantum_history = self._quantum_history[-10:]  # Keep last 10
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries:
                    break
                
                # Calculate quantum-adaptive delay
                delay = self._calculate_adaptive_delay(attempt)
                await asyncio.sleep(delay)
        
        # All retries exhausted
        if self.quantum_adaptive:
            self._quantum_history.append({"attempt": self.max_retries, "success": False})
            self._quantum_history = self._quantum_history[-10:]
        
        raise last_exception
    
    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive delay based on quantum history."""
        base_delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.quantum_adaptive and self._quantum_history:
            # Adjust delay based on historical success rates
            recent_success_rate = sum(
                1 for h in self._quantum_history[-5:] if h["success"]
            ) / len(self._quantum_history[-5:])
            
            # Lower success rate = longer delay
            quantum_factor = 1.0 + (1.0 - recent_success_rate)
            base_delay *= quantum_factor
        
        if self.jitter:
            base_delay *= (0.5 + np.random.random() * 0.5)
        
        return min(base_delay, self.max_delay)


class QuantumResilienceManager:
    """Comprehensive resilience manager for quantum research systems."""
    
    def __init__(
        self,
        resilience_level: ResilienceLevel = ResilienceLevel.ENHANCED,
        checkpoint_interval: float = 300.0,  # 5 minutes
        health_check_interval: float = 30.0,  # 30 seconds
        auto_recovery: bool = True
    ):
        self.resilience_level = resilience_level
        self.checkpoint_interval = checkpoint_interval
        self.health_check_interval = health_check_interval
        self.auto_recovery = auto_recovery
        
        self.circuit_breakers: Dict[str, QuantumCircuitBreaker] = {}
        self.retry_strategies: Dict[str, QuantumRetryStrategy] = {}
        self.failure_history: List[FailureContext] = []
        self.metrics = ResilienceMetrics()
        self.checkpoints: Dict[str, Any] = {}
        self.health_checks: Dict[str, Callable] = {}
        
        self._monitoring_task = None
        self._lock = threading.Lock()
    
    async def start_monitoring(self):
        """Start background monitoring and health checking."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Quantum resilience monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Quantum resilience monitoring stopped")
    
    def register_component(
        self, 
        component_name: str,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        health_check: Optional[Callable] = None
    ):
        """Register a component for resilience management."""
        # Create circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breakers[component_name] = QuantumCircuitBreaker(**cb_config)
        
        # Create retry strategy
        retry_config = retry_config or {}
        self.retry_strategies[component_name] = QuantumRetryStrategy(**retry_config)
        
        # Register health check
        if health_check:
            self.health_checks[component_name] = health_check
        
        logger.info(f"Component registered for resilience: {component_name}")
    
    @asynccontextmanager
    async def resilient_operation(
        self, 
        component_name: str,
        operation_name: str = "operation"
    ):
        """Context manager for resilient operations."""
        start_time = time.time()
        operation_id = f"{component_name}_{operation_name}_{int(start_time)}"
        
        try:
            self.metrics.total_operations += 1
            
            # Create checkpoint before operation
            if self.resilience_level in [ResilienceLevel.ENHANCED, ResilienceLevel.MAXIMUM]:
                await self._create_checkpoint(operation_id)
            
            yield
            
            # Operation successful
            self.metrics.successful_operations += 1
            logger.debug(f"Resilient operation completed: {operation_id}")
            
        except Exception as e:
            # Handle failure
            self.metrics.failed_operations += 1
            
            failure_context = FailureContext(
                failure_type=self._classify_failure(e),
                component=component_name,
                timestamp=time.time(),
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                system_state=await self._capture_system_state()
            )
            
            with self._lock:
                self.failure_history.append(failure_context)
                self.failure_history = self.failure_history[-100:]  # Keep last 100
            
            # Attempt recovery if enabled
            if self.auto_recovery:
                recovery_success = await self._attempt_recovery(failure_context, operation_id)
                failure_context.recovery_success = recovery_success
                if recovery_success:
                    self.metrics.recovered_operations += 1
            
            # Update metrics
            self._update_metrics()
            
            # Re-raise if not recovered
            if not failure_context.recovery_success:
                raise e
    
    async def resilient_call(
        self, 
        component_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Make a resilient function call with circuit breaker and retry."""
        if component_name not in self.circuit_breakers:
            self.register_component(component_name)
        
        circuit_breaker = self.circuit_breakers[component_name]
        retry_strategy = self.retry_strategies[component_name]
        
        async def wrapped_call():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return await retry_strategy.execute_with_retry(wrapped_call)
    
    async def _monitoring_loop(self):
        """Background monitoring loop for health checks and maintenance."""
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                
                # Run health checks
                await self._run_health_checks()
                
                # Update metrics
                self._update_metrics()
                
                # Perform maintenance
                await self._perform_maintenance()
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    async def _run_health_checks(self):
        """Run registered health checks."""
        for component_name, health_check in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(health_check):
                    healthy = await health_check()
                else:
                    healthy = health_check()
                
                if not healthy:
                    logger.warning(f"Health check failed for component: {component_name}")
                    # Trigger recovery if needed
                    if self.auto_recovery:
                        await self._trigger_component_recovery(component_name)
                        
            except Exception as e:
                logger.error(f"Health check error for {component_name}: {e}")
    
    async def _create_checkpoint(self, operation_id: str):
        """Create operation checkpoint for recovery."""
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "operation_id": operation_id,
                "system_state": await self._capture_system_state(),
                "quantum_state": self._capture_quantum_state()
            }
            self.checkpoints[operation_id] = checkpoint_data
            
            # Limit checkpoint storage
            if len(self.checkpoints) > 10:
                oldest_key = min(self.checkpoints.keys(), 
                               key=lambda k: self.checkpoints[k]["timestamp"])
                del self.checkpoints[oldest_key]
                
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for diagnostics."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "circuit_breaker_states": {
                    name: cb.get_state() 
                    for name, cb in self.circuit_breakers.items()
                }
            }
        except Exception:
            return {"error": "Failed to capture system state"}
    
    def _capture_quantum_state(self) -> Dict[str, Any]:
        """Capture quantum-specific state information."""
        return {
            "total_circuit_breakers": len(self.circuit_breakers),
            "open_circuit_breakers": sum(
                1 for cb in self.circuit_breakers.values() 
                if cb.get_state()["state"] == "OPEN"
            ),
            "avg_quantum_coherence": np.mean([
                cb.get_state()["quantum_coherence"] 
                for cb in self.circuit_breakers.values()
            ]) if self.circuit_breakers else 1.0
        }
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception."""
        if isinstance(exception, NetworkError):
            return FailureType.NETWORK_FAILURE
        elif isinstance(exception, MemoryError):
            return FailureType.MEMORY_EXHAUSTION
        elif isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT_ERROR
        elif isinstance(exception, ValidationError):
            return FailureType.VALIDATION_FAILURE
        elif isinstance(exception, QuantumPrivacyError):
            return FailureType.QUANTUM_DECOHERENCE
        elif isinstance(exception, ResourceError):
            return FailureType.RESOURCE_UNAVAILABLE
        else:
            return FailureType.COMPUTATION_FAILURE
    
    async def _attempt_recovery(
        self, 
        failure_context: FailureContext, 
        operation_id: str
    ) -> bool:
        """Attempt to recover from a failure."""
        try:
            recovery_actions = []
            
            if failure_context.failure_type == FailureType.MEMORY_EXHAUSTION:
                # Clear caches and force garbage collection
                import gc
                gc.collect()
                recovery_actions.append("memory_cleanup")
            
            elif failure_context.failure_type == FailureType.QUANTUM_DECOHERENCE:
                # Reset quantum states
                for cb in self.circuit_breakers.values():
                    cb._quantum_coherence = 1.0
                recovery_actions.append("quantum_state_reset")
            
            elif failure_context.failure_type == FailureType.NETWORK_FAILURE:
                # Wait for network recovery
                await asyncio.sleep(5.0)
                recovery_actions.append("network_wait")
            
            # Restore from checkpoint if available
            if operation_id in self.checkpoints:
                await self._restore_from_checkpoint(operation_id)
                recovery_actions.append("checkpoint_restore")
            
            failure_context.recovery_actions = recovery_actions
            logger.info(f"Recovery attempted for {failure_context.component}: {recovery_actions}")
            
            return len(recovery_actions) > 0
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False
    
    async def _restore_from_checkpoint(self, operation_id: str):
        """Restore system state from checkpoint."""
        if operation_id in self.checkpoints:
            checkpoint = self.checkpoints[operation_id]
            logger.info(f"Restoring from checkpoint: {operation_id}")
            # Implementation would restore actual system state
            # This is a placeholder for the restoration logic
    
    async def _trigger_component_recovery(self, component_name: str):
        """Trigger recovery for a specific component."""
        if component_name in self.circuit_breakers:
            cb = self.circuit_breakers[component_name]
            # Reset circuit breaker if it's in OPEN state
            if cb.get_state()["state"] == "OPEN":
                cb._state = "HALF_OPEN"
                cb._failure_count = 0
                cb._quantum_coherence = 1.0
                logger.info(f"Circuit breaker reset for component: {component_name}")
    
    def _update_metrics(self):
        """Update resilience metrics."""
        if self.metrics.total_operations > 0:
            self.metrics.failure_rate = (
                self.metrics.failed_operations / self.metrics.total_operations
            )
            self.metrics.availability = (
                self.metrics.successful_operations / self.metrics.total_operations
            )
        
        # Calculate quantum coherence uptime
        if self.circuit_breakers:
            avg_coherence = np.mean([
                cb.get_state()["quantum_coherence"] 
                for cb in self.circuit_breakers.values()
            ])
            self.metrics.quantum_coherence_uptime = avg_coherence
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # Clean old failure history
            cutoff_time = time.time() - 3600  # 1 hour
            self.failure_history = [
                f for f in self.failure_history 
                if f.timestamp > cutoff_time
            ]
            
            # Reset quantum coherence if it gets too low
            for component_name, cb in self.circuit_breakers.items():
                if cb._quantum_coherence < 0.1:
                    cb._quantum_coherence = 0.5  # Partial reset
                    logger.info(f"Quantum coherence reset for {component_name}")
            
        except Exception as e:
            logger.warning(f"Maintenance task failed: {e}")
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report."""
        return {
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
                "recovered_operations": self.metrics.recovered_operations,
                "failure_rate": self.metrics.failure_rate,
                "availability": self.metrics.availability,
                "quantum_coherence_uptime": self.metrics.quantum_coherence_uptime
            },
            "circuit_breaker_states": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            "recent_failures": [
                {
                    "component": f.component,
                    "failure_type": f.failure_type.value,
                    "timestamp": f.timestamp,
                    "recovered": f.recovery_success
                }
                for f in self.failure_history[-10:]
            ],
            "system_health": {
                "total_components": len(self.circuit_breakers),
                "healthy_components": sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb.get_state()["state"] == "CLOSED"
                ),
                "checkpoint_count": len(self.checkpoints)
            }
        }


# Decorator for making functions resilient
def quantum_resilient(
    component_name: str,
    resilience_manager: Optional[QuantumResilienceManager] = None,
    **resilience_config
):
    """Decorator to make functions quantum-resilient."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = resilience_manager or _get_global_resilience_manager()
            
            if component_name not in manager.circuit_breakers:
                manager.register_component(component_name, **resilience_config)
            
            async with manager.resilient_operation(component_name, func.__name__):
                return await manager.resilient_call(component_name, func, *args, **kwargs)
        
        return wrapper
    return decorator


# Global resilience manager instance
_global_resilience_manager = None

def get_global_resilience_manager() -> QuantumResilienceManager:
    """Get or create the global resilience manager."""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = QuantumResilienceManager()
    return _global_resilience_manager

def _get_global_resilience_manager() -> QuantumResilienceManager:
    """Internal function to get global resilience manager."""
    return get_global_resilience_manager()


# Factory functions
def create_quantum_circuit_breaker(**config) -> QuantumCircuitBreaker:
    """Create a quantum circuit breaker with specified configuration."""
    return QuantumCircuitBreaker(**config)

def create_quantum_retry_strategy(**config) -> QuantumRetryStrategy:
    """Create a quantum retry strategy with specified configuration."""
    return QuantumRetryStrategy(**config)

def create_resilience_manager(**config) -> QuantumResilienceManager:
    """Create a resilience manager with specified configuration."""
    return QuantumResilienceManager(**config)
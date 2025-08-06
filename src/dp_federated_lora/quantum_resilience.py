"""
Quantum-Enhanced Resilience and Fault Tolerance

Implements circuit breakers, retry mechanisms, and fault tolerance specifically
designed for quantum-inspired federated learning components.
"""

import asyncio
import logging
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from functools import wraps
import inspect

import numpy as np
import torch
from pydantic import BaseModel, Field

from .exceptions import (
    QuantumSchedulingError, 
    QuantumPrivacyError, 
    QuantumOptimizationError,
    DPFederatedLoRAError
)
from .quantum_monitoring import QuantumMetricsCollector, QuantumMetricType
from .config import FederatedConfig


class QuantumCircuitBreakerState(Enum):
    """States for quantum-enhanced circuit breaker"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, blocking requests
    HALF_OPEN = "half_open"    # Testing if service recovered
    QUANTUM_COHERENT = "quantum_coherent"      # Quantum-enhanced recovery mode
    QUANTUM_DECOHERENT = "quantum_decoherent"  # Quantum noise detected


@dataclass
class QuantumRetryConfig:
    """Configuration for quantum-inspired retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    quantum_amplification: bool = True
    coherence_threshold: float = 0.5
    backoff_strategy: str = "quantum_exponential"  # quantum_exponential, linear, fibonacci


@dataclass
class QuantumCircuitBreakerConfig:
    """Configuration for quantum circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    quantum_coherence_threshold: float = 0.3
    decoherence_penalty_factor: float = 1.5
    success_threshold: int = 3  # Successes needed to close circuit


class QuantumRetryStrategy:
    """Quantum-inspired retry strategy with coherence-based backoff"""
    
    def __init__(self, config: QuantumRetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_delay(
        self, 
        attempt: int, 
        quantum_coherence: float = 1.0,
        last_error: Optional[Exception] = None
    ) -> float:
        """Calculate delay with quantum-inspired modifications"""
        if self.config.backoff_strategy == "quantum_exponential":
            delay = self._quantum_exponential_backoff(attempt, quantum_coherence)
        elif self.config.backoff_strategy == "linear":
            delay = self._linear_backoff(attempt)
        elif self.config.backoff_strategy == "fibonacci":
            delay = self._fibonacci_backoff(attempt)
        else:
            delay = self._quantum_exponential_backoff(attempt, quantum_coherence)
            
        # Add jitter if enabled
        if self.config.jitter:
            jitter_factor = 0.1 + 0.1 * (1 - quantum_coherence)  # More jitter with lower coherence
            delay *= (1 + random.uniform(-jitter_factor, jitter_factor))
            
        # Apply quantum amplification based on error type
        if self.config.quantum_amplification and last_error:
            delay *= self._get_quantum_error_multiplier(last_error)
            
        return min(delay, self.config.max_delay)
        
    def _quantum_exponential_backoff(self, attempt: int, coherence: float) -> float:
        """Quantum-enhanced exponential backoff"""
        base_delay = self.config.base_delay
        
        # Quantum coherence affects base delay
        coherence_factor = 1.0 + (1.0 - coherence) * 0.5  # Higher delay for lower coherence
        
        # Quantum superposition effect: sometimes shorter, sometimes longer delays
        superposition_factor = 1.0
        if coherence > self.config.coherence_threshold:
            # In coherent state, use quantum interference patterns
            phase = 2 * np.pi * attempt / 8  # 8-level quantum system
            superposition_factor = 1.0 + 0.2 * np.sin(phase)
            
        delay = base_delay * (self.config.exponential_base ** (attempt - 1))
        delay *= coherence_factor * superposition_factor
        
        return delay
        
    def _linear_backoff(self, attempt: int) -> float:
        """Linear backoff strategy"""
        return self.config.base_delay * attempt
        
    def _fibonacci_backoff(self, attempt: int) -> float:
        """Fibonacci sequence backoff"""
        if attempt <= 2:
            return self.config.base_delay
            
        # Generate fibonacci number for attempt
        a, b = 1, 1
        for _ in range(attempt - 2):
            a, b = b, a + b
            
        return self.config.base_delay * b
        
    def _get_quantum_error_multiplier(self, error: Exception) -> float:
        """Get multiplier based on quantum error types"""
        if isinstance(error, QuantumSchedulingError):
            return 1.2  # Scheduling errors need moderate delays
        elif isinstance(error, QuantumPrivacyError):
            return 1.5  # Privacy errors are more serious
        elif isinstance(error, QuantumOptimizationError):
            return 0.8  # Optimization errors might resolve quickly
        else:
            return 1.0


class QuantumCircuitBreaker:
    """Quantum-enhanced circuit breaker with coherence-based recovery"""
    
    def __init__(
        self, 
        name: str,
        config: QuantumCircuitBreakerConfig,
        metrics_collector: Optional[QuantumMetricsCollector] = None
    ):
        self.name = name
        self.config = config
        self.metrics_collector = metrics_collector
        
        self.state = QuantumCircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.quantum_coherence = 1.0
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == QuantumCircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = QuantumCircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise QuantumSchedulingError(
                        f"Circuit breaker {self.name} is OPEN",
                        {"state": self.state.value, "failure_count": self.failure_count}
                    )
                    
            elif self.state == QuantumCircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = QuantumCircuitBreakerState.OPEN
                    raise QuantumSchedulingError(
                        f"Circuit breaker {self.name} exceeded half-open call limit"
                    )
                    
            elif self.state == QuantumCircuitBreakerState.QUANTUM_DECOHERENT:
                # In decoherent state, apply probabilistic blocking
                decoherence_factor = 1.0 - self.quantum_coherence
                if random.random() < decoherence_factor * 0.5:  # Up to 50% blocking
                    raise QuantumSchedulingError(
                        f"Circuit breaker {self.name} blocked due to quantum decoherence"
                    )
                    
        # Execute the function
        start_time = time.time()
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            execution_time = time.time() - start_time
            await self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_failure(e, execution_time)
            raise
            
    async def _record_success(self, execution_time: float) -> None:
        """Record successful execution"""
        with self._lock:
            self.success_count += 1
            
            if self.state == QuantumCircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = QuantumCircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.quantum_coherence = min(1.0, self.quantum_coherence + 0.1)
                    self.logger.info(f"Circuit breaker {self.name} reset to CLOSED")
                    
            elif self.state in [QuantumCircuitBreakerState.QUANTUM_COHERENT, 
                              QuantumCircuitBreakerState.QUANTUM_DECOHERENT]:
                # Improve quantum coherence on success
                self.quantum_coherence = min(1.0, self.quantum_coherence + 0.05)
                
                if self.quantum_coherence > self.config.quantum_coherence_threshold:
                    self.state = QuantumCircuitBreakerState.CLOSED
                    
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_quantum_metric(
                QuantumMetricType.QUANTUM_EFFICIENCY,
                1.0 / max(execution_time, 0.001),  # Efficiency as inverse of time
                additional_data={"circuit_breaker": self.name, "success": True}
            )
            
    async def _record_failure(self, error: Exception, execution_time: float) -> None:
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Degrade quantum coherence on failure
            coherence_degradation = 0.1
            if isinstance(error, QuantumPrivacyError):
                coherence_degradation = 0.2  # Privacy errors cause more decoherence
            elif isinstance(error, QuantumOptimizationError):
                coherence_degradation = 0.05  # Optimization errors cause less decoherence
                
            self.quantum_coherence = max(0.0, self.quantum_coherence - coherence_degradation)
            
            # State transitions
            if self.failure_count >= self.config.failure_threshold:
                if self.quantum_coherence < self.config.quantum_coherence_threshold:
                    self.state = QuantumCircuitBreakerState.QUANTUM_DECOHERENT
                else:
                    self.state = QuantumCircuitBreakerState.OPEN
                    
                self.logger.warning(
                    f"Circuit breaker {self.name} opened due to failures: "
                    f"count={self.failure_count}, coherence={self.quantum_coherence:.3f}"
                )
                
            elif self.quantum_coherence < self.config.quantum_coherence_threshold * 0.5:
                self.state = QuantumCircuitBreakerState.QUANTUM_DECOHERENT
                
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_quantum_metric(
                QuantumMetricType.QUANTUM_ERROR_RATE,
                self.failure_count / max(self.success_count + self.failure_count, 1),
                additional_data={
                    "circuit_breaker": self.name, 
                    "error_type": type(error).__name__,
                    "success": False
                }
            )
            
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit breaker should attempt reset"""
        time_since_failure = time.time() - self.last_failure_time
        
        # Base timeout consideration
        if time_since_failure < self.config.recovery_timeout:
            return False
            
        # Quantum coherence affects recovery probability
        recovery_probability = self.quantum_coherence * 0.8 + 0.2  # 20-100% based on coherence
        
        return random.random() < recovery_probability
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "quantum_coherence": self.quantum_coherence,
            "half_open_calls": self.half_open_calls,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else 0
        }


class QuantumResilienceManager:
    """Manages quantum-enhanced resilience patterns"""
    
    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        metrics_collector: Optional[QuantumMetricsCollector] = None
    ):
        self.config = config or FederatedConfig()
        self.metrics_collector = metrics_collector
        
        self.circuit_breakers: Dict[str, QuantumCircuitBreaker] = {}
        self.retry_strategies: Dict[str, QuantumRetryStrategy] = {}
        
        self.logger = logging.getLogger(__name__)
        
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[QuantumCircuitBreakerConfig] = None
    ) -> QuantumCircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            cb_config = config or QuantumCircuitBreakerConfig()
            self.circuit_breakers[name] = QuantumCircuitBreaker(
                name, cb_config, self.metrics_collector
            )
            
        return self.circuit_breakers[name]
        
    def get_retry_strategy(
        self,
        name: str,
        config: Optional[QuantumRetryConfig] = None
    ) -> QuantumRetryStrategy:
        """Get or create retry strategy"""
        if name not in self.retry_strategies:
            retry_config = config or QuantumRetryConfig()
            self.retry_strategies[name] = QuantumRetryStrategy(retry_config)
            
        return self.retry_strategies[name]
        
    async def execute_with_resilience(
        self,
        func: Callable,
        circuit_breaker_name: Optional[str] = None,
        retry_strategy_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full resilience patterns"""
        
        # Get components
        circuit_breaker = None
        if circuit_breaker_name:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
            
        retry_strategy = None
        if retry_strategy_name:
            retry_strategy = self.get_retry_strategy(retry_strategy_name)
            
        max_attempts = max_retries or (retry_strategy.config.max_attempts if retry_strategy else 1)
        
        last_error = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                if circuit_breaker:
                    return await circuit_breaker.call(func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
            except Exception as e:
                last_error = e
                
                if attempt == max_attempts:
                    self.logger.error(f"All {max_attempts} attempts failed for resilient execution")
                    raise
                    
                if retry_strategy:
                    # Get quantum coherence from circuit breaker if available
                    quantum_coherence = 1.0
                    if circuit_breaker:
                        quantum_coherence = circuit_breaker.quantum_coherence
                        
                    delay = retry_strategy.calculate_delay(attempt, quantum_coherence, e)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    # No retry strategy, re-raise immediately
                    raise
                    
        # Should never reach here, but just in case
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Unexpected state in resilient execution")
            
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get status of all resilience components"""
        return {
            "circuit_breakers": {
                name: cb.get_state_info()
                for name, cb in self.circuit_breakers.items()
            },
            "retry_strategies": {
                name: {
                    "max_attempts": rs.config.max_attempts,
                    "base_delay": rs.config.base_delay,
                    "backoff_strategy": rs.config.backoff_strategy
                }
                for name, rs in self.retry_strategies.items()
            }
        }


# Decorators for quantum resilience
def quantum_circuit_breaker(
    name: str,
    config: Optional[QuantumCircuitBreakerConfig] = None,
    resilience_manager: Optional[QuantumResilienceManager] = None
):
    """Decorator to add quantum circuit breaker to function"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = resilience_manager or get_global_resilience_manager()
            circuit_breaker = manager.get_circuit_breaker(name, config)
            return await circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def quantum_retry(
    name: str,
    config: Optional[QuantumRetryConfig] = None,
    resilience_manager: Optional[QuantumResilienceManager] = None
):
    """Decorator to add quantum retry to function"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = resilience_manager or get_global_resilience_manager()
            return await manager.execute_with_resilience(
                func, 
                retry_strategy_name=name,
                *args, 
                **kwargs
            )
        return wrapper
    return decorator


def quantum_resilient(
    circuit_breaker_name: Optional[str] = None,
    retry_strategy_name: Optional[str] = None,
    max_retries: Optional[int] = None,
    resilience_manager: Optional[QuantumResilienceManager] = None
):
    """Decorator to add full quantum resilience to function"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = resilience_manager or get_global_resilience_manager()
            return await manager.execute_with_resilience(
                func,
                circuit_breaker_name=circuit_breaker_name,
                retry_strategy_name=retry_strategy_name,
                max_retries=max_retries,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


# Global resilience manager
_global_resilience_manager: Optional[QuantumResilienceManager] = None


def get_global_resilience_manager(
    config: Optional[FederatedConfig] = None,
    metrics_collector: Optional[QuantumMetricsCollector] = None
) -> QuantumResilienceManager:
    """Get global resilience manager instance"""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = QuantumResilienceManager(config, metrics_collector)
    return _global_resilience_manager


def initialize_quantum_resilience(
    config: Optional[FederatedConfig] = None,
    metrics_collector: Optional[QuantumMetricsCollector] = None
) -> QuantumResilienceManager:
    """Initialize quantum resilience system"""
    global _global_resilience_manager
    _global_resilience_manager = QuantumResilienceManager(config, metrics_collector)
    return _global_resilience_manager
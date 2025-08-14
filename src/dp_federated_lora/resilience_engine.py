"""
Resilience Engine: Advanced Fault Tolerance and Recovery System.

Comprehensive resilience implementation including:
- Multi-level circuit breakers and bulkheads
- Adaptive retry strategies with jitter
- Graceful degradation and fallback mechanisms
- Real-time health monitoring and auto-healing
- Chaos engineering and failure injection
- Distributed system recovery coordination
"""

import asyncio
import logging
import random
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from collections import deque, defaultdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Overall system health states"""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()
    FAILING = auto()
    RECOVERY = auto()

class ComponentState(Enum):
    """Individual component states"""
    OPERATIONAL = auto()
    SLOW = auto()
    UNSTABLE = auto()
    FAILING = auto()
    DOWN = auto()
    RECOVERING = auto()

class FailureMode(Enum):
    """Types of system failures"""
    TIMEOUT = auto()
    CONNECTION_ERROR = auto()
    RESOURCE_EXHAUSTION = auto()
    AUTHENTICATION_FAILURE = auto()
    DATA_CORRUPTION = auto()
    PERFORMANCE_DEGRADATION = auto()
    CASCADING_FAILURE = auto()
    BYZANTINE_FAILURE = auto()

@dataclass
class HealthMetrics:
    """Comprehensive health metrics for system components"""
    component_name: str
    timestamp: datetime
    response_time: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_depth: int
    throughput: float
    availability: float
    
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        # Weighted combination of metrics
        score = (
            self.success_rate * 0.3 +
            (1.0 - self.error_rate) * 0.2 +
            self.availability * 0.2 +
            min(1.0, self.throughput / 1000) * 0.1 +
            (1.0 - min(1.0, self.response_time / 1000)) * 0.1 +
            (1.0 - self.cpu_usage) * 0.05 +
            (1.0 - self.memory_usage) * 0.05
        )
        return max(0.0, min(1.0, score))

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    success_threshold: int = 3
    timeout: float = 10.0
    max_requests: int = 100
    sliding_window_size: int = 100

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing fast
    HALF_OPEN = auto() # Testing recovery

class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque(maxlen=config.sliding_window_size)
        self.adaptive_threshold = config.failure_threshold
        self._lock = threading.RLock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
                    
        try:
            start_time = time.time()
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
                
            execution_time = time.time() - start_time
            
            # Record success
            await self._record_success(execution_time)
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure(FailureMode.TIMEOUT)
            raise
        except Exception as e:
            await self._record_failure(self._classify_error(e))
            raise
            
    async def _record_success(self, execution_time: float):
        """Record successful execution"""
        with self._lock:
            self.request_history.append({
                "timestamp": datetime.now(),
                "success": True,
                "execution_time": execution_time
            })
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
                    
            elif self.state == CircuitBreakerState.CLOSED:
                # Adapt threshold based on performance
                self._adapt_threshold()
                
    async def _record_failure(self, failure_mode: FailureMode):
        """Record failed execution"""
        with self._lock:
            self.request_history.append({
                "timestamp": datetime.now(),
                "success": False,
                "failure_mode": failure_mode.name
            })
            
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.adaptive_threshold:
                if self.state == CircuitBreakerState.CLOSED:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker {self.name} OPENED due to failures")
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    self.success_count = 0
                    logger.warning(f"Circuit breaker {self.name} returned to OPEN")
                    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
            
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout
        
    def _adapt_threshold(self):
        """Adapt failure threshold based on recent performance"""
        if len(self.request_history) < 20:
            return
            
        recent_requests = list(self.request_history)[-20:]
        success_rate = sum(1 for r in recent_requests if r["success"]) / len(recent_requests)
        
        # Adjust threshold based on success rate
        if success_rate > 0.95:
            self.adaptive_threshold = min(self.config.failure_threshold + 2, 10)
        elif success_rate < 0.8:
            self.adaptive_threshold = max(self.config.failure_threshold - 1, 2)
            
    def _classify_error(self, error: Exception) -> FailureMode:
        """Classify error type"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureMode.TIMEOUT
        elif "connection" in error_str:
            return FailureMode.CONNECTION_ERROR
        elif "auth" in error_str:
            return FailureMode.AUTHENTICATION_FAILURE
        else:
            return FailureMode.PERFORMANCE_DEGRADATION
            
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            recent_success_rate = 0.0
            if self.request_history:
                recent_requests = list(self.request_history)[-10:]
                recent_success_rate = sum(1 for r in recent_requests if r["success"]) / len(recent_requests)
                
            return {
                "name": self.name,
                "state": self.state.name,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "adaptive_threshold": self.adaptive_threshold,
                "recent_success_rate": recent_success_rate,
                "total_requests": len(self.request_history)
            }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class AdaptiveRetryStrategy:
    """Adaptive retry strategy with exponential backoff and jitter"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.retry_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def execute_with_retry(self, 
                               operation_id: str,
                               func: Callable,
                               *args,
                               **kwargs) -> Any:
        """Execute function with adaptive retry"""
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                # Record successful retry
                self._record_retry_attempt(operation_id, attempt, True, None)
                return result
                
            except Exception as e:
                self._record_retry_attempt(operation_id, attempt, False, str(e))
                
                if attempt == self.max_retries:
                    logger.error(f"Operation {operation_id} failed after {self.max_retries} retries")
                    raise
                    
                # Calculate delay with adaptive backoff
                delay = self._calculate_delay(operation_id, attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {operation_id}, retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
                
    def _calculate_delay(self, operation_id: str, attempt: int) -> float:
        """Calculate adaptive delay for retry"""
        # Base exponential backoff
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
            
        # Adapt based on historical success
        success_rate = self._get_historical_success_rate(operation_id)
        if success_rate < 0.5:
            delay *= 1.5  # Increase delay for frequently failing operations
        elif success_rate > 0.9:
            delay *= 0.8  # Decrease delay for reliable operations
            
        return delay
        
    def _record_retry_attempt(self, 
                            operation_id: str,
                            attempt: int,
                            success: bool,
                            error: Optional[str]):
        """Record retry attempt for adaptive learning"""
        self.retry_history[operation_id].append({
            "timestamp": datetime.now(),
            "attempt": attempt,
            "success": success,
            "error": error
        })
        
        # Keep only recent history
        if len(self.retry_history[operation_id]) > 100:
            self.retry_history[operation_id] = self.retry_history[operation_id][-100:]
            
    def _get_historical_success_rate(self, operation_id: str) -> float:
        """Get historical success rate for operation"""
        history = self.retry_history.get(operation_id, [])
        if not history:
            return 0.5  # Default
            
        recent_history = history[-20:]  # Last 20 attempts
        success_count = sum(1 for h in recent_history if h["success"])
        return success_count / len(recent_history)

class BulkheadIsolation:
    """Bulkhead pattern for resource isolation"""
    
    def __init__(self):
        self.resource_pools: Dict[str, ThreadPoolExecutor] = {}
        self.pool_configs: Dict[str, Dict[str, Any]] = {}
        self.resource_usage: Dict[str, Dict[str, float]] = {}
        
    def create_resource_pool(self, 
                           pool_name: str,
                           max_workers: int = 10,
                           queue_size: int = 100):
        """Create isolated resource pool"""
        self.resource_pools[pool_name] = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"bulkhead-{pool_name}"
        )
        
        self.pool_configs[pool_name] = {
            "max_workers": max_workers,
            "queue_size": queue_size
        }
        
        self.resource_usage[pool_name] = {
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "queue_depth": 0
        }
        
        logger.info(f"Created resource pool '{pool_name}' with {max_workers} workers")
        
    async def execute_in_pool(self, 
                            pool_name: str,
                            func: Callable,
                            *args,
                            **kwargs) -> Any:
        """Execute function in isolated resource pool"""
        if pool_name not in self.resource_pools:
            raise ValueError(f"Resource pool '{pool_name}' not found")
            
        pool = self.resource_pools[pool_name]
        usage = self.resource_usage[pool_name]
        
        # Check if pool is overloaded
        if usage["queue_depth"] >= self.pool_configs[pool_name]["queue_size"]:
            raise ResourceExhaustionError(f"Resource pool '{pool_name}' is overloaded")
            
        try:
            usage["active_tasks"] += 1
            usage["queue_depth"] += 1
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, func, *args, **kwargs)
            
            usage["completed_tasks"] += 1
            return result
            
        except Exception as e:
            usage["failed_tasks"] += 1
            raise
        finally:
            usage["active_tasks"] -= 1
            usage["queue_depth"] -= 1
            
    def get_pool_status(self, pool_name: str) -> Dict[str, Any]:
        """Get resource pool status"""
        if pool_name not in self.resource_pools:
            return {}
            
        pool = self.resource_pools[pool_name]
        usage = self.resource_usage[pool_name]
        config = self.pool_configs[pool_name]
        
        return {
            "pool_name": pool_name,
            "max_workers": config["max_workers"],
            "active_tasks": usage["active_tasks"],
            "completed_tasks": usage["completed_tasks"],
            "failed_tasks": usage["failed_tasks"],
            "queue_depth": usage["queue_depth"],
            "utilization": usage["active_tasks"] / config["max_workers"],
            "success_rate": (usage["completed_tasks"] / 
                           max(1, usage["completed_tasks"] + usage["failed_tasks"]))
        }
        
    def shutdown_pools(self):
        """Gracefully shutdown all resource pools"""
        for pool_name, pool in self.resource_pools.items():
            logger.info(f"Shutting down resource pool '{pool_name}'")
            pool.shutdown(wait=True)

class ResourceExhaustionError(Exception):
    """Exception raised when resources are exhausted"""
    pass

class HealthMonitor:
    """Real-time health monitoring system"""
    
    def __init__(self):
        self.component_health: Dict[str, HealthMetrics] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_thresholds = {
            "response_time": 1000,  # ms
            "success_rate": 0.95,
            "error_rate": 0.05,
            "cpu_usage": 0.8,
            "memory_usage": 0.8,
            "availability": 0.99
        }
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.is_monitoring = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect health metrics from all components
                await self._collect_health_metrics()
                
                # Analyze health trends
                await self._analyze_health_trends()
                
                # Trigger alerts if needed
                await self._check_health_alerts()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _collect_health_metrics(self):
        """Collect health metrics from system components"""
        # This would integrate with actual system monitoring
        # For now, simulate metric collection
        
        components = ["federated_server", "client_manager", "aggregator", "privacy_engine"]
        
        for component in components:
            metrics = await self._simulate_health_metrics(component)
            await self.record_health_metrics(component, metrics)
            
    async def _simulate_health_metrics(self, component_name: str) -> HealthMetrics:
        """Simulate health metrics collection"""
        # In production, this would collect real metrics
        
        base_performance = 0.9 + random.random() * 0.1
        noise = random.normalvariate(0, 0.05)
        
        return HealthMetrics(
            component_name=component_name,
            timestamp=datetime.now(),
            response_time=max(50, random.normalvariate(200, 50)),
            success_rate=max(0.7, min(1.0, base_performance + noise)),
            error_rate=max(0.0, min(0.3, 0.05 + abs(noise))),
            cpu_usage=max(0.1, min(0.9, random.normalvariate(0.5, 0.1))),
            memory_usage=max(0.1, min(0.9, random.normalvariate(0.6, 0.1))),
            active_connections=random.randint(50, 200),
            queue_depth=random.randint(0, 50),
            throughput=max(100, random.normalvariate(500, 100)),
            availability=max(0.95, min(1.0, base_performance + 0.05))
        )
        
    async def record_health_metrics(self, component_name: str, metrics: HealthMetrics):
        """Record health metrics for component"""
        self.component_health[component_name] = metrics
        self.health_history[component_name].append(metrics)
        
    async def _analyze_health_trends(self):
        """Analyze health trends and predict issues"""
        for component_name, history in self.health_history.items():
            if len(history) < 10:
                continue
                
            recent_metrics = list(history)[-10:]
            
            # Analyze trends
            response_times = [m.response_time for m in recent_metrics]
            success_rates = [m.success_rate for m in recent_metrics]
            
            # Check for degrading trends
            if len(response_times) >= 5:
                recent_avg_rt = np.mean(response_times[-5:])
                older_avg_rt = np.mean(response_times[-10:-5])
                
                if recent_avg_rt > older_avg_rt * 1.5:
                    logger.warning(f"Degrading response time trend in {component_name}")
                    
            if len(success_rates) >= 5:
                recent_avg_sr = np.mean(success_rates[-5:])
                if recent_avg_sr < 0.9:
                    logger.warning(f"Low success rate trend in {component_name}")
                    
    async def _check_health_alerts(self):
        """Check for health threshold violations"""
        for component_name, metrics in self.component_health.items():
            violations = []
            
            if metrics.response_time > self.health_thresholds["response_time"]:
                violations.append(f"High response time: {metrics.response_time:.1f}ms")
                
            if metrics.success_rate < self.health_thresholds["success_rate"]:
                violations.append(f"Low success rate: {metrics.success_rate:.2%}")
                
            if metrics.error_rate > self.health_thresholds["error_rate"]:
                violations.append(f"High error rate: {metrics.error_rate:.2%}")
                
            if violations:
                await self._trigger_health_alert(component_name, violations)
                
    async def _trigger_health_alert(self, component_name: str, violations: List[str]):
        """Trigger health alert"""
        alert_data = {
            "component": component_name,
            "violations": violations,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"🏥 HEALTH ALERT: {component_name} - {', '.join(violations)}")
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.component_health:
            return {"status": "unknown", "components": {}}
            
        component_scores = {}
        total_score = 0
        
        for component_name, metrics in self.component_health.items():
            score = metrics.health_score()
            component_scores[component_name] = {
                "health_score": score,
                "status": self._categorize_health(score),
                "last_update": metrics.timestamp.isoformat()
            }
            total_score += score
            
        avg_score = total_score / len(self.component_health)
        
        return {
            "overall_health_score": avg_score,
            "system_status": self._categorize_health(avg_score),
            "components": component_scores,
            "monitoring_active": self.is_monitoring
        }
        
    def _categorize_health(self, score: float) -> str:
        """Categorize health score"""
        if score >= 0.9:
            return "HEALTHY"
        elif score >= 0.7:
            return "DEGRADED"
        elif score >= 0.5:
            return "CRITICAL"
        else:
            return "FAILING"
            
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("Health monitoring stopped")

class ResilienceEngine:
    """Main resilience orchestration engine"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.retry_strategy = AdaptiveRetryStrategy()
        self.bulkhead = BulkheadIsolation()
        self.health_monitor = HealthMonitor()
        self.fallback_strategies: Dict[str, Callable] = {}
        self.chaos_enabled = False
        
    async def initialize_resilience(self):
        """Initialize resilience mechanisms"""
        logger.info("🛡️ Initializing resilience engine")
        
        # Create circuit breakers for critical components
        critical_components = [
            "federated_server",
            "client_communication",
            "model_aggregation", 
            "privacy_engine",
            "data_storage"
        ]
        
        for component in critical_components:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30,
                success_threshold=3
            )
            self.circuit_breakers[component] = AdvancedCircuitBreaker(component, config)
            
        # Create resource pools
        self.bulkhead.create_resource_pool("training_pool", max_workers=10)
        self.bulkhead.create_resource_pool("communication_pool", max_workers=20)
        self.bulkhead.create_resource_pool("aggregation_pool", max_workers=5)
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        logger.info("Resilience engine initialized")
        
    async def execute_with_resilience(self,
                                    operation_name: str,
                                    func: Callable,
                                    pool_name: Optional[str] = None,
                                    fallback: Optional[Callable] = None,
                                    *args,
                                    **kwargs) -> Any:
        """Execute operation with full resilience protection"""
        
        # Get circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation_name)
        
        try:
            if pool_name:
                # Execute in isolated resource pool
                if circuit_breaker:
                    result = await circuit_breaker.call(
                        self.bulkhead.execute_in_pool,
                        pool_name, func, *args, **kwargs
                    )
                else:
                    result = await self.bulkhead.execute_in_pool(
                        pool_name, func, *args, **kwargs
                    )
            else:
                # Execute with circuit breaker only
                if circuit_breaker:
                    result = await circuit_breaker.call(func, *args, **kwargs)
                else:
                    # Execute with retry strategy
                    result = await self.retry_strategy.execute_with_retry(
                        operation_name, func, *args, **kwargs
                    )
                    
            return result
            
        except Exception as e:
            logger.error(f"Operation {operation_name} failed: {e}")
            
            # Try fallback strategy
            if fallback:
                logger.info(f"Attempting fallback for {operation_name}")
                try:
                    return await self._execute_fallback(fallback, *args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    
            raise
            
    async def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fallback strategy"""
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(*args, **kwargs)
        else:
            return fallback(*args, **kwargs)
            
    def register_fallback_strategy(self, operation_name: str, fallback: Callable):
        """Register fallback strategy for operation"""
        self.fallback_strategies[operation_name] = fallback
        logger.info(f"Registered fallback strategy for {operation_name}")
        
    async def inject_chaos(self, failure_rate: float = 0.1):
        """Inject chaos for resilience testing"""
        if not self.chaos_enabled:
            return
            
        if random.random() < failure_rate:
            failure_types = [
                "random_delay",
                "timeout_error", 
                "connection_error",
                "resource_exhaustion"
            ]
            
            failure_type = random.choice(failure_types)
            
            if failure_type == "random_delay":
                delay = random.uniform(1, 5)
                logger.warning(f"🌪️ Chaos: Injecting {delay:.1f}s delay")
                await asyncio.sleep(delay)
                
            elif failure_type == "timeout_error":
                logger.warning("🌪️ Chaos: Injecting timeout error")
                raise asyncio.TimeoutError("Chaos-injected timeout")
                
            elif failure_type == "connection_error":
                logger.warning("🌪️ Chaos: Injecting connection error")
                raise ConnectionError("Chaos-injected connection failure")
                
            elif failure_type == "resource_exhaustion":
                logger.warning("🌪️ Chaos: Injecting resource exhaustion")
                raise ResourceExhaustionError("Chaos-injected resource exhaustion")
                
    def enable_chaos_engineering(self):
        """Enable chaos engineering"""
        self.chaos_enabled = True
        logger.info("🌪️ Chaos engineering enabled")
        
    def disable_chaos_engineering(self):
        """Disable chaos engineering"""
        self.chaos_enabled = False
        logger.info("Chaos engineering disabled")
        
    async def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status"""
        circuit_breaker_status = {
            name: cb.get_status() 
            for name, cb in self.circuit_breakers.items()
        }
        
        bulkhead_status = {
            pool_name: self.bulkhead.get_pool_status(pool_name)
            for pool_name in self.bulkhead.resource_pools.keys()
        }
        
        health_status = self.health_monitor.get_system_health()
        
        return {
            "circuit_breakers": circuit_breaker_status,
            "resource_pools": bulkhead_status,
            "health_monitoring": health_status,
            "chaos_enabled": self.chaos_enabled,
            "registered_fallbacks": len(self.fallback_strategies)
        }
        
    async def shutdown(self):
        """Gracefully shutdown resilience engine"""
        logger.info("Shutting down resilience engine")
        
        await self.health_monitor.stop_monitoring()
        self.bulkhead.shutdown_pools()
        
        logger.info("Resilience engine shutdown complete")

# Factory function
def create_resilience_engine() -> ResilienceEngine:
    """Create configured resilience engine"""
    return ResilienceEngine()

# Example usage
async def main():
    """Example resilience engine usage"""
    engine = create_resilience_engine()
    await engine.initialize_resilience()
    
    # Example resilient operation
    async def unreliable_operation():
        """Simulate unreliable operation"""
        if random.random() < 0.3:  # 30% failure rate
            raise ConnectionError("Simulated connection failure")
        return "Operation successful"
        
    # Fallback strategy
    async def fallback_operation():
        """Fallback when main operation fails"""
        return "Fallback result"
        
    engine.register_fallback_strategy("test_operation", fallback_operation)
    
    # Execute with resilience
    try:
        result = await engine.execute_with_resilience(
            "test_operation",
            unreliable_operation,
            pool_name="training_pool",
            fallback=fallback_operation
        )
        logger.info(f"Operation result: {result}")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        
    # Get status
    status = await engine.get_resilience_status()
    logger.info(f"Resilience status: {json.dumps(status, indent=2, default=str)}")
    
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
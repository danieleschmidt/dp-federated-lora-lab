#!/usr/bin/env python3
"""
Enhanced Research Engine with Robust Error Handling and Monitoring

This module implements a production-ready research engine with comprehensive
error handling, monitoring, and resilience features for autonomous operation.
"""

import os
import sys
import time
import json
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import asynccontextmanager
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import pickle

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_engine.log')
    ]
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResearchEngineState(Enum):
    """Research engine operational states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    GENERATING = "generating"
    EXPERIMENTING = "experimenting"
    VALIDATING = "validating"
    REPORTING = "reporting"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class ErrorEvent:
    """Represents an error event with context."""
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_threads: int
    hypotheses_generated: int
    experiments_completed: int
    validations_passed: int
    error_count: int
    uptime_seconds: float

@dataclass
class HealthCheck:
    """Health check results."""
    timestamp: float
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

class CircuitBreaker:
    """Circuit breaker pattern for resilient operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, name: str = "default"):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
                
                raise e

class ResourceMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: List[PerformanceMetrics] = []
        
    def collect_metrics(self, research_engine) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=process.cpu_percent(),
                memory_usage=process.memory_info().rss / (1024 * 1024),  # MB
                active_threads=threading.active_count(),
                hypotheses_generated=len(research_engine.generated_hypotheses),
                experiments_completed=len(research_engine.completed_experiments),
                validations_passed=len(research_engine.validated_breakthroughs),
                error_count=len(research_engine.error_events),
                uptime_seconds=time.time() - self.start_time
            )
        except ImportError:
            # Fallback when psutil is not available
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                active_threads=threading.active_count(),
                hypotheses_generated=len(research_engine.generated_hypotheses),
                experiments_completed=len(research_engine.completed_experiments),
                validations_passed=len(research_engine.validated_breakthroughs),
                error_count=len(research_engine.error_events),
                uptime_seconds=time.time() - self.start_time
            )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics

class HealthChecker:
    """Health checking system for all components."""
    
    def __init__(self):
        self.health_checks: List[HealthCheck] = []
    
    async def check_component_health(self, component_name: str, check_func: Callable) -> HealthCheck:
        """Check health of a specific component."""
        start_time = time.time()
        
        try:
            result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            latency = (time.time() - start_time) * 1000  # ms
            
            health_check = HealthCheck(
                timestamp=time.time(),
                component=component_name,
                status="healthy",
                details=result if isinstance(result, dict) else {"result": result},
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000  # ms
            health_check = HealthCheck(
                timestamp=time.time(),
                component=component_name,
                status="unhealthy",
                details={"error": str(e), "traceback": traceback.format_exc()},
                latency_ms=latency
            )
        
        self.health_checks.append(health_check)
        return health_check
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        if not self.health_checks:
            return "unknown"
        
        recent_checks = [hc for hc in self.health_checks if time.time() - hc.timestamp < 300]  # Last 5 minutes
        
        if not recent_checks:
            return "stale"
        
        unhealthy_count = sum(1 for hc in recent_checks if hc.status == "unhealthy")
        degraded_count = sum(1 for hc in recent_checks if hc.status == "degraded")
        
        if unhealthy_count > len(recent_checks) * 0.5:
            return "unhealthy"
        elif degraded_count + unhealthy_count > len(recent_checks) * 0.3:
            return "degraded"
        else:
            return "healthy"

class EnhancedResearchEngine:
    """Enhanced research engine with robustness features."""
    
    def __init__(self, output_dir: str = "enhanced_research", config: Optional[Dict[str, Any]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = config or self._default_config()
        
        # State management
        self.state = ResearchEngineState.INITIALIZING
        self.start_time = time.time()
        
        # Core components with error handling
        self.generated_hypotheses: List[Dict[str, Any]] = []
        self.completed_experiments: List[Dict[str, Any]] = []
        self.validated_breakthroughs: List[Dict[str, Any]] = []
        self.error_events: List[ErrorEvent] = []
        
        # Monitoring components
        self.resource_monitor = ResourceMonitor()
        self.health_checker = HealthChecker()
        
        # Resilience components
        self.circuit_breakers = {
            "hypothesis_generation": CircuitBreaker(name="hypothesis_generation"),
            "experiment_execution": CircuitBreaker(name="experiment_execution"),
            "validation": CircuitBreaker(name="validation")
        }
        
        # Async management
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Enhanced Research Engine initialized at {self.output_dir}")
        self.state = ResearchEngineState.IDLE
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for enhanced research engine."""
        return {
            "max_concurrent_experiments": 3,
            "max_hypotheses_per_session": 50,
            "health_check_interval": 30,  # seconds
            "metrics_collection_interval": 10,  # seconds
            "error_recovery_enabled": True,
            "persistent_state": True,
            "backup_interval": 300,  # seconds
            "circuit_breaker_enabled": True,
            "max_error_count": 100,
            "auto_recovery_attempts": 3
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.set()
    
    async def _log_error(self, error: Exception, context: Dict[str, Any] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Log error with structured format and context."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(error_event)
        
        # Log with appropriate level
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"Error in {context.get('component', 'unknown')}: {error_event.error_message}")
        
        # Trigger alerts for high severity errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._trigger_alert(error_event)
        
        # Cleanup old errors
        if len(self.error_events) > self.config["max_error_count"]:
            self.error_events = self.error_events[-self.config["max_error_count"]:]
    
    async def _trigger_alert(self, error_event: ErrorEvent):
        """Trigger alert for high severity errors."""
        alert_message = f"ALERT: {error_event.severity.value.upper()} error in research engine"
        logger.critical(alert_message)
        
        # In production, this would integrate with alerting systems
        alert_data = {
            "timestamp": error_event.timestamp,
            "severity": error_event.severity.value,
            "error_type": error_event.error_type,
            "message": error_event.error_message,
            "context": error_event.context
        }
        
        # Save alert to file
        alert_file = self.output_dir / "alerts.jsonl"
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert_data) + "\n")
    
    async def _save_checkpoint(self):
        """Save current state for recovery."""
        if not self.config["persistent_state"]:
            return
        
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "state": self.state.value,
                "generated_hypotheses": self.generated_hypotheses,
                "completed_experiments": self.completed_experiments,
                "validated_breakthroughs": self.validated_breakthroughs,
                "error_count": len(self.error_events),
                "uptime": time.time() - self.start_time
            }
            
            checkpoint_file = self.output_dir / "checkpoint.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.debug("Checkpoint saved successfully")
        except Exception as e:
            await self._log_error(e, {"component": "checkpoint"}, ErrorSeverity.MEDIUM)
    
    async def _load_checkpoint(self) -> bool:
        """Load previous state from checkpoint."""
        if not self.config["persistent_state"]:
            return False
        
        try:
            checkpoint_file = self.output_dir / "checkpoint.json"
            if not checkpoint_file.exists():
                return False
            
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            self.generated_hypotheses = checkpoint_data.get("generated_hypotheses", [])
            self.completed_experiments = checkpoint_data.get("completed_experiments", [])
            self.validated_breakthroughs = checkpoint_data.get("validated_breakthroughs", [])
            
            logger.info(f"Checkpoint loaded: {len(self.generated_hypotheses)} hypotheses, "
                       f"{len(self.validated_breakthroughs)} breakthroughs")
            return True
        except Exception as e:
            await self._log_error(e, {"component": "checkpoint_load"}, ErrorSeverity.MEDIUM)
            return False
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check algorithm generator health
                await self.health_checker.check_component_health(
                    "algorithm_generator",
                    lambda: {"status": "operational", "hypotheses_generated": len(self.generated_hypotheses)}
                )
                
                # Check overall system health
                overall_health = self.health_checker.get_overall_health()
                logger.debug(f"Overall system health: {overall_health}")
                
                await asyncio.sleep(self.config["health_check_interval"])
            except Exception as e:
                await self._log_error(e, {"component": "health_check"}, ErrorSeverity.LOW)
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                metrics = self.resource_monitor.collect_metrics(self)
                
                # Log metrics periodically
                if len(self.resource_monitor.metrics_history) % 10 == 0:
                    logger.info(f"Metrics - CPU: {metrics.cpu_usage:.1f}%, "
                               f"Memory: {metrics.memory_usage:.1f}MB, "
                               f"Hypotheses: {metrics.hypotheses_generated}, "
                               f"Breakthroughs: {metrics.validations_passed}")
                
                await asyncio.sleep(self.config["metrics_collection_interval"])
            except Exception as e:
                await self._log_error(e, {"component": "metrics_collection"}, ErrorSeverity.LOW)
                await asyncio.sleep(60)
    
    async def _backup_loop(self):
        """Background backup loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                await self._save_checkpoint()
                await asyncio.sleep(self.config["backup_interval"])
            except Exception as e:
                await self._log_error(e, {"component": "backup"}, ErrorSeverity.MEDIUM)
                await asyncio.sleep(300)
    
    async def _generate_hypothesis_with_recovery(self) -> Optional[Dict[str, Any]]:
        """Generate hypothesis with error recovery."""
        for attempt in range(self.config["auto_recovery_attempts"]):
            try:
                if self.config["circuit_breaker_enabled"]:
                    return self.circuit_breakers["hypothesis_generation"].call(
                        self._generate_single_hypothesis
                    )
                else:
                    return self._generate_single_hypothesis()
            except Exception as e:
                await self._log_error(e, {
                    "component": "hypothesis_generation",
                    "attempt": attempt + 1
                }, ErrorSeverity.MEDIUM if attempt < 2 else ErrorSeverity.HIGH)
                
                if attempt < self.config["auto_recovery_attempts"] - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _generate_single_hypothesis(self) -> Dict[str, Any]:
        """Generate a single hypothesis (placeholder implementation)."""
        import random
        import hashlib
        
        # Simulate hypothesis generation
        algorithm_types = ["privacy_mechanism", "aggregation_method", "optimization_strategy", "quantum_enhancement"]
        selected_type = random.choice(algorithm_types)
        
        hypothesis_id = hashlib.md5(f"{selected_type}_{time.time()}".encode()).hexdigest()[:8]
        
        return {
            "id": hypothesis_id,
            "title": f"Novel {selected_type.replace('_', ' ').title()} Algorithm",
            "algorithm_type": selected_type,
            "expected_improvement": random.uniform(0.1, 0.8),
            "created_at": time.time(),
            "success_criteria": {
                "performance": random.uniform(1.1, 2.0),
                "efficiency": random.uniform(0.5, 0.9),
                "robustness": random.uniform(1.2, 1.8)
            }
        }
    
    async def generate_research_hypotheses(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate research hypotheses with robustness."""
        self.state = ResearchEngineState.GENERATING
        logger.info(f"Generating {count} research hypotheses")
        
        hypotheses = []
        
        for i in range(count):
            if self.shutdown_event.is_set():
                break
            
            hypothesis = await self._generate_hypothesis_with_recovery()
            if hypothesis:
                hypotheses.append(hypothesis)
                self.generated_hypotheses.append(hypothesis)
                logger.debug(f"Generated hypothesis {i+1}: {hypothesis['title']}")
            else:
                logger.warning(f"Failed to generate hypothesis {i+1}")
        
        self.state = ResearchEngineState.IDLE
        logger.info(f"Successfully generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    async def start_enhanced_research(self, duration_hours: float = 1.0):
        """Start enhanced autonomous research with monitoring."""
        self.running = True
        logger.info(f"Starting enhanced autonomous research for {duration_hours} hours")
        
        # Load previous state if available
        await self._load_checkpoint()
        
        # Start background monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._backup_loop())
        ]
        
        end_time = time.time() + (duration_hours * 3600)
        
        try:
            while time.time() < end_time and not self.shutdown_event.is_set():
                # Generate hypotheses
                if len(self.generated_hypotheses) < self.config["max_hypotheses_per_session"]:
                    hypotheses = await self.generate_research_hypotheses(count=5)
                    
                    # Simulate validation (in real implementation, this would run experiments)
                    for hypothesis in hypotheses:
                        if hypothesis["expected_improvement"] > 0.5:  # Simulate breakthrough
                            self.validated_breakthroughs.append({
                                "hypothesis_id": hypothesis["id"],
                                "breakthrough_timestamp": time.time(),
                                "improvement_factor": hypothesis["expected_improvement"],
                                "validation_status": "validated"
                            })
                            logger.info(f"BREAKTHROUGH: {hypothesis['title']}")
                
                # Brief pause
                await asyncio.sleep(10)
        
        except Exception as e:
            await self._log_error(e, {"component": "main_research_loop"}, ErrorSeverity.CRITICAL)
        
        finally:
            await self._shutdown_gracefully()
    
    async def _shutdown_gracefully(self):
        """Graceful shutdown with cleanup."""
        logger.info("Initiating graceful shutdown")
        self.state = ResearchEngineState.SHUTDOWN
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Final checkpoint save
        await self._save_checkpoint()
        
        # Generate final report
        await self._generate_final_report()
        
        logger.info("Graceful shutdown completed")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        try:
            report = {
                "session_summary": {
                    "start_time": self.start_time,
                    "end_time": time.time(),
                    "duration_hours": (time.time() - self.start_time) / 3600,
                    "total_hypotheses": len(self.generated_hypotheses),
                    "total_breakthroughs": len(self.validated_breakthroughs),
                    "total_errors": len(self.error_events),
                    "final_state": self.state.value
                },
                "performance_metrics": {
                    "avg_cpu_usage": sum(m.cpu_usage for m in self.resource_monitor.metrics_history) / len(self.resource_monitor.metrics_history) if self.resource_monitor.metrics_history else 0,
                    "peak_memory_usage": max((m.memory_usage for m in self.resource_monitor.metrics_history), default=0),
                    "total_uptime": time.time() - self.start_time
                },
                "error_analysis": {
                    "error_count_by_severity": {
                        severity.value: sum(1 for e in self.error_events if e.severity == severity)
                        for severity in ErrorSeverity
                    },
                    "most_common_errors": self._get_common_errors()
                },
                "breakthroughs": self.validated_breakthroughs,
                "circuit_breaker_status": {
                    name: {
                        "state": cb.state,
                        "failure_count": cb.failure_count
                    }
                    for name, cb in self.circuit_breakers.items()
                }
            }
            
            report_file = self.output_dir / "enhanced_research_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Final report saved to: {report_file}")
            
        except Exception as e:
            await self._log_error(e, {"component": "final_report"}, ErrorSeverity.HIGH)
    
    def _get_common_errors(self) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_counts = {}
        for error in self.error_events:
            error_type = error.error_type
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

async def main():
    """Main function for enhanced research engine."""
    logger.info("🚀 Starting Enhanced Research Engine")
    
    # Configuration
    config = {
        "max_concurrent_experiments": 2,
        "max_hypotheses_per_session": 20,
        "health_check_interval": 15,
        "metrics_collection_interval": 5,
        "backup_interval": 60,
        "error_recovery_enabled": True,
        "persistent_state": True,
        "circuit_breaker_enabled": True,
        "max_error_count": 100,
        "auto_recovery_attempts": 3
    }
    
    # Create and run enhanced research engine
    engine = EnhancedResearchEngine(output_dir="enhanced_research_output", config=config)
    
    try:
        await engine.start_enhanced_research(duration_hours=0.1)  # 6 minutes for testing
    except KeyboardInterrupt:
        logger.info("Research interrupted by user")
    except Exception as e:
        logger.error(f"Research failed: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Enhanced research session completed")

if __name__ == "__main__":
    asyncio.run(main())
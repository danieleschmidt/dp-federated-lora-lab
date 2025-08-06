"""
Quantum-Enhanced Monitoring and Observability

Comprehensive monitoring, metrics collection, and observability for quantum-inspired
federated learning components with real-time performance tracking and anomaly detection.
"""

import asyncio
import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import threading
import json
import hashlib

import numpy as np
import torch
from pydantic import BaseModel, Field

from .monitoring import MetricsCollector
from .exceptions import MonitoringError, QuantumSchedulingError
from .config import FederatedConfig


class QuantumMetricType(Enum):
    """Types of quantum metrics"""
    COHERENCE_TIME = "coherence_time"
    ENTANGLEMENT_STRENGTH = "entanglement_strength"
    QUANTUM_FIDELITY = "quantum_fidelity"
    DECOHERENCE_RATE = "decoherence_rate"
    QUANTUM_EFFICIENCY = "quantum_efficiency"
    OPTIMIZATION_CONVERGENCE = "optimization_convergence"
    PRIVACY_AMPLIFICATION = "privacy_amplification"
    SUPERPOSITION_QUALITY = "superposition_quality"
    QUANTUM_ERROR_RATE = "quantum_error_rate"
    CIRCUIT_DEPTH = "circuit_depth"


@dataclass
class QuantumMetric:
    """Quantum metric measurement"""
    metric_type: QuantumMetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None
    round_number: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "client_id": self.client_id,
            "round_number": self.round_number,
            "additional_data": self.additional_data
        }


class QuantumAnomalyDetector:
    """Anomaly detection for quantum metrics using statistical methods"""
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 2.0,  # Number of standard deviations
        min_samples: int = 20
    ):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.metric_histories: Dict[QuantumMetricType, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.anomaly_counts: Dict[QuantumMetricType, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
        
    def add_measurement(
        self, 
        metric_type: QuantumMetricType, 
        value: float
    ) -> Tuple[bool, float]:
        """
        Add measurement and detect anomalies
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        history = self.metric_histories[metric_type]
        history.append(value)
        
        if len(history) < self.min_samples:
            return False, 0.0
            
        # Calculate statistical measures
        values = list(history)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if stdev == 0:
            return False, 0.0
            
        # Calculate z-score
        z_score = abs(value - mean) / stdev
        is_anomaly = z_score > self.sensitivity
        
        if is_anomaly:
            self.anomaly_counts[metric_type] += 1
            self.logger.warning(
                f"Quantum anomaly detected in {metric_type.value}: "
                f"value={value:.4f}, z_score={z_score:.2f}"
            )
            
        return is_anomaly, z_score
        
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        total_measurements = sum(len(history) for history in self.metric_histories.values())
        total_anomalies = sum(self.anomaly_counts.values())
        
        return {
            "total_measurements": total_measurements,
            "total_anomalies": total_anomalies,
            "anomaly_rate": total_anomalies / total_measurements if total_measurements > 0 else 0.0,
            "anomalies_by_metric": dict(self.anomaly_counts),
            "active_metrics": list(self.metric_histories.keys())
        }


class QuantumPerformanceTracker:
    """Tracks performance of quantum algorithms and components"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.execution_times: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.success_rates: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.convergence_rates: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._lock = threading.Lock()
        
    def record_execution(
        self,
        operation_name: str,
        execution_time: float,
        success: bool = True,
        convergence_iterations: Optional[int] = None
    ) -> None:
        """Record execution metrics"""
        with self._lock:
            self.execution_times[operation_name].append(execution_time)
            self.success_rates[operation_name].append(1.0 if success else 0.0)
            
            if convergence_iterations is not None:
                self.convergence_rates[operation_name].append(convergence_iterations)
                
    def get_performance_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get performance summary for an operation"""
        with self._lock:
            exec_times = list(self.execution_times[operation_name])
            success_vals = list(self.success_rates[operation_name])
            convergence_vals = list(self.convergence_rates[operation_name])
            
            if not exec_times:
                return {"error": "No data available"}
                
            summary = {
                "operation": operation_name,
                "sample_count": len(exec_times),
                "avg_execution_time": statistics.mean(exec_times),
                "median_execution_time": statistics.median(exec_times),
                "min_execution_time": min(exec_times),
                "max_execution_time": max(exec_times),
                "success_rate": statistics.mean(success_vals) if success_vals else 0.0,
            }
            
            if len(exec_times) > 1:
                summary["execution_time_std"] = statistics.stdev(exec_times)
                
            if convergence_vals:
                summary["avg_convergence_iterations"] = statistics.mean(convergence_vals)
                summary["median_convergence_iterations"] = statistics.median(convergence_vals)
                
            return summary
            
    def get_all_performance_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summaries for all tracked operations"""
        return {
            operation: self.get_performance_summary(operation)
            for operation in self.execution_times.keys()
        }


class QuantumDashboardData:
    """Data structure for quantum dashboard metrics"""
    
    def __init__(self):
        self.quantum_metrics: List[QuantumMetric] = []
        self.system_health: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Any] = {}
        self.anomaly_alerts: List[Dict[str, Any]] = []
        self.client_quantum_states: Dict[str, Dict[str, Any]] = {}
        self.last_update: datetime = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "quantum_metrics": [m.to_dict() for m in self.quantum_metrics[-100:]],  # Last 100
            "system_health": self.system_health,
            "performance_stats": self.performance_stats,
            "anomaly_alerts": self.anomaly_alerts[-50:],  # Last 50 alerts
            "client_quantum_states": self.client_quantum_states,
            "last_update": self.last_update.isoformat()
        }


class QuantumMetricsCollector(MetricsCollector):
    """Enhanced metrics collector with quantum-specific capabilities"""
    
    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        enable_anomaly_detection: bool = True
    ):
        super().__init__()
        self.config = config
        self.quantum_metrics: List[QuantumMetric] = []
        self.anomaly_detector = QuantumAnomalyDetector() if enable_anomaly_detection else None
        self.performance_tracker = QuantumPerformanceTracker()
        self.dashboard_data = QuantumDashboardData()
        self.logger = logging.getLogger(__name__)
        
        # Background task for metric processing
        self._stop_processing = threading.Event()
        self._processing_thread = threading.Thread(target=self._process_metrics_loop)
        self._processing_thread.start()
        
    def record_quantum_metric(
        self,
        metric_type: QuantumMetricType,
        value: float,
        client_id: Optional[str] = None,
        round_number: Optional[int] = None,
        **additional_data
    ) -> None:
        """Record a quantum-specific metric"""
        metric = QuantumMetric(
            metric_type=metric_type,
            value=value,
            client_id=client_id,
            round_number=round_number,
            additional_data=additional_data
        )
        
        self.quantum_metrics.append(metric)
        
        # Anomaly detection
        if self.anomaly_detector:
            is_anomaly, anomaly_score = self.anomaly_detector.add_measurement(
                metric_type, value
            )
            
            if is_anomaly:
                self.dashboard_data.anomaly_alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "metric_type": metric_type.value,
                    "value": value,
                    "anomaly_score": anomaly_score,
                    "client_id": client_id
                })
                
        # Also record in base collector
        metric_name = f"quantum_{metric_type.value}"
        if client_id:
            metric_name += f"_client_{client_id}"
        self.record_metric(metric_name, value)
        
    def record_quantum_performance(
        self,
        operation_name: str,
        execution_time: float,
        success: bool = True,
        convergence_iterations: Optional[int] = None
    ) -> None:
        """Record quantum algorithm performance"""
        self.performance_tracker.record_execution(
            operation_name, execution_time, success, convergence_iterations
        )
        
        # Record in base collector as well
        self.record_metric(f"quantum_perf_{operation_name}_time", execution_time)
        self.record_metric(f"quantum_perf_{operation_name}_success", 1.0 if success else 0.0)
        
        if convergence_iterations is not None:
            self.record_metric(f"quantum_perf_{operation_name}_convergence", convergence_iterations)
            
    def get_quantum_state_summary(self, time_window: timedelta = timedelta(minutes=10)) -> Dict[str, Any]:
        """Get quantum state summary for recent time window"""
        cutoff_time = datetime.now() - time_window
        recent_metrics = [
            m for m in self.quantum_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent quantum metrics available"}
            
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
            
        summary = {}
        for metric_type, values in metrics_by_type.items():
            summary[metric_type.value] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0
            }
            
        return summary
        
    def get_client_quantum_state(self, client_id: str) -> Dict[str, Any]:
        """Get quantum state for specific client"""
        client_metrics = [
            m for m in self.quantum_metrics
            if m.client_id == client_id
        ]
        
        if not client_metrics:
            return {"error": f"No quantum metrics found for client {client_id}"}
            
        # Get most recent metrics for each type
        latest_metrics = {}
        for metric in sorted(client_metrics, key=lambda x: x.timestamp, reverse=True):
            if metric.metric_type not in latest_metrics:
                latest_metrics[metric.metric_type] = metric
                
        client_state = {}
        for metric_type, metric in latest_metrics.items():
            client_state[metric_type.value] = {
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "round_number": metric.round_number
            }
            
        return client_state
        
    def get_dashboard_data(self) -> QuantumDashboardData:
        """Get formatted data for quantum dashboard"""
        # Update dashboard data
        self.dashboard_data.quantum_metrics = self.quantum_metrics[-200:]  # Last 200 metrics
        self.dashboard_data.performance_stats = self.performance_tracker.get_all_performance_summaries()
        self.dashboard_data.last_update = datetime.now()
        
        if self.anomaly_detector:
            self.dashboard_data.system_health["anomaly_stats"] = self.anomaly_detector.get_anomaly_stats()
            
        # Update client quantum states
        client_ids = set(m.client_id for m in self.quantum_metrics if m.client_id)
        for client_id in client_ids:
            self.dashboard_data.client_quantum_states[client_id] = self.get_client_quantum_state(client_id)
            
        return self.dashboard_data
        
    def export_quantum_metrics(self, format: str = "json") -> str:
        """Export quantum metrics in specified format"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_count": len(self.quantum_metrics),
            "metrics": [m.to_dict() for m in self.quantum_metrics],
            "performance_stats": self.performance_tracker.get_all_performance_summaries()
        }
        
        if self.anomaly_detector:
            data["anomaly_stats"] = self.anomaly_detector.get_anomaly_stats()
            
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _process_metrics_loop(self) -> None:
        """Background loop for processing metrics"""
        while not self._stop_processing.wait(1.0):  # Process every second
            try:
                # Clean old metrics (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.quantum_metrics = [
                    m for m in self.quantum_metrics
                    if m.timestamp >= cutoff_time
                ]
                
                # Clean old anomaly alerts
                self.dashboard_data.anomaly_alerts = [
                    alert for alert in self.dashboard_data.anomaly_alerts
                    if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
                ]
                
            except Exception as e:
                self.logger.error(f"Error in metrics processing loop: {e}")
                
    def cleanup(self) -> None:
        """Cleanup resources"""
        self._stop_processing.set()
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
        super().cleanup()


class QuantumHealthCheck:
    """Health check system for quantum components"""
    
    def __init__(self, metrics_collector: QuantumMetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
    async def check_quantum_scheduler_health(self) -> Dict[str, Any]:
        """Check health of quantum scheduler"""
        try:
            # Check if quantum scheduler is responsive
            from .quantum_scheduler import get_quantum_scheduler
            scheduler = get_quantum_scheduler()
            
            start_time = time.time()
            metrics = await scheduler.get_quantum_state_metrics()
            response_time = time.time() - start_time
            
            health_status = {
                "status": "healthy",
                "response_time": response_time,
                "metrics": metrics,
                "checks_passed": []
            }
            
            # Check response time
            if response_time < 1.0:
                health_status["checks_passed"].append("response_time")
            else:
                health_status["status"] = "degraded"
                health_status["issues"] = health_status.get("issues", [])
                health_status["issues"].append("slow_response_time")
                
            # Check if we have active tasks and clients
            if metrics.get("total_tasks", 0) > 0:
                health_status["checks_passed"].append("has_tasks")
            if metrics.get("total_clients", 0) > 0:
                health_status["checks_passed"].append("has_clients")
                
            # Check coherence levels
            avg_coherence = metrics.get("average_coherence", 0)
            if avg_coherence > 0.5:
                health_status["checks_passed"].append("quantum_coherence")
            elif avg_coherence > 0:
                health_status["status"] = "degraded"
                health_status["issues"] = health_status.get("issues", [])
                health_status["issues"].append("low_coherence")
                
            return health_status
            
        except Exception as e:
            self.logger.error(f"Quantum scheduler health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "checks_passed": []
            }
            
    async def check_quantum_privacy_health(self) -> Dict[str, Any]:
        """Check health of quantum privacy engine"""
        try:
            # Basic health check for quantum privacy
            from .quantum_privacy import create_quantum_privacy_engine
            
            start_time = time.time()
            privacy_engine = create_quantum_privacy_engine()
            health_check_time = time.time() - start_time
            
            health_status = {
                "status": "healthy",
                "initialization_time": health_check_time,
                "checks_passed": ["initialization"]
            }
            
            # Check privacy accountant
            try:
                epsilon, delta = privacy_engine.get_privacy_spent()
                health_status["privacy_spent"] = {"epsilon": epsilon, "delta": delta}
                health_status["checks_passed"].append("privacy_accounting")
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["issues"] = health_status.get("issues", [])
                health_status["issues"].append(f"privacy_accounting_error: {e}")
                
            return health_status
            
        except Exception as e:
            self.logger.error(f"Quantum privacy health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "checks_passed": []
            }
            
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check on all quantum components"""
        start_time = time.time()
        
        # Run individual health checks
        scheduler_health = await self.check_quantum_scheduler_health()
        privacy_health = await self.check_quantum_privacy_health()
        
        # Aggregate results
        all_healthy = all(
            health["status"] == "healthy" 
            for health in [scheduler_health, privacy_health]
        )
        
        any_degraded = any(
            health["status"] == "degraded"
            for health in [scheduler_health, privacy_health]
        )
        
        overall_status = "healthy"
        if any_degraded and all_healthy:
            overall_status = "degraded"
        elif not all_healthy:
            overall_status = "unhealthy"
            
        total_time = time.time() - start_time
        
        return {
            "overall_status": overall_status,
            "total_check_time": total_time,
            "component_health": {
                "quantum_scheduler": scheduler_health,
                "quantum_privacy": privacy_health
            },
            "metrics_summary": self.metrics_collector.get_quantum_state_summary(),
            "timestamp": datetime.now().isoformat()
        }


# Global quantum metrics collector instance
_quantum_metrics_collector: Optional[QuantumMetricsCollector] = None


def get_quantum_metrics_collector(
    config: Optional[FederatedConfig] = None,
    enable_anomaly_detection: bool = True
) -> QuantumMetricsCollector:
    """Get global quantum metrics collector instance"""
    global _quantum_metrics_collector
    if _quantum_metrics_collector is None:
        _quantum_metrics_collector = QuantumMetricsCollector(config, enable_anomaly_detection)
    return _quantum_metrics_collector


def create_quantum_health_checker(
    metrics_collector: Optional[QuantumMetricsCollector] = None
) -> QuantumHealthCheck:
    """Create quantum health checker"""
    if metrics_collector is None:
        metrics_collector = get_quantum_metrics_collector()
    return QuantumHealthCheck(metrics_collector)
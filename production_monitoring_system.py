#!/usr/bin/env python3
"""
Production monitoring and observability system for DP-Federated LoRA.

This system provides comprehensive monitoring, alerting, distributed tracing,
and observability for production federated learning deployments.
"""

import logging
import time
import threading
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = time.time()


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "ok"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = "ok"):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add log to span."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        })


class MetricsCollector:
    """Metrics collection and aggregation system."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.retention_seconds = retention_hours * 3600
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Initialized metrics collector with {retention_hours}h retention")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._generate_metric_key(name, labels)
            self.counters[key] += value
            
            metric = Metric(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            key = self._generate_metric_key(name, labels)
            self.gauges[key] = value
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a value in a histogram metric."""
        with self._lock:
            key = self._generate_metric_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values for histograms
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def time_operation(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    def _generate_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Generate unique key for metric."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metric_summary(self, name: str, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time period."""
        with self._lock:
            cutoff_time = time.time() - duration_seconds
            
            # Find all metrics matching the name
            matching_metrics = []
            for key, metric_list in self.metrics.items():
                if key.startswith(name):
                    recent_metrics = [m for m in metric_list if m.timestamp > cutoff_time]
                    matching_metrics.extend(recent_metrics)
            
            if not matching_metrics:
                return {"name": name, "count": 0}
            
            # Calculate statistics
            values = [m.value for m in matching_metrics]
            metric_type = matching_metrics[0].metric_type
            
            summary = {
                "name": name,
                "type": metric_type.value,
                "count": len(values),
                "latest_value": values[-1] if values else 0,
                "timestamp": time.time()
            }
            
            if metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM, MetricType.TIMER]:
                summary.update({
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p50": self._percentile(values, 0.5) if values else 0,
                    "p95": self._percentile(values, 0.95) if values else 0,
                    "p99": self._percentile(values, 0.99) if values else 0
                })
            
            return summary
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(p * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory leaks."""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                
                cutoff_time = time.time() - self.retention_seconds
                
                with self._lock:
                    for key in list(self.metrics.keys()):
                        # Filter out old metrics
                        self.metrics[key] = [
                            m for m in self.metrics[key]
                            if m.timestamp > cutoff_time
                        ]
                        
                        # Remove empty entries
                        if not self.metrics[key]:
                            del self.metrics[key]
                
                logger.debug("Completed metrics cleanup")
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        with self._lock:
            return {
                "total_metric_keys": len(self.metrics),
                "total_data_points": sum(len(metrics) for metrics in self.metrics.values()),
                "counters_count": len(self.counters),
                "gauges_count": len(self.gauges),
                "histograms_count": len(self.histograms),
                "retention_hours": self.retention_seconds / 3600
            }


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Record as histogram
        self.collector.observe_histogram(
            f"{self.name}_duration_seconds",
            duration,
            self.labels
        )
        
        # Also record success/failure counter
        status = "success" if exc_type is None else "failure"
        labels = dict(self.labels) if self.labels else {}
        labels["status"] = status
        
        self.collector.increment_counter(
            f"{self.name}_total",
            1.0,
            labels
        )


class AlertManager:
    """Alert management system."""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        self.max_alerts = max_alerts
        
        self._lock = threading.RLock()
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        logger.info("Initialized alert manager")
    
    def add_alert_rule(self, rule_func: Callable[[Dict[str, Any]], Optional[Alert]]) -> None:
        """Add an alert rule function."""
        self.alert_rules.append(rule_func)
        logger.info(f"Added alert rule: {rule_func.__name__}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        new_alerts = []
        
        for rule_func in self.alert_rules:
            try:
                alert = rule_func(metrics)
                if alert and alert.id not in self.alerts:
                    self.create_alert(alert)
                    new_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_func.__name__}: {e}")
        
        return new_alerts
    
    def create_alert(self, alert: Alert) -> None:
        """Create a new alert."""
        with self._lock:
            # Ensure we don't exceed max alerts
            if len(self.alerts) >= self.max_alerts:
                # Remove oldest resolved alert
                oldest_resolved = None
                oldest_time = float('inf')
                
                for alert_id, existing_alert in self.alerts.items():
                    if existing_alert.resolved and existing_alert.timestamp < oldest_time:
                        oldest_time = existing_alert.timestamp
                        oldest_resolved = alert_id
                
                if oldest_resolved:
                    del self.alerts[oldest_resolved]
            
            self.alerts[alert.id] = alert
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in notification handler: {e}")
            
            logger.warning(f"Alert created: {alert.title} [{alert.severity.value}]")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolve()
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(self.alerts) - len(active_alerts),
                "severity_breakdown": dict(severity_counts)
            }


class DistributedTracer:
    """Distributed tracing system."""
    
    def __init__(self, service_name: str, max_spans: int = 10000):
        """Initialize distributed tracer."""
        self.service_name = service_name
        self.spans: Dict[str, TraceSpan] = {}
        self.traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [span_ids]
        self.max_spans = max_spans
        
        self._lock = threading.RLock()
        
        logger.info(f"Initialized distributed tracer for service: {service_name}")
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> TraceSpan:
        """Start a new span."""
        span_id = self._generate_span_id()
        trace_id = self._generate_trace_id() if parent_span_id is None else self._get_trace_id(parent_span_id)
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Add service information
        span.tags["service.name"] = self.service_name
        
        with self._lock:
            # Manage span storage limits
            if len(self.spans) >= self.max_spans:
                self._cleanup_old_spans()
            
            self.spans[span_id] = span
            self.traces[trace_id].append(span_id)
        
        return span
    
    def finish_span(self, span_id: str, status: str = "ok") -> None:
        """Finish a span."""
        with self._lock:
            if span_id in self.spans:
                self.spans[span_id].finish(status)
    
    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        import uuid
        return str(uuid.uuid4())[:16]
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        import uuid
        return str(uuid.uuid4())[:32]
    
    def _get_trace_id(self, span_id: str) -> str:
        """Get trace ID for a span."""
        with self._lock:
            if span_id in self.spans:
                return self.spans[span_id].trace_id
            return self._generate_trace_id()  # Fallback
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            span_ids = self.traces.get(trace_id, [])
            return [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
    
    def _cleanup_old_spans(self) -> None:
        """Clean up old spans to prevent memory leaks."""
        cutoff_time = time.time() - 3600  # Keep spans for 1 hour
        
        with self._lock:
            old_span_ids = [
                span_id for span_id, span in self.spans.items()
                if span.start_time < cutoff_time
            ]
            
            for span_id in old_span_ids:
                span = self.spans[span_id]
                del self.spans[span_id]
                
                # Clean up trace references
                if span.trace_id in self.traces:
                    self.traces[span.trace_id] = [
                        sid for sid in self.traces[span.trace_id]
                        if sid != span_id
                    ]
                    
                    # Remove empty traces
                    if not self.traces[span.trace_id]:
                        del self.traces[span.trace_id]
    
    def get_tracing_summary(self) -> Dict[str, Any]:
        """Get tracing system summary."""
        with self._lock:
            active_spans = sum(1 for span in self.spans.values() if span.end_time is None)
            
            return {
                "service_name": self.service_name,
                "total_spans": len(self.spans),
                "active_spans": active_spans,
                "completed_spans": len(self.spans) - active_spans,
                "total_traces": len(self.traces),
                "max_spans": self.max_spans
            }


# Sample alert rules
def high_cpu_alert_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
    """Alert rule for high CPU usage."""
    cpu_metric = metrics.get("cpu_utilization")
    if cpu_metric and cpu_metric.get("latest_value", 0) > 80:
        return Alert(
            id="high_cpu_usage",
            title="High CPU Usage",
            description=f"CPU utilization is {cpu_metric['latest_value']:.1f}%",
            severity=AlertSeverity.WARNING,
            component="system",
            metadata={"cpu_percent": cpu_metric["latest_value"]}
        )
    return None


def memory_leak_alert_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
    """Alert rule for potential memory leaks."""
    memory_metric = metrics.get("memory_usage")
    if memory_metric and memory_metric.get("latest_value", 0) > 90:
        return Alert(
            id="high_memory_usage",
            title="High Memory Usage",
            description=f"Memory usage is {memory_metric['latest_value']:.1f}%",
            severity=AlertSeverity.ERROR,
            component="system",
            metadata={"memory_percent": memory_metric["latest_value"]}
        )
    return None


def privacy_budget_alert_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
    """Alert rule for privacy budget exhaustion."""
    privacy_metric = metrics.get("privacy_budget_remaining")
    if privacy_metric and privacy_metric.get("latest_value", 100) < 10:
        return Alert(
            id="privacy_budget_low",
            title="Privacy Budget Nearly Exhausted",
            description=f"Only {privacy_metric['latest_value']:.1f}% privacy budget remaining",
            severity=AlertSeverity.CRITICAL,
            component="privacy",
            metadata={"remaining_percent": privacy_metric["latest_value"]}
        )
    return None


def test_production_monitoring_system():
    """Test the production monitoring system."""
    logger.info("=== Testing Production Monitoring System ===")
    
    try:
        # Test 1: Metrics Collection
        logger.info("--- Test 1: Metrics Collection ---")
        
        collector = MetricsCollector(retention_hours=1)
        
        # Collect various metrics
        collector.increment_counter("http_requests_total", labels={"method": "POST", "status": "200"})
        collector.increment_counter("http_requests_total", labels={"method": "POST", "status": "200"})
        collector.increment_counter("http_requests_total", labels={"method": "GET", "status": "404"})
        
        collector.set_gauge("cpu_utilization", 75.5)
        collector.set_gauge("memory_usage", 65.2)
        
        collector.observe_histogram("request_duration_seconds", 0.125)
        collector.observe_histogram("request_duration_seconds", 0.089)
        collector.observe_histogram("request_duration_seconds", 0.245)
        
        # Test timing context manager
        with collector.time_operation("database_query"):
            time.sleep(0.01)  # Simulate work
        
        # Get metrics summary
        http_summary = collector.get_metric_summary("http_requests_total")
        cpu_summary = collector.get_metric_summary("cpu_utilization")
        duration_summary = collector.get_metric_summary("request_duration_seconds")
        
        assert http_summary["count"] > 0, "Should have HTTP request metrics"
        assert cpu_summary["latest_value"] == 75.5, "Should track CPU utilization"
        assert duration_summary["p95"] > 0, "Should calculate percentiles"
        
        overall_summary = collector.get_all_metrics_summary()
        logger.info(f"✓ Collected metrics: {overall_summary['total_data_points']} data points")
        
        # Test 2: Alert Management
        logger.info("--- Test 2: Alert Management ---")
        
        alert_manager = AlertManager()
        
        # Add alert rules
        alert_manager.add_alert_rule(high_cpu_alert_rule)
        alert_manager.add_alert_rule(memory_leak_alert_rule)
        alert_manager.add_alert_rule(privacy_budget_alert_rule)
        
        # Add notification handler
        notifications = []
        def test_notification_handler(alert: Alert):
            notifications.append(alert)
        
        alert_manager.add_notification_handler(test_notification_handler)
        
        # Evaluate rules with high CPU
        test_metrics = {
            "cpu_utilization": {"latest_value": 85.0},
            "memory_usage": {"latest_value": 60.0},
            "privacy_budget_remaining": {"latest_value": 50.0}
        }
        
        new_alerts = alert_manager.evaluate_rules(test_metrics)
        assert len(new_alerts) > 0, "Should generate alerts for high CPU"
        assert len(notifications) > 0, "Should send notifications"
        
        # Test alert resolution
        alert_id = new_alerts[0].id
        resolved = alert_manager.resolve_alert(alert_id)
        assert resolved, "Should resolve alert"
        
        alert_summary = alert_manager.get_alert_summary()
        logger.info(f"✓ Alert summary: {alert_summary['active_alerts']} active, {alert_summary['resolved_alerts']} resolved")
        
        # Test 3: Distributed Tracing
        logger.info("--- Test 3: Distributed Tracing ---")
        
        tracer = DistributedTracer("test-service")
        
        # Create trace with multiple spans
        parent_span = tracer.start_span("federated_training_round")
        parent_span.tags["round"] = 1
        parent_span.tags["num_clients"] = 5
        
        # Child spans
        aggregation_span = tracer.start_span("parameter_aggregation", parent_span.span_id)
        aggregation_span.log("Starting aggregation", level="info", num_params=1000)
        
        validation_span = tracer.start_span("privacy_validation", parent_span.span_id)
        validation_span.tags["epsilon"] = 8.0
        
        # Finish spans
        time.sleep(0.001)  # Simulate work
        tracer.finish_span(aggregation_span.span_id, "ok")
        tracer.finish_span(validation_span.span_id, "ok")
        tracer.finish_span(parent_span.span_id, "ok")
        
        # Get trace
        trace_spans = tracer.get_trace(parent_span.trace_id)
        assert len(trace_spans) == 3, "Should have 3 spans in trace"
        
        tracing_summary = tracer.get_tracing_summary()
        logger.info(f"✓ Tracing: {tracing_summary['total_spans']} spans, {tracing_summary['total_traces']} traces")
        
        logger.info("✅ Production monitoring system test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Production monitoring system test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run production monitoring system tests."""
    logger.info("Starting production monitoring system tests...")
    
    tests = [
        test_production_monitoring_system,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Production Monitoring Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All production monitoring tests PASSED!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. See logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
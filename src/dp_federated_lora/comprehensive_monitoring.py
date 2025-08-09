"""
Comprehensive Monitoring System for DP-Federated LoRA.

This module implements advanced monitoring, alerting, and observability features
for the federated learning system, including real-time metrics, health checks,
performance monitoring, and automated anomaly detection.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import pickle
import numpy as np
import torch
import psutil
from pathlib import Path

from .config import FederatedConfig
from .exceptions import MonitoringError, DPFederatedLoRAError


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Container for metric values with metadata."""
    
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthCheck:
    """Health check result."""
    
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class MetricsCollector:
    """Advanced metrics collection and storage system."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector."""
        self.max_history = max_history
        self.metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(lambda: defaultdict(deque))
        self.lock = threading.RLock()
        
        # Built-in metrics
        self.start_time = time.time()
        self._initialize_system_metrics()
        
        logger.info("Advanced metrics collector initialized")
    
    def _initialize_system_metrics(self):
        """Initialize system-level metrics collection."""
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self.system_metrics_thread.start()
    
    def _collect_system_metrics(self):
        """Continuously collect system metrics."""
        while True:
            try:
                # CPU metrics
                self.record_gauge("system.cpu.usage", psutil.cpu_percent(interval=1))
                self.record_gauge("system.cpu.cores", psutil.cpu_count())
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge("system.memory.usage", memory.percent)
                self.record_gauge("system.memory.available", memory.available)
                self.record_gauge("system.memory.total", memory.total)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_gauge("system.disk.usage", (disk.used / disk.total) * 100)
                self.record_gauge("system.disk.free", disk.free)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.record_counter("system.network.bytes_sent", network.bytes_sent)
                self.record_counter("system.network.bytes_recv", network.bytes_recv)
                
                # GPU metrics (if available)
                try:
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(i)
                            memory_reserved = torch.cuda.memory_reserved(i)
                            self.record_gauge(f"gpu.{i}.memory_allocated", memory_allocated)
                            self.record_gauge(f"gpu.{i}.memory_reserved", memory_reserved)
                except Exception:
                    pass  # GPU metrics not critical
                
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(60)
    
    def record_counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        self._record_metric(name, MetricType.COUNTER, value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self._record_metric(name, MetricType.GAUGE, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels)
    
    def record_timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        self._record_metric(name, MetricType.TIMER, value, labels)
    
    def _record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric with thread safety."""
        with self.lock:
            metric_value = MetricValue(
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            metric_queue = self.metrics[name][metric_type]
            metric_queue.append(metric_value)
            
            # Maintain max history
            while len(metric_queue) > self.max_history:
                metric_queue.popleft()
    
    def get_metric_values(
        self,
        name: str,
        metric_type: MetricType,
        since: Optional[float] = None
    ) -> List[MetricValue]:
        """Get metric values, optionally filtered by time."""
        with self.lock:
            if name not in self.metrics or metric_type not in self.metrics[name]:
                return []
            
            values = list(self.metrics[name][metric_type])
            
            if since is not None:
                values = [v for v in values if v.timestamp >= since]
            
            return values
    
    def get_metric_summary(self, name: str, metric_type: MetricType) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        values = self.get_metric_values(name, metric_type)
        
        if not values:
            return {"count": 0}
        
        numeric_values = [v.value for v in values]
        
        return {
            "count": len(values),
            "mean": np.mean(numeric_values),
            "median": np.median(numeric_values),
            "std": np.std(numeric_values),
            "min": np.min(numeric_values),
            "max": np.max(numeric_values),
            "p95": np.percentile(numeric_values, 95),
            "p99": np.percentile(numeric_values, 99),
            "latest": values[-1].value,
            "oldest": values[0].value
        }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics in Prometheus format."""
        exported = {}
        
        with self.lock:
            for metric_name, metric_types in self.metrics.items():
                exported[metric_name] = {}
                for metric_type, values in metric_types.items():
                    exported[metric_name][metric_type.value] = [
                        {
                            "value": v.value,
                            "timestamp": v.timestamp,
                            "labels": v.labels
                        }
                        for v in values
                    ]
        
        return exported


class AlertManager:
    """Advanced alerting and notification system."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()
        
        # Built-in alert rules
        self._setup_default_alert_rules()
        
        logger.info("Alert manager initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for common issues."""
        self.add_alert_rule(
            "high_cpu_usage",
            condition=lambda metrics: self._check_high_cpu(metrics),
            severity=AlertSeverity.WARNING,
            title="High CPU Usage",
            description="CPU usage is above 80% for more than 5 minutes"
        )
        
        self.add_alert_rule(
            "high_memory_usage",
            condition=lambda metrics: self._check_high_memory(metrics),
            severity=AlertSeverity.WARNING,
            title="High Memory Usage",
            description="Memory usage is above 85% for more than 3 minutes"
        )
        
        self.add_alert_rule(
            "federated_training_stuck",
            condition=lambda metrics: self._check_training_stuck(metrics),
            severity=AlertSeverity.ERROR,
            title="Federated Training Stuck",
            description="No training progress detected for more than 10 minutes"
        )
    
    def add_alert_rule(
        self,
        rule_name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        title: str,
        description: str
    ):
        """Add a new alert rule."""
        self.alert_rules[rule_name] = {
            "condition": condition,
            "severity": severity,
            "title": title,
            "description": description,
            "last_triggered": 0.0
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def add_notification_channel(self, handler: Callable[[Alert], None]):
        """Add a notification channel."""
        self.notification_channels.append(handler)
        logger.info("Added notification channel")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule_config in self.alert_rules.items():
            try:
                if rule_config["condition"](metrics):
                    # Prevent spam - only trigger once per 5 minutes
                    if current_time - rule_config["last_triggered"] > 300:
                        alert = Alert(
                            alert_id=f"{rule_name}_{int(current_time)}",
                            severity=rule_config["severity"],
                            title=rule_config["title"],
                            description=rule_config["description"],
                            timestamp=current_time,
                            source="alert_manager"
                        )
                        
                        self.trigger_alert(alert)
                        rule_config["last_triggered"] = current_time
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications."""
        with self.lock:
            self.alerts.append(alert)
            
            # Send notifications
            for channel in self.notification_channels:
                try:
                    channel(alert)
                except Exception as e:
                    logger.error(f"Error sending alert notification: {e}")
            
            logger.warning(f"ALERT: {alert.title} - {alert.description}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_time = time.time()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def _check_high_cpu(self, metrics: Dict[str, Any]) -> bool:
        """Check for high CPU usage."""
        cpu_values = metrics.get("system.cpu.usage", {}).get("gauge", [])
        if len(cpu_values) < 5:
            return False
        
        # Check if CPU > 80% for last 5 values (5 minutes)
        recent_values = [v["value"] for v in cpu_values[-5:]]
        return all(v > 80 for v in recent_values)
    
    def _check_high_memory(self, metrics: Dict[str, Any]) -> bool:
        """Check for high memory usage."""
        memory_values = metrics.get("system.memory.usage", {}).get("gauge", [])
        if len(memory_values) < 3:
            return False
        
        # Check if memory > 85% for last 3 values
        recent_values = [v["value"] for v in memory_values[-3:]]
        return all(v > 85 for v in recent_values)
    
    def _check_training_stuck(self, metrics: Dict[str, Any]) -> bool:
        """Check if federated training appears stuck."""
        progress_values = metrics.get("federated.training.progress", {}).get("gauge", [])
        if not progress_values:
            return False
        
        # Check if no progress in last 10 minutes
        current_time = time.time()
        recent_progress = [
            v for v in progress_values 
            if current_time - v["timestamp"] < 600
        ]
        
        if len(recent_progress) < 2:
            return True
        
        # Check if progress values are the same
        progress_vals = [v["value"] for v in recent_progress]
        return len(set(progress_vals)) <= 1  # All values are the same


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.health_history: List[HealthCheck] = []
        self.lock = threading.RLock()
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("Health monitor initialized")
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("process_health", self._check_process_health)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a new health check."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.execution_time = time.time() - start_time
                
                with self.lock:
                    self.health_history.append(result)
                    # Keep only last 1000 health checks
                    if len(self.health_history) > 1000:
                        self.health_history.pop(0)
                
                results[name] = result
                
            except Exception as e:
                error_check = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time()
                )
                results[name] = error_check
                logger.error(f"Health check {name} failed: {e}")
        
        return results
    
    def get_overall_health(self) -> Tuple[HealthStatus, str]:
        """Get overall system health status."""
        health_results = self.run_health_checks()
        
        if not health_results:
            return HealthStatus.UNHEALTHY, "No health checks available"
        
        # Determine overall status based on individual checks
        critical_count = sum(1 for r in health_results.values() if r.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for r in health_results.values() if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in health_results.values() if r.status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            return HealthStatus.CRITICAL, f"{critical_count} critical issues detected"
        elif unhealthy_count > 0:
            return HealthStatus.UNHEALTHY, f"{unhealthy_count} unhealthy components"
        elif degraded_count > 0:
            return HealthStatus.DEGRADED, f"{degraded_count} degraded components"
        else:
            return HealthStatus.HEALTHY, "All systems operational"
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 90 or memory_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Critical resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        elif cpu_percent > 80 or memory_percent > 85:
            status = HealthStatus.UNHEALTHY
            message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        elif cpu_percent > 70 or memory_percent > 75:
            status = HealthStatus.DEGRADED
            message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            }
        )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        disk_usage = psutil.disk_usage('/')
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Critical disk usage: {usage_percent:.1f}%"
        elif usage_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High disk usage: {usage_percent:.1f}%"
        elif usage_percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Elevated disk usage: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {usage_percent:.1f}%"
        
        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                "usage_percent": usage_percent,
                "free_bytes": disk_usage.free,
                "total_bytes": disk_usage.total
            }
        )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage patterns."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_critical = memory.percent > 95
        swap_critical = swap.percent > 80 if swap.total > 0 else False
        
        if memory_critical or swap_critical:
            status = HealthStatus.CRITICAL
            message = f"Critical memory usage: RAM {memory.percent:.1f}%, Swap {swap.percent:.1f}%"
        elif memory.percent > 85:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: RAM {memory.percent:.1f}%, Swap {swap.percent:.1f}%"
        elif memory.percent > 75:
            status = HealthStatus.DEGRADED
            message = f"Elevated memory usage: RAM {memory.percent:.1f}%, Swap {swap.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: RAM {memory.percent:.1f}%, Swap {swap.percent:.1f}%"
        
        return HealthCheck(
            name="memory_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                "memory_percent": memory.percent,
                "swap_percent": swap.percent,
                "available_bytes": memory.available
            }
        )
    
    def _check_process_health(self) -> HealthCheck:
        """Check health of current process."""
        current_process = psutil.Process()
        
        # Check if process is responsive
        try:
            cpu_percent = current_process.cpu_percent()
            memory_info = current_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > 8192:  # > 8GB
                status = HealthStatus.UNHEALTHY
                message = f"High process memory usage: {memory_mb:.1f}MB"
            elif memory_mb > 4096:  # > 4GB
                status = HealthStatus.DEGRADED
                message = f"Elevated process memory usage: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process health normal: Memory {memory_mb:.1f}MB, CPU {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="process_health",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "num_threads": current_process.num_threads()
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Cannot access process information: {str(e)}",
                timestamp=time.time()
            )


class ComprehensiveMonitor:
    """Main comprehensive monitoring system."""
    
    def __init__(self, config: FederatedConfig):
        """Initialize comprehensive monitoring system."""
        self.config = config
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        
        # Monitoring state
        self.monitoring_active = True
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Setup notification channels
        self._setup_notification_channels()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Comprehensive monitoring system initialized")
    
    def _setup_notification_channels(self):
        """Setup default notification channels."""
        # Log-based notifications
        def log_notification(alert: Alert):
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            logger.log(level, f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.description}")
        
        self.alert_manager.add_notification_channel(log_notification)
        
        # File-based alert logging
        def file_notification(alert: Alert):
            try:
                alert_file = Path("alerts.log")
                with open(alert_file, "a") as f:
                    alert_data = {
                        "timestamp": alert.timestamp,
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "description": alert.description,
                        "source": alert.source
                    }
                    f.write(json.dumps(alert_data) + "\n")
            except Exception as e:
                logger.error(f"Failed to write alert to file: {e}")
        
        self.alert_manager.add_notification_channel(file_notification)
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Export current metrics
                current_metrics = self.metrics_collector.export_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                # Run health checks every 5 minutes
                if int(time.time()) % 300 == 0:
                    health_results = self.health_monitor.run_health_checks()
                    overall_health, health_message = self.health_monitor.get_overall_health()
                    
                    self.metrics_collector.record_gauge(
                        "system.health.overall",
                        1.0 if overall_health == HealthStatus.HEALTHY else 0.0
                    )
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def record_federated_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """Record federated learning specific metrics."""
        # Training progress
        self.metrics_collector.record_gauge("federated.training.round", round_num)
        self.metrics_collector.record_gauge("federated.training.progress", metrics.get("progress", 0.0))
        
        # Client metrics
        self.metrics_collector.record_gauge("federated.clients.active", metrics.get("active_clients", 0))
        self.metrics_collector.record_gauge("federated.clients.participating", metrics.get("participating_clients", 0))
        
        # Model performance
        if "accuracy" in metrics:
            self.metrics_collector.record_gauge("federated.model.accuracy", metrics["accuracy"])
        if "loss" in metrics:
            self.metrics_collector.record_gauge("federated.model.loss", metrics["loss"])
        
        # Privacy metrics
        if "privacy_spent" in metrics:
            self.metrics_collector.record_gauge("federated.privacy.epsilon_spent", metrics["privacy_spent"])
        if "privacy_remaining" in metrics:
            self.metrics_collector.record_gauge("federated.privacy.epsilon_remaining", metrics["privacy_remaining"])
        
        # Communication metrics
        if "communication_cost" in metrics:
            self.metrics_collector.record_gauge("federated.communication.cost", metrics["communication_cost"])
        if "aggregation_time" in metrics:
            self.metrics_collector.record_timer("federated.aggregation.time", metrics["aggregation_time"])
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        # Get current metrics
        current_metrics = self.metrics_collector.export_metrics()
        
        # Get health status
        overall_health, health_message = self.health_monitor.get_overall_health()
        health_checks = self.health_monitor.run_health_checks()
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # System overview
        system_overview = {
            "uptime": time.time() - self.metrics_collector.start_time,
            "health_status": overall_health.value,
            "health_message": health_message,
            "active_alerts": len(active_alerts),
            "monitoring_active": self.monitoring_active
        }
        
        # Recent metrics summary
        metrics_summary = {}
        for metric_name in ["system.cpu.usage", "system.memory.usage", "system.disk.usage"]:
            summary = self.metrics_collector.get_metric_summary(metric_name, MetricType.GAUGE)
            if summary["count"] > 0:
                metrics_summary[metric_name] = {
                    "current": summary["latest"],
                    "average": summary["mean"],
                    "max": summary["max"]
                }
        
        return {
            "system_overview": system_overview,
            "metrics_summary": metrics_summary,
            "health_checks": {name: check.status.value for name, check in health_checks.items()},
            "active_alerts": [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts
            ],
            "full_metrics": current_metrics
        }
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        current_metrics = self.metrics_collector.export_metrics()
        
        for metric_name, metric_types in current_metrics.items():
            for metric_type, values in metric_types.items():
                if not values:
                    continue
                
                # Add metric help and type
                lines.append(f"# HELP {metric_name} {metric_name} metric")
                lines.append(f"# TYPE {metric_name} {metric_type}")
                
                # Add latest value
                latest_value = values[-1]
                labels_str = ""
                if latest_value["labels"]:
                    labels_list = [f'{k}="{v}"' for k, v in latest_value["labels"].items()]
                    labels_str = "{" + ",".join(labels_list) + "}"
                
                lines.append(f"{metric_name}{labels_str} {latest_value['value']} {int(latest_value['timestamp']*1000)}")
        
        return "\n".join(lines)
    
    def create_performance_report(self) -> Dict[str, Any]:
        """Create comprehensive performance report."""
        current_time = time.time()
        last_hour = current_time - 3600
        
        # System performance
        cpu_values = self.metrics_collector.get_metric_values("system.cpu.usage", MetricType.GAUGE, last_hour)
        memory_values = self.metrics_collector.get_metric_values("system.memory.usage", MetricType.GAUGE, last_hour)
        
        cpu_summary = self.metrics_collector.get_metric_summary("system.cpu.usage", MetricType.GAUGE)
        memory_summary = self.metrics_collector.get_metric_summary("system.memory.usage", MetricType.GAUGE)
        
        # Training performance
        training_progress = self.metrics_collector.get_metric_values("federated.training.progress", MetricType.GAUGE, last_hour)
        model_accuracy = self.metrics_collector.get_metric_values("federated.model.accuracy", MetricType.GAUGE, last_hour)
        
        # Privacy performance
        privacy_spent = self.metrics_collector.get_metric_values("federated.privacy.epsilon_spent", MetricType.GAUGE, last_hour)
        
        return {
            "report_time": current_time,
            "time_range": "last_hour",
            "system_performance": {
                "cpu": cpu_summary,
                "memory": memory_summary,
                "health_status": self.health_monitor.get_overall_health()[0].value
            },
            "training_performance": {
                "progress_points": len(training_progress),
                "accuracy_points": len(model_accuracy),
                "latest_accuracy": model_accuracy[-1].value if model_accuracy else None,
                "accuracy_trend": "improving" if len(model_accuracy) >= 2 and model_accuracy[-1].value > model_accuracy[0].value else "stable"
            },
            "privacy_performance": {
                "privacy_points": len(privacy_spent),
                "latest_epsilon_spent": privacy_spent[-1].value if privacy_spent else None
            },
            "alerts_summary": {
                "total_alerts": len(self.alert_manager.alerts),
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "critical_alerts": len([a for a in self.alert_manager.get_active_alerts() if a.severity == AlertSeverity.CRITICAL])
            }
        }


def create_comprehensive_monitor(config: FederatedConfig) -> ComprehensiveMonitor:
    """
    Create comprehensive monitoring system.
    
    Args:
        config: Federated learning configuration
        
    Returns:
        Configured comprehensive monitor
    """
    return ComprehensiveMonitor(config)
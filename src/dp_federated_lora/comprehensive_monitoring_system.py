"""
Comprehensive Monitoring and Alerting System for Federated Learning.

This module provides real-time monitoring, alerting, and observability for
the federated learning system with focus on privacy, performance, and security.

Features:
- Real-time metrics collection and aggregation
- Distributed tracing and observability
- Automated alerting with escalation
- Performance profiling and optimization insights
- Privacy budget monitoring and forecasting
- Health checks and system diagnostics

Author: Terry (Terragon Labs)
"""

import asyncio
import logging
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
import psutil
import socket
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
from contextlib import contextmanager
import os
import sys


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTAGE = "percentage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.ACTIVE
    labels: Dict[str, str] = field(default_factory=dict)
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    resolution_time: Optional[float] = None
    acknowledged_by: Optional[str] = None
    escalation_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    response_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "response_time": self.response_time,
            "details": self.details
        }


class MetricsStorage:
    """Time-series metrics storage using SQLite."""
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize metrics storage."""
        self.db_path = db_path
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    labels TEXT,
                    unit TEXT,
                    description TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    labels TEXT,
                    metric_name TEXT,
                    threshold_value REAL,
                    actual_value REAL,
                    resolution_time REAL,
                    acknowledged_by TEXT,
                    escalation_level INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    response_time REAL,
                    details TEXT
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection from pool or create new one."""
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
            
            yield conn
            
        finally:
            if conn:
                with self.pool_lock:
                    if len(self.connection_pool) < 10:  # Max pool size
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
    
    def store_metric(self, metric: Metric):
        """Store a metric in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO metrics (name, value, metric_type, timestamp, labels, unit, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.metric_type.value,
                metric.timestamp,
                json.dumps(metric.labels),
                metric.unit,
                metric.description
            ))
            conn.commit()
    
    def store_metrics(self, metrics: List[Metric]):
        """Store multiple metrics efficiently."""
        with self._get_connection() as conn:
            data = [
                (m.name, m.value, m.metric_type.value, m.timestamp, 
                 json.dumps(m.labels), m.unit, m.description)
                for m in metrics
            ]
            
            conn.executemany("""
                INSERT INTO metrics (name, value, metric_type, timestamp, labels, unit, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
    
    def get_metrics(
        self,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve metrics from storage."""
        with self._get_connection() as conn:
            query = "SELECT * FROM metrics WHERE name = ?"
            params = [metric_name]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                metric_dict = dict(zip(columns, row))
                if metric_dict['labels']:
                    metric_dict['labels'] = json.loads(metric_dict['labels'])
                results.append(metric_dict)
            
            return results
    
    def store_alert(self, alert: Alert):
        """Store an alert in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    id, name, severity, message, timestamp, status, labels,
                    metric_name, threshold_value, actual_value, resolution_time,
                    acknowledged_by, escalation_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.name, alert.severity.value, alert.message,
                alert.timestamp, alert.status.value, json.dumps(alert.labels),
                alert.metric_name, alert.threshold_value, alert.actual_value,
                alert.resolution_time, alert.acknowledged_by, alert.escalation_level
            ))
            conn.commit()
    
    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve alerts from storage."""
        with self._get_connection() as conn:
            query = "SELECT * FROM alerts"
            params = []
            conditions = []
            
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            
            if severity:
                conditions.append("severity = ?")
                params.append(severity.value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                alert_dict = dict(zip(columns, row))
                if alert_dict['labels']:
                    alert_dict['labels'] = json.loads(alert_dict['labels'])
                results.append(alert_dict)
            
            return results
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old data from storage."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        with self._get_connection() as conn:
            # Clean up old metrics
            conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old health checks
            conn.execute("DELETE FROM health_checks WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up resolved alerts older than retention period
            conn.execute("""
                DELETE FROM alerts 
                WHERE status = 'resolved' AND resolution_time < ?
            """, (cutoff_time,))
            
            conn.commit()
            
            logger.info(f"Cleaned up data older than {retention_days} days")


class AlertManager:
    """Manages alert rules, notifications, and escalations."""
    
    def __init__(self, storage: MetricsStorage):
        """Initialize alert manager."""
        self.storage = storage
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.escalation_handlers: List[Callable[[Alert], None]] = []
        self._monitoring_active = False
        self._monitoring_thread = None
        
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,  # e.g., "value > 80"
        severity: AlertSeverity,
        message_template: str = None,
        labels: Dict[str, str] = None,
        cooldown_seconds: int = 300
    ):
        """Add an alert rule."""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "condition": condition,
            "severity": severity,
            "message_template": message_template or f"{metric_name} alert triggered",
            "labels": labels or {},
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": 0
        }
        
        logger.info(f"Added alert rule: {name} for metric {metric_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def add_escalation_handler(self, handler: Callable[[Alert], None]):
        """Add an escalation handler for unacknowledged alerts."""
        self.escalation_handlers.append(handler)
    
    def evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check cooldown
                if current_time - rule["last_triggered"] < rule["cooldown_seconds"]:
                    continue
                
                # Get recent metrics
                metrics = self.storage.get_metrics(
                    rule["metric_name"],
                    start_time=current_time - 300,  # Last 5 minutes
                    limit=10
                )
                
                if not metrics:
                    continue
                
                # Evaluate condition on most recent metric
                latest_metric = metrics[0]
                value = latest_metric["value"]
                
                # Simple condition evaluation (can be extended for complex expressions)
                condition_met = self._evaluate_condition(rule["condition"], value)
                
                if condition_met:
                    # Create alert
                    alert_id = f"{rule_name}_{int(current_time)}"
                    alert = Alert(
                        id=alert_id,
                        name=rule_name,
                        severity=rule["severity"],
                        message=rule["message_template"].format(
                            metric_name=rule["metric_name"],
                            value=value,
                            threshold=self._extract_threshold(rule["condition"])
                        ),
                        labels=rule["labels"].copy(),
                        metric_name=rule["metric_name"],
                        actual_value=value,
                        threshold_value=self._extract_threshold(rule["condition"])
                    )
                    
                    # Store alert
                    self.storage.store_alert(alert)
                    self.active_alerts[alert_id] = alert
                    
                    # Send notifications
                    self._send_notifications(alert)
                    
                    # Update last triggered time
                    rule["last_triggered"] = current_time
                    
                    logger.warning(f"Alert triggered: {rule_name} - {alert.message}")
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, condition: str, value: float) -> bool:
        """Evaluate a simple condition."""
        try:
            # Replace 'value' with actual value in condition string
            condition = condition.replace("value", str(value))
            return eval(condition)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _extract_threshold(self, condition: str) -> Optional[float]:
        """Extract threshold value from condition string."""
        try:
            # Simple regex-like extraction for common patterns
            import re
            match = re.search(r'([><=]+)\s*(\d+\.?\d*)', condition)
            if match:
                return float(match.group(2))
        except Exception:
            pass
        return None
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            self.storage.store_alert(alert)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        else:
            logger.warning(f"Alert {alert_id} not found")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolution_time = time.time()
            self.storage.store_alert(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved")
        else:
            logger.warning(f"Alert {alert_id} not found")
    
    def start_monitoring(self, check_interval: int = 60):
        """Start continuous alert monitoring."""
        if self._monitoring_active:
            logger.warning("Alert monitoring is already active")
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self.evaluate_alert_rules()
                    
                    # Check for escalations
                    current_time = time.time()
                    for alert in self.active_alerts.values():
                        if (alert.status == AlertStatus.ACTIVE and
                            current_time - alert.timestamp > 3600):  # 1 hour
                            alert.escalation_level += 1
                            
                            # Send escalation notifications
                            for handler in self.escalation_handlers:
                                try:
                                    handler(alert)
                                except Exception as e:
                                    logger.error(f"Error in escalation handler: {e}")
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in alert monitoring: {e}")
                    time.sleep(check_interval)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started alert monitoring")
    
    def stop_monitoring(self):
        """Stop continuous alert monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped alert monitoring")


class SystemMonitor:
    """System resource and health monitoring."""
    
    def __init__(self, storage: MetricsStorage):
        """Initialize system monitor."""
        self.storage = storage
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._monitoring_active = False
        self._monitoring_thread = None
        
    def add_health_check(self, name: str, check_function: Callable[[], HealthCheck]):
        """Add a health check function."""
        self.health_checks[name] = check_function
        logger.info(f"Added health check: {name}")
    
    def collect_system_metrics(self) -> List[Metric]:
        """Collect system resource metrics."""
        metrics = []
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric(
            name="system_cpu_usage_percent",
            value=cpu_percent,
            metric_type=MetricType.PERCENTAGE,
            timestamp=timestamp,
            unit="%",
            description="CPU usage percentage"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(Metric(
            name="system_memory_usage_percent",
            value=memory.percent,
            metric_type=MetricType.PERCENTAGE,
            timestamp=timestamp,
            unit="%",
            description="Memory usage percentage"
        ))
        
        metrics.append(Metric(
            name="system_memory_available_bytes",
            value=memory.available,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="bytes",
            description="Available memory in bytes"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(Metric(
            name="system_disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.PERCENTAGE,
            timestamp=timestamp,
            unit="%",
            description="Disk usage percentage"
        ))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics.append(Metric(
            name="system_network_bytes_sent",
            value=net_io.bytes_sent,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            unit="bytes",
            description="Total bytes sent"
        ))
        
        metrics.append(Metric(
            name="system_network_bytes_received",
            value=net_io.bytes_recv,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            unit="bytes",
            description="Total bytes received"
        ))
        
        # Process metrics
        process = psutil.Process()
        metrics.append(Metric(
            name="process_memory_usage_bytes",
            value=process.memory_info().rss,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="bytes",
            description="Process memory usage"
        ))
        
        metrics.append(Metric(
            name="process_cpu_percent",
            value=process.cpu_percent(),
            metric_type=MetricType.PERCENTAGE,
            timestamp=timestamp,
            unit="%",
            description="Process CPU usage percentage"
        ))
        
        return metrics
    
    def run_health_checks(self) -> List[HealthCheck]:
        """Run all registered health checks."""
        results = []
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                health_check = check_function()
                health_check.response_time = time.time() - start_time
                results.append(health_check)
                
                # Store in database
                with self.storage._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO health_checks (
                            component, status, message, timestamp, response_time, details
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        health_check.component,
                        health_check.status.value,
                        health_check.message,
                        health_check.timestamp,
                        health_check.response_time,
                        json.dumps(health_check.details)
                    ))
                    conn.commit()
                
            except Exception as e:
                error_check = HealthCheck(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )
                results.append(error_check)
                logger.error(f"Health check {name} failed: {e}")
        
        return results
    
    def start_monitoring(self, collect_interval: int = 60):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            logger.warning("System monitoring is already active")
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    # Collect system metrics
                    metrics = self.collect_system_metrics()
                    self.storage.store_metrics(metrics)
                    
                    # Run health checks
                    health_checks = self.run_health_checks()
                    
                    # Log summary
                    unhealthy_checks = [hc for hc in health_checks if hc.status != HealthStatus.HEALTHY]
                    if unhealthy_checks:
                        logger.warning(f"Unhealthy components: {[hc.component for hc in unhealthy_checks]}")
                    
                    time.sleep(collect_interval)
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(collect_interval)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped system monitoring")


class NotificationHandler:
    """Handles alert notifications via various channels."""
    
    def __init__(self):
        """Initialize notification handler."""
        self.email_config = None
        self.webhook_urls = []
        self.log_alerts = True
    
    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str]
    ):
        """Configure email notifications."""
        self.email_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails
        }
        logger.info("Configured email notifications")
    
    def add_webhook(self, webhook_url: str):
        """Add webhook URL for notifications."""
        self.webhook_urls.append(webhook_url)
        logger.info(f"Added webhook: {webhook_url}")
    
    def send_email_alert(self, alert: Alert):
        """Send alert via email."""
        if not self.email_config:
            logger.warning("Email not configured, cannot send email alert")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity.value}
            Message: {alert.message}
            Timestamp: {datetime.fromtimestamp(alert.timestamp)}
            
            Metric: {alert.metric_name}
            Threshold: {alert.threshold_value}
            Actual Value: {alert.actual_value}
            
            Labels: {alert.labels}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()
            
            logger.info(f"Sent email alert for {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, alert: Alert):
        """Send alert via webhook."""
        if not self.webhook_urls:
            return
        
        alert_data = alert.to_dict()
        
        for webhook_url in self.webhook_urls:
            try:
                response = requests.post(
                    webhook_url,
                    json=alert_data,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                logger.info(f"Sent webhook alert to {webhook_url}")
                
            except Exception as e:
                logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
    
    def send_log_alert(self, alert: Alert):
        """Log alert to application logs."""
        if self.log_alerts:
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.FATAL]:
                logger.critical(f"ALERT: {alert.name} - {alert.message}")
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(f"ALERT: {alert.name} - {alert.message}")
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(f"ALERT: {alert.name} - {alert.message}")
            else:
                logger.info(f"ALERT: {alert.name} - {alert.message}")
    
    def __call__(self, alert: Alert):
        """Handle an alert (called by AlertManager)."""
        # Always log alerts
        self.send_log_alert(alert)
        
        # Send email for ERROR, CRITICAL, FATAL
        if alert.severity.value in ['error', 'critical', 'fatal']:
            self.send_email_alert(alert)
        
        # Send webhook for all alerts
        self.send_webhook_alert(alert)


class ComprehensiveMonitoringSystem:
    """Main monitoring system that orchestrates all components."""
    
    def __init__(
        self,
        db_path: str = "monitoring.db",
        retention_days: int = 30
    ):
        """Initialize comprehensive monitoring system."""
        self.storage = MetricsStorage(db_path)
        self.alert_manager = AlertManager(self.storage)
        self.system_monitor = SystemMonitor(self.storage)
        self.notification_handler = NotificationHandler()
        self.retention_days = retention_days
        
        # Add notification handler to alert manager
        self.alert_manager.add_notification_handler(self.notification_handler)
        
        # Custom metrics
        self.custom_metrics: deque = deque(maxlen=10000)
        self._metrics_lock = threading.Lock()
        
        # Background tasks
        self._cleanup_thread = None
        self._cleanup_active = False
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Initialized comprehensive monitoring system")
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        def database_health_check():
            try:
                with self.storage._get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                return HealthCheck(
                    component="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful"
                )
            except Exception as e:
                return HealthCheck(
                    component="database",
                    status=HealthStatus.CRITICAL,
                    message=f"Database connection failed: {str(e)}"
                )
        
        def disk_space_health_check():
            try:
                disk = psutil.disk_usage('/')
                usage_percent = (disk.used / disk.total) * 100
                
                if usage_percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"Disk space critical: {usage_percent:.1f}% used"
                elif usage_percent > 80:
                    status = HealthStatus.WARNING
                    message = f"Disk space warning: {usage_percent:.1f}% used"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Disk space healthy: {usage_percent:.1f}% used"
                
                return HealthCheck(
                    component="disk_space",
                    status=status,
                    message=message,
                    details={"usage_percent": usage_percent}
                )
            except Exception as e:
                return HealthCheck(
                    component="disk_space",
                    status=HealthStatus.UNKNOWN,
                    message=f"Failed to check disk space: {str(e)}"
                )
        
        def memory_health_check():
            try:
                memory = psutil.virtual_memory()
                
                if memory.percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"Memory usage critical: {memory.percent:.1f}%"
                elif memory.percent > 80:
                    status = HealthStatus.WARNING
                    message = f"Memory usage warning: {memory.percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage healthy: {memory.percent:.1f}%"
                
                return HealthCheck(
                    component="memory",
                    status=status,
                    message=message,
                    details={"usage_percent": memory.percent}
                )
            except Exception as e:
                return HealthCheck(
                    component="memory",
                    status=HealthStatus.UNKNOWN,
                    message=f"Failed to check memory: {str(e)}"
                )
        
        self.system_monitor.add_health_check("database", database_health_check)
        self.system_monitor.add_health_check("disk_space", disk_space_health_check)
        self.system_monitor.add_health_check("memory", memory_health_check)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            metric_name="system_cpu_usage_percent",
            condition="value > 80",
            severity=AlertSeverity.WARNING,
            message_template="High CPU usage: {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=300
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            metric_name="system_memory_usage_percent",
            condition="value > 85",
            severity=AlertSeverity.WARNING,
            message_template="High memory usage: {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=300
        )
        
        # Critical memory usage
        self.alert_manager.add_alert_rule(
            name="critical_memory_usage",
            metric_name="system_memory_usage_percent",
            condition="value > 95",
            severity=AlertSeverity.CRITICAL,
            message_template="Critical memory usage: {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=180
        )
        
        # High disk usage
        self.alert_manager.add_alert_rule(
            name="high_disk_usage",
            metric_name="system_disk_usage_percent",
            condition="value > 85",
            severity=AlertSeverity.WARNING,
            message_template="High disk usage: {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=600
        )
        
        # Critical disk usage
        self.alert_manager.add_alert_rule(
            name="critical_disk_usage",
            metric_name="system_disk_usage_percent",
            condition="value > 95",
            severity=AlertSeverity.CRITICAL,
            message_template="Critical disk usage: {value:.1f}% (threshold: {threshold}%)",
            cooldown_seconds=300
        )
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
        unit: str = "",
        description: str = ""
    ):
        """Record a custom metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
            description=description
        )
        
        # Store in database
        self.storage.store_metric(metric)
        
        # Keep in memory for recent access
        with self._metrics_lock:
            self.custom_metrics.append(metric)
    
    def record_metrics(self, metrics: List[Metric]):
        """Record multiple custom metrics."""
        # Store in database
        self.storage.store_metrics(metrics)
        
        # Keep in memory for recent access
        with self._metrics_lock:
            self.custom_metrics.extend(metrics)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics over specified time period."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get all unique metric names
        with self.storage._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT name FROM metrics 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_time, end_time))
            metric_names = [row[0] for row in cursor.fetchall()]
        
        summary = {}
        for metric_name in metric_names:
            metrics = self.storage.get_metrics(metric_name, start_time, end_time, limit=10000)
            
            if metrics:
                values = [m['value'] for m in metrics]
                summary[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "latest": values[0] if values else None,  # Most recent first
                    "unit": metrics[0].get('unit', ''),
                    "description": metrics[0].get('description', '')
                }
                
                if len(values) > 1:
                    summary[metric_name]["stddev"] = statistics.stdev(values)
        
        return summary
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        active_alerts = self.alert_manager.storage.get_alerts(status=AlertStatus.ACTIVE)
        acknowledged_alerts = self.alert_manager.storage.get_alerts(status=AlertStatus.ACKNOWLEDGED)
        resolved_alerts = self.alert_manager.storage.get_alerts(status=AlertStatus.RESOLVED, limit=50)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts + acknowledged_alerts:
            severity_counts[alert['severity']] += 1
        
        return {
            "active_count": len(active_alerts),
            "acknowledged_count": len(acknowledged_alerts),
            "recent_resolved_count": len(resolved_alerts),
            "severity_distribution": dict(severity_counts),
            "recent_active_alerts": active_alerts[:10],
            "recent_resolved_alerts": resolved_alerts[:10]
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health."""
        health_checks = self.system_monitor.run_health_checks()
        
        status_counts = defaultdict(int)
        unhealthy_components = []
        
        for check in health_checks:
            status_counts[check.status.value] += 1
            if check.status != HealthStatus.HEALTHY:
                unhealthy_components.append({
                    "component": check.component,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time
                })
        
        # Overall system health
        if any(check.status == HealthStatus.CRITICAL for check in health_checks):
            overall_status = HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.UNHEALTHY for check in health_checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.WARNING for check in health_checks):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "total_components": len(health_checks),
            "status_distribution": dict(status_counts),
            "unhealthy_components": unhealthy_components,
            "last_check_time": time.time()
        }
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create comprehensive dashboard data."""
        return {
            "timestamp": time.time(),
            "metrics_summary": self.get_metrics_summary(hours=1),  # Last hour
            "alerts_summary": self.get_alerts_summary(),
            "health_summary": self.get_health_summary(),
            "system_info": {
                "hostname": socket.gethostname(),
                "python_version": sys.version,
                "uptime_seconds": time.time() - psutil.boot_time(),
                "monitoring_db_path": self.storage.db_path
            }
        }
    
    def start_all_monitoring(
        self,
        system_collect_interval: int = 60,
        alert_check_interval: int = 30
    ):
        """Start all monitoring components."""
        # Start system monitoring
        self.system_monitor.start_monitoring(system_collect_interval)
        
        # Start alert monitoring
        self.alert_manager.start_monitoring(alert_check_interval)
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("Started all monitoring components")
    
    def stop_all_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        self.alert_manager.stop_monitoring()
        self._stop_cleanup_task()
        
        logger.info("Stopped all monitoring components")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_active:
            return
        
        self._cleanup_active = True
        
        def cleanup_loop():
            while self._cleanup_active:
                try:
                    # Run cleanup daily
                    time.sleep(24 * 3600)
                    if self._cleanup_active:
                        self.storage.cleanup_old_data(self.retention_days)
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _stop_cleanup_task(self):
        """Stop background cleanup task."""
        self._cleanup_active = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)
    
    def export_monitoring_data(self, export_path: str, hours: int = 24):
        """Export monitoring data for analysis."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        export_data = {
            "export_metadata": {
                "export_time": end_time,
                "start_time": start_time,
                "end_time": end_time,
                "hours_covered": hours
            },
            "metrics_summary": self.get_metrics_summary(hours),
            "alerts": self.alert_manager.storage.get_alerts(limit=1000),
            "health_checks": [],  # Would need to query health_checks table
            "dashboard_data": self.create_dashboard_data()
        }
        
        # Get detailed metrics for top metrics
        metrics_summary = export_data["metrics_summary"]
        detailed_metrics = {}
        
        for metric_name in list(metrics_summary.keys())[:10]:  # Top 10 metrics
            detailed_metrics[metric_name] = self.storage.get_metrics(
                metric_name, start_time, end_time, limit=1000
            )
        
        export_data["detailed_metrics"] = detailed_metrics
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported monitoring data to {export_path}")


# Factory function for easy instantiation
def create_monitoring_system(
    db_path: str = "monitoring.db",
    retention_days: int = 30,
    auto_start: bool = True
) -> ComprehensiveMonitoringSystem:
    """Create and optionally start a comprehensive monitoring system."""
    system = ComprehensiveMonitoringSystem(db_path, retention_days)
    
    if auto_start:
        system.start_all_monitoring()
    
    return system


if __name__ == "__main__":
    # Demonstration of comprehensive monitoring system
    import random
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Monitoring System Demo")
    parser.add_argument("--duration", type=int, default=300, help="Demo duration in seconds")
    parser.add_argument("--db-path", type=str, default="demo_monitoring.db", help="Database path")
    
    args = parser.parse_args()
    
    # Create monitoring system
    monitoring = create_monitoring_system(db_path=args.db_path, auto_start=True)
    
    # Configure notifications (example)
    monitoring.notification_handler.configure_email(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="demo@example.com",
        password="demo_password",
        from_email="monitoring@example.com",
        to_emails=["admin@example.com"]
    )
    
    print(f"\nStarted comprehensive monitoring system demonstration")
    print(f"Database: {args.db_path}")
    print(f"Duration: {args.duration} seconds")
    print("-" * 60)
    
    # Simulate some application metrics
    start_time = time.time()
    
    while time.time() - start_time < args.duration:
        try:
            # Simulate federated learning metrics
            monitoring.record_metric(
                name="federated_clients_active",
                value=random.randint(5, 15),
                metric_type=MetricType.GAUGE,
                labels={"environment": "demo"},
                unit="clients",
                description="Number of active federated clients"
            )
            
            monitoring.record_metric(
                name="privacy_budget_consumed",
                value=random.uniform(0, 100),
                metric_type=MetricType.GAUGE,
                labels={"budget_type": "epsilon"},
                unit="epsilon",
                description="Total privacy budget consumed"
            )
            
            monitoring.record_metric(
                name="model_accuracy",
                value=random.uniform(0.7, 0.95),
                metric_type=MetricType.GAUGE,
                labels={"model": "llama-7b"},
                unit="accuracy",
                description="Global model accuracy"
            )
            
            monitoring.record_metric(
                name="training_round_duration",
                value=random.uniform(30, 180),
                metric_type=MetricType.HISTOGRAM,
                labels={"round_type": "standard"},
                unit="seconds",
                description="Training round duration"
            )
            
            # Occasionally trigger alerts with high values
            if random.random() < 0.1:  # 10% chance
                monitoring.record_metric(
                    name="system_cpu_usage_percent",
                    value=random.uniform(85, 95),
                    metric_type=MetricType.PERCENTAGE,
                    unit="%",
                    description="High CPU usage for alert demo"
                )
            
            # Print status every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                dashboard_data = monitoring.create_dashboard_data()
                print(f"Time: {int(time.time() - start_time)}s | "
                      f"Metrics: {len(dashboard_data['metrics_summary'])} | "
                      f"Active Alerts: {dashboard_data['alerts_summary']['active_count']} | "
                      f"Health: {dashboard_data['health_summary']['overall_status']}")
            
            time.sleep(5)  # Collect metrics every 5 seconds
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in demo loop: {e}")
    
    print(f"\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)
    
    # Generate final dashboard
    dashboard_data = monitoring.create_dashboard_data()
    
    print(f"Metrics Collected:")
    for metric_name, summary in dashboard_data['metrics_summary'].items():
        print(f"  {metric_name}: {summary['count']} points, "
              f"latest={summary['latest']:.2f} {summary['unit']}")
    
    print(f"\nAlerts:")
    alerts_summary = dashboard_data['alerts_summary']
    print(f"  Active: {alerts_summary['active_count']}")
    print(f"  Acknowledged: {alerts_summary['acknowledged_count']}")
    print(f"  Recently Resolved: {alerts_summary['recent_resolved_count']}")
    
    print(f"\nSystem Health:")
    health_summary = dashboard_data['health_summary']
    print(f"  Overall Status: {health_summary['overall_status']}")
    print(f"  Components Monitored: {health_summary['total_components']}")
    
    if health_summary['unhealthy_components']:
        print(f"  Unhealthy Components:")
        for component in health_summary['unhealthy_components']:
            print(f"    {component['component']}: {component['status']} - {component['message']}")
    
    # Export data
    export_file = f"monitoring_export_{int(time.time())}.json"
    monitoring.export_monitoring_data(export_file, hours=1)
    print(f"\nMonitoring data exported to: {export_file}")
    
    # Stop monitoring
    monitoring.stop_all_monitoring()
    print(f"\nComprehensive monitoring system demonstration completed!")
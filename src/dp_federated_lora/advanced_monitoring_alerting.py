"""
Advanced Monitoring and Alerting System

Comprehensive monitoring, alerting, and observability system with:
- Quantum-aware health checks and performance monitoring
- Advanced anomaly detection with quantum statistics
- Performance degradation detection for quantum components
- Real-time alerting with intelligent severity assessment
- Distributed tracing and comprehensive logging
"""

import asyncio
import logging
import time
import statistics
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
import hashlib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .quantum_monitoring import QuantumMetricsCollector, QuantumMetricType, QuantumHealthCheck
from .quantum_error_recovery import QuantumErrorRecoverySystem
from .security_fortress import SecurityFortress, ThreatLevel
from .exceptions import MonitoringError, QuantumSchedulingError
from .config import FederatedConfig


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class AlertCategory(Enum):
    """Alert categories"""
    PERFORMANCE = auto()
    SECURITY = auto()
    QUANTUM = auto()
    SYSTEM = auto()
    BUSINESS = auto()
    COMPLIANCE = auto()


class MonitoringScope(Enum):
    """Monitoring scope levels"""
    CLIENT = auto()
    SERVER = auto()
    GLOBAL = auto()
    QUANTUM_COMPONENT = auto()


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    scope: MonitoringScope
    source_component: str
    detection_time: datetime
    metric_values: Dict[str, Any]
    recommended_actions: List[str]
    auto_resolution_attempted: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.name,
            "category": self.category.name,
            "scope": self.scope.name,
            "source_component": self.source_component,
            "detection_time": self.detection_time.isoformat(),
            "metric_values": self.metric_values,
            "recommended_actions": self.recommended_actions,
            "auto_resolution_attempted": self.auto_resolution_attempted,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection"""
    component_id: str
    metric_name: str
    baseline_value: float
    variance: float
    sample_count: int
    last_updated: datetime
    confidence_interval: Tuple[float, float]
    
    def is_anomaly(self, value: float, sensitivity: float = 2.0) -> bool:
        """Check if value is anomalous compared to baseline"""
        if self.variance == 0:
            return abs(value - self.baseline_value) > 0.01  # Small threshold for zero variance
        
        z_score = abs(value - self.baseline_value) / (self.variance ** 0.5)
        return z_score > sensitivity


class QuantumAwareAnomalyDetector:
    """Advanced anomaly detection with quantum statistics support"""
    
    def __init__(self, 
                 window_size: int = 200,
                 contamination: float = 0.1,
                 quantum_sensitivity: float = 1.5):
        self.window_size = window_size
        self.contamination = contamination
        self.quantum_sensitivity = quantum_sensitivity
        
        # Machine learning models for anomaly detection
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
        # Data storage
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.quantum_baselines: Dict[str, PerformanceBaseline] = {}
        self.trained_models: Dict[str, bool] = defaultdict(bool)
        
        self.logger = logging.getLogger(__name__)
        
    def add_metric_sample(self, 
                         component_id: str,
                         metric_name: str, 
                         value: float,
                         timestamp: datetime = None) -> Tuple[bool, float]:
        """Add metric sample and detect anomalies"""
        if timestamp is None:
            timestamp = datetime.now()
            
        metric_key = f"{component_id}_{metric_name}"
        history = self.metric_history[metric_key]
        
        # Add sample to history
        history.append({
            "value": value,
            "timestamp": timestamp,
            "component_id": component_id,
            "metric_name": metric_name
        })
        
        # Check for quantum-specific anomalies
        if "quantum" in metric_name.lower():
            return self._detect_quantum_anomaly(metric_key, value)
        else:
            return self._detect_standard_anomaly(metric_key, value)
            
    def _detect_quantum_anomaly(self, metric_key: str, value: float) -> Tuple[bool, float]:
        """Detect anomalies in quantum metrics using quantum-aware methods"""
        history = self.metric_history[metric_key]
        
        if len(history) < 10:  # Need minimum samples
            return False, 0.0
            
        values = [sample["value"] for sample in history]
        
        # Quantum coherence analysis
        if "coherence" in metric_key.lower():
            # Coherence should be between 0 and 1
            if value < 0 or value > 1:
                return True, 10.0  # Definite anomaly
                
            # Check for rapid coherence loss
            recent_values = values[-5:]
            if len(recent_values) >= 3:
                coherence_loss_rate = (recent_values[0] - recent_values[-1]) / len(recent_values)
                if coherence_loss_rate > 0.2:  # 20% loss per sample
                    return True, coherence_loss_rate * 5
                    
        # Entanglement analysis
        elif "entanglement" in metric_key.lower():
            # Check for entanglement sudden death
            if value < 0.1 and statistics.mean(values[-10:]) > 0.5:
                return True, 5.0
                
        # Quantum fidelity analysis
        elif "fidelity" in metric_key.lower():
            # Fidelity should be between 0 and 1
            if value < 0 or value > 1:
                return True, 10.0
                
            # Check for fidelity degradation
            if len(values) >= 5:
                recent_avg = statistics.mean(values[-5:])
                older_avg = statistics.mean(values[-10:-5]) if len(values) >= 10 else recent_avg
                
                if recent_avg < older_avg - 0.1:  # Significant degradation
                    return True, (older_avg - recent_avg) * 10
                    
        # Standard statistical analysis with quantum sensitivity
        return self._statistical_anomaly_check(values, value, self.quantum_sensitivity)
        
    def _detect_standard_anomaly(self, metric_key: str, value: float) -> Tuple[bool, float]:
        """Detect anomalies in standard metrics"""
        history = self.metric_history[metric_key]
        
        if len(history) < 20:  # Need more samples for standard metrics
            return False, 0.0
            
        values = [sample["value"] for sample in history]
        return self._statistical_anomaly_check(values, value, 2.0)
        
    def _statistical_anomaly_check(self, values: List[float], current_value: float, sensitivity: float) -> Tuple[bool, float]:
        """Perform statistical anomaly check"""
        if len(values) < 5:
            return False, 0.0
            
        try:
            mean_val = statistics.mean(values[:-1])  # Exclude current value
            stdev_val = statistics.stdev(values[:-1]) if len(values) > 2 else 0.0
            
            if stdev_val == 0:
                return False, 0.0
                
            z_score = abs(current_value - mean_val) / stdev_val
            is_anomaly = z_score > sensitivity
            
            return is_anomaly, z_score
            
        except Exception as e:
            self.logger.error(f"Statistical anomaly check failed: {e}")
            return False, 0.0
            
    def get_baseline_for_metric(self, component_id: str, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get performance baseline for metric"""
        metric_key = f"{component_id}_{metric_name}"
        history = self.metric_history[metric_key]
        
        if len(history) < 30:  # Need sufficient data for baseline
            return None
            
        values = [sample["value"] for sample in history]
        
        baseline = PerformanceBaseline(
            component_id=component_id,
            metric_name=metric_name,
            baseline_value=statistics.mean(values),
            variance=statistics.variance(values) if len(values) > 1 else 0.0,
            sample_count=len(values),
            last_updated=datetime.now(),
            confidence_interval=(
                min(values),
                max(values)
            )
        )
        
        self.quantum_baselines[metric_key] = baseline
        return baseline
        
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get comprehensive anomaly detection report"""
        total_metrics = len(self.metric_history)
        total_samples = sum(len(history) for history in self.metric_history.values())
        
        # Calculate baseline coverage
        baseline_coverage = len(self.quantum_baselines) / total_metrics if total_metrics > 0 else 0.0
        
        return {
            "total_monitored_metrics": total_metrics,
            "total_samples": total_samples,
            "baseline_coverage": baseline_coverage,
            "active_baselines": len(self.quantum_baselines),
            "model_settings": {
                "window_size": self.window_size,
                "contamination": self.contamination,
                "quantum_sensitivity": self.quantum_sensitivity
            }
        }


class PerformanceDegradationDetector:
    """Detects performance degradation in quantum and classical components"""
    
    def __init__(self, degradation_threshold: float = 0.15):
        self.degradation_threshold = degradation_threshold
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.degradation_alerts: List[Alert] = []
        self.logger = logging.getLogger(__name__)
        
    def record_performance_sample(self,
                                component_id: str,
                                performance_metrics: Dict[str, float],
                                timestamp: datetime = None) -> List[Alert]:
        """Record performance sample and check for degradation"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Record sample
        sample = {
            "timestamp": timestamp,
            "metrics": performance_metrics.copy()
        }
        self.performance_history[component_id].append(sample)
        
        # Keep only recent history (last 100 samples)
        if len(self.performance_history[component_id]) > 100:
            self.performance_history[component_id] = self.performance_history[component_id][-100:]
            
        # Check for degradation
        return self._check_performance_degradation(component_id)
        
    def _check_performance_degradation(self, component_id: str) -> List[Alert]:
        """Check for performance degradation patterns"""
        history = self.performance_history[component_id]
        alerts = []
        
        if len(history) < 10:  # Need minimum samples
            return alerts
            
        # Analyze each metric
        for metric_name in history[-1]["metrics"].keys():
            degradation_detected, severity = self._analyze_metric_degradation(
                component_id, metric_name, history
            )
            
            if degradation_detected:
                alert = Alert(
                    alert_id=f"perf_deg_{component_id}_{metric_name}_{int(time.time())}",
                    title=f"Performance Degradation Detected: {metric_name}",
                    description=f"Component {component_id} showing degradation in {metric_name}",
                    severity=severity,
                    category=AlertCategory.PERFORMANCE,
                    scope=MonitoringScope.QUANTUM_COMPONENT if "quantum" in component_id.lower() else MonitoringScope.CLIENT,
                    source_component=component_id,
                    detection_time=datetime.now(),
                    metric_values={"degraded_metric": metric_name, "component": component_id},
                    recommended_actions=self._get_degradation_remediation_actions(metric_name)
                )
                alerts.append(alert)
                self.degradation_alerts.append(alert)
                
        return alerts
        
    def _analyze_metric_degradation(self, 
                                  component_id: str,
                                  metric_name: str,
                                  history: List[Dict[str, Any]]) -> Tuple[bool, AlertSeverity]:
        """Analyze specific metric for degradation"""
        
        # Extract metric values
        values = []
        for sample in history:
            if metric_name in sample["metrics"]:
                values.append(sample["metrics"][metric_name])
                
        if len(values) < 10:
            return False, AlertSeverity.INFO
            
        # Compare recent performance to baseline
        recent_values = values[-5:]  # Last 5 samples
        baseline_values = values[-20:-5] if len(values) >= 20 else values[:-5]
        
        if not baseline_values:
            return False, AlertSeverity.INFO
            
        recent_avg = statistics.mean(recent_values)
        baseline_avg = statistics.mean(baseline_values)
        
        # Calculate degradation percentage
        if baseline_avg == 0:
            return False, AlertSeverity.INFO
            
        # For metrics where higher is better (like fidelity, coherence)
        if "fidelity" in metric_name.lower() or "coherence" in metric_name.lower() or "efficiency" in metric_name.lower():
            degradation_ratio = (baseline_avg - recent_avg) / baseline_avg
        else:
            # For metrics where lower is better (like error_rate, latency)
            degradation_ratio = (recent_avg - baseline_avg) / baseline_avg
            
        # Determine severity based on degradation
        if degradation_ratio > 0.5:  # 50% degradation
            return True, AlertSeverity.CRITICAL
        elif degradation_ratio > 0.3:  # 30% degradation
            return True, AlertSeverity.ERROR
        elif degradation_ratio > self.degradation_threshold:
            return True, AlertSeverity.WARNING
        else:
            return False, AlertSeverity.INFO
            
    def _get_degradation_remediation_actions(self, metric_name: str) -> List[str]:
        """Get recommended actions for performance degradation"""
        actions = []
        
        if "quantum" in metric_name.lower():
            actions.extend([
                "Check quantum circuit depth and complexity",
                "Verify quantum error correction is functioning",
                "Review quantum noise levels and coherence time",
                "Consider circuit optimization and error mitigation"
            ])
            
        if "fidelity" in metric_name.lower():
            actions.extend([
                "Recalibrate quantum gates",
                "Check for environmental interference",
                "Review quantum state preparation"
            ])
            
        if "coherence" in metric_name.lower():
            actions.extend([
                "Check temperature and electromagnetic environment",
                "Review isolation and shielding",
                "Consider dynamic decoupling sequences"
            ])
            
        if "latency" in metric_name.lower() or "response" in metric_name.lower():
            actions.extend([
                "Check network connectivity and bandwidth",
                "Review computational resource allocation",
                "Consider load balancing and scaling"
            ])
            
        if not actions:  # Generic actions
            actions = [
                "Review system logs for errors",
                "Check resource utilization",
                "Verify network connectivity",
                "Consider system restart if issues persist"
            ]
            
        return actions


class IntelligentAlertManager:
    """Intelligent alert management with auto-resolution and escalation"""
    
    def __init__(self, 
                 auto_resolution_enabled: bool = True,
                 escalation_timeout: timedelta = timedelta(minutes=15)):
        self.auto_resolution_enabled = auto_resolution_enabled
        self.escalation_timeout = escalation_timeout
        
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.alert_rules: Dict[AlertCategory, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        
        self.logger = logging.getLogger(__name__)
        self._initialize_alert_rules()
        
    def _initialize_alert_rules(self):
        """Initialize alert processing rules"""
        self.alert_rules = {
            AlertCategory.QUANTUM: {
                "auto_resolution_attempts": 3,
                "escalation_levels": [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL],
                "notification_required": True
            },
            AlertCategory.SECURITY: {
                "auto_resolution_attempts": 1,  # Security alerts need human attention
                "escalation_levels": [AlertSeverity.ERROR, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
                "notification_required": True
            },
            AlertCategory.PERFORMANCE: {
                "auto_resolution_attempts": 5,
                "escalation_levels": [AlertSeverity.WARNING, AlertSeverity.ERROR],
                "notification_required": False
            },
            AlertCategory.SYSTEM: {
                "auto_resolution_attempts": 3,
                "escalation_levels": [AlertSeverity.ERROR, AlertSeverity.CRITICAL],
                "notification_required": True
            }
        }
        
    async def process_alert(self, alert: Alert) -> Dict[str, Any]:
        """Process incoming alert with intelligent handling"""
        
        # Check for duplicate alerts
        if self._is_duplicate_alert(alert):
            self.logger.info(f"Duplicate alert suppressed: {alert.alert_id}")
            return {"status": "suppressed", "reason": "duplicate"}
            
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        
        # Get processing rules
        rules = self.alert_rules.get(alert.category, {})
        
        # Attempt auto-resolution if enabled
        resolution_result = None
        if self.auto_resolution_enabled and rules.get("auto_resolution_attempts", 0) > 0:
            resolution_result = await self._attempt_auto_resolution(alert)
            
        # Send notifications if required
        if rules.get("notification_required", False) or alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._send_notifications(alert)
            
        # Schedule escalation if not resolved
        if not resolution_result or not resolution_result.get("resolved", False):
            asyncio.create_task(self._schedule_escalation(alert))
            
        self.logger.info(f"Alert processed: {alert.alert_id} - {alert.title}")
        
        return {
            "status": "processed",
            "alert_id": alert.alert_id,
            "auto_resolution_attempted": resolution_result is not None,
            "auto_resolved": resolution_result.get("resolved", False) if resolution_result else False
        }
        
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if alert is a duplicate of existing active alert"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.source_component == new_alert.source_component and
                existing_alert.category == new_alert.category and
                existing_alert.title == new_alert.title and
                not existing_alert.resolved):
                return True
        return False
        
    async def _attempt_auto_resolution(self, alert: Alert) -> Dict[str, Any]:
        """Attempt automatic resolution of alert"""
        self.logger.info(f"Attempting auto-resolution for alert: {alert.alert_id}")
        
        alert.auto_resolution_attempted = True
        resolution_success = False
        resolution_notes = ""
        
        try:
            if alert.category == AlertCategory.QUANTUM:
                resolution_success = await self._auto_resolve_quantum_alert(alert)
            elif alert.category == AlertCategory.PERFORMANCE:
                resolution_success = await self._auto_resolve_performance_alert(alert)
            elif alert.category == AlertCategory.SYSTEM:
                resolution_success = await self._auto_resolve_system_alert(alert)
            else:
                resolution_notes = "No auto-resolution strategy available"
                
        except Exception as e:
            self.logger.error(f"Auto-resolution failed for {alert.alert_id}: {e}")
            resolution_notes = f"Auto-resolution error: {str(e)}"
            
        if resolution_success:
            await self.resolve_alert(alert.alert_id, f"Auto-resolved: {resolution_notes}")
            
        return {
            "resolved": resolution_success,
            "notes": resolution_notes
        }
        
    async def _auto_resolve_quantum_alert(self, alert: Alert) -> bool:
        """Auto-resolve quantum-specific alerts"""
        # Quantum circuit optimization
        if "circuit" in alert.title.lower():
            # Simulate circuit optimization
            await asyncio.sleep(0.1)
            return True
            
        # Coherence recovery
        if "coherence" in alert.title.lower():
            # Simulate coherence recovery procedure
            await asyncio.sleep(0.2)
            return True
            
        # Error correction
        if "error" in alert.title.lower():
            # Simulate error correction
            await asyncio.sleep(0.1)
            return True
            
        return False
        
    async def _auto_resolve_performance_alert(self, alert: Alert) -> bool:
        """Auto-resolve performance alerts"""
        # Resource scaling
        if "resource" in alert.title.lower() or "cpu" in alert.title.lower():
            # Simulate resource scaling
            await asyncio.sleep(0.1)
            return True
            
        # Memory optimization
        if "memory" in alert.title.lower():
            # Simulate memory cleanup
            await asyncio.sleep(0.1)
            return True
            
        return False
        
    async def _auto_resolve_system_alert(self, alert: Alert) -> bool:
        """Auto-resolve system alerts"""
        # Service restart
        if "service" in alert.title.lower() or "connection" in alert.title.lower():
            # Simulate service restart
            await asyncio.sleep(0.2)
            return True
            
        return False
        
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                self.logger.error(f"Notification failed: {e}")
                
    async def _schedule_escalation(self, alert: Alert):
        """Schedule alert escalation if not resolved"""
        await asyncio.sleep(self.escalation_timeout.total_seconds())
        
        # Check if alert is still active
        if alert.alert_id in self.active_alerts and not alert.resolved:
            await self._escalate_alert(alert)
            
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert severity and notifications"""
        rules = self.alert_rules.get(alert.category, {})
        escalation_levels = rules.get("escalation_levels", [])
        
        # Find next escalation level
        current_index = -1
        for i, level in enumerate(escalation_levels):
            if alert.severity == level:
                current_index = i
                break
                
        if current_index >= 0 and current_index < len(escalation_levels) - 1:
            # Escalate to next level
            new_severity = escalation_levels[current_index + 1]
            alert.severity = new_severity
            
            self.logger.warning(f"Alert escalated: {alert.alert_id} -> {new_severity.name}")
            
            # Send escalation notifications
            await self._send_notifications(alert)
            
    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Manually resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Move to resolved alerts
            self.resolved_alerts.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id}")
            
    def add_notification_channel(self, channel: Callable):
        """Add notification channel"""
        self.notification_channels.append(channel)
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert management statistics"""
        total_alerts = len(self.active_alerts) + len(self.resolved_alerts)
        
        # Calculate resolution rate
        resolved_count = len(self.resolved_alerts)
        resolution_rate = resolved_count / total_alerts if total_alerts > 0 else 0.0
        
        # Calculate average resolution time
        resolution_times = [
            (alert.resolution_time - alert.detection_time).total_seconds()
            for alert in self.resolved_alerts
            if alert.resolution_time
        ]
        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0.0
        
        # Auto-resolution rate
        auto_resolved = sum(1 for alert in self.resolved_alerts if alert.auto_resolution_attempted and alert.resolved)
        auto_resolution_rate = auto_resolved / resolved_count if resolved_count > 0 else 0.0
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": resolved_count,
            "resolution_rate": resolution_rate,
            "avg_resolution_time_seconds": avg_resolution_time,
            "auto_resolution_rate": auto_resolution_rate,
            "alerts_by_severity": self._count_alerts_by_severity(),
            "alerts_by_category": self._count_alerts_by_category()
        }
        
    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count alerts by severity"""
        counts = defaultdict(int)
        for alert in list(self.active_alerts.values()) + self.resolved_alerts:
            counts[alert.severity.name] += 1
        return dict(counts)
        
    def _count_alerts_by_category(self) -> Dict[str, int]:
        """Count alerts by category"""
        counts = defaultdict(int)
        for alert in list(self.active_alerts.values()) + self.resolved_alerts:
            counts[alert.category.name] += 1
        return dict(counts)


class AdvancedMonitoringSystem:
    """Main advanced monitoring and alerting orchestration system"""
    
    def __init__(self, 
                 config: Optional[FederatedConfig] = None,
                 quantum_metrics_collector: Optional[QuantumMetricsCollector] = None,
                 security_fortress: Optional[SecurityFortress] = None,
                 error_recovery_system: Optional[QuantumErrorRecoverySystem] = None):
        
        self.config = config or FederatedConfig()
        self.quantum_metrics = quantum_metrics_collector
        self.security_fortress = security_fortress
        self.error_recovery = error_recovery_system
        
        # Initialize components
        self.anomaly_detector = QuantumAwareAnomalyDetector()
        self.degradation_detector = PerformanceDegradationDetector()
        self.alert_manager = IntelligentAlertManager()
        self.health_checker = QuantumHealthCheck(quantum_metrics_collector) if quantum_metrics_collector else None
        
        # System state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Setup notification channels
        self._setup_notification_channels()
        
    def _setup_notification_channels(self):
        """Setup notification channels"""
        # Console notification
        async def console_notification(alert: Alert):
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.EMERGENCY: "🆘"
            }
            
            emoji = severity_emoji.get(alert.severity, "🔔")
            self.logger.warning(f"{emoji} ALERT: {alert.title} - {alert.description}")
            
        self.alert_manager.add_notification_channel(console_notification)
        
    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
            
        self.monitoring_active = True
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
        
        self.logger.info("Advanced monitoring system started")
        
    async def stop_monitoring(self):
        """Stop monitoring"""
        if not self.monitoring_active:
            return
            
        self.stop_event.set()
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Advanced monitoring system stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.wait(1.0):  # Check every second
            try:
                asyncio.run(self._perform_monitoring_cycle())
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                
    async def _perform_monitoring_cycle(self):
        """Perform one monitoring cycle"""
        
        # 1. Collect quantum metrics and check for anomalies
        await self._check_quantum_anomalies()
        
        # 2. Check performance degradation
        await self._check_performance_degradation()
        
        # 3. Run health checks
        await self._run_health_checks()
        
        # 4. Check security status
        await self._check_security_status()
        
    async def _check_quantum_anomalies(self):
        """Check for quantum metric anomalies"""
        if not self.quantum_metrics:
            return
            
        # Get recent quantum metrics
        quantum_state = self.quantum_metrics.get_quantum_state_summary(timedelta(minutes=1))
        
        if "error" in quantum_state:
            return
            
        # Check each metric type for anomalies
        for metric_type, stats in quantum_state.items():
            if isinstance(stats, dict) and "mean" in stats:
                component_id = "quantum_system"
                is_anomaly, anomaly_score = self.anomaly_detector.add_metric_sample(
                    component_id, metric_type, stats["mean"]
                )
                
                if is_anomaly:
                    alert = Alert(
                        alert_id=f"quantum_anomaly_{metric_type}_{int(time.time())}",
                        title=f"Quantum Anomaly Detected: {metric_type}",
                        description=f"Anomalous behavior detected in quantum metric {metric_type}",
                        severity=AlertSeverity.WARNING if anomaly_score < 5 else AlertSeverity.ERROR,
                        category=AlertCategory.QUANTUM,
                        scope=MonitoringScope.QUANTUM_COMPONENT,
                        source_component=component_id,
                        detection_time=datetime.now(),
                        metric_values={"metric_type": metric_type, "anomaly_score": anomaly_score, "stats": stats},
                        recommended_actions=[
                            "Review quantum circuit parameters",
                            "Check for environmental interference",
                            "Verify quantum error correction"
                        ]
                    )
                    
                    await self.alert_manager.process_alert(alert)
                    
    async def _check_performance_degradation(self):
        """Check for performance degradation"""
        # Get performance metrics from various components
        components_to_check = ["quantum_optimizer", "federated_server", "privacy_engine"]
        
        for component_id in components_to_check:
            # Simulate performance metrics collection
            performance_metrics = self._collect_component_performance(component_id)
            
            if performance_metrics:
                alerts = self.degradation_detector.record_performance_sample(
                    component_id, performance_metrics
                )
                
                for alert in alerts:
                    await self.alert_manager.process_alert(alert)
                    
    def _collect_component_performance(self, component_id: str) -> Dict[str, float]:
        """Collect performance metrics for component"""
        # This would integrate with actual component metrics in production
        # For now, simulate some metrics
        base_metrics = {
            "response_time": 0.1,
            "throughput": 100.0,
            "error_rate": 0.01,
            "cpu_usage": 0.5,
            "memory_usage": 0.6
        }
        
        # Add some quantum-specific metrics for quantum components
        if "quantum" in component_id:
            base_metrics.update({
                "quantum_fidelity": 0.95,
                "coherence_time": 0.8,
                "gate_error_rate": 0.001
            })
            
        return base_metrics
        
    async def _run_health_checks(self):
        """Run comprehensive health checks"""
        if not self.health_checker:
            return
            
        try:
            health_status = await self.health_checker.run_comprehensive_health_check()
            
            # Check overall health status
            if health_status["overall_status"] == "unhealthy":
                alert = Alert(
                    alert_id=f"health_check_failure_{int(time.time())}",
                    title="System Health Check Failed",
                    description="Comprehensive health check indicates system issues",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.SYSTEM,
                    scope=MonitoringScope.GLOBAL,
                    source_component="health_checker",
                    detection_time=datetime.now(),
                    metric_values={"health_status": health_status},
                    recommended_actions=[
                        "Review system logs",
                        "Check component status",
                        "Verify resource availability",
                        "Consider system restart"
                    ]
                )
                
                await self.alert_manager.process_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
    async def _check_security_status(self):
        """Check security status"""
        if not self.security_fortress:
            return
            
        try:
            security_status = await self.security_fortress.get_security_status()
            
            # Check for security threats
            if security_status["active_threats"] > 0:
                alert = Alert(
                    alert_id=f"security_threats_{int(time.time())}",
                    title="Active Security Threats Detected",
                    description=f"{security_status['active_threats']} active security threats",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.SECURITY,
                    scope=MonitoringScope.GLOBAL,
                    source_component="security_fortress",
                    detection_time=datetime.now(),
                    metric_values={"security_status": security_status},
                    recommended_actions=[
                        "Review security logs",
                        "Investigate threat sources",
                        "Apply security mitigations",
                        "Block malicious IPs"
                    ]
                )
                
                await self.alert_manager.process_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Security check failed: {e}")
            
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "anomaly_detection": self.anomaly_detector.get_anomaly_report(),
            "alert_management": self.alert_manager.get_alert_statistics(),
            "active_alerts": len(self.alert_manager.active_alerts),
            "components_monitored": [
                "quantum_metrics" if self.quantum_metrics else None,
                "security_fortress" if self.security_fortress else None,
                "error_recovery" if self.error_recovery else None,
                "health_checker" if self.health_checker else None
            ],
            "last_check": datetime.now().isoformat()
        }


# Factory function
def create_advanced_monitoring_system(
    config: Optional[FederatedConfig] = None,
    quantum_metrics_collector: Optional[QuantumMetricsCollector] = None,
    security_fortress: Optional[SecurityFortress] = None,
    error_recovery_system: Optional[QuantumErrorRecoverySystem] = None
) -> AdvancedMonitoringSystem:
    """Create advanced monitoring system"""
    return AdvancedMonitoringSystem(
        config, quantum_metrics_collector, security_fortress, error_recovery_system
    )


# Example usage
async def main():
    """Example usage of advanced monitoring system"""
    # Create monitoring system
    monitoring_system = create_advanced_monitoring_system()
    
    # Start monitoring
    await monitoring_system.start_monitoring()
    
    # Simulate some operations
    await asyncio.sleep(5)
    
    # Get status
    status = await monitoring_system.get_monitoring_status()
    logging.info(f"Monitoring status: {json.dumps(status, indent=2, default=str)}")
    
    # Stop monitoring
    await monitoring_system.stop_monitoring()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
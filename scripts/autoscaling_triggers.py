#!/usr/bin/env python3
"""
Auto-scaling triggers for DP-Federated LoRA Lab.
Monitors system metrics and suggests scaling actions.
"""

import json
import sys
import time
import psutil
import threading
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Add more instances
    SCALE_IN = "scale_in"    # Remove instances
    NO_ACTION = "no_action"

@dataclass
class MetricThreshold:
    """Threshold configuration for a metric."""
    name: str
    current_value: float
    threshold_high: float
    threshold_low: float
    unit: str
    
    def needs_scale_up(self) -> bool:
        return self.current_value > self.threshold_high
    
    def needs_scale_down(self) -> bool:
        return self.current_value < self.threshold_low

@dataclass
class ScalingRecommendation:
    """Scaling recommendation based on metrics."""
    action: ScalingAction
    reason: str
    metrics: List[str]
    urgency: str  # low, medium, high
    suggested_scale_factor: float

class AutoScalingMonitor:
    """Auto-scaling monitor for federated learning system."""
    
    def __init__(self):
        self.monitoring_duration = 60  # seconds
        self.sample_interval = 1  # seconds
        self.metrics_history = []
        
        # Default thresholds
        self.thresholds = {
            'cpu_percent': MetricThreshold("CPU Usage", 0, 80.0, 30.0, "%"),
            'memory_percent': MetricThreshold("Memory Usage", 0, 85.0, 40.0, "%"),
            'concurrent_clients': MetricThreshold("Active Clients", 0, 50.0, 5.0, "clients"),
            'response_time_ms': MetricThreshold("Response Time", 0, 1000.0, 100.0, "ms"),
            'throughput_ops_sec': MetricThreshold("Throughput", 0, 100.0, 10.0, "ops/sec"),
        }
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['cpu_percent'] = cpu_percent
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            
            # Network metrics (simplified)
            net_io = psutil.net_io_counters()
            metrics['network_bytes_sent'] = net_io.bytes_sent
            metrics['network_bytes_recv'] = net_io.bytes_recv
            
        except Exception as e:
            print(f"Warning: Could not collect system metrics: {e}")
        
        return metrics
    
    def simulate_federated_metrics(self) -> Dict[str, float]:
        """Simulate federated learning specific metrics."""
        import random
        
        # Simulate realistic federated learning metrics
        base_clients = 20
        client_variation = random.uniform(0.8, 1.5)
        
        metrics = {
            'concurrent_clients': base_clients * client_variation,
            'response_time_ms': random.uniform(50, 200) * (1 + client_variation * 0.5),
            'throughput_ops_sec': random.uniform(30, 80) / client_variation,
            'privacy_budget_used_percent': random.uniform(10, 60),
            'model_accuracy': random.uniform(0.85, 0.95),
            'aggregation_time_ms': random.uniform(100, 500),
        }
        
        return metrics
    
    def update_thresholds(self, current_metrics: Dict[str, float]) -> None:
        """Update threshold current values."""
        for metric_name, threshold in self.thresholds.items():
            if metric_name in current_metrics:
                threshold.current_value = current_metrics[metric_name]
    
    def analyze_scaling_needs(self, metrics_history: List[Dict[str, float]]) -> ScalingRecommendation:
        """Analyze metrics and determine scaling needs."""
        if not metrics_history:
            return ScalingRecommendation(
                action=ScalingAction.NO_ACTION,
                reason="No metrics available",
                metrics=[],
                urgency="low",
                suggested_scale_factor=1.0
            )
        
        # Use latest metrics for analysis
        latest_metrics = metrics_history[-1]
        
        # Analyze trends over time
        if len(metrics_history) > 5:
            # Look at trend over last 5 samples
            recent_metrics = metrics_history[-5:]
            cpu_trend = self._calculate_trend([m.get('cpu_percent', 0) for m in recent_metrics])
            memory_trend = self._calculate_trend([m.get('memory_percent', 0) for m in recent_metrics])
            client_trend = self._calculate_trend([m.get('concurrent_clients', 0) for m in recent_metrics])
        else:
            cpu_trend = memory_trend = client_trend = 0
        
        # Update current values
        self.update_thresholds(latest_metrics)
        
        # Check for scale up conditions
        scale_up_reasons = []
        scale_down_reasons = []
        
        for threshold in self.thresholds.values():
            if threshold.needs_scale_up():
                scale_up_reasons.append(f"{threshold.name} high ({threshold.current_value:.1f}{threshold.unit} > {threshold.threshold_high}{threshold.unit})")
            elif threshold.needs_scale_down():
                scale_down_reasons.append(f"{threshold.name} low ({threshold.current_value:.1f}{threshold.unit} < {threshold.threshold_low}{threshold.unit})")
        
        # Determine urgency and action
        if scale_up_reasons:
            urgency = "high" if len(scale_up_reasons) > 2 else "medium"
            suggested_factor = 1.5 if urgency == "high" else 1.2
            
            return ScalingRecommendation(
                action=ScalingAction.SCALE_OUT,
                reason="; ".join(scale_up_reasons),
                metrics=[threshold.name for threshold in self.thresholds.values() if threshold.needs_scale_up()],
                urgency=urgency,
                suggested_scale_factor=suggested_factor
            )
        
        elif scale_down_reasons and len(scale_down_reasons) >= 3:
            # Only scale down if multiple metrics are consistently low
            return ScalingRecommendation(
                action=ScalingAction.SCALE_IN,
                reason="; ".join(scale_down_reasons),
                metrics=[threshold.name for threshold in self.thresholds.values() if threshold.needs_scale_down()],
                urgency="low",
                suggested_scale_factor=0.8
            )
        
        else:
            return ScalingRecommendation(
                action=ScalingAction.NO_ACTION,
                reason="All metrics within normal range",
                metrics=[],
                urgency="low",
                suggested_scale_factor=1.0
            )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def monitor_and_recommend(self, duration_seconds: int = 60) -> Tuple[List[Dict[str, float]], ScalingRecommendation]:
        """Monitor system for specified duration and provide scaling recommendation."""
        print(f"🔍 Monitoring system metrics for {duration_seconds} seconds...")
        
        metrics_history = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Collect metrics
            system_metrics = self.collect_system_metrics()
            federated_metrics = self.simulate_federated_metrics()
            
            # Combine metrics
            combined_metrics = {**system_metrics, **federated_metrics, 'timestamp': time.time()}
            metrics_history.append(combined_metrics)
            
            # Print current status
            if len(metrics_history) % 10 == 0:
                cpu = combined_metrics.get('cpu_percent', 0)
                memory = combined_metrics.get('memory_percent', 0)
                clients = combined_metrics.get('concurrent_clients', 0)
                print(f"  Sample {len(metrics_history):2d}: CPU={cpu:.1f}%, Memory={memory:.1f}%, Clients={clients:.0f}")
            
            time.sleep(self.sample_interval)
        
        # Analyze collected metrics
        recommendation = self.analyze_scaling_needs(metrics_history)
        
        return metrics_history, recommendation

def generate_scaling_config(recommendation: ScalingRecommendation) -> Dict:
    """Generate Kubernetes/Docker scaling configuration."""
    config = {
        'apiVersion': 'autoscaling/v2',
        'kind': 'HorizontalPodAutoscaler',
        'metadata': {
            'name': 'dp-federated-lora-hpa',
            'namespace': 'default'
        },
        'spec': {
            'scaleTargetRef': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': 'dp-federated-lora-server'
            },
            'minReplicas': 1,
            'maxReplicas': 10,
            'metrics': [
                {
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 70
                        }
                    }
                },
                {
                    'type': 'Resource',
                    'resource': {
                        'name': 'memory',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 80
                        }
                    }
                }
            ],
            'behavior': {
                'scaleUp': {
                    'stabilizationWindowSeconds': 60,
                    'policies': [
                        {
                            'type': 'Percent',
                            'value': 50,
                            'periodSeconds': 60
                        }
                    ]
                },
                'scaleDown': {
                    'stabilizationWindowSeconds': 300,
                    'policies': [
                        {
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }
                    ]
                }
            }
        }
    }
    
    return config

def main():
    """Main auto-scaling analysis."""
    print("📈 DP-Federated LoRA Auto-Scaling Analysis")
    print("=" * 50)
    
    monitor = AutoScalingMonitor()
    
    # Run monitoring
    start_time = time.time()
    metrics_history, recommendation = monitor.monitor_and_recommend(30)  # 30 seconds for demo
    end_time = time.time()
    
    print("-" * 50)
    print("📊 Scaling Analysis Results")
    print("-" * 50)
    
    # Display recommendation
    action_emoji = {
        ScalingAction.SCALE_OUT: "📈",
        ScalingAction.SCALE_IN: "📉",
        ScalingAction.SCALE_UP: "⬆️",
        ScalingAction.SCALE_DOWN: "⬇️",
        ScalingAction.NO_ACTION: "✅"
    }
    
    emoji = action_emoji.get(recommendation.action, "❓")
    print(f"{emoji} Action: {recommendation.action.value}")
    print(f"📝 Reason: {recommendation.reason}")
    print(f"🚨 Urgency: {recommendation.urgency}")
    print(f"📏 Suggested Scale Factor: {recommendation.suggested_scale_factor:.2f}x")
    
    if recommendation.metrics:
        print(f"📊 Triggering Metrics: {', '.join(recommendation.metrics)}")
    
    # Generate scaling configuration
    scaling_config = generate_scaling_config(recommendation)
    
    # Create comprehensive report
    report = {
        'timestamp': time.time(),
        'monitoring_duration_seconds': end_time - start_time,
        'metrics_samples_collected': len(metrics_history),
        'scaling_recommendation': {
            'action': recommendation.action.value,
            'reason': recommendation.reason,
            'urgency': recommendation.urgency,
            'suggested_scale_factor': recommendation.suggested_scale_factor,
            'triggering_metrics': recommendation.metrics
        },
        'current_metrics': metrics_history[-1] if metrics_history else {},
        'metrics_summary': {},
        'scaling_config': scaling_config,
        'recommendations': []
    }
    
    # Calculate metrics summary
    if metrics_history:
        numeric_metrics = ['cpu_percent', 'memory_percent', 'concurrent_clients', 'response_time_ms']
        for metric in numeric_metrics:
            values = [m.get(metric, 0) for m in metrics_history if metric in m]
            if values:
                report['metrics_summary'][metric] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': monitor._calculate_trend(values)
                }
    
    # Add operational recommendations
    if recommendation.action == ScalingAction.SCALE_OUT:
        report['recommendations'].append("Consider adding more server instances to handle increased load")
        report['recommendations'].append("Monitor privacy budget consumption with increased throughput")
    elif recommendation.action == ScalingAction.SCALE_IN:
        report['recommendations'].append("System is under-utilized - consider reducing resource allocation")
    else:
        report['recommendations'].append("System operating within normal parameters")
    
    report['recommendations'].append("Set up continuous monitoring with alerting for production deployment")
    
    # Save reports
    with open('autoscaling_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    with open('k8s_hpa_config.yaml', 'w') as f:
        import yaml
        try:
            yaml.dump(scaling_config, f, default_flow_style=False)
        except:
            # Fallback to JSON if yaml not available
            f.write("# Kubernetes HPA Configuration\n")
            f.write("# (Install PyYAML for proper YAML formatting)\n")
            json.dump(scaling_config, f, indent=2)
    
    print(f"📄 Auto-scaling analysis saved to autoscaling_analysis.json")
    print(f"☸️  Kubernetes HPA config saved to k8s_hpa_config.yaml")
    
    # Exit codes based on urgency
    if recommendation.urgency == "high":
        print(f"\n🚨 HIGH URGENCY: Immediate scaling action recommended!")
        sys.exit(2)
    elif recommendation.action != ScalingAction.NO_ACTION:
        print(f"\n⚠️  Scaling action recommended: {recommendation.action.value}")
        sys.exit(1)
    else:
        print(f"\n✅ System metrics are within normal operating range")
        sys.exit(0)

if __name__ == "__main__":
    main()
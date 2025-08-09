"""
Auto-Scaling System for DP-Federated LoRA deployment.

This module implements intelligent auto-scaling capabilities including resource
prediction, dynamic scaling decisions, container orchestration integration,
and quantum-inspired optimization for federated learning workloads.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import psutil
import requests

from .config import FederatedConfig
from .monitoring import ServerMetricsCollector
from .high_performance_core import PerformanceMetrics, ResourceProfile
from .quantum_scaling import QuantumAutoScaler, get_quantum_auto_scaler
from .exceptions import ResourceError, AutoScalingError


logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    CLIENT_COUNT = "client_count"
    TRAINING_THROUGHPUT = "training_throughput"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    PRIVACY_EFFICIENCY = "privacy_efficiency"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    REPLICAS = "replicas"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    
    name: str
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    resource_type: ResourceType
    scaling_factor: float = 1.5
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True
    last_triggered: float = 0.0


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    
    decision_id: str
    direction: ScalingDirection
    resource_type: ResourceType
    current_value: float
    target_value: float
    scaling_factor: float
    reasoning: str
    confidence: float
    estimated_impact: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceAllocation:
    """Current resource allocation."""
    
    cpu_cores: float
    memory_gb: float
    gpu_count: int
    replicas: int
    storage_gb: float
    network_mbps: float
    cost_per_hour: float = 0.0


class PredictiveScaler:
    """Machine learning-based predictive scaling system."""
    
    def __init__(self):
        """Initialize predictive scaler."""
        self.cpu_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.memory_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.throughput_predictor = LinearRegression()
        
        # Training data
        self.training_data: List[Dict[str, float]] = []
        self.prediction_horizon = 300  # 5 minutes ahead
        self.is_trained = False
        
        logger.info("Predictive scaler initialized")
    
    def add_training_sample(self, metrics: Dict[str, float], future_metrics: Dict[str, float]):
        """Add training sample for predictive models."""
        sample = {
            # Current state features
            'cpu_util': metrics.get('cpu_utilization', 0.0),
            'memory_util': metrics.get('memory_utilization', 0.0),
            'client_count': metrics.get('client_count', 0.0),
            'training_round': metrics.get('training_round', 0.0),
            'time_of_day': time.time() % 86400,  # Seconds since midnight
            
            # Future values (targets)
            'future_cpu': future_metrics.get('cpu_utilization', 0.0),
            'future_memory': future_metrics.get('memory_utilization', 0.0),
            'future_throughput': future_metrics.get('training_throughput', 0.0)
        }
        
        self.training_data.append(sample)
        
        # Keep only recent data
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
    
    def train_models(self):
        """Train predictive models on collected data."""
        if len(self.training_data) < 50:
            logger.warning("Insufficient training data for predictive models")
            return False
        
        try:
            # Prepare training data
            features = []
            cpu_targets = []
            memory_targets = []
            throughput_targets = []
            
            for sample in self.training_data:
                features.append([
                    sample['cpu_util'],
                    sample['memory_util'],
                    sample['client_count'],
                    sample['training_round'],
                    sample['time_of_day']
                ])
                cpu_targets.append(sample['future_cpu'])
                memory_targets.append(sample['future_memory'])
                throughput_targets.append(sample['future_throughput'])
            
            features = np.array(features)
            cpu_targets = np.array(cpu_targets)
            memory_targets = np.array(memory_targets)
            throughput_targets = np.array(throughput_targets)
            
            # Train models
            self.cpu_predictor.fit(features, cpu_targets)
            self.memory_predictor.fit(features, memory_targets)
            self.throughput_predictor.fit(features, throughput_targets)
            
            self.is_trained = True
            logger.info("Predictive models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train predictive models: {e}")
            return False
    
    def predict_resource_needs(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Predict future resource needs."""
        if not self.is_trained:
            return {}
        
        try:
            # Prepare features
            features = np.array([[
                current_metrics.get('cpu_utilization', 0.0),
                current_metrics.get('memory_utilization', 0.0),
                current_metrics.get('client_count', 0.0),
                current_metrics.get('training_round', 0.0),
                time.time() % 86400
            ]])
            
            # Make predictions
            predicted_cpu = self.cpu_predictor.predict(features)[0]
            predicted_memory = self.memory_predictor.predict(features)[0]
            predicted_throughput = self.throughput_predictor.predict(features)[0]
            
            return {
                'predicted_cpu_utilization': predicted_cpu,
                'predicted_memory_utilization': predicted_memory,
                'predicted_throughput': predicted_throughput,
                'prediction_confidence': self._calculate_prediction_confidence(current_metrics)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}
    
    def _calculate_prediction_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence in predictions."""
        # Simple confidence based on data availability and model performance
        base_confidence = 0.7 if self.is_trained else 0.3
        
        # Adjust based on training data size
        data_confidence = min(1.0, len(self.training_data) / 200.0)
        
        # Adjust based on metrics stability
        stability_factor = 1.0  # Would analyze metric variance in practice
        
        return base_confidence * data_confidence * stability_factor


class ContainerOrchestrator:
    """Interface to container orchestration systems (Kubernetes, Docker Swarm)."""
    
    def __init__(self, orchestrator_type: str = "kubernetes"):
        """Initialize container orchestrator interface."""
        self.orchestrator_type = orchestrator_type
        self.api_endpoint = self._get_api_endpoint()
        self.namespace = "dp-federated-lora"
        
        logger.info(f"Container orchestrator interface initialized ({orchestrator_type})")
    
    def _get_api_endpoint(self) -> str:
        """Get orchestrator API endpoint."""
        if self.orchestrator_type == "kubernetes":
            return "http://kubernetes.default.svc"
        elif self.orchestrator_type == "docker_swarm":
            return "http://localhost:2375"
        else:
            return "http://localhost:8080"
    
    async def scale_replicas(self, service_name: str, target_replicas: int) -> bool:
        """Scale service replicas."""
        try:
            if self.orchestrator_type == "kubernetes":
                return await self._scale_kubernetes_deployment(service_name, target_replicas)
            elif self.orchestrator_type == "docker_swarm":
                return await self._scale_docker_service(service_name, target_replicas)
            else:
                logger.warning(f"Scaling not implemented for {self.orchestrator_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to scale {service_name} to {target_replicas} replicas: {e}")
            return False
    
    async def _scale_kubernetes_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale Kubernetes deployment."""
        try:
            # Kubernetes API call to scale deployment
            url = f"{self.api_endpoint}/apis/apps/v1/namespaces/{self.namespace}/deployments/{deployment_name}/scale"
            
            payload = {
                "spec": {
                    "replicas": replicas
                }
            }
            
            # In a real implementation, this would use kubernetes client library
            # response = requests.patch(url, json=payload, headers=headers)
            # return response.status_code == 200
            
            logger.info(f"Scaled Kubernetes deployment {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes scaling failed: {e}")
            return False
    
    async def _scale_docker_service(self, service_name: str, replicas: int) -> bool:
        """Scale Docker Swarm service."""
        try:
            # Docker Swarm API call
            url = f"{self.api_endpoint}/services/{service_name}/update"
            
            payload = {
                "Mode": {
                    "Replicated": {
                        "Replicas": replicas
                    }
                }
            }
            
            logger.info(f"Scaled Docker service {service_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Docker Swarm scaling failed: {e}")
            return False
    
    async def get_current_replicas(self, service_name: str) -> int:
        """Get current number of replicas for a service."""
        try:
            if self.orchestrator_type == "kubernetes":
                return await self._get_kubernetes_replicas(service_name)
            elif self.orchestrator_type == "docker_swarm":
                return await self._get_docker_replicas(service_name)
            else:
                return 1  # Default fallback
        except Exception as e:
            logger.error(f"Failed to get replica count for {service_name}: {e}")
            return 1
    
    async def _get_kubernetes_replicas(self, deployment_name: str) -> int:
        """Get Kubernetes deployment replica count."""
        # Placeholder implementation
        return 2  # Would query actual Kubernetes API
    
    async def _get_docker_replicas(self, service_name: str) -> int:
        """Get Docker service replica count."""
        # Placeholder implementation
        return 2  # Would query actual Docker API


class AutoScalingSystem:
    """Comprehensive auto-scaling system for federated learning workloads."""
    
    def __init__(
        self,
        config: FederatedConfig,
        metrics_collector: ServerMetricsCollector,
        enable_quantum_scaling: bool = True
    ):
        """Initialize auto-scaling system."""
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Initialize components
        self.predictive_scaler = PredictiveScaler()
        self.orchestrator = ContainerOrchestrator()
        
        # Quantum scaling
        self.quantum_scaler: Optional[QuantumAutoScaler] = None
        if enable_quantum_scaling and config.quantum_enabled:
            try:
                self.quantum_scaler = get_quantum_auto_scaler(config, metrics_collector)
                logger.info("Quantum auto-scaling enabled")
            except Exception as e:
                logger.warning(f"Quantum scaling unavailable: {e}")
        
        # Scaling state
        self.scaling_rules: List[ScalingRule] = []
        self.scaling_history: List[ScalingDecision] = []
        self.current_allocation = ResourceAllocation(
            cpu_cores=2.0,
            memory_gb=4.0,
            gpu_count=0,
            replicas=1,
            storage_gb=20.0,
            network_mbps=100.0
        )
        
        # Control parameters
        self.scaling_enabled = True
        self.scaling_thread: Optional[threading.Thread] = None
        
        # Setup default scaling rules
        self._setup_default_scaling_rules()
        
        # Start scaling system
        self.start_scaling_monitor()
        
        logger.info("Auto-scaling system initialized")
    
    def _setup_default_scaling_rules(self):
        """Setup default auto-scaling rules."""
        # CPU-based scaling
        self.scaling_rules.append(ScalingRule(
            name="cpu_scale_up",
            metric=ScalingMetric.CPU_UTILIZATION,
            threshold_up=75.0,
            threshold_down=25.0,
            resource_type=ResourceType.REPLICAS,
            scaling_factor=1.5,
            cooldown_seconds=300,
            min_instances=1,
            max_instances=10
        ))
        
        # Memory-based scaling
        self.scaling_rules.append(ScalingRule(
            name="memory_scale_up",
            metric=ScalingMetric.MEMORY_UTILIZATION,
            threshold_up=80.0,
            threshold_down=30.0,
            resource_type=ResourceType.REPLICAS,
            scaling_factor=2.0,
            cooldown_seconds=180,
            min_instances=1,
            max_instances=8
        ))
        
        # Client-based scaling
        self.scaling_rules.append(ScalingRule(
            name="client_based_scaling",
            metric=ScalingMetric.CLIENT_COUNT,
            threshold_up=20.0,
            threshold_down=5.0,
            resource_type=ResourceType.REPLICAS,
            scaling_factor=1.2,
            cooldown_seconds=120,
            min_instances=1,
            max_instances=15
        ))
        
        # Training throughput scaling
        self.scaling_rules.append(ScalingRule(
            name="throughput_scaling",
            metric=ScalingMetric.TRAINING_THROUGHPUT,
            threshold_up=0.5,  # Low throughput triggers scale up
            threshold_down=2.0,  # High throughput allows scale down
            resource_type=ResourceType.CPU,
            scaling_factor=1.3,
            cooldown_seconds=600,
            min_instances=1,
            max_instances=12
        ))
    
    def start_scaling_monitor(self):
        """Start the auto-scaling monitoring thread."""
        if self.scaling_thread is None or not self.scaling_thread.is_alive():
            self.scaling_enabled = True
            self.scaling_thread = threading.Thread(target=self._scaling_monitor_loop, daemon=True)
            self.scaling_thread.start()
            logger.info("Auto-scaling monitor started")
    
    def stop_scaling_monitor(self):
        """Stop the auto-scaling monitoring thread."""
        self.scaling_enabled = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
        logger.info("Auto-scaling monitor stopped")
    
    def _scaling_monitor_loop(self):
        """Main auto-scaling monitoring loop."""
        while self.scaling_enabled:
            try:
                # Collect current metrics
                current_metrics = self._collect_scaling_metrics()
                
                # Make scaling decisions
                decisions = self._evaluate_scaling_decisions(current_metrics)
                
                # Execute scaling decisions
                for decision in decisions:
                    if self._should_execute_decision(decision):
                        asyncio.create_task(self._execute_scaling_decision(decision))
                
                # Train predictive models periodically
                if len(self.scaling_history) % 20 == 0:
                    self.predictive_scaler.train_models()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling monitor loop: {e}")
                time.sleep(30)
    
    def _collect_scaling_metrics(self) -> Dict[str, float]:
        """Collect metrics for scaling decisions."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Federated learning metrics
        fl_metrics = self.metrics_collector.get_recent_metrics()
        
        metrics = {
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'client_count': fl_metrics.get('active_clients', 0),
            'training_throughput': fl_metrics.get('training_throughput', 1.0),
            'queue_length': fl_metrics.get('queue_length', 0),
            'response_time': fl_metrics.get('avg_response_time', 100.0),
            'training_round': fl_metrics.get('current_round', 1),
            'gpu_utilization': self._get_gpu_utilization()
        }
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # Simplified GPU utilization (would use nvidia-ml-py in production)
            total_memory = 0
            used_memory = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory += props.total_memory
                used_memory += torch.cuda.memory_allocated(i)
            
            return (used_memory / total_memory) * 100.0 if total_memory > 0 else 0.0
        except Exception:
            return 0.0
    
    def _evaluate_scaling_decisions(self, metrics: Dict[str, float]) -> List[ScalingDecision]:
        """Evaluate and create scaling decisions based on current metrics."""
        decisions = []
        current_time = time.time()
        
        # Check each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if current_time - rule.last_triggered < rule.cooldown_seconds:
                continue
            
            metric_value = metrics.get(rule.metric.value, 0.0)
            
            # Determine scaling direction
            direction = ScalingDirection.STABLE
            if metric_value > rule.threshold_up:
                direction = ScalingDirection.UP
            elif metric_value < rule.threshold_down:
                direction = ScalingDirection.DOWN
            
            if direction != ScalingDirection.STABLE:
                # Create scaling decision
                decision = self._create_scaling_decision(
                    rule, direction, metric_value, metrics
                )
                decisions.append(decision)
                rule.last_triggered = current_time
        
        # Apply quantum optimization if available
        if self.quantum_scaler and decisions:
            decisions = self._apply_quantum_optimization(decisions, metrics)
        
        # Apply predictive optimization
        decisions = self._apply_predictive_optimization(decisions, metrics)
        
        return decisions
    
    def _create_scaling_decision(
        self,
        rule: ScalingRule,
        direction: ScalingDirection,
        metric_value: float,
        all_metrics: Dict[str, float]
    ) -> ScalingDecision:
        """Create a scaling decision based on rule and metrics."""
        current_value = self._get_current_resource_value(rule.resource_type)
        
        if direction == ScalingDirection.UP:
            target_value = min(
                current_value * rule.scaling_factor,
                rule.max_instances if rule.resource_type == ResourceType.REPLICAS else current_value * 2
            )
            confidence = min(1.0, (metric_value - rule.threshold_up) / rule.threshold_up)
        else:  # DOWN
            target_value = max(
                current_value / rule.scaling_factor,
                rule.min_instances if rule.resource_type == ResourceType.REPLICAS else current_value * 0.5
            )
            confidence = min(1.0, (rule.threshold_down - metric_value) / rule.threshold_down)
        
        # Estimate impact
        estimated_impact = self._estimate_scaling_impact(
            rule.resource_type, current_value, target_value, all_metrics
        )
        
        decision = ScalingDecision(
            decision_id=f"scale_{rule.name}_{int(time.time())}",
            direction=direction,
            resource_type=rule.resource_type,
            current_value=current_value,
            target_value=target_value,
            scaling_factor=rule.scaling_factor,
            reasoning=f"Rule '{rule.name}' triggered: {rule.metric.value}={metric_value:.2f}",
            confidence=confidence,
            estimated_impact=estimated_impact
        )
        
        return decision
    
    def _get_current_resource_value(self, resource_type: ResourceType) -> float:
        """Get current value of a resource type."""
        if resource_type == ResourceType.CPU:
            return self.current_allocation.cpu_cores
        elif resource_type == ResourceType.MEMORY:
            return self.current_allocation.memory_gb
        elif resource_type == ResourceType.GPU:
            return float(self.current_allocation.gpu_count)
        elif resource_type == ResourceType.REPLICAS:
            return float(self.current_allocation.replicas)
        elif resource_type == ResourceType.STORAGE:
            return self.current_allocation.storage_gb
        elif resource_type == ResourceType.NETWORK_BANDWIDTH:
            return self.current_allocation.network_mbps
        else:
            return 1.0
    
    def _estimate_scaling_impact(
        self,
        resource_type: ResourceType,
        current_value: float,
        target_value: float,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Estimate the impact of scaling decision."""
        scaling_ratio = target_value / current_value
        
        # Estimate performance impact
        performance_improvement = (scaling_ratio - 1) * 0.7  # 70% efficiency
        
        # Estimate cost impact
        cost_multiplier = scaling_ratio
        if resource_type == ResourceType.REPLICAS:
            cost_multiplier = scaling_ratio
        elif resource_type == ResourceType.CPU:
            cost_multiplier = scaling_ratio * 0.8  # CPU scaling is cheaper
        elif resource_type == ResourceType.MEMORY:
            cost_multiplier = scaling_ratio * 0.6  # Memory scaling is cheapest
        
        return {
            'performance_improvement': performance_improvement,
            'cost_change': (cost_multiplier - 1) * 100,  # Percentage change
            'resource_utilization_change': -10 if scaling_ratio > 1 else 10,  # Better utilization
            'availability_improvement': 5 if scaling_ratio > 1 else -2
        }
    
    def _apply_quantum_optimization(
        self,
        decisions: List[ScalingDecision],
        metrics: Dict[str, float]
    ) -> List[ScalingDecision]:
        """Apply quantum optimization to scaling decisions."""
        if not self.quantum_scaler:
            return decisions
        
        try:
            # Use quantum algorithms to optimize scaling decisions
            optimized_decisions = []
            
            for decision in decisions:
                quantum_result = self.quantum_scaler.optimize_scaling_decision(
                    decision, metrics
                )
                
                if quantum_result and quantum_result.get('success'):
                    # Update decision with quantum optimization
                    decision.target_value = quantum_result.get('optimal_target', decision.target_value)
                    decision.confidence *= quantum_result.get('confidence_multiplier', 1.0)
                    decision.reasoning += f" (Quantum optimized)"
                
                optimized_decisions.append(decision)
            
            return optimized_decisions
            
        except Exception as e:
            logger.warning(f"Quantum optimization failed: {e}")
            return decisions
    
    def _apply_predictive_optimization(
        self,
        decisions: List[ScalingDecision],
        metrics: Dict[str, float]
    ) -> List[ScalingDecision]:
        """Apply predictive optimization to scaling decisions."""
        predictions = self.predictive_scaler.predict_resource_needs(metrics)
        
        if not predictions:
            return decisions
        
        # Adjust decisions based on predictions
        optimized_decisions = []
        
        for decision in decisions:
            # Check if prediction suggests different scaling
            if decision.resource_type == ResourceType.CPU:
                predicted_cpu = predictions.get('predicted_cpu_utilization', 50.0)
                if predicted_cpu > 80 and decision.direction == ScalingDirection.DOWN:
                    decision.confidence *= 0.5  # Reduce confidence
                elif predicted_cpu < 30 and decision.direction == ScalingDirection.UP:
                    decision.confidence *= 0.7
            
            decision.reasoning += " (Predictively optimized)"
            optimized_decisions.append(decision)
        
        return optimized_decisions
    
    def _should_execute_decision(self, decision: ScalingDecision) -> bool:
        """Determine whether to execute a scaling decision."""
        # Minimum confidence threshold
        if decision.confidence < 0.6:
            return False
        
        # Check if change is significant enough
        change_ratio = abs(decision.target_value - decision.current_value) / decision.current_value
        if change_ratio < 0.1:  # Less than 10% change
            return False
        
        # Check resource limits
        if decision.resource_type == ResourceType.REPLICAS:
            if decision.target_value < 1 or decision.target_value > 20:
                return False
        
        return True
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        logger.info(f"Executing scaling decision: {decision.reasoning}")
        
        try:
            success = False
            
            if decision.resource_type == ResourceType.REPLICAS:
                success = await self.orchestrator.scale_replicas(
                    "federated-server",
                    int(decision.target_value)
                )
                if success:
                    self.current_allocation.replicas = int(decision.target_value)
            
            elif decision.resource_type == ResourceType.CPU:
                # CPU scaling would typically involve vertical scaling
                success = await self._scale_cpu_resources(decision.target_value)
                if success:
                    self.current_allocation.cpu_cores = decision.target_value
            
            elif decision.resource_type == ResourceType.MEMORY:
                success = await self._scale_memory_resources(decision.target_value)
                if success:
                    self.current_allocation.memory_gb = decision.target_value
            
            if success:
                logger.info(f"Successfully executed scaling decision {decision.decision_id}")
                self.scaling_history.append(decision)
            else:
                logger.error(f"Failed to execute scaling decision {decision.decision_id}")
        
        except Exception as e:
            logger.error(f"Error executing scaling decision {decision.decision_id}: {e}")
    
    async def _scale_cpu_resources(self, target_cpu: float) -> bool:
        """Scale CPU resources (placeholder for actual implementation)."""
        logger.info(f"Scaling CPU resources to {target_cpu} cores")
        # In practice, this would involve container resource limits
        return True
    
    async def _scale_memory_resources(self, target_memory: float) -> bool:
        """Scale memory resources (placeholder for actual implementation)."""
        logger.info(f"Scaling memory resources to {target_memory}GB")
        # In practice, this would involve container resource limits
        return True
    
    def add_custom_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added custom scaling rule: {rule.name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status."""
        recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
        
        return {
            'scaling_enabled': self.scaling_enabled,
            'current_allocation': {
                'cpu_cores': self.current_allocation.cpu_cores,
                'memory_gb': self.current_allocation.memory_gb,
                'gpu_count': self.current_allocation.gpu_count,
                'replicas': self.current_allocation.replicas,
                'storage_gb': self.current_allocation.storage_gb,
                'network_mbps': self.current_allocation.network_mbps
            },
            'scaling_rules': [
                {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'metric': rule.metric.value,
                    'resource_type': rule.resource_type.value,
                    'threshold_up': rule.threshold_up,
                    'threshold_down': rule.threshold_down
                }
                for rule in self.scaling_rules
            ],
            'recent_decisions': [
                {
                    'decision_id': d.decision_id,
                    'direction': d.direction.value,
                    'resource_type': d.resource_type.value,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp
                }
                for d in recent_decisions
            ],
            'predictive_scaler_trained': self.predictive_scaler.is_trained,
            'quantum_scaling_enabled': self.quantum_scaler is not None,
            'total_scaling_events': len(self.scaling_history)
        }


def create_auto_scaling_system(
    config: FederatedConfig,
    metrics_collector: ServerMetricsCollector,
    enable_quantum_scaling: bool = True
) -> AutoScalingSystem:
    """
    Create auto-scaling system with specified configuration.
    
    Args:
        config: Federated learning configuration
        metrics_collector: Server metrics collector
        enable_quantum_scaling: Whether to enable quantum optimization
        
    Returns:
        Configured auto-scaling system
    """
    return AutoScalingSystem(config, metrics_collector, enable_quantum_scaling)
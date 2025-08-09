"""
Intelligence Engine for DP-Federated LoRA system.

This module implements an AI-driven intelligence engine that makes autonomous
decisions about training optimization, client management, and system adaptation
using advanced machine learning and quantum-inspired algorithms.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json

from .config import FederatedConfig
from .monitoring import ServerMetricsCollector, UtilityMonitor
from .adaptive_optimization import AdaptiveOptimizer, OptimizationMetrics
from .quantum_scheduler import QuantumTaskScheduler
from .quantum_privacy import QuantumPrivacyEngine
from .exceptions import DPFederatedLoRAError


logger = logging.getLogger(__name__)


class IntelligenceLevel(Enum):
    """Intelligence operation levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    QUANTUM_ENHANCED = "quantum_enhanced"


class DecisionType(Enum):
    """Types of autonomous decisions."""
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    SCALING = "scaling"
    CLIENT_MANAGEMENT = "client_management"
    PRIVACY_ADAPTATION = "privacy_adaptation"
    EMERGENCY_RESPONSE = "emergency_response"


@dataclass
class IntelligenceDecision:
    """Decision made by the intelligence engine."""
    
    decision_id: str
    decision_type: DecisionType
    confidence: float
    parameters: Dict[str, Any]
    reasoning: str
    expected_impact: Dict[str, float]
    execution_priority: int = 1
    requires_human_approval: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemState:
    """Current system state for decision making."""
    
    training_metrics: OptimizationMetrics
    client_status: Dict[str, Any]
    resource_utilization: Dict[str, float]
    privacy_budget_status: Dict[str, float]
    security_alerts: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]


class MachineLearningPredictor:
    """Machine learning models for prediction and optimization."""
    
    def __init__(self):
        """Initialize ML predictor with ensemble models."""
        self.accuracy_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.privacy_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.communication_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_names = [
            'round_number', 'learning_rate', 'client_sampling_rate',
            'privacy_epsilon', 'lora_rank', 'batch_size',
            'num_clients', 'convergence_rate', 'privacy_spent'
        ]
    
    def add_training_data(self, features: Dict[str, float], targets: Dict[str, float]):
        """Add new training data point."""
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        
        self.training_data.append({
            'features': feature_vector,
            'accuracy': targets.get('accuracy', 0.0),
            'privacy_efficiency': targets.get('privacy_efficiency', 0.0),
            'communication_cost': targets.get('communication_cost', 0.0)
        })
    
    def train_models(self):
        """Train ML models on collected data."""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for ML models")
            return
        
        features = np.array([d['features'] for d in self.training_data])
        accuracy_targets = np.array([d['accuracy'] for d in self.training_data])
        privacy_targets = np.array([d['privacy_efficiency'] for d in self.training_data])
        comm_targets = np.array([d['communication_cost'] for d in self.training_data])
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models
        self.accuracy_predictor.fit(features_scaled, accuracy_targets)
        self.privacy_predictor.fit(features_scaled, privacy_targets)
        self.communication_predictor.fit(features_scaled, comm_targets)
        
        self.is_trained = True
        logger.info("ML models trained successfully")
    
    def predict_performance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict performance metrics for given configuration."""
        if not self.is_trained:
            return {"accuracy": 0.8, "privacy_efficiency": 0.7, "communication_cost": 50.0}
        
        feature_vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        predictions = {
            'accuracy': float(self.accuracy_predictor.predict(feature_vector_scaled)[0]),
            'privacy_efficiency': float(self.privacy_predictor.predict(feature_vector_scaled)[0]),
            'communication_cost': float(self.communication_predictor.predict(feature_vector_scaled)[0])
        }
        
        return predictions


class IntelligenceEngine:
    """
    AI-driven intelligence engine for autonomous federated learning management.
    
    Combines machine learning, quantum-inspired algorithms, and rule-based systems
    to make intelligent decisions about training optimization and system management.
    """
    
    def __init__(
        self,
        config: FederatedConfig,
        metrics_collector: ServerMetricsCollector,
        utility_monitor: UtilityMonitor,
        adaptive_optimizer: AdaptiveOptimizer,
        intelligence_level: IntelligenceLevel = IntelligenceLevel.QUANTUM_ENHANCED
    ):
        """
        Initialize intelligence engine.
        
        Args:
            config: Federated learning configuration
            metrics_collector: Server metrics collector
            utility_monitor: Utility monitor
            adaptive_optimizer: Adaptive optimizer
            intelligence_level: Level of intelligence to operate at
        """
        self.config = config
        self.metrics_collector = metrics_collector
        self.utility_monitor = utility_monitor
        self.adaptive_optimizer = adaptive_optimizer
        self.intelligence_level = intelligence_level
        
        # Decision making components
        self.ml_predictor = MachineLearningPredictor()
        self.decision_history: List[IntelligenceDecision] = []
        self.system_state_history: List[SystemState] = []
        
        # Quantum components
        self.quantum_scheduler: Optional[QuantumTaskScheduler] = None
        self.quantum_privacy: Optional[QuantumPrivacyEngine] = None
        
        if intelligence_level == IntelligenceLevel.QUANTUM_ENHANCED and config.quantum_enabled:
            try:
                from .quantum_scheduler import get_quantum_scheduler
                from .quantum_privacy import create_quantum_privacy_engine
                
                self.quantum_scheduler = get_quantum_scheduler(config, metrics_collector)
                self.quantum_privacy = create_quantum_privacy_engine(config.privacy)
                
                logger.info("Quantum-enhanced intelligence engine initialized")
            except ImportError:
                logger.warning("Quantum components not available, falling back to advanced mode")
                self.intelligence_level = IntelligenceLevel.ADVANCED
        
        # Intelligence parameters
        self.decision_threshold = 0.7
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.confidence_decay = 0.95
        
        # Performance tracking
        self.performance_baseline = None
        self.improvement_tracking = {}
        
        logger.info(f"Intelligence engine initialized at {intelligence_level.value} level")
    
    def analyze_system_state(self, round_num: int) -> SystemState:
        """
        Analyze current system state for decision making.
        
        Args:
            round_num: Current training round
            
        Returns:
            Current system state
        """
        # Get current metrics
        training_metrics = self.adaptive_optimizer.collect_metrics(
            round_num, 
            self.metrics_collector.get_round_summary(round_num)
        )
        
        # Analyze client status
        client_status = {
            "active_clients": self.metrics_collector.get_active_client_count(),
            "participation_rate": training_metrics.client_participation_rate,
            "avg_performance": self.metrics_collector.get_average_client_performance()
        }
        
        # Check resource utilization
        resource_utilization = {
            "cpu_usage": self.metrics_collector.get_cpu_usage(),
            "memory_usage": self.metrics_collector.get_memory_usage(),
            "network_bandwidth": self.metrics_collector.get_network_usage(),
            "gpu_utilization": self.metrics_collector.get_gpu_usage()
        }
        
        # Privacy budget status
        privacy_budget_status = {
            "epsilon_spent": training_metrics.privacy_spent,
            "epsilon_remaining": max(0, self.config.privacy.epsilon - training_metrics.privacy_spent),
            "budget_utilization": training_metrics.privacy_spent / self.config.privacy.epsilon
        }
        
        # Security alerts (placeholder for real security monitoring)
        security_alerts = self._analyze_security_threats(training_metrics)
        
        # Performance trends
        performance_trends = self._calculate_performance_trends()
        
        system_state = SystemState(
            training_metrics=training_metrics,
            client_status=client_status,
            resource_utilization=resource_utilization,
            privacy_budget_status=privacy_budget_status,
            security_alerts=security_alerts,
            performance_trends=performance_trends
        )
        
        self.system_state_history.append(system_state)
        return system_state
    
    def make_intelligent_decision(self, system_state: SystemState) -> List[IntelligenceDecision]:
        """
        Make intelligent decisions based on system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            List of decisions to implement
        """
        decisions = []
        
        # Optimization decisions
        opt_decision = self._make_optimization_decision(system_state)
        if opt_decision:
            decisions.append(opt_decision)
        
        # Security decisions
        security_decision = self._make_security_decision(system_state)
        if security_decision:
            decisions.append(security_decision)
        
        # Scaling decisions
        scaling_decision = self._make_scaling_decision(system_state)
        if scaling_decision:
            decisions.append(scaling_decision)
        
        # Client management decisions
        client_decision = self._make_client_management_decision(system_state)
        if client_decision:
            decisions.append(client_decision)
        
        # Privacy adaptation decisions
        privacy_decision = self._make_privacy_adaptation_decision(system_state)
        if privacy_decision:
            decisions.append(privacy_decision)
        
        # Emergency response decisions
        emergency_decision = self._make_emergency_decision(system_state)
        if emergency_decision:
            decisions.append(emergency_decision)
        
        # Sort by priority and confidence
        decisions.sort(key=lambda d: (d.execution_priority, -d.confidence))
        
        # Update decision history
        self.decision_history.extend(decisions)
        
        # Train ML models with new data
        self._update_ml_models(system_state)
        
        return decisions
    
    def _make_optimization_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make optimization-related decisions."""
        metrics = system_state.training_metrics
        
        # Use ML predictor if available
        if self.ml_predictor.is_trained:
            current_features = {
                'round_number': metrics.round_number,
                'learning_rate': self.config.learning_rate,
                'client_sampling_rate': self.config.security.client_sampling_rate,
                'privacy_epsilon': self.config.privacy.epsilon,
                'lora_rank': self.config.lora.r,
                'batch_size': self.config.batch_size,
                'num_clients': system_state.client_status['active_clients'],
                'convergence_rate': metrics.convergence_rate,
                'privacy_spent': metrics.privacy_spent
            }
            
            # Test different configurations
            best_config = None
            best_score = -np.inf
            
            for lr_mult in [0.5, 0.8, 1.0, 1.2, 1.5]:
                for sampling_mult in [0.7, 0.9, 1.0, 1.1, 1.3]:
                    test_features = current_features.copy()
                    test_features['learning_rate'] *= lr_mult
                    test_features['client_sampling_rate'] = min(1.0, test_features['client_sampling_rate'] * sampling_mult)
                    
                    predictions = self.ml_predictor.predict_performance(test_features)
                    
                    # Multi-objective score
                    score = (
                        0.4 * predictions['accuracy'] +
                        0.3 * predictions['privacy_efficiency'] -
                        0.3 * predictions['communication_cost'] / 100.0
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'learning_rate': test_features['learning_rate'],
                            'client_sampling_rate': test_features['client_sampling_rate'],
                            'expected_accuracy': predictions['accuracy'],
                            'expected_privacy_eff': predictions['privacy_efficiency']
                        }
            
            if best_config and best_score > 0.1:  # Minimum improvement threshold
                return IntelligenceDecision(
                    decision_id=f"opt_{int(time.time())}",
                    decision_type=DecisionType.OPTIMIZATION,
                    confidence=min(0.9, best_score),
                    parameters=best_config,
                    reasoning=f"ML predictor suggests configuration with expected score {best_score:.3f}",
                    expected_impact={
                        'accuracy_improvement': best_config['expected_accuracy'] - metrics.model_accuracy,
                        'privacy_efficiency_improvement': best_config['expected_privacy_eff'] - metrics.privacy_efficiency
                    },
                    execution_priority=2
                )
        
        # Fallback rule-based optimization
        if metrics.convergence_rate < 0.01 and metrics.round_number > 5:
            return IntelligenceDecision(
                decision_id=f"opt_fallback_{int(time.time())}",
                decision_type=DecisionType.OPTIMIZATION,
                confidence=0.6,
                parameters={'learning_rate': self.config.learning_rate * 1.5},
                reasoning="Slow convergence detected, increasing learning rate",
                expected_impact={'convergence_improvement': 0.02},
                execution_priority=3
            )
        
        return None
    
    def _make_security_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make security-related decisions."""
        if system_state.security_alerts:
            high_priority_alerts = [alert for alert in system_state.security_alerts if alert.get('severity') == 'high']
            
            if high_priority_alerts:
                return IntelligenceDecision(
                    decision_id=f"sec_{int(time.time())}",
                    decision_type=DecisionType.SECURITY,
                    confidence=0.9,
                    parameters={
                        'aggregation_method': 'krum',
                        'increase_security_level': True,
                        'reduce_client_trust': 0.8
                    },
                    reasoning=f"Detected {len(high_priority_alerts)} high-priority security alerts",
                    expected_impact={'security_improvement': 0.3},
                    execution_priority=1,
                    requires_human_approval=True
                )
        
        return None
    
    def _make_scaling_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make scaling-related decisions."""
        resource_util = system_state.resource_utilization
        
        # Scale up if resources are highly utilized
        if (resource_util.get('cpu_usage', 0) > 0.8 or 
            resource_util.get('memory_usage', 0) > 0.8):
            
            return IntelligenceDecision(
                decision_id=f"scale_up_{int(time.time())}",
                decision_type=DecisionType.SCALING,
                confidence=0.8,
                parameters={
                    'scale_factor': 1.5,
                    'resource_type': 'compute',
                    'priority': 'high'
                },
                reasoning="High resource utilization detected",
                expected_impact={'performance_improvement': 0.2},
                execution_priority=2
            )
        
        # Scale down if resources are underutilized
        elif (resource_util.get('cpu_usage', 0) < 0.3 and 
              resource_util.get('memory_usage', 0) < 0.3):
            
            return IntelligenceDecision(
                decision_id=f"scale_down_{int(time.time())}",
                decision_type=DecisionType.SCALING,
                confidence=0.7,
                parameters={
                    'scale_factor': 0.8,
                    'resource_type': 'compute',
                    'priority': 'low'
                },
                reasoning="Low resource utilization detected",
                expected_impact={'cost_reduction': 0.2},
                execution_priority=4
            )
        
        return None
    
    def _make_client_management_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make client management decisions."""
        participation_rate = system_state.client_status['participation_rate']
        
        if participation_rate < 0.5:
            return IntelligenceDecision(
                decision_id=f"client_mgmt_{int(time.time())}",
                decision_type=DecisionType.CLIENT_MANAGEMENT,
                confidence=0.7,
                parameters={
                    'increase_incentives': True,
                    'reduce_requirements': True,
                    'extend_deadline': True
                },
                reasoning=f"Low client participation rate: {participation_rate:.2f}",
                expected_impact={'participation_improvement': 0.2},
                execution_priority=3
            )
        
        return None
    
    def _make_privacy_adaptation_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make privacy adaptation decisions."""
        budget_status = system_state.privacy_budget_status
        
        # If privacy budget is running low, adjust strategy
        if budget_status['budget_utilization'] > 0.8:
            rounds_remaining = self.config.num_rounds - system_state.training_metrics.round_number
            
            if rounds_remaining > 5:  # Still many rounds left
                return IntelligenceDecision(
                    decision_id=f"privacy_{int(time.time())}",
                    decision_type=DecisionType.PRIVACY_ADAPTATION,
                    confidence=0.8,
                    parameters={
                        'reduce_noise_multiplier': True,
                        'increase_clipping_norm': True,
                        'adaptive_privacy_budget': True
                    },
                    reasoning="Privacy budget running low with many rounds remaining",
                    expected_impact={'privacy_efficiency_improvement': 0.15},
                    execution_priority=2
                )
        
        return None
    
    def _make_emergency_decision(self, system_state: SystemState) -> Optional[IntelligenceDecision]:
        """Make emergency response decisions."""
        # Check for critical failures
        if (system_state.training_metrics.model_accuracy < 0.3 or
            system_state.resource_utilization.get('memory_usage', 0) > 0.95):
            
            return IntelligenceDecision(
                decision_id=f"emergency_{int(time.time())}",
                decision_type=DecisionType.EMERGENCY_RESPONSE,
                confidence=0.95,
                parameters={
                    'immediate_action': 'pause_training',
                    'diagnostic_mode': True,
                    'alert_administrators': True
                },
                reasoning="Critical system state detected",
                expected_impact={'system_stability': 0.8},
                execution_priority=1,
                requires_human_approval=True
            )
        
        return None
    
    def _analyze_security_threats(self, metrics: OptimizationMetrics) -> List[Dict[str, Any]]:
        """Analyze potential security threats."""
        alerts = []
        
        # Check for unusual accuracy patterns (potential Byzantine attacks)
        if len(self.system_state_history) >= 3:
            recent_accuracies = [s.training_metrics.model_accuracy for s in self.system_state_history[-3:]]
            if max(recent_accuracies) - min(recent_accuracies) > 0.1:
                alerts.append({
                    'type': 'byzantine_attack_suspected',
                    'severity': 'high',
                    'description': 'Unusual accuracy variance detected',
                    'confidence': 0.7
                })
        
        # Check for communication anomalies
        if metrics.communication_cost > 200:  # Threshold for suspicious activity
            alerts.append({
                'type': 'communication_anomaly',
                'severity': 'medium',
                'description': 'Unusually high communication costs',
                'confidence': 0.6
            })
        
        return alerts
    
    def _calculate_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends over recent rounds."""
        if len(self.system_state_history) < 5:
            return {}
        
        recent_states = self.system_state_history[-10:]  # Last 10 rounds
        
        return {
            'accuracy_trend': [s.training_metrics.model_accuracy for s in recent_states],
            'privacy_efficiency_trend': [s.training_metrics.privacy_efficiency for s in recent_states],
            'convergence_trend': [s.training_metrics.convergence_rate for s in recent_states],
            'participation_trend': [s.client_status['participation_rate'] for s in recent_states]
        }
    
    def _update_ml_models(self, system_state: SystemState):
        """Update ML models with new training data."""
        metrics = system_state.training_metrics
        
        features = {
            'round_number': metrics.round_number,
            'learning_rate': self.config.learning_rate,
            'client_sampling_rate': self.config.security.client_sampling_rate,
            'privacy_epsilon': self.config.privacy.epsilon,
            'lora_rank': self.config.lora.r,
            'batch_size': self.config.batch_size,
            'num_clients': system_state.client_status['active_clients'],
            'convergence_rate': metrics.convergence_rate,
            'privacy_spent': metrics.privacy_spent
        }
        
        targets = {
            'accuracy': metrics.model_accuracy,
            'privacy_efficiency': metrics.privacy_efficiency,
            'communication_cost': metrics.communication_cost
        }
        
        self.ml_predictor.add_training_data(features, targets)
        
        # Retrain periodically
        if len(self.ml_predictor.training_data) % 20 == 0:
            self.ml_predictor.train_models()
    
    def execute_decision(self, decision: IntelligenceDecision) -> Dict[str, Any]:
        """
        Execute an intelligence decision.
        
        Args:
            decision: Decision to execute
            
        Returns:
            Execution results
        """
        logger.info(f"Executing decision {decision.decision_id}: {decision.reasoning}")
        
        results = {
            'decision_id': decision.decision_id,
            'executed': True,
            'timestamp': time.time(),
            'parameters_applied': decision.parameters
        }
        
        try:
            if decision.decision_type == DecisionType.OPTIMIZATION:
                self._execute_optimization_decision(decision)
            elif decision.decision_type == DecisionType.SECURITY:
                self._execute_security_decision(decision)
            elif decision.decision_type == DecisionType.SCALING:
                self._execute_scaling_decision(decision)
            elif decision.decision_type == DecisionType.CLIENT_MANAGEMENT:
                self._execute_client_management_decision(decision)
            elif decision.decision_type == DecisionType.PRIVACY_ADAPTATION:
                self._execute_privacy_decision(decision)
            elif decision.decision_type == DecisionType.EMERGENCY_RESPONSE:
                self._execute_emergency_decision(decision)
            
            results['success'] = True
            logger.info(f"Successfully executed decision {decision.decision_id}")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"Failed to execute decision {decision.decision_id}: {e}")
        
        return results
    
    def _execute_optimization_decision(self, decision: IntelligenceDecision):
        """Execute optimization decision."""
        params = decision.parameters
        
        if 'learning_rate' in params:
            self.config.learning_rate = params['learning_rate']
        
        if 'client_sampling_rate' in params:
            self.config.security.client_sampling_rate = params['client_sampling_rate']
    
    def _execute_security_decision(self, decision: IntelligenceDecision):
        """Execute security decision."""
        params = decision.parameters
        
        if 'aggregation_method' in params:
            from .config import AggregationMethod
            self.config.security.aggregation_method = AggregationMethod(params['aggregation_method'])
    
    def _execute_scaling_decision(self, decision: IntelligenceDecision):
        """Execute scaling decision."""
        logger.info(f"Scaling decision executed: {decision.parameters}")
        # In a real implementation, this would interact with container orchestration
    
    def _execute_client_management_decision(self, decision: IntelligenceDecision):
        """Execute client management decision."""
        logger.info(f"Client management decision executed: {decision.parameters}")
        # In a real implementation, this would adjust client incentives and requirements
    
    def _execute_privacy_decision(self, decision: IntelligenceDecision):
        """Execute privacy adaptation decision."""
        params = decision.parameters
        
        if 'reduce_noise_multiplier' in params:
            self.config.privacy.noise_multiplier *= 0.9
        
        if 'increase_clipping_norm' in params:
            self.config.privacy.max_grad_norm *= 1.1
    
    def _execute_emergency_decision(self, decision: IntelligenceDecision):
        """Execute emergency response decision."""
        logger.warning(f"EMERGENCY DECISION EXECUTED: {decision.reasoning}")
        # In a real implementation, this would trigger emergency protocols
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive intelligence report.
        
        Returns:
            Intelligence engine report
        """
        total_decisions = len(self.decision_history)
        successful_decisions = len([d for d in self.decision_history if hasattr(d, 'success') and d.success])
        
        decision_types_count = {}
        for decision in self.decision_history:
            decision_types_count[decision.decision_type.value] = decision_types_count.get(decision.decision_type.value, 0) + 1
        
        return {
            'intelligence_level': self.intelligence_level.value,
            'total_decisions': total_decisions,
            'success_rate': successful_decisions / max(total_decisions, 1),
            'decision_breakdown': decision_types_count,
            'ml_model_status': {
                'trained': self.ml_predictor.is_trained,
                'training_samples': len(self.ml_predictor.training_data)
            },
            'recent_decisions': [
                {
                    'id': d.decision_id,
                    'type': d.decision_type.value,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning
                }
                for d in self.decision_history[-5:]
            ],
            'performance_tracking': self.improvement_tracking,
            'quantum_components': {
                'scheduler_active': self.quantum_scheduler is not None,
                'privacy_engine_active': self.quantum_privacy is not None
            }
        }


def create_intelligence_engine(
    config: FederatedConfig,
    metrics_collector: ServerMetricsCollector,
    utility_monitor: UtilityMonitor,
    adaptive_optimizer: AdaptiveOptimizer,
    intelligence_level: IntelligenceLevel = IntelligenceLevel.QUANTUM_ENHANCED
) -> IntelligenceEngine:
    """
    Create intelligence engine with specified configuration.
    
    Args:
        config: Federated learning configuration
        metrics_collector: Server metrics collector
        utility_monitor: Utility monitor
        adaptive_optimizer: Adaptive optimizer
        intelligence_level: Intelligence level
        
    Returns:
        Configured intelligence engine
    """
    return IntelligenceEngine(
        config=config,
        metrics_collector=metrics_collector,
        utility_monitor=utility_monitor,
        adaptive_optimizer=adaptive_optimizer,
        intelligence_level=intelligence_level
    )
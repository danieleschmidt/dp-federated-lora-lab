"""
Adaptive optimization module for DP-Federated LoRA training.

This module implements adaptive learning strategies that adjust training parameters
based on real-time performance metrics and quantum-inspired optimization algorithms.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from .config import FederatedConfig, PrivacyConfig, LoRAConfig
from .monitoring import ServerMetricsCollector, UtilityMonitor
from .quantum_optimizer import QuantumInspiredOptimizer, get_quantum_optimizer


logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Supported optimization strategies."""
    ADAPTIVE_LEARNING_RATE = "adaptive_lr"
    DYNAMIC_PRIVACY_BUDGET = "dynamic_privacy"
    QUANTUM_ANNEALING = "quantum_annealing"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OptimizationMetrics:
    """Metrics used for optimization decisions."""
    
    round_number: int
    convergence_rate: float
    privacy_efficiency: float
    communication_cost: float
    client_participation_rate: float
    model_accuracy: float
    training_loss: float
    privacy_spent: float
    computation_time: float
    network_latency: float


@dataclass
class AdaptationDecision:
    """Decision made by the adaptive optimizer."""
    
    new_learning_rate: Optional[float] = None
    new_privacy_budget: Optional[Dict[str, float]] = None
    new_client_sampling_rate: Optional[float] = None
    new_aggregation_method: Optional[str] = None
    quantum_parameters: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    reasoning: str = ""


class AdaptiveOptimizer:
    """
    Adaptive optimizer that adjusts training parameters based on real-time metrics.
    
    Uses quantum-inspired algorithms and machine learning to optimize federated
    training performance while maintaining privacy guarantees.
    """
    
    def __init__(
        self,
        config: FederatedConfig,
        metrics_collector: ServerMetricsCollector,
        utility_monitor: UtilityMonitor
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            config: Federated learning configuration
            metrics_collector: Server metrics collector
            utility_monitor: Utility monitoring system
        """
        self.config = config
        self.metrics_collector = metrics_collector
        self.utility_monitor = utility_monitor
        
        # Optimization state
        self.optimization_history: List[OptimizationMetrics] = []
        self.adaptation_history: List[AdaptationDecision] = []
        self.performance_baseline: Optional[Dict[str, float]] = None
        
        # Quantum optimizer
        self.quantum_optimizer = get_quantum_optimizer(config, metrics_collector)
        
        # Adaptive parameters
        self.learning_rate_momentum = 0.9
        self.privacy_efficiency_target = 0.85
        self.convergence_patience = 5
        self.min_improvement_threshold = 0.01
        
        # Optimization strategies
        self.enabled_strategies = {
            OptimizationStrategy.ADAPTIVE_LEARNING_RATE: True,
            OptimizationStrategy.DYNAMIC_PRIVACY_BUDGET: True,
            OptimizationStrategy.QUANTUM_ANNEALING: config.quantum_enabled,
            OptimizationStrategy.MULTI_OBJECTIVE: True
        }
        
        logger.info("Adaptive optimizer initialized with quantum-inspired algorithms")
    
    def collect_metrics(self, round_num: int, training_state: Dict[str, Any]) -> OptimizationMetrics:
        """
        Collect optimization metrics for the current round.
        
        Args:
            round_num: Current training round
            training_state: Current training state
            
        Returns:
            Optimization metrics
        """
        # Get server metrics
        server_metrics = self.metrics_collector.get_round_summary(round_num)
        utility_metrics = self.utility_monitor.get_current_metrics()
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate(round_num)
        
        # Calculate privacy efficiency
        privacy_spent = training_state.get("privacy_spent", 0.0)
        model_accuracy = training_state.get("accuracy", 0.0)
        privacy_efficiency = model_accuracy / max(privacy_spent, 0.001)
        
        metrics = OptimizationMetrics(
            round_number=round_num,
            convergence_rate=convergence_rate,
            privacy_efficiency=privacy_efficiency,
            communication_cost=server_metrics.get("communication_cost", 0.0),
            client_participation_rate=server_metrics.get("client_participation_rate", 0.0),
            model_accuracy=model_accuracy,
            training_loss=training_state.get("loss", 0.0),
            privacy_spent=privacy_spent,
            computation_time=server_metrics.get("computation_time", 0.0),
            network_latency=server_metrics.get("network_latency", 0.0)
        )
        
        self.optimization_history.append(metrics)
        return metrics
    
    def optimize_training_parameters(
        self,
        current_metrics: OptimizationMetrics,
        training_state: Dict[str, Any]
    ) -> AdaptationDecision:
        """
        Optimize training parameters based on current metrics.
        
        Args:
            current_metrics: Current optimization metrics
            training_state: Current training state
            
        Returns:
            Adaptation decision with new parameters
        """
        logger.info(f"Optimizing parameters for round {current_metrics.round_number}")
        
        # Initialize decision
        decision = AdaptationDecision()
        adaptations = []
        
        # Apply optimization strategies
        if self.enabled_strategies[OptimizationStrategy.ADAPTIVE_LEARNING_RATE]:
            lr_decision = self._optimize_learning_rate(current_metrics)
            if lr_decision:
                decision.new_learning_rate = lr_decision
                adaptations.append("learning_rate")
        
        if self.enabled_strategies[OptimizationStrategy.DYNAMIC_PRIVACY_BUDGET]:
            privacy_decision = self._optimize_privacy_budget(current_metrics)
            if privacy_decision:
                decision.new_privacy_budget = privacy_decision
                adaptations.append("privacy_budget")
        
        if self.enabled_strategies[OptimizationStrategy.QUANTUM_ANNEALING]:
            quantum_decision = self._apply_quantum_optimization(current_metrics, training_state)
            if quantum_decision:
                decision.quantum_parameters = quantum_decision
                adaptations.append("quantum_parameters")
        
        if self.enabled_strategies[OptimizationStrategy.MULTI_OBJECTIVE]:
            multi_obj_decision = self._multi_objective_optimization(current_metrics)
            if multi_obj_decision.get("client_sampling_rate"):
                decision.new_client_sampling_rate = multi_obj_decision["client_sampling_rate"]
                adaptations.append("client_sampling")
            if multi_obj_decision.get("aggregation_method"):
                decision.new_aggregation_method = multi_obj_decision["aggregation_method"]
                adaptations.append("aggregation_method")
        
        # Calculate confidence score
        decision.confidence_score = self._calculate_confidence_score(current_metrics, adaptations)
        decision.reasoning = f"Applied {', '.join(adaptations)} based on performance analysis"
        
        self.adaptation_history.append(decision)
        
        logger.info(
            f"Optimization decision: {len(adaptations)} adaptations, "
            f"confidence: {decision.confidence_score:.3f}"
        )
        
        return decision
    
    def _optimize_learning_rate(self, metrics: OptimizationMetrics) -> Optional[float]:
        """Optimize learning rate based on convergence metrics."""
        if len(self.optimization_history) < 3:
            return None
        
        recent_losses = [m.training_loss for m in self.optimization_history[-3:]]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        current_lr = self.config.learning_rate
        
        # Decrease learning rate if loss is increasing
        if loss_trend > 0:
            new_lr = current_lr * 0.8
            logger.info(f"Decreasing learning rate: {current_lr:.6f} -> {new_lr:.6f}")
            return new_lr
        
        # Increase learning rate if convergence is slow
        if metrics.convergence_rate < 0.01 and loss_trend < -0.001:
            new_lr = min(current_lr * 1.2, 0.01)  # Cap at reasonable maximum
            logger.info(f"Increasing learning rate: {current_lr:.6f} -> {new_lr:.6f}")
            return new_lr
        
        return None
    
    def _optimize_privacy_budget(self, metrics: OptimizationMetrics) -> Optional[Dict[str, float]]:
        """Optimize privacy budget allocation based on efficiency metrics."""
        if metrics.privacy_efficiency < self.privacy_efficiency_target:
            # Allocate more budget to improve utility
            rounds_remaining = self.config.num_rounds - metrics.round_number
            if rounds_remaining > 0:
                remaining_epsilon = self.config.privacy.epsilon - metrics.privacy_spent
                new_epsilon_per_round = remaining_epsilon / rounds_remaining * 1.1
                
                logger.info(f"Increasing privacy budget allocation: {new_epsilon_per_round:.3f} ε per round")
                return {"epsilon": new_epsilon_per_round, "delta": self.config.privacy.delta}
        
        return None
    
    def _apply_quantum_optimization(
        self,
        metrics: OptimizationMetrics,
        training_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply quantum-inspired optimization algorithms."""
        if not self.quantum_optimizer:
            return None
        
        # Define optimization objective
        def objective_function(params):
            # Multi-objective: minimize loss, maximize privacy efficiency
            loss_weight = 0.6
            privacy_weight = 0.4
            
            predicted_loss = params[0]  # Simplified prediction
            predicted_privacy_eff = params[1]
            
            return loss_weight * predicted_loss - privacy_weight * predicted_privacy_eff
        
        # Use quantum annealing for parameter optimization
        try:
            quantum_result = self.quantum_optimizer.optimize_parameters(
                objective_function=objective_function,
                current_state=training_state,
                constraints={
                    "privacy_budget": self.config.privacy.epsilon,
                    "min_clients": self.config.security.min_clients
                }
            )
            
            if quantum_result and quantum_result.get("success"):
                logger.info("Applied quantum-inspired parameter optimization")
                return {
                    "quantum_state": quantum_result.get("quantum_state"),
                    "optimization_params": quantum_result.get("optimal_params"),
                    "expected_improvement": quantum_result.get("expected_improvement", 0.0)
                }
        except Exception as e:
            logger.warning(f"Quantum optimization failed: {e}")
        
        return None
    
    def _multi_objective_optimization(self, metrics: OptimizationMetrics) -> Dict[str, Any]:
        """Perform multi-objective optimization for federated parameters."""
        decisions = {}
        
        # Optimize client sampling rate based on participation and performance
        current_participation = metrics.client_participation_rate
        if current_participation < 0.7:
            # Increase sampling rate to get more participants
            new_sampling_rate = min(self.config.security.client_sampling_rate * 1.2, 1.0)
            decisions["client_sampling_rate"] = new_sampling_rate
        elif current_participation > 0.95 and metrics.communication_cost > 100:
            # Reduce sampling rate to decrease communication cost
            new_sampling_rate = max(self.config.security.client_sampling_rate * 0.9, 0.3)
            decisions["client_sampling_rate"] = new_sampling_rate
        
        # Optimize aggregation method based on Byzantine resilience needs
        if len(self.optimization_history) >= 5:
            recent_accuracy_variance = np.var([m.model_accuracy for m in self.optimization_history[-5:]])
            if recent_accuracy_variance > 0.01:  # High variance suggests Byzantine attacks
                decisions["aggregation_method"] = "krum"
            elif metrics.communication_cost < 50:  # Low cost, can use more robust methods
                decisions["aggregation_method"] = "trimmed_mean"
        
        return decisions
    
    def _calculate_convergence_rate(self, round_num: int) -> float:
        """Calculate convergence rate based on recent performance."""
        if len(self.optimization_history) < 2:
            return 0.0
        
        recent_metrics = self.optimization_history[-2:]
        accuracy_change = recent_metrics[-1].model_accuracy - recent_metrics[0].model_accuracy
        return max(0.0, accuracy_change)
    
    def _calculate_confidence_score(
        self,
        metrics: OptimizationMetrics,
        adaptations: List[str]
    ) -> float:
        """Calculate confidence score for optimization decisions."""
        base_confidence = 0.5
        
        # Increase confidence with more historical data
        if len(self.optimization_history) >= 10:
            base_confidence += 0.2
        elif len(self.optimization_history) >= 5:
            base_confidence += 0.1
        
        # Increase confidence if recent performance is consistent
        if len(self.optimization_history) >= 3:
            recent_accuracies = [m.model_accuracy for m in self.optimization_history[-3:]]
            if np.std(recent_accuracies) < 0.05:  # Low variance
                base_confidence += 0.1
        
        # Decrease confidence if making many adaptations
        adaptation_penalty = min(0.2, len(adaptations) * 0.05)
        base_confidence -= adaptation_penalty
        
        # Quantum optimization boost
        if "quantum_parameters" in adaptations:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def should_apply_adaptation(self, decision: AdaptationDecision) -> bool:
        """
        Determine whether to apply the optimization decision.
        
        Args:
            decision: Adaptation decision to evaluate
            
        Returns:
            True if decision should be applied
        """
        # Apply if confidence is high enough
        if decision.confidence_score >= 0.7:
            return True
        
        # Apply if performance is degrading significantly
        if len(self.optimization_history) >= 3:
            recent_accuracies = [m.model_accuracy for m in self.optimization_history[-3:]]
            if max(recent_accuracies) - min(recent_accuracies) > 0.05:  # Significant degradation
                return True
        
        # Conservative approach for low-confidence decisions
        return decision.confidence_score >= 0.8
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Returns:
            Optimization report with metrics and decisions
        """
        if not self.optimization_history:
            return {"status": "no_data"}
        
        latest_metrics = self.optimization_history[-1]
        total_adaptations = len(self.adaptation_history)
        
        # Calculate performance trends
        if len(self.optimization_history) >= 5:
            accuracy_trend = np.polyfit(
                range(len(self.optimization_history[-5:])),
                [m.model_accuracy for m in self.optimization_history[-5:]],
                1
            )[0]
            privacy_efficiency_trend = np.polyfit(
                range(len(self.optimization_history[-5:])),
                [m.privacy_efficiency for m in self.optimization_history[-5:]],
                1
            )[0]
        else:
            accuracy_trend = 0.0
            privacy_efficiency_trend = 0.0
        
        report = {
            "optimization_summary": {
                "total_rounds": len(self.optimization_history),
                "total_adaptations": total_adaptations,
                "latest_accuracy": latest_metrics.model_accuracy,
                "latest_privacy_efficiency": latest_metrics.privacy_efficiency,
                "accuracy_trend": accuracy_trend,
                "privacy_efficiency_trend": privacy_efficiency_trend
            },
            "performance_metrics": {
                "convergence_rate": latest_metrics.convergence_rate,
                "client_participation": latest_metrics.client_participation_rate,
                "communication_efficiency": 1.0 / max(latest_metrics.communication_cost, 1.0),
                "computation_efficiency": 1.0 / max(latest_metrics.computation_time, 1.0)
            },
            "adaptation_strategies": {
                strategy.value: enabled for strategy, enabled in self.enabled_strategies.items()
            },
            "recent_decisions": [
                {
                    "round": decision.new_learning_rate if hasattr(decision, 'round') else 'unknown',
                    "adaptations": [
                        k for k, v in {
                            "learning_rate": decision.new_learning_rate,
                            "privacy_budget": decision.new_privacy_budget,
                            "client_sampling": decision.new_client_sampling_rate,
                            "aggregation": decision.new_aggregation_method,
                            "quantum": decision.quantum_parameters
                        }.items() if v is not None
                    ],
                    "confidence": decision.confidence_score,
                    "reasoning": decision.reasoning
                }
                for decision in self.adaptation_history[-5:]
            ]
        }
        
        return report


def create_adaptive_optimizer(
    config: FederatedConfig,
    metrics_collector: ServerMetricsCollector,
    utility_monitor: UtilityMonitor
) -> AdaptiveOptimizer:
    """
    Create adaptive optimizer with default configuration.
    
    Args:
        config: Federated learning configuration
        metrics_collector: Server metrics collector
        utility_monitor: Utility monitor
        
    Returns:
        Configured adaptive optimizer
    """
    return AdaptiveOptimizer(config, metrics_collector, utility_monitor)
"""
Monitoring and metrics collection for federated learning.

This module provides comprehensive monitoring capabilities for privacy,
performance, and system metrics in the DP-Federated LoRA system.
"""

import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


@dataclass
class PrivacyMetrics:
    """Privacy-related metrics."""
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    epsilon_remaining: float = 0.0
    budget_utilization: float = 0.0
    noise_multiplier: float = 0.0
    clipping_threshold: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance and utility metrics."""
    training_loss: float = float('inf')
    validation_loss: float = float('inf')
    accuracy: float = 0.0
    perplexity: float = float('inf')
    convergence_rate: float = 0.0
    utility_privacy_ratio: float = 0.0


@dataclass
class SystemMetrics:
    """System and resource metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0


@dataclass
class FederatedMetrics:
    """Federated learning specific metrics."""
    num_clients_selected: int = 0
    num_clients_completed: int = 0
    client_participation_rate: float = 0.0
    aggregation_time: float = 0.0
    communication_overhead: float = 0.0
    byzantine_detected: int = 0


class MetricsCollector:
    """Base class for metrics collection."""
    
    def __init__(self, entity_id: str):
        """
        Initialize metrics collector.
        
        Args:
            entity_id: Identifier for the entity being monitored
        """
        self.entity_id = entity_id
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.last_update = time.time()
        
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics with timestamp.
        
        Args:
            metrics: Dictionary of metrics to record
        """
        timestamp = time.time()
        metrics_entry = {
            "timestamp": timestamp,
            "elapsed_time": timestamp - self.start_time,
            "entity_id": self.entity_id,
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        self.last_update = timestamp
        
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics entry."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_range(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metrics within a time range.
        
        Args:
            start_time: Start timestamp (None for beginning)
            end_time: End timestamp (None for current time)
            
        Returns:
            List of metrics entries in the range
        """
        filtered = []
        for entry in self.metrics_history:
            timestamp = entry["timestamp"]
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            filtered.append(entry)
        return filtered
    
    def compute_summary_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Compute summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        values = [
            entry.get(metric_name, 0.0)
            for entry in self.metrics_history
            if metric_name in entry
        ]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1]
        }


class LocalMetricsCollector(MetricsCollector):
    """Metrics collector for federated learning clients."""
    
    def __init__(self, client_id: str):
        """Initialize client metrics collector."""
        super().__init__(client_id)
        self.training_rounds: List[Dict[str, Any]] = []
        self.privacy_spent_history: List[PrivacyMetrics] = []
        
    def record_training_round(self, round_data: Dict[str, Any]) -> None:
        """
        Record metrics for a training round.
        
        Args:
            round_data: Training round metrics
        """
        round_entry = {
            "round": round_data.get("round", 0),
            "timestamp": time.time(),
            "client_id": self.entity_id,
            **round_data
        }
        
        self.training_rounds.append(round_entry)
        self.record_metrics(round_data)
        
    def record_privacy_metrics(self, privacy_data: Dict[str, float]) -> None:
        """
        Record privacy metrics.
        
        Args:
            privacy_data: Privacy-related metrics
        """
        privacy_metrics = PrivacyMetrics(
            epsilon_spent=privacy_data.get("epsilon", 0.0),
            delta_spent=privacy_data.get("delta", 0.0),
            epsilon_remaining=privacy_data.get("epsilon_remaining", 0.0),
            budget_utilization=privacy_data.get("budget_utilization", 0.0),
            noise_multiplier=privacy_data.get("noise_multiplier", 0.0),
            clipping_threshold=privacy_data.get("clipping_threshold", 0.0)
        )
        
        self.privacy_spent_history.append(privacy_metrics)
        self.record_metrics(privacy_data)
        
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress summary."""
        if not self.training_rounds:
            return {}
        
        latest_round = self.training_rounds[-1]
        total_rounds = len(self.training_rounds)
        
        # Compute loss trend
        losses = [r.get("loss", float('inf')) for r in self.training_rounds]
        loss_trend = "improving" if len(losses) > 1 and losses[-1] < losses[0] else "stable"
        
        return {
            "total_rounds": total_rounds,
            "latest_round": latest_round.get("round", 0),
            "latest_loss": latest_round.get("loss", float('inf')),
            "loss_trend": loss_trend,
            "privacy_epsilon": latest_round.get("privacy_epsilon", 0.0),
            "examples_processed": sum(r.get("examples_processed", 0) for r in self.training_rounds)
        }


class ServerMetricsCollector(MetricsCollector):
    """Metrics collector for the federated learning server."""
    
    def __init__(self):
        """Initialize server metrics collector."""
        super().__init__("federated_server")
        self.round_metrics: List[Dict[str, Any]] = []
        self.client_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.aggregation_history: List[Dict[str, Any]] = []
        
    def record_round_start(self, round_num: int, selected_clients: List[str]) -> None:
        """
        Record the start of a training round.
        
        Args:
            round_num: Round number
            selected_clients: List of selected client IDs
        """
        round_data = {
            "event": "round_start",
            "round": round_num,
            "selected_clients": selected_clients,
            "num_selected": len(selected_clients),
            "start_time": time.time()
        }
        
        self.record_metrics(round_data)
        
    def record_round_completion(
        self,
        round_num: int,
        participated_clients: List[str],
        aggregation_time: float,
        round_metrics: Dict[str, Any]
    ) -> None:
        """
        Record the completion of a training round.
        
        Args:
            round_num: Round number
            participated_clients: Clients that participated
            aggregation_time: Time spent on aggregation
            round_metrics: Additional round metrics
        """
        completion_data = {
            "event": "round_completion",
            "round": round_num,
            "participated_clients": participated_clients,
            "num_participated": len(participated_clients),
            "aggregation_time": aggregation_time,
            "completion_time": time.time(),
            **round_metrics
        }
        
        self.round_metrics.append(completion_data)
        self.record_metrics(completion_data)
        
    def record_aggregation(self, aggregation_data: Dict[str, Any]) -> None:
        """
        Record aggregation metrics.
        
        Args:
            aggregation_data: Aggregation-related metrics
        """
        agg_entry = {
            "timestamp": time.time(),
            **aggregation_data
        }
        
        self.aggregation_history.append(agg_entry)
        self.record_metrics(aggregation_data)
        
    def record_client_update(
        self,
        client_id: str,
        round_num: int,
        client_metrics: Dict[str, Any]
    ) -> None:
        """
        Record metrics from a client update.
        
        Args:
            client_id: Client identifier
            round_num: Round number
            client_metrics: Metrics from the client
        """
        update_data = {
            "client_id": client_id,
            "round": round_num,
            "timestamp": time.time(),
            **client_metrics
        }
        
        self.client_metrics[client_id].append(update_data)
        self.record_metrics(update_data)
        
    def get_server_summary(self) -> Dict[str, Any]:
        """Get comprehensive server metrics summary."""
        total_rounds = len(self.round_metrics)
        total_clients = len(self.client_metrics)
        
        if not self.round_metrics:
            return {"status": "no_training_data"}
        
        # Compute aggregation performance
        agg_times = [r.get("aggregation_time", 0.0) for r in self.round_metrics]
        avg_agg_time = statistics.mean(agg_times) if agg_times else 0.0
        
        # Compute client participation
        participation_rates = [
            r.get("num_participated", 0) / max(r.get("num_selected", 1), 1)
            for r in self.round_metrics
        ]
        avg_participation = statistics.mean(participation_rates) if participation_rates else 0.0
        
        return {
            "total_rounds": total_rounds,
            "total_clients": total_clients,
            "avg_aggregation_time": avg_agg_time,
            "avg_participation_rate": avg_participation,
            "latest_round": self.round_metrics[-1] if self.round_metrics else None,
            "training_duration": time.time() - self.start_time
        }


class UtilityMonitor:
    """
    Monitor privacy-utility tradeoffs in federated learning.
    
    Tracks the relationship between privacy spending and model performance
    to optimize the privacy-utility balance.
    """
    
    def __init__(self):
        """Initialize utility monitor."""
        self.privacy_utility_history: List[Dict[str, float]] = []
        self.baseline_utility: Optional[float] = None
        self.privacy_budgets: List[float] = []
        self.utility_scores: List[float] = []
        
    def track_metrics(self, func):
        """
        Decorator to track privacy-utility metrics.
        
        Args:
            func: Function that computes utility metrics
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                self.record_utility_point(result)
            
            return result
        
        return wrapper
    
    def record_utility_point(self, metrics: Dict[str, float]) -> None:
        """
        Record a privacy-utility data point.
        
        Args:
            metrics: Dictionary containing utility and privacy metrics
        """
        # Extract privacy and utility metrics
        epsilon = metrics.get("epsilon", 0.0)
        utility = metrics.get("accuracy", metrics.get("utility", 0.0))
        
        # Record data point
        data_point = {
            "timestamp": time.time(),
            "epsilon": epsilon,
            "utility": utility,
            "utility_privacy_ratio": utility / max(epsilon, 1e-8),
            **metrics
        }
        
        self.privacy_utility_history.append(data_point)
        self.privacy_budgets.append(epsilon)
        self.utility_scores.append(utility)
        
    def set_baseline_utility(self, baseline: float) -> None:
        """
        Set baseline utility for comparison.
        
        Args:
            baseline: Baseline utility (e.g., non-private model performance)
        """
        self.baseline_utility = baseline
        
    def compute_utility_degradation(self, current_utility: float) -> float:
        """
        Compute utility degradation compared to baseline.
        
        Args:
            current_utility: Current utility score
            
        Returns:
            Utility degradation percentage
        """
        if self.baseline_utility is None:
            return 0.0
        
        degradation = (self.baseline_utility - current_utility) / self.baseline_utility
        return max(0.0, degradation * 100)
    
    def find_optimal_privacy_budget(
        self,
        target_utility: float,
        tolerance: float = 0.05
    ) -> Optional[float]:
        """
        Find optimal privacy budget for target utility.
        
        Args:
            target_utility: Target utility score
            tolerance: Acceptable tolerance around target
            
        Returns:
            Optimal epsilon value or None if not found
        """
        candidates = []
        
        for point in self.privacy_utility_history:
            utility = point["utility"]
            epsilon = point["epsilon"]
            
            if abs(utility - target_utility) <= tolerance:
                candidates.append((epsilon, abs(utility - target_utility)))
        
        if candidates:
            # Return epsilon with smallest deviation from target
            return min(candidates, key=lambda x: x[1])[0]
        
        return None
    
    def plot_privacy_utility_curve(
        self,
        save_path: Optional[str] = None,
        show_baseline: bool = True
    ) -> None:
        """
        Plot privacy-utility tradeoff curve.
        
        Args:
            save_path: Path to save the plot
            show_baseline: Whether to show baseline utility
        """
        if not self.privacy_utility_history:
            logger.warning("No privacy-utility data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract data
        epsilons = [p["epsilon"] for p in self.privacy_utility_history]
        utilities = [p["utility"] for p in self.privacy_utility_history]
        
        # Create scatter plot
        plt.scatter(epsilons, utilities, alpha=0.6, s=50)
        
        # Add trend line
        if len(epsilons) > 1:
            z = np.polyfit(epsilons, utilities, 1)
            p = np.poly1d(z)
            plt.plot(sorted(epsilons), p(sorted(epsilons)), "r--", alpha=0.8, linewidth=2)
        
        # Add baseline line
        if show_baseline and self.baseline_utility is not None:
            plt.axhline(
                y=self.baseline_utility,
                color='green',
                linestyle='-',
                alpha=0.7,
                label=f'Baseline Utility: {self.baseline_utility:.3f}'
            )
        
        plt.xlabel('Privacy Budget (ε)')
        plt.ylabel('Model Utility')
        plt.title('Privacy-Utility Tradeoff')
        plt.grid(True, alpha=0.3)
        
        if show_baseline and self.baseline_utility is not None:
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Privacy-utility curve saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive privacy-utility report.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.privacy_utility_history:
            return {"error": "No data available"}
        
        # Basic statistics
        epsilons = [p["epsilon"] for p in self.privacy_utility_history]
        utilities = [p["utility"] for p in self.privacy_utility_history]
        
        # Compute correlations
        if len(epsilons) > 1:
            correlation = np.corrcoef(epsilons, utilities)[0, 1]
        else:
            correlation = 0.0
        
        # Find optimal points
        max_utility_point = max(self.privacy_utility_history, key=lambda x: x["utility"])
        min_epsilon_point = min(self.privacy_utility_history, key=lambda x: x["epsilon"])
        
        # Compute efficiency metrics
        efficiency_scores = [p["utility_privacy_ratio"] for p in self.privacy_utility_history]
        best_efficiency_point = max(self.privacy_utility_history, key=lambda x: x["utility_privacy_ratio"])
        
        report = {
            "summary": {
                "total_experiments": len(self.privacy_utility_history),
                "epsilon_range": [min(epsilons), max(epsilons)] if epsilons else [0, 0],
                "utility_range": [min(utilities), max(utilities)] if utilities else [0, 0],
                "privacy_utility_correlation": correlation
            },
            "optimal_points": {
                "max_utility": {
                    "epsilon": max_utility_point["epsilon"],
                    "utility": max_utility_point["utility"],
                    "ratio": max_utility_point["utility_privacy_ratio"]
                },
                "min_epsilon": {
                    "epsilon": min_epsilon_point["epsilon"],
                    "utility": min_epsilon_point["utility"],
                    "ratio": min_epsilon_point["utility_privacy_ratio"]
                },
                "best_efficiency": {
                    "epsilon": best_efficiency_point["epsilon"],
                    "utility": best_efficiency_point["utility"],
                    "ratio": best_efficiency_point["utility_privacy_ratio"]
                }
            },
            "statistics": {
                "avg_epsilon": statistics.mean(epsilons),
                "avg_utility": statistics.mean(utilities),
                "avg_efficiency": statistics.mean(efficiency_scores),
                "std_epsilon": statistics.stdev(epsilons) if len(epsilons) > 1 else 0,
                "std_utility": statistics.stdev(utilities) if len(utilities) > 1 else 0
            }
        }
        
        # Add baseline comparison if available
        if self.baseline_utility is not None:
            avg_degradation = self.compute_utility_degradation(statistics.mean(utilities))
            report["baseline_comparison"] = {
                "baseline_utility": self.baseline_utility,
                "avg_degradation_percent": avg_degradation,
                "best_degradation_percent": self.compute_utility_degradation(max_utility_point["utility"])
            }
        
        return report


class DashboardMetrics:
    """
    Real-time metrics for monitoring dashboards.
    
    Provides formatted metrics suitable for display in monitoring
    dashboards and web interfaces.
    """
    
    def __init__(self):
        """Initialize dashboard metrics."""
        self.metrics_cache: Dict[str, Any] = {}
        self.last_update = time.time()
        self.update_interval = 5.0  # seconds
        
    def update_cache(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics cache.
        
        Args:
            metrics: New metrics to cache
        """
        self.metrics_cache.update(metrics)
        self.last_update = time.time()
        
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard display."""
        return {
            "timestamp": time.time(),
            "last_update": self.last_update,
            "cache_age": time.time() - self.last_update,
            **self.metrics_cache
        }
    
    def format_for_display(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Format metrics for human-readable display.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            Formatted metrics for display
        """
        formatted = {}
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if "epsilon" in key.lower() or "delta" in key.lower():
                    formatted[key] = f"{value:.4f}"
                elif "accuracy" in key.lower() or "utility" in key.lower():
                    formatted[key] = f"{value:.2%}"
                elif "time" in key.lower() or "duration" in key.lower():
                    formatted[key] = f"{value:.1f}s"
                else:
                    formatted[key] = f"{value:.3f}"
            elif isinstance(value, int):
                formatted[key] = f"{value:,}"
            else:
                formatted[key] = str(value)
        
        return formatted
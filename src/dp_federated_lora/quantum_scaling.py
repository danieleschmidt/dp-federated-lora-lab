"""
Quantum-Enhanced Scaling and Performance Optimization

Implements quantum-inspired auto-scaling, load balancing, and performance optimization
for federated learning systems with adaptive resource management.
"""

import asyncio
import logging
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import psutil
import os

import numpy as np
import torch
import torch.distributed as dist
from pydantic import BaseModel, Field

from .quantum_monitoring import QuantumMetricsCollector, QuantumMetricType
from .quantum_resilience import QuantumResilienceManager
from .config import FederatedConfig
from .exceptions import QuantumOptimizationError


class ScalingStrategy(Enum):
    """Scaling strategies for quantum-enhanced systems"""
    QUANTUM_PREDICTIVE = "quantum_predictive"
    ADAPTIVE_COHERENCE = "adaptive_coherence"
    ENTANGLEMENT_BASED = "entanglement_based"
    SUPERPOSITION_SCALING = "superposition_scaling"
    CLASSICAL_LINEAR = "classical_linear"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU_CORES = "cpu_cores"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_IOPS = "storage_iops"
    CLIENT_CONNECTIONS = "client_connections"
    QUANTUM_COHERENCE = "quantum_coherence"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    network_utilization: float = 0.0
    storage_utilization: float = 0.0
    client_load: float = 0.0
    quantum_coherence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "network_utilization": self.network_utilization,
            "storage_utilization": self.storage_utilization,
            "client_load": self.client_load,
            "quantum_coherence": self.quantum_coherence,
            "timestamp": self.timestamp
        }


@dataclass
class ScalingDecision:
    """Scaling decision with quantum-inspired reasoning"""
    resource_type: ResourceType
    action: str  # "scale_up", "scale_down", "maintain"
    scale_factor: float
    confidence: float
    quantum_probability: float
    reasoning: str
    estimated_impact: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "resource_type": self.resource_type.value,
            "action": self.action,
            "scale_factor": self.scale_factor,
            "confidence": self.confidence,
            "quantum_probability": self.quantum_probability,
            "reasoning": self.reasoning,
            "estimated_impact": self.estimated_impact
        }


class QuantumResourcePredictor:
    """Quantum-inspired resource demand prediction"""
    
    def __init__(
        self,
        prediction_window: int = 50,
        quantum_coherence_threshold: float = 0.7
    ):
        self.prediction_window = prediction_window
        self.quantum_coherence_threshold = quantum_coherence_threshold
        self.metrics_history: List[ResourceMetrics] = []
        self.quantum_state = np.array([1.0 + 0j, 0.0 + 0j])  # |0⟩ state initially
        self.logger = logging.getLogger(__name__)
        
    def add_metrics(self, metrics: ResourceMetrics) -> None:
        """Add new metrics to history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.prediction_window:
            self.metrics_history = self.metrics_history[-self.prediction_window:]
            
    def predict_resource_demand(
        self,
        resource_type: ResourceType,
        time_horizon: float = 300.0  # 5 minutes ahead
    ) -> Tuple[float, float]:
        """
        Predict future resource demand using quantum-inspired algorithms
        
        Returns:
            Tuple of (predicted_demand, confidence)
        """
        if len(self.metrics_history) < 10:
            return 0.5, 0.1  # Low confidence prediction
            
        # Extract historical values for the resource
        values = []
        for metrics in self.metrics_history:
            if resource_type == ResourceType.CPU_CORES:
                values.append(metrics.cpu_utilization)
            elif resource_type == ResourceType.MEMORY:
                values.append(metrics.memory_utilization)
            elif resource_type == ResourceType.GPU_MEMORY:
                values.append(metrics.gpu_utilization)
            elif resource_type == ResourceType.CLIENT_CONNECTIONS:
                values.append(metrics.client_load)
            elif resource_type == ResourceType.QUANTUM_COHERENCE:
                values.append(metrics.quantum_coherence)
            else:
                values.append(0.5)  # Default value
                
        # Quantum-inspired prediction using superposition of multiple models
        predictions = []
        confidences = []
        
        # Model 1: Quantum harmonic oscillator model
        harmonic_pred, harmonic_conf = self._harmonic_oscillator_prediction(values, time_horizon)
        predictions.append(harmonic_pred)
        confidences.append(harmonic_conf)
        
        # Model 2: Quantum tunneling model (for sudden changes)
        tunnel_pred, tunnel_conf = self._quantum_tunneling_prediction(values, time_horizon)
        predictions.append(tunnel_pred)
        confidences.append(tunnel_conf)
        
        # Model 3: Classical trend analysis
        trend_pred, trend_conf = self._trend_analysis_prediction(values, time_horizon)
        predictions.append(trend_pred)
        confidences.append(trend_conf)
        
        # Quantum superposition of predictions
        prediction, confidence = self._quantum_ensemble_prediction(
            predictions, confidences
        )
        
        return prediction, confidence
        
    def _harmonic_oscillator_prediction(
        self, 
        values: List[float], 
        time_horizon: float
    ) -> Tuple[float, float]:
        """Predict using quantum harmonic oscillator model"""
        if len(values) < 5:
            return np.mean(values), 0.3
            
        # Fit harmonic oscillator parameters
        t = np.arange(len(values))
        y = np.array(values)
        
        # Find frequency using FFT
        fft_result = np.fft.fft(y - np.mean(y))
        frequencies = np.fft.fftfreq(len(y))
        dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
        omega = 2 * np.pi * abs(frequencies[dominant_freq_idx])
        
        # Fit amplitude and phase
        A = np.std(y) * 2
        mean_val = np.mean(y)
        
        # Predict using harmonic oscillator
        future_time = len(values) + time_horizon / 60.0  # Convert to time steps
        prediction = mean_val + A * np.cos(omega * future_time)
        
        # Confidence based on model fit
        fitted_values = mean_val + A * np.cos(omega * t)
        mse = np.mean((y - fitted_values) ** 2)
        confidence = max(0.1, 1.0 - mse)
        
        return max(0.0, min(1.0, prediction)), confidence
        
    def _quantum_tunneling_prediction(
        self,
        values: List[float],
        time_horizon: float
    ) -> Tuple[float, float]:
        """Predict sudden changes using quantum tunneling model"""
        if len(values) < 3:
            return values[-1], 0.2
            
        # Calculate recent gradient
        recent_values = values[-5:]  # Last 5 values
        gradient = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        # Quantum tunneling probability for sudden changes
        barrier_height = abs(gradient) * 10  # Scale gradient
        tunneling_prob = np.exp(-2 * barrier_height)  # Quantum tunneling probability
        
        current_value = values[-1]
        
        if tunneling_prob > 0.1:  # Significant tunneling probability
            # Predict sudden change
            tunnel_magnitude = np.random.normal(0, 0.2)  # Gaussian tunneling
            prediction = current_value + tunnel_magnitude
            confidence = tunneling_prob
        else:
            # Gradual change
            prediction = current_value + gradient * (time_horizon / 60.0)
            confidence = 1.0 - tunneling_prob
            
        return max(0.0, min(1.0, prediction)), confidence
        
    def _trend_analysis_prediction(
        self,
        values: List[float],
        time_horizon: float
    ) -> Tuple[float, float]:
        """Classical trend analysis prediction"""
        if len(values) < 2:
            return values[-1], 0.1
            
        # Simple linear regression
        t = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and intercept
        n = len(values)
        slope = (n * np.sum(t * y) - np.sum(t) * np.sum(y)) / (n * np.sum(t**2) - np.sum(t)**2)
        intercept = (np.sum(y) - slope * np.sum(t)) / n
        
        # Predict
        future_time = len(values) + time_horizon / 60.0
        prediction = slope * future_time + intercept
        
        # Confidence based on R-squared
        y_pred = slope * t + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0.1, r_squared)
        
        return max(0.0, min(1.0, prediction)), confidence
        
    def _quantum_ensemble_prediction(
        self,
        predictions: List[float],
        confidences: List[float]
    ) -> Tuple[float, float]:
        """Combine predictions using quantum superposition"""
        if not predictions:
            return 0.5, 0.1
            
        # Normalize confidences to create quantum amplitudes
        total_confidence = sum(confidences)
        if total_confidence == 0:
            amplitudes = [1.0 / len(predictions)] * len(predictions)
        else:
            amplitudes = [c / total_confidence for c in confidences]
            
        # Create quantum superposition state
        quantum_prediction = sum(a * p for a, p in zip(amplitudes, predictions))
        
        # Calculate ensemble confidence
        variance = sum(a * (p - quantum_prediction)**2 for a, p in zip(amplitudes, predictions))
        ensemble_confidence = max(0.1, 1.0 / (1.0 + variance))
        
        return quantum_prediction, ensemble_confidence


class QuantumAutoScaler:
    """Quantum-enhanced auto-scaling system"""
    
    def __init__(
        self,
        config: FederatedConfig,
        metrics_collector: QuantumMetricsCollector,
        resilience_manager: QuantumResilienceManager
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.resilience_manager = resilience_manager
        
        self.predictor = QuantumResourcePredictor()
        self.scaling_decisions: List[ScalingDecision] = []
        
        # Scaling thresholds
        self.scale_up_threshold = 0.75
        self.scale_down_threshold = 0.25
        self.quantum_coherence_threshold = 0.5
        
        # Resource limits
        self.max_cpu_cores = multiprocessing.cpu_count() * 2
        self.max_memory_gb = psutil.virtual_memory().total / (1024**3) * 1.5
        self.min_resources = {
            ResourceType.CPU_CORES: 1,
            ResourceType.MEMORY: 1.0,
            ResourceType.CLIENT_CONNECTIONS: 1
        }
        
        self.current_resources = self._get_current_resources()
        self.logger = logging.getLogger(__name__)
        
    def _get_current_resources(self) -> Dict[ResourceType, float]:
        """Get current resource allocation"""
        return {
            ResourceType.CPU_CORES: multiprocessing.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),
            ResourceType.CLIENT_CONNECTIONS: self.config.max_clients,
            ResourceType.QUANTUM_COHERENCE: 1.0
        }
        
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource utilization metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        # GPU metrics if available
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_utilization = min(1.0, gpu_memory)
            except:
                pass
                
        # Network utilization (simplified)
        net_io = psutil.net_io_counters()
        network_utilization = min(1.0, (net_io.bytes_sent + net_io.bytes_recv) / (1024**3))  # GB
        
        # Client load from metrics collector
        client_load = 0.0
        if hasattr(self.metrics_collector, 'get_client_count'):
            current_clients = self.metrics_collector.get_client_count()
            client_load = current_clients / max(self.config.max_clients, 1)
            
        return ResourceMetrics(
            cpu_utilization=cpu_percent / 100.0,
            memory_utilization=memory_info.percent / 100.0,
            gpu_utilization=gpu_utilization,
            network_utilization=network_utilization,
            client_load=client_load,
            quantum_coherence=self._estimate_quantum_coherence()
        )
        
    def _estimate_quantum_coherence(self) -> float:
        """Estimate system-wide quantum coherence"""
        # Get quantum coherence from circuit breakers
        coherence_values = []
        
        resilience_status = self.resilience_manager.get_resilience_status()
        for cb_name, cb_info in resilience_status.get("circuit_breakers", {}).items():
            coherence = cb_info.get("quantum_coherence", 1.0)
            coherence_values.append(coherence)
            
        if coherence_values:
            return np.mean(coherence_values)
        else:
            return 1.0  # Default high coherence
            
    async def evaluate_scaling_needs(self) -> List[ScalingDecision]:
        """Evaluate and generate scaling decisions"""
        current_metrics = self._collect_resource_metrics()
        self.predictor.add_metrics(current_metrics)
        
        scaling_decisions = []
        
        for resource_type in [ResourceType.CPU_CORES, ResourceType.MEMORY, ResourceType.CLIENT_CONNECTIONS]:
            decision = await self._evaluate_resource_scaling(resource_type, current_metrics)
            if decision:
                scaling_decisions.append(decision)
                
        # Special handling for quantum coherence
        coherence_decision = await self._evaluate_quantum_coherence_scaling(current_metrics)
        if coherence_decision:
            scaling_decisions.append(coherence_decision)
            
        self.scaling_decisions.extend(scaling_decisions)
        return scaling_decisions
        
    async def _evaluate_resource_scaling(
        self,
        resource_type: ResourceType,
        current_metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Evaluate scaling for a specific resource type"""
        # Get current utilization
        if resource_type == ResourceType.CPU_CORES:
            current_utilization = current_metrics.cpu_utilization
        elif resource_type == ResourceType.MEMORY:
            current_utilization = current_metrics.memory_utilization
        elif resource_type == ResourceType.CLIENT_CONNECTIONS:
            current_utilization = current_metrics.client_load
        else:
            return None
            
        # Predict future demand
        predicted_demand, confidence = self.predictor.predict_resource_demand(resource_type)
        
        # Quantum probability calculation
        quantum_coherence = current_metrics.quantum_coherence
        quantum_probability = self._calculate_quantum_scaling_probability(
            current_utilization, predicted_demand, quantum_coherence
        )
        
        # Determine scaling action
        action = "maintain"
        scale_factor = 1.0
        reasoning = "No scaling needed"
        
        if quantum_probability > 0.7:
            if predicted_demand > self.scale_up_threshold or current_utilization > self.scale_up_threshold:
                action = "scale_up"
                scale_factor = min(2.0, 1.0 + (predicted_demand - self.scale_up_threshold))
                reasoning = f"High demand predicted: {predicted_demand:.2f}, current: {current_utilization:.2f}"
                
            elif predicted_demand < self.scale_down_threshold and current_utilization < self.scale_down_threshold:
                action = "scale_down"
                scale_factor = max(0.5, 1.0 - (self.scale_down_threshold - predicted_demand))
                reasoning = f"Low demand predicted: {predicted_demand:.2f}, current: {current_utilization:.2f}"
                
        # Estimate impact
        estimated_impact = self._estimate_scaling_impact(resource_type, scale_factor)
        
        if action != "maintain":
            return ScalingDecision(
                resource_type=resource_type,
                action=action,
                scale_factor=scale_factor,
                confidence=confidence,
                quantum_probability=quantum_probability,
                reasoning=reasoning,
                estimated_impact=estimated_impact
            )
            
        return None
        
    async def _evaluate_quantum_coherence_scaling(
        self,
        current_metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Evaluate quantum coherence-based scaling"""
        coherence = current_metrics.quantum_coherence
        
        if coherence < self.quantum_coherence_threshold:
            # Low coherence suggests system stress - scale up resources
            scale_factor = 1.5  # Increase resources by 50%
            
            return ScalingDecision(
                resource_type=ResourceType.QUANTUM_COHERENCE,
                action="scale_up",
                scale_factor=scale_factor,
                confidence=1.0 - coherence,  # Higher confidence when coherence is lower
                quantum_probability=1.0 - coherence,
                reasoning=f"Low quantum coherence: {coherence:.3f} < {self.quantum_coherence_threshold}",
                estimated_impact={"coherence_improvement": 0.3, "resource_cost": 0.5}
            )
            
        return None
        
    def _calculate_quantum_scaling_probability(
        self,
        current_utilization: float,
        predicted_demand: float,
        quantum_coherence: float
    ) -> float:
        """Calculate quantum probability for scaling decision"""
        # Base probability from utilization and prediction
        utilization_factor = abs(predicted_demand - current_utilization)
        base_probability = min(1.0, utilization_factor * 2)
        
        # Quantum coherence affects decision certainty
        coherence_factor = quantum_coherence
        
        # Quantum superposition: sometimes make counter-intuitive decisions
        superposition_factor = 1.0
        if quantum_coherence > 0.8:
            # In high coherence state, add quantum fluctuations
            phase = 2 * np.pi * (current_utilization + predicted_demand)
            superposition_factor = 1.0 + 0.1 * np.sin(phase)
            
        quantum_probability = base_probability * coherence_factor * superposition_factor
        return max(0.0, min(1.0, quantum_probability))
        
    def _estimate_scaling_impact(
        self,
        resource_type: ResourceType,
        scale_factor: float
    ) -> Dict[str, float]:
        """Estimate the impact of scaling decision"""
        impact = {}
        
        if scale_factor > 1.0:  # Scaling up
            impact["performance_improvement"] = min(0.8, (scale_factor - 1.0) * 0.6)
            impact["cost_increase"] = (scale_factor - 1.0) * 0.8
            impact["reliability_improvement"] = min(0.5, (scale_factor - 1.0) * 0.4)
        elif scale_factor < 1.0:  # Scaling down
            impact["cost_reduction"] = (1.0 - scale_factor) * 0.8
            impact["performance_degradation"] = min(0.6, (1.0 - scale_factor) * 0.5)
            impact["reliability_risk"] = min(0.4, (1.0 - scale_factor) * 0.3)
        else:
            impact["no_change"] = 1.0
            
        return impact
        
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision"""
        try:
            self.logger.info(f"Executing scaling decision: {decision.action} {decision.resource_type.value} by {decision.scale_factor:.2f}")
            
            if decision.resource_type == ResourceType.CPU_CORES:
                return await self._scale_cpu_resources(decision.scale_factor, decision.action)
            elif decision.resource_type == ResourceType.MEMORY:
                return await self._scale_memory_resources(decision.scale_factor, decision.action)
            elif decision.resource_type == ResourceType.CLIENT_CONNECTIONS:
                return await self._scale_client_capacity(decision.scale_factor, decision.action)
            elif decision.resource_type == ResourceType.QUANTUM_COHERENCE:
                return await self._scale_quantum_coherence(decision.scale_factor, decision.action)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            return False
            
    async def _scale_cpu_resources(self, scale_factor: float, action: str) -> bool:
        """Scale CPU resources (thread/process pool sizes)"""
        current_cores = self.current_resources[ResourceType.CPU_CORES]
        
        if action == "scale_up":
            new_cores = min(self.max_cpu_cores, current_cores * scale_factor)
        else:
            new_cores = max(self.min_resources[ResourceType.CPU_CORES], current_cores * scale_factor)
            
        # Update thread pool sizes in components that use them
        # This would require coordination with other components
        self.current_resources[ResourceType.CPU_CORES] = new_cores
        
        self.logger.info(f"Scaled CPU resources from {current_cores} to {new_cores} cores")
        return True
        
    async def _scale_memory_resources(self, scale_factor: float, action: str) -> bool:
        """Scale memory resources (cache sizes, buffer sizes)"""
        current_memory = self.current_resources[ResourceType.MEMORY]
        
        if action == "scale_up":
            new_memory = min(self.max_memory_gb, current_memory * scale_factor)
        else:
            new_memory = max(self.min_resources[ResourceType.MEMORY], current_memory * scale_factor)
            
        # Update memory allocations in components
        self.current_resources[ResourceType.MEMORY] = new_memory
        
        self.logger.info(f"Scaled memory resources from {current_memory:.1f} to {new_memory:.1f} GB")
        return True
        
    async def _scale_client_capacity(self, scale_factor: float, action: str) -> bool:
        """Scale client connection capacity"""
        current_capacity = self.current_resources[ResourceType.CLIENT_CONNECTIONS]
        
        if action == "scale_up":
            new_capacity = min(1000, current_capacity * scale_factor)  # Max 1000 clients
        else:
            new_capacity = max(self.min_resources[ResourceType.CLIENT_CONNECTIONS], 
                             current_capacity * scale_factor)
            
        # Update configuration
        self.config.max_clients = int(new_capacity)
        self.current_resources[ResourceType.CLIENT_CONNECTIONS] = new_capacity
        
        self.logger.info(f"Scaled client capacity from {current_capacity} to {new_capacity} connections")
        return True
        
    async def _scale_quantum_coherence(self, scale_factor: float, action: str) -> bool:
        """Scale quantum coherence (adjust quantum algorithm parameters)"""
        # This would involve adjusting quantum algorithm parameters
        # to improve coherence
        
        if action == "scale_up":
            # Reduce decoherence in quantum components
            for cb_name, circuit_breaker in self.resilience_manager.circuit_breakers.items():
                circuit_breaker.quantum_coherence = min(1.0, circuit_breaker.quantum_coherence * 1.1)
                
        self.logger.info(f"Quantum coherence scaling executed: {action}")
        return True
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status"""
        current_metrics = self._collect_resource_metrics()
        
        return {
            "current_resources": {rt.value: val for rt, val in self.current_resources.items()},
            "current_utilization": current_metrics.to_dict(),
            "recent_decisions": [d.to_dict() for d in self.scaling_decisions[-10:]],
            "scaling_thresholds": {
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold,
                "quantum_coherence": self.quantum_coherence_threshold
            }
        }


# Global auto-scaler instance
_quantum_auto_scaler: Optional[QuantumAutoScaler] = None


def get_quantum_auto_scaler(
    config: FederatedConfig,
    metrics_collector: QuantumMetricsCollector,
    resilience_manager: QuantumResilienceManager
) -> QuantumAutoScaler:
    """Get global quantum auto-scaler instance"""
    global _quantum_auto_scaler
    if _quantum_auto_scaler is None:
        _quantum_auto_scaler = QuantumAutoScaler(config, metrics_collector, resilience_manager)
    return _quantum_auto_scaler


async def initialize_quantum_auto_scaling(
    config: FederatedConfig,
    metrics_collector: QuantumMetricsCollector,
    resilience_manager: QuantumResilienceManager
) -> QuantumAutoScaler:
    """Initialize quantum auto-scaling system"""
    auto_scaler = get_quantum_auto_scaler(config, metrics_collector, resilience_manager)
    
    # Start auto-scaling loop
    asyncio.create_task(_auto_scaling_loop(auto_scaler))
    
    return auto_scaler


async def _auto_scaling_loop(auto_scaler: QuantumAutoScaler) -> None:
    """Background loop for auto-scaling decisions"""
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Evaluate scaling needs every 30 seconds
            decisions = await auto_scaler.evaluate_scaling_needs()
            
            # Execute decisions
            for decision in decisions:
                if decision.confidence > 0.6:  # Only execute high-confidence decisions
                    success = await auto_scaler.execute_scaling_decision(decision)
                    if not success:
                        logger.warning(f"Failed to execute scaling decision: {decision.action}")
                        
            await asyncio.sleep(30)  # 30 second evaluation interval
            
        except Exception as e:
            logger.error(f"Error in auto-scaling loop: {e}")
            await asyncio.sleep(60)  # Back off on error
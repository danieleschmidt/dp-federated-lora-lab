"""
Quantum-Enhanced Adaptive Load Balancer
Generation 3 Scalability Enhancement

This module implements advanced quantum-enhanced adaptive load balancing for 
federated learning systems, featuring quantum-inspired client selection algorithms,
dynamic traffic distribution, and adaptive resource allocation with quantum advantage.
"""

import asyncio
import logging
import time
import math
import numpy as np
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
from collections import defaultdict, deque
from datetime import datetime, timezone
import threading
import json

import torch
import torch.nn as nn
from scipy.optimize import minimize, differential_evolution
from scipy.stats import entropy
from sklearn.cluster import KMeans

from .multi_region_deployment import DeploymentRegion, ClientGeoLocation
from .quantum_global_orchestration import QuantumGlobalLoadBalancer
from .quantum_entanglement_coordinator import QuantumEntanglementCoordinator, EntanglementType
from .config import FederatedConfig
from .exceptions import QuantumOptimizationError


logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Advanced load balancing strategies"""
    QUANTUM_ANNEALING = "quantum_annealing"
    VARIATIONAL_QUANTUM = "variational_quantum"
    QUANTUM_CLUSTERING = "quantum_clustering"
    ENTANGLEMENT_BASED = "entanglement_based"
    ADAPTIVE_SUPERPOSITION = "adaptive_superposition"
    HYBRID_QUANTUM_CLASSICAL = "hybrid_quantum_classical"


class ClientPriority(Enum):
    """Client priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ClientProfile:
    """Comprehensive client profile for load balancing"""
    client_id: str
    location: ClientGeoLocation
    priority: ClientPriority
    compute_capacity: float  # Relative compute power
    network_quality: float  # Network stability/speed
    data_quality: float  # Quality of training data
    participation_history: List[float]  # Historical participation rates
    resource_contribution: float  # GPU/CPU contribution
    latency_sensitivity: float  # How sensitive to latency
    privacy_requirements: List[str]  # Specific privacy needs
    preferred_regions: List[DeploymentRegion]
    training_frequency: float  # How often client trains
    model_size_preference: float  # Preferred model size
    quantum_capability: float  # Quantum processing capability
    last_activity: datetime
    
    def calculate_utility_score(self, current_load: float = 0.5) -> float:
        """Calculate overall utility score for load balancing"""
        scores = [
            self.compute_capacity * 0.25,
            self.network_quality * 0.20,
            self.data_quality * 0.20,
            self.resource_contribution * 0.15,
            (1.0 - self.latency_sensitivity) * 0.10,  # Lower sensitivity = higher score
            self.quantum_capability * 0.10
        ]
        
        # Penalty for high load
        load_penalty = current_load * 0.1
        
        base_score = sum(scores) - load_penalty
        
        # Priority multiplier
        priority_multipliers = {
            ClientPriority.CRITICAL: 2.0,
            ClientPriority.HIGH: 1.5,
            ClientPriority.NORMAL: 1.0,
            ClientPriority.LOW: 0.7,
            ClientPriority.BACKGROUND: 0.3
        }
        
        return base_score * priority_multipliers[self.priority]


@dataclass
class LoadBalancingDecision:
    """Load balancing decision with quantum reasoning"""
    decision_id: str
    client_id: str
    assigned_region: DeploymentRegion
    alternative_regions: List[DeploymentRegion]
    quantum_confidence: float
    classical_confidence: float
    quantum_advantage: float
    decision_factors: Dict[str, float]
    expected_latency: float
    expected_throughput: float
    reasoning: str
    timestamp: datetime
    execution_success: Optional[bool] = None
    actual_performance: Optional[Dict[str, float]] = None


@dataclass
class RegionalLoadState:
    """Current load state for a region"""
    region: DeploymentRegion
    current_clients: int
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    quantum_coherence: float
    average_latency: float
    throughput_capacity: float
    error_rate: float
    client_satisfaction: float
    last_update: datetime
    
    def calculate_load_score(self) -> float:
        """Calculate normalized load score (0.0 = no load, 1.0 = fully loaded)"""
        utilization_score = (
            self.cpu_utilization * 0.3 +
            self.memory_utilization * 0.3 +
            self.network_utilization * 0.2 +
            (self.current_clients / 100.0) * 0.2  # Assume max 100 clients per region
        )
        
        # Quality penalties
        latency_penalty = min(0.2, self.average_latency / 500.0)  # Penalty for >500ms latency
        error_penalty = min(0.2, self.error_rate * 2.0)  # Penalty for >10% error rate
        
        return min(1.0, utilization_score + latency_penalty + error_penalty)


class QuantumClientSelector:
    """Quantum-inspired client selection algorithm"""
    
    def __init__(self, quantum_threshold: float = 0.8):
        self.quantum_threshold = quantum_threshold
        self.selection_history: List[Dict[str, Any]] = []
        
    def quantum_client_selection(
        self,
        available_clients: List[ClientProfile],
        target_count: int,
        selection_criteria: Dict[str, float]
    ) -> Tuple[List[ClientProfile], float]:
        """Select clients using quantum-inspired algorithm"""
        
        if len(available_clients) <= target_count:
            return available_clients, 1.0
        
        # Create quantum superposition of client states
        n_clients = len(available_clients)
        
        # Calculate client amplitudes based on utility scores
        amplitudes = []
        for client in available_clients:
            utility = client.calculate_utility_score()
            
            # Apply selection criteria weights
            weighted_utility = utility
            for criterion, weight in selection_criteria.items():
                if hasattr(client, criterion):
                    criterion_value = getattr(client, criterion)
                    if isinstance(criterion_value, (int, float)):
                        weighted_utility += criterion_value * weight
            
            amplitudes.append(math.sqrt(max(0.1, weighted_utility)))
        
        # Normalize amplitudes
        total_amplitude = sum(a**2 for a in amplitudes)
        if total_amplitude > 0:
            amplitudes = [a / math.sqrt(total_amplitude) for a in amplitudes]
        else:
            amplitudes = [1.0 / math.sqrt(n_clients)] * n_clients
        
        # Quantum measurement simulation
        selected_indices = set()
        quantum_confidence = 0.0
        
        for _ in range(target_count):
            # Calculate measurement probabilities
            probabilities = [a**2 for a in amplitudes]
            
            # Select client based on quantum probabilities
            selected_idx = np.random.choice(n_clients, p=probabilities)
            
            if selected_idx not in selected_indices:
                selected_indices.add(selected_idx)
                quantum_confidence += probabilities[selected_idx]
                
                # Collapse amplitude for selected client (can't be selected again)
                amplitudes[selected_idx] = 0.0
                
                # Renormalize remaining amplitudes
                remaining_amplitude = sum(a**2 for a in amplitudes)
                if remaining_amplitude > 0:
                    amplitudes = [a / math.sqrt(remaining_amplitude) for a in amplitudes]
        
        # Fill remaining slots if needed
        while len(selected_indices) < target_count and len(selected_indices) < n_clients:
            remaining_indices = set(range(n_clients)) - selected_indices
            if remaining_indices:
                selected_indices.add(next(iter(remaining_indices)))
        
        selected_clients = [available_clients[i] for i in selected_indices]
        average_confidence = quantum_confidence / len(selected_indices) if selected_indices else 0.0
        
        # Record selection
        self.selection_history.append({
            "timestamp": datetime.now(timezone.utc),
            "total_clients": n_clients,
            "selected_count": len(selected_clients),
            "quantum_confidence": average_confidence,
            "selection_criteria": selection_criteria
        })
        
        return selected_clients, average_confidence


class QuantumTrafficDistributor:
    """Quantum-enhanced traffic distribution algorithm"""
    
    def __init__(self, coherence_threshold: float = 0.7):
        self.coherence_threshold = coherence_threshold
        self.distribution_state = np.array([1.0, 0.0], dtype=complex)  # |0⟩ state
        self.traffic_history: deque = deque(maxlen=1000)
        
    def distribute_traffic(
        self,
        traffic_load: float,
        regional_states: List[RegionalLoadState],
        quantum_coherence: float
    ) -> Dict[DeploymentRegion, float]:
        """Distribute traffic using quantum superposition"""
        
        if not regional_states:
            return {}
        
        n_regions = len(regional_states)
        
        # Create quantum state for traffic distribution
        if quantum_coherence > self.coherence_threshold:
            # High coherence: use quantum superposition
            distribution = self._quantum_superposition_distribution(traffic_load, regional_states)
        else:
            # Low coherence: fall back to classical algorithm
            distribution = self._classical_distribution(traffic_load, regional_states)
        
        # Record traffic distribution
        self.traffic_history.append({
            "timestamp": time.time(),
            "traffic_load": traffic_load,
            "quantum_coherence": quantum_coherence,
            "distribution": distribution,
            "algorithm_used": "quantum" if quantum_coherence > self.coherence_threshold else "classical"
        })
        
        return distribution
    
    def _quantum_superposition_distribution(
        self,
        traffic_load: float,
        regional_states: List[RegionalLoadState]
    ) -> Dict[DeploymentRegion, float]:
        """Distribute traffic using quantum superposition of optimal solutions"""
        
        n_regions = len(regional_states)
        
        # Create superposition state representing all possible distributions
        superposition_states = []
        
        # Generate multiple distribution strategies
        strategies = [
            "load_balanced",
            "latency_optimized", 
            "throughput_optimized",
            "quantum_coherence_optimized"
        ]
        
        for strategy in strategies:
            if strategy == "load_balanced":
                # Distribute inversely proportional to current load
                weights = [1.0 / (state.calculate_load_score() + 0.1) for state in regional_states]
                
            elif strategy == "latency_optimized":
                # Favor regions with low latency
                weights = [1.0 / (state.average_latency + 1.0) for state in regional_states]
                
            elif strategy == "throughput_optimized":
                # Favor regions with high throughput capacity
                weights = [state.throughput_capacity for state in regional_states]
                
            elif strategy == "quantum_coherence_optimized":
                # Favor regions with high quantum coherence
                weights = [state.quantum_coherence for state in regional_states]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
            else:
                normalized_weights = [1.0 / n_regions] * n_regions
            
            superposition_states.append(normalized_weights)
        
        # Create quantum superposition of strategies
        n_strategies = len(superposition_states)
        strategy_amplitudes = np.ones(n_strategies) / math.sqrt(n_strategies)
        
        # Quantum interference: combine strategies with interference effects
        final_distribution = np.zeros(n_regions)
        
        for i, amplitude in enumerate(strategy_amplitudes):
            strategy_distribution = np.array(superposition_states[i])
            
            # Add quantum phase for interference
            phase = 2 * np.pi * i / n_strategies
            quantum_contribution = amplitude * np.exp(1j * phase) * strategy_distribution
            
            # Take real part for final distribution
            final_distribution += np.real(quantum_contribution)
        
        # Ensure positive distribution
        final_distribution = np.abs(final_distribution)
        
        # Normalize to sum to traffic_load
        total_distribution = np.sum(final_distribution)
        if total_distribution > 0:
            final_distribution = (final_distribution / total_distribution) * traffic_load
        else:
            final_distribution = np.ones(n_regions) * (traffic_load / n_regions)
        
        # Convert to dictionary
        distribution = {}
        for i, state in enumerate(regional_states):
            distribution[state.region] = final_distribution[i]
        
        return distribution
    
    def _classical_distribution(
        self,
        traffic_load: float,
        regional_states: List[RegionalLoadState]
    ) -> Dict[DeploymentRegion, float]:
        """Classical weighted round-robin distribution"""
        
        # Calculate weights based on inverse load and capacity
        weights = []
        for state in regional_states:
            load_score = state.calculate_load_score()
            capacity_factor = state.throughput_capacity / max(1.0, state.average_latency)
            weight = (1.0 - load_score) * capacity_factor
            weights.append(max(0.1, weight))  # Minimum weight
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Distribute traffic proportionally
        distribution = {}
        for i, state in enumerate(regional_states):
            distribution[state.region] = traffic_load * normalized_weights[i]
        
        return distribution


class QuantumAdaptiveLoadBalancer:
    """Main quantum-enhanced adaptive load balancer"""
    
    def __init__(
        self,
        config: FederatedConfig,
        entanglement_coordinator: QuantumEntanglementCoordinator
    ):
        self.config = config
        self.entanglement_coordinator = entanglement_coordinator
        
        # Core components
        self.client_selector = QuantumClientSelector()
        self.traffic_distributor = QuantumTrafficDistributor()
        
        # State management
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.regional_states: Dict[DeploymentRegion, RegionalLoadState] = {}
        self.load_balancing_decisions: List[LoadBalancingDecision] = []
        
        # Adaptive parameters
        self.adaptation_rate = 0.1  # How quickly to adapt to changes
        self.quantum_coherence_threshold = 0.7
        self.load_balancing_strategy = LoadBalancingStrategy.ADAPTIVE_SUPERPOSITION
        
        # Performance metrics
        self.performance_metrics = {
            "total_decisions": 0,
            "quantum_decisions": 0,
            "classical_decisions": 0,
            "average_latency": 0.0,
            "average_throughput": 0.0,
            "client_satisfaction": 0.0,
            "quantum_advantage": 0.0,
            "adaptation_efficiency": 0.0
        }
        
        # Adaptive learning
        self.decision_outcomes: deque = deque(maxlen=1000)
        self.learning_rate = 0.01
        self.quantum_advantage_history: deque = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
        
    def register_client(
        self,
        client_id: str,
        location: ClientGeoLocation,
        capabilities: Dict[str, Any]
    ) -> ClientProfile:
        """Register new client with comprehensive profiling"""
        
        # Create client profile
        profile = ClientProfile(
            client_id=client_id,
            location=location,
            priority=ClientPriority(capabilities.get("priority", "normal")),
            compute_capacity=capabilities.get("compute_capacity", 0.5),
            network_quality=capabilities.get("network_quality", 0.7),
            data_quality=capabilities.get("data_quality", 0.8),
            participation_history=[],
            resource_contribution=capabilities.get("resource_contribution", 0.5),
            latency_sensitivity=capabilities.get("latency_sensitivity", 0.5),
            privacy_requirements=capabilities.get("privacy_requirements", []),
            preferred_regions=capabilities.get("preferred_regions", []),
            training_frequency=capabilities.get("training_frequency", 1.0),
            model_size_preference=capabilities.get("model_size_preference", 1.0),
            quantum_capability=capabilities.get("quantum_capability", 0.0),
            last_activity=datetime.now(timezone.utc)
        )
        
        self.client_profiles[client_id] = profile
        
        self.logger.info(f"Registered client {client_id} with quantum capability {profile.quantum_capability:.3f}")
        
        return profile
    
    def update_regional_state(
        self,
        region: DeploymentRegion,
        metrics: Dict[str, Any]
    ):
        """Update regional load state with latest metrics"""
        
        state = RegionalLoadState(
            region=region,
            current_clients=metrics.get("current_clients", 0),
            cpu_utilization=metrics.get("cpu_utilization", 0.0),
            memory_utilization=metrics.get("memory_utilization", 0.0),
            network_utilization=metrics.get("network_utilization", 0.0),
            quantum_coherence=metrics.get("quantum_coherence", 1.0),
            average_latency=metrics.get("average_latency", 100.0),
            throughput_capacity=metrics.get("throughput_capacity", 100.0),
            error_rate=metrics.get("error_rate", 0.0),
            client_satisfaction=metrics.get("client_satisfaction", 0.8),
            last_update=datetime.now(timezone.utc)
        )
        
        self.regional_states[region] = state
    
    async def make_load_balancing_decision(
        self,
        client_id: str,
        request_type: str,
        available_regions: List[DeploymentRegion],
        requirements: Optional[Dict[str, Any]] = None
    ) -> LoadBalancingDecision:
        """Make intelligent load balancing decision using quantum algorithms"""
        
        if client_id not in self.client_profiles:
            raise QuantumOptimizationError(f"Client {client_id} not registered")
        
        client_profile = self.client_profiles[client_id]
        requirements = requirements or {}
        
        # Determine if we should use quantum or classical approach
        quantum_coherence = self._calculate_system_coherence()
        use_quantum = quantum_coherence > self.quantum_coherence_threshold
        
        if use_quantum:
            decision = await self._quantum_load_balancing_decision(
                client_profile, request_type, available_regions, requirements
            )
        else:
            decision = await self._classical_load_balancing_decision(
                client_profile, request_type, available_regions, requirements
            )
        
        # Record decision
        self.load_balancing_decisions.append(decision)
        self.performance_metrics["total_decisions"] += 1
        
        if use_quantum:
            self.performance_metrics["quantum_decisions"] += 1
        else:
            self.performance_metrics["classical_decisions"] += 1
        
        # Update client activity
        client_profile.last_activity = datetime.now(timezone.utc)
        
        self.logger.info(f"Load balancing decision for client {client_id}: "
                        f"assigned to {decision.assigned_region.value} "
                        f"({decision.quantum_advantage:.3f} quantum advantage)")
        
        return decision
    
    async def _quantum_load_balancing_decision(
        self,
        client_profile: ClientProfile,
        request_type: str,
        available_regions: List[DeploymentRegion],
        requirements: Dict[str, Any]
    ) -> LoadBalancingDecision:
        """Make load balancing decision using quantum algorithms"""
        
        n_regions = len(available_regions)
        
        # Create quantum superposition of all possible assignments
        region_amplitudes = []
        
        for region in available_regions:
            regional_state = self.regional_states.get(region)
            if not regional_state:
                # Default state for unknown regions
                regional_state = RegionalLoadState(
                    region=region,
                    current_clients=0,
                    cpu_utilization=0.5,
                    memory_utilization=0.5,
                    network_utilization=0.5,
                    quantum_coherence=0.8,
                    average_latency=100.0,
                    throughput_capacity=100.0,
                    error_rate=0.01,
                    client_satisfaction=0.8,
                    last_update=datetime.now(timezone.utc)
                )
            
            # Calculate quantum amplitude based on multiple factors
            amplitude = self._calculate_quantum_amplitude(
                client_profile, regional_state, request_type, requirements
            )
            region_amplitudes.append(amplitude)
        
        # Normalize amplitudes
        total_amplitude_squared = sum(a**2 for a in region_amplitudes)
        if total_amplitude_squared > 0:
            region_amplitudes = [a / math.sqrt(total_amplitude_squared) for a in region_amplitudes]
        else:
            region_amplitudes = [1.0 / math.sqrt(n_regions)] * n_regions
        
        # Quantum measurement to select region
        probabilities = [a**2 for a in region_amplitudes]
        selected_region_idx = np.random.choice(n_regions, p=probabilities)
        selected_region = available_regions[selected_region_idx]
        
        # Calculate quantum confidence
        quantum_confidence = probabilities[selected_region_idx]
        
        # Calculate classical confidence for comparison
        classical_scores = []
        for region in available_regions:
            regional_state = self.regional_states.get(region)
            if regional_state:
                load_score = regional_state.calculate_load_score()
                latency_score = 1.0 / (regional_state.average_latency + 1.0)
                throughput_score = regional_state.throughput_capacity / 100.0
                classical_score = (1.0 - load_score) * 0.4 + latency_score * 0.3 + throughput_score * 0.3
                classical_scores.append(classical_score)
            else:
                classical_scores.append(0.5)
        
        best_classical_score = max(classical_scores)
        classical_confidence = classical_scores[selected_region_idx] / best_classical_score if best_classical_score > 0 else 0.5
        
        # Calculate quantum advantage
        quantum_advantage = max(0.0, quantum_confidence - classical_confidence)
        
        # Alternative regions (other high-probability regions)
        alternative_regions = []
        for i, prob in enumerate(probabilities):
            if i != selected_region_idx and prob > 0.1:  # At least 10% probability
                alternative_regions.append(available_regions[i])
        
        # Decision factors
        decision_factors = {
            "load_score": self.regional_states[selected_region].calculate_load_score() if selected_region in self.regional_states else 0.5,
            "latency_factor": 1.0 / (self.regional_states[selected_region].average_latency + 1.0) if selected_region in self.regional_states else 0.01,
            "quantum_coherence": self.regional_states[selected_region].quantum_coherence if selected_region in self.regional_states else 0.8,
            "client_utility": client_profile.calculate_utility_score(),
            "entanglement_potential": self._calculate_entanglement_potential(selected_region)
        }
        
        # Expected performance
        expected_latency = self.regional_states[selected_region].average_latency if selected_region in self.regional_states else 100.0
        expected_throughput = self.regional_states[selected_region].throughput_capacity if selected_region in self.regional_states else 100.0
        
        # Create decision
        decision = LoadBalancingDecision(
            decision_id=str(uuid.uuid4()),
            client_id=client_profile.client_id,
            assigned_region=selected_region,
            alternative_regions=alternative_regions,
            quantum_confidence=quantum_confidence,
            classical_confidence=classical_confidence,
            quantum_advantage=quantum_advantage,
            decision_factors=decision_factors,
            expected_latency=expected_latency,
            expected_throughput=expected_throughput,
            reasoning=f"Quantum algorithm selected {selected_region.value} with {quantum_confidence:.3f} confidence",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Track quantum advantage
        self.quantum_advantage_history.append(quantum_advantage)
        
        return decision
    
    async def _classical_load_balancing_decision(
        self,
        client_profile: ClientProfile,
        request_type: str,
        available_regions: List[DeploymentRegion],
        requirements: Dict[str, Any]
    ) -> LoadBalancingDecision:
        """Make load balancing decision using classical algorithms"""
        
        # Score each region based on classical metrics
        region_scores = {}
        
        for region in available_regions:
            regional_state = self.regional_states.get(region)
            if regional_state:
                # Calculate composite score
                load_factor = 1.0 - regional_state.calculate_load_score()
                latency_factor = 1.0 / (regional_state.average_latency + 1.0)
                throughput_factor = regional_state.throughput_capacity / 100.0
                satisfaction_factor = regional_state.client_satisfaction
                
                # Weight factors based on client preferences
                if client_profile.latency_sensitivity > 0.7:
                    score = latency_factor * 0.5 + load_factor * 0.3 + throughput_factor * 0.2
                elif client_profile.priority == ClientPriority.CRITICAL:
                    score = throughput_factor * 0.4 + load_factor * 0.4 + satisfaction_factor * 0.2
                else:
                    score = load_factor * 0.4 + latency_factor * 0.3 + throughput_factor * 0.3
                
                region_scores[region] = score
            else:
                region_scores[region] = 0.5  # Default score
        
        # Select best region
        selected_region = max(region_scores.keys(), key=region_scores.get)
        
        # Calculate confidence
        best_score = region_scores[selected_region]
        scores = list(region_scores.values())
        classical_confidence = best_score / max(scores) if max(scores) > 0 else 0.5
        
        # Alternative regions
        sorted_regions = sorted(region_scores.keys(), key=region_scores.get, reverse=True)
        alternative_regions = sorted_regions[1:4]  # Top 3 alternatives
        
        # Decision factors
        decision_factors = {
            "load_score": self.regional_states[selected_region].calculate_load_score() if selected_region in self.regional_states else 0.5,
            "latency_factor": 1.0 / (self.regional_states[selected_region].average_latency + 1.0) if selected_region in self.regional_states else 0.01,
            "classical_score": best_score,
            "client_utility": client_profile.calculate_utility_score()
        }
        
        # Expected performance
        expected_latency = self.regional_states[selected_region].average_latency if selected_region in self.regional_states else 100.0
        expected_throughput = self.regional_states[selected_region].throughput_capacity if selected_region in self.regional_states else 100.0
        
        # Create decision
        decision = LoadBalancingDecision(
            decision_id=str(uuid.uuid4()),
            client_id=client_profile.client_id,
            assigned_region=selected_region,
            alternative_regions=alternative_regions,
            quantum_confidence=0.0,  # No quantum in classical approach
            classical_confidence=classical_confidence,
            quantum_advantage=0.0,
            decision_factors=decision_factors,
            expected_latency=expected_latency,
            expected_throughput=expected_throughput,
            reasoning=f"Classical algorithm selected {selected_region.value} with score {best_score:.3f}",
            timestamp=datetime.now(timezone.utc)
        )
        
        return decision
    
    def _calculate_quantum_amplitude(
        self,
        client_profile: ClientProfile,
        regional_state: RegionalLoadState,
        request_type: str,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate quantum amplitude for region assignment"""
        
        # Base amplitude from region capacity
        load_score = regional_state.calculate_load_score()
        capacity_amplitude = math.sqrt(1.0 - load_score + 0.1)  # Add small positive offset
        
        # Client-region compatibility
        compatibility_factors = []
        
        # Latency compatibility
        if client_profile.latency_sensitivity > 0.5:
            latency_compatibility = 1.0 / (regional_state.average_latency / 100.0 + 1.0)
        else:
            latency_compatibility = 0.8  # Less sensitive clients
        compatibility_factors.append(latency_compatibility)
        
        # Quantum capability alignment
        if client_profile.quantum_capability > 0.3:
            quantum_compatibility = regional_state.quantum_coherence
        else:
            quantum_compatibility = 0.8  # Classical clients don't need high coherence
        compatibility_factors.append(quantum_compatibility)
        
        # Priority alignment
        if client_profile.priority in [ClientPriority.CRITICAL, ClientPriority.HIGH]:
            priority_factor = regional_state.throughput_capacity / 100.0
        else:
            priority_factor = 0.7  # Normal priority
        compatibility_factors.append(priority_factor)
        
        # Combine factors
        compatibility_score = np.mean(compatibility_factors)
        
        # Final amplitude
        amplitude = capacity_amplitude * math.sqrt(compatibility_score)
        
        return max(0.1, amplitude)  # Minimum amplitude to ensure all regions are possible
    
    def _calculate_entanglement_potential(self, region: DeploymentRegion) -> float:
        """Calculate potential for quantum entanglement with region"""
        
        # Check existing entangled pairs involving this region
        entanglement_strength = 0.0
        
        for pair in self.entanglement_coordinator.entangled_pairs.values():
            if pair.region_a == region or pair.region_b == region:
                entanglement_strength += pair.entanglement_strength * pair.fidelity
        
        # Check quantum clusters
        for cluster in self.entanglement_coordinator.quantum_clusters.values():
            if region in cluster.regions:
                entanglement_strength += cluster.get_cluster_fidelity() * 0.5
        
        return min(1.0, entanglement_strength)
    
    def _calculate_system_coherence(self) -> float:
        """Calculate overall system quantum coherence"""
        
        coherence_values = []
        
        # Regional coherence
        for state in self.regional_states.values():
            coherence_values.append(state.quantum_coherence)
        
        # Entanglement coherence
        for pair in self.entanglement_coordinator.entangled_pairs.values():
            coherence_values.append(pair.fidelity)
        
        # Cluster coherence
        for cluster in self.entanglement_coordinator.quantum_clusters.values():
            coherence_values.append(cluster.get_cluster_fidelity())
        
        if coherence_values:
            return np.mean(coherence_values)
        else:
            return 0.5  # Default moderate coherence
    
    async def adapt_strategy(self):
        """Adapt load balancing strategy based on performance feedback"""
        
        if len(self.decision_outcomes) < 10:
            return  # Need more data
        
        # Analyze recent performance
        recent_outcomes = list(self.decision_outcomes)[-50:]  # Last 50 decisions
        
        quantum_decisions = [o for o in recent_outcomes if o.get("used_quantum", False)]
        classical_decisions = [o for o in recent_outcomes if not o.get("used_quantum", False)]
        
        # Compare quantum vs classical performance
        if quantum_decisions and classical_decisions:
            quantum_performance = np.mean([o.get("performance_score", 0.5) for o in quantum_decisions])
            classical_performance = np.mean([o.get("performance_score", 0.5) for o in classical_decisions])
            
            performance_difference = quantum_performance - classical_performance
            
            # Adapt coherence threshold based on performance
            if performance_difference > 0.1:
                # Quantum is significantly better, lower threshold
                self.quantum_coherence_threshold = max(0.3, self.quantum_coherence_threshold - self.adaptation_rate)
            elif performance_difference < -0.1:
                # Classical is significantly better, raise threshold
                self.quantum_coherence_threshold = min(0.9, self.quantum_coherence_threshold + self.adaptation_rate)
            
            self.logger.info(f"Adapted quantum coherence threshold to {self.quantum_coherence_threshold:.3f} "
                           f"based on performance difference {performance_difference:.3f}")
    
    def record_decision_outcome(
        self,
        decision_id: str,
        actual_latency: float,
        actual_throughput: float,
        client_satisfaction: float,
        success: bool
    ):
        """Record actual outcome of load balancing decision"""
        
        # Find the decision
        decision = None
        for d in self.load_balancing_decisions:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            return
        
        # Update decision with actual performance
        decision.execution_success = success
        decision.actual_performance = {
            "latency": actual_latency,
            "throughput": actual_throughput,
            "satisfaction": client_satisfaction
        }
        
        # Calculate performance score
        latency_score = min(1.0, decision.expected_latency / max(1.0, actual_latency))
        throughput_score = min(1.0, actual_throughput / max(1.0, decision.expected_throughput))
        
        performance_score = (latency_score + throughput_score + client_satisfaction) / 3.0
        
        # Record outcome for adaptation
        outcome = {
            "decision_id": decision_id,
            "used_quantum": decision.quantum_advantage > 0,
            "performance_score": performance_score,
            "quantum_advantage": decision.quantum_advantage,
            "success": success,
            "timestamp": time.time()
        }
        
        self.decision_outcomes.append(outcome)
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update global performance metrics"""
        
        if not self.decision_outcomes:
            return
        
        recent_outcomes = list(self.decision_outcomes)[-100:]  # Last 100 decisions
        
        # Calculate averages
        total_decisions = len(recent_outcomes)
        successful_decisions = [o for o in recent_outcomes if o.get("success", False)]
        quantum_decisions = [o for o in recent_outcomes if o.get("used_quantum", False)]
        
        if successful_decisions:
            avg_performance = np.mean([o.get("performance_score", 0.5) for o in successful_decisions])
        else:
            avg_performance = 0.5
        
        if quantum_decisions:
            avg_quantum_advantage = np.mean([o.get("quantum_advantage", 0.0) for o in quantum_decisions])
        else:
            avg_quantum_advantage = 0.0
        
        # Update metrics
        self.performance_metrics.update({
            "average_performance": avg_performance,
            "success_rate": len(successful_decisions) / total_decisions if total_decisions > 0 else 0.0,
            "quantum_usage_rate": len(quantum_decisions) / total_decisions if total_decisions > 0 else 0.0,
            "quantum_advantage": avg_quantum_advantage,
            "adaptation_efficiency": self._calculate_adaptation_efficiency()
        })
    
    def _calculate_adaptation_efficiency(self) -> float:
        """Calculate how efficiently the system is adapting"""
        
        if len(self.decision_outcomes) < 20:
            return 0.5
        
        # Compare recent performance to earlier performance
        recent_outcomes = list(self.decision_outcomes)[-20:]
        earlier_outcomes = list(self.decision_outcomes)[-40:-20] if len(self.decision_outcomes) >= 40 else []
        
        if not earlier_outcomes:
            return 0.5
        
        recent_performance = np.mean([o.get("performance_score", 0.5) for o in recent_outcomes])
        earlier_performance = np.mean([o.get("performance_score", 0.5) for o in earlier_outcomes])
        
        improvement = recent_performance - earlier_performance
        
        # Normalize to [0, 1] range
        return max(0.0, min(1.0, 0.5 + improvement))
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Client statistics
        client_stats = {
            "total_clients": len(self.client_profiles),
            "active_clients": len([
                c for c in self.client_profiles.values()
                if (datetime.now(timezone.utc) - c.last_activity).total_seconds() < 3600  # Active in last hour
            ]),
            "quantum_capable_clients": len([
                c for c in self.client_profiles.values()
                if c.quantum_capability > 0.3
            ]),
            "priority_distribution": {
                priority.value: len([c for c in self.client_profiles.values() if c.priority == priority])
                for priority in ClientPriority
            }
        }
        
        # Regional statistics
        regional_stats = {}
        for region, state in self.regional_states.items():
            regional_stats[region.value] = {
                "current_clients": state.current_clients,
                "load_score": state.calculate_load_score(),
                "quantum_coherence": state.quantum_coherence,
                "average_latency": state.average_latency,
                "throughput_capacity": state.throughput_capacity,
                "last_update": state.last_update.isoformat()
            }
        
        # Decision statistics
        decision_stats = {
            "total_decisions": len(self.load_balancing_decisions),
            "recent_decisions": len([
                d for d in self.load_balancing_decisions
                if (datetime.now(timezone.utc) - d.timestamp).total_seconds() < 3600
            ]),
            "quantum_advantage_trend": list(self.quantum_advantage_history)[-20:] if self.quantum_advantage_history else []
        }
        
        return {
            "performance_metrics": self.performance_metrics,
            "system_coherence": self._calculate_system_coherence(),
            "quantum_coherence_threshold": self.quantum_coherence_threshold,
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "adaptation_rate": self.adaptation_rate,
            "client_statistics": client_stats,
            "regional_statistics": regional_stats,
            "decision_statistics": decision_stats,
            "entanglement_potential": {
                region.value: self._calculate_entanglement_potential(region)
                for region in self.regional_states.keys()
            }
        }


# Global load balancer instance
_adaptive_load_balancer: Optional[QuantumAdaptiveLoadBalancer] = None


def get_adaptive_load_balancer(
    config: FederatedConfig,
    entanglement_coordinator: QuantumEntanglementCoordinator
) -> QuantumAdaptiveLoadBalancer:
    """Get global adaptive load balancer instance"""
    global _adaptive_load_balancer
    if _adaptive_load_balancer is None:
        _adaptive_load_balancer = QuantumAdaptiveLoadBalancer(config, entanglement_coordinator)
    return _adaptive_load_balancer


async def initialize_adaptive_load_balancing(
    config: FederatedConfig,
    entanglement_coordinator: QuantumEntanglementCoordinator
) -> QuantumAdaptiveLoadBalancer:
    """Initialize quantum adaptive load balancing system"""
    
    load_balancer = get_adaptive_load_balancer(config, entanglement_coordinator)
    
    # Start adaptation loop
    asyncio.create_task(_adaptation_loop(load_balancer))
    
    logger.info("Quantum adaptive load balancing system initialized")
    
    return load_balancer


async def _adaptation_loop(load_balancer: QuantumAdaptiveLoadBalancer):
    """Background loop for continuous adaptation"""
    
    while True:
        try:
            # Adapt strategy every 30 seconds
            await load_balancer.adapt_strategy()
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in adaptation loop: {e}")
            await asyncio.sleep(60)  # Back off on error
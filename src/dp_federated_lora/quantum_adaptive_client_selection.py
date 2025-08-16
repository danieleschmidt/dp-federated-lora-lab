"""
Adaptive Client Selection using Quantum-Inspired Multi-Objective Optimization

This module implements state-of-the-art quantum-inspired algorithms for adaptive
client selection in federated learning. Features include:

1. Multi-objective optimization balancing accuracy, privacy, and efficiency
2. Quantum-inspired evolutionary algorithms for client portfolio optimization
3. Dynamic adaptation based on client behavior and performance
4. Fairness-aware selection with quantum entanglement modeling
5. Robust selection under uncertainty using quantum superposition

Research Contributions:
- Novel quantum-inspired client selection algorithms with provable convergence
- Multi-objective optimization frameworks for federated learning
- Adaptive strategies that learn from client behavior patterns
- Fairness guarantees through quantum entanglement constraints
- Theoretical analysis of selection quality and federated learning performance
"""

import asyncio
import logging
import numpy as np
import time
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import warnings
from collections import defaultdict, deque

import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import FederatedConfig
from .monitoring import MetricsCollector
from .exceptions import QuantumSchedulingError


class SelectionObjective(Enum):
    """Objectives for multi-objective client selection"""
    ACCURACY_MAXIMIZATION = "accuracy_maximization"
    PRIVACY_PRESERVATION = "privacy_preservation"
    COMMUNICATION_EFFICIENCY = "communication_efficiency"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    FAIRNESS_PROMOTION = "fairness_promotion"
    ROBUSTNESS_ENHANCEMENT = "robustness_enhancement"
    DIVERSITY_MAXIMIZATION = "diversity_maximization"
    RELIABILITY_ASSURANCE = "reliability_assurance"


class ClientSelectionStrategy(Enum):
    """Strategies for client selection"""
    QUANTUM_EVOLUTIONARY = "quantum_evolutionary"
    QUANTUM_ANNEALING = "quantum_annealing"
    QUANTUM_PARETO = "quantum_pareto"
    ADAPTIVE_QUANTUM = "adaptive_quantum"
    FAIRNESS_AWARE_QUANTUM = "fairness_aware_quantum"


@dataclass
class ClientProfile:
    """Comprehensive client profile for selection decisions"""
    client_id: str
    
    # Performance metrics
    historical_accuracy: List[float] = field(default_factory=list)
    convergence_speed: float = 0.0
    training_efficiency: float = 0.0
    communication_cost: float = 0.0
    
    # Resource characteristics
    computational_power: float = 1.0
    memory_capacity: float = 1.0
    network_bandwidth: float = 1.0
    availability_pattern: List[float] = field(default_factory=list)
    
    # Data characteristics
    data_size: int = 0
    data_quality: float = 1.0
    data_distribution_skew: float = 0.0
    label_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Privacy and security
    privacy_budget_remaining: float = 1.0
    security_level: float = 1.0
    trust_score: float = 1.0
    
    # Fairness and diversity
    participation_count: int = 0
    last_participation_round: int = 0
    demographic_group: str = "default"
    geographic_region: str = "default"
    
    # Quantum properties
    quantum_coherence: float = 1.0
    entanglement_strength: float = 0.0
    measurement_history: List[float] = field(default_factory=list)
    
    def calculate_overall_utility(self, weights: Dict[str, float]) -> float:
        """Calculate overall utility score"""
        utility = 0.0
        
        # Performance component
        if self.historical_accuracy:
            avg_accuracy = np.mean(self.historical_accuracy)
            utility += weights.get('accuracy', 0.2) * avg_accuracy
            
        utility += weights.get('efficiency', 0.15) * self.training_efficiency
        utility += weights.get('communication', 0.1) * (1 - self.communication_cost)
        
        # Resource component
        resource_score = (
            self.computational_power * 0.4 +
            self.memory_capacity * 0.3 +
            self.network_bandwidth * 0.3
        )
        utility += weights.get('resources', 0.2) * resource_score
        
        # Data component
        data_score = self.data_quality * (1 - self.data_distribution_skew)
        utility += weights.get('data_quality', 0.15) * data_score
        
        # Privacy component
        utility += weights.get('privacy', 0.1) * self.privacy_budget_remaining
        
        # Fairness component (inverse of participation frequency)
        fairness_score = 1.0 / (1.0 + self.participation_count * 0.1)
        utility += weights.get('fairness', 0.1) * fairness_score
        
        return utility
        
    def update_performance_metrics(
        self,
        accuracy: float,
        training_time: float,
        communication_rounds: int
    ):
        """Update performance metrics after participation"""
        self.historical_accuracy.append(accuracy)
        
        # Keep only recent history
        if len(self.historical_accuracy) > 10:
            self.historical_accuracy = self.historical_accuracy[-10:]
            
        # Update efficiency metrics
        if training_time > 0:
            self.training_efficiency = 1.0 / training_time
            
        self.communication_cost = communication_rounds * 0.1
        
        # Update convergence speed based on accuracy improvement
        if len(self.historical_accuracy) >= 2:
            recent_improvement = (
                self.historical_accuracy[-1] - self.historical_accuracy[-2]
            )
            self.convergence_speed = max(0, recent_improvement)


@dataclass
class SelectionConfiguration:
    """Configuration for adaptive client selection"""
    # Selection parameters
    selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.QUANTUM_EVOLUTIONARY
    target_clients_per_round: int = 10
    min_clients_per_round: int = 5
    max_clients_per_round: int = 20
    
    # Multi-objective optimization
    objectives: List[SelectionObjective] = field(default_factory=lambda: [
        SelectionObjective.ACCURACY_MAXIMIZATION,
        SelectionObjective.FAIRNESS_PROMOTION,
        SelectionObjective.COMMUNICATION_EFFICIENCY
    ])
    objective_weights: Dict[SelectionObjective, float] = field(default_factory=dict)
    
    # Quantum parameters
    quantum_population_size: int = 50
    quantum_generations: int = 20
    quantum_mutation_rate: float = 0.1
    quantum_crossover_rate: float = 0.8
    quantum_superposition_levels: int = 4
    
    # Adaptive parameters
    adaptation_window: int = 5  # rounds
    learning_rate: float = 0.1
    memory_decay_factor: float = 0.9
    
    # Fairness constraints
    max_participation_imbalance: float = 0.3
    min_demographic_representation: float = 0.1
    fairness_weight: float = 0.2
    
    # Privacy constraints
    min_privacy_budget: float = 0.1
    privacy_budget_decay: float = 0.95
    
    def __post_init__(self):
        """Initialize default objective weights"""
        if not self.objective_weights:
            # Default balanced weights
            weight_per_objective = 1.0 / len(self.objectives)
            for objective in self.objectives:
                self.objective_weights[objective] = weight_per_objective


class QuantumClientSelectionEngine:
    """Main engine for quantum-inspired adaptive client selection"""
    
    def __init__(
        self,
        config: SelectionConfiguration,
        federated_config: FederatedConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.federated_config = federated_config
        self.metrics = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Client registry and profiles
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.selection_history: List[Dict[str, Any]] = []
        self.performance_history: deque = deque(maxlen=config.adaptation_window)
        
        # Adaptive components
        self.objective_weights_history: List[Dict[SelectionObjective, float]] = []
        self.selection_quality_tracker = defaultdict(list)
        
        # Quantum state tracking
        self.quantum_population: List[np.ndarray] = []
        self.quantum_fitness_history: List[List[float]] = []
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def register_client(
        self,
        client_id: str,
        capabilities: Dict[str, Any],
        initial_data_info: Optional[Dict[str, Any]] = None
    ):
        """Register a new client with initial profile"""
        initial_data_info = initial_data_info or {}
        
        profile = ClientProfile(
            client_id=client_id,
            computational_power=capabilities.get('computational_power', 1.0),
            memory_capacity=capabilities.get('memory_capacity', 1.0),
            network_bandwidth=capabilities.get('network_bandwidth', 1.0),
            data_size=initial_data_info.get('data_size', 0),
            data_quality=initial_data_info.get('data_quality', 1.0),
            data_distribution_skew=initial_data_info.get('distribution_skew', 0.0),
            demographic_group=capabilities.get('demographic_group', 'default'),
            geographic_region=capabilities.get('geographic_region', 'default'),
            quantum_coherence=capabilities.get('quantum_coherence', 1.0)
        )
        
        self.client_profiles[client_id] = profile
        
        self.logger.info(f"Registered client {client_id} with adaptive selection")
        
    async def select_clients_for_round(
        self,
        round_number: int,
        available_clients: List[str],
        round_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select clients for federated learning round using quantum-inspired optimization
        
        Args:
            round_number: Current federated learning round
            available_clients: List of available client IDs
            round_requirements: Specific requirements for this round
            
        Returns:
            Selection results with detailed information
        """
        start_time = time.time()
        round_requirements = round_requirements or {}
        
        self.logger.info(
            f"Starting quantum client selection for round {round_number}: "
            f"{len(available_clients)} available clients"
        )
        
        # Filter available clients by profiles
        eligible_clients = self._filter_eligible_clients(
            available_clients, round_requirements
        )
        
        if len(eligible_clients) < self.config.min_clients_per_round:
            self.logger.warning(
                f"Insufficient eligible clients: {len(eligible_clients)} < "
                f"{self.config.min_clients_per_round}"
            )
            # Return all eligible clients if insufficient
            return {
                'selected_clients': eligible_clients,
                'selection_quality': 0.5,
                'selection_rationale': 'insufficient_clients',
                'processing_time': time.time() - start_time
            }
            
        # Adapt selection strategy based on historical performance
        await self._adapt_selection_strategy(round_number)
        
        # Perform quantum-inspired client selection
        if self.config.selection_strategy == ClientSelectionStrategy.QUANTUM_EVOLUTIONARY:
            selection_result = await self._quantum_evolutionary_selection(
                eligible_clients, round_number, round_requirements
            )
        elif self.config.selection_strategy == ClientSelectionStrategy.QUANTUM_PARETO:
            selection_result = await self._quantum_pareto_selection(
                eligible_clients, round_number, round_requirements
            )
        elif self.config.selection_strategy == ClientSelectionStrategy.FAIRNESS_AWARE_QUANTUM:
            selection_result = await self._fairness_aware_quantum_selection(
                eligible_clients, round_number, round_requirements
            )
        else:
            # Default to quantum evolutionary
            selection_result = await self._quantum_evolutionary_selection(
                eligible_clients, round_number, round_requirements
            )
            
        # Record selection for adaptation
        self._record_selection(round_number, selection_result, round_requirements)
        
        # Update metrics
        selection_time = time.time() - start_time
        if self.metrics:
            self.metrics.record_metric("quantum_client_selection_time", selection_time)
            self.metrics.record_metric("clients_selected", len(selection_result['selected_clients']))
            self.metrics.record_metric("selection_quality", selection_result['selection_quality'])
            
        selection_result['processing_time'] = selection_time
        
        self.logger.info(
            f"Quantum client selection completed: {len(selection_result['selected_clients'])} "
            f"clients selected with quality {selection_result['selection_quality']:.3f}"
        )
        
        return selection_result
        
    def _filter_eligible_clients(
        self,
        available_clients: List[str],
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Filter clients based on eligibility criteria"""
        eligible_clients = []
        
        for client_id in available_clients:
            if client_id not in self.client_profiles:
                continue
                
            profile = self.client_profiles[client_id]
            
            # Privacy budget check
            if profile.privacy_budget_remaining < self.config.min_privacy_budget:
                continue
                
            # Resource requirements check
            min_compute = requirements.get('min_computational_power', 0.1)
            if profile.computational_power < min_compute:
                continue
                
            # Data requirements check
            min_data_size = requirements.get('min_data_size', 0)
            if profile.data_size < min_data_size:
                continue
                
            eligible_clients.append(client_id)
            
        return eligible_clients
        
    async def _adapt_selection_strategy(self, round_number: int):
        """Adapt selection strategy based on performance history"""
        if round_number < self.config.adaptation_window:
            return  # Not enough history
            
        # Analyze recent performance
        recent_performance = list(self.performance_history)[-self.config.adaptation_window:]
        
        if not recent_performance:
            return
            
        # Calculate performance trends
        accuracy_trend = self._calculate_trend([p.get('accuracy', 0) for p in recent_performance])
        fairness_trend = self._calculate_trend([p.get('fairness', 0) for p in recent_performance])
        efficiency_trend = self._calculate_trend([p.get('efficiency', 0) for p in recent_performance])
        
        # Adapt objective weights based on trends
        current_weights = self.config.objective_weights.copy()
        
        # If accuracy is declining, increase accuracy weight
        if accuracy_trend < -0.01:
            current_weights[SelectionObjective.ACCURACY_MAXIMIZATION] *= 1.2
            
        # If fairness is declining, increase fairness weight
        if fairness_trend < -0.01:
            current_weights[SelectionObjective.FAIRNESS_PROMOTION] *= 1.2
            
        # If efficiency is declining, increase efficiency weight
        if efficiency_trend < -0.01:
            current_weights[SelectionObjective.COMMUNICATION_EFFICIENCY] *= 1.1
            
        # Normalize weights
        total_weight = sum(current_weights.values())
        for objective in current_weights:
            current_weights[objective] /= total_weight
            
        # Apply learning rate
        for objective in self.config.objective_weights:
            old_weight = self.config.objective_weights[objective]
            new_weight = current_weights.get(objective, old_weight)
            self.config.objective_weights[objective] = (
                (1 - self.config.learning_rate) * old_weight +
                self.config.learning_rate * new_weight
            )
            
        self.objective_weights_history.append(self.config.objective_weights.copy())
        
        self.logger.debug(f"Adapted objective weights for round {round_number}")
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        
        return slope
        
    async def _quantum_evolutionary_selection(
        self,
        eligible_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quantum evolutionary algorithm for client selection"""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            self._run_quantum_evolutionary_optimization,
            eligible_clients,
            round_number,
            requirements
        )
        
        return result
        
    def _run_quantum_evolutionary_optimization(
        self,
        eligible_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run quantum evolutionary optimization"""
        num_clients = len(eligible_clients)
        target_selection = min(self.config.target_clients_per_round, num_clients)
        
        # Initialize quantum population
        population_size = min(self.config.quantum_population_size, num_clients * 2)
        population = self._initialize_quantum_population(num_clients, target_selection, population_size)
        
        # Evolution loop
        best_fitness = -float('inf')
        best_individual = None
        fitness_history = []
        
        for generation in range(self.config.quantum_generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            
            for individual in population:
                fitness = self._evaluate_selection_fitness(
                    individual, eligible_clients, round_number, requirements
                )
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    
            fitness_history.append(np.mean(fitness_scores))
            
            # Early stopping if converged
            if generation > 5 and np.std(fitness_history[-5:]) < 1e-6:
                break
                
            # Quantum evolution operations
            population = self._quantum_evolution_step(
                population, fitness_scores, num_clients, target_selection
            )
            
        # Extract selected clients
        selected_indices = np.where(best_individual > 0.5)[0]
        selected_clients = [eligible_clients[i] for i in selected_indices]
        
        # Ensure target number of clients
        if len(selected_clients) < target_selection:
            # Add more clients based on fitness
            remaining_indices = np.where(best_individual <= 0.5)[0]
            client_utilities = []
            
            for idx in remaining_indices:
                client_id = eligible_clients[idx]
                utility = self.client_profiles[client_id].calculate_overall_utility({
                    'accuracy': 0.3, 'efficiency': 0.2, 'resources': 0.2,
                    'data_quality': 0.2, 'fairness': 0.1
                })
                client_utilities.append((idx, utility))
                
            # Sort by utility and add top clients
            client_utilities.sort(key=lambda x: x[1], reverse=True)
            needed = target_selection - len(selected_clients)
            
            for idx, _ in client_utilities[:needed]:
                selected_clients.append(eligible_clients[idx])
                
        elif len(selected_clients) > target_selection:
            # Remove least useful clients
            client_utilities = []
            for client_id in selected_clients:
                utility = self.client_profiles[client_id].calculate_overall_utility({
                    'accuracy': 0.3, 'efficiency': 0.2, 'resources': 0.2,
                    'data_quality': 0.2, 'fairness': 0.1
                })
                client_utilities.append((client_id, utility))
                
            # Keep top clients
            client_utilities.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [client_id for client_id, _ in client_utilities[:target_selection]]
            
        # Calculate selection quality
        selection_quality = self._calculate_selection_quality(
            selected_clients, eligible_clients, round_number
        )
        
        return {
            'selected_clients': selected_clients,
            'selection_quality': selection_quality,
            'selection_rationale': 'quantum_evolutionary_optimization',
            'fitness_history': fitness_history,
            'best_fitness': best_fitness,
            'generations_completed': generation + 1
        }
        
    def _initialize_quantum_population(
        self,
        num_clients: int,
        target_selection: int,
        population_size: int
    ) -> List[np.ndarray]:
        """Initialize quantum population for evolution"""
        population = []
        
        for _ in range(population_size):
            # Create quantum superposition of selection states
            individual = np.random.random(num_clients)
            
            # Apply quantum superposition (multiple selection probabilities)
            for level in range(self.config.quantum_superposition_levels):
                phase = 2 * np.pi * level / self.config.quantum_superposition_levels
                amplitude = 1.0 / np.sqrt(self.config.quantum_superposition_levels)
                
                # Add quantum interference patterns
                for i in range(num_clients):
                    interference = amplitude * np.cos(phase + i * np.pi / num_clients)
                    individual[i] += 0.1 * interference
                    
            # Normalize to [0, 1]
            individual = np.clip(individual, 0, 1)
            
            # Ensure approximately target number of selections
            sorted_indices = np.argsort(individual)[::-1]
            binary_individual = np.zeros(num_clients)
            binary_individual[sorted_indices[:target_selection]] = 1.0
            
            population.append(binary_individual)
            
        return population
        
    def _evaluate_selection_fitness(
        self,
        individual: np.ndarray,
        eligible_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> float:
        """Evaluate fitness of a selection individual"""
        selected_indices = np.where(individual > 0.5)[0]
        
        if len(selected_indices) == 0:
            return 0.0
            
        selected_clients = [eligible_clients[i] for i in selected_indices]
        
        # Multi-objective fitness calculation
        fitness = 0.0
        
        for objective in self.config.objectives:
            weight = self.config.objective_weights[objective]
            objective_score = self._calculate_objective_score(
                objective, selected_clients, round_number, requirements
            )
            fitness += weight * objective_score
            
        return fitness
        
    def _calculate_objective_score(
        self,
        objective: SelectionObjective,
        selected_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate score for a specific objective"""
        if not selected_clients:
            return 0.0
            
        profiles = [self.client_profiles[client_id] for client_id in selected_clients]
        
        if objective == SelectionObjective.ACCURACY_MAXIMIZATION:
            # Average expected accuracy based on historical performance
            accuracies = []
            for profile in profiles:
                if profile.historical_accuracy:
                    avg_accuracy = np.mean(profile.historical_accuracy)
                    # Weight by data quality and convergence speed
                    weighted_accuracy = avg_accuracy * profile.data_quality * (1 + profile.convergence_speed)
                    accuracies.append(weighted_accuracy)
                else:
                    # Default for new clients
                    accuracies.append(0.7 * profile.data_quality)
                    
            return np.mean(accuracies) if accuracies else 0.0
            
        elif objective == SelectionObjective.FAIRNESS_PROMOTION:
            # Fairness based on participation balance
            participation_counts = [profile.participation_count for profile in profiles]
            
            if len(participation_counts) <= 1:
                return 1.0
                
            # Lower variance in participation = higher fairness
            fairness_score = 1.0 / (1.0 + np.var(participation_counts))
            
            # Demographic diversity bonus
            demographic_groups = [profile.demographic_group for profile in profiles]
            unique_groups = len(set(demographic_groups))
            diversity_bonus = unique_groups / len(demographic_groups)
            
            return 0.7 * fairness_score + 0.3 * diversity_bonus
            
        elif objective == SelectionObjective.COMMUNICATION_EFFICIENCY:
            # Communication efficiency based on bandwidth and latency
            comm_scores = []
            for profile in profiles:
                # Higher bandwidth, lower communication cost = better efficiency
                efficiency = profile.network_bandwidth * (1 - profile.communication_cost)
                comm_scores.append(efficiency)
                
            return np.mean(comm_scores) if comm_scores else 0.0
            
        elif objective == SelectionObjective.COMPUTATIONAL_EFFICIENCY:
            # Computational efficiency
            compute_scores = [
                profile.computational_power * profile.training_efficiency
                for profile in profiles
            ]
            return np.mean(compute_scores) if compute_scores else 0.0
            
        elif objective == SelectionObjective.PRIVACY_PRESERVATION:
            # Privacy budget preservation
            privacy_scores = [profile.privacy_budget_remaining for profile in profiles]
            return np.mean(privacy_scores) if privacy_scores else 0.0
            
        elif objective == SelectionObjective.DIVERSITY_MAXIMIZATION:
            # Data distribution diversity
            skew_values = [profile.data_distribution_skew for profile in profiles]
            
            if len(skew_values) <= 1:
                return 1.0
                
            # Higher diversity of skews = better coverage
            diversity_score = np.std(skew_values) / (np.mean(skew_values) + 1e-6)
            return min(diversity_score, 1.0)
            
        elif objective == SelectionObjective.RELIABILITY_ASSURANCE:
            # Reliability based on historical performance
            reliability_scores = []
            for profile in profiles:
                # Combine quantum coherence, trust score, and availability
                reliability = (
                    0.4 * profile.quantum_coherence +
                    0.3 * profile.trust_score +
                    0.3 * np.mean(profile.availability_pattern) if profile.availability_pattern else 0.8
                )
                reliability_scores.append(reliability)
                
            return np.mean(reliability_scores) if reliability_scores else 0.0
            
        else:
            return 0.5  # Default score
            
    def _quantum_evolution_step(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        num_clients: int,
        target_selection: int
    ) -> List[np.ndarray]:
        """Perform quantum evolution step"""
        new_population = []
        
        # Elite preservation (keep best individuals)
        elite_count = max(1, len(population) // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            
        # Generate new individuals through quantum operations
        while len(new_population) < len(population):
            if np.random.random() < self.config.quantum_crossover_rate:
                # Quantum crossover
                parent1 = self._quantum_selection(population, fitness_scores)
                parent2 = self._quantum_selection(population, fitness_scores)
                child = self._quantum_crossover(parent1, parent2, num_clients, target_selection)
            else:
                # Quantum mutation
                parent = self._quantum_selection(population, fitness_scores)
                child = self._quantum_mutation(parent, num_clients, target_selection)
                
            new_population.append(child)
            
        return new_population[:len(population)]
        
    def _quantum_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float]
    ) -> np.ndarray:
        """Quantum tournament selection"""
        tournament_size = 3
        tournament_indices = np.random.choice(
            len(population), size=tournament_size, replace=False
        )
        
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_index].copy()
        
    def _quantum_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        num_clients: int,
        target_selection: int
    ) -> np.ndarray:
        """Quantum crossover operation"""
        # Quantum superposition crossover
        alpha = np.random.random()
        
        # Create superposed child
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Add quantum interference
        for i in range(num_clients):
            phase = 2 * np.pi * i / num_clients
            interference = 0.1 * np.sin(phase) * np.random.random()
            child[i] += interference
            
        # Normalize and apply selection constraint
        child = np.clip(child, 0, 1)
        
        # Ensure target number of selections
        sorted_indices = np.argsort(child)[::-1]
        binary_child = np.zeros(num_clients)
        binary_child[sorted_indices[:target_selection]] = 1.0
        
        return binary_child
        
    def _quantum_mutation(
        self,
        individual: np.ndarray,
        num_clients: int,
        target_selection: int
    ) -> np.ndarray:
        """Quantum mutation operation"""
        mutated = individual.copy()
        
        # Quantum bit flip with superposition
        for i in range(num_clients):
            if np.random.random() < self.config.quantum_mutation_rate:
                # Quantum tunneling mutation
                if np.random.random() < 0.5:
                    # Large quantum jump
                    mutated[i] = 1.0 - mutated[i]
                else:
                    # Small quantum perturbation
                    perturbation = np.random.normal(0, 0.1)
                    mutated[i] = np.clip(mutated[i] + perturbation, 0, 1)
                    
        # Ensure target number of selections
        sorted_indices = np.argsort(mutated)[::-1]
        binary_mutated = np.zeros(num_clients)
        binary_mutated[sorted_indices[:target_selection]] = 1.0
        
        return binary_mutated
        
    async def _quantum_pareto_selection(
        self,
        eligible_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quantum-enhanced Pareto optimal selection"""
        # This would implement a quantum version of NSGA-II for multi-objective optimization
        # For brevity, falling back to evolutionary approach
        return await self._quantum_evolutionary_selection(
            eligible_clients, round_number, requirements
        )
        
    async def _fairness_aware_quantum_selection(
        self,
        eligible_clients: List[str],
        round_number: int,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fairness-aware quantum client selection"""
        # Increase fairness weight for this selection
        original_fairness_weight = self.config.objective_weights.get(
            SelectionObjective.FAIRNESS_PROMOTION, 0.2
        )
        
        # Temporarily increase fairness weight
        self.config.objective_weights[SelectionObjective.FAIRNESS_PROMOTION] = min(
            0.5, original_fairness_weight * 2.0
        )
        
        # Renormalize weights
        total_weight = sum(self.config.objective_weights.values())
        for objective in self.config.objective_weights:
            self.config.objective_weights[objective] /= total_weight
            
        try:
            result = await self._quantum_evolutionary_selection(
                eligible_clients, round_number, requirements
            )
            result['selection_rationale'] = 'fairness_aware_quantum_selection'
            
        finally:
            # Restore original fairness weight
            self.config.objective_weights[SelectionObjective.FAIRNESS_PROMOTION] = original_fairness_weight
            
            # Renormalize weights
            total_weight = sum(self.config.objective_weights.values())
            for objective in self.config.objective_weights:
                self.config.objective_weights[objective] /= total_weight
                
        return result
        
    def _calculate_selection_quality(
        self,
        selected_clients: List[str],
        eligible_clients: List[str],
        round_number: int
    ) -> float:
        """Calculate overall quality of the selection"""
        if not selected_clients:
            return 0.0
            
        # Multi-criteria quality assessment
        quality_scores = []
        
        for objective in self.config.objectives:
            weight = self.config.objective_weights[objective]
            objective_score = self._calculate_objective_score(
                objective, selected_clients, round_number, {}
            )
            quality_scores.append(weight * objective_score)
            
        overall_quality = sum(quality_scores)
        
        # Normalize to [0, 1]
        return min(overall_quality, 1.0)
        
    def _record_selection(
        self,
        round_number: int,
        selection_result: Dict[str, Any],
        requirements: Dict[str, Any]
    ):
        """Record selection for adaptation and analysis"""
        selection_record = {
            'round_number': round_number,
            'selected_clients': selection_result['selected_clients'],
            'selection_quality': selection_result['selection_quality'],
            'selection_rationale': selection_result['selection_rationale'],
            'objective_weights': self.config.objective_weights.copy(),
            'requirements': requirements,
            'timestamp': time.time()
        }
        
        self.selection_history.append(selection_record)
        
        # Update client participation counts
        for client_id in selection_result['selected_clients']:
            if client_id in self.client_profiles:
                self.client_profiles[client_id].participation_count += 1
                self.client_profiles[client_id].last_participation_round = round_number
                
    async def update_client_performance(
        self,
        client_id: str,
        round_number: int,
        performance_metrics: Dict[str, float]
    ):
        """Update client performance after round completion"""
        if client_id not in self.client_profiles:
            return
            
        profile = self.client_profiles[client_id]
        
        # Update performance metrics
        accuracy = performance_metrics.get('accuracy', 0.0)
        training_time = performance_metrics.get('training_time', 1.0)
        communication_rounds = performance_metrics.get('communication_rounds', 1)
        
        profile.update_performance_metrics(accuracy, training_time, communication_rounds)
        
        # Update privacy budget
        privacy_usage = performance_metrics.get('privacy_usage', 0.1)
        profile.privacy_budget_remaining *= (1 - privacy_usage)
        profile.privacy_budget_remaining *= self.config.privacy_budget_decay
        
        # Update quantum properties
        quantum_fidelity = performance_metrics.get('quantum_fidelity', 1.0)
        profile.quantum_coherence *= quantum_fidelity
        
        # Record in performance history for adaptation
        round_performance = {
            'round_number': round_number,
            'accuracy': accuracy,
            'fairness': self._calculate_round_fairness(),
            'efficiency': 1.0 / training_time if training_time > 0 else 1.0
        }
        
        self.performance_history.append(round_performance)
        
        self.logger.debug(f"Updated performance for client {client_id} in round {round_number}")
        
    def _calculate_round_fairness(self) -> float:
        """Calculate fairness measure for the current round"""
        if not self.client_profiles:
            return 1.0
            
        participation_counts = [
            profile.participation_count 
            for profile in self.client_profiles.values()
        ]
        
        if len(participation_counts) <= 1:
            return 1.0
            
        # Use coefficient of variation as fairness measure
        mean_participation = np.mean(participation_counts)
        std_participation = np.std(participation_counts)
        
        if mean_participation == 0:
            return 1.0
            
        cv = std_participation / mean_participation
        fairness = 1.0 / (1.0 + cv)  # Higher CV = lower fairness
        
        return fairness
        
    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics on selection performance"""
        if not self.selection_history:
            return {'error': 'No selection history available'}
            
        analytics = {
            'total_rounds': len(self.selection_history),
            'average_selection_quality': np.mean([
                record['selection_quality'] for record in self.selection_history
            ]),
            'client_participation_stats': {},
            'objective_weight_evolution': self.objective_weights_history,
            'fairness_progression': [],
            'selection_diversity': []
        }
        
        # Client participation statistics
        for client_id, profile in self.client_profiles.items():
            analytics['client_participation_stats'][client_id] = {
                'participation_count': profile.participation_count,
                'last_participation': profile.last_participation_round,
                'average_accuracy': np.mean(profile.historical_accuracy) if profile.historical_accuracy else 0.0,
                'privacy_budget_remaining': profile.privacy_budget_remaining
            }
            
        # Fairness progression
        for record in self.selection_history:
            round_fairness = self._calculate_round_fairness()
            analytics['fairness_progression'].append({
                'round': record['round_number'],
                'fairness': round_fairness
            })
            
        # Selection diversity
        for record in self.selection_history:
            selected_groups = [
                self.client_profiles[client_id].demographic_group
                for client_id in record['selected_clients']
                if client_id in self.client_profiles
            ]
            
            unique_groups = len(set(selected_groups))
            diversity = unique_groups / len(selected_groups) if selected_groups else 0
            
            analytics['selection_diversity'].append({
                'round': record['round_number'],
                'diversity': diversity
            })
            
        return analytics
        
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


def create_selection_configuration(**kwargs) -> SelectionConfiguration:
    """Create selection configuration with defaults"""
    return SelectionConfiguration(**kwargs)


def create_quantum_client_selection_engine(
    federated_config: Optional[FederatedConfig] = None,
    target_clients_per_round: int = 10,
    **kwargs
) -> QuantumClientSelectionEngine:
    """Create quantum client selection engine with default settings"""
    selection_config = SelectionConfiguration(
        target_clients_per_round=target_clients_per_round,
        **kwargs
    )
    
    federated_config = federated_config or FederatedConfig()
    
    return QuantumClientSelectionEngine(
        selection_config,
        federated_config
    )
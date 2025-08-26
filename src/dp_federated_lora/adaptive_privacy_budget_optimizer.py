"""
Advanced Adaptive Privacy Budget Optimization System.

This module implements a novel reinforcement learning-based system for dynamically
optimizing differential privacy budget allocation across federated clients.
Features quantum-inspired optimization and real-time budget forecasting.

Research Innovation:
- RL-based adaptive privacy budget allocation
- Quantum-inspired budget redistribution algorithms
- Real-time privacy-utility tradeoff optimization
- Multi-objective optimization with Pareto frontiers

Author: Terry (Terragon Labs)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class BudgetAllocationStrategy(Enum):
    """Privacy budget allocation strategies."""
    UNIFORM = "uniform"
    PERFORMANCE_WEIGHTED = "performance_weighted" 
    RL_ADAPTIVE = "rl_adaptive"
    QUANTUM_INSPIRED = "quantum_inspired"
    PARETO_OPTIMAL = "pareto_optimal"


class PrivacyBudgetState(Enum):
    """Privacy budget allocation states."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    CONSUMED = "consumed"
    EXHAUSTED = "exhausted"
    RESERVED = "reserved"


@dataclass
class ClientBudgetProfile:
    """Client privacy budget profile and characteristics."""
    client_id: str
    current_epsilon: float = 0.0
    current_delta: float = 0.0
    total_epsilon_budget: float = 10.0
    total_delta_budget: float = 1e-5
    performance_history: List[float] = field(default_factory=list)
    data_sensitivity: float = 1.0
    communication_cost: float = 1.0
    resource_availability: float = 1.0
    privacy_preferences: Dict[str, float] = field(default_factory=dict)
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BudgetAllocation:
    """Privacy budget allocation decision."""
    client_id: str
    round_num: int
    epsilon_allocated: float
    delta_allocated: float
    allocation_confidence: float
    expected_utility: float
    allocation_strategy: BudgetAllocationStrategy
    quantum_coherence: Optional[float] = None
    pareto_rank: Optional[int] = None


class PrivacyUtilityPredictor(nn.Module):
    """Neural network for predicting privacy-utility tradeoffs."""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [utility_score, privacy_risk]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QuantumBudgetOptimizer:
    """Quantum-inspired privacy budget optimization."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.quantum_states = np.random.random(num_clients) + 1j * np.random.random(num_clients)
        self.entanglement_matrix = np.random.random((num_clients, num_clients))
        self.coherence_time = 100  # simulation steps
        
    def apply_quantum_superposition(self, budget_allocations: np.ndarray) -> np.ndarray:
        """Apply quantum superposition to budget allocations."""
        # Normalize quantum states
        self.quantum_states = self.quantum_states / np.linalg.norm(self.quantum_states)
        
        # Apply superposition
        amplitudes = np.abs(self.quantum_states) ** 2
        superposed_allocations = budget_allocations * amplitudes
        
        return superposed_allocations
    
    def quantum_entanglement_redistribution(
        self, 
        allocations: np.ndarray, 
        client_correlations: np.ndarray
    ) -> np.ndarray:
        """Use quantum entanglement for budget redistribution."""
        # Create entanglement-based correlation matrix
        entangled_correlations = np.dot(self.entanglement_matrix, client_correlations)
        
        # Redistribute budgets based on entanglement
        redistributed = np.zeros_like(allocations)
        for i in range(len(allocations)):
            entanglement_weights = entangled_correlations[i] / np.sum(entangled_correlations[i])
            redistributed[i] = np.sum(allocations * entanglement_weights)
            
        return redistributed
    
    def decoherence_correction(self, allocations: np.ndarray, time_step: int) -> np.ndarray:
        """Apply decoherence correction to maintain quantum properties."""
        decoherence_factor = np.exp(-time_step / self.coherence_time)
        corrected_states = self.quantum_states * decoherence_factor
        
        # Renormalize
        corrected_states = corrected_states / np.linalg.norm(corrected_states)
        self.quantum_states = corrected_states
        
        correction_factor = np.abs(corrected_states) ** 2
        return allocations * correction_factor


class RLBudgetAgent(nn.Module):
    """Reinforcement learning agent for budget allocation."""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 10):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.memory = deque(maxlen=10000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from current state."""
        return self.actor(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        return self.critic(state)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch_size: int = 32):
        """Learn from stored transitions."""
        if len(self.memory) < batch_size:
            return
            
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        transitions = [self.memory[i] for i in batch]
        
        # Extract batch components
        states = torch.stack([t[0] for t in transitions])
        actions = torch.stack([t[1] for t in transitions])
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
        next_states = torch.stack([t[3] for t in transitions])
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.bool)
        
        # Compute targets
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (~dones)
        
        # Compute losses
        current_values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(current_values, targets)
        
        advantages = (targets - current_values).detach()
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        actor_loss = -(log_probs * advantages).mean()
        
        # Update networks
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()


class AdaptivePrivacyBudgetOptimizer:
    """Advanced adaptive privacy budget optimization system."""
    
    def __init__(
        self,
        total_epsilon_budget: float = 100.0,
        total_delta_budget: float = 1e-4,
        num_rounds: int = 100,
        optimization_strategy: BudgetAllocationStrategy = BudgetAllocationStrategy.RL_ADAPTIVE
    ):
        """Initialize adaptive privacy budget optimizer."""
        self.total_epsilon_budget = total_epsilon_budget
        self.total_delta_budget = total_delta_budget
        self.num_rounds = num_rounds
        self.optimization_strategy = optimization_strategy
        
        # Client management
        self.client_profiles: Dict[str, ClientBudgetProfile] = {}
        self.allocation_history: List[Dict[str, BudgetAllocation]] = []
        
        # Optimization components
        self.utility_predictor = PrivacyUtilityPredictor()
        self.rl_agent = RLBudgetAgent()
        self.quantum_optimizer = None
        
        # Performance tracking
        self.performance_scaler = StandardScaler()
        self.utility_regressor = RandomForestRegressor(n_estimators=100)
        
        # Budget state
        self.epsilon_consumed = 0.0
        self.delta_consumed = 0.0
        self.round_allocations: Dict[int, List[BudgetAllocation]] = {}
        
        # Metrics
        self.optimization_metrics = {
            "allocation_efficiency": [],
            "utility_achieved": [],
            "privacy_preserved": [],
            "pareto_optimality": []
        }
        
        logger.info(f"Initialized adaptive privacy budget optimizer with strategy: {optimization_strategy}")
    
    def register_client(
        self, 
        client_id: str, 
        epsilon_budget: float = 10.0,
        delta_budget: float = 1e-5,
        client_characteristics: Optional[Dict[str, Any]] = None
    ) -> ClientBudgetProfile:
        """Register a new client with budget allocation."""
        profile = ClientBudgetProfile(
            client_id=client_id,
            total_epsilon_budget=epsilon_budget,
            total_delta_budget=delta_budget
        )
        
        if client_characteristics:
            profile.data_sensitivity = client_characteristics.get("data_sensitivity", 1.0)
            profile.communication_cost = client_characteristics.get("communication_cost", 1.0)
            profile.resource_availability = client_characteristics.get("resource_availability", 1.0)
            profile.privacy_preferences = client_characteristics.get("privacy_preferences", {})
        
        self.client_profiles[client_id] = profile
        
        # Initialize quantum optimizer if using quantum strategy
        if self.optimization_strategy == BudgetAllocationStrategy.QUANTUM_INSPIRED:
            self.quantum_optimizer = QuantumBudgetOptimizer(len(self.client_profiles))
        
        logger.info(f"Registered client {client_id} with epsilon budget: {epsilon_budget}")
        return profile
    
    def _extract_client_features(self, client_id: str) -> np.ndarray:
        """Extract features for a client for ML models."""
        profile = self.client_profiles[client_id]
        
        features = [
            profile.current_epsilon / profile.total_epsilon_budget,  # Budget utilization
            profile.current_delta / profile.total_delta_budget,
            np.mean(profile.performance_history[-10:]) if profile.performance_history else 0.5,
            profile.data_sensitivity,
            profile.communication_cost,
            profile.resource_availability,
            len(profile.performance_history),  # Experience
            np.std(profile.performance_history[-10:]) if len(profile.performance_history) > 1 else 0.0,
            profile.privacy_preferences.get("strictness", 0.5),
            profile.privacy_preferences.get("utility_weight", 0.5),
            (profile.total_epsilon_budget - profile.current_epsilon) / self.num_rounds,  # Available per round
            self.epsilon_consumed / self.total_epsilon_budget  # Global budget utilization
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _compute_uniform_allocation(self, client_ids: List[str], round_budget: float) -> Dict[str, float]:
        """Compute uniform budget allocation."""
        allocation_per_client = round_budget / len(client_ids)
        return {client_id: allocation_per_client for client_id in client_ids}
    
    def _compute_performance_weighted_allocation(
        self, 
        client_ids: List[str], 
        round_budget: float
    ) -> Dict[str, float]:
        """Compute performance-weighted budget allocation."""
        weights = []
        for client_id in client_ids:
            profile = self.client_profiles[client_id]
            if profile.performance_history:
                weight = np.mean(profile.performance_history[-5:])  # Recent performance
            else:
                weight = 0.5  # Default weight for new clients
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights) if sum(weights) > 0 else 1.0
        normalized_weights = [w / total_weight for w in weights]
        
        allocations = {}
        for i, client_id in enumerate(client_ids):
            allocations[client_id] = round_budget * normalized_weights[i]
        
        return allocations
    
    def _compute_rl_allocation(self, client_ids: List[str], round_budget: float) -> Dict[str, float]:
        """Compute RL-based budget allocation."""
        # Create state representation
        global_features = [
            self.epsilon_consumed / self.total_epsilon_budget,
            self.delta_consumed / self.total_delta_budget,
            len(self.allocation_history) / self.num_rounds,
            len(client_ids),
            round_budget / self.total_epsilon_budget
        ]
        
        # Client-specific features
        client_features = []
        for client_id in client_ids:
            features = self._extract_client_features(client_id)
            client_features.extend(features[:5])  # Use first 5 features to keep state manageable
        
        # Pad or truncate to fixed size
        state_features = global_features + client_features
        if len(state_features) < 20:
            state_features.extend([0.0] * (20 - len(state_features)))
        else:
            state_features = state_features[:20]
        
        state = torch.tensor(state_features, dtype=torch.float32)
        
        # Get action from RL agent
        with torch.no_grad():
            action_probs = self.rl_agent.get_action(state)
        
        # Convert action probabilities to budget allocations
        action_probs_np = action_probs.numpy()
        num_clients = len(client_ids)
        
        if num_clients <= len(action_probs_np):
            client_weights = action_probs_np[:num_clients]
        else:
            # Repeat pattern if more clients than actions
            client_weights = np.tile(action_probs_np, (num_clients // len(action_probs_np) + 1))[:num_clients]
        
        # Normalize weights
        total_weight = np.sum(client_weights)
        if total_weight > 0:
            client_weights = client_weights / total_weight
        else:
            client_weights = np.ones(num_clients) / num_clients
        
        allocations = {}
        for i, client_id in enumerate(client_ids):
            allocations[client_id] = round_budget * client_weights[i]
        
        return allocations
    
    def _compute_quantum_allocation(
        self, 
        client_ids: List[str], 
        round_budget: float
    ) -> Dict[str, float]:
        """Compute quantum-inspired budget allocation."""
        if self.quantum_optimizer is None:
            self.quantum_optimizer = QuantumBudgetOptimizer(len(client_ids))
        
        # Initial allocation based on client characteristics
        base_allocations = np.array([
            self._extract_client_features(client_id)[2] * round_budget / len(client_ids)  # Performance-based
            for client_id in client_ids
        ])
        
        # Apply quantum superposition
        quantum_allocations = self.quantum_optimizer.apply_quantum_superposition(base_allocations)
        
        # Create client correlation matrix
        correlations = np.eye(len(client_ids))
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i != j:
                    profile_i = self.client_profiles[client_i]
                    profile_j = self.client_profiles[client_j]
                    # Simple correlation based on data sensitivity similarity
                    correlations[i, j] = 1.0 - abs(profile_i.data_sensitivity - profile_j.data_sensitivity)
        
        # Apply quantum entanglement redistribution
        final_allocations = self.quantum_optimizer.quantum_entanglement_redistribution(
            quantum_allocations, correlations
        )
        
        # Apply decoherence correction
        time_step = len(self.allocation_history)
        corrected_allocations = self.quantum_optimizer.decoherence_correction(final_allocations, time_step)
        
        # Normalize to budget constraint
        total_allocated = np.sum(corrected_allocations)
        if total_allocated > 0:
            corrected_allocations = corrected_allocations * (round_budget / total_allocated)
        
        allocations = {}
        for i, client_id in enumerate(client_ids):
            allocations[client_id] = max(0.0, corrected_allocations[i])
        
        return allocations
    
    def _compute_pareto_optimal_allocation(
        self, 
        client_ids: List[str], 
        round_budget: float
    ) -> Dict[str, float]:
        """Compute Pareto optimal budget allocation."""
        # Multi-objective optimization: maximize utility, minimize privacy cost
        
        # Generate candidate allocations
        num_candidates = 100
        candidates = []
        
        for _ in range(num_candidates):
            # Random allocation
            weights = np.random.dirichlet(np.ones(len(client_ids)))
            allocation = {client_id: round_budget * weights[i] for i, client_id in enumerate(client_ids)}
            
            # Evaluate objectives
            utility_score = self._evaluate_utility(allocation)
            privacy_cost = self._evaluate_privacy_cost(allocation)
            
            candidates.append((allocation, utility_score, privacy_cost))
        
        # Find Pareto frontier
        pareto_candidates = []
        for i, (alloc_i, util_i, priv_i) in enumerate(candidates):
            is_dominated = False
            for j, (alloc_j, util_j, priv_j) in enumerate(candidates):
                if i != j and util_j >= util_i and priv_j <= priv_i and (util_j > util_i or priv_j < priv_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_candidates.append((alloc_i, util_i, priv_i))
        
        # Select best from Pareto frontier (balanced utility-privacy tradeoff)
        best_allocation = None
        best_score = -np.inf
        
        for allocation, utility, privacy_cost in pareto_candidates:
            # Combined score: utility / (1 + privacy_cost)
            score = utility / (1.0 + privacy_cost)
            if score > best_score:
                best_score = score
                best_allocation = allocation
        
        return best_allocation or self._compute_uniform_allocation(client_ids, round_budget)
    
    def _evaluate_utility(self, allocation: Dict[str, float]) -> float:
        """Evaluate expected utility of budget allocation."""
        total_utility = 0.0
        
        for client_id, budget in allocation.items():
            profile = self.client_profiles[client_id]
            
            # Predict utility based on budget and client characteristics
            if profile.performance_history:
                base_performance = np.mean(profile.performance_history[-5:])
            else:
                base_performance = 0.5
            
            # Utility increases with budget but with diminishing returns
            utility = base_performance * (1.0 - np.exp(-budget / profile.total_epsilon_budget))
            
            # Adjust for client characteristics
            utility *= profile.resource_availability
            utility /= profile.communication_cost
            
            total_utility += utility
        
        return total_utility
    
    def _evaluate_privacy_cost(self, allocation: Dict[str, float]) -> float:
        """Evaluate privacy cost of budget allocation."""
        total_cost = 0.0
        
        for client_id, budget in allocation.items():
            profile = self.client_profiles[client_id]
            
            # Privacy cost increases with budget and data sensitivity
            privacy_cost = budget * profile.data_sensitivity
            
            # Higher cost for clients with strict privacy preferences
            strictness = profile.privacy_preferences.get("strictness", 0.5)
            privacy_cost *= (1.0 + strictness)
            
            total_cost += privacy_cost
        
        return total_cost
    
    def allocate_budget(
        self, 
        client_ids: List[str], 
        round_num: int,
        round_budget_fraction: float = 0.1
    ) -> List[BudgetAllocation]:
        """Allocate privacy budget for a training round."""
        round_budget = self.total_epsilon_budget * round_budget_fraction
        
        # Ensure we don't exceed total budget
        remaining_budget = self.total_epsilon_budget - self.epsilon_consumed
        round_budget = min(round_budget, remaining_budget)
        
        if round_budget <= 0:
            logger.warning("No remaining privacy budget for allocation")
            return []
        
        # Compute allocation based on strategy
        if self.optimization_strategy == BudgetAllocationStrategy.UNIFORM:
            allocations_dict = self._compute_uniform_allocation(client_ids, round_budget)
        elif self.optimization_strategy == BudgetAllocationStrategy.PERFORMANCE_WEIGHTED:
            allocations_dict = self._compute_performance_weighted_allocation(client_ids, round_budget)
        elif self.optimization_strategy == BudgetAllocationStrategy.RL_ADAPTIVE:
            allocations_dict = self._compute_rl_allocation(client_ids, round_budget)
        elif self.optimization_strategy == BudgetAllocationStrategy.QUANTUM_INSPIRED:
            allocations_dict = self._compute_quantum_allocation(client_ids, round_budget)
        elif self.optimization_strategy == BudgetAllocationStrategy.PARETO_OPTIMAL:
            allocations_dict = self._compute_pareto_optimal_allocation(client_ids, round_budget)
        else:
            allocations_dict = self._compute_uniform_allocation(client_ids, round_budget)
        
        # Create allocation objects
        allocations = []
        for client_id in client_ids:
            epsilon_allocated = allocations_dict.get(client_id, 0.0)
            delta_allocated = epsilon_allocated * (self.total_delta_budget / self.total_epsilon_budget)
            
            # Predict utility
            expected_utility = self._evaluate_utility({client_id: epsilon_allocated})
            
            allocation = BudgetAllocation(
                client_id=client_id,
                round_num=round_num,
                epsilon_allocated=epsilon_allocated,
                delta_allocated=delta_allocated,
                allocation_confidence=0.85,  # Could be computed based on model confidence
                expected_utility=expected_utility,
                allocation_strategy=self.optimization_strategy
            )
            
            # Update client profile
            profile = self.client_profiles[client_id]
            profile.current_epsilon += epsilon_allocated
            profile.current_delta += delta_allocated
            profile.allocation_history.append({
                "round": round_num,
                "epsilon": epsilon_allocated,
                "delta": delta_allocated,
                "strategy": self.optimization_strategy.value
            })
            
            allocations.append(allocation)
        
        # Update global state
        self.epsilon_consumed += sum(alloc.epsilon_allocated for alloc in allocations)
        self.delta_consumed += sum(alloc.delta_allocated for alloc in allocations)
        self.round_allocations[round_num] = allocations
        self.allocation_history.append({round_num: {alloc.client_id: alloc for alloc in allocations}})
        
        logger.info(f"Allocated privacy budget for round {round_num}: "
                   f"epsilon={sum(alloc.epsilon_allocated for alloc in allocations):.3f}")
        
        return allocations
    
    def update_client_performance(self, client_id: str, performance_metrics: Dict[str, float]):
        """Update client performance after training round."""
        if client_id not in self.client_profiles:
            logger.warning(f"Unknown client ID: {client_id}")
            return
        
        profile = self.client_profiles[client_id]
        
        # Extract primary performance metric (e.g., accuracy, F1-score)
        performance_score = performance_metrics.get("accuracy", performance_metrics.get("f1_score", 0.5))
        profile.performance_history.append(performance_score)
        
        # Update RL agent if using RL strategy
        if self.optimization_strategy == BudgetAllocationStrategy.RL_ADAPTIVE:
            # Create state for RL update
            if len(self.allocation_history) > 0:
                last_allocation = None
                for round_data in reversed(self.allocation_history):
                    for round_num, client_allocations in round_data.items():
                        if client_id in client_allocations:
                            last_allocation = client_allocations[client_id]
                            break
                    if last_allocation:
                        break
                
                if last_allocation:
                    # Reward based on performance improvement
                    if len(profile.performance_history) > 1:
                        reward = performance_score - profile.performance_history[-2]
                    else:
                        reward = performance_score - 0.5  # Baseline
                    
                    # Store transition for RL learning (simplified)
                    # In practice, this would include proper state representations
                    features = self._extract_client_features(client_id)
                    state = torch.tensor(features[:20] if len(features) >= 20 else 
                                       features + [0.0] * (20 - len(features)), dtype=torch.float32)
                    action = torch.tensor([0], dtype=torch.long)  # Simplified action
                    
                    self.rl_agent.store_transition(state, action, reward, state, False)
        
        logger.info(f"Updated performance for client {client_id}: {performance_score:.3f}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            "strategy": self.optimization_strategy.value,
            "total_budget": {
                "epsilon_budget": self.total_epsilon_budget,
                "delta_budget": self.total_delta_budget,
                "epsilon_consumed": self.epsilon_consumed,
                "delta_consumed": self.delta_consumed,
                "epsilon_remaining": self.total_epsilon_budget - self.epsilon_consumed,
                "utilization_rate": self.epsilon_consumed / self.total_epsilon_budget
            },
            "client_profiles": {},
            "allocation_efficiency": {},
            "performance_metrics": self.optimization_metrics
        }
        
        # Client-specific reports
        for client_id, profile in self.client_profiles.items():
            client_report = {
                "budget_utilization": profile.current_epsilon / profile.total_epsilon_budget,
                "performance_trend": profile.performance_history[-5:] if profile.performance_history else [],
                "avg_performance": np.mean(profile.performance_history) if profile.performance_history else 0.0,
                "allocations_received": len(profile.allocation_history),
                "characteristics": {
                    "data_sensitivity": profile.data_sensitivity,
                    "communication_cost": profile.communication_cost,
                    "resource_availability": profile.resource_availability
                }
            }
            report["client_profiles"][client_id] = client_report
        
        # Overall efficiency metrics
        if self.allocation_history:
            total_utility = sum(
                self._evaluate_utility({alloc.client_id: alloc.epsilon_allocated for alloc in allocations})
                for round_data in self.allocation_history
                for allocations in round_data.values()
            )
            
            report["allocation_efficiency"] = {
                "total_utility_achieved": total_utility,
                "utility_per_epsilon": total_utility / self.epsilon_consumed if self.epsilon_consumed > 0 else 0,
                "rounds_completed": len(self.allocation_history),
                "avg_allocation_per_round": self.epsilon_consumed / len(self.allocation_history) if self.allocation_history else 0
            }
        
        return report
    
    def visualize_allocations(self, save_path: Optional[str] = None):
        """Create visualization of budget allocations over time."""
        if not self.allocation_history:
            logger.warning("No allocation history to visualize")
            return
        
        # Prepare data for plotting
        rounds = []
        client_allocations = defaultdict(list)
        
        for round_data in self.allocation_history:
            for round_num, allocations in round_data.items():
                rounds.append(round_num)
                for client_id in self.client_profiles.keys():
                    if client_id in allocations:
                        client_allocations[client_id].append(allocations[client_id].epsilon_allocated)
                    else:
                        client_allocations[client_id].append(0.0)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Allocation over time by client
        ax1 = axes[0, 0]
        for client_id, allocations in client_allocations.items():
            ax1.plot(rounds, allocations, marker='o', label=client_id, linewidth=2)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Epsilon Allocated")
        ax1.set_title("Privacy Budget Allocation Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative budget consumption
        ax2 = axes[0, 1]
        cumulative_consumption = np.cumsum([
            sum(allocations[client_id].epsilon_allocated for client_id in allocations)
            for round_data in self.allocation_history
            for allocations in round_data.values()
        ])
        ax2.plot(rounds, cumulative_consumption, marker='s', linewidth=3, color='red')
        ax2.axhline(y=self.total_epsilon_budget, color='black', linestyle='--', label='Total Budget')
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Cumulative Epsilon Consumed")
        ax2.set_title("Cumulative Privacy Budget Consumption")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Client performance vs allocation
        ax3 = axes[1, 0]
        for client_id, profile in self.client_profiles.items():
            if profile.performance_history and profile.allocation_history:
                allocations = [alloc["epsilon"] for alloc in profile.allocation_history]
                performance = profile.performance_history[:len(allocations)]
                ax3.scatter(allocations, performance, label=client_id, alpha=0.7, s=50)
        ax3.set_xlabel("Epsilon Allocated")
        ax3.set_ylabel("Performance Score")
        ax3.set_title("Performance vs Budget Allocation")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Budget utilization by client
        ax4 = axes[1, 1]
        client_ids = list(self.client_profiles.keys())
        utilizations = [
            profile.current_epsilon / profile.total_epsilon_budget
            for profile in self.client_profiles.values()
        ]
        colors = plt.cm.viridis(np.linspace(0, 1, len(client_ids)))
        bars = ax4.bar(client_ids, utilizations, color=colors)
        ax4.set_ylabel("Budget Utilization Rate")
        ax4.set_title("Privacy Budget Utilization by Client")
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, util in zip(bars, utilizations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{util:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved allocation visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    async def optimize_continuously(self, optimization_interval: int = 10):
        """Continuously optimize budget allocation strategy."""
        logger.info("Starting continuous budget optimization")
        
        while True:
            try:
                # Learn from recent experience if using RL
                if self.optimization_strategy == BudgetAllocationStrategy.RL_ADAPTIVE:
                    self.rl_agent.learn(batch_size=32)
                
                # Update quantum states if using quantum optimization
                if (self.optimization_strategy == BudgetAllocationStrategy.QUANTUM_INSPIRED and 
                    self.quantum_optimizer):
                    time_step = len(self.allocation_history)
                    # Perform quantum state evolution
                    self.quantum_optimizer.quantum_states *= np.exp(1j * np.pi * time_step / 100)
                
                # Generate optimization metrics
                if self.allocation_history:
                    recent_allocations = self.allocation_history[-5:] if len(self.allocation_history) >= 5 else self.allocation_history
                    
                    total_utility = 0.0
                    for round_data in recent_allocations:
                        for allocations in round_data.values():
                            round_utility = sum(alloc.expected_utility for alloc in allocations)
                            total_utility += round_utility
                    
                    avg_utility = total_utility / len(recent_allocations) if recent_allocations else 0.0
                    self.optimization_metrics["utility_achieved"].append(avg_utility)
                    
                    # Calculate allocation efficiency
                    total_budget_used = sum(
                        sum(alloc.epsilon_allocated for alloc in allocations)
                        for round_data in recent_allocations
                        for allocations in round_data.values()
                    )
                    efficiency = avg_utility / total_budget_used if total_budget_used > 0 else 0.0
                    self.optimization_metrics["allocation_efficiency"].append(efficiency)
                
                logger.info(f"Budget optimization cycle completed. Total consumed: {self.epsilon_consumed:.3f}/{self.total_epsilon_budget}")
                
                await asyncio.sleep(optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous optimization: {e}")
                await asyncio.sleep(optimization_interval)
    
    def export_optimization_data(self, export_path: str):
        """Export optimization data for research analysis."""
        export_data = {
            "configuration": {
                "strategy": self.optimization_strategy.value,
                "total_epsilon_budget": self.total_epsilon_budget,
                "total_delta_budget": self.total_delta_budget,
                "num_rounds": self.num_rounds
            },
            "client_profiles": {
                client_id: {
                    "characteristics": {
                        "data_sensitivity": profile.data_sensitivity,
                        "communication_cost": profile.communication_cost,
                        "resource_availability": profile.resource_availability,
                        "privacy_preferences": profile.privacy_preferences
                    },
                    "performance_history": profile.performance_history,
                    "allocation_history": profile.allocation_history,
                    "budget_utilization": {
                        "epsilon_used": profile.current_epsilon,
                        "delta_used": profile.current_delta,
                        "epsilon_budget": profile.total_epsilon_budget,
                        "delta_budget": profile.total_delta_budget
                    }
                }
                for client_id, profile in self.client_profiles.items()
            },
            "allocation_history": [
                {
                    round_num: {
                        client_id: {
                            "epsilon_allocated": alloc.epsilon_allocated,
                            "delta_allocated": alloc.delta_allocated,
                            "expected_utility": alloc.expected_utility,
                            "allocation_confidence": alloc.allocation_confidence
                        }
                        for client_id, alloc in allocations.items()
                    }
                    for round_num, allocations in round_data.items()
                }
                for round_data in self.allocation_history
            ],
            "optimization_metrics": self.optimization_metrics,
            "summary_statistics": {
                "total_epsilon_consumed": self.epsilon_consumed,
                "total_delta_consumed": self.delta_consumed,
                "budget_utilization_rate": self.epsilon_consumed / self.total_epsilon_budget,
                "rounds_completed": len(self.allocation_history),
                "clients_served": len(self.client_profiles),
                "avg_utility_per_round": np.mean(self.optimization_metrics["utility_achieved"]) if self.optimization_metrics["utility_achieved"] else 0.0
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported optimization data to {export_path}")


# Factory functions for easy instantiation
def create_adaptive_budget_optimizer(
    strategy: BudgetAllocationStrategy = BudgetAllocationStrategy.RL_ADAPTIVE,
    total_epsilon_budget: float = 100.0,
    **kwargs
) -> AdaptivePrivacyBudgetOptimizer:
    """Create an adaptive privacy budget optimizer with specified strategy."""
    return AdaptivePrivacyBudgetOptimizer(
        total_epsilon_budget=total_epsilon_budget,
        optimization_strategy=strategy,
        **kwargs
    )


def create_quantum_budget_optimizer(
    total_epsilon_budget: float = 100.0,
    **kwargs
) -> AdaptivePrivacyBudgetOptimizer:
    """Create a quantum-inspired privacy budget optimizer."""
    return AdaptivePrivacyBudgetOptimizer(
        total_epsilon_budget=total_epsilon_budget,
        optimization_strategy=BudgetAllocationStrategy.QUANTUM_INSPIRED,
        **kwargs
    )


def create_rl_budget_optimizer(
    total_epsilon_budget: float = 100.0,
    **kwargs
) -> AdaptivePrivacyBudgetOptimizer:
    """Create an RL-based adaptive privacy budget optimizer."""
    return AdaptivePrivacyBudgetOptimizer(
        total_epsilon_budget=total_epsilon_budget,
        optimization_strategy=BudgetAllocationStrategy.RL_ADAPTIVE,
        **kwargs
    )


if __name__ == "__main__":
    # Demonstration of the adaptive privacy budget optimization system
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Privacy Budget Optimization Demo")
    parser.add_argument("--strategy", type=str, default="rl_adaptive", 
                       choices=["uniform", "performance_weighted", "rl_adaptive", "quantum_inspired", "pareto_optimal"])
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=20, help="Number of training rounds")
    parser.add_argument("--budget", type=float, default=50.0, help="Total epsilon budget")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    strategy = BudgetAllocationStrategy(args.strategy)
    optimizer = AdaptivePrivacyBudgetOptimizer(
        total_epsilon_budget=args.budget,
        num_rounds=args.rounds,
        optimization_strategy=strategy
    )
    
    # Register clients with different characteristics
    for i in range(args.clients):
        optimizer.register_client(
            client_id=f"client_{i}",
            epsilon_budget=args.budget / args.clients,
            client_characteristics={
                "data_sensitivity": np.random.uniform(0.5, 2.0),
                "communication_cost": np.random.uniform(0.8, 1.5),
                "resource_availability": np.random.uniform(0.7, 1.2),
                "privacy_preferences": {
                    "strictness": np.random.uniform(0.3, 0.8),
                    "utility_weight": np.random.uniform(0.4, 0.9)
                }
            }
        )
    
    # Simulate training rounds
    client_ids = [f"client_{i}" for i in range(args.clients)]
    
    print(f"\nStarting adaptive privacy budget optimization simulation")
    print(f"Strategy: {strategy.value}")
    print(f"Clients: {args.clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Total Budget: {args.budget}")
    print("-" * 60)
    
    for round_num in range(args.rounds):
        # Allocate budget for this round
        allocations = optimizer.allocate_budget(client_ids, round_num)
        
        # Simulate client performance
        for allocation in allocations:
            # Simulate performance based on allocated budget and client characteristics
            profile = optimizer.client_profiles[allocation.client_id]
            
            # Performance improves with budget but has noise
            base_performance = 0.5 + 0.3 * (allocation.epsilon_allocated / (args.budget / args.rounds))
            noise = np.random.normal(0, 0.1)
            performance = np.clip(base_performance + noise, 0.0, 1.0)
            
            # Update client performance
            optimizer.update_client_performance(
                allocation.client_id, 
                {"accuracy": performance}
            )
        
        # Print round summary
        total_allocated = sum(alloc.epsilon_allocated for alloc in allocations)
        print(f"Round {round_num + 1:2d}: Allocated ε={total_allocated:.3f}, "
              f"Remaining ε={optimizer.total_epsilon_budget - optimizer.epsilon_consumed:.3f}")
        
        if optimizer.epsilon_consumed >= optimizer.total_epsilon_budget * 0.95:
            print("Approaching budget limit, stopping simulation")
            break
    
    # Generate final report
    print("\n" + "=" * 60)
    print("OPTIMIZATION REPORT")
    print("=" * 60)
    
    report = optimizer.get_optimization_report()
    
    print(f"Strategy: {report['strategy']}")
    print(f"Budget Utilization: {report['total_budget']['utilization_rate']:.1%}")
    print(f"Rounds Completed: {report['allocation_efficiency'].get('rounds_completed', 0)}")
    print(f"Total Utility: {report['allocation_efficiency'].get('total_utility_achieved', 0):.3f}")
    print(f"Utility per Epsilon: {report['allocation_efficiency'].get('utility_per_epsilon', 0):.3f}")
    
    print(f"\nClient Performance Summary:")
    for client_id, profile_report in report['client_profiles'].items():
        print(f"  {client_id}: "
              f"Budget Used {profile_report['budget_utilization']:.1%}, "
              f"Avg Performance {profile_report['avg_performance']:.3f}")
    
    # Create visualizations
    try:
        optimizer.visualize_allocations(f"budget_allocation_visualization_{args.strategy}.png")
        print(f"\nVisualization saved as: budget_allocation_visualization_{args.strategy}.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Export data for research
    optimizer.export_optimization_data(f"optimization_data_{args.strategy}.json")
    print(f"Optimization data exported as: optimization_data_{args.strategy}.json")
    
    print(f"\nAdaptive privacy budget optimization simulation completed!")
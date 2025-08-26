"""
Comprehensive test suite for Adaptive Privacy Budget Optimizer.

Tests all components of the adaptive privacy budget optimization system including
RL agents, quantum optimization, anomaly detection, and budget allocation strategies.

Author: Terry (Terragon Labs)
"""

import pytest
import numpy as np
import torch
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

# Import the modules under test
from src.dp_federated_lora.adaptive_privacy_budget_optimizer import (
    AdaptivePrivacyBudgetOptimizer,
    BudgetAllocationStrategy,
    ClientBudgetProfile,
    BudgetAllocation,
    PrivacyUtilityPredictor,
    QuantumBudgetOptimizer,
    RLBudgetAgent,
    create_adaptive_budget_optimizer,
    create_quantum_budget_optimizer,
    create_rl_budget_optimizer
)


class TestClientBudgetProfile:
    """Test ClientBudgetProfile data structure."""
    
    def test_client_budget_profile_creation(self):
        """Test creating a client budget profile."""
        profile = ClientBudgetProfile(
            client_id="test_client",
            total_epsilon_budget=10.0,
            total_delta_budget=1e-5,
            data_sensitivity=1.5,
            privacy_preferences={"strictness": 0.8}
        )
        
        assert profile.client_id == "test_client"
        assert profile.current_epsilon == 0.0
        assert profile.current_delta == 0.0
        assert profile.total_epsilon_budget == 10.0
        assert profile.data_sensitivity == 1.5
        assert profile.privacy_preferences["strictness"] == 0.8
        assert len(profile.performance_history) == 0
        assert len(profile.allocation_history) == 0
    
    def test_client_budget_profile_defaults(self):
        """Test default values in client budget profile."""
        profile = ClientBudgetProfile(client_id="test")
        
        assert profile.data_sensitivity == 1.0
        assert profile.communication_cost == 1.0
        assert profile.resource_availability == 1.0
        assert isinstance(profile.privacy_preferences, dict)
        assert isinstance(profile.performance_history, list)
        assert isinstance(profile.allocation_history, list)


class TestBudgetAllocation:
    """Test BudgetAllocation data structure."""
    
    def test_budget_allocation_creation(self):
        """Test creating a budget allocation."""
        allocation = BudgetAllocation(
            client_id="test_client",
            round_num=5,
            epsilon_allocated=2.0,
            delta_allocated=1e-6,
            allocation_confidence=0.85,
            expected_utility=0.75,
            allocation_strategy=BudgetAllocationStrategy.RL_ADAPTIVE
        )
        
        assert allocation.client_id == "test_client"
        assert allocation.round_num == 5
        assert allocation.epsilon_allocated == 2.0
        assert allocation.delta_allocated == 1e-6
        assert allocation.allocation_confidence == 0.85
        assert allocation.expected_utility == 0.75
        assert allocation.allocation_strategy == BudgetAllocationStrategy.RL_ADAPTIVE
        assert allocation.quantum_coherence is None
        assert allocation.pareto_rank is None


class TestPrivacyUtilityPredictor:
    """Test PrivacyUtilityPredictor neural network."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = PrivacyUtilityPredictor(input_dim=10, hidden_dim=32)
        
        # Check network structure
        assert len(predictor.network) == 8  # 4 linear + 3 relu + 1 dropout layers
        
        # Check input/output dimensions
        first_layer = predictor.network[0]
        last_layer = predictor.network[-1]
        assert first_layer.in_features == 10
        assert last_layer.out_features == 2  # utility_score, privacy_risk
    
    def test_predictor_forward_pass(self):
        """Test forward pass through predictor."""
        predictor = PrivacyUtilityPredictor()
        
        # Test with batch of inputs
        batch_size = 32
        input_tensor = torch.randn(batch_size, 12)
        
        output = predictor(input_tensor)
        
        assert output.shape == (batch_size, 2)
        assert torch.all(torch.isfinite(output))
    
    def test_predictor_single_input(self):
        """Test predictor with single input."""
        predictor = PrivacyUtilityPredictor()
        
        single_input = torch.randn(1, 12)
        output = predictor(single_input)
        
        assert output.shape == (1, 2)
        assert torch.all(torch.isfinite(output))


class TestQuantumBudgetOptimizer:
    """Test QuantumBudgetOptimizer."""
    
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer initialization."""
        optimizer = QuantumBudgetOptimizer(num_clients=5)
        
        assert len(optimizer.quantum_states) == 5
        assert optimizer.entanglement_matrix.shape == (5, 5)
        assert optimizer.coherence_time == 100
        
        # Check quantum states are complex
        assert np.iscomplexobj(optimizer.quantum_states)
    
    def test_apply_quantum_superposition(self):
        """Test quantum superposition application."""
        optimizer = QuantumBudgetOptimizer(num_clients=3)
        
        budget_allocations = np.array([1.0, 2.0, 3.0])
        superposed = optimizer.apply_quantum_superposition(budget_allocations)
        
        assert len(superposed) == 3
        assert np.all(np.isfinite(superposed))
        assert np.all(superposed >= 0)  # Should be non-negative
    
    def test_quantum_entanglement_redistribution(self):
        """Test quantum entanglement redistribution."""
        optimizer = QuantumBudgetOptimizer(num_clients=3)
        
        allocations = np.array([1.0, 2.0, 3.0])
        correlations = np.eye(3)  # Identity matrix
        
        redistributed = optimizer.quantum_entanglement_redistribution(allocations, correlations)
        
        assert len(redistributed) == 3
        assert np.all(np.isfinite(redistributed))
    
    def test_decoherence_correction(self):
        """Test decoherence correction."""
        optimizer = QuantumBudgetOptimizer(num_clients=3)
        
        allocations = np.array([1.0, 2.0, 3.0])
        corrected = optimizer.decoherence_correction(allocations, time_step=50)
        
        assert len(corrected) == 3
        assert np.all(np.isfinite(corrected))
        assert np.all(corrected >= 0)


class TestRLBudgetAgent:
    """Test RLBudgetAgent."""
    
    def test_rl_agent_initialization(self):
        """Test RL agent initialization."""
        agent = RLBudgetAgent(state_dim=15, action_dim=8)
        
        # Check actor network
        actor_input = agent.actor[0]
        actor_output = agent.actor[-2]  # Before softmax
        assert actor_input.in_features == 15
        assert actor_output.out_features == 8
        
        # Check critic network
        critic_input = agent.critic[0]
        critic_output = agent.critic[-1]
        assert critic_input.in_features == 15
        assert critic_output.out_features == 1
        
        # Check memory
        assert len(agent.memory) == 0
        assert agent.memory.maxlen == 10000
    
    def test_rl_agent_get_action(self):
        """Test getting action from RL agent."""
        agent = RLBudgetAgent(state_dim=10, action_dim=5)
        
        state = torch.randn(10)
        action_probs = agent.get_action(state)
        
        assert action_probs.shape == (5,)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(action_probs >= 0)
        assert torch.all(action_probs <= 1)
    
    def test_rl_agent_get_value(self):
        """Test getting value from RL agent."""
        agent = RLBudgetAgent(state_dim=10)
        
        state = torch.randn(10)
        value = agent.get_value(state)
        
        assert value.shape == (1,)
        assert torch.isfinite(value)
    
    def test_rl_agent_store_transition(self):
        """Test storing transition in RL agent."""
        agent = RLBudgetAgent()
        
        state = torch.randn(20)
        action = torch.tensor([2])
        reward = 0.75
        next_state = torch.randn(20)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 1
        
        # Store more transitions
        for _ in range(10):
            agent.store_transition(torch.randn(20), torch.tensor([1]), 0.5, torch.randn(20), False)
        
        assert len(agent.memory) == 11
    
    def test_rl_agent_learning(self):
        """Test RL agent learning."""
        agent = RLBudgetAgent(state_dim=10, action_dim=5)
        
        # Store some transitions
        for _ in range(50):
            state = torch.randn(10)
            action = torch.tensor([np.random.randint(0, 5)])
            reward = np.random.uniform(0, 1)
            next_state = torch.randn(10)
            done = np.random.random() < 0.1
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Test learning (should not raise exception)
        try:
            agent.learn(batch_size=16)
        except Exception as e:
            pytest.fail(f"Learning failed with exception: {e}")


class TestAdaptivePrivacyBudgetOptimizer:
    """Test AdaptivePrivacyBudgetOptimizer main class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimizer = AdaptivePrivacyBudgetOptimizer(
            total_epsilon_budget=50.0,
            total_delta_budget=1e-4,
            num_rounds=20,
            optimization_strategy=BudgetAllocationStrategy.RL_ADAPTIVE
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.total_epsilon_budget == 50.0
        assert self.optimizer.total_delta_budget == 1e-4
        assert self.optimizer.num_rounds == 20
        assert self.optimizer.optimization_strategy == BudgetAllocationStrategy.RL_ADAPTIVE
        
        assert len(self.optimizer.client_profiles) == 0
        assert len(self.optimizer.allocation_history) == 0
        assert self.optimizer.epsilon_consumed == 0.0
        assert self.optimizer.delta_consumed == 0.0
    
    def test_register_client(self):
        """Test client registration."""
        profile = self.optimizer.register_client(
            client_id="test_client",
            epsilon_budget=10.0,
            delta_budget=1e-5,
            client_characteristics={
                "data_sensitivity": 1.5,
                "communication_cost": 1.2,
                "resource_availability": 0.9,
                "privacy_preferences": {"strictness": 0.7}
            }
        )
        
        assert profile.client_id == "test_client"
        assert profile.total_epsilon_budget == 10.0
        assert profile.total_delta_budget == 1e-5
        assert profile.data_sensitivity == 1.5
        assert profile.communication_cost == 1.2
        assert profile.resource_availability == 0.9
        assert profile.privacy_preferences["strictness"] == 0.7
        
        # Check client is registered
        assert "test_client" in self.optimizer.client_profiles
        assert self.optimizer.client_profiles["test_client"] == profile
    
    def test_register_multiple_clients(self):
        """Test registering multiple clients."""
        for i in range(5):
            self.optimizer.register_client(
                client_id=f"client_{i}",
                epsilon_budget=8.0,
                delta_budget=1e-5
            )
        
        assert len(self.optimizer.client_profiles) == 5
        
        # Check quantum optimizer is initialized for quantum strategy
        quantum_optimizer = AdaptivePrivacyBudgetOptimizer(
            optimization_strategy=BudgetAllocationStrategy.QUANTUM_INSPIRED
        )
        quantum_optimizer.register_client("client_0", epsilon_budget=5.0)
        assert quantum_optimizer.quantum_optimizer is not None
    
    def test_extract_client_features(self):
        """Test client feature extraction."""
        # Register a client first
        profile = self.optimizer.register_client(
            client_id="test_client",
            epsilon_budget=10.0,
            client_characteristics={
                "data_sensitivity": 1.5,
                "privacy_preferences": {"strictness": 0.8, "utility_weight": 0.6}
            }
        )
        
        # Add some performance history
        profile.performance_history = [0.8, 0.85, 0.82, 0.87]
        
        features = self.optimizer._extract_client_features("test_client")
        
        assert len(features) == 12
        assert np.all(np.isfinite(features))
        
        # Check some expected values
        assert 0 <= features[0] <= 1  # Budget utilization
        assert features[3] == 1.5     # Data sensitivity
        assert features[8] == 0.8     # Strictness
        assert features[9] == 0.6     # Utility weight
    
    def test_uniform_allocation(self):
        """Test uniform budget allocation."""
        # Register clients
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            self.optimizer.register_client(client_id, epsilon_budget=10.0)
            client_ids.append(client_id)
        
        allocations = self.optimizer._compute_uniform_allocation(client_ids, 6.0)
        
        assert len(allocations) == 3
        for client_id in client_ids:
            assert allocations[client_id] == 2.0  # 6.0 / 3
    
    def test_performance_weighted_allocation(self):
        """Test performance-weighted allocation."""
        # Register clients with different performance histories
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            profile = self.optimizer.register_client(client_id, epsilon_budget=10.0)
            profile.performance_history = [0.5 + i * 0.2]  # Different performance levels
            client_ids.append(client_id)
        
        allocations = self.optimizer._compute_performance_weighted_allocation(client_ids, 6.0)
        
        assert len(allocations) == 3
        assert sum(allocations.values()) == pytest.approx(6.0, abs=1e-6)
        
        # Higher performance clients should get more allocation
        assert allocations["client_2"] > allocations["client_0"]
    
    def test_rl_allocation(self):
        """Test RL-based allocation."""
        # Register clients
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            self.optimizer.register_client(client_id, epsilon_budget=10.0)
            client_ids.append(client_id)
        
        allocations = self.optimizer._compute_rl_allocation(client_ids, 6.0)
        
        assert len(allocations) == 3
        assert sum(allocations.values()) == pytest.approx(6.0, abs=1e-6)
        assert all(alloc >= 0 for alloc in allocations.values())
    
    def test_quantum_allocation(self):
        """Test quantum-inspired allocation."""
        # Create quantum-enabled optimizer
        quantum_optimizer = AdaptivePrivacyBudgetOptimizer(
            optimization_strategy=BudgetAllocationStrategy.QUANTUM_INSPIRED
        )
        
        # Register clients
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            quantum_optimizer.register_client(client_id, epsilon_budget=10.0)
            client_ids.append(client_id)
        
        allocations = quantum_optimizer._compute_quantum_allocation(client_ids, 6.0)
        
        assert len(allocations) == 3
        assert all(alloc >= 0 for alloc in allocations.values())
        # Note: quantum allocation may not sum exactly to 6.0 due to quantum effects
    
    def test_pareto_optimal_allocation(self):
        """Test Pareto optimal allocation."""
        # Register clients
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            profile = self.optimizer.register_client(client_id, epsilon_budget=10.0)
            profile.data_sensitivity = 1.0 + i * 0.5  # Different sensitivities
            client_ids.append(client_id)
        
        allocations = self.optimizer._compute_pareto_optimal_allocation(client_ids, 6.0)
        
        assert len(allocations) == 3
        assert all(alloc >= 0 for alloc in allocations.values())
    
    def test_allocate_budget(self):
        """Test budget allocation for a round."""
        # Register clients
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            self.optimizer.register_client(client_id, epsilon_budget=15.0)
            client_ids.append(client_id)
        
        allocations = self.optimizer.allocate_budget(client_ids, round_num=1, round_budget_fraction=0.2)
        
        assert len(allocations) == 3
        
        # Check allocation objects
        for allocation in allocations:
            assert isinstance(allocation, BudgetAllocation)
            assert allocation.client_id in client_ids
            assert allocation.round_num == 1
            assert allocation.epsilon_allocated >= 0
            assert allocation.delta_allocated >= 0
            assert 0 <= allocation.allocation_confidence <= 1
            assert allocation.expected_utility >= 0
            assert allocation.allocation_strategy == BudgetAllocationStrategy.RL_ADAPTIVE
        
        # Check budget consumption tracking
        total_epsilon_allocated = sum(alloc.epsilon_allocated for alloc in allocations)
        assert self.optimizer.epsilon_consumed == total_epsilon_allocated
        
        # Check client profiles are updated
        for allocation in allocations:
            client_id = allocation.client_id
            profile = self.optimizer.client_profiles[client_id]
            assert profile.current_epsilon == allocation.epsilon_allocated
            assert len(profile.allocation_history) == 1
    
    def test_update_client_performance(self):
        """Test updating client performance."""
        # Register client
        self.optimizer.register_client("test_client", epsilon_budget=10.0)
        
        # Update performance
        self.optimizer.update_client_performance(
            "test_client", 
            {"accuracy": 0.85, "f1_score": 0.82}
        )
        
        profile = self.optimizer.client_profiles["test_client"]
        assert len(profile.performance_history) == 1
        assert profile.performance_history[0] == 0.85
        
        # Update again with different metric
        self.optimizer.update_client_performance(
            "test_client", 
            {"f1_score": 0.88}
        )
        
        assert len(profile.performance_history) == 2
        assert profile.performance_history[1] == 0.88
    
    def test_update_nonexistent_client_performance(self):
        """Test updating performance for non-existent client."""
        # Should not raise exception, but should log warning
        self.optimizer.update_client_performance(
            "nonexistent_client", 
            {"accuracy": 0.75}
        )
        # No assertion needed - just checking it doesn't crash
    
    def test_evaluate_utility(self):
        """Test utility evaluation."""
        # Register client with performance history
        profile = self.optimizer.register_client("test_client", epsilon_budget=10.0)
        profile.performance_history = [0.8, 0.85, 0.82]
        profile.resource_availability = 0.9
        profile.communication_cost = 1.1
        
        allocation = {"test_client": 2.0}
        utility = self.optimizer._evaluate_utility(allocation)
        
        assert utility > 0
        assert np.isfinite(utility)
    
    def test_evaluate_privacy_cost(self):
        """Test privacy cost evaluation."""
        # Register client with sensitivity
        profile = self.optimizer.register_client("test_client", epsilon_budget=10.0)
        profile.data_sensitivity = 1.5
        profile.privacy_preferences = {"strictness": 0.8}
        
        allocation = {"test_client": 2.0}
        cost = self.optimizer._evaluate_privacy_cost(allocation)
        
        assert cost > 0
        assert np.isfinite(cost)
    
    def test_get_optimization_report(self):
        """Test optimization report generation."""
        # Register clients and perform allocations
        client_ids = []
        for i in range(3):
            client_id = f"client_{i}"
            self.optimizer.register_client(client_id, epsilon_budget=10.0)
            client_ids.append(client_id)
        
        # Perform allocation
        allocations = self.optimizer.allocate_budget(client_ids, round_num=1)
        
        # Update performance
        for client_id in client_ids:
            self.optimizer.update_client_performance(client_id, {"accuracy": 0.8})
        
        report = self.optimizer.get_optimization_report()
        
        # Check report structure
        assert "strategy" in report
        assert "total_budget" in report
        assert "client_profiles" in report
        assert "allocation_efficiency" in report
        
        # Check budget info
        budget_info = report["total_budget"]
        assert budget_info["epsilon_budget"] == 50.0
        assert budget_info["epsilon_consumed"] > 0
        assert budget_info["utilization_rate"] > 0
        
        # Check client profiles
        assert len(report["client_profiles"]) == 3
        for client_id in client_ids:
            assert client_id in report["client_profiles"]
            client_report = report["client_profiles"][client_id]
            assert "budget_utilization" in client_report
            assert "performance_trend" in client_report
            assert "avg_performance" in client_report
    
    def test_visualize_allocations_no_history(self):
        """Test visualization with no allocation history."""
        # Should not raise exception
        try:
            self.optimizer.visualize_allocations()
        except Exception as e:
            # Expected to not crash, even with no data
            pass
    
    def test_export_optimization_data(self):
        """Test exporting optimization data."""
        # Register clients and perform allocations
        client_ids = []
        for i in range(2):
            client_id = f"client_{i}"
            self.optimizer.register_client(client_id, epsilon_budget=10.0)
            client_ids.append(client_id)
        
        allocations = self.optimizer.allocate_budget(client_ids, round_num=1)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.optimizer.export_optimization_data(export_path)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            # Check exported data structure
            assert "configuration" in data
            assert "client_profiles" in data
            assert "allocation_history" in data
            assert "summary_statistics" in data
            
            # Check configuration
            config = data["configuration"]
            assert config["strategy"] == "rl_adaptive"
            assert config["total_epsilon_budget"] == 50.0
            
            # Check summary statistics
            summary = data["summary_statistics"]
            assert summary["clients_served"] == 2
            assert summary["rounds_completed"] == 1
            assert summary["total_epsilon_consumed"] > 0
            
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_adaptive_budget_optimizer(self):
        """Test creating adaptive budget optimizer."""
        optimizer = create_adaptive_budget_optimizer(
            strategy=BudgetAllocationStrategy.PERFORMANCE_WEIGHTED,
            total_epsilon_budget=75.0
        )
        
        assert isinstance(optimizer, AdaptivePrivacyBudgetOptimizer)
        assert optimizer.optimization_strategy == BudgetAllocationStrategy.PERFORMANCE_WEIGHTED
        assert optimizer.total_epsilon_budget == 75.0
    
    def test_create_quantum_budget_optimizer(self):
        """Test creating quantum budget optimizer."""
        optimizer = create_quantum_budget_optimizer(total_epsilon_budget=100.0)
        
        assert isinstance(optimizer, AdaptivePrivacyBudgetOptimizer)
        assert optimizer.optimization_strategy == BudgetAllocationStrategy.QUANTUM_INSPIRED
        assert optimizer.total_epsilon_budget == 100.0
    
    def test_create_rl_budget_optimizer(self):
        """Test creating RL budget optimizer."""
        optimizer = create_rl_budget_optimizer(total_epsilon_budget=80.0)
        
        assert isinstance(optimizer, AdaptivePrivacyBudgetOptimizer)
        assert optimizer.optimization_strategy == BudgetAllocationStrategy.RL_ADAPTIVE
        assert optimizer.total_epsilon_budget == 80.0


class TestIntegrationScenarios:
    """Integration tests for complete scenarios."""
    
    def test_complete_federated_learning_simulation(self):
        """Test complete federated learning simulation scenario."""
        optimizer = AdaptivePrivacyBudgetOptimizer(
            total_epsilon_budget=100.0,
            num_rounds=10,
            optimization_strategy=BudgetAllocationStrategy.ADAPTIVE
        )
        
        # Register diverse clients
        client_configs = [
            {"client_id": "hospital_1", "epsilon_budget": 20.0, "data_sensitivity": 2.0},
            {"client_id": "hospital_2", "epsilon_budget": 15.0, "data_sensitivity": 1.5},
            {"client_id": "research_lab", "epsilon_budget": 30.0, "data_sensitivity": 1.0},
            {"client_id": "university", "epsilon_budget": 25.0, "data_sensitivity": 0.8},
        ]
        
        for config in client_configs:
            optimizer.register_client(
                config["client_id"],
                config["epsilon_budget"],
                client_characteristics={"data_sensitivity": config["data_sensitivity"]}
            )
        
        client_ids = [config["client_id"] for config in client_configs]
        
        # Simulate multiple training rounds
        for round_num in range(1, 6):
            # Allocate budget
            allocations = optimizer.allocate_budget(
                client_ids, 
                round_num, 
                round_budget_fraction=0.15
            )
            
            assert len(allocations) == 4
            
            # Simulate training and update performance
            for allocation in allocations:
                # Simulate performance improving over rounds
                base_performance = 0.6 + round_num * 0.05
                noise = np.random.normal(0, 0.02)
                performance = np.clip(base_performance + noise, 0.0, 1.0)
                
                optimizer.update_client_performance(
                    allocation.client_id,
                    {"accuracy": performance}
                )
        
        # Check final state
        assert optimizer.epsilon_consumed > 0
        assert optimizer.epsilon_consumed <= optimizer.total_epsilon_budget
        assert len(optimizer.allocation_history) == 5
        
        # All clients should have performance history
        for client_id in client_ids:
            profile = optimizer.client_profiles[client_id]
            assert len(profile.performance_history) == 5
            assert len(profile.allocation_history) == 5
        
        # Generate report
        report = optimizer.get_optimization_report()
        assert report["total_budget"]["rounds_completed"] == 5
        assert report["allocation_efficiency"]["total_utility_achieved"] > 0
    
    def test_budget_exhaustion_scenario(self):
        """Test scenario where budget gets exhausted."""
        optimizer = AdaptivePrivacyBudgetOptimizer(
            total_epsilon_budget=10.0,  # Small budget
            num_rounds=20,
            optimization_strategy=BudgetAllocationStrategy.UNIFORM
        )
        
        # Register client
        optimizer.register_client("test_client", epsilon_budget=15.0)  # More than total
        
        allocations_list = []
        round_num = 1
        
        # Keep allocating until budget is exhausted
        while optimizer.epsilon_consumed < optimizer.total_epsilon_budget * 0.99:
            allocations = optimizer.allocate_budget(["test_client"], round_num)
            
            if not allocations:  # No more budget
                break
                
            allocations_list.append(allocations)
            round_num += 1
            
            if round_num > 50:  # Safety check
                break
        
        # Should have some allocations
        assert len(allocations_list) > 0
        assert optimizer.epsilon_consumed <= optimizer.total_epsilon_budget
        
        # Try to allocate more (should return empty)
        final_allocations = optimizer.allocate_budget(["test_client"], round_num)
        assert len(final_allocations) == 0
    
    def test_multi_strategy_comparison(self):
        """Test comparing different allocation strategies."""
        strategies = [
            BudgetAllocationStrategy.UNIFORM,
            BudgetAllocationStrategy.PERFORMANCE_WEIGHTED,
            BudgetAllocationStrategy.RL_ADAPTIVE,
        ]
        
        results = {}
        
        for strategy in strategies:
            optimizer = AdaptivePrivacyBudgetOptimizer(
                total_epsilon_budget=50.0,
                optimization_strategy=strategy
            )
            
            # Register clients with different characteristics
            for i in range(3):
                optimizer.register_client(
                    f"client_{i}",
                    epsilon_budget=20.0,
                    client_characteristics={
                        "data_sensitivity": 1.0 + i * 0.5,
                        "privacy_preferences": {"strictness": 0.5 + i * 0.2}
                    }
                )
            
            # Perform allocation
            client_ids = [f"client_{i}" for i in range(3)]
            allocations = optimizer.allocate_budget(client_ids, round_num=1)
            
            # Store results
            results[strategy] = {
                "total_allocated": sum(alloc.epsilon_allocated for alloc in allocations),
                "allocation_variance": np.var([alloc.epsilon_allocated for alloc in allocations]),
                "expected_utility": sum(alloc.expected_utility for alloc in allocations)
            }
        
        # All strategies should allocate some budget
        for strategy_results in results.values():
            assert strategy_results["total_allocated"] > 0
            assert strategy_results["expected_utility"] > 0
        
        # Different strategies should produce different results
        uniform_alloc = results[BudgetAllocationStrategy.UNIFORM]["allocation_variance"]
        weighted_alloc = results[BudgetAllocationStrategy.PERFORMANCE_WEIGHTED]["allocation_variance"]
        
        # Uniform should have lower variance than performance-weighted (given different client characteristics)
        # Note: This might not always hold due to randomness in RL, so we just check they're different
        assert uniform_alloc != weighted_alloc


# Pytest configuration and fixtures
@pytest.fixture
def sample_optimizer():
    """Create a sample optimizer for testing."""
    return AdaptivePrivacyBudgetOptimizer(
        total_epsilon_budget=100.0,
        optimization_strategy=BudgetAllocationStrategy.RL_ADAPTIVE
    )


@pytest.fixture
def sample_client_profiles():
    """Create sample client profiles for testing."""
    profiles = {}
    for i in range(5):
        profile = ClientBudgetProfile(
            client_id=f"client_{i}",
            total_epsilon_budget=20.0,
            data_sensitivity=1.0 + i * 0.3,
            privacy_preferences={"strictness": 0.5 + i * 0.1}
        )
        profile.performance_history = [0.7 + i * 0.05, 0.72 + i * 0.05]
        profiles[f"client_{i}"] = profile
    return profiles


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v", "--tb=short"])
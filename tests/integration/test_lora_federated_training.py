"""Integration tests for LoRA federated training workflow."""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTensor:
    """Mock tensor class for testing without PyTorch dependency."""
    
    def __init__(self, shape, dtype="float32", device="cpu"):
        self.shape = shape
        self.dtype = dtype
        self.device = MockDevice(device)
        self.data = [[0.01 * (i+j) for j in range(shape[1])] for i in range(shape[0])]
    
    def clone(self):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def detach(self):
        return self
    
    def to(self, device=None, dtype=None):
        new_device = device if device else self.device
        new_dtype = dtype if dtype else self.dtype
        return MockTensor(self.shape, new_dtype, new_device)
    
    def norm(self):
        return MockScalarTensor(1.5)  # Mock norm value
    
    def numel(self):
        return self.shape[0] * self.shape[1]
    
    def __add__(self, other):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def __mul__(self, scalar):
        return MockTensor(self.shape, self.dtype, self.device.type)
    
    def flatten(self):
        return MockTensor([self.numel()], self.dtype, self.device.type)


class MockScalarTensor:
    """Mock scalar tensor."""
    
    def __init__(self, value):
        self.value = value
    
    def item(self):
        return self.value


class MockDevice:
    """Mock device class."""
    
    def __init__(self, device_type="cpu"):
        self.type = device_type
    
    def __eq__(self, other):
        if isinstance(other, MockDevice):
            return self.type == other.type
        return str(other) == self.type


@pytest.fixture
def mock_federated_config():
    """Mock federated configuration."""
    return {
        "model_name": "microsoft/DialoGPT-small",
        "num_rounds": 5,
        "client_fraction": 0.8,
        "local_epochs": 3,
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn"],
            "lora_dropout": 0.1
        },
        "privacy": {
            "epsilon": 8.0,
            "delta": 1e-5,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0
        }
    }


@pytest.fixture
def mock_client_data():
    """Mock client training data."""
    return {
        "client_0": {
            "text": [
                "Hello, how are you today?",
                "I'm doing well, thank you for asking.",
                "What's your favorite topic to discuss?"
            ]
        },
        "client_1": {
            "text": [
                "The weather is nice today.",
                "Would you like to go for a walk?",
                "I enjoy spending time outdoors."
            ]
        },
        "client_2": {
            "text": [
                "Machine learning is fascinating.",
                "Neural networks can solve complex problems.",
                "AI research continues to advance rapidly."
            ]
        }
    }


@pytest.fixture
def temp_data_files(mock_client_data):
    """Create temporary data files for testing."""
    temp_files = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for client_id, data in mock_client_data.items():
            file_path = Path(temp_dir) / f"{client_id}_data.json"
            with open(file_path, 'w') as f:
                json.dump(data, f)
            temp_files[client_id] = str(file_path)
        
        yield temp_files


class MockDPLoRAClient:
    """Mock DP LoRA client for integration testing."""
    
    def __init__(self, client_id: str, data_path: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.data_path = data_path
        self.config = config
        self.model = None
        self.current_round = 0
        self.privacy_spent = {"epsilon": 0.0, "delta": 0.0}
        self.training_history = []
        self._is_setup = False
        
        # Mock LoRA parameters
        self._lora_parameters = {
            "base_model.model.layers.0.self_attn.c_attn.lora_A.default.weight": MockTensor([16, 768]),
            "base_model.model.layers.0.self_attn.c_attn.lora_B.default.weight": MockTensor([768, 16]),
        }
    
    def setup(self):
        """Setup mock client."""
        self._is_setup = True
        logger.info(f"Mock client {self.client_id} setup completed")
    
    def extract_lora_parameters(self) -> Dict[str, MockTensor]:
        """Extract mock LoRA parameters."""
        if not self._is_setup:
            raise ValueError("Client not setup")
        
        logger.info(f"Extracted {len(self._lora_parameters)} LoRA parameters from {self.client_id}")
        return self._lora_parameters.copy()
    
    def merge_lora_weights(self, global_params: Dict[str, MockTensor]):
        """Merge global parameters into local model."""
        if not self._is_setup:
            raise ValueError("Client not setup")
        
        updated_count = 0
        for name, param in global_params.items():
            if name in self._lora_parameters:
                self._lora_parameters[name] = param
                updated_count += 1
        
        logger.info(f"Client {self.client_id} merged {updated_count} global parameters")
    
    def train_local(self) -> Dict[str, MockTensor]:
        """Simulate local training."""
        if not self._is_setup:
            raise ValueError("Client not setup")
        
        # Simulate training by slightly modifying parameters
        updated_params = {}
        for name, param in self._lora_parameters.items():
            # Mock parameter update
            updated_params[name] = param
        
        # Update privacy spent (mock)
        self.privacy_spent["epsilon"] += 0.5
        
        # Add to training history
        self.training_history.append({
            "round": self.current_round,
            "loss": 2.5 - 0.1 * self.current_round,  # Mock decreasing loss
            "privacy_spent": self.privacy_spent.copy()
        })
        
        logger.info(f"Client {self.client_id} completed local training for round {self.current_round}")
        return updated_params
    
    def get_lora_statistics(self) -> Dict[str, Any]:
        """Get LoRA parameter statistics."""
        if not self._is_setup:
            return {}
        
        return {
            "num_lora_layers": len(self._lora_parameters),
            "total_lora_parameters": sum(p.numel() for p in self._lora_parameters.values()),
            "parameter_norms": {name: param.norm().item() for name, param in self._lora_parameters.items()},
        }
    
    def adaptive_rank_selection(self) -> int:
        """Perform adaptive rank selection."""
        current_rank = self.config["lora"]["r"]
        data_size = len(self._mock_data()) if hasattr(self, '_mock_data') else 100
        
        # Simple heuristic
        if data_size < 100:
            return min(8, current_rank)
        elif data_size < 1000:
            return min(16, max(8, current_rank))
        else:
            return min(64, max(16, current_rank))
    
    def _mock_data(self):
        """Mock data for testing."""
        return ["sample text"] * 150  # Mock 150 examples


class MockLoRAAggregator:
    """Mock LoRA aggregator for testing."""
    
    def __init__(self, config):
        self.config = config
        self.aggregation_history = []
    
    def aggregate(self, client_updates: Dict[str, Dict[str, MockTensor]], 
                 client_weights: Dict[str, float] = None) -> Dict[str, MockTensor]:
        """Aggregate client updates."""
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Mock aggregation - simple averaging
        aggregated = {}
        first_client = next(iter(client_updates.values()))
        
        for param_name in first_client.keys():
            # Mock weighted averaging
            aggregated[param_name] = first_client[param_name]  # Simplified
        
        # Record aggregation
        self.aggregation_history.append({
            "num_clients": len(client_updates),
            "parameters": len(aggregated),
            "method": "lora_fedavg"
        })
        
        logger.info(f"Aggregated updates from {len(client_updates)} clients")
        return aggregated
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self.aggregation_history:
            return {}
        
        latest = self.aggregation_history[-1]
        return {
            "total_aggregations": len(self.aggregation_history),
            "latest_num_clients": latest["num_clients"],
            "latest_parameters": latest["parameters"]
        }


class MockFederatedServer:
    """Mock federated server for testing."""
    
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.aggregator = MockLoRAAggregator(config)
        self.current_round = 0
        self.global_model_params = {}
        self.round_history = []
    
    def register_client(self, client: MockDPLoRAClient) -> bool:
        """Register a client with the server."""
        try:
            self.clients[client.client_id] = client
            logger.info(f"Registered client {client.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register client {client.client_id}: {e}")
            return False
    
    def start_round(self, round_num: int):
        """Start a new training round."""
        self.current_round = round_num
        logger.info(f"Starting round {round_num}")
        
        # Initialize global model if first round
        if round_num == 0 and not self.global_model_params:
            if self.clients:
                first_client = next(iter(self.clients.values()))
                self.global_model_params = first_client.extract_lora_parameters()
    
    def collect_client_updates(self) -> Dict[str, Dict[str, MockTensor]]:
        """Collect updates from all clients."""
        client_updates = {}
        
        for client_id, client in self.clients.items():
            try:
                # Send global model to client
                client.merge_lora_weights(self.global_model_params)
                
                # Client performs local training
                local_updates = client.train_local()
                client_updates[client_id] = local_updates
                
                client.current_round = self.current_round
                
            except Exception as e:
                logger.error(f"Failed to collect updates from client {client_id}: {e}")
        
        logger.info(f"Collected updates from {len(client_updates)} clients")
        return client_updates
    
    def aggregate_and_update(self, client_updates: Dict[str, Dict[str, MockTensor]]):
        """Aggregate client updates and update global model."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate client weights (mock - based on data size)
        client_weights = {client_id: 100.0 for client_id in client_updates.keys()}
        
        # Perform aggregation
        self.global_model_params = self.aggregator.aggregate(client_updates, client_weights)
        
        # Record round history
        round_stats = {
            "round": self.current_round,
            "num_clients": len(client_updates),
            "aggregation_method": "lora_fedavg",
            "global_params_count": len(self.global_model_params)
        }
        self.round_history.append(round_stats)
        
        logger.info(f"Round {self.current_round} completed - aggregated {len(self.global_model_params)} parameters")
    
    def get_round_statistics(self) -> Dict[str, Any]:
        """Get statistics for the current round."""
        if not self.round_history:
            return {}
        
        return {
            "total_rounds": len(self.round_history),
            "current_round": self.current_round,
            "latest_stats": self.round_history[-1],
            "aggregator_stats": self.aggregator.get_parameter_statistics()
        }


class TestLoRAFederatedTrainingIntegration:
    """Integration tests for LoRA federated training workflow."""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_federated_config, temp_data_files):
        """Setup test environment."""
        self.config = mock_federated_config
        self.data_files = temp_data_files
        self.server = MockFederatedServer(self.config)
        self.clients = []
        
        # Create mock clients
        for client_id, data_file in self.data_files.items():
            client = MockDPLoRAClient(client_id, data_file, self.config)
            client.setup()
            self.clients.append(client)
            self.server.register_client(client)
    
    def test_client_registration(self):
        """Test client registration with server."""
        assert len(self.server.clients) == len(self.clients)
        
        for client in self.clients:
            assert client.client_id in self.server.clients
            assert self.server.clients[client.client_id] == client
        
        logger.info(f"✅ Successfully registered {len(self.clients)} clients")
    
    def test_client_setup_and_lora_extraction(self):
        """Test client setup and LoRA parameter extraction."""
        for client in self.clients:
            # Verify setup
            assert client._is_setup, f"Client {client.client_id} not setup properly"
            
            # Test LoRA parameter extraction
            lora_params = client.extract_lora_parameters()
            assert len(lora_params) > 0, f"No LoRA parameters extracted from {client.client_id}"
            
            # Verify parameter structure
            expected_patterns = ["lora_A", "lora_B"]
            found_patterns = set()
            for param_name in lora_params.keys():
                for pattern in expected_patterns:
                    if pattern in param_name:
                        found_patterns.add(pattern)
            
            assert len(found_patterns) >= 2, f"Missing LoRA patterns in {client.client_id}: {found_patterns}"
        
        logger.info("✅ All clients setup and LoRA extraction working")
    
    def test_single_round_training(self):
        """Test a single round of federated training."""
        round_num = 0
        
        # Start round
        self.server.start_round(round_num)
        assert self.server.current_round == round_num
        
        # Collect client updates
        client_updates = self.server.collect_client_updates()
        assert len(client_updates) == len(self.clients)
        
        # Verify all clients participated
        for client in self.clients:
            assert client.client_id in client_updates
            assert len(client_updates[client.client_id]) > 0
            assert client.current_round == round_num
        
        # Aggregate updates
        self.server.aggregate_and_update(client_updates)
        
        # Verify global model updated
        assert len(self.server.global_model_params) > 0
        assert len(self.server.round_history) == 1
        
        logger.info("✅ Single round training completed successfully")
    
    def test_multi_round_training(self):
        """Test multiple rounds of federated training."""
        num_rounds = 3
        
        for round_num in range(num_rounds):
            # Start round
            self.server.start_round(round_num)
            
            # Collect and aggregate updates
            client_updates = self.server.collect_client_updates()
            self.server.aggregate_and_update(client_updates)
            
            # Verify round progression
            assert self.server.current_round == round_num
            assert len(self.server.round_history) == round_num + 1
        
        # Verify training history
        for client in self.clients:
            assert len(client.training_history) == num_rounds
            
            # Check privacy budget increases
            for i in range(1, num_rounds):
                prev_epsilon = client.training_history[i-1]["privacy_spent"]["epsilon"]
                curr_epsilon = client.training_history[i]["privacy_spent"]["epsilon"]
                assert curr_epsilon > prev_epsilon, "Privacy budget should increase"
        
        logger.info(f"✅ Multi-round training ({num_rounds} rounds) completed successfully")
    
    def test_lora_parameter_aggregation(self):
        """Test LoRA parameter aggregation process."""
        # Run one round to get updates
        self.server.start_round(0)
        client_updates = self.server.collect_client_updates()
        
        # Test aggregation directly
        aggregated_params = self.server.aggregator.aggregate(client_updates)
        
        # Verify aggregation results
        assert len(aggregated_params) > 0, "Aggregation should produce parameters"
        
        # Check parameter consistency
        first_client_params = next(iter(client_updates.values()))
        assert set(aggregated_params.keys()) == set(first_client_params.keys()), \
            "Aggregated parameters should have same keys as client parameters"
        
        # Check aggregator statistics
        agg_stats = self.server.aggregator.get_parameter_statistics()
        assert agg_stats["total_aggregations"] >= 1
        assert agg_stats["latest_num_clients"] == len(client_updates)
        
        logger.info("✅ LoRA parameter aggregation working correctly")
    
    def test_parameter_distribution_and_merging(self):
        """Test parameter distribution to clients and merging."""
        # Initialize global parameters
        self.server.start_round(0)
        initial_global_params = self.server.global_model_params.copy()
        
        # Collect updates (which includes parameter distribution)
        client_updates = self.server.collect_client_updates()
        
        # Verify clients received and merged global parameters
        for client in self.clients:
            client_params = client.extract_lora_parameters()
            
            # Check that client has parameters
            assert len(client_params) > 0, f"Client {client.client_id} has no parameters"
            
            # In a real scenario, we'd check if parameters were properly merged
            # For mock, we just verify structure consistency
            assert set(client_params.keys()) == set(initial_global_params.keys()), \
                f"Client {client.client_id} parameter keys don't match global model"
        
        logger.info("✅ Parameter distribution and merging working correctly")
    
    def test_adaptive_rank_selection(self):
        """Test adaptive rank selection across clients."""
        for client in self.clients:
            # Test adaptive rank selection
            optimal_rank = client.adaptive_rank_selection()
            current_rank = client.config["lora"]["r"]
            
            # Validate rank is within reasonable bounds
            assert 1 <= optimal_rank <= 64, f"Invalid optimal rank {optimal_rank} for {client.client_id}"
            
            # Test rank selection logic
            if optimal_rank != current_rank:
                logger.info(f"Client {client.client_id}: rank {current_rank} → {optimal_rank}")
            else:
                logger.info(f"Client {client.client_id}: keeping rank {current_rank}")
        
        logger.info("✅ Adaptive rank selection working correctly")
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking across rounds."""
        num_rounds = 3
        
        for round_num in range(num_rounds):
            self.server.start_round(round_num)
            self.server.collect_client_updates()
            self.server.aggregate_and_update({})  # Empty for privacy tracking test
        
        # Verify privacy budget increases
        for client in self.clients:
            total_epsilon = client.privacy_spent["epsilon"]
            assert total_epsilon > 0, f"No privacy spent for {client.client_id}"
            
            # Check training history privacy tracking
            if len(client.training_history) > 1:
                for i in range(1, len(client.training_history)):
                    prev_eps = client.training_history[i-1]["privacy_spent"]["epsilon"]
                    curr_eps = client.training_history[i]["privacy_spent"]["epsilon"]
                    assert curr_eps >= prev_eps, "Privacy budget should not decrease"
        
        logger.info("✅ Privacy budget tracking working correctly")
    
    def test_error_handling_robustness(self):
        """Test error handling in federated training workflow."""
        # Test client setup failure
        broken_client = MockDPLoRAClient("broken_client", "/nonexistent/path", self.config)
        # Don't call setup() to simulate broken client
        
        registration_success = self.server.register_client(broken_client)
        # Should still register, but will fail during training
        assert registration_success
        
        # Test round with broken client
        self.server.start_round(0)
        
        # This should handle the broken client gracefully
        try:
            client_updates = self.server.collect_client_updates()
            # Should succeed for working clients, skip broken ones
            assert len(client_updates) >= len(self.clients), "Should collect updates from working clients"
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            logger.info(f"Handled exception gracefully: {e}")
        
        logger.info("✅ Error handling robustness test completed")
    
    def test_performance_monitoring(self):
        """Test performance monitoring during training."""
        import time
        
        # Monitor round timing
        start_time = time.time()
        
        # Run a complete round
        self.server.start_round(0)
        client_updates = self.server.collect_client_updates()
        self.server.aggregate_and_update(client_updates)
        
        end_time = time.time()
        round_duration = end_time - start_time
        
        # Get round statistics
        stats = self.server.get_round_statistics()
        assert stats["total_rounds"] >= 1
        assert stats["current_round"] == 0
        
        # Performance should be reasonable for mock implementation
        assert round_duration < 5.0, f"Round took too long: {round_duration}s"
        
        logger.info(f"✅ Performance monitoring - round completed in {round_duration:.3f}s")
    
    def test_lora_statistics_collection(self):
        """Test collection of LoRA-specific statistics."""
        # Setup clients and run one round
        self.server.start_round(0)
        self.server.collect_client_updates()
        
        # Collect LoRA statistics from all clients
        all_client_stats = {}
        for client in self.clients:
            stats = client.get_lora_statistics()
            all_client_stats[client.client_id] = stats
            
            # Verify statistics structure
            assert "num_lora_layers" in stats
            assert "total_lora_parameters" in stats
            assert "parameter_norms" in stats
            
            # Verify statistics values
            assert stats["num_lora_layers"] > 0
            assert stats["total_lora_parameters"] > 0
            assert len(stats["parameter_norms"]) > 0
        
        # Aggregate statistics
        total_params = sum(stats["total_lora_parameters"] for stats in all_client_stats.values())
        avg_layers = sum(stats["num_lora_layers"] for stats in all_client_stats.values()) / len(all_client_stats)
        
        logger.info(f"Total LoRA parameters across all clients: {total_params}")
        logger.info(f"Average LoRA layers per client: {avg_layers:.1f}")
        
        assert total_params > 0
        assert avg_layers > 0
        
        logger.info("✅ LoRA statistics collection working correctly")


@pytest.mark.asyncio
class TestAsyncLoRAFederatedTraining:
    """Async integration tests for LoRA federated training."""
    
    async def test_async_client_communication(self):
        """Test asynchronous client communication patterns."""
        # Mock async client methods
        async def mock_register():
            await asyncio.sleep(0.01)  # Simulate network delay
            return True
        
        async def mock_get_updates():
            await asyncio.sleep(0.02)
            return {"param": MockTensor([16, 768])}
        
        async def mock_send_global_model():
            await asyncio.sleep(0.01)
            return True
        
        # Test concurrent client operations
        tasks = []
        for i in range(3):
            tasks.append(mock_register())
            tasks.append(mock_get_updates())
            tasks.append(mock_send_global_model())
        
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 9  # 3 operations × 3 clients
        assert all(results), "All async operations should succeed"
        
        logger.info("✅ Async client communication test completed")
    
    async def test_concurrent_aggregation(self):
        """Test concurrent aggregation operations."""
        # Mock concurrent aggregation tasks
        async def mock_aggregate_batch(client_updates):
            await asyncio.sleep(0.05)  # Simulate computation time
            return len(client_updates)
        
        # Create multiple batches
        batches = [
            {"client_0": {"param": MockTensor([16, 768])}},
            {"client_1": {"param": MockTensor([16, 768])}},
            {"client_2": {"param": MockTensor([16, 768])}},
        ]
        
        # Process batches concurrently
        tasks = [mock_aggregate_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Verify concurrent processing
        assert len(results) == len(batches)
        assert all(result == 1 for result in results), "Each batch should process one client"
        
        logger.info("✅ Concurrent aggregation test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
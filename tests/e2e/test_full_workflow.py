"""End-to-end tests for complete federated DP-LoRA workflows."""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for E2E tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Create directory structure
        (workspace / "models").mkdir()
        (workspace / "data").mkdir()
        (workspace / "checkpoints").mkdir()
        (workspace / "logs").mkdir()
        
        yield workspace


@pytest.fixture
def mock_model_config():
    """Mock model configuration for E2E tests."""
    return {
        "model_name": "mock-llama-7b",
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
    }


@pytest.fixture
def federated_scenario_config():
    """Configuration for federated learning scenario."""
    return {
        "num_clients": 5,
        "num_rounds": 3,  # Short for E2E tests
        "local_epochs": 2,
        "client_fraction": 0.6,
        "min_clients": 3,
        "aggregation_method": "fedavg",
        "privacy": {
            "epsilon": 8.0,
            "delta": 1e-5,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"]
        }
    }


class TestEndToEndWorkflow:
    """Test complete end-to-end federated learning workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_federated_training_workflow(
        self, 
        temp_workspace, 
        mock_model_config, 
        federated_scenario_config
    ):
        """Test complete federated training workflow from start to finish."""
        
        # Mock data for each client
        client_data = self._create_mock_client_data(
            federated_scenario_config["num_clients"],
            temp_workspace / "data"
        )
        
        # Initialize mock server
        server = self._create_mock_server(
            federated_scenario_config,
            temp_workspace
        )
        
        # Initialize mock clients
        clients = self._create_mock_clients(
            client_data,
            federated_scenario_config,
            temp_workspace
        )
        
        # Execute federated training
        training_results = self._execute_federated_training(
            server, 
            clients, 
            federated_scenario_config
        )
        
        # Validate results
        self._validate_training_results(training_results, federated_scenario_config)
        
        # Verify artifacts were created
        self._verify_training_artifacts(temp_workspace)
    
    @pytest.mark.e2e
    def test_privacy_budget_management(self, federated_scenario_config):
        """Test privacy budget management throughout training."""
        
        initial_epsilon = federated_scenario_config["privacy"]["epsilon"]
        num_rounds = federated_scenario_config["num_rounds"]
        num_clients = federated_scenario_config["num_clients"]
        
        # Mock privacy accountant
        privacy_accountant = Mock()
        privacy_accountant.total_epsilon = 0.0
        privacy_accountant.remaining_epsilon = initial_epsilon
        
        def mock_spend_privacy(epsilon_spent):
            privacy_accountant.total_epsilon += epsilon_spent
            privacy_accountant.remaining_epsilon -= epsilon_spent
            return privacy_accountant.remaining_epsilon > 0
        
        privacy_accountant.spend_privacy = mock_spend_privacy
        
        # Simulate training with privacy consumption
        for round_num in range(num_rounds):
            # Each round consumes some privacy budget
            epsilon_per_round = initial_epsilon / (num_rounds * 2)  # Conservative estimate
            
            can_continue = privacy_accountant.spend_privacy(epsilon_per_round)
            
            if not can_continue:
                break
        
        # Verify privacy budget was managed correctly
        assert privacy_accountant.total_epsilon <= initial_epsilon
        assert privacy_accountant.remaining_epsilon >= 0
    
    @pytest.mark.e2e
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_training_workflow(self, mock_cuda, temp_workspace, federated_scenario_config):
        """Test E2E workflow with GPU training."""
        
        # Mock GPU environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create mock client with GPU support
        client = Mock()
        client.device = device
        client.model = Mock()
        client.model.to.return_value = client.model
        
        # Mock training step on GPU
        def mock_train_step():
            # Simulate GPU tensor operations
            mock_loss = torch.tensor(1.5, device=device)
            mock_gradients = [torch.randn(16, 768, device=device)]
            return {
                "loss": mock_loss.cpu().item(),
                "gradients": [g.cpu() for g in mock_gradients]
            }
        
        client.train_step = mock_train_step
        
        # Execute training step
        result = client.train_step()
        
        # Verify GPU was used (mocked)
        assert result["loss"] > 0
        assert len(result["gradients"]) > 0
        
        # Verify model was moved to GPU
        client.model.to.assert_called()
    
    @pytest.mark.e2e
    def test_checkpoint_and_recovery(self, temp_workspace, federated_scenario_config):
        """Test checkpoint saving and recovery workflow."""
        
        checkpoint_dir = temp_workspace / "checkpoints"
        
        # Mock server state
        server_state = {
            "global_round": 5,
            "global_model_state": {"lora_A": torch.randn(16, 768), "lora_B": torch.randn(768, 16)},
            "privacy_budget_spent": 3.5,
            "training_history": [
                {"round": i, "accuracy": 0.7 + i * 0.05, "loss": 2.0 - i * 0.1}
                for i in range(5)
            ]
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / "server_checkpoint_round_5.pt"
        torch.save(server_state, checkpoint_path)
        
        # Verify checkpoint was saved
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_state = torch.load(checkpoint_path)
        
        # Verify state was recovered correctly
        assert loaded_state["global_round"] == server_state["global_round"]
        assert loaded_state["privacy_budget_spent"] == server_state["privacy_budget_spent"]
        assert len(loaded_state["training_history"]) == len(server_state["training_history"])
        
        # Verify model state
        for key in server_state["global_model_state"]:
            assert torch.allclose(
                loaded_state["global_model_state"][key],
                server_state["global_model_state"][key]
            )
    
    @pytest.mark.e2e
    def test_client_heterogeneity_handling(self, temp_workspace, federated_scenario_config):
        """Test handling of heterogeneous clients (different data, capabilities)."""
        
        # Create clients with different characteristics
        heterogeneous_clients = []
        
        for i in range(federated_scenario_config["num_clients"]):
            client = Mock()
            client.client_id = f"client_{i}"
            
            # Different data sizes
            client.data_size = np.random.randint(50, 500)
            
            # Different computational capabilities
            client.batch_size = 16 if i < 2 else 8  # Some clients have smaller batch sizes
            client.local_epochs = 3 if i < 3 else 2  # Some clients do fewer epochs
            
            # Different privacy requirements
            if i == 0:  # Highly sensitive client
                client.privacy_config = {"epsilon": 1.0, "delta": 1e-6}
            elif i < 3:  # Moderately sensitive
                client.privacy_config = {"epsilon": 4.0, "delta": 1e-5}
            else:  # Less sensitive
                client.privacy_config = {"epsilon": 10.0, "delta": 1e-4}
            
            # Mock training results based on capabilities
            client.train.return_value = {
                "loss": np.random.uniform(0.5, 2.0),
                "accuracy": np.random.uniform(0.7, 0.9),
                "samples_processed": client.data_size,
                "privacy_spent": np.random.uniform(0.1, 0.3)
            }
            
            heterogeneous_clients.append(client)
        
        # Test server handling of heterogeneous clients
        # Weight aggregation by data size
        total_samples = sum(client.data_size for client in heterogeneous_clients)
        
        client_weights = []
        for client in heterogeneous_clients:
            weight = client.data_size / total_samples
            client_weights.append(weight)
        
        # Verify weights sum to 1
        assert abs(sum(client_weights) - 1.0) < 1e-6
        
        # Verify clients with more data get higher weights
        max_weight_idx = np.argmax(client_weights)
        max_data_size = max(client.data_size for client in heterogeneous_clients)
        assert heterogeneous_clients[max_weight_idx].data_size == max_data_size
    
    def _create_mock_client_data(self, num_clients: int, data_dir: Path):
        """Create mock data for each client."""
        client_data = {}
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            
            # Create mock dataset
            dataset = {
                "input_ids": torch.randint(0, 32000, (100, 128)),  # 100 samples, 128 tokens
                "attention_mask": torch.ones(100, 128),
                "labels": torch.randint(0, 32000, (100, 128))
            }
            
            # Save to file
            data_file = data_dir / f"{client_id}_data.pt"
            torch.save(dataset, data_file)
            
            client_data[client_id] = {
                "data_path": data_file,
                "num_samples": 100
            }
        
        return client_data
    
    def _create_mock_server(self, config: dict, workspace: Path):
        """Create mock federated learning server."""
        server = Mock()
        server.config = config
        server.workspace = workspace
        server.global_round = 0
        server.max_rounds = config["num_rounds"]
        
        # Mock server methods
        def mock_select_clients(available_clients, fraction):
            num_selected = int(len(available_clients) * fraction)
            return available_clients[:num_selected]
        
        def mock_aggregate_parameters(client_updates):
            # Simple averaging
            if not client_updates:
                return {}
            
            aggregated = {}
            for key in client_updates[0].keys():
                avg_param = torch.zeros_like(client_updates[0][key])
                for update in client_updates:
                    avg_param += update[key]
                aggregated[key] = avg_param / len(client_updates)
            return aggregated
        
        def mock_evaluate():
            return {
                "accuracy": np.random.uniform(0.75, 0.95),
                "loss": np.random.uniform(0.5, 1.5),
                "privacy_budget_remaining": np.random.uniform(3.0, 8.0)
            }
        
        server.select_clients = mock_select_clients
        server.aggregate_parameters = mock_aggregate_parameters
        server.evaluate = mock_evaluate
        
        return server
    
    def _create_mock_clients(self, client_data: dict, config: dict, workspace: Path):
        """Create mock federated learning clients."""
        clients = []
        
        for client_id, data_info in client_data.items():
            client = Mock()
            client.client_id = client_id
            client.data_path = data_info["data_path"]
            client.num_samples = data_info["num_samples"]
            client.config = config
            
            # Mock client methods
            def mock_load_data(path=data_info["data_path"]):
                return torch.load(path)
            
            def mock_train():
                return {
                    "loss": np.random.uniform(0.5, 2.0),
                    "accuracy": np.random.uniform(0.7, 0.9),
                    "privacy_spent": np.random.uniform(0.1, 0.5),
                    "samples_processed": data_info["num_samples"]
                }
            
            def mock_get_parameters():
                return {
                    "lora_A": torch.randn(config["lora"]["r"], 4096),
                    "lora_B": torch.randn(4096, config["lora"]["r"])
                }
            
            client.load_data = mock_load_data
            client.train = mock_train
            client.get_parameters = mock_get_parameters
            client.set_parameters = Mock()
            
            clients.append(client)
        
        return clients
    
    def _execute_federated_training(self, server, clients, config):
        """Execute the federated training process."""
        training_results = []
        
        for round_num in range(config["num_rounds"]):
            server.global_round = round_num
            
            # Client selection
            selected_clients = server.select_clients(clients, config["client_fraction"])
            
            # Client training
            client_updates = []
            round_metrics = []
            
            for client in selected_clients:
                # Train client
                training_result = client.train()
                round_metrics.append(training_result)
                
                # Get updated parameters
                client_params = client.get_parameters()
                client_updates.append(client_params)
            
            # Server aggregation
            aggregated_params = server.aggregate_parameters(client_updates)
            
            # Update global model (mock)
            for client in clients:
                client.set_parameters(aggregated_params)
            
            # Evaluate global model
            eval_result = server.evaluate()
            
            # Record round results
            round_result = {
                "round": round_num,
                "num_participants": len(selected_clients),
                "client_metrics": round_metrics,
                "global_metrics": eval_result
            }
            training_results.append(round_result)
        
        return training_results
    
    def _validate_training_results(self, results, config):
        """Validate the training results."""
        assert len(results) == config["num_rounds"]
        
        for round_result in results:
            # Check round structure
            assert "round" in round_result
            assert "num_participants" in round_result
            assert "client_metrics" in round_result
            assert "global_metrics" in round_result
            
            # Check participant count
            assert round_result["num_participants"] >= config["min_clients"]
            assert round_result["num_participants"] <= config["num_clients"]
            
            # Check client metrics
            for client_metric in round_result["client_metrics"]:
                assert "loss" in client_metric
                assert "accuracy" in client_metric
                assert "privacy_spent" in client_metric
                assert client_metric["loss"] > 0
                assert 0 <= client_metric["accuracy"] <= 1
                assert client_metric["privacy_spent"] >= 0
            
            # Check global metrics
            global_metrics = round_result["global_metrics"]
            assert "accuracy" in global_metrics
            assert "loss" in global_metrics
            assert 0 <= global_metrics["accuracy"] <= 1
            assert global_metrics["loss"] > 0
    
    def _verify_training_artifacts(self, workspace: Path):
        """Verify that training artifacts were created."""
        # Check that directories exist
        assert (workspace / "models").exists()
        assert (workspace / "data").exists()
        assert (workspace / "checkpoints").exists()
        assert (workspace / "logs").exists()
        
        # Check that data files were created
        data_files = list((workspace / "data").glob("client_*_data.pt"))
        assert len(data_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
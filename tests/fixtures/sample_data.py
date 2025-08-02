"""Sample data fixtures for testing federated learning components."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path


class SampleDataGenerator:
    """Generate sample data for testing federated learning scenarios."""
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    def generate_text_dataset(
        self, 
        num_samples: int, 
        min_length: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Generate a mock text dataset for language modeling."""
        
        # Generate random token sequences
        lengths = self.rng.randint(min_length, self.max_length + 1, num_samples)
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for length in lengths:
            # Generate input sequence
            seq = self.rng.randint(1, self.vocab_size, length)  # Avoid 0 (padding token)
            
            # Pad to max_length
            padded_seq = np.zeros(self.max_length, dtype=np.int64)
            padded_seq[:length] = seq
            
            # Create attention mask
            attention_mask = np.zeros(self.max_length, dtype=np.int64)
            attention_mask[:length] = 1
            
            # Labels for causal language modeling (shifted input)
            labels_seq = np.full(self.max_length, -100, dtype=np.int64)  # -100 = ignore index
            if length > 1:
                labels_seq[:length-1] = seq[1:]  # Predict next token
            
            input_ids.append(padded_seq)
            attention_masks.append(attention_mask)
            labels.append(labels_seq)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def generate_federated_datasets(
        self,
        num_clients: int,
        samples_per_client: List[int],
        heterogeneity: str = "iid"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate datasets for multiple federated clients."""
        
        assert len(samples_per_client) == num_clients, \
            "samples_per_client must have length equal to num_clients"
        
        client_datasets = {}
        
        if heterogeneity == "iid":
            # IID: Each client gets random samples from same distribution
            for i in range(num_clients):
                client_id = f"client_{i}"
                client_datasets[client_id] = self.generate_text_dataset(samples_per_client[i])
        
        elif heterogeneity == "non_iid_quantity":
            # Non-IID quantity: Clients have different amounts of data
            for i in range(num_clients):
                client_id = f"client_{i}"
                client_datasets[client_id] = self.generate_text_dataset(samples_per_client[i])
        
        elif heterogeneity == "non_iid_label":
            # Non-IID label: Clients have different vocabulary distributions
            for i in range(num_clients):
                client_id = f"client_{i}"
                
                # Create biased vocabulary for each client
                vocab_bias = self._create_vocabulary_bias(i, num_clients)
                client_datasets[client_id] = self._generate_biased_dataset(
                    samples_per_client[i], vocab_bias
                )
        
        else:
            raise ValueError(f"Unknown heterogeneity type: {heterogeneity}")
        
        return client_datasets
    
    def _create_vocabulary_bias(self, client_id: int, num_clients: int) -> np.ndarray:
        """Create vocabulary bias for non-IID label distribution."""
        # Create probability distribution favoring certain vocab ranges
        bias = np.ones(self.vocab_size)
        
        # Each client favors different vocabulary ranges
        vocab_per_client = self.vocab_size // num_clients
        start_idx = client_id * vocab_per_client
        end_idx = min((client_id + 1) * vocab_per_client, self.vocab_size)
        
        # Increase probability for client's vocabulary range
        bias[start_idx:end_idx] *= 3.0
        
        # Normalize to create probability distribution
        bias = bias / bias.sum()
        
        return bias
    
    def _generate_biased_dataset(
        self, 
        num_samples: int, 
        vocab_bias: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Generate dataset with vocabulary bias."""
        
        lengths = self.rng.randint(10, self.max_length + 1, num_samples)
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for length in lengths:
            # Generate biased token sequence
            seq = self.rng.choice(
                self.vocab_size, 
                size=length, 
                p=vocab_bias
            )
            seq = np.maximum(seq, 1)  # Avoid padding token
            
            # Pad to max_length
            padded_seq = np.zeros(self.max_length, dtype=np.int64)
            padded_seq[:length] = seq
            
            # Create attention mask
            attention_mask = np.zeros(self.max_length, dtype=np.int64)
            attention_mask[:length] = 1
            
            # Labels
            labels_seq = np.full(self.max_length, -100, dtype=np.int64)
            if length > 1:
                labels_seq[:length-1] = seq[1:]
            
            input_ids.append(padded_seq)
            attention_masks.append(attention_mask)
            labels.append(labels_seq)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class PrivacyTestDataGenerator:
    """Generate data for privacy-specific tests."""
    
    @staticmethod
    def generate_gradient_data(
        batch_size: int, 
        model_params: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """Generate mock gradients for privacy testing."""
        gradients = []
        
        for param_shape in model_params:
            grad = torch.randn(*param_shape, requires_grad=False)
            gradients.append(grad)
        
        return gradients
    
    @staticmethod
    def generate_privacy_scenarios() -> List[Dict[str, Any]]:
        """Generate various privacy scenarios for testing."""
        scenarios = [
            {
                "name": "high_privacy",
                "epsilon": 1.0,
                "delta": 1e-6,
                "noise_multiplier": 2.0,
                "max_grad_norm": 0.5,
                "expected_utility_loss": 0.15  # High privacy = more utility loss
            },
            {
                "name": "medium_privacy",
                "epsilon": 4.0,
                "delta": 1e-5,
                "noise_multiplier": 1.1,
                "max_grad_norm": 1.0,
                "expected_utility_loss": 0.08
            },
            {
                "name": "low_privacy",
                "epsilon": 10.0,
                "delta": 1e-4,
                "noise_multiplier": 0.8,
                "max_grad_norm": 2.0,
                "expected_utility_loss": 0.03
            }
        ]
        
        return scenarios


class FederatedScenarioGenerator:
    """Generate federated learning scenarios for testing."""
    
    @staticmethod
    def generate_healthcare_scenario() -> Dict[str, Any]:
        """Generate healthcare federated learning scenario."""
        return {
            "name": "healthcare_federation",
            "num_clients": 10,
            "client_names": [f"hospital_{i}" for i in range(10)],
            "data_distribution": "non_iid_label",  # Each hospital has different patient populations
            "samples_per_client": [100, 150, 200, 80, 120, 300, 250, 90, 180, 110],
            "privacy_requirements": {
                "epsilon": 1.0,  # HIPAA-compliant strict privacy
                "delta": 1e-6
            },
            "heterogeneity": {
                "compute_power": [0.8, 1.0, 0.6, 1.2, 0.9, 1.5, 1.1, 0.7, 1.0, 0.8],
                "network_bandwidth": ["low", "high", "medium", "high", "medium", "high", "high", "low", "medium", "medium"]
            }
        }
    
    @staticmethod
    def generate_financial_scenario() -> Dict[str, Any]:
        """Generate financial federated learning scenario."""
        return {
            "name": "financial_federation",
            "num_clients": 5,
            "client_names": [f"bank_{i}" for i in range(5)],
            "data_distribution": "non_iid_quantity",  # Different sized banks
            "samples_per_client": [1000, 500, 2000, 800, 300],  # Large variation in data size
            "privacy_requirements": {
                "epsilon": 4.0,  # Moderate privacy for financial data
                "delta": 1e-5
            },
            "heterogeneity": {
                "compute_power": [2.0, 1.5, 3.0, 1.8, 1.0],  # Banks have good compute resources
                "network_bandwidth": ["high", "high", "high", "medium", "medium"]
            }
        }
    
    @staticmethod
    def generate_research_scenario() -> Dict[str, Any]:
        """Generate research collaboration scenario."""
        return {
            "name": "research_federation",
            "num_clients": 8,
            "client_names": [f"university_{i}" for i in range(8)],
            "data_distribution": "iid",  # Research data is more uniform
            "samples_per_client": [200] * 8,  # Equal data distribution
            "privacy_requirements": {
                "epsilon": 8.0,  # More relaxed privacy for research
                "delta": 1e-5
            },
            "heterogeneity": {
                "compute_power": [1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0],
                "network_bandwidth": ["medium"] * 8
            }
        }


def save_test_datasets(datasets: Dict[str, Any], output_dir: Path) -> None:
    """Save test datasets to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for client_id, dataset in datasets.items():
        # Save PyTorch tensors
        dataset_file = output_dir / f"{client_id}_dataset.pt"
        torch.save(dataset, dataset_file)
        
        # Save metadata
        metadata = {
            "client_id": client_id,
            "num_samples": len(dataset["input_ids"]),
            "vocab_size": dataset["input_ids"].max().item() + 1,
            "max_length": dataset["input_ids"].shape[1]
        }
        
        metadata_file = output_dir / f"{client_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def load_test_dataset(dataset_path: Path) -> Dict[str, torch.Tensor]:
    """Load test dataset from file."""
    return torch.load(dataset_path)


# Predefined test fixtures
SMALL_TEST_DATASETS = {
    "client_0": {
        "input_ids": torch.randint(1, 1000, (10, 32)),
        "attention_mask": torch.ones(10, 32),
        "labels": torch.randint(1, 1000, (10, 32))
    },
    "client_1": {
        "input_ids": torch.randint(1, 1000, (15, 32)),
        "attention_mask": torch.ones(15, 32),
        "labels": torch.randint(1, 1000, (15, 32))
    }
}

MOCK_MODEL_PARAMETERS = [
    (768, 16),    # LoRA A for q_proj
    (16, 768),    # LoRA B for q_proj
    (768, 16),    # LoRA A for v_proj
    (16, 768),    # LoRA B for v_proj
]

PRIVACY_TEST_CONFIGS = [
    {"epsilon": 1.0, "delta": 1e-6, "noise_multiplier": 2.0},
    {"epsilon": 4.0, "delta": 1e-5, "noise_multiplier": 1.1},
    {"epsilon": 8.0, "delta": 1e-5, "noise_multiplier": 0.8},
]
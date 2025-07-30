"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing."""
    model = Mock()
    model.parameters.return_value = [
        torch.randn(10, 10, requires_grad=True),
        torch.randn(10, requires_grad=True),
    ]
    return model


@pytest.fixture
def privacy_config():
    """Standard privacy configuration for tests."""
    return {
        "epsilon": 8.0,
        "delta": 1e-5,
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
    }


@pytest.fixture
def lora_config():
    """Standard LoRA configuration for tests."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    }


@pytest.fixture
def federated_config():
    """Standard federated learning configuration."""
    return {
        "num_clients": 10,
        "rounds": 50,
        "local_epochs": 3,
        "client_sampling_rate": 0.5,
    }


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    return {
        "input_ids": torch.randint(0, 1000, (32, 128)),
        "attention_mask": torch.ones(32, 128),
        "labels": torch.randint(0, 1000, (32, 128)),
    }


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (128,)),
            "attention_mask": torch.ones(128),
            "labels": torch.randint(0, 1000, (128,)),
        }


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return MockDataset()


@pytest.fixture
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    return torch.cuda.is_available()


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "privacy: mark test as privacy-related")
    config.addinivalue_line("markers", "federated: mark test as federated learning test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run GPU tests"
    )
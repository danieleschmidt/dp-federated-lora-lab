"""
DP-Federated LoRA Lab: Differentially Private Federated Learning with LoRA.

This package provides implementations of differentially private federated learning
algorithms specifically designed for fine-tuning large language models using
Low-Rank Adaptation (LoRA) techniques.

Key Components:
- Federated server and client implementations
- Differential privacy mechanisms (DP-SGD, RDP accounting)
- Secure aggregation protocols
- Byzantine-robust algorithms
- Privacy-utility monitoring and benchmarking

Example:
    >>> from dp_federated_lora import FederatedServer, DPLoRAClient
    >>> server = FederatedServer(model_name="meta-llama/Llama-2-7b-hf")
    >>> client = DPLoRAClient(client_id="client_1", data_path="data.json")
    >>> history = server.train(clients=[client], rounds=10)
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"
__license__ = "MIT"

# Core components
from .client import DPLoRAClient
from .server import FederatedServer

# Privacy components
from .privacy import PrivacyEngine, PrivacyAccountant

# Aggregation protocols
from .aggregation import SecureAggregator, ByzantineRobustAggregator

# Monitoring and utilities
from .monitoring import UtilityMonitor
from .config import FederatedConfig, PrivacyConfig, LoRAConfig

__all__ = [
    # Core
    "FederatedServer",
    "DPLoRAClient",
    # Privacy
    "PrivacyEngine", 
    "PrivacyAccountant",
    # Aggregation
    "SecureAggregator",
    "ByzantineRobustAggregator",
    # Utilities
    "UtilityMonitor",
    "FederatedConfig",
    "PrivacyConfig", 
    "LoRAConfig",
]
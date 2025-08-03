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
try:
    from .client import DPLoRAClient
    from .server import FederatedServer, TrainingHistory
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import core components: {e}")
    DPLoRAClient = None
    FederatedServer = None
    TrainingHistory = None

# Privacy components  
try:
    from .privacy import PrivacyEngine, PrivacyAccountant
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import privacy components: {e}")
    PrivacyEngine = None
    PrivacyAccountant = None

# Aggregation protocols
try:
    from .aggregation import SecureAggregator, ByzantineRobustAggregator, create_aggregator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import aggregation components: {e}")
    SecureAggregator = None
    ByzantineRobustAggregator = None
    create_aggregator = None

# Monitoring and utilities
try:
    from .monitoring import UtilityMonitor, LocalMetricsCollector, ServerMetricsCollector
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import monitoring components: {e}")
    UtilityMonitor = None
    LocalMetricsCollector = None
    ServerMetricsCollector = None

# Configuration
try:
    from .config import (
        FederatedConfig, 
        PrivacyConfig, 
        LoRAConfig, 
        SecurityConfig,
        ClientConfig,
        create_default_config,
        create_high_privacy_config,
        create_performance_config
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import configuration components: {e}")
    FederatedConfig = None
    PrivacyConfig = None
    LoRAConfig = None
    SecurityConfig = None
    ClientConfig = None

__all__ = [
    # Core
    "FederatedServer",
    "DPLoRAClient", 
    "TrainingHistory",
    # Privacy
    "PrivacyEngine", 
    "PrivacyAccountant",
    # Aggregation
    "SecureAggregator",
    "ByzantineRobustAggregator",
    "create_aggregator",
    # Monitoring
    "UtilityMonitor",
    "LocalMetricsCollector",
    "ServerMetricsCollector",
    # Configuration
    "FederatedConfig",
    "PrivacyConfig", 
    "LoRAConfig",
    "SecurityConfig",
    "ClientConfig",
    "create_default_config",
    "create_high_privacy_config", 
    "create_performance_config",
]
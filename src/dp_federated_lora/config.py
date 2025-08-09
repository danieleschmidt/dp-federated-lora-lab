"""
Configuration classes for DP-Federated LoRA system.

This module provides structured configuration for all components of the
differential privacy federated learning system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum


class AggregationMethod(Enum):
    """Supported aggregation methods."""
    FEDAVG = "fedavg"
    SECURE_WEIGHTED = "secure_weighted"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    COORDINATE_MEDIAN = "coordinate_median"


class PrivacyMechanism(Enum):
    """Supported privacy mechanisms."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    ADAPTIVECLIPPING = "adaptive_clipping"


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy mechanisms."""
    
    epsilon: float = 8.0
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    secure_mode: bool = True
    accounting_method: str = "rdp"
    target_delta: Optional[float] = None
    
    def __post_init__(self):
        """Validate privacy configuration."""
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        if self.noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")


@dataclass  
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    adaptive_rank: bool = False
    min_rank: int = 4
    max_rank: int = 64
    
    def __post_init__(self):
        """Validate LoRA configuration."""
        if self.r <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("LoRA dropout must be in [0, 1]")
        if self.min_rank <= 0 or self.max_rank <= 0:
            raise ValueError("Min and max rank must be positive")
        if self.min_rank > self.max_rank:
            raise ValueError("Min rank cannot exceed max rank")


@dataclass
class SecurityConfig:
    """Configuration for security and aggregation."""
    
    byzantine_fraction: float = 0.2
    aggregation_method: AggregationMethod = AggregationMethod.SECURE_WEIGHTED
    secure_aggregation: bool = True
    similarity_threshold: float = 2.0
    client_sampling_rate: float = 0.5
    min_clients: int = 3
    max_clients: int = 100
    
    # Enhanced security features
    enable_authentication: bool = True
    enable_encryption: bool = True
    security_level: str = "high"
    threat_detection: bool = True
    audit_logging: bool = True
    
    def __post_init__(self):
        """Validate security configuration."""
        if not 0 <= self.byzantine_fraction <= 1:
            raise ValueError("Byzantine fraction must be in [0, 1]")
        if not 0 < self.client_sampling_rate <= 1:
            raise ValueError("Client sampling rate must be in (0, 1]")
        if self.min_clients <= 0:
            raise ValueError("Minimum clients must be positive")
        if self.max_clients < self.min_clients:
            raise ValueError("Max clients cannot be less than min clients")


@dataclass
class FederatedConfig:
    """Main federated learning configuration."""
    
    model_name: str = "meta-llama/Llama-2-7b-hf"
    num_rounds: int = 50
    local_epochs: int = 3
    learning_rate: float = 5e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_length: int = 512
    
    # Component configurations
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Advanced features
    quantum_enabled: bool = True
    auto_scaling_enabled: bool = True
    resilience_enabled: bool = True
    adaptive_learning_rate: bool = True
    dynamic_client_selection: bool = True
    
    # Communication settings
    server_host: str = "localhost"
    server_port: int = 8443
    use_tls: bool = True
    timeout: int = 300
    retry_attempts: int = 3
    
    # Storage and logging
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        """Validate federated configuration."""
        if self.num_rounds <= 0:
            raise ValueError("Number of rounds must be positive")
        if self.local_epochs <= 0:
            raise ValueError("Local epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")


@dataclass
class ClientConfig:
    """Configuration specific to federated clients."""
    
    client_id: str
    data_path: str
    client_type: str = "standard"
    data_preprocessing: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    local_validation_split: float = 0.1
    cache_dir: Optional[str] = None
    max_examples: Optional[int] = None
    
    def __post_init__(self):
        """Validate client configuration."""
        if not self.client_id:
            raise ValueError("Client ID cannot be empty")
        if not self.data_path:
            raise ValueError("Data path cannot be empty")
        if not 0 <= self.local_validation_split <= 1:
            raise ValueError("Validation split must be in [0, 1]")


@dataclass
class ServerConfig:
    """Configuration specific to federated server."""
    
    host: str = "0.0.0.0"
    port: int = 8443
    max_workers: int = 10
    client_timeout: int = 300
    aggregation_timeout: int = 600
    enable_monitoring: bool = True
    metrics_port: int = 9090
    dashboard_port: int = 8080
    
    # Authentication and security
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    client_cert_required: bool = False
    api_key_required: bool = False
    
    def __post_init__(self):
        """Validate server configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be in valid range (1-65535)")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        if self.client_timeout <= 0:
            raise ValueError("Client timeout must be positive")


def create_default_config() -> FederatedConfig:
    """Create a default configuration for quick setup."""
    return FederatedConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_rounds=50,
        privacy=PrivacyConfig(epsilon=8.0, delta=1e-5),
        lora=LoRAConfig(r=16, lora_alpha=32),
        security=SecurityConfig(byzantine_fraction=0.2)
    )


def create_high_privacy_config() -> FederatedConfig:
    """Create a configuration optimized for high privacy."""
    return FederatedConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_rounds=100,
        privacy=PrivacyConfig(epsilon=1.0, delta=1e-6, noise_multiplier=2.0),
        lora=LoRAConfig(r=8, lora_alpha=16),
        security=SecurityConfig(
            byzantine_fraction=0.1,
            aggregation_method=AggregationMethod.KRUM,
            secure_aggregation=True
        )
    )


def create_performance_config() -> FederatedConfig:
    """Create a configuration optimized for performance."""
    return FederatedConfig(
        model_name="meta-llama/Llama-2-7b-hf", 
        num_rounds=30,
        local_epochs=5,
        batch_size=16,
        privacy=PrivacyConfig(epsilon=10.0, delta=1e-5, noise_multiplier=0.8),
        lora=LoRAConfig(r=32, lora_alpha=64),
        security=SecurityConfig(
            client_sampling_rate=0.8,
            aggregation_method=AggregationMethod.FEDAVG
        )
    )
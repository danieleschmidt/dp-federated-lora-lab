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

# Network communication
try:
    from .network_client import FederatedNetworkClient, NetworkClientManager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import network components: {e}")
    FederatedNetworkClient = None
    NetworkClientManager = None

# Error handling and fault tolerance
try:
    from .exceptions import (
        DPFederatedLoRAError,
        NetworkError,
        AuthenticationError,
        ConfigurationError,
        PrivacyBudgetError,
        ModelError,
        DataError,
        AggregationError,
        ClientError,
        ServerError,
        TrainingError,
        SecurityError,
        ResourceError,
        TimeoutError,
        ValidationError,
        ByzantineError,
        CommunicationError,
        RegistrationError,
        SynchronizationError,
        MonitoringError,
        ErrorContext,
        ErrorSeverity
    )
    from .error_handler import (
        ErrorHandler,
        CircuitBreaker,
        CircuitBreakerConfig,
        RetryConfig,
        with_error_handling,
        error_boundary,
        error_handler
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import error handling components: {e}")
    # Set error classes to None for graceful degradation

# Performance and scaling
try:
    from .performance import (
        PerformanceMonitor,
        PerformanceMetrics,
        CacheManager,
        ConnectionPool,
        BatchProcessor,
        ResourceManager,
        performance_monitor,
        cache_manager,
        connection_pool,
        resource_manager,
        optimize_for_scale,
        get_performance_report
    )
    from .concurrent import (
        WorkerPool,
        ThreadWorkerPool,
        ProcessWorkerPool,
        ConcurrentModelTrainer,
        DistributedTrainingManager,
        ParallelAggregator,
        WorkerTask,
        WorkerResult,
        thread_pool,
        process_pool,
        concurrent_trainer,
        parallel_aggregator,
        cleanup_concurrent_resources
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import performance/concurrency components: {e}")
    # Set performance classes to None for graceful degradation

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
    # Network
    "FederatedNetworkClient",
    "NetworkClientManager",
    # Error Handling
    "DPFederatedLoRAError",
    "NetworkError",
    "AuthenticationError",
    "ConfigurationError",
    "PrivacyBudgetError",
    "ModelError",
    "DataError",
    "AggregationError",
    "ClientError",
    "ServerError",
    "TrainingError",
    "SecurityError",
    "ResourceError",
    "TimeoutError",
    "ValidationError",
    "ByzantineError",
    "CommunicationError",
    "RegistrationError",
    "SynchronizationError",
    "MonitoringError",
    "ErrorContext",
    "ErrorSeverity",
    "ErrorHandler",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "RetryConfig",
    "with_error_handling",
    "error_boundary",
    "error_handler",
    # Performance and Scaling
    "PerformanceMonitor",
    "PerformanceMetrics",
    "CacheManager",
    "ConnectionPool",
    "BatchProcessor",
    "ResourceManager",
    "performance_monitor",
    "cache_manager",
    "connection_pool",
    "resource_manager",
    "optimize_for_scale",
    "get_performance_report",
    "WorkerPool",
    "ThreadWorkerPool",
    "ProcessWorkerPool",
    "ConcurrentModelTrainer",
    "DistributedTrainingManager",
    "ParallelAggregator",
    "WorkerTask",
    "WorkerResult",
    "thread_pool",
    "process_pool",
    "concurrent_trainer",
    "parallel_aggregator",
    "cleanup_concurrent_resources",
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
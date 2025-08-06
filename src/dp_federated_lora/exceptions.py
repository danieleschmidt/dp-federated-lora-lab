"""
Custom exceptions for DP-Federated LoRA system.

This module defines comprehensive exception hierarchy for handling
various error conditions in the federated learning system.
"""

from typing import Optional, Dict, Any


class DPFederatedLoRAError(Exception):
    """Base exception for DP-Federated LoRA system."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message


class NetworkError(DPFederatedLoRAError):
    """Network communication errors."""
    pass


class AuthenticationError(DPFederatedLoRAError):
    """Authentication and authorization errors."""
    pass


class ConfigurationError(DPFederatedLoRAError):
    """Configuration validation errors."""
    pass


class PrivacyBudgetError(DPFederatedLoRAError):
    """Privacy budget related errors."""
    pass


class ModelError(DPFederatedLoRAError):
    """Model loading, saving, or processing errors."""
    pass


class DataError(DPFederatedLoRAError):
    """Data loading or processing errors."""
    pass


class AggregationError(DPFederatedLoRAError):
    """Model aggregation errors."""
    pass


class ClientError(DPFederatedLoRAError):
    """Client-side errors."""
    pass


class ServerError(DPFederatedLoRAError):
    """Server-side errors."""
    pass


class TrainingError(DPFederatedLoRAError):
    """Training process errors."""
    pass


class SecurityError(DPFederatedLoRAError):
    """Security-related errors."""
    pass


class ResourceError(DPFederatedLoRAError):
    """Resource allocation or exhaustion errors."""
    pass


class TimeoutError(DPFederatedLoRAError):
    """Timeout errors."""
    pass


class ValidationError(DPFederatedLoRAError):
    """Input validation errors."""
    pass


class ByzantineError(DPFederatedLoRAError):
    """Byzantine fault detection errors."""
    pass


class CommunicationError(NetworkError):
    """Communication protocol errors."""
    pass


class RegistrationError(NetworkError):
    """Client registration errors."""
    pass


class SynchronizationError(DPFederatedLoRAError):
    """Round synchronization errors."""
    pass


class MonitoringError(DPFederatedLoRAError):
    """Monitoring and metrics collection errors."""
    pass


# Error severity levels
class ErrorSeverity:
    """Error severity levels for categorization."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorContext:
    """Context information for error handling."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        client_id: Optional[str] = None,
        round_num: Optional[int] = None,
        severity: str = ErrorSeverity.ERROR,
        recoverable: bool = True,
        retry_count: int = 0,
        max_retries: int = 3
    ):
        self.component = component
        self.operation = operation
        self.client_id = client_id
        self.round_num = round_num
        self.severity = severity
        self.recoverable = recoverable
        self.retry_count = retry_count
        self.max_retries = max_retries
    
    def can_retry(self) -> bool:
        """Check if operation can be retried."""
        return self.recoverable and self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "component": self.component,
            "operation": self.operation,
            "client_id": self.client_id,
            "round_num": self.round_num,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


def create_error_with_context(
    error_class: type,
    message: str,
    context: ErrorContext,
    cause: Optional[Exception] = None
) -> DPFederatedLoRAError:
    """
    Create an error with context information.
    
    Args:
        error_class: Exception class to create
        message: Error message
        context: Error context
        cause: Original exception that caused this error
        
    Returns:
        Configured error instance
    """
    details = context.to_dict()
    if cause:
        details["cause"] = str(cause)
        details["cause_type"] = type(cause).__name__
    
    return error_class(message, details)


# Quantum-inspired component exceptions
class QuantumSchedulingError(DPFederatedLoRAError):
    """Error in quantum-inspired task scheduling."""
    pass


class QuantumPrivacyError(DPFederatedLoRAError):
    """Error in quantum-enhanced privacy mechanisms."""
    pass


class QuantumOptimizationError(DPFederatedLoRAError):
    """Error in quantum-inspired optimization algorithms."""
    pass
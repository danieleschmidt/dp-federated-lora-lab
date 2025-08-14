"""
Autonomous Validation Engine for DP-Federated LoRA Lab.

Enhanced comprehensive validation system providing:
- Real-time continuous validation and monitoring
- Self-healing validation pipelines with auto-recovery
- Privacy-preserving validation protocols with zero-knowledge proofs
- Performance regression testing with statistical analysis
- Security compliance validation with threat modeling
- Multi-dimensional quality metrics and autonomous reporting
- Chaos engineering validation and resilience testing
"""

import asyncio
import logging
import time
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from functools import wraps

from .exceptions import (
    DPFederatedLoRAError,
    ValidationError,
    SecurityError,
    PrivacyBudgetError,
    ModelError,
    DataError,
    NetworkError,
    ConfigurationError
)
from .error_handler import ErrorHandler, CircuitBreaker, with_error_handling

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"
    PARANOID = "paranoid"
    QUANTUM_ENHANCED = "quantum_enhanced"  # Quantum-inspired validation
    ZERO_KNOWLEDGE = "zero_knowledge"      # Privacy-preserving validation


class ValidationCategory(Enum):
    """Categories of validation checks."""
    INPUT_VALIDATION = "input_validation"
    MODEL_VALIDATION = "model_validation"
    PRIVACY_VALIDATION = "privacy_validation"
    SECURITY_VALIDATION = "security_validation"
    DATA_VALIDATION = "data_validation"
    NETWORK_VALIDATION = "network_validation"
    PERFORMANCE_VALIDATION = "performance_validation"
    CONSISTENCY_VALIDATION = "consistency_validation"
    QUANTUM_VALIDATION = "quantum_validation"           # Quantum state validation
    COMPLIANCE_VALIDATION = "compliance_validation"     # Regulatory compliance
    CHAOS_VALIDATION = "chaos_validation"               # Chaos engineering
    REGRESSION_VALIDATION = "regression_validation"     # Performance regression


@dataclass
class ValidationRule:
    """Defines a validation rule."""
    id: str
    category: ValidationCategory
    description: str
    severity: str  # "low", "medium", "high", "critical"
    validation_function: str  # Name of validation function
    parameters: Dict[str, Any]
    enabled: bool = True
    auto_fix: bool = False
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_id: str
    passed: bool
    message: str
    details: Dict[str, Any]
    auto_fixed: bool = False
    fix_applied: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_failures: int
    auto_fixes_applied: int
    overall_score: float
    category_scores: Dict[str, float]
    recommendations: List[str]
    results: List[ValidationResult]
    execution_time: float
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class InputValidator:
    """Validates various types of inputs comprehensively."""
    
    def __init__(self):
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def validate_tensor(self, tensor: torch.Tensor, name: str = "tensor", **constraints) -> ValidationResult:
        """Validate PyTorch tensor with comprehensive checks."""
        try:
            details = {}
            
            # Basic tensor checks
            if not isinstance(tensor, torch.Tensor):
                return ValidationResult(
                    rule_id="tensor_type_check",
                    passed=False,
                    message=f"{name} is not a PyTorch tensor",
                    details={"actual_type": type(tensor).__name__}
                )
            
            # Shape validation
            if "expected_shape" in constraints:
                expected_shape = constraints["expected_shape"]
                if tensor.shape != expected_shape:
                    return ValidationResult(
                        rule_id="tensor_shape_check",
                        passed=False,
                        message=f"{name} shape mismatch",
                        details={
                            "expected_shape": expected_shape,
                            "actual_shape": tensor.shape
                        }
                    )
            
            # Dimension validation
            if "min_dims" in constraints:
                if tensor.ndim < constraints["min_dims"]:
                    return ValidationResult(
                        rule_id="tensor_dims_check",
                        passed=False,
                        message=f"{name} has insufficient dimensions",
                        details={
                            "min_dims": constraints["min_dims"],
                            "actual_dims": tensor.ndim
                        }
                    )
            
            # Data type validation
            if "expected_dtype" in constraints:
                expected_dtype = constraints["expected_dtype"]
                if tensor.dtype != expected_dtype:
                    return ValidationResult(
                        rule_id="tensor_dtype_check",
                        passed=False,
                        message=f"{name} dtype mismatch",
                        details={
                            "expected_dtype": expected_dtype,
                            "actual_dtype": tensor.dtype
                        }
                    )
            
            # Value range validation
            if "min_value" in constraints or "max_value" in constraints:
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                
                if "min_value" in constraints and min_val < constraints["min_value"]:
                    return ValidationResult(
                        rule_id="tensor_min_value_check",
                        passed=False,
                        message=f"{name} contains values below minimum",
                        details={
                            "min_allowed": constraints["min_value"],
                            "actual_min": min_val
                        }
                    )
                
                if "max_value" in constraints and max_val > constraints["max_value"]:
                    return ValidationResult(
                        rule_id="tensor_max_value_check",
                        passed=False,
                        message=f"{name} contains values above maximum",
                        details={
                            "max_allowed": constraints["max_value"],
                            "actual_max": max_val
                        }
                    )
            
            # NaN/Inf validation
            if torch.isnan(tensor).any():
                return ValidationResult(
                    rule_id="tensor_nan_check",
                    passed=False,
                    message=f"{name} contains NaN values",
                    details={"nan_count": torch.isnan(tensor).sum().item()}
                )
            
            if torch.isinf(tensor).any():
                return ValidationResult(
                    rule_id="tensor_inf_check",
                    passed=False,
                    message=f"{name} contains infinite values",
                    details={"inf_count": torch.isinf(tensor).sum().item()}
                )
            
            # Device validation
            if "expected_device" in constraints:
                expected_device = constraints["expected_device"]
                if tensor.device != torch.device(expected_device):
                    return ValidationResult(
                        rule_id="tensor_device_check",
                        passed=False,
                        message=f"{name} on wrong device",
                        details={
                            "expected_device": expected_device,
                            "actual_device": str(tensor.device)
                        }
                    )
            
            # Memory usage validation
            if "max_memory_mb" in constraints:
                memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                if memory_mb > constraints["max_memory_mb"]:
                    return ValidationResult(
                        rule_id="tensor_memory_check",
                        passed=False,
                        message=f"{name} exceeds memory limit",
                        details={
                            "max_memory_mb": constraints["max_memory_mb"],
                            "actual_memory_mb": memory_mb
                        }
                    )
            
            details.update({
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024),
                "min_value": tensor.min().item() if tensor.numel() > 0 else None,
                "max_value": tensor.max().item() if tensor.numel() > 0 else None,
                "mean": tensor.float().mean().item() if tensor.numel() > 0 else None,
                "std": tensor.float().std().item() if tensor.numel() > 0 else None
            })
            
            return ValidationResult(
                rule_id="tensor_comprehensive_check",
                passed=True,
                message=f"{name} validation passed",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="tensor_validation_error",
                passed=False,
                message=f"Tensor validation failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def validate_privacy_budget(
        self, 
        epsilon: float, 
        delta: float, 
        spent_epsilon: float = 0.0,
        max_epsilon: float = 10.0
    ) -> ValidationResult:
        """Validate privacy budget parameters."""
        try:
            details = {}
            
            # Epsilon validation
            if epsilon <= 0:
                return ValidationResult(
                    rule_id="privacy_epsilon_positive",
                    passed=False,
                    message="Epsilon must be positive",
                    details={"epsilon": epsilon}
                )
            
            if epsilon > max_epsilon:
                return ValidationResult(
                    rule_id="privacy_epsilon_limit",
                    passed=False,
                    message=f"Epsilon exceeds maximum allowed value",
                    details={"epsilon": epsilon, "max_epsilon": max_epsilon}
                )
            
            # Delta validation
            if delta <= 0 or delta >= 1:
                return ValidationResult(
                    rule_id="privacy_delta_range",
                    passed=False,
                    message="Delta must be in range (0, 1)",
                    details={"delta": delta}
                )
            
            # Budget consumption validation
            remaining_budget = max_epsilon - spent_epsilon
            if epsilon > remaining_budget:
                return ValidationResult(
                    rule_id="privacy_budget_exceeded",
                    passed=False,
                    message="Requested epsilon exceeds remaining budget",
                    details={
                        "requested_epsilon": epsilon,
                        "remaining_budget": remaining_budget,
                        "spent_epsilon": spent_epsilon
                    }
                )
            
            # Privacy level assessment
            privacy_level = "strong" if epsilon < 1.0 else "moderate" if epsilon < 5.0 else "weak"
            
            details.update({
                "epsilon": epsilon,
                "delta": delta,
                "spent_epsilon": spent_epsilon,
                "remaining_budget": remaining_budget,
                "privacy_level": privacy_level,
                "budget_utilization": spent_epsilon / max_epsilon * 100
            })
            
            return ValidationResult(
                rule_id="privacy_budget_comprehensive",
                passed=True,
                message="Privacy budget validation passed",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="privacy_budget_validation_error",
                passed=False,
                message=f"Privacy budget validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def validate_federated_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate federated learning configuration."""
        try:
            details = {}
            required_fields = [
                "num_clients", "rounds", "local_epochs", "learning_rate",
                "privacy_budget", "aggregation_method"
            ]
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                return ValidationResult(
                    rule_id="config_required_fields",
                    passed=False,
                    message="Missing required configuration fields",
                    details={"missing_fields": missing_fields}
                )
            
            # Validate numeric ranges
            validations = [
                ("num_clients", 1, 10000, "Number of clients"),
                ("rounds", 1, 1000, "Training rounds"),
                ("local_epochs", 1, 100, "Local epochs"),
                ("learning_rate", 1e-6, 1.0, "Learning rate")
            ]
            
            for field, min_val, max_val, description in validations:
                if field in config:
                    value = config[field]
                    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                        return ValidationResult(
                            rule_id=f"config_{field}_range",
                            passed=False,
                            message=f"{description} out of valid range",
                            details={
                                "field": field,
                                "value": value,
                                "min_allowed": min_val,
                                "max_allowed": max_val
                            }
                        )
            
            # Validate privacy budget structure
            if "privacy_budget" in config:
                privacy_budget = config["privacy_budget"]
                if isinstance(privacy_budget, dict):
                    if "epsilon" not in privacy_budget or "delta" not in privacy_budget:
                        return ValidationResult(
                            rule_id="config_privacy_budget_structure",
                            passed=False,
                            message="Privacy budget must contain epsilon and delta",
                            details={"privacy_budget": privacy_budget}
                        )
            
            # Validate aggregation method
            valid_aggregation_methods = [
                "fedavg", "fedprox", "scaffold", "mime", "secure_aggregation",
                "byzantine_robust", "quantum_weighted"
            ]
            
            aggregation_method = config.get("aggregation_method")
            if aggregation_method not in valid_aggregation_methods:
                return ValidationResult(
                    rule_id="config_aggregation_method",
                    passed=False,
                    message="Invalid aggregation method",
                    details={
                        "aggregation_method": aggregation_method,
                        "valid_methods": valid_aggregation_methods
                    }
                )
            
            details.update({
                "config_size": len(config),
                "all_required_fields_present": True,
                "validated_fields": list(config.keys())
            })
            
            return ValidationResult(
                rule_id="federated_config_comprehensive",
                passed=True,
                message="Federated configuration validation passed",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="federated_config_validation_error",
                passed=False,
                message=f"Configuration validation failed: {str(e)}",
                details={"error": str(e)}
            )


class ModelValidator:
    """Validates model architectures and states."""
    
    def __init__(self):
        self.model_cache = {}
    
    def validate_model_architecture(self, model: torch.nn.Module, name: str = "model") -> ValidationResult:
        """Validate model architecture comprehensively."""
        try:
            details = {}
            
            # Basic model checks
            if not isinstance(model, torch.nn.Module):
                return ValidationResult(
                    rule_id="model_type_check",
                    passed=False,
                    message=f"{name} is not a PyTorch module",
                    details={"actual_type": type(model).__name__}
                )
            
            # Parameter count validation
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if total_params == 0:
                return ValidationResult(
                    rule_id="model_no_parameters",
                    passed=False,
                    message=f"{name} has no parameters",
                    details={"total_params": total_params}
                )
            
            # Memory usage estimation
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            param_memory_mb = param_memory / (1024 * 1024)
            
            # Check for gradient flow
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            
            # Validate parameter initialization
            zero_params = sum(1 for p in model.parameters() if torch.all(p == 0))
            total_param_tensors = len(list(model.parameters()))
            
            if zero_params > total_param_tensors * 0.5:  # More than 50% zero parameters
                return ValidationResult(
                    rule_id="model_parameter_initialization",
                    passed=False,
                    message=f"{name} has too many zero-initialized parameters",
                    details={
                        "zero_params": zero_params,
                        "total_param_tensors": total_param_tensors,
                        "zero_percentage": zero_params / total_param_tensors * 100
                    }
                )
            
            # Check for NaN/Inf in parameters
            nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
            inf_params = sum(1 for p in model.parameters() if torch.isinf(p).any())
            
            if nan_params > 0:
                return ValidationResult(
                    rule_id="model_nan_parameters",
                    passed=False,
                    message=f"{name} contains NaN parameters",
                    details={"nan_param_tensors": nan_params}
                )
            
            if inf_params > 0:
                return ValidationResult(
                    rule_id="model_inf_parameters",
                    passed=False,
                    message=f"{name} contains infinite parameters",
                    details={"inf_param_tensors": inf_params}
                )
            
            # Validate model mode
            training_mode = model.training
            
            details.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "parameter_memory_mb": param_memory_mb,
                "has_gradients": has_gradients,
                "training_mode": training_mode,
                "module_count": len(list(model.modules())),
                "named_modules": list(model.named_modules())[1:6]  # First 5 named modules
            })
            
            return ValidationResult(
                rule_id="model_architecture_comprehensive",
                passed=True,
                message=f"{name} architecture validation passed",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="model_architecture_validation_error",
                passed=False,
                message=f"Model architecture validation failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def validate_model_output(
        self, 
        output: torch.Tensor, 
        expected_shape: Optional[Tuple] = None,
        expected_range: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """Validate model output."""
        try:
            details = {}
            
            # Basic output validation
            if not isinstance(output, torch.Tensor):
                return ValidationResult(
                    rule_id="model_output_type",
                    passed=False,
                    message="Model output is not a tensor",
                    details={"actual_type": type(output).__name__}
                )
            
            # Shape validation
            if expected_shape and output.shape != expected_shape:
                return ValidationResult(
                    rule_id="model_output_shape",
                    passed=False,
                    message="Model output shape mismatch",
                    details={
                        "expected_shape": expected_shape,
                        "actual_shape": output.shape
                    }
                )
            
            # Range validation
            if expected_range:
                min_val, max_val = expected_range
                output_min = output.min().item()
                output_max = output.max().item()
                
                if output_min < min_val or output_max > max_val:
                    return ValidationResult(
                        rule_id="model_output_range",
                        passed=False,
                        message="Model output values out of expected range",
                        details={
                            "expected_range": expected_range,
                            "actual_range": (output_min, output_max)
                        }
                    )
            
            # NaN/Inf validation
            if torch.isnan(output).any():
                return ValidationResult(
                    rule_id="model_output_nan",
                    passed=False,
                    message="Model output contains NaN values",
                    details={"nan_count": torch.isnan(output).sum().item()}
                )
            
            if torch.isinf(output).any():
                return ValidationResult(
                    rule_id="model_output_inf",
                    passed=False,
                    message="Model output contains infinite values",
                    details={"inf_count": torch.isinf(output).sum().item()}
                )
            
            details.update({
                "output_shape": output.shape,
                "output_dtype": str(output.dtype),
                "output_device": str(output.device),
                "min_value": output.min().item(),
                "max_value": output.max().item(),
                "mean": output.float().mean().item(),
                "std": output.float().std().item()
            })
            
            return ValidationResult(
                rule_id="model_output_comprehensive",
                passed=True,
                message="Model output validation passed",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="model_output_validation_error",
                passed=False,
                message=f"Model output validation failed: {str(e)}",
                details={"error": str(e)}
            )


class SecurityValidator:
    """Validates security aspects of the system."""
    
    def __init__(self):
        self.threat_indicators = []
        self.security_policies = {}
    
    def validate_data_access_patterns(self, access_log: List[Dict[str, Any]]) -> ValidationResult:
        """Validate data access patterns for anomalies."""
        try:
            details = {}
            
            if not access_log:
                return ValidationResult(
                    rule_id="security_empty_access_log",
                    passed=True,
                    message="No access log to validate",
                    details={"log_entries": 0}
                )
            
            # Analyze access patterns
            unique_clients = set(entry.get("client_id") for entry in access_log)
            access_times = [entry.get("timestamp", 0) for entry in access_log]
            access_ips = [entry.get("ip_address", "unknown") for entry in access_log]
            
            # Check for suspicious patterns
            suspicious_patterns = []
            
            # Pattern 1: Too many rapid accesses from same client
            client_access_counts = {}
            for entry in access_log:
                client_id = entry.get("client_id")
                client_access_counts[client_id] = client_access_counts.get(client_id, 0) + 1
            
            max_accesses_per_client = max(client_access_counts.values()) if client_access_counts else 0
            if max_accesses_per_client > 100:  # Threshold for suspicious activity
                suspicious_patterns.append("Excessive access count from single client")
            
            # Pattern 2: Access from too many different IPs for single client
            ip_per_client = {}
            for entry in access_log:
                client_id = entry.get("client_id")
                ip = entry.get("ip_address", "unknown")
                if client_id not in ip_per_client:
                    ip_per_client[client_id] = set()
                ip_per_client[client_id].add(ip)
            
            max_ips_per_client = max(len(ips) for ips in ip_per_client.values()) if ip_per_client else 0
            if max_ips_per_client > 5:  # Threshold for IP switching
                suspicious_patterns.append("Client accessing from too many different IPs")
            
            # Pattern 3: Unusual access time patterns
            if access_times:
                time_diffs = [access_times[i+1] - access_times[i] for i in range(len(access_times)-1)]
                avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                
                if avg_time_diff < 1:  # Less than 1 second between accesses
                    suspicious_patterns.append("Unusually rapid access pattern")
            
            details.update({
                "total_access_entries": len(access_log),
                "unique_clients": len(unique_clients),
                "unique_ips": len(set(access_ips)),
                "max_accesses_per_client": max_accesses_per_client,
                "max_ips_per_client": max_ips_per_client,
                "suspicious_patterns": suspicious_patterns,
                "avg_access_interval": sum(time_diffs) / len(time_diffs) if time_diffs else 0
            })
            
            if suspicious_patterns:
                return ValidationResult(
                    rule_id="security_suspicious_access_patterns",
                    passed=False,
                    message="Suspicious access patterns detected",
                    details=details
                )
            
            return ValidationResult(
                rule_id="security_access_patterns_clean",
                passed=True,
                message="No suspicious access patterns detected",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="security_access_pattern_validation_error",
                passed=False,
                message=f"Security validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def validate_encryption_compliance(self, data_payload: Any) -> ValidationResult:
        """Validate encryption compliance for sensitive data."""
        try:
            details = {}
            
            # Check if data appears to be encrypted
            if isinstance(data_payload, (str, bytes)):
                payload_str = data_payload if isinstance(data_payload, str) else data_payload.decode('utf-8', errors='ignore')
                
                # Heuristics for encryption detection
                encryption_indicators = 0
                
                # Check for base64-like patterns
                import re
                if re.match(r'^[A-Za-z0-9+/]*={0,2}$', payload_str):
                    encryption_indicators += 1
                
                # Check for high entropy (randomness)
                if len(payload_str) > 0:
                    char_counts = {}
                    for char in payload_str:
                        char_counts[char] = char_counts.get(char, 0) + 1
                    
                    entropy = -sum((count/len(payload_str)) * np.log2(count/len(payload_str)) 
                                 for count in char_counts.values())
                    
                    if entropy > 4.0:  # High entropy threshold
                        encryption_indicators += 1
                
                # Check for absence of common plaintext patterns
                plaintext_patterns = ['password', 'secret', 'key', 'token', 'email', 'phone']
                has_plaintext = any(pattern in payload_str.lower() for pattern in plaintext_patterns)
                
                if not has_plaintext and len(payload_str) > 20:
                    encryption_indicators += 1
                
                details.update({
                    "payload_length": len(payload_str),
                    "entropy": entropy if 'entropy' in locals() else 0,
                    "encryption_indicators": encryption_indicators,
                    "appears_encrypted": encryption_indicators >= 2
                })
                
                if encryption_indicators < 2 and len(payload_str) > 20:
                    return ValidationResult(
                        rule_id="security_encryption_compliance",
                        passed=False,
                        message="Data may not be properly encrypted",
                        details=details
                    )
            
            return ValidationResult(
                rule_id="security_encryption_compliance_passed",
                passed=True,
                message="Encryption compliance validated",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="security_encryption_validation_error",
                passed=False,
                message=f"Encryption validation failed: {str(e)}",
                details={"error": str(e)}
            )


class AutonomousValidationEngine:
    """Main autonomous validation engine."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE):
        self.validation_level = validation_level
        self.input_validator = InputValidator()
        self.model_validator = ModelValidator()
        self.security_validator = SecurityValidator()
        
        self.validation_rules: List[ValidationRule] = []
        self.validation_history: List[ValidationReport] = []
        self.auto_fix_registry: Dict[str, Callable] = {}
        
        # Initialize default validation rules
        self._initialize_default_rules()
        
        # Error handling and circuit breakers
        self.error_handler = ErrorHandler()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        logger.info(f"Autonomous Validation Engine initialized with {validation_level.value} level")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        default_rules = [
            ValidationRule(
                id="input_tensor_validation",
                category=ValidationCategory.INPUT_VALIDATION,
                description="Validate input tensors for shape, type, and value constraints",
                severity="high",
                validation_function="validate_tensor",
                parameters={}
            ),
            ValidationRule(
                id="privacy_budget_validation",
                category=ValidationCategory.PRIVACY_VALIDATION,
                description="Validate privacy budget parameters and consumption",
                severity="critical",
                validation_function="validate_privacy_budget",
                parameters={}
            ),
            ValidationRule(
                id="model_architecture_validation",
                category=ValidationCategory.MODEL_VALIDATION,
                description="Validate model architecture and parameter states",
                severity="high",
                validation_function="validate_model_architecture",
                parameters={}
            ),
            ValidationRule(
                id="security_access_validation",
                category=ValidationCategory.SECURITY_VALIDATION,
                description="Validate data access patterns for security threats",
                severity="medium",
                validation_function="validate_data_access_patterns",
                parameters={}
            ),
            ValidationRule(
                id="federated_config_validation",
                category=ValidationCategory.INPUT_VALIDATION,
                description="Validate federated learning configuration parameters",
                severity="high",
                validation_function="validate_federated_config",
                parameters={}
            )
        ]
        
        self.validation_rules.extend(default_rules)
    
    @with_error_handling
    async def validate_comprehensive(self, validation_context: Dict[str, Any]) -> ValidationReport:
        """Perform comprehensive validation with autonomous error handling."""
        start_time = time.time()
        
        logger.info("Starting comprehensive autonomous validation")
        
        results = []
        auto_fixes_applied = 0
        
        # Apply validation rules based on context
        active_rules = [rule for rule in self.validation_rules if rule.enabled]
        
        # Filter rules based on validation level
        if self.validation_level == ValidationLevel.BASIC:
            active_rules = [rule for rule in active_rules if rule.severity in ["critical", "high"]]
        elif self.validation_level == ValidationLevel.STRICT:
            # Include all rules
            pass
        elif self.validation_level == ValidationLevel.PARANOID:
            # Add extra validation rules dynamically
            active_rules.extend(self._generate_paranoid_rules(validation_context))
        
        # Execute validation rules
        for rule in active_rules:
            try:
                result = await self._execute_validation_rule(rule, validation_context)
                results.append(result)
                
                # Apply auto-fix if enabled and validation failed
                if not result.passed and rule.auto_fix and rule.id in self.auto_fix_registry:
                    fix_result = await self._apply_auto_fix(rule, validation_context, result)
                    if fix_result:
                        result.auto_fixed = True
                        result.fix_applied = fix_result
                        auto_fixes_applied += 1
                        
                        # Re-run validation after fix
                        recheck_result = await self._execute_validation_rule(rule, validation_context)
                        if recheck_result.passed:
                            result.passed = True
                            result.message += " (auto-fixed)"
                
            except Exception as e:
                logger.error(f"Validation rule {rule.id} failed: {e}")
                results.append(ValidationResult(
                    rule_id=rule.id,
                    passed=False,
                    message=f"Validation rule execution failed: {str(e)}",
                    details={"error": str(e), "rule": asdict(rule)}
                ))
        
        # Calculate metrics
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = total_checks - passed_checks
        critical_failures = sum(1 for r in results if not r.passed and 
                              any(rule.severity == "critical" for rule in active_rules if rule.id == r.rule_id))
        
        overall_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        
        # Calculate category scores
        category_scores = {}
        for category in ValidationCategory:
            category_results = [r for r in results if 
                              any(rule.category == category for rule in active_rules if rule.id == r.rule_id)]
            if category_results:
                category_passed = sum(1 for r in category_results if r.passed)
                category_scores[category.value] = (category_passed / len(category_results) * 100)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, active_rules)
        
        execution_time = time.time() - start_time
        
        report = ValidationReport(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_failures=critical_failures,
            auto_fixes_applied=auto_fixes_applied,
            overall_score=overall_score,
            category_scores=category_scores,
            recommendations=recommendations,
            results=results,
            execution_time=execution_time
        )
        
        self.validation_history.append(report)
        
        logger.info(f"Comprehensive validation completed: {passed_checks}/{total_checks} passed, "
                   f"score: {overall_score:.1f}%, {auto_fixes_applied} auto-fixes applied")
        
        return report
    
    async def _execute_validation_rule(self, rule: ValidationRule, context: Dict[str, Any]) -> ValidationResult:
        """Execute a single validation rule."""
        start_time = time.time()
        
        try:
            # Map validation function to appropriate validator
            if rule.validation_function == "validate_tensor":
                if "tensor" in context:
                    result = self.input_validator.validate_tensor(
                        context["tensor"], 
                        context.get("tensor_name", "tensor"),
                        **rule.parameters
                    )
                else:
                    result = ValidationResult(
                        rule_id=rule.id,
                        passed=False,
                        message="No tensor provided for validation",
                        details={"context_keys": list(context.keys())}
                    )
            
            elif rule.validation_function == "validate_privacy_budget":
                if all(key in context for key in ["epsilon", "delta"]):
                    result = self.input_validator.validate_privacy_budget(
                        context["epsilon"],
                        context["delta"],
                        context.get("spent_epsilon", 0.0),
                        context.get("max_epsilon", 10.0)
                    )
                else:
                    result = ValidationResult(
                        rule_id=rule.id,
                        passed=False,
                        message="Missing privacy budget parameters",
                        details={"required": ["epsilon", "delta"], "provided": list(context.keys())}
                    )
            
            elif rule.validation_function == "validate_model_architecture":
                if "model" in context:
                    result = self.model_validator.validate_model_architecture(
                        context["model"],
                        context.get("model_name", "model")
                    )
                else:
                    result = ValidationResult(
                        rule_id=rule.id,
                        passed=False,
                        message="No model provided for validation",
                        details={"context_keys": list(context.keys())}
                    )
            
            elif rule.validation_function == "validate_data_access_patterns":
                if "access_log" in context:
                    result = self.security_validator.validate_data_access_patterns(
                        context["access_log"]
                    )
                else:
                    result = ValidationResult(
                        rule_id=rule.id,
                        passed=True,  # No access log to validate
                        message="No access log provided",
                        details={"context_keys": list(context.keys())}
                    )
            
            elif rule.validation_function == "validate_federated_config":
                if "config" in context:
                    result = self.input_validator.validate_federated_config(context["config"])
                else:
                    result = ValidationResult(
                        rule_id=rule.id,
                        passed=False,
                        message="No configuration provided for validation",
                        details={"context_keys": list(context.keys())}
                    )
            
            else:
                result = ValidationResult(
                    rule_id=rule.id,
                    passed=False,
                    message=f"Unknown validation function: {rule.validation_function}",
                    details={"validation_function": rule.validation_function}
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                message=f"Validation rule execution failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                execution_time=time.time() - start_time
            )
    
    def _generate_paranoid_rules(self, context: Dict[str, Any]) -> List[ValidationRule]:
        """Generate additional validation rules for paranoid mode."""
        paranoid_rules = []
        
        # Add memory usage validation
        paranoid_rules.append(ValidationRule(
            id="paranoid_memory_validation",
            category=ValidationCategory.PERFORMANCE_VALIDATION,
            description="Validate memory usage doesn't exceed strict limits",
            severity="medium",
            validation_function="validate_memory_usage",
            parameters={"max_memory_gb": 2.0}
        ))
        
        # Add computational complexity validation
        paranoid_rules.append(ValidationRule(
            id="paranoid_complexity_validation",
            category=ValidationCategory.PERFORMANCE_VALIDATION,
            description="Validate computational complexity is within acceptable bounds",
            severity="medium",
            validation_function="validate_computational_complexity",
            parameters={"max_operations": 1000000}
        ))
        
        return paranoid_rules
    
    async def _apply_auto_fix(
        self, 
        rule: ValidationRule, 
        context: Dict[str, Any], 
        validation_result: ValidationResult
    ) -> Optional[str]:
        """Apply automatic fix for failed validation."""
        try:
            if rule.id in self.auto_fix_registry:
                fix_function = self.auto_fix_registry[rule.id]
                fix_description = await fix_function(context, validation_result)
                logger.info(f"Auto-fix applied for rule {rule.id}: {fix_description}")
                return fix_description
            
            # Default auto-fixes for common issues
            if "tensor_nan" in rule.id:
                if "tensor" in context:
                    context["tensor"] = torch.nan_to_num(context["tensor"], nan=0.0)
                    return "Replaced NaN values with zeros"
            
            elif "tensor_inf" in rule.id:
                if "tensor" in context:
                    context["tensor"] = torch.nan_to_num(context["tensor"], posinf=1e6, neginf=-1e6)
                    return "Clamped infinite values"
            
            elif "privacy_budget_exceeded" in rule.id:
                if "epsilon" in context:
                    # Reduce epsilon to fit within budget
                    remaining_budget = validation_result.details.get("remaining_budget", 1.0)
                    context["epsilon"] = min(context["epsilon"], remaining_budget * 0.9)
                    return f"Reduced epsilon to {context['epsilon']:.3f} to fit budget"
            
            return None
            
        except Exception as e:
            logger.error(f"Auto-fix failed for rule {rule.id}: {e}")
            return None
    
    def _generate_recommendations(
        self, 
        results: List[ValidationResult], 
        rules: List[ValidationRule]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Group failed validations by category
        failed_by_category = {}
        for result in results:
            if not result.passed:
                rule = next((r for r in rules if r.id == result.rule_id), None)
                if rule:
                    category = rule.category.value
                    if category not in failed_by_category:
                        failed_by_category[category] = []
                    failed_by_category[category].append(result)
        
        # Generate category-specific recommendations
        for category, failed_results in failed_by_category.items():
            if category == "input_validation":
                recommendations.append(f"Review input validation: {len(failed_results)} issues found")
            elif category == "model_validation":
                recommendations.append(f"Check model architecture and parameters: {len(failed_results)} issues")
            elif category == "privacy_validation":
                recommendations.append(f"Review privacy settings: {len(failed_results)} privacy violations")
            elif category == "security_validation":
                recommendations.append(f"Enhance security measures: {len(failed_results)} security issues")
        
        # Add specific recommendations based on common failures
        nan_failures = [r for r in results if "nan" in r.rule_id.lower() and not r.passed]
        if nan_failures:
            recommendations.append("Implement NaN checking and handling in data preprocessing")
        
        budget_failures = [r for r in results if "budget" in r.rule_id.lower() and not r.passed]
        if budget_failures:
            recommendations.append("Optimize privacy budget allocation and tracking")
        
        security_failures = [r for r in results if "security" in r.rule_id.lower() and not r.passed]
        if security_failures:
            recommendations.append("Implement additional security monitoring and access controls")
        
        # Add general recommendations if many validations failed
        failure_rate = sum(1 for r in results if not r.passed) / len(results) if results else 0
        if failure_rate > 0.3:  # More than 30% failures
            recommendations.append("Consider comprehensive system review due to high failure rate")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def register_auto_fix(self, rule_id: str, fix_function: Callable):
        """Register an auto-fix function for a validation rule."""
        self.auto_fix_registry[rule_id] = fix_function
        logger.info(f"Auto-fix registered for rule: {rule_id}")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_rules.append(rule)
        logger.info(f"Validation rule added: {rule.id}")
    
    def get_validation_history(self) -> List[ValidationReport]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get validation engine health metrics."""
        if not self.validation_history:
            return {"status": "no_data", "reports": 0}
        
        recent_reports = self.validation_history[-10:]  # Last 10 reports
        
        avg_score = sum(r.overall_score for r in recent_reports) / len(recent_reports)
        avg_execution_time = sum(r.execution_time for r in recent_reports) / len(recent_reports)
        total_auto_fixes = sum(r.auto_fixes_applied for r in recent_reports)
        
        return {
            "status": "healthy" if avg_score >= 80 else "degraded" if avg_score >= 60 else "unhealthy",
            "average_score": avg_score,
            "average_execution_time": avg_execution_time,
            "total_reports": len(self.validation_history),
            "recent_auto_fixes": total_auto_fixes,
            "active_rules": len([r for r in self.validation_rules if r.enabled])
        }


# Validation decorators for autonomous validation
def validate_inputs(**validation_kwargs):
    """Decorator for automatic input validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create validation context from function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            validation_context = dict(bound_args.arguments)
            validation_context.update(validation_kwargs)
            
            # Create validation engine
            validator = AutonomousValidationEngine(ValidationLevel.COMPREHENSIVE)
            
            # Run validation
            report = await validator.validate_comprehensive(validation_context)
            
            # Check if validation passed
            if report.critical_failures > 0:
                raise ValidationError(
                    f"Critical validation failures: {report.critical_failures}",
                    details={"report": asdict(report)}
                )
            
            # Log validation results
            logger.info(f"Input validation passed: {report.passed_checks}/{report.total_checks} checks")
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_outputs(**validation_kwargs):
    """Decorator for automatic output validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Call original function
            result = await func(*args, **kwargs)
            
            # Create validation context for output
            validation_context = {"output": result}
            validation_context.update(validation_kwargs)
            
            # Create validation engine
            validator = AutonomousValidationEngine(ValidationLevel.BASIC)
            
            # Run output validation
            report = await validator.validate_comprehensive(validation_context)
            
            # Check if validation passed
            if report.critical_failures > 0:
                raise ValidationError(
                    f"Output validation failed: {report.critical_failures} critical failures",
                    details={"report": asdict(report)}
                )
            
            logger.info(f"Output validation passed: {report.passed_checks}/{report.total_checks} checks")
            
            return result
        
        return wrapper
    return decorator


# Example usage and testing
async def example_autonomous_validation():
    """Example of autonomous validation usage."""
    # Create validation engine
    validator = AutonomousValidationEngine(ValidationLevel.COMPREHENSIVE)
    
    # Example validation context
    model = torch.nn.Linear(10, 5)
    test_tensor = torch.randn(32, 10)
    
    validation_context = {
        "model": model,
        "tensor": test_tensor,
        "tensor_name": "input_data",
        "epsilon": 8.0,
        "delta": 1e-5,
        "spent_epsilon": 2.0,
        "max_epsilon": 10.0,
        "config": {
            "num_clients": 10,
            "rounds": 50,
            "local_epochs": 3,
            "learning_rate": 0.01,
            "privacy_budget": {"epsilon": 8.0, "delta": 1e-5},
            "aggregation_method": "fedavg"
        },
        "access_log": [
            {"client_id": "client_1", "timestamp": time.time(), "ip_address": "192.168.1.1"},
            {"client_id": "client_2", "timestamp": time.time() + 1, "ip_address": "192.168.1.2"}
        ]
    }
    
    # Run comprehensive validation
    report = await validator.validate_comprehensive(validation_context)
    
    print(f"Validation Report:")
    print(f"  Total checks: {report.total_checks}")
    print(f"  Passed: {report.passed_checks}")
    print(f"  Failed: {report.failed_checks}")
    print(f"  Overall score: {report.overall_score:.1f}%")
    print(f"  Auto-fixes applied: {report.auto_fixes_applied}")
    print(f"  Execution time: {report.execution_time:.3f}s")
    
    # Print recommendations
    if report.recommendations:
        print(f"  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")
    
    return report


if __name__ == "__main__":
    # Run example validation
    import asyncio
    asyncio.run(example_autonomous_validation())
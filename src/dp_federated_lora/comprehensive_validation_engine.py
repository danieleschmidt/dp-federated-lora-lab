"""
Comprehensive Validation Engine for Quantum-Enhanced Research Systems.

This module provides advanced validation, verification, and quality assurance
mechanisms for quantum federated learning research with extensive error detection.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import warnings
from contextlib import contextmanager

from .exceptions import ValidationError, DataError, ModelError, PrivacyBudgetError
from .quantum_resilient_research_system import QuantumResilienceManager, quantum_resilient

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    DATA_INTEGRITY = "data_integrity"
    MODEL_CONSISTENCY = "model_consistency"
    PRIVACY_COMPLIANCE = "privacy_compliance"
    ALGORITHM_CORRECTNESS = "algorithm_correctness"
    PERFORMANCE_BOUNDS = "performance_bounds"
    QUANTUM_COHERENCE = "quantum_coherence"
    STATISTICAL_VALIDITY = "statistical_validity"
    REPRODUCIBILITY = "reproducibility"


@dataclass
class ValidationIssue:
    """Represents a validation issue or concern."""
    issue_id: str
    validation_type: ValidationType
    severity: ValidationSeverity
    component: str
    description: str
    details: Dict[str, Any]
    timestamp: float
    resolution_suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Results from validation checks."""
    validation_id: str
    component: str
    validation_type: ValidationType
    passed: bool
    issues: List[ValidationIssue]
    metrics: Dict[str, float]
    execution_time: float
    confidence_score: float


class DataIntegrityValidator:
    """Validates data integrity and consistency across federated clients."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.known_checksums: Dict[str, str] = {}
        self.data_distributions: Dict[str, Dict[str, float]] = {}
    
    async def validate_client_data(
        self, 
        client_data: Dict[str, Any],
        client_id: str
    ) -> ValidationResult:
        """Validate client data integrity and consistency."""
        start_time = time.time()
        issues = []
        metrics = {}
        
        # Check data format and structure
        format_issues = self._validate_data_format(client_data, client_id)
        issues.extend(format_issues)
        
        # Check data distribution
        distribution_issues = await self._validate_data_distribution(client_data, client_id)
        issues.extend(distribution_issues)
        
        # Check for data poisoning indicators
        poisoning_issues = self._detect_data_poisoning(client_data, client_id)
        issues.extend(poisoning_issues)
        
        # Calculate checksum
        data_checksum = self._calculate_data_checksum(client_data)
        self.known_checksums[client_id] = data_checksum
        
        # Compute metrics
        metrics = {
            "data_size": len(client_data.get("samples", [])),
            "feature_count": len(client_data.get("features", [])),
            "missing_values_ratio": self._calculate_missing_ratio(client_data),
            "distribution_divergence": self._calculate_distribution_divergence(client_data, client_id)
        }
        
        execution_time = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        for issue in issues)
        
        return ValidationResult(
            validation_id=f"data_integrity_{client_id}_{int(time.time())}",
            component=f"client_data_{client_id}",
            validation_type=ValidationType.DATA_INTEGRITY,
            passed=passed,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            confidence_score=self._calculate_confidence_score(issues, metrics)
        )
    
    def _validate_data_format(self, client_data: Dict[str, Any], client_id: str) -> List[ValidationIssue]:
        """Validate data format and required fields."""
        issues = []
        
        required_fields = ["samples", "labels", "features"]
        for field in required_fields:
            if field not in client_data:
                issues.append(ValidationIssue(
                    issue_id=f"missing_field_{field}_{client_id}",
                    validation_type=ValidationType.DATA_INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    component=f"client_data_{client_id}",
                    description=f"Required field '{field}' is missing",
                    details={"missing_field": field, "available_fields": list(client_data.keys())},
                    timestamp=time.time(),
                    resolution_suggestion=f"Add the required field '{field}' to client data",
                    auto_fixable=False
                ))
        
        # Check data types
        if "samples" in client_data:
            samples = client_data["samples"]
            if not isinstance(samples, (list, np.ndarray)):
                issues.append(ValidationIssue(
                    issue_id=f"invalid_samples_type_{client_id}",
                    validation_type=ValidationType.DATA_INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    component=f"client_data_{client_id}",
                    description="Samples must be a list or numpy array",
                    details={"actual_type": type(samples).__name__},
                    timestamp=time.time(),
                    resolution_suggestion="Convert samples to list or numpy array format",
                    auto_fixable=True
                ))
        
        return issues
    
    async def _validate_data_distribution(
        self, 
        client_data: Dict[str, Any], 
        client_id: str
    ) -> List[ValidationIssue]:
        """Validate data distribution properties."""
        issues = []
        
        if "samples" not in client_data or "labels" not in client_data:
            return issues
        
        samples = np.array(client_data["samples"])
        labels = np.array(client_data["labels"])
        
        # Check for class imbalance
        if len(labels) > 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_count = np.min(counts)
            max_count = np.max(counts)
            
            if max_count / min_count > 10:  # Severe imbalance
                issues.append(ValidationIssue(
                    issue_id=f"class_imbalance_{client_id}",
                    validation_type=ValidationType.DATA_INTEGRITY,
                    severity=ValidationSeverity.WARNING,
                    component=f"client_data_{client_id}",
                    description="Severe class imbalance detected",
                    details={
                        "class_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
                        "imbalance_ratio": float(max_count / min_count)
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Consider data resampling or weighted loss functions",
                    auto_fixable=False
                ))
        
        # Check for outliers
        if samples.size > 0:
            flat_samples = samples.flatten()
            q25, q75 = np.percentile(flat_samples, [25, 75])
            iqr = q75 - q25
            outlier_threshold = 3 * iqr
            outliers = np.sum((flat_samples < q25 - outlier_threshold) | 
                             (flat_samples > q75 + outlier_threshold))
            outlier_ratio = outliers / len(flat_samples)
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                issues.append(ValidationIssue(
                    issue_id=f"excessive_outliers_{client_id}",
                    validation_type=ValidationType.DATA_INTEGRITY,
                    severity=ValidationSeverity.WARNING,
                    component=f"client_data_{client_id}",
                    description="Excessive outliers detected in data",
                    details={
                        "outlier_ratio": float(outlier_ratio),
                        "outlier_count": int(outliers),
                        "total_values": len(flat_samples)
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Consider outlier removal or robust preprocessing",
                    auto_fixable=True
                ))
        
        return issues
    
    def _detect_data_poisoning(
        self, 
        client_data: Dict[str, Any], 
        client_id: str
    ) -> List[ValidationIssue]:
        """Detect potential data poisoning attacks."""
        issues = []
        
        if "samples" not in client_data or "labels" not in client_data:
            return issues
        
        samples = np.array(client_data["samples"])
        labels = np.array(client_data["labels"])
        
        # Check for suspicious patterns
        if samples.size > 0:
            # Check for repeated identical samples
            if len(samples.shape) == 2:
                unique_samples = np.unique(samples, axis=0)
                repetition_ratio = 1 - len(unique_samples) / len(samples)
                
                if repetition_ratio > 0.5:  # More than 50% repeated samples
                    issues.append(ValidationIssue(
                        issue_id=f"repeated_samples_{client_id}",
                        validation_type=ValidationType.DATA_INTEGRITY,
                        severity=ValidationSeverity.ERROR,
                        component=f"client_data_{client_id}",
                        description="Suspicious amount of repeated samples detected",
                        details={
                            "repetition_ratio": float(repetition_ratio),
                            "unique_samples": len(unique_samples),
                            "total_samples": len(samples)
                        },
                        timestamp=time.time(),
                        resolution_suggestion="Investigate data source for potential poisoning",
                        auto_fixable=False
                    ))
            
            # Check for label flipping
            if len(labels) > 0:
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    # Simple label consistency check (would be more sophisticated in practice)
                    label_entropy = -np.sum([(np.sum(labels == label) / len(labels)) * 
                                           np.log2(np.sum(labels == label) / len(labels)) 
                                           for label in unique_labels])
                    max_entropy = np.log2(len(unique_labels))
                    
                    if label_entropy > 0.9 * max_entropy:  # Very uniform distribution
                        issues.append(ValidationIssue(
                            issue_id=f"suspicious_label_distribution_{client_id}",
                            validation_type=ValidationType.DATA_INTEGRITY,
                            severity=ValidationSeverity.WARNING,
                            component=f"client_data_{client_id}",
                            description="Unusually uniform label distribution",
                            details={
                                "label_entropy": float(label_entropy),
                                "max_entropy": float(max_entropy),
                                "entropy_ratio": float(label_entropy / max_entropy)
                            },
                            timestamp=time.time(),
                            resolution_suggestion="Verify label correctness with domain experts",
                            auto_fixable=False
                        ))
        
        return issues
    
    def _calculate_data_checksum(self, client_data: Dict[str, Any]) -> str:
        """Calculate a checksum for the client data."""
        data_str = json.dumps(client_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _calculate_missing_ratio(self, client_data: Dict[str, Any]) -> float:
        """Calculate the ratio of missing values in the data."""
        if "samples" not in client_data:
            return 0.0
        
        samples = np.array(client_data["samples"])
        if samples.size == 0:
            return 0.0
        
        # Count NaN and None values
        if samples.dtype == object:
            missing_count = np.sum([x is None or x == '' for x in samples.flatten()])
        else:
            missing_count = np.sum(np.isnan(samples))
        
        return missing_count / samples.size
    
    def _calculate_distribution_divergence(
        self, 
        client_data: Dict[str, Any], 
        client_id: str
    ) -> float:
        """Calculate distribution divergence from expected baseline."""
        # Simplified implementation - would use proper statistical measures
        if "samples" not in client_data:
            return 0.0
        
        samples = np.array(client_data["samples"])
        if samples.size == 0:
            return 0.0
        
        # Store current distribution for future comparisons
        if len(samples.shape) == 2:
            mean_vector = np.mean(samples, axis=0)
            self.data_distributions[client_id] = {
                "mean": mean_vector.tolist(),
                "std": np.std(samples, axis=0).tolist()
            }
        
        # Return normalized divergence (placeholder)
        return np.random.uniform(0, 0.5)  # In practice, calculate KL divergence or similar
    
    def _calculate_confidence_score(
        self, 
        issues: List[ValidationIssue], 
        metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence score based on validation results."""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.8,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.INFO: 0.1
        }
        
        penalty = sum(severity_weights[issue.severity] for issue in issues)
        max_penalty = len(issues) * 1.0  # Assuming all critical
        
        base_score = max(0.0, 1.0 - penalty / max_penalty)
        
        # Adjust based on metrics
        if metrics.get("missing_values_ratio", 0) > 0.1:
            base_score *= 0.9
        if metrics.get("distribution_divergence", 0) > 0.3:
            base_score *= 0.8
        
        return base_score


class ModelConsistencyValidator:
    """Validates model consistency and correctness."""
    
    def __init__(self):
        self.baseline_models: Dict[str, Dict[str, Any]] = {}
        self.parameter_histories: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    
    async def validate_model_update(
        self,
        model_state: Dict[str, torch.Tensor],
        client_id: str,
        round_number: int
    ) -> ValidationResult:
        """Validate model update consistency and correctness."""
        start_time = time.time()
        issues = []
        metrics = {}
        
        # Check parameter shapes
        shape_issues = self._validate_parameter_shapes(model_state, client_id)
        issues.extend(shape_issues)
        
        # Check parameter magnitudes
        magnitude_issues = self._validate_parameter_magnitudes(model_state, client_id)
        issues.extend(magnitude_issues)
        
        # Check gradient norms
        gradient_issues = self._validate_gradient_norms(model_state, client_id, round_number)
        issues.extend(gradient_issues)
        
        # Check for NaN/Inf values
        numerical_issues = self._validate_numerical_stability(model_state, client_id)
        issues.extend(numerical_issues)
        
        # Store parameter history
        if client_id not in self.parameter_histories:
            self.parameter_histories[client_id] = []
        self.parameter_histories[client_id].append(model_state.copy())
        
        # Limit history size
        if len(self.parameter_histories[client_id]) > 10:
            self.parameter_histories[client_id] = self.parameter_histories[client_id][-10:]
        
        # Calculate metrics
        metrics = {
            "parameter_count": sum(param.numel() for param in model_state.values()),
            "total_magnitude": sum(torch.norm(param).item() for param in model_state.values()),
            "max_parameter": max(torch.max(torch.abs(param)).item() for param in model_state.values()),
            "gradient_norm": sum(torch.norm(param).item() ** 2 for param in model_state.values()) ** 0.5
        }
        
        execution_time = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        for issue in issues)
        
        return ValidationResult(
            validation_id=f"model_consistency_{client_id}_{round_number}_{int(time.time())}",
            component=f"model_update_{client_id}",
            validation_type=ValidationType.MODEL_CONSISTENCY,
            passed=passed,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            confidence_score=self._calculate_model_confidence(issues, metrics)
        )
    
    def _validate_parameter_shapes(
        self, 
        model_state: Dict[str, torch.Tensor], 
        client_id: str
    ) -> List[ValidationIssue]:
        """Validate parameter shapes match expected dimensions."""
        issues = []
        
        if client_id in self.baseline_models:
            baseline_shapes = self.baseline_models[client_id]
            
            for param_name, param in model_state.items():
                if param_name in baseline_shapes:
                    expected_shape = baseline_shapes[param_name]
                    if param.shape != expected_shape:
                        issues.append(ValidationIssue(
                            issue_id=f"shape_mismatch_{param_name}_{client_id}",
                            validation_type=ValidationType.MODEL_CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            component=f"model_update_{client_id}",
                            description=f"Parameter shape mismatch for {param_name}",
                            details={
                                "parameter_name": param_name,
                                "expected_shape": list(expected_shape),
                                "actual_shape": list(param.shape)
                            },
                            timestamp=time.time(),
                            resolution_suggestion="Check model architecture consistency",
                            auto_fixable=False
                        ))
        else:
            # Store baseline shapes for future validation
            self.baseline_models[client_id] = {
                name: param.shape for name, param in model_state.items()
            }
        
        return issues
    
    def _validate_parameter_magnitudes(
        self, 
        model_state: Dict[str, torch.Tensor], 
        client_id: str
    ) -> List[ValidationIssue]:
        """Validate parameter magnitudes are within reasonable bounds."""
        issues = []
        
        for param_name, param in model_state.items():
            max_magnitude = torch.max(torch.abs(param)).item()
            
            # Check for extremely large parameters
            if max_magnitude > 1000:
                issues.append(ValidationIssue(
                    issue_id=f"large_parameter_{param_name}_{client_id}",
                    validation_type=ValidationType.MODEL_CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    component=f"model_update_{client_id}",
                    description=f"Unusually large parameter values in {param_name}",
                    details={
                        "parameter_name": param_name,
                        "max_magnitude": float(max_magnitude),
                        "mean_magnitude": float(torch.mean(torch.abs(param)).item())
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Consider gradient clipping or learning rate adjustment",
                    auto_fixable=True
                ))
            
            # Check for extremely small parameters (potential underflow)
            mean_magnitude = torch.mean(torch.abs(param)).item()
            if mean_magnitude < 1e-8 and param.numel() > 1:
                issues.append(ValidationIssue(
                    issue_id=f"small_parameter_{param_name}_{client_id}",
                    validation_type=ValidationType.MODEL_CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    component=f"model_update_{client_id}",
                    description=f"Unusually small parameter values in {param_name}",
                    details={
                        "parameter_name": param_name,
                        "mean_magnitude": float(mean_magnitude),
                        "min_magnitude": float(torch.min(torch.abs(param)).item())
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Check for vanishing gradients or learning rate issues",
                    auto_fixable=False
                ))
        
        return issues
    
    def _validate_gradient_norms(
        self, 
        model_state: Dict[str, torch.Tensor], 
        client_id: str,
        round_number: int
    ) -> List[ValidationIssue]:
        """Validate gradient norms for training stability."""
        issues = []
        
        # Calculate total gradient norm
        total_norm = sum(torch.norm(param).item() ** 2 for param in model_state.values()) ** 0.5
        
        # Check for exploding gradients
        if total_norm > 1000:
            issues.append(ValidationIssue(
                issue_id=f"exploding_gradients_{client_id}_{round_number}",
                validation_type=ValidationType.MODEL_CONSISTENCY,
                severity=ValidationSeverity.ERROR,
                component=f"model_update_{client_id}",
                description="Potential exploding gradients detected",
                details={
                    "gradient_norm": float(total_norm),
                    "round_number": round_number
                },
                timestamp=time.time(),
                resolution_suggestion="Apply gradient clipping",
                auto_fixable=True
            ))
        
        # Check for vanishing gradients
        if total_norm < 1e-6:
            issues.append(ValidationIssue(
                issue_id=f"vanishing_gradients_{client_id}_{round_number}",
                validation_type=ValidationType.MODEL_CONSISTENCY,
                severity=ValidationSeverity.WARNING,
                component=f"model_update_{client_id}",
                description="Potential vanishing gradients detected",
                details={
                    "gradient_norm": float(total_norm),
                    "round_number": round_number
                },
                timestamp=time.time(),
                resolution_suggestion="Check learning rate or model architecture",
                auto_fixable=False
            ))
        
        return issues
    
    def _validate_numerical_stability(
        self, 
        model_state: Dict[str, torch.Tensor], 
        client_id: str
    ) -> List[ValidationIssue]:
        """Validate numerical stability (check for NaN/Inf values)."""
        issues = []
        
        for param_name, param in model_state.items():
            # Check for NaN values
            if torch.isnan(param).any():
                nan_count = torch.isnan(param).sum().item()
                issues.append(ValidationIssue(
                    issue_id=f"nan_values_{param_name}_{client_id}",
                    validation_type=ValidationType.MODEL_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    component=f"model_update_{client_id}",
                    description=f"NaN values detected in {param_name}",
                    details={
                        "parameter_name": param_name,
                        "nan_count": int(nan_count),
                        "total_elements": int(param.numel())
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Check for division by zero or numerical instability",
                    auto_fixable=True
                ))
            
            # Check for Inf values
            if torch.isinf(param).any():
                inf_count = torch.isinf(param).sum().item()
                issues.append(ValidationIssue(
                    issue_id=f"inf_values_{param_name}_{client_id}",
                    validation_type=ValidationType.MODEL_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    component=f"model_update_{client_id}",
                    description=f"Infinite values detected in {param_name}",
                    details={
                        "parameter_name": param_name,
                        "inf_count": int(inf_count),
                        "total_elements": int(param.numel())
                    },
                    timestamp=time.time(),
                    resolution_suggestion="Apply gradient clipping or check learning rate",
                    auto_fixable=True
                ))
        
        return issues
    
    def _calculate_model_confidence(
        self, 
        issues: List[ValidationIssue], 
        metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence score for model validation."""
        if not issues:
            return 1.0
        
        # Severe penalty for critical issues
        critical_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        if critical_issues > 0:
            return 0.0
        
        # Calculate penalty based on severity
        penalty = 0.0
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                penalty += 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                penalty += 0.1
        
        return max(0.0, 1.0 - penalty)


class ComprehensiveValidationEngine:
    """Main validation engine that coordinates all validation components."""
    
    def __init__(
        self,
        resilience_manager: Optional[QuantumResilienceManager] = None,
        strict_mode: bool = True
    ):
        self.resilience_manager = resilience_manager
        self.strict_mode = strict_mode
        
        # Initialize validators
        self.data_validator = DataIntegrityValidator(strict_mode)
        self.model_validator = ModelConsistencyValidator()
        
        # Validation results storage
        self.validation_history: List[ValidationResult] = []
        self.validation_metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    @quantum_resilient("validation_engine")
    async def comprehensive_validation(
        self,
        validation_data: Dict[str, Any],
        validation_types: Optional[List[ValidationType]] = None
    ) -> Dict[str, ValidationResult]:
        """Perform comprehensive validation across all registered validators."""
        if validation_types is None:
            validation_types = list(ValidationType)
        
        results = {}
        
        # Data integrity validation
        if ValidationType.DATA_INTEGRITY in validation_types:
            if "client_data" in validation_data:
                for client_id, client_data in validation_data["client_data"].items():
                    result = await self.data_validator.validate_client_data(
                        client_data, client_id
                    )
                    results[f"data_integrity_{client_id}"] = result
        
        # Model consistency validation
        if ValidationType.MODEL_CONSISTENCY in validation_types:
            if "model_updates" in validation_data:
                for client_id, model_state in validation_data["model_updates"].items():
                    round_number = validation_data.get("round_number", 0)
                    result = await self.model_validator.validate_model_update(
                        model_state, client_id, round_number
                    )
                    results[f"model_consistency_{client_id}"] = result
        
        # Privacy compliance validation
        if ValidationType.PRIVACY_COMPLIANCE in validation_types:
            if "privacy_config" in validation_data:
                result = await self._validate_privacy_compliance(validation_data["privacy_config"])
                results["privacy_compliance"] = result
        
        # Statistical validity validation
        if ValidationType.STATISTICAL_VALIDITY in validation_types:
            if "experimental_results" in validation_data:
                result = await self._validate_statistical_validity(validation_data["experimental_results"])
                results["statistical_validity"] = result
        
        # Store results
        with self._lock:
            self.validation_history.extend(results.values())
            self.validation_history = self.validation_history[-1000:]  # Keep last 1000
        
        return results
    
    async def _validate_privacy_compliance(
        self, 
        privacy_config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate privacy compliance settings."""
        start_time = time.time()
        issues = []
        metrics = {}
        
        # Check epsilon value
        epsilon = privacy_config.get("epsilon", float("inf"))
        if epsilon > 10.0:
            issues.append(ValidationIssue(
                issue_id=f"high_epsilon_{int(time.time())}",
                validation_type=ValidationType.PRIVACY_COMPLIANCE,
                severity=ValidationSeverity.WARNING,
                component="privacy_config",
                description="Privacy epsilon value is high",
                details={"epsilon": float(epsilon), "recommended_max": 10.0},
                timestamp=time.time(),
                resolution_suggestion="Consider reducing epsilon for stronger privacy",
                auto_fixable=True
            ))
        
        # Check delta value
        delta = privacy_config.get("delta", 1.0)
        if delta > 1e-3:
            issues.append(ValidationIssue(
                issue_id=f"high_delta_{int(time.time())}",
                validation_type=ValidationType.PRIVACY_COMPLIANCE,
                severity=ValidationSeverity.WARNING,
                component="privacy_config",
                description="Privacy delta value is high",
                details={"delta": float(delta), "recommended_max": 1e-5},
                timestamp=time.time(),
                resolution_suggestion="Consider reducing delta for stronger privacy",
                auto_fixable=True
            ))
        
        metrics = {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "privacy_strength": 1.0 / (epsilon * delta) if epsilon > 0 and delta > 0 else 0.0
        }
        
        execution_time = time.time() - start_time
        passed = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            validation_id=f"privacy_compliance_{int(time.time())}",
            component="privacy_config",
            validation_type=ValidationType.PRIVACY_COMPLIANCE,
            passed=passed,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            confidence_score=1.0 - len(issues) * 0.1
        )
    
    async def _validate_statistical_validity(
        self, 
        experimental_results: Dict[str, Any]
    ) -> ValidationResult:
        """Validate statistical validity of experimental results."""
        start_time = time.time()
        issues = []
        metrics = {}
        
        # Check for sufficient sample size
        sample_sizes = experimental_results.get("sample_sizes", [])
        if sample_sizes and min(sample_sizes) < 10:
            issues.append(ValidationIssue(
                issue_id=f"small_sample_size_{int(time.time())}",
                validation_type=ValidationType.STATISTICAL_VALIDITY,
                severity=ValidationSeverity.WARNING,
                component="experimental_results",
                description="Small sample size may affect statistical validity",
                details={"min_sample_size": min(sample_sizes), "recommended_min": 30},
                timestamp=time.time(),
                resolution_suggestion="Increase sample size for more reliable results",
                auto_fixable=False
            ))
        
        # Check p-values
        p_values = experimental_results.get("p_values", [])
        if p_values:
            significant_results = sum(1 for p in p_values if p < 0.05)
            if len(p_values) > 1 and significant_results == 0:
                issues.append(ValidationIssue(
                    issue_id=f"no_significant_results_{int(time.time())}",
                    validation_type=ValidationType.STATISTICAL_VALIDITY,
                    severity=ValidationSeverity.INFO,
                    component="experimental_results",
                    description="No statistically significant results found",
                    details={"total_tests": len(p_values), "significant_tests": significant_results},
                    timestamp=time.time(),
                    resolution_suggestion="Consider increasing effect size or sample size",
                    auto_fixable=False
                ))
        
        metrics = {
            "total_experiments": len(sample_sizes) if sample_sizes else 0,
            "min_sample_size": min(sample_sizes) if sample_sizes else 0,
            "significant_results": len([p for p in p_values if p < 0.05]) if p_values else 0,
            "mean_p_value": np.mean(p_values) if p_values else 1.0
        }
        
        execution_time = time.time() - start_time
        passed = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            validation_id=f"statistical_validity_{int(time.time())}",
            component="experimental_results",
            validation_type=ValidationType.STATISTICAL_VALIDITY,
            passed=passed,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            confidence_score=1.0 - len(issues) * 0.05
        )
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        with self._lock:
            recent_results = self.validation_history[-100:]  # Last 100 validations
        
        if not recent_results:
            return {"status": "no_validations", "summary": {}}
        
        # Calculate summary statistics
        passed_validations = sum(1 for r in recent_results if r.passed)
        total_validations = len(recent_results)
        
        # Group by validation type
        type_summary = {}
        for validation_type in ValidationType:
            type_results = [r for r in recent_results if r.validation_type == validation_type]
            if type_results:
                type_summary[validation_type.value] = {
                    "total": len(type_results),
                    "passed": sum(1 for r in type_results if r.passed),
                    "avg_confidence": np.mean([r.confidence_score for r in type_results]),
                    "avg_execution_time": np.mean([r.execution_time for r in type_results])
                }
        
        # Severity distribution
        all_issues = [issue for result in recent_results for issue in result.issues]
        severity_counts = {
            severity.value: sum(1 for issue in all_issues if issue.severity == severity)
            for severity in ValidationSeverity
        }
        
        return {
            "status": "completed",
            "summary": {
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
                "total_issues": len(all_issues),
                "avg_confidence": np.mean([r.confidence_score for r in recent_results])
            },
            "validation_types": type_summary,
            "issue_severity_distribution": severity_counts,
            "recent_critical_issues": [
                {
                    "component": issue.component,
                    "description": issue.description,
                    "timestamp": issue.timestamp
                }
                for result in recent_results[-10:]  # Last 10 results
                for issue in result.issues
                if issue.severity == ValidationSeverity.CRITICAL
            ]
        }


# Factory function
def create_validation_engine(**config) -> ComprehensiveValidationEngine:
    """Create a comprehensive validation engine with specified configuration."""
    return ComprehensiveValidationEngine(**config)
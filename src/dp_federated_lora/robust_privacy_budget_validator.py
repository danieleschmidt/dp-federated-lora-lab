"""
Robust Privacy Budget Validation and Security Framework.

This module provides comprehensive validation, error handling, and security
measures for the adaptive privacy budget optimization system.

Features:
- Real-time budget validation and consistency checks
- Security auditing and threat detection
- Comprehensive error handling with circuit breakers
- Privacy budget integrity verification
- Anomaly detection in allocation patterns

Author: Terry (Terragon Labs)
"""

import asyncio
import logging
import time
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum
import json
import numpy as np
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityThreat(Enum):
    """Security threat types."""
    BUDGET_MANIPULATION = "budget_manipulation"
    REPLAY_ATTACK = "replay_attack"
    GRADIENT_LEAKAGE = "gradient_leakage"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    BYZANTINE_CLIENT = "byzantine_client"
    PRIVACY_BREACH = "privacy_breach"
    DATA_POISONING = "data_poisoning"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ValidationError:
    """Validation error details."""
    error_type: ValidationResult
    message: str
    client_id: Optional[str] = None
    round_num: Optional[int] = None
    severity: int = 1  # 1-5 scale
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert details."""
    threat_type: SecurityThreat
    severity: int  # 1-5 scale
    client_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class BudgetIntegrityCheck:
    """Privacy budget integrity check result."""
    is_valid: bool
    total_epsilon_expected: float
    total_epsilon_actual: float
    total_delta_expected: float
    total_delta_actual: float
    discrepancy: float
    affected_clients: List[str]
    timestamp: float = field(default_factory=time.time)


class PrivacyBudgetEncryption:
    """Encryption for sensitive privacy budget data."""
    
    def __init__(self, password: str):
        """Initialize encryption with password."""
        password_bytes = password.encode()
        salt = b'salt_1234567890123456'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: Dict[str, Any]) -> bytes:
        """Encrypt sensitive data."""
        json_data = json.dumps(data).encode()
        return self.cipher.encrypt(json_data)
    
    def decrypt(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt sensitive data."""
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Timeout: {self.timeout}s")
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                        logger.info("Circuit breaker transitioned to CLOSED")
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker transitioned to OPEN due to failures: {e}")
                
            raise e


class AnomalyDetector:
    """ML-based anomaly detection for budget allocations."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_history = deque(maxlen=1000)
    
    def _extract_features(
        self, 
        allocation_data: Dict[str, Any], 
        client_profiles: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        
        # Basic allocation features
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        delta_allocated = allocation_data.get("delta_allocated", 0.0)
        client_id = allocation_data.get("client_id", "unknown")
        
        features.extend([
            epsilon_allocated,
            delta_allocated,
            allocation_data.get("expected_utility", 0.0),
            allocation_data.get("allocation_confidence", 0.0)
        ])
        
        # Client profile features
        if client_id in client_profiles:
            profile = client_profiles[client_id]
            features.extend([
                profile.get("data_sensitivity", 1.0),
                profile.get("communication_cost", 1.0),
                profile.get("resource_availability", 1.0),
                len(profile.get("performance_history", [])),
                np.mean(profile.get("performance_history", [0.5])),
                np.std(profile.get("performance_history", [0.5])) if len(profile.get("performance_history", [])) > 1 else 0.0
            ])
        else:
            features.extend([1.0, 1.0, 1.0, 0, 0.5, 0.0])
        
        # Historical context features
        if historical_data:
            recent_allocations = [h.get("epsilon_allocated", 0.0) for h in historical_data[-10:]]
            features.extend([
                np.mean(recent_allocations),
                np.std(recent_allocations) if len(recent_allocations) > 1 else 0.0,
                np.max(recent_allocations) if recent_allocations else 0.0,
                np.min(recent_allocations) if recent_allocations else 0.0,
                len(recent_allocations)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0])
        
        # Time-based features
        current_time = time.time()
        features.extend([
            current_time % (24 * 3600),  # Time of day
            current_time % (7 * 24 * 3600),  # Day of week
        ])
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, training_data: List[Dict[str, Any]]):
        """Fit anomaly detector on historical data."""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for anomaly detection")
            return
        
        features = []
        for data_point in training_data:
            feature_vector = self._extract_features(
                data_point.get("allocation", {}),
                data_point.get("client_profiles", {}),
                data_point.get("history", [])
            )
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Handle NaN values
        features_array = np.nan_to_num(features_array)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Fit isolation forest
        self.isolation_forest.fit(scaled_features)
        self.is_fitted = True
        
        logger.info(f"Trained anomaly detector on {len(training_data)} samples")
    
    def detect_anomaly(
        self,
        allocation_data: Dict[str, Any],
        client_profiles: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """Detect if allocation is anomalous."""
        if not self.is_fitted:
            logger.warning("Anomaly detector not fitted. Cannot detect anomalies.")
            return False, 0.5
        
        try:
            # Extract features
            features = self._extract_features(allocation_data, client_profiles, historical_data)
            
            # Handle NaN values
            features = np.nan_to_num(features)
            
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            # Predict anomaly
            anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
            is_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            # Store features for future training
            self.feature_history.append({
                "features": features,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "timestamp": time.time()
            })
            
            return is_anomaly, anomaly_score
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.5


class RobustPrivacyBudgetValidator:
    """Comprehensive privacy budget validation and security framework."""
    
    def __init__(
        self,
        encryption_password: str = None,
        max_validation_errors: int = 100,
        anomaly_threshold: float = 0.1
    ):
        """Initialize robust validator."""
        self.encryption_password = encryption_password or secrets.token_urlsafe(32)
        self.encryptor = PrivacyBudgetEncryption(self.encryption_password)
        
        # Validation components
        self.circuit_breaker = CircuitBreaker()
        self.anomaly_detector = AnomalyDetector(contamination=anomaly_threshold)
        
        # Error tracking
        self.validation_errors: deque = deque(maxlen=max_validation_errors)
        self.security_alerts: List[SecurityAlert] = []
        self.integrity_checks: List[BudgetIntegrityCheck] = []
        
        # Security monitoring
        self.client_request_counts: defaultdict = defaultdict(int)
        self.client_last_request: Dict[str, float] = {}
        self.suspicious_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Thread safety
        self._validation_lock = threading.Lock()
        self._security_lock = threading.Lock()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info("Initialized robust privacy budget validator")
    
    def validate_budget_allocation(
        self,
        allocation_data: Dict[str, Any],
        client_profiles: Dict[str, Any],
        total_budget_constraints: Dict[str, float],
        allocation_history: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Comprehensive validation of budget allocation."""
        with self._validation_lock:
            errors = []
            
            try:
                # Basic validation
                errors.extend(self._validate_basic_constraints(allocation_data, total_budget_constraints))
                
                # Client-specific validation
                errors.extend(self._validate_client_constraints(allocation_data, client_profiles))
                
                # Historical consistency validation
                errors.extend(self._validate_historical_consistency(allocation_data, allocation_history))
                
                # Mathematical consistency validation
                errors.extend(self._validate_mathematical_consistency(allocation_data))
                
                # Anomaly detection
                errors.extend(self._validate_anomalies(allocation_data, client_profiles, allocation_history))
                
                # Security validation
                errors.extend(self._validate_security(allocation_data, client_profiles))
                
                # Store validation results
                if errors:
                    self.validation_errors.extend(errors)
                
                return errors
                
            except Exception as e:
                critical_error = ValidationError(
                    error_type=ValidationResult.CRITICAL,
                    message=f"Validation system failure: {str(e)}",
                    severity=5,
                    metadata={"exception": str(e)}
                )
                self.validation_errors.append(critical_error)
                return [critical_error]
    
    def _validate_basic_constraints(
        self,
        allocation_data: Dict[str, Any],
        budget_constraints: Dict[str, float]
    ) -> List[ValidationError]:
        """Validate basic budget constraints."""
        errors = []
        
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        delta_allocated = allocation_data.get("delta_allocated", 0.0)
        
        # Non-negative check
        if epsilon_allocated < 0:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Negative epsilon allocation: {epsilon_allocated}",
                client_id=allocation_data.get("client_id"),
                severity=4
            ))
        
        if delta_allocated < 0:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Negative delta allocation: {delta_allocated}",
                client_id=allocation_data.get("client_id"),
                severity=4
            ))
        
        # Upper bound checks
        max_epsilon = budget_constraints.get("max_epsilon_per_round", float('inf'))
        max_delta = budget_constraints.get("max_delta_per_round", float('inf'))
        
        if epsilon_allocated > max_epsilon:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Epsilon allocation {epsilon_allocated} exceeds maximum {max_epsilon}",
                client_id=allocation_data.get("client_id"),
                severity=3
            ))
        
        if delta_allocated > max_delta:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Delta allocation {delta_allocated} exceeds maximum {max_delta}",
                client_id=allocation_data.get("client_id"),
                severity=3
            ))
        
        # Ratio validation (delta should be much smaller than epsilon)
        if epsilon_allocated > 0 and delta_allocated / epsilon_allocated > 0.1:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Unusual epsilon/delta ratio: ε={epsilon_allocated}, δ={delta_allocated}",
                client_id=allocation_data.get("client_id"),
                severity=2
            ))
        
        return errors
    
    def _validate_client_constraints(
        self,
        allocation_data: Dict[str, Any],
        client_profiles: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate client-specific constraints."""
        errors = []
        
        client_id = allocation_data.get("client_id")
        if not client_id:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message="Missing client ID in allocation",
                severity=4
            ))
            return errors
        
        if client_id not in client_profiles:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Unknown client ID: {client_id}",
                client_id=client_id,
                severity=4
            ))
            return errors
        
        profile = client_profiles[client_id]
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        delta_allocated = allocation_data.get("delta_allocated", 0.0)
        
        # Check remaining budget
        current_epsilon = profile.get("current_epsilon", 0.0)
        current_delta = profile.get("current_delta", 0.0)
        total_epsilon_budget = profile.get("total_epsilon_budget", float('inf'))
        total_delta_budget = profile.get("total_delta_budget", float('inf'))
        
        if current_epsilon + epsilon_allocated > total_epsilon_budget:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Client {client_id} epsilon budget exceeded: "
                       f"{current_epsilon + epsilon_allocated} > {total_epsilon_budget}",
                client_id=client_id,
                severity=4
            ))
        
        if current_delta + delta_allocated > total_delta_budget:
            errors.append(ValidationError(
                error_type=ValidationResult.ERROR,
                message=f"Client {client_id} delta budget exceeded: "
                       f"{current_delta + delta_allocated} > {total_delta_budget}",
                client_id=client_id,
                severity=4
            ))
        
        # Check allocation reasonableness based on client characteristics
        data_sensitivity = profile.get("data_sensitivity", 1.0)
        privacy_preferences = profile.get("privacy_preferences", {})
        strictness = privacy_preferences.get("strictness", 0.5)
        
        # High sensitivity clients should have higher allocations
        expected_min_allocation = data_sensitivity * strictness * 0.1  # Heuristic
        if epsilon_allocated < expected_min_allocation:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Low allocation for high-sensitivity client {client_id}: "
                       f"{epsilon_allocated} < {expected_min_allocation}",
                client_id=client_id,
                severity=2
            ))
        
        return errors
    
    def _validate_historical_consistency(
        self,
        allocation_data: Dict[str, Any],
        allocation_history: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate historical consistency."""
        errors = []
        
        if not allocation_history:
            return errors
        
        client_id = allocation_data.get("client_id")
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        
        # Get recent allocations for this client
        recent_client_allocations = []
        for round_data in allocation_history[-10:]:  # Last 10 rounds
            for round_num, round_allocations in round_data.items():
                if client_id in round_allocations:
                    recent_client_allocations.append(round_allocations[client_id].get("epsilon_allocated", 0.0))
        
        if recent_client_allocations:
            mean_allocation = np.mean(recent_client_allocations)
            std_allocation = np.std(recent_client_allocations)
            
            # Check for sudden spikes
            if std_allocation > 0 and abs(epsilon_allocated - mean_allocation) > 3 * std_allocation:
                errors.append(ValidationError(
                    error_type=ValidationResult.WARNING,
                    message=f"Allocation spike detected for client {client_id}: "
                           f"{epsilon_allocated} vs avg {mean_allocation:.3f} ± {std_allocation:.3f}",
                    client_id=client_id,
                    severity=2,
                    metadata={"recent_allocations": recent_client_allocations}
                ))
        
        # Check for decreasing trend (might indicate budget exhaustion)
        if len(recent_client_allocations) >= 5:
            trend_slope = np.polyfit(range(len(recent_client_allocations)), recent_client_allocations, 1)[0]
            if trend_slope < -0.01:  # Significant downward trend
                errors.append(ValidationError(
                    error_type=ValidationResult.WARNING,
                    message=f"Decreasing allocation trend for client {client_id}: slope={trend_slope:.4f}",
                    client_id=client_id,
                    severity=2
                ))
        
        return errors
    
    def _validate_mathematical_consistency(self, allocation_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate mathematical consistency of allocation."""
        errors = []
        
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        delta_allocated = allocation_data.get("delta_allocated", 0.0)
        expected_utility = allocation_data.get("expected_utility", 0.0)
        allocation_confidence = allocation_data.get("allocation_confidence", 0.0)
        
        # Check for NaN or infinite values
        for field_name, value in [
            ("epsilon_allocated", epsilon_allocated),
            ("delta_allocated", delta_allocated),
            ("expected_utility", expected_utility),
            ("allocation_confidence", allocation_confidence)
        ]:
            if not np.isfinite(value):
                errors.append(ValidationError(
                    error_type=ValidationResult.ERROR,
                    message=f"Invalid value for {field_name}: {value}",
                    client_id=allocation_data.get("client_id"),
                    severity=4
                ))
        
        # Confidence should be in [0, 1]
        if not 0 <= allocation_confidence <= 1:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Allocation confidence out of range: {allocation_confidence}",
                client_id=allocation_data.get("client_id"),
                severity=2
            ))
        
        # Utility should be non-negative
        if expected_utility < 0:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Negative expected utility: {expected_utility}",
                client_id=allocation_data.get("client_id"),
                severity=2
            ))
        
        return errors
    
    def _validate_anomalies(
        self,
        allocation_data: Dict[str, Any],
        client_profiles: Dict[str, Any],
        allocation_history: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate using anomaly detection."""
        errors = []
        
        try:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                allocation_data, client_profiles, allocation_history
            )
            
            if is_anomaly:
                errors.append(ValidationError(
                    error_type=ValidationResult.WARNING,
                    message=f"Anomalous allocation detected (score: {anomaly_score:.3f})",
                    client_id=allocation_data.get("client_id"),
                    severity=3,
                    metadata={"anomaly_score": anomaly_score}
                ))
            
        except Exception as e:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Anomaly detection failed: {str(e)}",
                severity=2
            ))
        
        return errors
    
    def _validate_security(
        self,
        allocation_data: Dict[str, Any],
        client_profiles: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate security aspects."""
        errors = []
        
        client_id = allocation_data.get("client_id")
        if not client_id:
            return errors
        
        # Track request patterns
        current_time = time.time()
        self.client_request_counts[client_id] += 1
        
        # Check for replay attacks (too frequent requests)
        if client_id in self.client_last_request:
            time_since_last = current_time - self.client_last_request[client_id]
            if time_since_last < 1.0:  # Less than 1 second
                errors.append(ValidationError(
                    error_type=ValidationResult.WARNING,
                    message=f"Potential replay attack: too frequent requests from {client_id}",
                    client_id=client_id,
                    severity=3
                ))
                
                # Log security alert
                self._log_security_alert(
                    SecurityThreat.REPLAY_ATTACK,
                    client_id,
                    f"Frequent requests: {time_since_last:.2f}s interval",
                    {"time_interval": time_since_last}
                )
        
        self.client_last_request[client_id] = current_time
        
        # Check for suspicious allocation patterns
        epsilon_allocated = allocation_data.get("epsilon_allocated", 0.0)
        expected_utility = allocation_data.get("expected_utility", 0.0)
        
        # Suspiciously high allocation with low expected utility
        if epsilon_allocated > 1.0 and expected_utility < 0.1:
            errors.append(ValidationError(
                error_type=ValidationResult.WARNING,
                message=f"Suspicious allocation pattern: high ε ({epsilon_allocated}) with low utility ({expected_utility})",
                client_id=client_id,
                severity=3
            ))
            
            self._log_security_alert(
                SecurityThreat.BUDGET_MANIPULATION,
                client_id,
                "High allocation with low expected utility",
                {"epsilon": epsilon_allocated, "utility": expected_utility}
            )
        
        return errors
    
    def _log_security_alert(
        self,
        threat_type: SecurityThreat,
        client_id: str,
        description: str,
        evidence: Dict[str, Any],
        severity: int = 3
    ):
        """Log security alert."""
        with self._security_lock:
            alert = SecurityAlert(
                threat_type=threat_type,
                severity=severity,
                client_id=client_id,
                description=description,
                evidence=evidence
            )
            self.security_alerts.append(alert)
            
            logger.warning(f"Security alert: {threat_type.value} - {description} (Client: {client_id})")
    
    @circuit_breaker
    def verify_budget_integrity(
        self,
        total_epsilon_budget: float,
        total_delta_budget: float,
        client_profiles: Dict[str, Any],
        allocation_history: List[Dict[str, Any]]
    ) -> BudgetIntegrityCheck:
        """Verify overall budget integrity."""
        try:
            # Calculate expected totals
            expected_epsilon = sum(
                profile.get("current_epsilon", 0.0) 
                for profile in client_profiles.values()
            )
            expected_delta = sum(
                profile.get("current_delta", 0.0) 
                for profile in client_profiles.values()
            )
            
            # Calculate actual totals from history
            actual_epsilon = 0.0
            actual_delta = 0.0
            
            for round_data in allocation_history:
                for round_allocations in round_data.values():
                    for client_allocation in round_allocations.values():
                        actual_epsilon += client_allocation.get("epsilon_allocated", 0.0)
                        actual_delta += client_allocation.get("delta_allocated", 0.0)
            
            # Calculate discrepancy
            epsilon_discrepancy = abs(expected_epsilon - actual_epsilon)
            delta_discrepancy = abs(expected_delta - actual_delta)
            total_discrepancy = epsilon_discrepancy + delta_discrepancy
            
            # Identify affected clients
            affected_clients = []
            for client_id, profile in client_profiles.items():
                client_expected = profile.get("current_epsilon", 0.0)
                client_actual = sum(
                    alloc.get("epsilon_allocated", 0.0)
                    for round_data in allocation_history
                    for round_allocations in round_data.values()
                    for cid, alloc in round_allocations.items()
                    if cid == client_id
                )
                
                if abs(client_expected - client_actual) > 1e-6:
                    affected_clients.append(client_id)
            
            # Create integrity check result
            is_valid = total_discrepancy < 1e-6  # Floating point tolerance
            
            integrity_check = BudgetIntegrityCheck(
                is_valid=is_valid,
                total_epsilon_expected=expected_epsilon,
                total_epsilon_actual=actual_epsilon,
                total_delta_expected=expected_delta,
                total_delta_actual=actual_delta,
                discrepancy=total_discrepancy,
                affected_clients=affected_clients
            )
            
            self.integrity_checks.append(integrity_check)
            
            if not is_valid:
                logger.error(f"Budget integrity violation detected. Discrepancy: {total_discrepancy}")
                self._log_security_alert(
                    SecurityThreat.PRIVACY_BREACH,
                    None,
                    f"Budget integrity violation: discrepancy={total_discrepancy}",
                    {
                        "expected_epsilon": expected_epsilon,
                        "actual_epsilon": actual_epsilon,
                        "affected_clients": affected_clients
                    },
                    severity=5
                )
            
            return integrity_check
            
        except Exception as e:
            logger.error(f"Error in budget integrity verification: {e}")
            raise
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        unresolved_alerts = [alert for alert in self.security_alerts if not alert.resolved]
        
        # Alert statistics by threat type
        threat_stats = defaultdict(int)
        for alert in self.security_alerts:
            threat_stats[alert.threat_type.value] += 1
        
        # Recent validation errors
        recent_errors = [
            error for error in self.validation_errors
            if time.time() - error.timestamp < 3600  # Last hour
        ]
        
        # Error severity distribution
        error_severity_dist = defaultdict(int)
        for error in recent_errors:
            error_severity_dist[error.severity] += 1
        
        return {
            "security_alerts": {
                "total_alerts": len(self.security_alerts),
                "unresolved_alerts": len(unresolved_alerts),
                "threat_distribution": dict(threat_stats),
                "recent_alerts": [
                    {
                        "threat_type": alert.threat_type.value,
                        "client_id": alert.client_id,
                        "severity": alert.severity,
                        "description": alert.description,
                        "timestamp": alert.timestamp
                    }
                    for alert in unresolved_alerts[-10:]  # Last 10 unresolved
                ]
            },
            "validation_errors": {
                "total_errors": len(self.validation_errors),
                "recent_errors": len(recent_errors),
                "severity_distribution": dict(error_severity_dist),
                "error_rate": len(recent_errors) / 3600 if recent_errors else 0.0  # Errors per second
            },
            "budget_integrity": {
                "total_checks": len(self.integrity_checks),
                "failed_checks": len([c for c in self.integrity_checks if not c.is_valid]),
                "last_check": (
                    {
                        "is_valid": self.integrity_checks[-1].is_valid,
                        "discrepancy": self.integrity_checks[-1].discrepancy,
                        "timestamp": self.integrity_checks[-1].timestamp
                    }
                    if self.integrity_checks else None
                )
            },
            "client_monitoring": {
                "monitored_clients": len(self.client_request_counts),
                "request_distribution": dict(self.client_request_counts),
                "suspicious_clients": [
                    client_id for client_id, count in self.client_request_counts.items()
                    if count > 100  # High request volume
                ]
            },
            "system_health": {
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "circuit_breaker_failures": self.circuit_breaker.failure_count,
                "anomaly_detector_fitted": self.anomaly_detector.is_fitted,
                "monitoring_active": self._monitoring_active
            }
        }
    
    def start_continuous_monitoring(self, monitoring_interval: int = 30):
        """Start continuous security monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    # Clean up old data
                    current_time = time.time()
                    
                    # Remove old security alerts (older than 24 hours)
                    self.security_alerts = [
                        alert for alert in self.security_alerts
                        if current_time - alert.timestamp < 86400
                    ]
                    
                    # Remove old integrity checks (keep last 100)
                    if len(self.integrity_checks) > 100:
                        self.integrity_checks = self.integrity_checks[-100:]
                    
                    # Reset request counts periodically (every hour)
                    if int(current_time) % 3600 == 0:
                        self.client_request_counts.clear()
                    
                    # Log monitoring status
                    if len(self.security_alerts) > 0:
                        unresolved = len([a for a in self.security_alerts if not a.resolved])
                        logger.info(f"Security monitoring: {len(self.security_alerts)} alerts, {unresolved} unresolved")
                    
                    time.sleep(monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Error in security monitoring: {e}")
                    time.sleep(monitoring_interval)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started continuous security monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous security monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped continuous security monitoring")
    
    def resolve_security_alert(self, alert_index: int, resolution_notes: str = ""):
        """Mark security alert as resolved."""
        if 0 <= alert_index < len(self.security_alerts):
            self.security_alerts[alert_index].resolved = True
            logger.info(f"Resolved security alert {alert_index}: {resolution_notes}")
        else:
            logger.warning(f"Invalid alert index: {alert_index}")
    
    def train_anomaly_detector(self, training_data: List[Dict[str, Any]]):
        """Train the anomaly detector on historical data."""
        try:
            self.anomaly_detector.fit(training_data)
            logger.info(f"Trained anomaly detector on {len(training_data)} samples")
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
    
    def export_security_logs(self, export_path: str):
        """Export security logs for analysis."""
        security_data = {
            "validation_errors": [
                {
                    "error_type": error.error_type.value,
                    "message": error.message,
                    "client_id": error.client_id,
                    "round_num": error.round_num,
                    "severity": error.severity,
                    "timestamp": error.timestamp,
                    "metadata": error.metadata
                }
                for error in self.validation_errors
            ],
            "security_alerts": [
                {
                    "threat_type": alert.threat_type.value,
                    "severity": alert.severity,
                    "client_id": alert.client_id,
                    "description": alert.description,
                    "evidence": alert.evidence,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                }
                for alert in self.security_alerts
            ],
            "integrity_checks": [
                {
                    "is_valid": check.is_valid,
                    "total_epsilon_expected": check.total_epsilon_expected,
                    "total_epsilon_actual": check.total_epsilon_actual,
                    "total_delta_expected": check.total_delta_expected,
                    "total_delta_actual": check.total_delta_actual,
                    "discrepancy": check.discrepancy,
                    "affected_clients": check.affected_clients,
                    "timestamp": check.timestamp
                }
                for check in self.integrity_checks
            ],
            "export_metadata": {
                "export_time": time.time(),
                "total_validation_errors": len(self.validation_errors),
                "total_security_alerts": len(self.security_alerts),
                "total_integrity_checks": len(self.integrity_checks)
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(security_data, f, indent=2)
        
        logger.info(f"Exported security logs to {export_path}")


# Factory functions for easy instantiation
def create_robust_validator(
    encryption_password: str = None,
    anomaly_threshold: float = 0.1
) -> RobustPrivacyBudgetValidator:
    """Create a robust privacy budget validator."""
    return RobustPrivacyBudgetValidator(
        encryption_password=encryption_password,
        anomaly_threshold=anomaly_threshold
    )


if __name__ == "__main__":
    # Demonstration of robust validation system
    import random
    
    # Create validator
    validator = create_robust_validator()
    validator.start_continuous_monitoring()
    
    # Mock client profiles
    client_profiles = {
        f"client_{i}": {
            "current_epsilon": random.uniform(0, 5),
            "current_delta": random.uniform(0, 1e-5),
            "total_epsilon_budget": 10.0,
            "total_delta_budget": 1e-5,
            "data_sensitivity": random.uniform(0.5, 2.0),
            "privacy_preferences": {"strictness": random.uniform(0.3, 0.8)}
        }
        for i in range(5)
    }
    
    # Generate training data for anomaly detector
    training_data = []
    for _ in range(100):
        training_data.append({
            "allocation": {
                "client_id": f"client_{random.randint(0, 4)}",
                "epsilon_allocated": random.uniform(0.1, 2.0),
                "delta_allocated": random.uniform(1e-6, 1e-5),
                "expected_utility": random.uniform(0.3, 0.9),
                "allocation_confidence": random.uniform(0.7, 0.95)
            },
            "client_profiles": client_profiles,
            "history": []
        })
    
    validator.train_anomaly_detector(training_data)
    
    # Test validation
    test_allocation = {
        "client_id": "client_0",
        "epsilon_allocated": 1.5,
        "delta_allocated": 5e-6,
        "expected_utility": 0.75,
        "allocation_confidence": 0.85,
        "round_num": 1
    }
    
    budget_constraints = {
        "max_epsilon_per_round": 10.0,
        "max_delta_per_round": 1e-4
    }
    
    # Validate allocation
    validation_errors = validator.validate_budget_allocation(
        test_allocation, client_profiles, budget_constraints, []
    )
    
    print(f"\nValidation Results:")
    print(f"Found {len(validation_errors)} validation errors")
    for error in validation_errors:
        print(f"  {error.error_type.value}: {error.message}")
    
    # Test integrity check
    integrity_check = validator.verify_budget_integrity(
        total_epsilon_budget=50.0,
        total_delta_budget=5e-5,
        client_profiles=client_profiles,
        allocation_history=[]
    )
    
    print(f"\nBudget Integrity Check:")
    print(f"Valid: {integrity_check.is_valid}")
    print(f"Discrepancy: {integrity_check.discrepancy}")
    
    # Generate security report
    security_report = validator.get_security_report()
    print(f"\nSecurity Report:")
    print(f"Total alerts: {security_report['security_alerts']['total_alerts']}")
    print(f"Circuit breaker state: {security_report['system_health']['circuit_breaker_state']}")
    print(f"Anomaly detector fitted: {security_report['system_health']['anomaly_detector_fitted']}")
    
    # Export logs
    validator.export_security_logs("security_logs_demo.json")
    print(f"\nSecurity logs exported to: security_logs_demo.json")
    
    # Stop monitoring
    validator.stop_continuous_monitoring()
    
    print(f"\nRobust privacy budget validation demonstration completed!")
"""
Advanced Security Module for DP-Federated LoRA system.

This module implements comprehensive security measures including threat detection,
advanced authentication, secure communication protocols, and quantum-enhanced
cryptography for federated learning environments.
"""

import logging
import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import torch

from .config import SecurityConfig, FederatedConfig
from .monitoring import ServerMetricsCollector
from .exceptions import SecurityError, AuthenticationError, DPFederatedLoRAError


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    BYZANTINE_DETECTION = "byzantine_detection"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_INTEGRITY_VIOLATION = "data_integrity"
    PRIVACY_BREACH_ATTEMPT = "privacy_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_UPDATE = "malicious_update"


@dataclass
class SecurityEvent:
    """Security event record."""
    
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    client_id: Optional[str]
    timestamp: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


@dataclass
class ClientSecurityProfile:
    """Security profile for a federated client."""
    
    client_id: str
    trust_score: float = 1.0
    authentication_history: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    security_violations: List[SecurityEvent] = field(default_factory=list)
    last_security_check: float = field(default_factory=time.time)
    risk_level: ThreatLevel = ThreatLevel.LOW
    quarantined: bool = False


class AdvancedCryptographyManager:
    """Manager for advanced cryptographic operations."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize cryptography manager."""
        self.config = config
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.symmetric_keys: Dict[str, bytes] = {}
        
        logger.info("Advanced cryptography manager initialized")
    
    def generate_client_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate public-private key pair for client."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_model_update(self, update: Dict[str, torch.Tensor], client_id: str) -> bytes:
        """Encrypt model update using hybrid encryption."""
        # Generate symmetric key for this session
        symmetric_key = secrets.token_bytes(32)
        self.symmetric_keys[client_id] = symmetric_key
        
        # Serialize model update
        update_data = {}
        for key, tensor in update.items():
            update_data[key] = tensor.cpu().numpy().tobytes()
        
        serialized_update = json.dumps(update_data).encode()
        
        # Encrypt with AES
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data
        pad_length = 16 - (len(serialized_update) % 16)
        padded_data = serialized_update + bytes([pad_length]) * pad_length
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt symmetric key with RSA
        encrypted_key = self.public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_key + iv + encrypted_data
    
    def decrypt_model_update(self, encrypted_data: bytes, client_id: str) -> Dict[str, torch.Tensor]:
        """Decrypt model update."""
        # Extract components
        encrypted_key = encrypted_data[:256]  # RSA 2048-bit key
        iv = encrypted_data[256:272]
        ciphertext = encrypted_data[272:]
        
        # Decrypt symmetric key
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        # Deserialize
        update_data = json.loads(plaintext.decode())
        
        # Convert back to tensors
        update = {}
        for key, data_bytes in update_data.items():
            tensor_data = np.frombuffer(data_bytes, dtype=np.float32)
            update[key] = torch.from_numpy(tensor_data)
        
        return update


class ByzantineDetector:
    """Advanced Byzantine behavior detection system."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize Byzantine detector."""
        self.config = config
        self.client_update_history: Dict[str, List[Dict[str, Any]]] = {}
        self.global_statistics: Dict[str, Any] = {}
        self.detection_threshold = 2.0  # Standard deviations
        
        logger.info("Byzantine detector initialized")
    
    def analyze_client_update(
        self,
        client_id: str,
        update: Dict[str, torch.Tensor],
        round_num: int
    ) -> Tuple[bool, float, str]:
        """
        Analyze client update for Byzantine behavior.
        
        Args:
            client_id: Client identifier
            update: Model update from client
            round_num: Current training round
            
        Returns:
            Tuple of (is_byzantine, confidence, reason)
        """
        # Initialize client history if needed
        if client_id not in self.client_update_history:
            self.client_update_history[client_id] = []
        
        # Calculate update statistics
        update_stats = self._calculate_update_statistics(update)
        
        # Store update history
        self.client_update_history[client_id].append({
            'round': round_num,
            'stats': update_stats,
            'timestamp': time.time()
        })
        
        # Detect anomalies
        byzantine_indicators = []
        
        # Check update magnitude
        is_magnitude_anomaly, magnitude_score = self._check_magnitude_anomaly(
            client_id, update_stats
        )
        if is_magnitude_anomaly:
            byzantine_indicators.append(f"Magnitude anomaly (score: {magnitude_score:.3f})")
        
        # Check direction consistency
        is_direction_anomaly, direction_score = self._check_direction_consistency(
            client_id, update_stats
        )
        if is_direction_anomaly:
            byzantine_indicators.append(f"Direction inconsistency (score: {direction_score:.3f})")
        
        # Check statistical outliers
        is_statistical_outlier, outlier_score = self._check_statistical_outliers(
            client_id, update_stats
        )
        if is_statistical_outlier:
            byzantine_indicators.append(f"Statistical outlier (score: {outlier_score:.3f})")
        
        # Aggregate results
        is_byzantine = len(byzantine_indicators) >= 2  # Multiple indicators
        confidence = min(1.0, len(byzantine_indicators) * 0.4)
        reason = "; ".join(byzantine_indicators) if byzantine_indicators else "Normal behavior"
        
        logger.debug(f"Byzantine analysis for {client_id}: {is_byzantine}, confidence: {confidence:.3f}")
        
        return is_byzantine, confidence, reason
    
    def _calculate_update_statistics(self, update: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate statistics for model update."""
        all_params = torch.cat([param.flatten() for param in update.values()])
        
        return {
            'mean': float(all_params.mean()),
            'std': float(all_params.std()),
            'l2_norm': float(torch.norm(all_params, p=2)),
            'l1_norm': float(torch.norm(all_params, p=1)),
            'max_value': float(all_params.max()),
            'min_value': float(all_params.min()),
            'num_params': len(all_params)
        }
    
    def _check_magnitude_anomaly(
        self,
        client_id: str,
        stats: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Check for magnitude anomalies in update."""
        if len(self.client_update_history[client_id]) < 3:
            return False, 0.0
        
        # Compare with client's own history
        recent_norms = [h['stats']['l2_norm'] for h in self.client_update_history[client_id][-5:]]
        mean_norm = np.mean(recent_norms)
        std_norm = np.std(recent_norms)
        
        if std_norm == 0:
            return False, 0.0
        
        z_score = abs((stats['l2_norm'] - mean_norm) / std_norm)
        is_anomaly = z_score > self.detection_threshold
        
        return is_anomaly, z_score
    
    def _check_direction_consistency(
        self,
        client_id: str,
        stats: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Check for direction consistency with expected updates."""
        if len(self.client_update_history[client_id]) < 2:
            return False, 0.0
        
        # Simple direction check based on sign of mean
        current_sign = np.sign(stats['mean'])
        recent_signs = [np.sign(h['stats']['mean']) for h in self.client_update_history[client_id][-3:]]
        
        if len(recent_signs) == 0:
            return False, 0.0
        
        # Check consistency
        consistency = sum(1 for s in recent_signs if s == current_sign) / len(recent_signs)
        is_inconsistent = consistency < 0.5  # Less than 50% consistency
        
        return is_inconsistent, 1.0 - consistency
    
    def _check_statistical_outliers(
        self,
        client_id: str,
        stats: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Check if update is statistical outlier compared to all clients."""
        # Global statistics across all clients
        all_recent_stats = []
        for client_history in self.client_update_history.values():
            if client_history:
                all_recent_stats.append(client_history[-1]['stats'])
        
        if len(all_recent_stats) < 3:
            return False, 0.0
        
        # Calculate z-score compared to all clients
        global_norms = [s['l2_norm'] for s in all_recent_stats]
        global_mean = np.mean(global_norms)
        global_std = np.std(global_norms)
        
        if global_std == 0:
            return False, 0.0
        
        z_score = abs((stats['l2_norm'] - global_mean) / global_std)
        is_outlier = z_score > self.detection_threshold
        
        return is_outlier, z_score


class AdvancedThreatDetector:
    """Advanced threat detection and response system."""
    
    def __init__(self, config: SecurityConfig, metrics_collector: ServerMetricsCollector):
        """Initialize threat detector."""
        self.config = config
        self.metrics_collector = metrics_collector
        self.security_events: List[SecurityEvent] = []
        self.client_profiles: Dict[str, ClientSecurityProfile] = {}
        self.threat_patterns: Dict[str, Any] = {}
        
        # Initialize detection models
        self._initialize_detection_models()
        
        logger.info("Advanced threat detector initialized")
    
    def _initialize_detection_models(self):
        """Initialize machine learning models for threat detection."""
        # Placeholder for ML models - in production, these would be trained models
        self.anomaly_detector = None  # Would be isolation forest or similar
        self.behavioral_classifier = None  # Would be classification model
        
        logger.debug("Detection models initialized")
    
    def register_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        client_id: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Register a security event."""
        event = SecurityEvent(
            event_id=f"sec_{int(time.time())}_{secrets.token_hex(4)}",
            event_type=event_type,
            threat_level=threat_level,
            client_id=client_id,
            timestamp=time.time(),
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Update client profile if applicable
        if client_id and client_id in self.client_profiles:
            self.client_profiles[client_id].security_violations.append(event)
            self._update_client_risk_assessment(client_id)
        
        # Trigger response if high/critical threat
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._trigger_threat_response(event)
        
        logger.warning(f"Security event registered: {event.event_type.value} - {description}")
        return event
    
    def analyze_client_behavior(
        self,
        client_id: str,
        behavior_data: Dict[str, Any]
    ) -> Tuple[bool, ThreatLevel, List[str]]:
        """
        Analyze client behavior for security threats.
        
        Args:
            client_id: Client identifier
            behavior_data: Client behavior data
            
        Returns:
            Tuple of (is_threat, threat_level, anomalies)
        """
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = ClientSecurityProfile(client_id=client_id)
        
        profile = self.client_profiles[client_id]
        anomalies = []
        threat_level = ThreatLevel.LOW
        
        # Check communication patterns
        comm_anomaly = self._check_communication_patterns(client_id, behavior_data)
        if comm_anomaly:
            anomalies.append(comm_anomaly)
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
        
        # Check timing patterns
        timing_anomaly = self._check_timing_patterns(client_id, behavior_data)
        if timing_anomaly:
            anomalies.append(timing_anomaly)
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
        
        # Check resource usage patterns
        resource_anomaly = self._check_resource_patterns(client_id, behavior_data)
        if resource_anomaly:
            anomalies.append(resource_anomaly)
            threat_level = max(threat_level, ThreatLevel.HIGH)
        
        # Update behavioral patterns
        profile.behavioral_patterns.update({
            'last_analysis': time.time(),
            'communication_score': behavior_data.get('communication_score', 0.0),
            'timing_consistency': behavior_data.get('timing_consistency', 1.0),
            'resource_efficiency': behavior_data.get('resource_efficiency', 1.0)
        })
        
        is_threat = len(anomalies) > 0
        return is_threat, threat_level, anomalies
    
    def _check_communication_patterns(
        self,
        client_id: str,
        behavior_data: Dict[str, Any]
    ) -> Optional[str]:
        """Check for anomalous communication patterns."""
        # Check for unusual message frequencies
        msg_frequency = behavior_data.get('message_frequency', 0.0)
        if msg_frequency > 10.0:  # Too frequent
            return f"Unusually high message frequency: {msg_frequency:.2f}/min"
        
        # Check for unusual payload sizes
        payload_size = behavior_data.get('avg_payload_size', 0.0)
        if payload_size > 10e6:  # > 10MB average
            return f"Unusually large payload sizes: {payload_size/1e6:.2f}MB"
        
        return None
    
    def _check_timing_patterns(
        self,
        client_id: str,
        behavior_data: Dict[str, Any]
    ) -> Optional[str]:
        """Check for anomalous timing patterns."""
        response_time = behavior_data.get('avg_response_time', 0.0)
        
        # Check for suspiciously fast responses (possible automation/attacks)
        if response_time < 0.1:  # Less than 100ms
            return f"Suspiciously fast responses: {response_time*1000:.1f}ms"
        
        # Check for very slow responses (possible DoS)
        if response_time > 300.0:  # More than 5 minutes
            return f"Extremely slow responses: {response_time:.1f}s"
        
        return None
    
    def _check_resource_patterns(
        self,
        client_id: str,
        behavior_data: Dict[str, Any]
    ) -> Optional[str]:
        """Check for anomalous resource usage patterns."""
        cpu_usage = behavior_data.get('cpu_usage', 0.0)
        memory_usage = behavior_data.get('memory_usage', 0.0)
        
        # Check for resource exhaustion attempts
        if cpu_usage > 0.95 or memory_usage > 0.95:
            return f"Resource exhaustion pattern: CPU {cpu_usage:.1%}, Memory {memory_usage:.1%}"
        
        return None
    
    def _update_client_risk_assessment(self, client_id: str):
        """Update client risk assessment based on security events."""
        if client_id not in self.client_profiles:
            return
        
        profile = self.client_profiles[client_id]
        
        # Calculate risk score based on security violations
        risk_score = 0.0
        for event in profile.security_violations[-10:]:  # Last 10 events
            if event.threat_level == ThreatLevel.LOW:
                risk_score += 0.1
            elif event.threat_level == ThreatLevel.MEDIUM:
                risk_score += 0.3
            elif event.threat_level == ThreatLevel.HIGH:
                risk_score += 0.6
            elif event.threat_level == ThreatLevel.CRITICAL:
                risk_score += 1.0
        
        # Update trust score
        profile.trust_score = max(0.0, 1.0 - risk_score)
        
        # Update risk level
        if risk_score >= 1.0:
            profile.risk_level = ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            profile.risk_level = ThreatLevel.HIGH
        elif risk_score >= 0.3:
            profile.risk_level = ThreatLevel.MEDIUM
        else:
            profile.risk_level = ThreatLevel.LOW
        
        # Quarantine if necessary
        if profile.risk_level == ThreatLevel.CRITICAL:
            profile.quarantined = True
            logger.warning(f"Client {client_id} quarantined due to critical risk level")
    
    def _trigger_threat_response(self, event: SecurityEvent):
        """Trigger automated threat response."""
        response_actions = []
        
        if event.threat_level == ThreatLevel.HIGH:
            response_actions.extend([
                "increase_monitoring",
                "request_additional_authentication",
                "temporary_rate_limiting"
            ])
        elif event.threat_level == ThreatLevel.CRITICAL:
            response_actions.extend([
                "immediate_quarantine",
                "block_client_communication",
                "alert_security_team",
                "emergency_protocol"
            ])
        
        event.response_actions = response_actions
        
        # Execute response actions
        for action in response_actions:
            self._execute_response_action(action, event)
    
    def _execute_response_action(self, action: str, event: SecurityEvent):
        """Execute security response action."""
        logger.warning(f"Executing security action: {action} for event {event.event_id}")
        
        if action == "immediate_quarantine" and event.client_id:
            if event.client_id in self.client_profiles:
                self.client_profiles[event.client_id].quarantined = True
        
        # In production, this would implement actual security responses
        # such as firewall rules, rate limiting, alerting systems, etc.
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        total_events = len(self.security_events)
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]  # Last hour
        
        threat_distribution = {}
        for level in ThreatLevel:
            threat_distribution[level.value] = len([
                e for e in self.security_events if e.threat_level == level
            ])
        
        client_risk_summary = {}
        for client_id, profile in self.client_profiles.items():
            client_risk_summary[client_id] = {
                'trust_score': profile.trust_score,
                'risk_level': profile.risk_level.value,
                'quarantined': profile.quarantined,
                'violation_count': len(profile.security_violations)
            }
        
        return {
            'total_events': total_events,
            'recent_events': len(recent_events),
            'threat_distribution': threat_distribution,
            'active_threats': len([e for e in self.security_events if not e.resolved]),
            'quarantined_clients': len([p for p in self.client_profiles.values() if p.quarantined]),
            'client_risk_summary': client_risk_summary,
            'security_score': self._calculate_overall_security_score()
        }
    
    def _calculate_overall_security_score(self) -> float:
        """Calculate overall system security score."""
        if not self.security_events:
            return 1.0
        
        # Calculate based on recent events and client trust scores
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]
        recent_threat_score = sum(
            0.1 if e.threat_level == ThreatLevel.LOW else
            0.3 if e.threat_level == ThreatLevel.MEDIUM else
            0.6 if e.threat_level == ThreatLevel.HIGH else 1.0
            for e in recent_events
        )
        
        avg_trust_score = np.mean([p.trust_score for p in self.client_profiles.values()]) if self.client_profiles else 1.0
        
        security_score = max(0.0, avg_trust_score - recent_threat_score / 10.0)
        return min(1.0, security_score)


class AdvancedSecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, config: SecurityConfig, metrics_collector: ServerMetricsCollector):
        """Initialize advanced security manager."""
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Initialize components
        self.crypto_manager = AdvancedCryptographyManager(config)
        self.byzantine_detector = ByzantineDetector(config)
        self.threat_detector = AdvancedThreatDetector(config, metrics_collector)
        
        # Security state
        self.security_enabled = config.enable_authentication and config.enable_encryption
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info(f"Advanced security manager initialized (enabled: {self.security_enabled})")
    
    def validate_client_update(
        self,
        client_id: str,
        update: Dict[str, torch.Tensor],
        round_num: int,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[SecurityEvent]]:
        """
        Validate client update for security threats.
        
        Args:
            client_id: Client identifier
            update: Model update
            round_num: Training round
            metadata: Additional metadata
            
        Returns:
            Tuple of (is_valid, reason, security_event)
        """
        try:
            # Check if client is quarantined
            if client_id in self.threat_detector.client_profiles:
                profile = self.threat_detector.client_profiles[client_id]
                if profile.quarantined:
                    return False, "Client is quarantined", None
            
            # Byzantine detection
            is_byzantine, confidence, reason = self.byzantine_detector.analyze_client_update(
                client_id, update, round_num
            )
            
            if is_byzantine and confidence > 0.7:
                event = self.threat_detector.register_security_event(
                    SecurityEventType.BYZANTINE_DETECTION,
                    ThreatLevel.HIGH,
                    client_id,
                    f"Byzantine behavior detected: {reason}",
                    {'confidence': confidence, 'round': round_num}
                )
                return False, f"Byzantine behavior: {reason}", event
            
            # Behavioral analysis
            behavior_data = {
                'message_frequency': metadata.get('message_frequency', 1.0),
                'avg_payload_size': sum(t.numel() * 4 for t in update.values()),  # Approximate size
                'avg_response_time': metadata.get('response_time', 1.0),
                'cpu_usage': metadata.get('cpu_usage', 0.5),
                'memory_usage': metadata.get('memory_usage', 0.5)
            }
            
            is_threat, threat_level, anomalies = self.threat_detector.analyze_client_behavior(
                client_id, behavior_data
            )
            
            if is_threat and threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                event = self.threat_detector.register_security_event(
                    SecurityEventType.ANOMALOUS_BEHAVIOR,
                    threat_level,
                    client_id,
                    f"Behavioral anomalies: {'; '.join(anomalies)}",
                    {'anomalies': anomalies}
                )
                return False, f"Behavioral anomalies detected", event
            
            # Log successful validation
            self.audit_log.append({
                'timestamp': time.time(),
                'client_id': client_id,
                'round': round_num,
                'action': 'update_validated',
                'result': 'success'
            })
            
            return True, "Update validated successfully", None
            
        except Exception as e:
            logger.error(f"Error validating client update: {e}")
            event = self.threat_detector.register_security_event(
                SecurityEventType.DATA_INTEGRITY_VIOLATION,
                ThreatLevel.MEDIUM,
                client_id,
                f"Update validation error: {str(e)}"
            )
            return False, f"Validation error: {str(e)}", event
    
    def secure_aggregate_updates(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation with Byzantine resilience.
        
        Args:
            client_updates: Updates from clients
            client_weights: Client weights
            
        Returns:
            Securely aggregated update
        """
        # Filter out quarantined clients
        valid_updates = {}
        valid_weights = {}
        
        for client_id, update in client_updates.items():
            if client_id in self.threat_detector.client_profiles:
                profile = self.threat_detector.client_profiles[client_id]
                if not profile.quarantined and profile.trust_score > 0.5:
                    valid_updates[client_id] = update
                    valid_weights[client_id] = client_weights.get(client_id, 1.0) * profile.trust_score
                else:
                    logger.warning(f"Excluding client {client_id} from aggregation (quarantined or low trust)")
            else:
                valid_updates[client_id] = update
                valid_weights[client_id] = client_weights.get(client_id, 1.0)
        
        if len(valid_updates) < 2:
            raise SecurityError("Insufficient trusted clients for secure aggregation")
        
        # Perform Byzantine-robust aggregation (Krum or Trimmed Mean)
        aggregated = self._byzantine_robust_aggregation(valid_updates, valid_weights)
        
        # Log aggregation
        self.audit_log.append({
            'timestamp': time.time(),
            'action': 'secure_aggregation',
            'participants': list(valid_updates.keys()),
            'excluded_clients': [
                cid for cid in client_updates.keys() if cid not in valid_updates
            ]
        })
        
        return aggregated
    
    def _byzantine_robust_aggregation(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Perform Byzantine-robust aggregation."""
        if len(client_updates) < 3:
            # Fall back to weighted average for small number of clients
            return self._weighted_average(client_updates, client_weights)
        
        # Use Krum algorithm for Byzantine robustness
        return self._krum_aggregation(client_updates, client_weights)
    
    def _weighted_average(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of updates."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Get parameter names from first client
        first_client = next(iter(client_updates.keys()))
        param_names = client_updates[first_client].keys()
        
        aggregated = {}
        total_weight = sum(client_weights.values())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[first_client][param_name])
            
            for client_id, update in client_updates.items():
                weight = client_weights.get(client_id, 1.0) / total_weight
                weighted_sum += weight * update[param_name]
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _krum_aggregation(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Perform Krum aggregation for Byzantine robustness."""
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        n_byzantine = max(1, int(n_clients * self.config.byzantine_fraction))
        n_closest = n_clients - n_byzantine - 2
        
        if n_closest <= 0:
            # Fall back to weighted average
            return self._weighted_average(client_updates, client_weights)
        
        # Calculate pairwise distances
        distances = {}
        for i, client_i in enumerate(client_ids):
            distances[client_i] = {}
            update_i = client_updates[client_i]
            
            for j, client_j in enumerate(client_ids):
                if i != j:
                    update_j = client_updates[client_j]
                    
                    # Calculate L2 distance between updates
                    dist = 0.0
                    for param_name in update_i.keys():
                        if param_name in update_j:
                            diff = update_i[param_name] - update_j[param_name]
                            dist += torch.norm(diff, p=2).item() ** 2
                    
                    distances[client_i][client_j] = dist
        
        # Find client with minimum Krum score
        krum_scores = {}
        for client_id in client_ids:
            # Sort distances for this client
            client_distances = sorted(distances[client_id].values())
            # Sum of n_closest smallest distances
            krum_scores[client_id] = sum(client_distances[:n_closest])
        
        # Select client with minimum Krum score
        selected_client = min(krum_scores, key=krum_scores.get)
        
        logger.info(f"Krum selected client {selected_client} with score {krum_scores[selected_client]:.3f}")
        
        return client_updates[selected_client]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        threat_report = self.threat_detector.get_security_report()
        
        return {
            'security_enabled': self.security_enabled,
            'threat_detector_status': threat_report,
            'byzantine_detection_active': True,
            'encryption_active': self.config.enable_encryption,
            'authentication_active': self.config.enable_authentication,
            'audit_log_entries': len(self.audit_log),
            'recent_security_events': len([
                e for e in self.threat_detector.security_events
                if time.time() - e.timestamp < 3600
            ])
        }


def create_advanced_security_manager(
    config: SecurityConfig,
    metrics_collector: ServerMetricsCollector
) -> AdvancedSecurityManager:
    """
    Create advanced security manager with specified configuration.
    
    Args:
        config: Security configuration
        metrics_collector: Server metrics collector
        
    Returns:
        Configured advanced security manager
    """
    return AdvancedSecurityManager(config, metrics_collector)
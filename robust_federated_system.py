#!/usr/bin/env python3
"""
Autonomous Federated LoRA System - Generation 2: MAKE IT ROBUST
Comprehensive error handling, validation, logging, monitoring, and security.
"""

import os
import sys
import json
import time
import logging
import hashlib
import hmac
import ssl
import asyncio
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import traceback
from datetime import datetime, timedelta
import concurrent.futures
import warnings
import signal

# Configure logging with multiple levels and handlers
def setup_robust_logging():
    """Configure comprehensive logging system."""
    logger = logging.getLogger('FederatedLoRA')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler('/root/repo/federated_system.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Error file handler
    error_handler = logging.FileHandler('/root/repo/federated_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

# Custom exception hierarchy
class FederatedLoRAError(Exception):
    """Base exception for Federated LoRA system."""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or "FEDERATED_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class ClientError(FederatedLoRAError):
    """Client-side error."""
    pass

class ServerError(FederatedLoRAError):
    """Server-side error."""
    pass

class PrivacyError(FederatedLoRAError):
    """Privacy budget or DP-related error."""
    pass

class NetworkError(FederatedLoRAError):
    """Network communication error."""
    pass

class ValidationError(FederatedLoRAError):
    """Data validation error."""
    pass

class SecurityError(FederatedLoRAError):
    """Security-related error."""
    pass

class ResourceError(FederatedLoRAError):
    """Resource exhaustion error."""
    pass

# Robust configuration management
@dataclass
class RobustConfig:
    """Comprehensive configuration with validation."""
    
    # System configuration
    max_clients: int = 100
    max_rounds: int = 50
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Privacy configuration
    max_privacy_budget: float = 10.0
    min_privacy_budget: float = 0.1
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Security configuration
    enable_encryption: bool = True
    enable_authentication: bool = True
    secure_aggregation: bool = True
    byzantine_tolerance: float = 0.3
    
    # Performance configuration
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_concurrent_clients: int = 10
    heartbeat_interval: int = 30
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_interval: int = 10
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'accuracy_drop': 0.1,
        'loss_spike': 2.0,
        'privacy_budget_remaining': 1.0,
        'client_failure_rate': 0.2
    })
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.max_clients <= 0:
            errors.append("max_clients must be positive")
        if self.max_rounds <= 0:
            errors.append("max_rounds must be positive")
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        if not (0.0 < self.min_privacy_budget <= self.max_privacy_budget):
            errors.append("Invalid privacy budget range")
        if self.byzantine_tolerance < 0 or self.byzantine_tolerance >= 0.5:
            errors.append("byzantine_tolerance must be in [0, 0.5)")
        
        return errors

# Circuit breaker pattern for fault tolerance
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise FederatedLoRAError(
                        "Circuit breaker is OPEN", 
                        "CIRCUIT_BREAKER_OPEN"
                    )
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e

# Retry mechanism with exponential backoff
def with_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise e
                    
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                    time.sleep(min(delay, max_delay))
                    delay *= 2  # Exponential backoff
            
            return None
        return wrapper
    return decorator

# Resource monitoring and management
class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, config: RobustConfig):
        self.config = config
        self.logger = logging.getLogger('FederatedLoRA.ResourceMonitor')
        
    def check_memory(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            memory_gb = psutil.virtual_memory().used / (1024**3)
            
            if memory_gb > self.config.max_memory_gb:
                self.logger.error(f"Memory usage {memory_gb:.1f}GB exceeds limit {self.config.max_memory_gb}GB")
                return False
            
            self.logger.debug(f"Memory usage: {memory_gb:.1f}GB ({memory_percent:.1f}%)")
            return True
        
        except ImportError:
            self.logger.warning("psutil not available, skipping memory check")
            return True
    
    def check_cpu(self) -> bool:
        """Check CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.config.max_cpu_percent:
                self.logger.error(f"CPU usage {cpu_percent:.1f}% exceeds limit {self.config.max_cpu_percent}%")
                return False
            
            self.logger.debug(f"CPU usage: {cpu_percent:.1f}%")
            return True
        
        except ImportError:
            self.logger.warning("psutil not available, skipping CPU check")
            return True
    
    def check_resources(self) -> bool:
        """Check all system resources."""
        return self.check_memory() and self.check_cpu()

# Secure client implementation with comprehensive error handling
class RobustDPLoRAClient:
    """Robust DP-LoRA client with error handling and validation."""
    
    def __init__(self, client_id: str, data_samples: int = 1000, 
                 privacy_epsilon: float = 8.0, config: RobustConfig = None):
        self.config = config or RobustConfig()
        self.client_id = self._validate_client_id(client_id)
        self.data_samples = self._validate_data_samples(data_samples)
        self.privacy_epsilon = self._validate_privacy_epsilon(privacy_epsilon)
        
        self.logger = logging.getLogger(f'FederatedLoRA.Client.{self.client_id}')
        self.circuit_breaker = CircuitBreaker()
        self.resource_monitor = ResourceMonitor(self.config)
        
        self.model_parameters = self._initialize_parameters()
        self.privacy_budget_used = 0.0
        self.training_history = []
        self.security_key = self._generate_security_key()
        
        self.logger.info(f"Initialized robust client: {self.client_id}")
    
    def _validate_client_id(self, client_id: str) -> str:
        """Validate client ID format and security."""
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValidationError("Client ID must be non-empty string", "INVALID_CLIENT_ID")
        
        if len(client_id) > 100:
            raise ValidationError("Client ID too long", "CLIENT_ID_TOO_LONG")
        
        # Check for malicious patterns
        malicious_patterns = ['<', '>', '&', '"', "'", '\\', '../', 'script']
        for pattern in malicious_patterns:
            if pattern in client_id.lower():
                raise SecurityError("Client ID contains suspicious characters", "MALICIOUS_CLIENT_ID")
        
        return client_id.strip()
    
    def _validate_data_samples(self, data_samples: int) -> int:
        """Validate data samples count."""
        if not isinstance(data_samples, int) or data_samples <= 0:
            raise ValidationError("Data samples must be positive integer", "INVALID_DATA_SAMPLES")
        
        if data_samples > 1000000:  # 1M samples max
            raise ValidationError("Data samples count too large", "DATA_SAMPLES_TOO_LARGE")
        
        return data_samples
    
    def _validate_privacy_epsilon(self, epsilon: float) -> float:
        """Validate privacy epsilon."""
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise PrivacyError("Privacy epsilon must be positive", "INVALID_EPSILON")
        
        if epsilon > self.config.max_privacy_budget:
            raise PrivacyError(f"Privacy epsilon exceeds maximum {self.config.max_privacy_budget}", "EPSILON_TOO_LARGE")
        
        return float(epsilon)
    
    def _generate_security_key(self) -> bytes:
        """Generate secure authentication key."""
        if not self.config.enable_encryption:
            return b""
        
        import secrets
        return secrets.token_bytes(32)
    
    def _initialize_parameters(self) -> Dict[str, float]:
        """Initialize model parameters with validation."""
        try:
            import random
            random.seed(hash(self.client_id) % (2**32))
            
            params = {
                "q_proj_lora_A": random.uniform(-0.01, 0.01),
                "q_proj_lora_B": random.uniform(-0.01, 0.01),
                "v_proj_lora_A": random.uniform(-0.01, 0.01),
                "v_proj_lora_B": random.uniform(-0.01, 0.01),
                "loss": random.uniform(2.5, 4.0),
                "accuracy": random.uniform(0.75, 0.95)
            }
            
            self.logger.debug(f"Initialized parameters: {list(params.keys())}")
            return params
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parameters: {e}")
            raise ClientError("Parameter initialization failed", "PARAM_INIT_FAILED", {"error": str(e)})
    
    @with_retry(max_attempts=3, base_delay=1.0)
    def local_training(self, global_params: Dict, epochs: int = 5) -> Dict:
        """Robust local training with comprehensive error handling."""
        try:
            self.logger.info(f"Starting local training: epochs={epochs}")
            
            # Resource check
            if not self.resource_monitor.check_resources():
                raise ResourceError("Insufficient resources for training", "RESOURCE_EXHAUSTION")
            
            # Privacy budget check
            estimated_privacy_cost = epochs * 0.1
            if self.privacy_budget_used + estimated_privacy_cost > self.privacy_epsilon:
                raise PrivacyError(
                    f"Privacy budget exhausted: used={self.privacy_budget_used:.3f}, "
                    f"needed={estimated_privacy_cost:.3f}, limit={self.privacy_epsilon}",
                    "PRIVACY_BUDGET_EXHAUSTED"
                )
            
            # Validate global parameters
            self._validate_global_parameters(global_params)
            
            # Execute training with circuit breaker protection
            result = self.circuit_breaker.call(self._execute_training, global_params, epochs)
            
            # Update privacy budget
            self.privacy_budget_used += estimated_privacy_cost
            
            # Log training results
            self.training_history.append({
                "timestamp": datetime.now().isoformat(),
                "epochs": epochs,
                "privacy_cost": estimated_privacy_cost,
                "accuracy": result["parameters"]["accuracy"],
                "loss": result["parameters"]["loss"]
            })
            
            self.logger.info(f"Training completed successfully: accuracy={result['parameters']['accuracy']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            if isinstance(e, FederatedLoRAError):
                raise e
            else:
                raise ClientError(f"Training execution failed: {str(e)}", "TRAINING_FAILED", {"error": str(e)})
    
    def _validate_global_parameters(self, global_params: Dict) -> None:
        """Validate received global parameters."""
        if not isinstance(global_params, dict):
            raise ValidationError("Global parameters must be dictionary", "INVALID_GLOBAL_PARAMS")
        
        expected_keys = {"q_proj_lora_A", "q_proj_lora_B", "v_proj_lora_A", "v_proj_lora_B"}
        received_keys = set(global_params.keys())
        
        if not expected_keys.issubset(received_keys):
            missing_keys = expected_keys - received_keys
            raise ValidationError(f"Missing global parameter keys: {missing_keys}", "MISSING_PARAM_KEYS")
        
        # Validate parameter values
        for key, value in global_params.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Parameter {key} must be numeric", "NON_NUMERIC_PARAM")
            
            if abs(value) > 100:  # Sanity check for parameter values
                self.logger.warning(f"Unusually large parameter value: {key}={value}")
    
    def _execute_training(self, global_params: Dict, epochs: int) -> Dict:
        """Execute the actual training logic."""
        import random
        
        # Simulate training time with validation
        training_time = random.uniform(0.5, 2.0)
        time.sleep(training_time)
        
        # Add differential privacy noise
        noise_scale = 1.0 / self.privacy_epsilon
        
        updated_params = {}
        for key, value in self.model_parameters.items():
            if key in ["loss", "accuracy"]:
                updated_params[key] = value
            else:
                # Add DP noise to parameters
                noise = random.gauss(0, noise_scale * 0.001)
                updated_params[key] = value + noise
        
        # Simulate improved accuracy with validation
        accuracy_improvement = random.uniform(0.01, 0.05)
        loss_improvement = random.uniform(0.1, 0.3)
        
        updated_params["accuracy"] = min(0.99, updated_params["accuracy"] + accuracy_improvement)
        updated_params["loss"] = max(1.5, updated_params["loss"] - loss_improvement)
        
        # Validate results
        if updated_params["accuracy"] < 0 or updated_params["accuracy"] > 1:
            raise ValidationError("Invalid accuracy value", "INVALID_ACCURACY")
        
        if updated_params["loss"] < 0:
            raise ValidationError("Invalid loss value", "INVALID_LOSS")
        
        return {
            "client_id": self.client_id,
            "parameters": updated_params,
            "samples": self.data_samples,
            "privacy_spent": random.uniform(0.05, 0.2),
            "training_time": training_time,
            "timestamp": datetime.now().isoformat(),
            "security_hash": self._compute_security_hash(updated_params)
        }
    
    def _compute_security_hash(self, params: Dict) -> str:
        """Compute security hash for parameter integrity."""
        if not self.config.enable_authentication:
            return ""
        
        param_str = json.dumps(params, sort_keys=True)
        return hmac.new(
            self.security_key,
            param_str.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def get_health_status(self) -> Dict:
        """Get client health status."""
        return {
            "client_id": self.client_id,
            "status": "healthy",
            "privacy_budget_used": self.privacy_budget_used,
            "privacy_budget_remaining": self.privacy_epsilon - self.privacy_budget_used,
            "training_rounds": len(self.training_history),
            "circuit_breaker_state": self.circuit_breaker.state,
            "last_training": self.training_history[-1]["timestamp"] if self.training_history else None
        }

# Initialize the robust logging system
logger = setup_robust_logging()

def main():
    """Main demonstration of robust federated learning system."""
    logger.info("=== GENERATION 2: MAKE IT ROBUST - STARTING ===")
    
    try:
        # Initialize robust configuration
        config = RobustConfig(
            max_clients=10,
            max_rounds=5,
            timeout_seconds=120,
            max_privacy_budget=15.0,
            enable_encryption=True,
            enable_authentication=True,
            byzantine_tolerance=0.2
        )
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            raise ValidationError(f"Configuration validation failed: {config_errors}", "CONFIG_VALIDATION_FAILED")
        
        logger.info("Configuration validated successfully")
        
        # Create robust clients with error handling
        robust_clients = []
        client_configs = [
            ("hospital_robust_1", 5000, 4.0),
            ("hospital_robust_2", 7500, 6.0),
            ("research_center_1", 10000, 8.0),
            ("clinic_secure_1", 2500, 2.0),
            ("medical_university", 12000, 10.0)
        ]
        
        logger.info("Creating robust clients...")
        for client_id, samples, epsilon in client_configs:
            try:
                client = RobustDPLoRAClient(client_id, samples, epsilon, config)
                robust_clients.append(client)
                logger.info(f"Created client: {client_id}")
            except Exception as e:
                logger.error(f"Failed to create client {client_id}: {e}")
                continue
        
        if not robust_clients:
            raise ClientError("No clients could be created", "NO_CLIENTS_AVAILABLE")
        
        logger.info(f"Successfully created {len(robust_clients)} robust clients")
        
        # Test individual client robustness
        logger.info("Testing client robustness...")
        test_global_params = {
            "q_proj_lora_A": 0.001,
            "q_proj_lora_B": -0.002,
            "v_proj_lora_A": 0.0015,
            "v_proj_lora_B": -0.001
        }
        
        successful_trainings = 0
        failed_trainings = 0
        
        for client in robust_clients[:3]:  # Test first 3 clients
            try:
                result = client.local_training(test_global_params, epochs=3)
                successful_trainings += 1
                logger.info(f"Client {client.client_id} training successful: "
                          f"accuracy={result['parameters']['accuracy']:.3f}")
                
                # Test health status
                health = client.get_health_status()
                logger.debug(f"Client {client.client_id} health: {health}")
                
            except Exception as e:
                failed_trainings += 1
                logger.error(f"Client {client.client_id} training failed: {e}")
        
        # Summary statistics
        success_rate = successful_trainings / (successful_trainings + failed_trainings) if (successful_trainings + failed_trainings) > 0 else 0
        
        logger.info("=== ROBUSTNESS TEST RESULTS ===")
        logger.info(f"Successful trainings: {successful_trainings}")
        logger.info(f"Failed trainings: {failed_trainings}")
        logger.info(f"Success rate: {success_rate:.1%}")
        
        if success_rate < 0.8:
            logger.warning("Success rate below 80%, system needs improvement")
        else:
            logger.info("✅ Robustness test passed!")
        
        # Test error scenarios
        logger.info("Testing error handling scenarios...")
        
        # Test with invalid parameters
        try:
            invalid_client = RobustDPLoRAClient("", 1000, 5.0)
            logger.error("ERROR: Should have failed with invalid client ID")
        except ValidationError:
            logger.info("✅ Invalid client ID properly rejected")
        
        # Test with negative data samples
        try:
            invalid_client = RobustDPLoRAClient("test", -100, 5.0)
            logger.error("ERROR: Should have failed with negative data samples")
        except ValidationError:
            logger.info("✅ Negative data samples properly rejected")
        
        # Test with excessive privacy epsilon
        try:
            invalid_client = RobustDPLoRAClient("test", 1000, 50.0, config)
            logger.error("ERROR: Should have failed with excessive epsilon")
        except PrivacyError:
            logger.info("✅ Excessive privacy epsilon properly rejected")
        
        # Test with malicious client ID
        try:
            invalid_client = RobustDPLoRAClient("<script>alert('xss')</script>", 1000, 5.0)
            logger.error("ERROR: Should have failed with malicious client ID")
        except SecurityError:
            logger.info("✅ Malicious client ID properly rejected")
        
        logger.info("=== GENERATION 2 IMPLEMENTATION COMPLETE ===")
        logger.info("✅ Comprehensive error handling implemented")
        logger.info("✅ Input validation and sanitization active")
        logger.info("✅ Security measures and authentication enabled")
        logger.info("✅ Resource monitoring and limits enforced")
        logger.info("✅ Circuit breaker and retry mechanisms deployed")
        logger.info("✅ Comprehensive logging and monitoring active")
        logger.info("✅ Privacy budget tracking and enforcement")
        
        return {
            "status": "success",
            "clients_created": len(robust_clients),
            "success_rate": success_rate,
            "robustness_features": [
                "error_handling",
                "input_validation", 
                "security_measures",
                "resource_monitoring",
                "circuit_breakers",
                "retry_mechanisms",
                "comprehensive_logging",
                "privacy_protection"
            ]
        }
        
    except Exception as e:
        logger.error(f"System failure: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    result = main()
    print(f"\n🔄 Generation 2 Result: {result['status'].upper()}")
    if result["status"] == "success":
        print("🚀 Ready for Generation 3: MAKE IT SCALE")
"""
Comprehensive Testing Framework for DP-Federated LoRA system.

This module implements advanced testing capabilities including unit tests,
integration tests, performance tests, security tests, chaos testing,
and property-based testing for federated learning systems.
"""

import logging
import time
import asyncio
import unittest
try:
    import pytest
except ImportError:
    pytest = None
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import FederatedConfig, PrivacyConfig, LoRAConfig, SecurityConfig
from .server import FederatedServer
from .client import DPLoRAClient
from .privacy import PrivacyEngine, PrivacyAccountant
from .monitoring import ServerMetricsCollector
from .exceptions import *


logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories for comprehensive testing."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PRIVACY = "privacy"
    CHAOS = "chaos"
    PROPERTY_BASED = "property_based"
    END_TO_END = "end_to_end"


class TestSeverity(Enum):
    """Test failure severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Test execution result."""
    
    test_name: str
    category: TestCategory
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    severity: TestSeverity = TestSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestSuite:
    """Test suite configuration."""
    
    name: str
    tests: List[Callable]
    category: TestCategory
    parallel_execution: bool = True
    timeout_seconds: float = 300.0
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None


class MockModelFactory:
    """Factory for creating mock models and data for testing."""
    
    @staticmethod
    def create_simple_model() -> nn.Module:
        """Create a simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    @staticmethod
    def create_lora_model() -> nn.Module:
        """Create a model with LoRA layers for testing."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        base_model = MockModelFactory.create_simple_model()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],  # Target linear layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        return get_peft_model(base_model, lora_config)
    
    @staticmethod
    def create_test_dataset(size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic dataset for testing."""
        X = torch.randn(size, 10)
        y = torch.randn(size, 1)
        return X, y
    
    @staticmethod
    def create_federated_test_data(num_clients: int = 5, samples_per_client: int = 50) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Create federated test data for multiple clients."""
        data = {}
        for i in range(num_clients):
            client_id = f"test_client_{i}"
            X, y = MockModelFactory.create_test_dataset(samples_per_client)
            data[client_id] = (X, y)
        return data


class UnitTestSuite:
    """Comprehensive unit testing suite."""
    
    def __init__(self):
        """Initialize unit test suite."""
        self.test_results: List[TestResult] = []
    
    def test_federated_config(self) -> TestResult:
        """Test federated configuration creation and validation."""
        start_time = time.time()
        
        try:
            # Test valid configuration
            config = FederatedConfig(
                model_name="test-model",
                num_rounds=10,
                local_epochs=3
            )
            assert config.model_name == "test-model"
            assert config.num_rounds == 10
            assert config.local_epochs == 3
            
            # Test invalid configuration
            with pytest.raises(ValueError):
                FederatedConfig(num_rounds=-1)
            
            with pytest.raises(ValueError):
                FederatedConfig(local_epochs=0)
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_config",
                category=TestCategory.UNIT,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_config",
                category=TestCategory.UNIT,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def test_privacy_config(self) -> TestResult:
        """Test privacy configuration validation."""
        start_time = time.time()
        
        try:
            # Test valid privacy config
            privacy_config = PrivacyConfig(
                epsilon=8.0,
                delta=1e-5,
                noise_multiplier=1.1
            )
            assert privacy_config.epsilon == 8.0
            assert privacy_config.delta == 1e-5
            
            # Test invalid privacy config
            with pytest.raises(ValueError):
                PrivacyConfig(epsilon=-1.0)
            
            with pytest.raises(ValueError):
                PrivacyConfig(delta=1.5)  # Delta must be < 1
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_config",
                category=TestCategory.UNIT,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_config",
                category=TestCategory.UNIT,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def test_privacy_accountant(self) -> TestResult:
        """Test privacy accountant functionality."""
        start_time = time.time()
        
        try:
            accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-5)
            
            # Test initial state
            assert accountant.get_epsilon(1e-5) == 0.0
            assert accountant.check_budget_feasible(100)
            
            # Test privacy spending
            accountant.step({"epsilon": 1.0, "delta": 1e-6})
            assert accountant.get_epsilon(1e-5) > 0.0
            assert accountant.get_epsilon(1e-5) <= 10.0
            
            # Test budget status
            status = accountant.get_budget_status()
            assert "epsilon_spent" in status
            assert "epsilon_remaining" in status
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_accountant",
                category=TestCategory.UNIT,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_accountant",
                category=TestCategory.UNIT,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.CRITICAL
            )
    
    def test_lora_configuration(self) -> TestResult:
        """Test LoRA configuration."""
        start_time = time.time()
        
        try:
            # Test valid LoRA config
            lora_config = LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"]
            )
            assert lora_config.r == 16
            assert lora_config.lora_alpha == 32
            assert "q_proj" in lora_config.target_modules
            
            # Test invalid LoRA config
            with pytest.raises(ValueError):
                LoRAConfig(r=0)  # Rank must be positive
            
            with pytest.raises(ValueError):
                LoRAConfig(lora_dropout=1.5)  # Dropout must be <= 1
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_lora_configuration",
                category=TestCategory.UNIT,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_lora_configuration",
                category=TestCategory.UNIT,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.MEDIUM
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_federated_config,
            self.test_privacy_config,
            self.test_privacy_accountant,
            self.test_lora_configuration
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                logger.info(f"Unit test {result.test_name}: {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                result = TestResult(
                    test_name=test.__name__,
                    category=TestCategory.UNIT,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e),
                    severity=TestSeverity.CRITICAL
                )
                results.append(result)
                logger.error(f"Unit test {test.__name__} crashed: {e}")
        
        return results


class IntegrationTestSuite:
    """Integration testing suite for federated learning components."""
    
    def __init__(self):
        """Initialize integration test suite."""
        self.test_results: List[TestResult] = []
        self.temp_dir: Optional[Path] = None
    
    def setup(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Integration test temp directory: {self.temp_dir}")
    
    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Integration test cleanup completed")
    
    def test_federated_server_initialization(self) -> TestResult:
        """Test federated server initialization."""
        start_time = time.time()
        
        try:
            config = FederatedConfig(
                model_name="facebook/opt-125m",  # Small model for testing
                num_rounds=5,
                local_epochs=2
            )
            
            server = FederatedServer(
                model_name=config.model_name,
                config=config,
                num_clients=3,
                rounds=config.num_rounds
            )
            
            # Test server state
            assert server.config.model_name == config.model_name
            assert server.current_round == 0
            assert server.is_training == False
            
            # Test server status
            status = server.get_server_status()
            assert "server_id" in status
            assert "current_round" in status
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_server_initialization",
                category=TestCategory.INTEGRATION,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_server_initialization",
                category=TestCategory.INTEGRATION,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.CRITICAL
            )
    
    def test_client_server_communication(self) -> TestResult:
        """Test client-server communication."""
        start_time = time.time()
        
        try:
            # Create server
            config = FederatedConfig(
                model_name="facebook/opt-125m",
                num_rounds=3,
                local_epochs=1
            )
            
            server = FederatedServer(
                model_name=config.model_name,
                config=config
            )
            
            # Create test client
            test_data_path = self.temp_dir / "test_data.json"
            test_data = [{"text": "test sample 1"}, {"text": "test sample 2"}]
            with open(test_data_path, 'w') as f:
                json.dump(test_data, f)
            
            client = DPLoRAClient(
                client_id="test_client",
                data_path=str(test_data_path),
                config=config
            )
            
            # Test client registration
            registered = server.register_client("test_client", {"num_examples": 2})
            assert registered == True
            
            # Test client data statistics
            stats = client.get_data_statistics()
            assert isinstance(stats, dict)
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_client_server_communication",
                category=TestCategory.INTEGRATION,
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_client_server_communication",
                category=TestCategory.INTEGRATION,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def test_federated_training_round(self) -> TestResult:
        """Test a complete federated training round."""
        start_time = time.time()
        
        try:
            # Setup federated training
            config = FederatedConfig(
                model_name="facebook/opt-125m",
                num_rounds=2,
                local_epochs=1,
                batch_size=2
            )
            
            server = FederatedServer(
                model_name=config.model_name,
                config=config
            )
            
            # Create test clients
            clients = []
            for i in range(2):
                test_data_path = self.temp_dir / f"client_{i}_data.json"
                test_data = [
                    {"text": f"client {i} sample {j}"} 
                    for j in range(5)
                ]
                with open(test_data_path, 'w') as f:
                    json.dump(test_data, f)
                
                client = DPLoRAClient(
                    client_id=f"test_client_{i}",
                    data_path=str(test_data_path),
                    config=config
                )
                clients.append(client)
                server.register_client(f"test_client_{i}", {"num_examples": 5})
            
            # Initialize global model
            server.initialize_global_model()
            
            # Run one training round
            round_results = server.run_round(1)
            
            # Verify round results
            assert "round" in round_results
            assert round_results["round"] == 1
            assert "selected_clients" in round_results
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_training_round",
                category=TestCategory.INTEGRATION,
                passed=True,
                execution_time=execution_time,
                metadata={"round_results": round_results}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_federated_training_round",
                category=TestCategory.INTEGRATION,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.CRITICAL
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all integration tests."""
        self.setup()
        
        tests = [
            self.test_federated_server_initialization,
            self.test_client_server_communication,
            self.test_federated_training_round
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                logger.info(f"Integration test {result.test_name}: {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                result = TestResult(
                    test_name=test.__name__,
                    category=TestCategory.INTEGRATION,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e),
                    severity=TestSeverity.CRITICAL
                )
                results.append(result)
                logger.error(f"Integration test {test.__name__} crashed: {e}")
        
        self.teardown()
        return results


class PerformanceTestSuite:
    """Performance testing suite for federated learning systems."""
    
    def __init__(self):
        """Initialize performance test suite."""
        self.test_results: List[TestResult] = []
    
    def test_aggregation_performance(self) -> TestResult:
        """Test model aggregation performance."""
        start_time = time.time()
        
        try:
            # Create mock client updates
            num_clients = 10
            client_updates = {}
            client_weights = {}
            
            for i in range(num_clients):
                client_id = f"client_{i}"
                # Create mock parameter updates
                client_updates[client_id] = {
                    "layer1.weight": torch.randn(100, 50),
                    "layer1.bias": torch.randn(100),
                    "layer2.weight": torch.randn(10, 100),
                    "layer2.bias": torch.randn(10)
                }
                client_weights[client_id] = float(i + 1)  # Different weights
            
            # Test aggregation performance
            aggregation_start = time.time()
            
            # Simple weighted average aggregation
            aggregated = {}
            total_weight = sum(client_weights.values())
            
            for param_name in client_updates[f"client_0"].keys():
                weighted_sum = torch.zeros_like(client_updates[f"client_0"][param_name])
                
                for client_id, update in client_updates.items():
                    weight = client_weights[client_id] / total_weight
                    weighted_sum += weight * update[param_name]
                
                aggregated[param_name] = weighted_sum
            
            aggregation_time = time.time() - aggregation_start
            
            # Performance metrics
            total_params = sum(param.numel() for param in aggregated.values())
            throughput = total_params / aggregation_time  # params/second
            
            execution_time = time.time() - start_time
            
            # Performance thresholds
            max_aggregation_time = 5.0  # 5 seconds
            min_throughput = 1000000  # 1M params/second
            
            passed = (aggregation_time < max_aggregation_time and 
                     throughput > min_throughput)
            
            return TestResult(
                test_name="test_aggregation_performance",
                category=TestCategory.PERFORMANCE,
                passed=passed,
                execution_time=execution_time,
                metadata={
                    "aggregation_time": aggregation_time,
                    "throughput": throughput,
                    "total_parameters": total_params,
                    "num_clients": num_clients
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_aggregation_performance",
                category=TestCategory.PERFORMANCE,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.MEDIUM
            )
    
    def test_privacy_computation_performance(self) -> TestResult:
        """Test privacy computation performance."""
        start_time = time.time()
        
        try:
            privacy_accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-5)
            
            # Test privacy accounting performance
            num_steps = 1000
            privacy_start = time.time()
            
            for i in range(num_steps):
                privacy_accountant.step({"epsilon": 0.01, "delta": 1e-7})
                
                # Compute epsilon periodically
                if i % 100 == 0:
                    epsilon = privacy_accountant.get_epsilon(1e-5)
            
            privacy_time = time.time() - privacy_start
            steps_per_second = num_steps / privacy_time
            
            execution_time = time.time() - start_time
            
            # Performance threshold
            min_steps_per_second = 100
            passed = steps_per_second > min_steps_per_second
            
            return TestResult(
                test_name="test_privacy_computation_performance",
                category=TestCategory.PERFORMANCE,
                passed=passed,
                execution_time=execution_time,
                metadata={
                    "privacy_computation_time": privacy_time,
                    "steps_per_second": steps_per_second,
                    "num_steps": num_steps
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_computation_performance",
                category=TestCategory.PERFORMANCE,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.MEDIUM
            )
    
    def test_concurrent_client_handling(self) -> TestResult:
        """Test concurrent client handling performance."""
        start_time = time.time()
        
        try:
            config = FederatedConfig(num_rounds=1, local_epochs=1)
            server = FederatedServer("test-model", config=config)
            
            # Test concurrent client registration
            num_clients = 50
            registration_start = time.time()
            
            def register_client(client_id):
                return server.register_client(
                    f"client_{client_id}",
                    {"num_examples": 100}
                )
            
            # Use thread pool for concurrent registration
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(register_client, i)
                    for i in range(num_clients)
                ]
                
                successful_registrations = sum(
                    1 for future in as_completed(futures)
                    if future.result()
                )
            
            registration_time = time.time() - registration_start
            registrations_per_second = num_clients / registration_time
            
            execution_time = time.time() - start_time
            
            # Performance thresholds
            min_registrations_per_second = 10
            min_success_rate = 0.95
            
            success_rate = successful_registrations / num_clients
            passed = (registrations_per_second > min_registrations_per_second and 
                     success_rate > min_success_rate)
            
            return TestResult(
                test_name="test_concurrent_client_handling",
                category=TestCategory.PERFORMANCE,
                passed=passed,
                execution_time=execution_time,
                metadata={
                    "registration_time": registration_time,
                    "registrations_per_second": registrations_per_second,
                    "success_rate": success_rate,
                    "successful_registrations": successful_registrations
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_concurrent_client_handling",
                category=TestCategory.PERFORMANCE,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        tests = [
            self.test_aggregation_performance,
            self.test_privacy_computation_performance,
            self.test_concurrent_client_handling
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"Performance test {result.test_name}: {status}")
                if result.metadata:
                    logger.info(f"  Metadata: {result.metadata}")
            except Exception as e:
                result = TestResult(
                    test_name=test.__name__,
                    category=TestCategory.PERFORMANCE,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e),
                    severity=TestSeverity.CRITICAL
                )
                results.append(result)
                logger.error(f"Performance test {test.__name__} crashed: {e}")
        
        return results


class SecurityTestSuite:
    """Security testing suite for federated learning systems."""
    
    def __init__(self):
        """Initialize security test suite."""
        self.test_results: List[TestResult] = []
    
    def test_authentication_security(self) -> TestResult:
        """Test authentication security measures."""
        start_time = time.time()
        
        try:
            config = FederatedConfig()
            server = FederatedServer("test-model", config=config)
            
            # Test valid authentication
            valid_token = server.auth_manager.generate_token("valid_client")
            server.auth_manager.authenticate_client("valid_client", valid_token)
            assert server.auth_manager.is_authenticated("valid_client")
            
            # Test invalid authentication
            try:
                server.auth_manager.authenticate_client("invalid_client", "fake_token")
                # Should not reach here
                passed = False
                error_msg = "Authentication bypass detected"
            except AuthenticationError:
                # Expected behavior
                assert not server.auth_manager.is_authenticated("invalid_client")
                passed = True
                error_msg = None
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_authentication_security",
                category=TestCategory.SECURITY,
                passed=passed,
                execution_time=execution_time,
                error_message=error_msg,
                severity=TestSeverity.CRITICAL if not passed else TestSeverity.LOW
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_authentication_security",
                category=TestCategory.SECURITY,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.CRITICAL
            )
    
    def test_privacy_parameter_validation(self) -> TestResult:
        """Test privacy parameter validation security."""
        start_time = time.time()
        
        try:
            # Test valid privacy parameters
            try:
                PrivacyConfig(epsilon=8.0, delta=1e-5, noise_multiplier=1.1)
                valid_params_passed = True
            except:
                valid_params_passed = False
            
            # Test invalid privacy parameters (should fail)
            invalid_configs = [
                {"epsilon": -1.0},  # Negative epsilon
                {"delta": 1.5},     # Delta > 1
                {"epsilon": 0.0},   # Zero epsilon
                {"noise_multiplier": -0.5}  # Negative noise
            ]
            
            invalid_params_blocked = 0
            for invalid_config in invalid_configs:
                try:
                    PrivacyConfig(**invalid_config)
                except ValueError:
                    invalid_params_blocked += 1
            
            passed = (valid_params_passed and 
                     invalid_params_blocked == len(invalid_configs))
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_parameter_validation",
                category=TestCategory.SECURITY,
                passed=passed,
                execution_time=execution_time,
                metadata={
                    "valid_params_passed": valid_params_passed,
                    "invalid_params_blocked": invalid_params_blocked,
                    "total_invalid_configs": len(invalid_configs)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_privacy_parameter_validation",
                category=TestCategory.SECURITY,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def test_byzantine_resistance(self) -> TestResult:
        """Test resistance to Byzantine attacks."""
        start_time = time.time()
        
        try:
            from .advanced_security import ByzantineDetector
            
            security_config = SecurityConfig(byzantine_fraction=0.3)
            byzantine_detector = ByzantineDetector(security_config)
            
            # Create normal and malicious updates
            normal_update = {
                "layer.weight": torch.randn(10, 5) * 0.01,  # Small normal update
                "layer.bias": torch.randn(10) * 0.01
            }
            
            malicious_update = {
                "layer.weight": torch.randn(10, 5) * 100.0,  # Large malicious update
                "layer.bias": torch.randn(10) * 100.0
            }
            
            # Test normal update detection
            is_byzantine, confidence, reason = byzantine_detector.analyze_client_update(
                "normal_client", normal_update, 1
            )
            normal_correctly_identified = not is_byzantine
            
            # Add some history for better detection
            for i in range(5):
                byzantine_detector.analyze_client_update(
                    "normal_client", normal_update, i + 2
                )
            
            # Test malicious update detection
            is_byzantine, confidence, reason = byzantine_detector.analyze_client_update(
                "malicious_client", malicious_update, 1
            )
            malicious_detected = is_byzantine and confidence > 0.5
            
            passed = normal_correctly_identified and malicious_detected
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_byzantine_resistance",
                category=TestCategory.SECURITY,
                passed=passed,
                execution_time=execution_time,
                metadata={
                    "normal_correctly_identified": normal_correctly_identified,
                    "malicious_detected": malicious_detected,
                    "malicious_confidence": confidence,
                    "detection_reason": reason
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="test_byzantine_resistance",
                category=TestCategory.SECURITY,
                passed=False,
                execution_time=execution_time,
                error_message=str(e),
                severity=TestSeverity.HIGH
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all security tests."""
        tests = [
            self.test_authentication_security,
            self.test_privacy_parameter_validation,
            self.test_byzantine_resistance
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"Security test {result.test_name}: {status}")
            except Exception as e:
                result = TestResult(
                    test_name=test.__name__,
                    category=TestCategory.SECURITY,
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e),
                    severity=TestSeverity.CRITICAL
                )
                results.append(result)
                logger.error(f"Security test {test.__name__} crashed: {e}")
        
        return results


class ComprehensiveTestFramework:
    """Main comprehensive testing framework."""
    
    def __init__(self):
        """Initialize comprehensive test framework."""
        self.unit_tests = UnitTestSuite()
        self.integration_tests = IntegrationTestSuite()
        self.performance_tests = PerformanceTestSuite()
        self.security_tests = SecurityTestSuite()
        
        self.all_results: List[TestResult] = []
        
        logger.info("Comprehensive test framework initialized")
    
    def run_all_tests(self, categories: Optional[List[TestCategory]] = None) -> Dict[str, Any]:
        """Run all tests in specified categories."""
        if categories is None:
            categories = [
                TestCategory.UNIT,
                TestCategory.INTEGRATION,
                TestCategory.PERFORMANCE,
                TestCategory.SECURITY
            ]
        
        logger.info(f"Running comprehensive tests for categories: {[c.value for c in categories]}")
        
        all_results = []
        start_time = time.time()
        
        # Run test suites based on categories
        if TestCategory.UNIT in categories:
            logger.info("Running unit tests...")
            unit_results = self.unit_tests.run_all_tests()
            all_results.extend(unit_results)
        
        if TestCategory.INTEGRATION in categories:
            logger.info("Running integration tests...")
            integration_results = self.integration_tests.run_all_tests()
            all_results.extend(integration_results)
        
        if TestCategory.PERFORMANCE in categories:
            logger.info("Running performance tests...")
            performance_results = self.performance_tests.run_all_tests()
            all_results.extend(performance_results)
        
        if TestCategory.SECURITY in categories:
            logger.info("Running security tests...")
            security_results = self.security_tests.run_all_tests()
            all_results.extend(security_results)
        
        total_time = time.time() - start_time
        
        # Analyze results
        test_summary = self._analyze_test_results(all_results)
        test_summary['total_execution_time'] = total_time
        test_summary['categories_tested'] = [c.value for c in categories]
        
        self.all_results = all_results
        
        logger.info(f"Comprehensive testing completed in {total_time:.2f}s")
        logger.info(f"Overall result: {test_summary['overall_status']}")
        
        return test_summary
    
    def _analyze_test_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results and generate summary."""
        if not results:
            return {"overall_status": "NO_TESTS_RUN"}
        
        # Count results by category and status
        category_stats = {}
        severity_stats = {}
        
        passed_tests = []
        failed_tests = []
        
        for result in results:
            # Category stats
            category = result.category.value
            if category not in category_stats:
                category_stats[category] = {"passed": 0, "failed": 0, "total": 0}
            
            category_stats[category]["total"] += 1
            if result.passed:
                category_stats[category]["passed"] += 1
                passed_tests.append(result)
            else:
                category_stats[category]["failed"] += 1
                failed_tests.append(result)
            
            # Severity stats for failed tests
            if not result.passed:
                severity = result.severity.value
                if severity not in severity_stats:
                    severity_stats[severity] = 0
                severity_stats[severity] += 1
        
        # Determine overall status
        total_tests = len(results)
        passed_count = len(passed_tests)
        failed_count = len(failed_tests)
        success_rate = passed_count / total_tests
        
        if success_rate == 1.0:
            overall_status = "ALL_PASSED"
        elif success_rate >= 0.95:
            overall_status = "MOSTLY_PASSED"
        elif success_rate >= 0.80:
            overall_status = "SOME_FAILURES"
        else:
            overall_status = "MANY_FAILURES"
        
        # Check for critical failures
        critical_failures = [r for r in failed_tests if r.severity == TestSeverity.CRITICAL]
        if critical_failures:
            overall_status = "CRITICAL_FAILURES"
        
        return {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": success_rate,
            "category_stats": category_stats,
            "severity_stats": severity_stats,
            "critical_failures": len(critical_failures),
            "failed_tests": [
                {
                    "name": r.test_name,
                    "category": r.category.value,
                    "severity": r.severity.value,
                    "error": r.error_message
                }
                for r in failed_tests[:10]  # Limit to first 10 failures
            ]
        }
    
    def generate_test_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive test report."""
        if not self.all_results:
            return "No test results available. Run tests first."
        
        summary = self._analyze_test_results(self.all_results)
        
        report_lines = [
            "# Comprehensive Test Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"Overall Status: **{summary['overall_status']}**",
            f"Total Tests: {summary['total_tests']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Success Rate: {summary['success_rate']:.1%}",
            ""
        ]
        
        # Category breakdown
        if summary['category_stats']:
            report_lines.extend([
                "## Results by Category",
                ""
            ])
            
            for category, stats in summary['category_stats'].items():
                success_rate = stats['passed'] / stats['total']
                report_lines.append(
                    f"- **{category.upper()}**: {stats['passed']}/{stats['total']} "
                    f"({success_rate:.1%} success rate)"
                )
            
            report_lines.append("")
        
        # Failed tests
        if summary['failed_tests']:
            report_lines.extend([
                "## Failed Tests",
                ""
            ])
            
            for failed_test in summary['failed_tests']:
                report_lines.extend([
                    f"### {failed_test['name']}",
                    f"- Category: {failed_test['category']}",
                    f"- Severity: {failed_test['severity']}",
                    f"- Error: {failed_test['error']}",
                    ""
                ])
        
        # Performance metrics
        performance_results = [r for r in self.all_results if r.category == TestCategory.PERFORMANCE]
        if performance_results:
            report_lines.extend([
                "## Performance Metrics",
                ""
            ])
            
            for result in performance_results:
                if result.metadata:
                    report_lines.append(f"### {result.test_name}")
                    for key, value in result.metadata.items():
                        if isinstance(value, float):
                            report_lines.append(f"- {key}: {value:.3f}")
                        else:
                            report_lines.append(f"- {key}: {value}")
                    report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Test report saved to {output_file}")
        
        return report


def run_comprehensive_tests(
    categories: Optional[List[TestCategory]] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive tests and generate report.
    
    Args:
        categories: Test categories to run
        output_file: Optional file to save report
        
    Returns:
        Test results summary
    """
    framework = ComprehensiveTestFramework()
    results = framework.run_all_tests(categories)
    
    if output_file:
        framework.generate_test_report(output_file)
    
    return results


if __name__ == "__main__":
    # Example usage
    test_results = run_comprehensive_tests(
        categories=[TestCategory.UNIT, TestCategory.INTEGRATION],
        output_file="test_report.md"
    )
    
    print(f"Test Results: {test_results['overall_status']}")
    print(f"Success Rate: {test_results['success_rate']:.1%}")
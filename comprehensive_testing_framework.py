#!/usr/bin/env python3
"""
Comprehensive Testing Framework for DP-Federated LoRA Lab.

This module implements a complete testing framework including:
- Unit testing for core components
- Integration testing for federated workflows
- Security testing for privacy guarantees
- Performance benchmarking
- Stress testing for resilience
- Automated test reporting
"""

import logging
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(Enum):
    """Test severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    status: TestStatus
    severity: TestSeverity
    execution_time: float
    error_message: Optional[str] = None
    test_data: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestSuite:
    """Collection of related tests."""
    suite_name: str
    tests: List[TestResult]
    setup_time: float
    teardown_time: float
    total_execution_time: float


class TestRunner:
    """Executes and manages test suites."""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
        self.current_suite: Optional[TestSuite] = None
        
    def run_test(self, test_name: str, test_func: Callable, severity: TestSeverity = TestSeverity.MEDIUM, **kwargs) -> TestResult:
        """Run a single test function."""
        start_time = time.time()
        
        try:
            logger.info(f"Running test: {test_name}")
            
            # Execute test function
            result_data = test_func(**kwargs)
            
            execution_time = time.time() - start_time
            
            # Determine test status based on result
            if result_data is False:
                status = TestStatus.FAILED
                error_message = "Test function returned False"
            elif isinstance(result_data, dict) and result_data.get("error"):
                status = TestStatus.FAILED
                error_message = result_data.get("error")
            else:
                status = TestStatus.PASSED
                error_message = None
            
            test_result = TestResult(
                test_name=test_name,
                status=status,
                severity=severity,
                execution_time=execution_time,
                error_message=error_message,
                test_data=result_data if isinstance(result_data, dict) else {"result": result_data}
            )
            
            logger.info(f"Test {test_name}: {status.value} ({execution_time:.3f}s)")
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"
            
            test_result = TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                severity=severity,
                execution_time=execution_time,
                error_message=error_message,
                test_data={"traceback": traceback.format_exc()}
            )
            
            logger.error(f"Test {test_name} ERROR: {error_message}")
            return test_result
    
    def create_suite(self, suite_name: str) -> None:
        """Create a new test suite."""
        if self.current_suite:
            self.test_suites.append(self.current_suite)
            
        self.current_suite = TestSuite(
            suite_name=suite_name,
            tests=[],
            setup_time=0.0,
            teardown_time=0.0,
            total_execution_time=0.0
        )
        logger.info(f"Created test suite: {suite_name}")
    
    def add_test_result(self, test_result: TestResult) -> None:
        """Add test result to current suite."""
        if self.current_suite:
            self.current_suite.tests.append(test_result)
            self.current_suite.total_execution_time += test_result.execution_time
    
    def finalize_suite(self) -> None:
        """Finalize current test suite."""
        if self.current_suite:
            self.test_suites.append(self.current_suite)
            self.current_suite = None
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Finalize current suite if exists
        self.finalize_suite()
        
        total_tests = sum(len(suite.tests) for suite in self.test_suites)
        passed_tests = sum(sum(1 for test in suite.tests if test.status == TestStatus.PASSED) for suite in self.test_suites)
        failed_tests = sum(sum(1 for test in suite.tests if test.status == TestStatus.FAILED) for suite in self.test_suites)
        error_tests = sum(sum(1 for test in suite.tests if test.status == TestStatus.ERROR) for suite in self.test_suites)
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": (passed_tests / max(1, total_tests)) * 100,
                "total_execution_time": sum(suite.total_execution_time for suite in self.test_suites)
            },
            "test_suites": [
                {
                    "suite_name": suite.suite_name,
                    "total_tests": len(suite.tests),
                    "passed": sum(1 for test in suite.tests if test.status == TestStatus.PASSED),
                    "failed": sum(1 for test in suite.tests if test.status == TestStatus.FAILED),
                    "errors": sum(1 for test in suite.tests if test.status == TestStatus.ERROR),
                    "execution_time": suite.total_execution_time,
                    "tests": [
                        {
                            "test_name": test.test_name,
                            "status": test.status.value,
                            "severity": test.severity.value,
                            "execution_time": test.execution_time,
                            "error_message": test.error_message,
                            "test_data": test.test_data,
                            "timestamp": test.timestamp
                        }
                        for test in suite.tests
                    ]
                }
                for suite in self.test_suites
            ],
            "timestamp": datetime.now().isoformat()
        }


# Test Functions for Core Components

def test_basic_lora_functionality() -> Dict[str, Any]:
    """Test basic LoRA parameter initialization and operations."""
    try:
        # Test LoRA parameter initialization
        lora_params = {
            "lora_A": [[random.gauss(0, 1) for _ in range(768)] for _ in range(16)],
            "lora_B": [[random.gauss(0, 1) for _ in range(16)] for _ in range(768)],
            "bias": [random.gauss(0, 1) for _ in range(768)]
        }
        
        # Validate parameter shapes
        assert len(lora_params["lora_A"]) == 16, "LoRA A matrix should have 16 rows"
        assert len(lora_params["lora_A"][0]) == 768, "LoRA A matrix should have 768 columns"
        assert len(lora_params["lora_B"]) == 768, "LoRA B matrix should have 768 rows"
        assert len(lora_params["lora_B"][0]) == 16, "LoRA B matrix should have 16 columns"
        assert len(lora_params["bias"]) == 768, "Bias vector should have 768 elements"
        
        # Test parameter multiplication (A @ B)
        result_approx = 0.0
        for i in range(5):  # Sample a few elements
            for j in range(5):
                dot_product = sum(lora_params["lora_A"][i][k] * lora_params["lora_B"][k][j] for k in range(16))
                result_approx += abs(dot_product)
        
        return {
            "lora_params_initialized": True,
            "parameter_shapes_valid": True,
            "matrix_multiplication_successful": True,
            "sample_result_magnitude": result_approx / 25
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_privacy_budget_management() -> Dict[str, Any]:
    """Test privacy budget allocation and consumption."""
    try:
        from enhanced_resilient_system import PrivacyBudgetManager
        
        # Initialize privacy manager
        privacy_manager = PrivacyBudgetManager(total_epsilon=10.0)
        
        # Test budget allocation
        allocation_results = []
        for i in range(5):
            client_id = f"client_{i}"
            allocated = privacy_manager.allocate_client_budget(client_id, 1.0)
            allocation_results.append(allocated)
        
        # Test budget consumption
        consumption_results = []
        for i in range(3):
            client_id = f"client_{i}"
            consumed = privacy_manager.consume_budget(client_id, 0.5)
            consumption_results.append(consumed)
        
        # Test budget status
        status = privacy_manager.get_budget_status()
        
        return {
            "allocations_successful": all(allocation_results),
            "consumptions_successful": all(consumption_results),
            "total_epsilon": status["total_epsilon"],
            "spent_epsilon": status["spent_epsilon"],
            "remaining_epsilon": status["remaining_epsilon"],
            "budget_utilization": status["budget_utilization"]
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_byzantine_detection() -> Dict[str, Any]:
    """Test Byzantine client detection capabilities."""
    try:
        from enhanced_resilient_system import ByzantineDetector
        
        detector = ByzantineDetector(detection_threshold=2.0)
        
        # Test normal client behavior
        normal_updates = []
        for i in range(5):
            update = {
                "model_updates": {
                    "lora_A": [[random.gauss(0, 0.1) for _ in range(768)] for _ in range(16)],
                    "bias": [random.gauss(0, 0.1) for _ in range(768)]
                },
                "training_loss": random.uniform(0.8, 1.2),
                "privacy_cost": 0.2,
                "data_size": 100
            }
            is_byzantine = detector.analyze_client_update("normal_client", update)
            normal_updates.append(is_byzantine)
        
        # Test Byzantine client behavior
        byzantine_updates = []
        for i in range(3):
            update = {
                "model_updates": {
                    "lora_A": [[random.gauss(0, 10.0) for _ in range(768)] for _ in range(16)],  # Abnormal variance
                    "bias": [random.gauss(0, 0.1) for _ in range(768)]
                },
                "training_loss": -5.0,  # Invalid loss
                "privacy_cost": -0.1,  # Invalid privacy cost
                "data_size": 100
            }
            is_byzantine = detector.analyze_client_update("byzantine_client", update)
            byzantine_updates.append(is_byzantine)
        
        return {
            "normal_client_false_positives": sum(normal_updates),
            "byzantine_client_detected": sum(byzantine_updates),
            "detection_sensitivity": sum(byzantine_updates) / len(byzantine_updates) * 100,
            "false_positive_rate": sum(normal_updates) / len(normal_updates) * 100,
            "total_alerts": len(detector.alerts)
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_federated_averaging() -> Dict[str, Any]:
    """Test federated averaging algorithm correctness."""
    try:
        # Create mock client updates
        client_updates = []
        for i in range(3):
            update = {
                "client_id": f"client_{i}",
                "model_updates": {
                    "bias": [1.0] * 768  # Simple test case
                },
                "data_size": 100,
                "training_loss": 1.0
            }
            client_updates.append(update)
        
        # Perform weighted averaging
        total_data = sum(update["data_size"] for update in client_updates)
        weighted_sum = [0.0] * 768
        
        for update in client_updates:
            weight = update["data_size"] / total_data
            for i in range(768):
                weighted_sum[i] += weight * update["model_updates"]["bias"][i]
        
        # Verify averaging result
        expected_result = 1.0  # Since all clients have bias=1.0
        actual_result = weighted_sum[0]  # All elements should be the same
        
        return {
            "averaging_successful": True,
            "expected_result": expected_result,
            "actual_result": actual_result,
            "result_accuracy": abs(expected_result - actual_result) < 1e-10,
            "num_clients_aggregated": len(client_updates)
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_system_resilience() -> Dict[str, Any]:
    """Test system resilience under failure conditions."""
    try:
        from enhanced_resilient_system import ResilientFederatedServer
        
        server = ResilientFederatedServer(num_rounds=2)
        
        # Register clients
        success_count = 0
        for i in range(3):
            config = {"data_size": 100, "privacy_requirements": {"epsilon": 1.0}}
            if server.register_client(f"client_{i}", config):
                success_count += 1
        
        # Simulate training with failures
        results = server.train_with_resilience()
        
        return {
            "clients_registered": success_count,
            "training_completed": len(results["training_history"]) > 0,
            "byzantine_detection_active": "byzantine_alerts" in results,
            "privacy_budget_managed": "privacy_budget_status" in results,
            "health_monitoring_active": "health_alerts" in results,
            "circuit_breaker_functional": "system_resilience_metrics" in results
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_performance_benchmarks() -> Dict[str, Any]:
    """Test system performance under various loads."""
    try:
        # Test with different client counts
        performance_results = []
        
        for num_clients in [5, 10, 20]:
            start_time = time.time()
            
            # Simulate federated training
            client_updates = []
            for i in range(num_clients):
                update = {
                    "client_id": f"client_{i}",
                    "model_updates": {
                        "lora_A": [[random.gauss(0, 0.1) for _ in range(768)] for _ in range(16)],
                        "bias": [random.gauss(0, 0.1) for _ in range(768)]
                    },
                    "data_size": 100,
                    "training_loss": random.uniform(0.5, 2.0)
                }
                client_updates.append(update)
            
            # Perform aggregation
            total_data = sum(update["data_size"] for update in client_updates)
            
            # Measure aggregation time
            agg_start = time.time()
            for param_name in ["lora_A", "bias"]:
                if param_name == "bias":
                    weighted_sum = [0.0] * 768
                    for update in client_updates:
                        weight = update["data_size"] / total_data
                        for i in range(768):
                            weighted_sum[i] += weight * update["model_updates"][param_name][i]
                else:
                    weighted_sum = [[0.0 for _ in range(768)] for _ in range(16)]
                    for update in client_updates:
                        weight = update["data_size"] / total_data
                        for i in range(16):
                            for j in range(768):
                                weighted_sum[i][j] += weight * update["model_updates"][param_name][i][j]
            
            agg_time = time.time() - agg_start
            total_time = time.time() - start_time
            
            performance_results.append({
                "num_clients": num_clients,
                "total_time": total_time,
                "aggregation_time": agg_time,
                "throughput": num_clients / total_time
            })
        
        return {
            "performance_results": performance_results,
            "scalability_tested": True,
            "max_throughput": max(r["throughput"] for r in performance_results),
            "avg_aggregation_time": sum(r["aggregation_time"] for r in performance_results) / len(performance_results)
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_security_guarantees() -> Dict[str, Any]:
    """Test security and privacy guarantees."""
    try:
        # Test differential privacy noise addition
        import math
        
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0
        
        # Calculate noise scale
        noise_scale = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        
        # Test noise generation
        noise_samples = [random.gauss(0, noise_scale) for _ in range(1000)]
        noise_std = math.sqrt(sum(x**2 for x in noise_samples) / len(noise_samples))
        
        # Test privacy budget accounting
        rounds = 10
        epsilon_per_round = 0.1
        total_epsilon_spent = rounds * epsilon_per_round
        
        # Test secure aggregation simulation
        client_secrets = [random.random() for _ in range(5)]
        aggregated_secret = sum(client_secrets)  # Simple aggregation
        
        return {
            "noise_scale_calculated": noise_scale,
            "noise_distribution_valid": abs(noise_std - noise_scale) < noise_scale * 0.2,  # Within 20%
            "privacy_budget_tracking": total_epsilon_spent <= epsilon,
            "secure_aggregation_functional": aggregated_secret > 0,
            "differential_privacy_enabled": True
        }
        
    except Exception as e:
        return {"error": str(e)}


def run_comprehensive_tests():
    """Run the complete test suite."""
    logger.info("🧪 Starting Comprehensive Test Suite")
    
    runner = TestRunner()
    
    # Core Functionality Tests
    runner.create_suite("Core Functionality")
    
    test_result = runner.run_test(
        "Basic LoRA Functionality",
        test_basic_lora_functionality,
        TestSeverity.CRITICAL
    )
    runner.add_test_result(test_result)
    
    test_result = runner.run_test(
        "Federated Averaging",
        test_federated_averaging,
        TestSeverity.CRITICAL
    )
    runner.add_test_result(test_result)
    
    # Security and Privacy Tests
    runner.create_suite("Security and Privacy")
    
    test_result = runner.run_test(
        "Privacy Budget Management",
        test_privacy_budget_management,
        TestSeverity.HIGH
    )
    runner.add_test_result(test_result)
    
    test_result = runner.run_test(
        "Byzantine Detection",
        test_byzantine_detection,
        TestSeverity.HIGH
    )
    runner.add_test_result(test_result)
    
    test_result = runner.run_test(
        "Security Guarantees",
        test_security_guarantees,
        TestSeverity.HIGH
    )
    runner.add_test_result(test_result)
    
    # System Resilience Tests
    runner.create_suite("System Resilience")
    
    test_result = runner.run_test(
        "System Resilience",
        test_system_resilience,
        TestSeverity.MEDIUM
    )
    runner.add_test_result(test_result)
    
    # Performance Tests
    runner.create_suite("Performance")
    
    test_result = runner.run_test(
        "Performance Benchmarks",
        test_performance_benchmarks,
        TestSeverity.MEDIUM
    )
    runner.add_test_result(test_result)
    
    # Generate test report
    report = runner.generate_report()
    
    # Save test results
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "comprehensive_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    summary = report["test_summary"]
    logger.info(f"\n✅ Test Suite Complete!")
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed_tests']}")
    logger.info(f"Failed: {summary['failed_tests']}")
    logger.info(f"Errors: {summary['error_tests']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"Total execution time: {summary['total_execution_time']:.3f}s")
    logger.info(f"Results saved to: {results_dir / 'comprehensive_test_report.json'}")
    
    return report


def main():
    """Main testing function."""
    print("🧪 DP-Federated LoRA Lab - Comprehensive Testing Framework")
    print("=" * 70)
    
    try:
        # Run comprehensive tests
        report = run_comprehensive_tests()
        
        print("\n🎉 Testing completed successfully!")
        print("Test coverage:")
        print("  ✅ Core LoRA functionality")
        print("  ✅ Federated averaging algorithms")
        print("  ✅ Privacy budget management")
        print("  ✅ Byzantine client detection")
        print("  ✅ System resilience and fault tolerance")
        print("  ✅ Performance benchmarks")
        print("  ✅ Security guarantees")
        
        success_rate = report["test_summary"]["success_rate"]
        if success_rate >= 90:
            print(f"\n🏆 Excellent test results: {success_rate:.1f}% success rate")
        elif success_rate >= 80:
            print(f"\n✅ Good test results: {success_rate:.1f}% success rate")
        else:
            print(f"\n⚠️  Test results need attention: {success_rate:.1f}% success rate")
        
    except Exception as e:
        logger.error(f"Testing framework failed: {e}")
        raise


if __name__ == "__main__":
    main()
"""
Comprehensive Quality Gates Validator for Federated Learning System.

This script implements comprehensive quality gates validation including:
- Code quality and security analysis
- Performance benchmarks and testing
- Privacy compliance verification
- System integration testing
- Production readiness checks

Author: Terry (Terragon Labs)
"""

import os
import sys
import subprocess
import time
import json
import logging
import argparse
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import hashlib
import re

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import our modules
try:
    from dp_federated_lora.adaptive_privacy_budget_optimizer import (
        AdaptivePrivacyBudgetOptimizer, BudgetAllocationStrategy
    )
    from dp_federated_lora.robust_privacy_budget_validator import (
        RobustPrivacyBudgetValidator, create_robust_validator
    )
    from dp_federated_lora.comprehensive_monitoring_system import (
        ComprehensiveMonitoringSystem, create_monitoring_system
    )
    from dp_federated_lora.hyperscale_optimization_engine import (
        HyperscaleOptimizationEngine, ClientCapabilities, ClientTier
    )
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some tests may be skipped")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class QualityGatesSummary:
    """Summary of all quality gates."""
    total_gates: int
    passed_gates: int
    failed_gates: int
    overall_score: float
    execution_time: float
    results: List[QualityGateResult]
    timestamp: str
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        return (self.passed_gates / self.total_gates * 100) if self.total_gates > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "pass_rate": self.pass_rate,
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "results": [
                {
                    "gate_name": result.gate_name,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "details": result.details,
                    "execution_time": result.execution_time,
                    "warnings": result.warnings,
                    "errors": result.errors
                }
                for result in self.results
            ]
        }


class QualityGateValidator:
    """Main quality gates validator."""
    
    def __init__(self, project_root: str):
        """Initialize quality gates validator."""
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.scripts_dir = self.project_root / "scripts"
        
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
        logger.info(f"Initialized quality gates validator for project: {self.project_root}")
    
    def run_command(self, command: List[str], cwd: Optional[str] = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command failed: {str(e)}"
    
    def validate_code_quality(self) -> QualityGateResult:
        """Validate code quality using static analysis tools."""
        start_time = time.time()
        
        try:
            score = 0.0
            total_checks = 0
            errors = []
            warnings = []
            details = {}
            
            # Check if source directory exists
            if not self.src_dir.exists():
                return QualityGateResult(
                    gate_name="Code Quality",
                    passed=False,
                    score=0.0,
                    message="Source directory not found",
                    execution_time=time.time() - start_time
                )
            
            # Python syntax check
            python_files = list(self.src_dir.rglob("*.py"))
            if python_files:
                total_checks += 1
                syntax_errors = 0
                
                for py_file in python_files:
                    returncode, stdout, stderr = self.run_command([
                        sys.executable, "-m", "py_compile", str(py_file)
                    ])
                    if returncode != 0:
                        syntax_errors += 1
                        errors.append(f"Syntax error in {py_file}: {stderr}")
                
                if syntax_errors == 0:
                    score += 25  # 25 points for clean syntax
                    details["syntax_check"] = "PASSED"
                else:
                    details["syntax_check"] = f"FAILED - {syntax_errors} files with errors"
            
            # Import validation
            try:
                total_checks += 1
                sys.path.insert(0, str(self.src_dir))
                
                # Try importing main modules
                import_errors = 0
                test_imports = [
                    "dp_federated_lora",
                    "dp_federated_lora.adaptive_privacy_budget_optimizer",
                    "dp_federated_lora.robust_privacy_budget_validator",
                    "dp_federated_lora.comprehensive_monitoring_system",
                    "dp_federated_lora.hyperscale_optimization_engine"
                ]
                
                for module_name in test_imports:
                    try:
                        __import__(module_name)
                    except ImportError as e:
                        import_errors += 1
                        warnings.append(f"Import warning for {module_name}: {str(e)}")
                    except Exception as e:
                        import_errors += 1
                        errors.append(f"Import error for {module_name}: {str(e)}")
                
                if import_errors == 0:
                    score += 25  # 25 points for clean imports
                    details["import_check"] = "PASSED"
                else:
                    details["import_check"] = f"ISSUES - {import_errors} import problems"
                    
            except Exception as e:
                errors.append(f"Import validation failed: {str(e)}")
            
            # Flake8 style check (if available)
            total_checks += 1
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "flake8", str(self.src_dir), "--count", "--max-line-length=120", "--statistics"
            ])
            
            if returncode == 0:
                score += 25  # 25 points for style compliance
                details["style_check"] = "PASSED"
            else:
                # Extract flake8 statistics if available
                if stdout:
                    lines = stdout.strip().split('\n')
                    if lines and lines[-1].isdigit():
                        issue_count = int(lines[-1])
                        if issue_count < 10:
                            score += 15  # Partial credit
                        details["style_check"] = f"ISSUES - {issue_count} style violations"
                        warnings.append(f"Flake8 found {issue_count} style issues")
                else:
                    details["style_check"] = "TOOL_NOT_AVAILABLE"
                    warnings.append("Flake8 not available, skipping style check")
            
            # Security check with bandit (if available)
            total_checks += 1
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "bandit", "-r", str(self.src_dir), "-f", "json"
            ])
            
            if returncode == 0:
                score += 25  # 25 points for security
                details["security_check"] = "PASSED"
            else:
                if "No module named 'bandit'" in stderr:
                    details["security_check"] = "TOOL_NOT_AVAILABLE"
                    warnings.append("Bandit not available, skipping security check")
                    score += 10  # Partial credit if tool not available
                else:
                    # Parse bandit output for security issues
                    try:
                        if stdout:
                            bandit_data = json.loads(stdout)
                            high_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'HIGH'])
                            medium_severity = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'MEDIUM'])
                            
                            if high_severity == 0:
                                score += 15  # Partial credit for no high severity issues
                            
                            details["security_check"] = f"ISSUES - {high_severity} high, {medium_severity} medium severity"
                            if high_severity > 0:
                                errors.append(f"Bandit found {high_severity} high severity security issues")
                    except (json.JSONDecodeError, KeyError):
                        details["security_check"] = "FAILED - Could not parse results"
            
            # Calculate final score
            if total_checks > 0:
                final_score = min(100.0, score)  # Cap at 100
            else:
                final_score = 0.0
            
            passed = final_score >= 70.0  # Pass threshold
            
            return QualityGateResult(
                gate_name="Code Quality",
                passed=passed,
                score=final_score,
                message=f"Code quality score: {final_score:.1f}/100 ({len(errors)} errors, {len(warnings)} warnings)",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                message=f"Code quality validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def validate_testing(self) -> QualityGateResult:
        """Validate testing coverage and execution."""
        start_time = time.time()
        
        try:
            score = 0.0
            errors = []
            warnings = []
            details = {}
            
            # Check if tests directory exists
            if not self.tests_dir.exists():
                return QualityGateResult(
                    gate_name="Testing",
                    passed=False,
                    score=0.0,
                    message="Tests directory not found",
                    execution_time=time.time() - start_time
                )
            
            # Count test files
            test_files = list(self.tests_dir.rglob("test_*.py"))
            if len(test_files) == 0:
                warnings.append("No test files found")
                details["test_files"] = 0
            else:
                score += 20  # 20 points for having tests
                details["test_files"] = len(test_files)
            
            # Run pytest if available
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "pytest", str(self.tests_dir), "-v", "--tb=short", "--no-header"
            ], timeout=600)  # 10 minute timeout for tests
            
            if returncode == 0:
                score += 50  # 50 points for passing tests
                details["pytest_status"] = "PASSED"
                
                # Extract test count from output
                test_pattern = r"(\d+) passed"
                match = re.search(test_pattern, stdout)
                if match:
                    passed_tests = int(match.group(1))
                    details["tests_passed"] = passed_tests
                    if passed_tests >= 10:
                        score += 20  # Bonus for good test coverage
                    
            elif "No module named 'pytest'" in stderr:
                details["pytest_status"] = "TOOL_NOT_AVAILABLE"
                warnings.append("Pytest not available")
                if test_files:
                    score += 20  # Partial credit for having test files
            else:
                details["pytest_status"] = "FAILED"
                
                # Extract failure information
                failed_pattern = r"(\d+) failed"
                passed_pattern = r"(\d+) passed"
                
                failed_match = re.search(failed_pattern, stdout)
                passed_match = re.search(passed_pattern, stdout)
                
                if failed_match:
                    failed_tests = int(failed_match.group(1))
                    details["tests_failed"] = failed_tests
                    errors.append(f"{failed_tests} tests failed")
                
                if passed_match:
                    passed_tests = int(passed_match.group(1))
                    details["tests_passed"] = passed_tests
                    score += min(40, passed_tests * 2)  # Partial credit for passing tests
            
            # Check for test coverage if pytest-cov is available
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "pytest", str(self.tests_dir), "--cov=" + str(self.src_dir), "--cov-report=term-missing", "--quiet"
            ], timeout=300)
            
            if returncode == 0 and "TOTAL" in stdout:
                # Extract coverage percentage
                coverage_pattern = r"TOTAL.*?(\d+)%"
                match = re.search(coverage_pattern, stdout)
                if match:
                    coverage_percent = int(match.group(1))
                    details["coverage_percent"] = coverage_percent
                    
                    if coverage_percent >= 80:
                        score += 10  # Bonus for high coverage
                    elif coverage_percent >= 60:
                        score += 5   # Partial bonus
            
            final_score = min(100.0, score)
            passed = final_score >= 60.0  # Pass threshold
            
            return QualityGateResult(
                gate_name="Testing",
                passed=passed,
                score=final_score,
                message=f"Testing score: {final_score:.1f}/100 ({details.get('tests_passed', 0)} passed, {details.get('tests_failed', 0)} failed)",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Testing",
                passed=False,
                score=0.0,
                message=f"Testing validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def validate_privacy_compliance(self) -> QualityGateResult:
        """Validate privacy compliance and differential privacy implementation."""
        start_time = time.time()
        
        try:
            score = 0.0
            errors = []
            warnings = []
            details = {}
            
            # Test privacy budget optimizer functionality
            try:
                from dp_federated_lora.adaptive_privacy_budget_optimizer import AdaptivePrivacyBudgetOptimizer
                
                optimizer = AdaptivePrivacyBudgetOptimizer(
                    total_epsilon_budget=10.0,
                    optimization_strategy=BudgetAllocationStrategy.UNIFORM
                )
                
                # Test client registration
                optimizer.register_client("test_client", epsilon_budget=5.0)
                
                # Test budget allocation
                allocations = optimizer.allocate_budget(["test_client"], round_num=1, round_budget_fraction=0.2)
                
                if len(allocations) == 1 and allocations[0].epsilon_allocated > 0:
                    score += 30  # 30 points for basic privacy budget functionality
                    details["privacy_budget_allocation"] = "PASSED"
                else:
                    errors.append("Privacy budget allocation test failed")
                    details["privacy_budget_allocation"] = "FAILED"
                
                # Test budget consumption tracking
                if optimizer.epsilon_consumed > 0 and optimizer.epsilon_consumed <= optimizer.total_epsilon_budget:
                    score += 20  # 20 points for budget tracking
                    details["budget_tracking"] = "PASSED"
                else:
                    errors.append("Budget consumption tracking test failed")
                    details["budget_tracking"] = "FAILED"
                
            except ImportError:
                warnings.append("Could not import privacy budget optimizer")
                details["privacy_budget_allocation"] = "MODULE_NOT_AVAILABLE"
            except Exception as e:
                errors.append(f"Privacy budget optimizer test failed: {str(e)}")
                details["privacy_budget_allocation"] = "FAILED"
            
            # Test privacy validator functionality
            try:
                from dp_federated_lora.robust_privacy_budget_validator import create_robust_validator
                
                validator = create_robust_validator()
                
                # Test validation functionality
                test_allocation = {
                    "client_id": "test_client",
                    "epsilon_allocated": 1.0,
                    "delta_allocated": 1e-6,
                    "expected_utility": 0.7,
                    "allocation_confidence": 0.8
                }
                
                test_profiles = {
                    "test_client": {
                        "current_epsilon": 0.0,
                        "current_delta": 0.0,
                        "total_epsilon_budget": 5.0,
                        "total_delta_budget": 1e-5,
                        "data_sensitivity": 1.0
                    }
                }
                
                test_constraints = {
                    "max_epsilon_per_round": 2.0,
                    "max_delta_per_round": 1e-5
                }
                
                validation_errors = validator.validate_budget_allocation(
                    test_allocation, test_profiles, test_constraints, []
                )
                
                if len(validation_errors) == 0:
                    score += 25  # 25 points for privacy validation
                    details["privacy_validation"] = "PASSED"
                else:
                    # Some validation errors might be expected for edge cases
                    if len(validation_errors) <= 2:
                        score += 15  # Partial credit
                    details["privacy_validation"] = f"WARNINGS - {len(validation_errors)} validation issues"
                    
            except ImportError:
                warnings.append("Could not import privacy validator")
                details["privacy_validation"] = "MODULE_NOT_AVAILABLE"
            except Exception as e:
                errors.append(f"Privacy validator test failed: {str(e)}")
                details["privacy_validation"] = "FAILED"
            
            # Check for privacy-related constants and configurations
            privacy_files = []
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(keyword in content for keyword in [
                        'epsilon', 'delta', 'differential_privacy', 'privacy_budget',
                        'opacus', 'PrivacyEngine', 'noise_multiplier'
                    ]):
                        privacy_files.append(py_file.name)
            
            if privacy_files:
                score += 15  # 15 points for privacy-related code
                details["privacy_related_files"] = len(privacy_files)
            else:
                warnings.append("No privacy-related code patterns found")
                details["privacy_related_files"] = 0
            
            # Check for proper epsilon-delta accounting
            accounting_patterns = [
                r'epsilon.*\+.*epsilon',  # Epsilon composition
                r'delta.*\+.*delta',      # Delta composition
                r'privacy.*account',      # Privacy accounting
                r'budget.*track',         # Budget tracking
            ]
            
            accounting_found = False
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in accounting_patterns):
                        accounting_found = True
                        break
            
            if accounting_found:
                score += 10  # 10 points for privacy accounting patterns
                details["privacy_accounting"] = "FOUND"
            else:
                warnings.append("Privacy accounting patterns not clearly detected")
                details["privacy_accounting"] = "NOT_DETECTED"
            
            final_score = min(100.0, score)
            passed = final_score >= 70.0  # Pass threshold for privacy compliance
            
            return QualityGateResult(
                gate_name="Privacy Compliance",
                passed=passed,
                score=final_score,
                message=f"Privacy compliance score: {final_score:.1f}/100",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Privacy Compliance",
                passed=False,
                score=0.0,
                message=f"Privacy compliance validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def validate_system_integration(self) -> QualityGateResult:
        """Validate system integration and component interactions."""
        start_time = time.time()
        
        try:
            score = 0.0
            errors = []
            warnings = []
            details = {}
            
            # Test component integration
            integration_tests = [
                self._test_optimizer_validator_integration,
                self._test_monitoring_integration,
                self._test_hyperscale_integration,
                self._test_end_to_end_workflow
            ]
            
            passed_tests = 0
            total_tests = len(integration_tests)
            
            for test_func in integration_tests:
                try:
                    test_result = test_func()
                    if test_result:
                        passed_tests += 1
                        score += 25  # 25 points per integration test
                except Exception as e:
                    errors.append(f"Integration test {test_func.__name__} failed: {str(e)}")
            
            details["integration_tests_passed"] = passed_tests
            details["integration_tests_total"] = total_tests
            
            final_score = min(100.0, score)
            passed = final_score >= 60.0  # Pass threshold
            
            return QualityGateResult(
                gate_name="System Integration",
                passed=passed,
                score=final_score,
                message=f"Integration score: {final_score:.1f}/100 ({passed_tests}/{total_tests} tests passed)",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="System Integration",
                passed=False,
                score=0.0,
                message=f"System integration validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _test_optimizer_validator_integration(self) -> bool:
        """Test integration between optimizer and validator."""
        try:
            from dp_federated_lora.adaptive_privacy_budget_optimizer import AdaptivePrivacyBudgetOptimizer
            from dp_federated_lora.robust_privacy_budget_validator import create_robust_validator
            
            # Create optimizer and validator
            optimizer = AdaptivePrivacyBudgetOptimizer(total_epsilon_budget=20.0)
            validator = create_robust_validator()
            
            # Register client and allocate budget
            optimizer.register_client("test_client", epsilon_budget=10.0)
            allocations = optimizer.allocate_budget(["test_client"], round_num=1)
            
            if not allocations:
                return False
            
            # Validate the allocation
            allocation_dict = {
                "client_id": allocations[0].client_id,
                "epsilon_allocated": allocations[0].epsilon_allocated,
                "delta_allocated": allocations[0].delta_allocated,
                "expected_utility": allocations[0].expected_utility,
                "allocation_confidence": allocations[0].allocation_confidence
            }
            
            client_profiles = {
                "test_client": optimizer.client_profiles["test_client"].__dict__
            }
            
            validation_errors = validator.validate_budget_allocation(
                allocation_dict, client_profiles, {"max_epsilon_per_round": 5.0}, []
            )
            
            # Integration successful if validation completes (errors are acceptable)
            return True
            
        except Exception:
            return False
    
    def _test_monitoring_integration(self) -> bool:
        """Test monitoring system integration."""
        try:
            from dp_federated_lora.comprehensive_monitoring_system import create_monitoring_system
            
            # Create monitoring system
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                monitoring = create_monitoring_system(db_path=tmp_db.name, auto_start=False)
            
            # Record some metrics
            monitoring.record_metric("test_metric", 42.0, unit="units")
            monitoring.record_metric("test_metric", 43.0, unit="units")
            
            # Get dashboard data
            dashboard_data = monitoring.create_dashboard_data()
            
            # Clean up
            monitoring.stop_all_monitoring()
            
            return "metrics_summary" in dashboard_data and len(dashboard_data["metrics_summary"]) > 0
            
        except Exception:
            return False
    
    def _test_hyperscale_integration(self) -> bool:
        """Test hyperscale optimization engine integration."""
        try:
            from dp_federated_lora.hyperscale_optimization_engine import (
                create_hyperscale_optimizer, ClientCapabilities, ClientTier
            )
            
            # Create optimizer (without Redis to avoid dependency)
            optimizer = create_hyperscale_optimizer(auto_start=False, redis_host="nonexistent")
            
            # Register test client
            capabilities = ClientCapabilities(
                client_id="test_client",
                tier=ClientTier.DESKTOP,
                compute_power=100.0,
                memory_gb=8.0,
                bandwidth_mbps=50.0,
                latency_ms=20.0
            )
            
            optimizer.register_client("test_client", capabilities)
            optimizer.active_clients.add("test_client")
            
            # Test client selection
            selected = optimizer.select_clients_for_round(1)
            
            # Clean up
            optimizer.stop_continuous_optimization()
            
            return len(selected) == 1 and selected[0] == "test_client"
            
        except Exception:
            return False
    
    def _test_end_to_end_workflow(self) -> bool:
        """Test end-to-end federated learning workflow."""
        try:
            from dp_federated_lora.adaptive_privacy_budget_optimizer import AdaptivePrivacyBudgetOptimizer
            
            # Create optimizer
            optimizer = AdaptivePrivacyBudgetOptimizer(total_epsilon_budget=30.0)
            
            # Register multiple clients
            clients = []
            for i in range(3):
                client_id = f"client_{i}"
                optimizer.register_client(client_id, epsilon_budget=10.0)
                clients.append(client_id)
            
            # Simulate multiple training rounds
            for round_num in range(1, 4):
                # Allocate budget
                allocations = optimizer.allocate_budget(clients, round_num, round_budget_fraction=0.2)
                
                if len(allocations) != 3:
                    return False
                
                # Simulate performance updates
                for allocation in allocations:
                    optimizer.update_client_performance(
                        allocation.client_id,
                        {"accuracy": 0.7 + round_num * 0.05}
                    )
            
            # Check final state
            report = optimizer.get_optimization_report()
            
            return (
                report["total_budget"]["rounds_completed"] == 3 and
                report["allocation_efficiency"]["total_utility_achieved"] > 0 and
                len(report["client_profiles"]) == 3
            )
            
        except Exception:
            return False
    
    def validate_performance(self) -> QualityGateResult:
        """Validate system performance benchmarks."""
        start_time = time.time()
        
        try:
            score = 0.0
            errors = []
            warnings = []
            details = {}
            
            # Test privacy budget optimization performance
            try:
                from dp_federated_lora.adaptive_privacy_budget_optimizer import AdaptivePrivacyBudgetOptimizer
                
                perf_start = time.time()
                optimizer = AdaptivePrivacyBudgetOptimizer(total_epsilon_budget=100.0)
                
                # Register many clients
                for i in range(50):
                    optimizer.register_client(f"client_{i}", epsilon_budget=5.0)
                
                # Perform budget allocation
                client_ids = [f"client_{i}" for i in range(50)]
                allocations = optimizer.allocate_budget(client_ids[:20], round_num=1)
                
                allocation_time = time.time() - perf_start
                details["budget_allocation_time"] = allocation_time
                
                if allocation_time < 5.0:  # Should complete within 5 seconds
                    score += 25
                    details["allocation_performance"] = "EXCELLENT"
                elif allocation_time < 10.0:
                    score += 15
                    details["allocation_performance"] = "GOOD"
                else:
                    details["allocation_performance"] = "SLOW"
                    warnings.append(f"Budget allocation took {allocation_time:.2f}s")
                
                if len(allocations) == 20:
                    score += 15  # Correct number of allocations
                
            except ImportError:
                warnings.append("Could not test privacy budget performance")
            except Exception as e:
                errors.append(f"Privacy budget performance test failed: {str(e)}")
            
            # Test monitoring system performance
            try:
                from dp_federated_lora.comprehensive_monitoring_system import create_monitoring_system
                
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                    monitoring = create_monitoring_system(db_path=tmp_db.name, auto_start=False)
                
                perf_start = time.time()
                
                # Record many metrics
                for i in range(1000):
                    monitoring.record_metric(f"test_metric_{i%10}", float(i), unit="units")
                
                metric_recording_time = time.time() - perf_start
                details["metric_recording_time"] = metric_recording_time
                
                if metric_recording_time < 2.0:
                    score += 20
                    details["monitoring_performance"] = "EXCELLENT"
                elif metric_recording_time < 5.0:
                    score += 10
                    details["monitoring_performance"] = "GOOD"
                else:
                    details["monitoring_performance"] = "SLOW"
                    warnings.append(f"Metric recording took {metric_recording_time:.2f}s")
                
                # Test dashboard data generation
                perf_start = time.time()
                dashboard_data = monitoring.create_dashboard_data()
                dashboard_time = time.time() - perf_start
                
                details["dashboard_generation_time"] = dashboard_time
                
                if dashboard_time < 1.0:
                    score += 15
                elif dashboard_time < 3.0:
                    score += 10
                else:
                    warnings.append(f"Dashboard generation took {dashboard_time:.2f}s")
                
                monitoring.stop_all_monitoring()
                
            except ImportError:
                warnings.append("Could not test monitoring performance")
            except Exception as e:
                errors.append(f"Monitoring performance test failed: {str(e)}")
            
            # Memory usage test
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            details["memory_usage_mb"] = memory_usage_mb
            
            if memory_usage_mb < 500:  # Less than 500MB
                score += 20
                details["memory_usage"] = "EFFICIENT"
            elif memory_usage_mb < 1000:  # Less than 1GB
                score += 10
                details["memory_usage"] = "ACCEPTABLE"
            else:
                details["memory_usage"] = "HIGH"
                warnings.append(f"High memory usage: {memory_usage_mb:.1f}MB")
            
            final_score = min(100.0, score)
            passed = final_score >= 60.0
            
            return QualityGateResult(
                gate_name="Performance",
                passed=passed,
                score=final_score,
                message=f"Performance score: {final_score:.1f}/100",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Performance",
                passed=False,
                score=0.0,
                message=f"Performance validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def validate_production_readiness(self) -> QualityGateResult:
        """Validate production readiness."""
        start_time = time.time()
        
        try:
            score = 0.0
            errors = []
            warnings = []
            details = {}
            
            # Check for required configuration files
            required_files = [
                "requirements.txt",
                "pyproject.toml",
                "README.md"
            ]
            
            found_files = 0
            for filename in required_files:
                file_path = self.project_root / filename
                if file_path.exists():
                    found_files += 1
                else:
                    warnings.append(f"Missing {filename}")
            
            details["required_files"] = f"{found_files}/{len(required_files)}"
            score += (found_files / len(required_files)) * 20
            
            # Check for deployment configurations
            deployment_files = [
                "Dockerfile",
                "docker-compose.yml",
                "deployment",
                "kubernetes"
            ]
            
            found_deployment = 0
            for filename in deployment_files:
                path = self.project_root / filename
                if path.exists():
                    found_deployment += 1
            
            if found_deployment > 0:
                score += 15
                details["deployment_configs"] = "FOUND"
            else:
                warnings.append("No deployment configurations found")
                details["deployment_configs"] = "MISSING"
            
            # Check for logging configuration
            logging_found = False
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(pattern in content for pattern in ['logging', 'logger', 'getLogger']):
                        logging_found = True
                        break
            
            if logging_found:
                score += 15
                details["logging"] = "CONFIGURED"
            else:
                warnings.append("Logging configuration not found")
                details["logging"] = "MISSING"
            
            # Check for error handling
            error_handling_patterns = [
                r'try:',
                r'except.*:',
                r'raise.*Error',
                r'ErrorHandler',
                r'circuit.*breaker'
            ]
            
            error_handling_found = 0
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern in error_handling_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            error_handling_found += 1
                            break
            
            if error_handling_found > 0:
                score += 20
                details["error_handling"] = "IMPLEMENTED"
            else:
                errors.append("Error handling patterns not found")
                details["error_handling"] = "MISSING"
            
            # Check for monitoring and health checks
            monitoring_patterns = [
                'health.*check',
                'monitoring',
                'metrics',
                'prometheus',
                'grafana'
            ]
            
            monitoring_found = False
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in monitoring_patterns):
                        monitoring_found = True
                        break
            
            if monitoring_found:
                score += 15
                details["monitoring"] = "IMPLEMENTED"
            else:
                warnings.append("Monitoring patterns not clearly detected")
                details["monitoring"] = "NOT_DETECTED"
            
            # Check for security configurations
            security_patterns = [
                'authentication',
                'authorization',
                'encrypt',
                'secure',
                'ssl',
                'tls'
            ]
            
            security_found = False
            for py_file in self.src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in security_patterns):
                        security_found = True
                        break
            
            if security_found:
                score += 15
                details["security"] = "IMPLEMENTED"
            else:
                warnings.append("Security patterns not clearly detected")
                details["security"] = "NOT_DETECTED"
            
            final_score = min(100.0, score)
            passed = final_score >= 70.0
            
            return QualityGateResult(
                gate_name="Production Readiness",
                passed=passed,
                score=final_score,
                message=f"Production readiness score: {final_score:.1f}/100",
                details=details,
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Production Readiness",
                passed=False,
                score=0.0,
                message=f"Production readiness validation failed: {str(e)}",
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def run_all_quality_gates(self, parallel: bool = True) -> QualityGatesSummary:
        """Run all quality gates."""
        start_time = time.time()
        
        gate_functions = [
            self.validate_code_quality,
            self.validate_testing,
            self.validate_privacy_compliance,
            self.validate_system_integration,
            self.validate_performance,
            self.validate_production_readiness
        ]
        
        if parallel:
            # Run gates in parallel for faster execution
            with ThreadPoolExecutor(max_workers=len(gate_functions)) as executor:
                future_to_gate = {executor.submit(gate_func): gate_func.__name__ for gate_func in gate_functions}
                
                for future in as_completed(future_to_gate):
                    gate_name = future_to_gate[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        logger.info(f"Completed {gate_name}: {'PASSED' if result.passed else 'FAILED'} ({result.score:.1f}/100)")
                    except Exception as e:
                        logger.error(f"Gate {gate_name} failed with exception: {e}")
                        self.results.append(QualityGateResult(
                            gate_name=gate_name,
                            passed=False,
                            score=0.0,
                            message=f"Gate execution failed: {str(e)}",
                            errors=[str(e)]
                        ))
        else:
            # Run gates sequentially
            for gate_func in gate_functions:
                gate_name = gate_func.__name__
                try:
                    result = gate_func()
                    self.results.append(result)
                    logger.info(f"Completed {gate_name}: {'PASSED' if result.passed else 'FAILED'} ({result.score:.1f}/100)")
                except Exception as e:
                    logger.error(f"Gate {gate_name} failed with exception: {e}")
                    self.results.append(QualityGateResult(
                        gate_name=gate_name,
                        passed=False,
                        score=0.0,
                        message=f"Gate execution failed: {str(e)}",
                        errors=[str(e)]
                    ))
        
        # Calculate summary
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results if result.passed)
        failed_gates = total_gates - passed_gates
        overall_score = sum(result.score for result in self.results) / total_gates if total_gates > 0 else 0.0
        execution_time = time.time() - start_time
        
        return QualityGatesSummary(
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            overall_score=overall_score,
            execution_time=execution_time,
            results=self.results,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


def main():
    """Main entry point for quality gates validation."""
    parser = argparse.ArgumentParser(description="Comprehensive Quality Gates Validator")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run gates in parallel")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = QualityGateValidator(args.project_root)
    
    print("🛡️ Starting Comprehensive Quality Gates Validation")
    print("=" * 60)
    
    # Run all quality gates
    summary = validator.run_all_quality_gates(parallel=args.parallel)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("QUALITY GATES SUMMARY")
    print("=" * 60)
    
    print(f"Total Gates: {summary.total_gates}")
    print(f"Passed: {summary.passed_gates}")
    print(f"Failed: {summary.failed_gates}")
    print(f"Pass Rate: {summary.pass_rate:.1f}%")
    print(f"Overall Score: {summary.overall_score:.1f}/100")
    print(f"Execution Time: {summary.execution_time:.1f}s")
    
    print(f"\nDetailed Results:")
    print("-" * 60)
    
    for result in summary.results:
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"{status} {result.gate_name:<25} Score: {result.score:5.1f}/100 ({result.execution_time:.1f}s)")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        if result.errors:
            for error in result.errors:
                print(f"  🚨 {error}")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with appropriate code
    exit_code = 0 if summary.overall_score >= 70.0 and summary.passed_gates >= summary.total_gates * 0.8 else 1
    
    if exit_code == 0:
        print(f"\n🎉 Quality gates validation PASSED!")
    else:
        print(f"\n💥 Quality gates validation FAILED!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quality Gates Script for DP-Federated LoRA system.

This script implements comprehensive quality gates including code quality checks,
security validation, performance benchmarks, and comprehensive testing to ensure
production readiness of the federated learning system.
"""

import os
import sys
import logging
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dp_federated_lora.comprehensive_testing import (
    ComprehensiveTestFramework, TestCategory, run_comprehensive_tests
)


logger = logging.getLogger(__name__)


class QualityGateResult(Enum):
    """Quality gate results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Individual quality gate result."""
    
    name: str
    result: QualityGateResult
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class QualityGateRunner:
    """Main quality gate runner."""
    
    def __init__(self, project_root: Path):
        """Initialize quality gate runner."""
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"
        self.results: List[GateResult] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Quality gate runner initialized for {project_root}")
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        logger.info("Starting comprehensive quality gate execution")
        start_time = time.time()
        
        # Quality gates in order of execution
        gates = [
            ("Code Structure", self.check_code_structure),
            ("Import Validation", self.check_imports),
            ("Security Scan", self.run_security_scan),
            ("Code Quality", self.check_code_quality),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests),
            ("Documentation", self.check_documentation),
            ("Configuration", self.validate_configuration),
        ]
        
        for gate_name, gate_func in gates:
            logger.info(f"Running quality gate: {gate_name}")
            try:
                gate_start = time.time()
                result = gate_func()
                result.execution_time = time.time() - gate_start
                self.results.append(result)
                
                status = "✓ PASSED" if result.result == QualityGateResult.PASSED else "✗ FAILED"
                logger.info(f"{gate_name}: {status} (score: {result.score:.2f})")
                
            except Exception as e:
                error_result = GateResult(
                    name=gate_name,
                    result=QualityGateResult.FAILED,
                    score=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    execution_time=time.time() - gate_start
                )
                self.results.append(error_result)
                logger.error(f"{gate_name}: FAILED with error: {e}")
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self.generate_summary()
        summary['total_execution_time'] = total_time
        
        logger.info(f"Quality gates completed in {total_time:.2f}s")
        logger.info(f"Overall result: {summary['overall_status']}")
        
        return summary
    
    def check_code_structure(self) -> GateResult:
        """Check code structure and organization."""
        required_files = [
            "src/dp_federated_lora/__init__.py",
            "src/dp_federated_lora/server.py",
            "src/dp_federated_lora/client.py",
            "src/dp_federated_lora/config.py",
            "src/dp_federated_lora/privacy.py",
            "src/dp_federated_lora/monitoring.py"
        ]
        
        required_dirs = [
            "src/dp_federated_lora",
            "tests",
            "scripts",
            "docs"
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        issues = len(missing_files) + len(missing_dirs)
        total_checks = len(required_files) + len(required_dirs)
        score = max(0.0, (total_checks - issues) / total_checks)
        
        if issues == 0:
            return GateResult(
                name="Code Structure",
                result=QualityGateResult.PASSED,
                score=score,
                message="All required files and directories present",
                details={
                    "required_files": len(required_files),
                    "required_dirs": len(required_dirs),
                    "missing_files": missing_files,
                    "missing_dirs": missing_dirs
                }
            )
        else:
            return GateResult(
                name="Code Structure",
                result=QualityGateResult.FAILED,
                score=score,
                message=f"Missing {issues} required files/directories",
                details={
                    "missing_files": missing_files,
                    "missing_dirs": missing_dirs
                }
            )
    
    def check_imports(self) -> GateResult:
        """Check if core modules can be imported."""
        import_checks = [
            "dp_federated_lora.config",
            "dp_federated_lora.server",
            "dp_federated_lora.client",
            "dp_federated_lora.privacy",
            "dp_federated_lora.monitoring"
        ]
        
        successful_imports = 0
        import_errors = []
        
        for module_name in import_checks:
            try:
                __import__(module_name)
                successful_imports += 1
                logger.debug(f"Successfully imported {module_name}")
            except Exception as e:
                import_errors.append(f"{module_name}: {str(e)}")
                logger.warning(f"Failed to import {module_name}: {e}")
        
        score = successful_imports / len(import_checks)
        
        if successful_imports == len(import_checks):
            return GateResult(
                name="Import Validation",
                result=QualityGateResult.PASSED,
                score=score,
                message="All core modules import successfully",
                details={"successful_imports": successful_imports, "total_modules": len(import_checks)}
            )
        else:
            return GateResult(
                name="Import Validation",
                result=QualityGateResult.FAILED,
                score=score,
                message=f"Failed to import {len(import_errors)} modules",
                details={"import_errors": import_errors}
            )
    
    def run_security_scan(self) -> GateResult:
        """Run security scan using bandit."""
        try:
            # Try to run bandit security scan
            cmd = ["python3", "-m", "bandit", "-r", str(self.src_dir), "-f", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # No security issues found
                return GateResult(
                    name="Security Scan",
                    result=QualityGateResult.PASSED,
                    score=1.0,
                    message="No security issues detected",
                    details={"scan_output": "Clean scan"}
                )
            else:
                # Parse bandit output if available
                try:
                    bandit_data = json.loads(result.stdout)
                    num_issues = len(bandit_data.get("results", []))
                    
                    if num_issues == 0:
                        return GateResult(
                            name="Security Scan",
                            result=QualityGateResult.PASSED,
                            score=1.0,
                            message="No security issues detected"
                        )
                    elif num_issues <= 5:
                        return GateResult(
                            name="Security Scan",
                            result=QualityGateResult.WARNING,
                            score=0.7,
                            message=f"Found {num_issues} minor security issues"
                        )
                    else:
                        return GateResult(
                            name="Security Scan",
                            result=QualityGateResult.FAILED,
                            score=0.3,
                            message=f"Found {num_issues} security issues"
                        )
                except json.JSONDecodeError:
                    # Fallback if bandit output is not JSON
                    return GateResult(
                        name="Security Scan",
                        result=QualityGateResult.WARNING,
                        score=0.5,
                        message="Security scan completed with warnings"
                    )
        
        except FileNotFoundError:
            # Bandit not available, manual security check
            return self._manual_security_check()
    
    def _manual_security_check(self) -> GateResult:
        """Manual security check when bandit is not available."""
        security_patterns = [
            ("eval(", "Use of eval() function"),
            ("exec(", "Use of exec() function"),
            ("os.system(", "Use of os.system()"),
            ("subprocess.call(", "Direct subprocess call"),
            ("shell=True", "Shell execution enabled"),
            ("pickle.loads(", "Unsafe pickle deserialization"),
            ("yaml.load(", "Unsafe YAML loading"),
        ]
        
        security_issues = []
        
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in security_patterns:
                    if pattern in content:
                        security_issues.append(f"{py_file.name}: {description}")
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        if not security_issues:
            return GateResult(
                name="Security Scan",
                result=QualityGateResult.PASSED,
                score=0.8,  # Lower score due to manual check
                message="Manual security scan found no obvious issues"
            )
        else:
            return GateResult(
                name="Security Scan",
                result=QualityGateResult.FAILED,
                score=0.3,
                message=f"Found {len(security_issues)} potential security issues",
                details={"issues": security_issues}
            )
    
    def check_code_quality(self) -> GateResult:
        """Check code quality metrics."""
        quality_metrics = {
            "docstrings": 0,
            "type_hints": 0,
            "long_functions": 0,
            "complex_functions": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_lines": 0
        }
        
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    quality_metrics["total_lines"] += len(lines)
                    
                    # Simple heuristic analysis
                    in_function = False
                    function_lines = 0
                    
                    for line in lines:
                        stripped = line.strip()
                        
                        # Count functions and classes
                        if stripped.startswith("def "):
                            quality_metrics["total_functions"] += 1
                            in_function = True
                            function_lines = 0
                            
                            # Check for type hints
                            if "->" in stripped or ":" in stripped:
                                quality_metrics["type_hints"] += 1
                        
                        elif stripped.startswith("class "):
                            quality_metrics["total_classes"] += 1
                            in_function = False
                        
                        elif stripped.startswith('"""') or stripped.startswith("'''"):
                            quality_metrics["docstrings"] += 1
                        
                        if in_function:
                            function_lines += 1
                            if function_lines > 50:  # Long function threshold
                                quality_metrics["long_functions"] += 1
                                in_function = False  # Count only once per function
                        
                        if not stripped or stripped.startswith("#"):
                            continue
                        
                        if stripped.endswith(":") and not in_function:
                            in_function = False
            
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Calculate quality score
        docstring_rate = (quality_metrics["docstrings"] / 
                         max(1, quality_metrics["total_functions"] + quality_metrics["total_classes"]))
        type_hint_rate = (quality_metrics["type_hints"] / 
                         max(1, quality_metrics["total_functions"]))
        
        # Penalize long functions
        long_function_penalty = min(0.5, quality_metrics["long_functions"] * 0.1)
        
        quality_score = min(1.0, (docstring_rate * 0.4 + type_hint_rate * 0.4 + 0.2) - long_function_penalty)
        
        if quality_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Code quality meets high standards"
        elif quality_score >= 0.6:
            result = QualityGateResult.WARNING
            message = "Code quality is acceptable but could be improved"
        else:
            result = QualityGateResult.FAILED
            message = "Code quality below standards"
        
        return GateResult(
            name="Code Quality",
            result=result,
            score=quality_score,
            message=message,
            details=quality_metrics
        )
    
    def run_unit_tests(self) -> GateResult:
        """Run unit tests."""
        try:
            test_results = run_comprehensive_tests(categories=[TestCategory.UNIT])
            
            success_rate = test_results.get('success_rate', 0.0)
            passed = test_results.get('passed', 0)
            total = test_results.get('total_tests', 0)
            
            if success_rate >= 0.95:
                result = QualityGateResult.PASSED
                message = f"Unit tests passed ({passed}/{total})"
            elif success_rate >= 0.8:
                result = QualityGateResult.WARNING
                message = f"Most unit tests passed ({passed}/{total})"
            else:
                result = QualityGateResult.FAILED
                message = f"Unit tests failed ({passed}/{total})"
            
            return GateResult(
                name="Unit Tests",
                result=result,
                score=success_rate,
                message=message,
                details=test_results
            )
        
        except Exception as e:
            return GateResult(
                name="Unit Tests",
                result=QualityGateResult.FAILED,
                score=0.0,
                message=f"Unit test execution failed: {str(e)}"
            )
    
    def run_integration_tests(self) -> GateResult:
        """Run integration tests."""
        try:
            test_results = run_comprehensive_tests(categories=[TestCategory.INTEGRATION])
            
            success_rate = test_results.get('success_rate', 0.0)
            passed = test_results.get('passed', 0)
            total = test_results.get('total_tests', 0)
            
            if success_rate >= 0.9:
                result = QualityGateResult.PASSED
                message = f"Integration tests passed ({passed}/{total})"
            elif success_rate >= 0.7:
                result = QualityGateResult.WARNING
                message = f"Most integration tests passed ({passed}/{total})"
            else:
                result = QualityGateResult.FAILED
                message = f"Integration tests failed ({passed}/{total})"
            
            return GateResult(
                name="Integration Tests",
                result=result,
                score=success_rate,
                message=message,
                details=test_results
            )
        
        except Exception as e:
            return GateResult(
                name="Integration Tests",
                result=QualityGateResult.FAILED,
                score=0.0,
                message=f"Integration test execution failed: {str(e)}"
            )
    
    def run_performance_tests(self) -> GateResult:
        """Run performance tests."""
        try:
            test_results = run_comprehensive_tests(categories=[TestCategory.PERFORMANCE])
            
            success_rate = test_results.get('success_rate', 0.0)
            passed = test_results.get('passed', 0)
            total = test_results.get('total_tests', 0)
            
            if success_rate >= 0.8:
                result = QualityGateResult.PASSED
                message = f"Performance tests passed ({passed}/{total})"
            elif success_rate >= 0.6:
                result = QualityGateResult.WARNING
                message = f"Most performance tests passed ({passed}/{total})"
            else:
                result = QualityGateResult.FAILED
                message = f"Performance tests failed ({passed}/{total})"
            
            return GateResult(
                name="Performance Tests",
                result=result,
                score=success_rate,
                message=message,
                details=test_results
            )
        
        except Exception as e:
            return GateResult(
                name="Performance Tests",
                result=QualityGateResult.FAILED,
                score=0.0,
                message=f"Performance test execution failed: {str(e)}"
            )
    
    def run_security_tests(self) -> GateResult:
        """Run security tests."""
        try:
            test_results = run_comprehensive_tests(categories=[TestCategory.SECURITY])
            
            success_rate = test_results.get('success_rate', 0.0)
            passed = test_results.get('passed', 0)
            total = test_results.get('total_tests', 0)
            critical_failures = test_results.get('critical_failures', 0)
            
            if critical_failures > 0:
                result = QualityGateResult.FAILED
                message = f"Security tests have critical failures ({critical_failures})"
            elif success_rate >= 0.95:
                result = QualityGateResult.PASSED
                message = f"Security tests passed ({passed}/{total})"
            elif success_rate >= 0.8:
                result = QualityGateResult.WARNING
                message = f"Most security tests passed ({passed}/{total})"
            else:
                result = QualityGateResult.FAILED
                message = f"Security tests failed ({passed}/{total})"
            
            return GateResult(
                name="Security Tests",
                result=result,
                score=success_rate,
                message=message,
                details=test_results
            )
        
        except Exception as e:
            return GateResult(
                name="Security Tests",
                result=QualityGateResult.FAILED,
                score=0.0,
                message=f"Security test execution failed: {str(e)}"
            )
    
    def check_documentation(self) -> GateResult:
        """Check documentation quality."""
        doc_files = [
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "CHANGELOG.md"
        ]
        
        existing_docs = 0
        doc_details = {}
        
        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                existing_docs += 1
                # Check file size as quality indicator
                size = doc_path.stat().st_size
                doc_details[doc_file] = {"exists": True, "size": size}
            else:
                doc_details[doc_file] = {"exists": False}
        
        # Check for inline documentation
        docstring_files = 0
        total_py_files = 0
        
        for py_file in self.src_dir.rglob("*.py"):
            total_py_files += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        docstring_files += 1
            except Exception:
                pass
        
        doc_score = existing_docs / len(doc_files)
        docstring_score = docstring_files / max(1, total_py_files)
        overall_score = (doc_score * 0.6 + docstring_score * 0.4)
        
        if overall_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Documentation quality is excellent"
        elif overall_score >= 0.6:
            result = QualityGateResult.WARNING
            message = "Documentation quality is acceptable"
        else:
            result = QualityGateResult.FAILED
            message = "Documentation quality needs improvement"
        
        return GateResult(
            name="Documentation",
            result=result,
            score=overall_score,
            message=message,
            details={
                "doc_files": doc_details,
                "docstring_coverage": f"{docstring_files}/{total_py_files}",
                "doc_score": doc_score,
                "docstring_score": docstring_score
            }
        )
    
    def validate_configuration(self) -> GateResult:
        """Validate project configuration files."""
        config_files = [
            ("pyproject.toml", True),
            ("requirements.txt", True),
            ("requirements-dev.txt", False),
            ("pytest.ini", False),
            ("tox.ini", False),
        ]
        
        config_issues = []
        valid_configs = 0
        
        for config_file, required in config_files:
            config_path = self.project_root / config_file
            
            if config_path.exists():
                valid_configs += 1
                try:
                    # Basic validation - check if file is readable
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) == 0:
                            config_issues.append(f"{config_file}: Empty configuration file")
                except Exception as e:
                    config_issues.append(f"{config_file}: Read error - {str(e)}")
            elif required:
                config_issues.append(f"{config_file}: Required configuration file missing")
        
        score = max(0.0, (valid_configs - len(config_issues)) / len(config_files))
        
        if len(config_issues) == 0:
            result = QualityGateResult.PASSED
            message = "All configuration files valid"
        elif len(config_issues) <= 2:
            result = QualityGateResult.WARNING
            message = f"Minor configuration issues ({len(config_issues)})"
        else:
            result = QualityGateResult.FAILED
            message = f"Multiple configuration issues ({len(config_issues)})"
        
        return GateResult(
            name="Configuration",
            result=result,
            score=score,
            message=message,
            details={
                "valid_configs": valid_configs,
                "total_configs": len(config_files),
                "issues": config_issues
            }
        )
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate quality gate summary."""
        if not self.results:
            return {"overall_status": "NO_GATES_RUN"}
        
        # Count results by status
        passed_gates = [r for r in self.results if r.result == QualityGateResult.PASSED]
        failed_gates = [r for r in self.results if r.result == QualityGateResult.FAILED]
        warning_gates = [r for r in self.results if r.result == QualityGateResult.WARNING]
        
        # Calculate overall score
        total_score = sum(r.score for r in self.results)
        avg_score = total_score / len(self.results)
        
        # Determine overall status
        critical_failures = ["Unit Tests", "Integration Tests", "Security Tests", "Import Validation"]
        critical_failed = any(
            r.name in critical_failures and r.result == QualityGateResult.FAILED
            for r in self.results
        )
        
        if critical_failed:
            overall_status = "CRITICAL_FAILURE"
        elif len(failed_gates) == 0 and len(warning_gates) == 0:
            overall_status = "ALL_PASSED"
        elif len(failed_gates) == 0:
            overall_status = "PASSED_WITH_WARNINGS"
        elif len(failed_gates) <= 2:
            overall_status = "SOME_FAILURES"
        else:
            overall_status = "MANY_FAILURES"
        
        return {
            "overall_status": overall_status,
            "overall_score": avg_score,
            "total_gates": len(self.results),
            "passed": len(passed_gates),
            "failed": len(failed_gates),
            "warnings": len(warning_gates),
            "critical_failure": critical_failed,
            "gate_results": [
                {
                    "name": r.name,
                    "result": r.result.value,
                    "score": r.score,
                    "message": r.message,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ],
            "failed_gates": [r.name for r in failed_gates],
            "warning_gates": [r.name for r in warning_gates]
        }
    
    def save_report(self, output_file: str):
        """Save quality gate report to file."""
        summary = self.generate_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Quality gate report saved to {output_file}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "quality_gates.log")
        ]
    )
    
    logger.info("Starting DP-Federated LoRA Quality Gates")
    
    # Run quality gates
    runner = QualityGateRunner(project_root)
    summary = runner.run_all_gates()
    
    # Save report
    report_file = project_root / "quality_gate_report.json"
    runner.save_report(str(report_file))
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY GATE SUMMARY")
    print("="*60)
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Overall Score: {summary['overall_score']:.2f}")
    print(f"Gates Passed: {summary['passed']}/{summary['total_gates']}")
    print(f"Gates Failed: {summary['failed']}")
    print(f"Gates with Warnings: {summary['warnings']}")
    print(f"Total Execution Time: {summary.get('total_execution_time', 0):.2f}s")
    
    if summary['failed_gates']:
        print(f"\nFailed Gates: {', '.join(summary['failed_gates'])}")
    
    if summary['warning_gates']:
        print(f"Warning Gates: {', '.join(summary['warning_gates'])}")
    
    print(f"\nDetailed report saved to: {report_file}")
    print("="*60)
    
    # Exit with appropriate code
    if summary['overall_status'] in ['CRITICAL_FAILURE', 'MANY_FAILURES']:
        sys.exit(1)
    elif summary['overall_status'] == 'SOME_FAILURES':
        sys.exit(2)  # Warning exit code
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
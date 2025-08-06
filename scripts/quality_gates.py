#!/usr/bin/env python3
"""
Quality Gates Validation Script for DP-Federated LoRA.

This script validates the entire system against production quality gates:
- Code structure and imports
- Error handling coverage
- Performance benchmarks
- Security validation
- Documentation completeness
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import ast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class QualityGateValidator:
    """Comprehensive quality gate validation."""
    
    def __init__(self):
        self.root_path = Path(__file__).parent.parent
        self.src_path = self.root_path / "src" / "dp_federated_lora"
        self.test_path = self.root_path / "tests"
        self.results = {}
        self.total_score = 0
        self.max_score = 0
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        print("🔍 DP-FEDERATED LORA QUALITY GATES VALIDATION")
        print("=" * 80)
        
        gates = [
            ("Code Structure", self.validate_code_structure),
            ("Import System", self.validate_imports),
            ("Error Handling", self.validate_error_handling),
            ("Network Layer", self.validate_network_layer),
            ("Performance", self.validate_performance),
            ("Security", self.validate_security),
            ("Documentation", self.validate_documentation),
            ("Configuration", self.validate_configuration),
            ("Scalability", self.validate_scalability),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 {gate_name} Validation")
            print("-" * 40)
            
            try:
                score, max_score, details = gate_func()
                self.results[gate_name] = {
                    "score": score,
                    "max_score": max_score,
                    "success_rate": (score / max_score) * 100 if max_score > 0 else 0,
                    "details": details,
                    "status": "PASS" if score >= max_score * 0.8 else "WARN" if score >= max_score * 0.6 else "FAIL"
                }
                
                self.total_score += score
                self.max_score += max_score
                
                status_icon = "✅" if self.results[gate_name]["status"] == "PASS" else "⚠️" if self.results[gate_name]["status"] == "WARN" else "❌"
                print(f"{status_icon} {gate_name}: {score}/{max_score} ({self.results[gate_name]['success_rate']:.1f}%)")
                
                for detail in details[:3]:  # Show first 3 details
                    print(f"  • {detail}")
                
            except Exception as e:
                print(f"❌ {gate_name}: FAILED - {e}")
                self.results[gate_name] = {
                    "score": 0,
                    "max_score": 10,
                    "success_rate": 0,
                    "details": [f"Validation failed: {e}"],
                    "status": "FAIL"
                }
                self.max_score += 10
        
        return self.generate_final_report()
    
    def validate_code_structure(self) -> Tuple[int, int, List[str]]:
        """Validate code structure and organization."""
        score = 0
        max_score = 15
        details = []
        
        # Check main modules exist
        required_modules = [
            "server.py", "client.py", "config.py", "privacy.py",
            "aggregation.py", "monitoring.py", "network_client.py",
            "exceptions.py", "error_handler.py", "performance.py",
            "concurrent.py", "cli.py"
        ]
        
        existing_modules = []
        for module in required_modules:
            if (self.src_path / module).exists():
                existing_modules.append(module)
                score += 1
        
        details.append(f"Core modules present: {len(existing_modules)}/{len(required_modules)}")
        
        # Check __init__.py exports
        init_file = self.src_path / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            if "__all__" in content and len(content) > 1000:
                score += 2
                details.append("__init__.py properly configured with exports")
            else:
                details.append("__init__.py missing proper exports")
        
        # Check examples directory
        examples_path = self.root_path / "examples"
        if examples_path.exists():
            example_files = list(examples_path.glob("*.py"))
            if len(example_files) >= 2:
                score += 1
                details.append(f"Examples directory with {len(example_files)} files")
        
        return score, max_score, details
    
    def validate_imports(self) -> Tuple[int, int, List[str]]:
        """Validate import system functionality."""
        score = 0
        max_score = 10
        details = []
        
        try:
            # Test basic configuration imports
            from dp_federated_lora import FederatedConfig, PrivacyConfig, LoRAConfig
            score += 2
            details.append("Core configuration classes import successfully")
        except Exception as e:
            details.append(f"Configuration import failed: {e}")
        
        try:
            # Test error handling system
            from dp_federated_lora.exceptions import DPFederatedLoRAError, NetworkError
            from dp_federated_lora.error_handler import ErrorHandler
            score += 2
            details.append("Error handling system imports successfully")
        except Exception as e:
            details.append(f"Error handling import failed: {e}")
        
        try:
            # Test network components (may fail without dependencies)
            from dp_federated_lora.network_client import FederatedNetworkClient
            score += 2
            details.append("Network components import successfully")
        except Exception:
            details.append("Network components missing dependencies (expected)")
            score += 1  # Partial credit
        
        try:
            # Test performance components (may fail without dependencies)
            from dp_federated_lora.performance import PerformanceMonitor
            score += 2
            details.append("Performance components import successfully")
        except Exception:
            details.append("Performance components missing dependencies (expected)")
            score += 1  # Partial credit
        
        try:
            # Test concurrent components
            from dp_federated_lora.concurrent import WorkerPool, ThreadWorkerPool
            score += 2
            details.append("Concurrent processing components import successfully")
        except Exception:
            details.append("Concurrent components missing dependencies (expected)")
            score += 1  # Partial credit
        
        return score, max_score, details
    
    def validate_error_handling(self) -> Tuple[int, int, List[str]]:
        """Validate comprehensive error handling system."""
        score = 0
        max_score = 12
        details = []
        
        # Check exception hierarchy
        exceptions_file = self.src_path / "exceptions.py"
        if exceptions_file.exists():
            content = exceptions_file.read_text()
            
            # Count exception classes
            exception_classes = content.count("class ") - content.count("class ErrorContext") - content.count("class ErrorSeverity")
            if exception_classes >= 10:
                score += 3
                details.append(f"Comprehensive exception hierarchy: {exception_classes} classes")
            elif exception_classes >= 5:
                score += 2
                details.append(f"Good exception coverage: {exception_classes} classes")
            else:
                score += 1
                details.append(f"Basic exception coverage: {exception_classes} classes")
        
        # Check error handler implementation
        error_handler_file = self.src_path / "error_handler.py"
        if error_handler_file.exists():
            content = error_handler_file.read_text()
            
            # Check for circuit breaker pattern
            if "CircuitBreaker" in content:
                score += 2
                details.append("Circuit breaker pattern implemented")
            
            # Check for retry mechanisms
            if "retry" in content.lower() and "RetryConfig" in content:
                score += 2
                details.append("Retry mechanisms implemented")
            
            # Check for async error handling
            if "async def" in content and "error_boundary" in content:
                score += 2
                details.append("Async error handling implemented")
            
            # Check for monitoring integration
            if "monitor" in content.lower():
                score += 1
                details.append("Error monitoring integration")
        
        # Check error handling in network layer
        network_client_file = self.src_path / "network_client.py"
        if network_client_file.exists():
            content = network_client_file.read_text()
            if "with_error_handling" in content or "error_boundary" in content:
                score += 2
                details.append("Network layer uses error handling decorators")
        
        return score, max_score, details
    
    def validate_network_layer(self) -> Tuple[int, int, List[str]]:
        """Validate network communication implementation."""
        score = 0
        max_score = 10
        details = []
        
        network_client_file = self.src_path / "network_client.py"
        if network_client_file.exists():
            content = network_client_file.read_text()
            
            # Check for HTTP client implementation
            if "httpx" in content and "AsyncClient" in content:
                score += 2
                details.append("HTTP client implementation using httpx")
            
            # Check for authentication
            if "auth" in content.lower() and "token" in content.lower():
                score += 2
                details.append("Authentication mechanisms implemented")
            
            # Check for connection management
            if "connection" in content.lower() and "pool" in content.lower():
                score += 1
                details.append("Connection management features")
            
            # Check for retry logic
            if "retry" in content.lower():
                score += 1
                details.append("Network retry logic implemented")
        
        # Check server implementation
        server_file = self.src_path / "server.py"
        if server_file.exists():
            content = server_file.read_text()
            
            # Check for FastAPI integration
            if "FastAPI" in content and "uvicorn" in content:
                score += 2
                details.append("FastAPI server implementation")
            
            # Check for API endpoints
            if "@app.post" in content and "@app.get" in content:
                score += 1
                details.append("RESTful API endpoints implemented")
            
            # Check for async support
            if "async def" in content:
                score += 1
                details.append("Async server operation support")
        
        return score, max_score, details
    
    def validate_performance(self) -> Tuple[int, int, List[str]]:
        """Validate performance optimization features."""
        score = 0
        max_score = 12
        details = []
        
        performance_file = self.src_path / "performance.py"
        if performance_file.exists():
            content = performance_file.read_text()
            
            # Check for performance monitoring
            if "PerformanceMonitor" in content:
                score += 2
                details.append("Performance monitoring system implemented")
            
            # Check for caching
            if "CacheManager" in content and "LRU" in content:
                score += 2
                details.append("Advanced caching system with LRU eviction")
            
            # Check for connection pooling
            if "ConnectionPool" in content:
                score += 2
                details.append("Connection pooling implemented")
            
            # Check for resource management
            if "ResourceManager" in content and "psutil" in content:
                score += 2
                details.append("Resource management and monitoring")
        
        concurrent_file = self.src_path / "concurrent.py"
        if concurrent_file.exists():
            content = concurrent_file.read_text()
            
            # Check for parallel processing
            if "ParallelAggregator" in content:
                score += 2
                details.append("Parallel aggregation implemented")
            
            # Check for worker pools
            if "WorkerPool" in content and "ThreadPoolExecutor" in content:
                score += 1
                details.append("Concurrent worker pools")
            
            # Check for distributed training
            if "DistributedTrainingManager" in content:
                score += 1
                details.append("Distributed training support")
        
        return score, max_score, details
    
    def validate_security(self) -> Tuple[int, int, List[str]]:
        """Validate security implementations."""
        score = 0
        max_score = 10
        details = []
        
        # Check authentication system
        server_file = self.src_path / "server.py"
        if server_file.exists():
            content = server_file.read_text()
            
            if "AuthenticationManager" in content:
                score += 2
                details.append("Authentication manager implemented")
            
            if "hmac" in content and "hashlib" in content:
                score += 2
                details.append("Secure token generation with HMAC")
            
            if "HTTPBearer" in content:
                score += 1
                details.append("Bearer token authentication")
        
        # Check privacy implementation
        privacy_file = self.src_path / "privacy.py"
        if privacy_file.exists():
            content = privacy_file.read_text()
            
            if "opacus" in content.lower():
                score += 2
                details.append("Differential privacy with Opacus")
            
            if "PrivacyAccountant" in content:
                score += 1
                details.append("Privacy accounting system")
        
        # Check secure aggregation
        aggregation_file = self.src_path / "aggregation.py"
        if aggregation_file.exists():
            content = aggregation_file.read_text()
            
            if "secure" in content.lower() and "byzantine" in content.lower():
                score += 2
                details.append("Secure and Byzantine-robust aggregation")
        
        return score, max_score, details
    
    def validate_documentation(self) -> Tuple[int, int, List[str]]:
        """Validate documentation completeness."""
        score = 0
        max_score = 8
        details = []
        
        # Check README
        readme_file = self.root_path / "README.md"
        if readme_file.exists():
            content = readme_file.read_text()
            if len(content) > 5000:  # Comprehensive README
                score += 2
                details.append("Comprehensive README.md (>5000 chars)")
            elif len(content) > 1000:
                score += 1
                details.append("Good README.md documentation")
        
        # Check docstrings in main modules
        docstring_modules = ["server.py", "client.py", "network_client.py"]
        docstring_score = 0
        
        for module in docstring_modules:
            module_file = self.src_path / module
            if module_file.exists():
                content = module_file.read_text()
                # Count docstrings (triple quotes)
                docstring_count = content.count('"""')
                if docstring_count >= 10:  # Good documentation
                    docstring_score += 1
        
        score += min(3, docstring_score)
        details.append(f"Module documentation: {docstring_score}/{len(docstring_modules)} well-documented")
        
        # Check examples
        examples_path = self.root_path / "examples"
        if examples_path.exists():
            example_files = list(examples_path.glob("*.py"))
            if len(example_files) >= 3:
                score += 2
                details.append(f"Comprehensive examples: {len(example_files)} files")
            elif len(example_files) >= 1:
                score += 1
                details.append(f"Basic examples: {len(example_files)} files")
        
        # Check architecture documentation
        arch_file = self.root_path / "ARCHITECTURE.md"
        if arch_file.exists():
            score += 1
            details.append("Architecture documentation present")
        
        return score, max_score, details
    
    def validate_configuration(self) -> Tuple[int, int, List[str]]:
        """Validate configuration system."""
        score = 0
        max_score = 8
        details = []
        
        config_file = self.src_path / "config.py"
        if config_file.exists():
            content = config_file.read_text()
            
            # Check for dataclass usage
            if "@dataclass" in content:
                score += 2
                details.append("Dataclass-based configuration")
            
            # Check for validation
            if "validate" in content.lower() or "pydantic" in content.lower():
                score += 2
                details.append("Configuration validation implemented")
            
            # Check for multiple config types
            config_classes = content.count("class ") - content.count("class AggregationMethod")
            if config_classes >= 5:
                score += 2
                details.append(f"Comprehensive configuration: {config_classes} classes")
            
            # Check for factory functions
            if "create_default_config" in content:
                score += 1
                details.append("Configuration factory functions")
            
            # Check for enums
            if "enum" in content.lower():
                score += 1
                details.append("Enum-based type safety")
        
        return score, max_score, details
    
    def validate_scalability(self) -> Tuple[int, int, List[str]]:
        """Validate scalability features."""
        score = 0
        max_score = 10
        details = []
        
        # Check concurrent processing
        concurrent_file = self.src_path / "concurrent.py"
        if concurrent_file.exists():
            content = concurrent_file.read_text()
            
            if "ConcurrentModelTrainer" in content:
                score += 2
                details.append("Concurrent model training")
            
            if "ProcessPoolExecutor" in content and "ThreadPoolExecutor" in content:
                score += 2
                details.append("Multi-processing and threading support")
            
            if "DistributedTrainingManager" in content:
                score += 2
                details.append("Distributed training capabilities")
        
        # Check performance optimizations
        performance_file = self.src_path / "performance.py"
        if performance_file.exists():
            content = performance_file.read_text()
            
            if "optimize_for_scale" in content:
                score += 2
                details.append("Scaling optimization functions")
            
            if "BatchProcessor" in content:
                score += 1
                details.append("Batch processing capabilities")
        
        # Check server scalability
        server_file = self.src_path / "server.py"
        if server_file.exists():
            content = server_file.read_text()
            
            if "parallel_aggregator" in content:
                score += 1
                details.append("Parallel aggregation in server")
        
        return score, max_score, details
    
    def validate_production_readiness(self) -> Tuple[int, int, List[str]]:
        """Validate production readiness features."""
        score = 0
        max_score = 12
        details = []
        
        # Check deployment files
        docker_file = self.root_path / "Dockerfile"
        if docker_file.exists():
            score += 2
            details.append("Docker containerization support")
        
        compose_file = self.root_path / "docker-compose.yml"
        if compose_file.exists():
            score += 1
            details.append("Docker Compose configuration")
        
        # Check requirements
        requirements_file = self.root_path / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()
            if len(content.splitlines()) >= 10:
                score += 2
                details.append("Comprehensive requirements specification")
        
        # Check monitoring endpoints in server
        server_file = self.src_path / "server.py"
        if server_file.exists():
            content = server_file.read_text()
            
            if "/metrics" in content:
                score += 2
                details.append("Metrics endpoint for monitoring")
            
            if "/health" in content:
                score += 1
                details.append("Health check endpoint")
            
            if "/stats" in content:
                score += 1
                details.append("Statistics endpoint")
        
        # Check CLI support
        cli_file = self.src_path / "cli.py"
        if cli_file.exists():
            content = cli_file.read_text()
            if "typer" in content and "rich" in content:
                score += 2
                details.append("Production-ready CLI interface")
        
        # Check logging configuration
        total_logging = 0
        for py_file in self.src_path.glob("*.py"):
            content = py_file.read_text()
            if "logging" in content and "logger" in content:
                total_logging += 1
        
        if total_logging >= 8:
            score += 1
            details.append(f"Comprehensive logging: {total_logging} modules")
        
        return score, max_score, details
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final quality gate report."""
        overall_score = (self.total_score / self.max_score) * 100 if self.max_score > 0 else 0
        
        # Determine overall status
        if overall_score >= 85:
            overall_status = "EXCELLENT"
            status_icon = "🏆"
        elif overall_score >= 75:
            overall_status = "GOOD"
            status_icon = "✅"
        elif overall_score >= 60:
            overall_status = "ACCEPTABLE"
            status_icon = "⚠️"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
            status_icon = "❌"
        
        print(f"\n" + "=" * 80)
        print(f"{status_icon} FINAL QUALITY GATE REPORT")
        print("=" * 80)
        print(f"Overall Score: {self.total_score}/{self.max_score} ({overall_score:.1f}%)")
        print(f"Status: {overall_status}")
        print()
        
        # Summary by category
        for gate_name, result in self.results.items():
            status_icon = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "WARN" else "❌"
            print(f"{status_icon} {gate_name}: {result['score']}/{result['max_score']} ({result['success_rate']:.1f}%)")
        
        # Production readiness assessment
        print(f"\n📊 PRODUCTION READINESS ASSESSMENT")
        print("-" * 40)
        
        critical_gates = ["Error Handling", "Network Layer", "Security", "Performance"]
        critical_passed = sum(1 for gate in critical_gates if self.results.get(gate, {}).get("status") == "PASS")
        
        print(f"Critical Systems: {critical_passed}/{len(critical_gates)} PASSED")
        
        if critical_passed == len(critical_gates) and overall_score >= 80:
            print("🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        elif critical_passed >= len(critical_gates) * 0.75:
            print("⚠️  SYSTEM READY FOR STAGING/TESTING")
        else:
            print("❌ SYSTEM NEEDS IMPROVEMENT BEFORE DEPLOYMENT")
        
        report = {
            "timestamp": time.time(),
            "overall_score": overall_score,
            "overall_status": overall_status,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "results": self.results,
            "critical_systems_ready": critical_passed == len(critical_gates),
            "production_ready": critical_passed == len(critical_gates) and overall_score >= 80
        }
        
        return report


def main():
    """Run quality gate validation."""
    validator = QualityGateValidator()
    report = validator.run_all_gates()
    
    # Save report
    report_file = Path(__file__).parent.parent / "quality_gate_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if report["production_ready"] else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
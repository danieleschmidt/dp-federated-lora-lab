#!/usr/bin/env python3
"""
Test runner script for dp-federated-lora-lab.

This script provides various test execution modes for different scenarios.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd)


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd)


def run_e2e_tests(verbose: bool = False) -> int:
    """Run end-to-end tests."""
    cmd = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # E2E tests can be slow
    cmd.extend(["--timeout=600", "--tb=short"])
    
    return run_command(cmd)


def run_privacy_tests(verbose: bool = False) -> int:
    """Run privacy-specific tests."""
    cmd = ["python", "-m", "pytest", "-m", "privacy"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd)


def run_fast_tests(verbose: bool = False) -> int:
    """Run fast tests (excluding slow, gpu, e2e)."""
    cmd = [
        "python", "-m", "pytest", 
        "-m", "not slow and not gpu and not e2e",
        "--maxfail=5"
    ]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd)


def run_gpu_tests(verbose: bool = False) -> int:
    """Run GPU tests."""
    cmd = ["python", "-m", "pytest", "-m", "gpu"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Check if GPU is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  GPU not available, skipping GPU tests")
            return 0
    except ImportError:
        print("⚠️  PyTorch not available, skipping GPU tests")
        return 0
    
    return run_command(cmd)


def run_all_tests(verbose: bool = False, include_slow: bool = False) -> int:
    """Run all tests."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if not include_slow:
        cmd.extend(["-m", "not slow"])
    
    cmd.extend([
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--durations=10"
    ])
    
    return run_command(cmd)


def run_security_tests(verbose: bool = False) -> int:
    """Run security-related tests."""
    cmd = ["python", "-m", "pytest", "-m", "security"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd)


def run_benchmarks(verbose: bool = False) -> int:
    """Run benchmark tests."""
    cmd = ["python", "-m", "pytest", "-m", "benchmark", "--benchmark-only"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    return run_command(cmd)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for dp-federated-lora-lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py unit                    # Run unit tests
  python scripts/run_tests.py integration             # Run integration tests
  python scripts/run_tests.py all --verbose           # Run all tests with verbose output
  python scripts/run_tests.py fast                    # Run fast tests only
  python scripts/run_tests.py privacy                 # Run privacy tests
  python scripts/run_tests.py gpu                     # Run GPU tests
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "e2e", "privacy", "security", 
            "fast", "gpu", "all", "benchmark"
        ],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting (for unit tests)"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests (for 'all' test type)"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    # Run appropriate tests
    if args.test_type == "unit":
        exit_code = run_unit_tests(args.verbose, not args.no_coverage)
    elif args.test_type == "integration":
        exit_code = run_integration_tests(args.verbose)
    elif args.test_type == "e2e":
        exit_code = run_e2e_tests(args.verbose)
    elif args.test_type == "privacy":
        exit_code = run_privacy_tests(args.verbose)
    elif args.test_type == "security":
        exit_code = run_security_tests(args.verbose)
    elif args.test_type == "fast":
        exit_code = run_fast_tests(args.verbose)
    elif args.test_type == "gpu":
        exit_code = run_gpu_tests(args.verbose)
    elif args.test_type == "benchmark":
        exit_code = run_benchmarks(args.verbose)
    elif args.test_type == "all":
        exit_code = run_all_tests(args.verbose, args.include_slow)
    else:
        print(f"Unknown test type: {args.test_type}")
        exit_code = 1
    
    # Report results
    if exit_code == 0:
        print(f"✅ {args.test_type.title()} tests passed!")
    else:
        print(f"❌ {args.test_type.title()} tests failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
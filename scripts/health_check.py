#!/usr/bin/env python3
"""
Health check script for DP-Federated LoRA Lab.
Validates system readiness, dependencies, and core functionality.
"""

import json
import sys
import time
import traceback
from typing import Dict, List, Tuple
import warnings

# Suppress warnings during health checks
warnings.filterwarnings('ignore')

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    try:
        if sys.version_info < (3, 9):
            return False, f"Python {sys.version} < 3.9 (required)"
        return True, f"Python {sys.version} ✅"
    except Exception as e:
        return False, f"Python version check failed: {e}"

def check_core_imports() -> Tuple[bool, str]:
    """Check if core package imports work."""
    try:
        sys.path.insert(0, 'src')
        import dp_federated_lora
        return True, "Core package imports ✅"
    except Exception as e:
        return False, f"Core imports failed: {e}"

def check_ml_dependencies() -> Tuple[bool, str]:
    """Check ML framework dependencies."""
    missing = []
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        missing.append("torch")
        torch_version = "missing"
    
    try:
        import transformers
        hf_version = transformers.__version__
    except ImportError:
        missing.append("transformers")
        hf_version = "missing"
    
    try:
        import datasets
        ds_version = datasets.__version__
    except ImportError:
        missing.append("datasets")  
        ds_version = "missing"
    
    if missing:
        return False, f"Missing ML deps: {missing}"
    
    return True, f"ML stack: torch={torch_version}, transformers={hf_version}, datasets={ds_version} ✅"

def check_privacy_dependencies() -> Tuple[bool, str]:
    """Check differential privacy dependencies."""
    try:
        import opacus
        return True, f"Opacus {opacus.__version__} ✅"
    except ImportError:
        return False, "Opacus (DP framework) missing"

def check_network_dependencies() -> Tuple[bool, str]:
    """Check network and API dependencies."""
    missing = []
    try:
        import fastapi
        fastapi_version = fastapi.__version__
    except ImportError:
        missing.append("fastapi")
        fastapi_version = "missing"
        
    try:
        import httpx
        httpx_version = httpx.__version__
    except ImportError:
        missing.append("httpx")
        httpx_version = "missing"
    
    if missing:
        return False, f"Missing network deps: {missing}"
    
    return True, f"Network: FastAPI={fastapi_version}, httpx={httpx_version} ✅"

def check_security_features() -> Tuple[bool, str]:
    """Check security and cryptography features."""
    try:
        import cryptography
        from cryptography.fernet import Fernet
        # Test key generation
        key = Fernet.generate_key()
        cipher = Fernet(key)
        test_data = b"test_encryption"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)
        
        if decrypted == test_data:
            return True, f"Cryptography {cryptography.__version__} ✅"
        else:
            return False, "Encryption/decryption test failed"
    except Exception as e:
        return False, f"Security check failed: {e}"

def check_file_structure() -> Tuple[bool, str]:
    """Check if required files and directories exist."""
    import os
    required_paths = [
        'src/dp_federated_lora',
        'src/dp_federated_lora/__init__.py',
        'src/dp_federated_lora/server.py',
        'src/dp_federated_lora/client.py',
        'tests/',
        'pyproject.toml',
        '.github/workflows',
    ]
    
    missing = [path for path in required_paths if not os.path.exists(path)]
    
    if missing:
        return False, f"Missing paths: {missing}"
    
    return True, f"File structure complete ✅"

def check_configuration() -> Tuple[bool, str]:
    """Check configuration files and settings."""
    try:
        # Try to import config module
        sys.path.insert(0, 'src')
        from dp_federated_lora import config
        return True, "Configuration module ✅"
    except Exception as e:
        return False, f"Configuration check failed: {e}"

def run_health_checks() -> Dict[str, Tuple[bool, str]]:
    """Run all health checks and return results."""
    checks = {
        'python_version': check_python_version,
        'core_imports': check_core_imports,
        'ml_dependencies': check_ml_dependencies,
        'privacy_dependencies': check_privacy_dependencies,
        'network_dependencies': check_network_dependencies,
        'security_features': check_security_features,
        'file_structure': check_file_structure,
        'configuration': check_configuration,
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            success, message = check_func()
            results[check_name] = (success, message)
        except Exception as e:
            results[check_name] = (False, f"Check failed with exception: {e}")
    
    return results

def main():
    """Main health check execution."""
    print("🏥 DP-Federated LoRA Health Check")
    print("=" * 50)
    
    start_time = time.time()
    results = run_health_checks()
    end_time = time.time()
    
    # Display results
    passed = 0
    total = len(results)
    
    for check_name, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {check_name:20} | {message}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Results: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    print(f"Duration: {end_time - start_time:.2f}s")
    
    # Create health report
    health_report = {
        'timestamp': time.time(),
        'total_checks': total,
        'passed_checks': passed,
        'success_rate': passed / total * 100,
        'duration_seconds': end_time - start_time,
        'details': {name: {'success': success, 'message': message} 
                   for name, (success, message) in results.items()},
    }
    
    # Save health report
    with open('health_report.json', 'w') as f:
        json.dump(health_report, f, indent=2)
    
    print(f"📊 Health report saved to health_report.json")
    
    # Exit with error code if any checks failed
    if passed < total:
        print(f"\n⚠️  {total - passed} health check(s) failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All health checks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
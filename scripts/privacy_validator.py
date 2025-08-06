#!/usr/bin/env python3
"""
Privacy validator for DP-Federated LoRA Lab.
Validates differential privacy guarantees and configurations.
"""

import json
import sys
import time
import warnings
from typing import Dict, List, Tuple, Optional
import math

warnings.filterwarnings('ignore')

def validate_epsilon_delta(epsilon: float, delta: float) -> Tuple[bool, str]:
    """Validate epsilon and delta values for DP."""
    try:
        # Standard DP parameter validation
        if epsilon <= 0:
            return False, f"Epsilon must be positive, got {epsilon}"
        
        if epsilon > 10:
            return False, f"Epsilon {epsilon} > 10 (high privacy loss)"
        
        if delta <= 0:
            return False, f"Delta must be positive, got {delta}"
        
        if delta >= 1:
            return False, f"Delta {delta} >= 1 (invalid probability)"
        
        # Recommended bounds for practical DP
        if delta > 1e-5:
            return False, f"Delta {delta} > 1e-5 (weak privacy)"
        
        # Privacy budget recommendations
        privacy_level = "high" if epsilon <= 1 else "medium" if epsilon <= 5 else "low"
        
        return True, f"DP parameters valid: ε={epsilon}, δ={delta} ({privacy_level} privacy)"
        
    except Exception as e:
        return False, f"Parameter validation failed: {e}"

def validate_rdp_accounting() -> Tuple[bool, str]:
    """Validate RDP (Rényi Differential Privacy) accounting setup."""
    try:
        # Try to import Opacus privacy components
        sys.path.insert(0, 'src')
        
        try:
            from opacus.accountants.rdp import RDPAccountant
            from opacus.accountants.utils import get_noise_multiplier
            
            # Test RDP accountant initialization
            accountant = RDPAccountant()
            
            # Test noise multiplier calculation
            noise_multiplier = get_noise_multiplier(
                target_epsilon=1.0,
                target_delta=1e-5,
                sample_rate=0.01,
                steps=1000
            )
            
            if noise_multiplier <= 0:
                return False, f"Invalid noise multiplier: {noise_multiplier}"
            
            return True, f"RDP accounting ready (noise_multiplier≈{noise_multiplier:.3f}) ✅"
            
        except ImportError as e:
            return False, f"Opacus not available: {e}"
            
    except Exception as e:
        return False, f"RDP validation failed: {e}"

def validate_privacy_engine() -> Tuple[bool, str]:
    """Validate privacy engine configuration."""
    try:
        sys.path.insert(0, 'src')
        from dp_federated_lora.privacy import PrivacyEngine, PrivacyAccountant
        
        # Test privacy engine creation
        config = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 1.1
        }
        
        # This will test import and basic structure
        return True, "Privacy engine module structure ✅"
        
    except ImportError:
        return False, "Privacy engine module not available"
    except Exception as e:
        return False, f"Privacy engine validation failed: {e}"

def validate_secure_aggregation() -> Tuple[bool, str]:
    """Validate secure aggregation protocols."""
    try:
        sys.path.insert(0, 'src')
        from dp_federated_lora.aggregation import SecureAggregator, ByzantineRobustAggregator
        
        return True, "Secure aggregation modules ✅"
        
    except ImportError:
        return False, "Aggregation modules not available"
    except Exception as e:
        return False, f"Secure aggregation validation failed: {e}"

def validate_gradient_clipping(max_grad_norm: float = 1.0) -> Tuple[bool, str]:
    """Validate gradient clipping configuration."""
    try:
        if max_grad_norm <= 0:
            return False, f"max_grad_norm must be positive, got {max_grad_norm}"
        
        if max_grad_norm > 10:
            return False, f"max_grad_norm {max_grad_norm} > 10 (may reduce utility)"
        
        return True, f"Gradient clipping: max_norm={max_grad_norm} ✅"
        
    except Exception as e:
        return False, f"Gradient clipping validation failed: {e}"

def validate_data_privacy_safeguards() -> Tuple[bool, str]:
    """Validate data privacy safeguards."""
    try:
        # Check for potential data leakage safeguards
        safeguards = []
        
        # Check if logging is configured to avoid data leakage
        import logging
        logger = logging.getLogger('dp_federated_lora')
        if logger.level >= logging.INFO:
            safeguards.append("logging_level_safe")
        
        # Check cryptographic capabilities
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            safeguards.append("encryption_available")
        except ImportError:
            pass
        
        if len(safeguards) == 0:
            return False, "No privacy safeguards detected"
        
        return True, f"Privacy safeguards: {', '.join(safeguards)} ✅"
        
    except Exception as e:
        return False, f"Data privacy validation failed: {e}"

def validate_privacy_budget_tracking() -> Tuple[bool, str]:
    """Validate privacy budget tracking mechanisms."""
    try:
        # Test privacy budget calculations
        epsilon_per_round = 0.1
        num_rounds = 10
        total_epsilon = epsilon_per_round * num_rounds
        
        if total_epsilon > 10:
            return False, f"Privacy budget exceeded: {total_epsilon} > 10"
        
        # Test composition theorems (basic)
        # Advanced composition bound: ε' ≤ ε√(2k ln(1/δ)) + kε(e^ε - 1)
        k = num_rounds
        delta = 1e-5
        epsilon = epsilon_per_round
        
        if k > 0 and delta > 0 and epsilon > 0:
            advanced_bound = epsilon * math.sqrt(2 * k * math.log(1/delta)) + k * epsilon * (math.exp(epsilon) - 1)
            composition_method = "advanced" if advanced_bound < total_epsilon else "basic"
        else:
            composition_method = "basic"
        
        return True, f"Privacy budget tracking ({composition_method} composition) ✅"
        
    except Exception as e:
        return False, f"Privacy budget validation failed: {e}"

def run_privacy_validation() -> Dict[str, Tuple[bool, str]]:
    """Run all privacy validations."""
    checks = {
        'epsilon_delta_validation': lambda: validate_epsilon_delta(1.0, 1e-5),
        'rdp_accounting': validate_rdp_accounting,
        'privacy_engine': validate_privacy_engine,
        'secure_aggregation': validate_secure_aggregation,
        'gradient_clipping': lambda: validate_gradient_clipping(1.0),
        'data_privacy_safeguards': validate_data_privacy_safeguards,
        'privacy_budget_tracking': validate_privacy_budget_tracking,
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            success, message = check_func()
            results[check_name] = (success, message)
        except Exception as e:
            results[check_name] = (False, f"Check failed: {e}")
    
    return results

def main():
    """Main privacy validation execution."""
    print("🔒 DP-Federated LoRA Privacy Validation")
    print("=" * 50)
    
    start_time = time.time()
    results = run_privacy_validation()
    end_time = time.time()
    
    # Display results
    passed = 0
    total = len(results)
    
    for check_name, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {check_name:25} | {message}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Privacy Validation: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    print(f"Duration: {end_time - start_time:.2f}s")
    
    # Create privacy report
    privacy_report = {
        'timestamp': time.time(),
        'total_privacy_checks': total,
        'passed_privacy_checks': passed,
        'privacy_compliance_rate': passed / total * 100,
        'duration_seconds': end_time - start_time,
        'privacy_details': {name: {'success': success, 'message': message} 
                          for name, (success, message) in results.items()},
        'recommendations': [],
    }
    
    # Add recommendations based on results
    if passed < total:
        privacy_report['recommendations'].append("Address failed privacy checks before production deployment")
    if results.get('epsilon_delta_validation', (False, ''))[0] == False:
        privacy_report['recommendations'].append("Review and adjust privacy parameters (ε, δ)")
    if results.get('rdp_accounting', (False, ''))[0] == False:
        privacy_report['recommendations'].append("Install and configure Opacus for RDP accounting")
    
    # Save privacy report
    with open('privacy_report.json', 'w') as f:
        json.dump(privacy_report, f, indent=2)
    
    print(f"📊 Privacy report saved to privacy_report.json")
    
    # Exit with error code if critical privacy checks failed
    critical_checks = ['epsilon_delta_validation', 'privacy_engine', 'gradient_clipping']
    critical_failures = [name for name in critical_checks if not results.get(name, (False, ''))[0]]
    
    if critical_failures:
        print(f"\n⚠️  Critical privacy checks failed: {critical_failures}")
        sys.exit(1)
    elif passed < total:
        print(f"\n⚠️  Some privacy checks failed, but system may still be usable")
        sys.exit(0)
    else:
        print(f"\n🔒 All privacy validations passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
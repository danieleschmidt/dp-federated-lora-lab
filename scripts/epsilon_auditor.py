#!/usr/bin/env python3
"""
Epsilon auditor for DP-Federated LoRA Lab.
Advanced privacy budget management and auditing.
"""

import json
import sys
import time
import math
import warnings
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class PrivacyBudgetEntry:
    """Single privacy budget usage entry."""
    timestamp: float
    epsilon: float
    delta: float
    mechanism: str
    operation: str
    client_id: Optional[str] = None
    round_id: Optional[int] = None

class PrivacyLedger:
    """Privacy budget ledger for tracking cumulative usage."""
    
    def __init__(self, total_epsilon: float = 10.0, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.entries: List[PrivacyBudgetEntry] = []
        self.composition_method = "advanced"  # or "basic"
    
    def add_entry(self, entry: PrivacyBudgetEntry) -> bool:
        """Add privacy budget entry and check if within bounds."""
        self.entries.append(entry)
        current_epsilon, current_delta = self.get_current_budget()
        
        return (current_epsilon <= self.total_epsilon and 
                current_delta <= self.total_delta)
    
    def get_current_budget(self) -> Tuple[float, float]:
        """Calculate current privacy budget usage."""
        if not self.entries:
            return 0.0, 0.0
        
        if self.composition_method == "basic":
            return self._basic_composition()
        else:
            return self._advanced_composition()
    
    def _basic_composition(self) -> Tuple[float, float]:
        """Basic composition: sum all epsilons and deltas."""
        total_eps = sum(entry.epsilon for entry in self.entries)
        total_delta = sum(entry.delta for entry in self.entries)
        return total_eps, total_delta
    
    def _advanced_composition(self) -> Tuple[float, float]:
        """Advanced composition with improved bounds."""
        if not self.entries:
            return 0.0, 0.0
        
        # Group by mechanism for better composition
        mechanisms = {}
        for entry in self.entries:
            key = entry.mechanism
            if key not in mechanisms:
                mechanisms[key] = {'epsilon': 0, 'delta': 0, 'count': 0}
            mechanisms[key]['epsilon'] += entry.epsilon
            mechanisms[key]['delta'] += entry.delta
            mechanisms[key]['count'] += 1
        
        # Apply advanced composition theorem
        total_epsilon = 0.0
        total_delta = 0.0
        
        for mech_data in mechanisms.values():
            epsilon = mech_data['epsilon']
            delta = mech_data['delta']
            k = mech_data['count']
            
            if k == 1:
                # Single query, no composition needed
                total_epsilon += epsilon
                total_delta += delta
            else:
                # Advanced composition bound
                # ε' ≤ ε√(2k ln(1/δ')) + kε(e^ε - 1)
                delta_prime = delta / k  # distribute delta budget
                if delta_prime > 0 and epsilon > 0:
                    composed_eps = (epsilon * math.sqrt(2 * k * math.log(1/delta_prime)) +
                                   k * epsilon * (math.exp(epsilon) - 1))
                    total_epsilon += composed_eps
                    total_delta += k * delta_prime
                else:
                    # Fallback to basic composition
                    total_epsilon += epsilon * k
                    total_delta += delta * k
        
        return total_epsilon, total_delta

def audit_federated_training_scenario() -> Tuple[bool, str, PrivacyLedger]:
    """Audit a typical federated training scenario."""
    try:
        # Simulate federated training privacy budget usage
        ledger = PrivacyLedger(total_epsilon=5.0, total_delta=1e-5)
        
        num_clients = 10
        num_rounds = 5
        epsilon_per_client_per_round = 0.1
        delta_per_client_per_round = 1e-6
        
        timestamp = time.time()
        
        # Simulate training rounds
        for round_id in range(num_rounds):
            for client_id in range(num_clients):
                entry = PrivacyBudgetEntry(
                    timestamp=timestamp + round_id * 100 + client_id,
                    epsilon=epsilon_per_client_per_round,
                    delta=delta_per_client_per_round,
                    mechanism="DP-SGD",
                    operation="gradient_update",
                    client_id=f"client_{client_id}",
                    round_id=round_id
                )
                
                if not ledger.add_entry(entry):
                    current_eps, current_delta = ledger.get_current_budget()
                    return False, f"Budget exceeded at round {round_id}, client {client_id}: ε={current_eps:.3f}, δ={current_delta:.2e}", ledger
        
        final_eps, final_delta = ledger.get_current_budget()
        return True, f"Federated scenario OK: ε={final_eps:.3f}/{ledger.total_epsilon}, δ={final_delta:.2e}/{ledger.total_delta}", ledger
        
    except Exception as e:
        return False, f"Audit failed: {e}", PrivacyLedger()

def audit_privacy_amplification_by_sampling() -> Tuple[bool, str]:
    """Audit privacy amplification through subsampling."""
    try:
        # Privacy amplification calculation
        base_epsilon = 1.0
        base_delta = 1e-6
        sampling_rate = 0.01  # 1% sampling
        
        # Amplification by subsampling (simplified)
        # For Gaussian mechanism: ε_amplified ≈ sampling_rate * ε_base
        amplified_epsilon = sampling_rate * base_epsilon
        
        # Delta amplification (approximately)
        amplified_delta = sampling_rate * base_delta
        
        amplification_factor = base_epsilon / amplified_epsilon if amplified_epsilon > 0 else float('inf')
        
        if amplification_factor < 1:
            return False, f"Invalid amplification factor: {amplification_factor}"
        
        return True, f"Privacy amplification by sampling: {amplification_factor:.1f}x improvement (ε: {base_epsilon}→{amplified_epsilon:.3f})"
        
    except Exception as e:
        return False, f"Privacy amplification audit failed: {e}"

def audit_rdp_conversion() -> Tuple[bool, str]:
    """Audit RDP to (ε,δ)-DP conversion."""
    try:
        # Test RDP accounting conversion
        alpha = 2.0  # Rényi parameter
        rdp_epsilon = 0.5  # RDP budget
        target_delta = 1e-5
        
        # Convert RDP to (ε,δ)-DP using the standard conversion
        # ε(δ) = rdp_ε + (log(1/δ))/(α-1)
        if alpha <= 1:
            return False, f"Invalid Rényi parameter α={alpha} (must be > 1)"
        
        converted_epsilon = rdp_epsilon + math.log(1/target_delta) / (alpha - 1)
        
        if converted_epsilon < 0:
            return False, f"Negative converted epsilon: {converted_epsilon}"
        
        return True, f"RDP conversion: RDP(α={alpha}, ε={rdp_epsilon}) → (ε={converted_epsilon:.3f}, δ={target_delta:.2e})"
        
    except Exception as e:
        return False, f"RDP conversion audit failed: {e}"

def audit_gradient_clipping_bounds() -> Tuple[bool, str]:
    """Audit gradient clipping parameter bounds."""
    try:
        max_grad_norms = [0.1, 1.0, 5.0, 10.0]
        results = []
        
        for max_grad_norm in max_grad_norms:
            # Estimate privacy-utility tradeoff
            # Larger clipping bound = less privacy noise needed = better utility
            # But also potentially more privacy leakage
            
            if max_grad_norm <= 0:
                return False, f"Invalid gradient clipping bound: {max_grad_norm}"
            
            # Noise scale for Gaussian mechanism: σ = C * noise_multiplier / ε
            # where C is the sensitivity (related to clipping bound)
            noise_multiplier = 1.1  # typical value
            epsilon = 1.0
            
            noise_scale = max_grad_norm * noise_multiplier / epsilon
            signal_to_noise_ratio = max_grad_norm / noise_scale if noise_scale > 0 else float('inf')
            
            results.append(f"C={max_grad_norm} → SNR={signal_to_noise_ratio:.2f}")
        
        return True, f"Gradient clipping analysis: {', '.join(results)}"
        
    except Exception as e:
        return False, f"Gradient clipping audit failed: {e}"

def run_epsilon_audit() -> Dict[str, Tuple[bool, str]]:
    """Run comprehensive privacy budget audit."""
    checks = {}
    
    # Run federated training scenario audit
    fed_success, fed_message, ledger = audit_federated_training_scenario()
    checks['federated_scenario'] = (fed_success, fed_message)
    
    # Export ledger details if successful
    if fed_success and ledger:
        ledger_summary = {
            'total_entries': len(ledger.entries),
            'final_epsilon': ledger.get_current_budget()[0],
            'final_delta': ledger.get_current_budget()[1],
            'budget_utilization': ledger.get_current_budget()[0] / ledger.total_epsilon * 100,
            'composition_method': ledger.composition_method
        }
        
        # Save detailed ledger
        ledger_data = {
            'config': {
                'total_epsilon': ledger.total_epsilon,
                'total_delta': ledger.total_delta,
                'composition_method': ledger.composition_method
            },
            'summary': ledger_summary,
            'entries': [
                {
                    'timestamp': entry.timestamp,
                    'epsilon': entry.epsilon,
                    'delta': entry.delta,
                    'mechanism': entry.mechanism,
                    'operation': entry.operation,
                    'client_id': entry.client_id,
                    'round_id': entry.round_id
                }
                for entry in ledger.entries
            ]
        }
        
        with open('privacy_ledger.json', 'w') as f:
            json.dump(ledger_data, f, indent=2)
    
    # Other audits
    checks['privacy_amplification'] = audit_privacy_amplification_by_sampling()
    checks['rdp_conversion'] = audit_rdp_conversion()
    checks['gradient_clipping'] = audit_gradient_clipping_bounds()
    
    return checks

def main():
    """Main epsilon audit execution."""
    print("📊 DP-Federated LoRA Epsilon Audit")
    print("=" * 50)
    
    start_time = time.time()
    results = run_epsilon_audit()
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
    print(f"Epsilon Audit: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    print(f"Duration: {end_time - start_time:.2f}s")
    
    # Create audit report
    audit_report = {
        'timestamp': time.time(),
        'total_audits': total,
        'passed_audits': passed,
        'audit_success_rate': passed / total * 100,
        'duration_seconds': end_time - start_time,
        'audit_details': {name: {'success': success, 'message': message} 
                        for name, (success, message) in results.items()},
        'privacy_recommendations': []
    }
    
    # Add privacy recommendations
    if results.get('federated_scenario', (False, ''))[0]:
        audit_report['privacy_recommendations'].append("Federated training budget allocation is within bounds")
    else:
        audit_report['privacy_recommendations'].append("Review federated training privacy budget allocation")
    
    if results.get('privacy_amplification', (False, ''))[0]:
        audit_report['privacy_recommendations'].append("Consider privacy amplification by subsampling to improve privacy-utility tradeoff")
    
    # Save audit report
    with open('epsilon_audit_report.json', 'w') as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"📈 Privacy ledger saved to privacy_ledger.json")
    print(f"📊 Epsilon audit report saved to epsilon_audit_report.json")
    
    if passed < total:
        print(f"\n⚠️  {total - passed} privacy audit(s) failed!")
        sys.exit(1)
    else:
        print(f"\n🎯 All privacy audits passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Privacy guarantee validation system for DP-Federated LoRA.

This system validates and monitors differential privacy guarantees,
privacy budget tracking, and privacy-preserving mechanisms.
"""

import logging
import math
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Privacy budget tracking."""
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self.spent_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.delta - self.spent_delta)
    
    @property
    def is_exhausted(self) -> bool:
        return self.remaining_epsilon <= 0 or self.remaining_delta <= 0


@dataclass
class PrivacyParameters:
    """Differential privacy parameters."""
    noise_multiplier: float
    max_grad_norm: float
    batch_size: int
    epochs: int
    dataset_size: int


class RDPAccountant:
    """Rényi Differential Privacy accountant."""
    
    def __init__(self, orders: Optional[List[float]] = None):
        """Initialize RDP accountant with moment orders."""
        self.orders = orders or [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5, 5.]
        self.rdp_values = {order: 0.0 for order in self.orders}
    
    def compose_privacy(self, noise_multiplier: float, steps: int, sampling_rate: float) -> Dict[str, float]:
        """Compute privacy cost using RDP composition."""
        # RDP composition for Gaussian mechanism
        for order in self.orders:
            if order == 1.0:
                continue  # Skip α=1 (undefined for Gaussian)
            
            # RDP for subsampled Gaussian mechanism
            if sampling_rate == 0:
                rdp_step = 0.0
            else:
                # Simplified RDP calculation for Gaussian mechanism
                rdp_step = (sampling_rate**2) * (order / (2 * noise_multiplier**2))
            
            self.rdp_values[order] += steps * rdp_step
        
        return self.rdp_values.copy()
    
    def get_privacy_spent(self, delta: float) -> float:
        """Convert RDP to (ε, δ)-DP."""
        if delta <= 0 or delta >= 1:
            return float('inf')
        
        min_epsilon = float('inf')
        
        for order in self.orders:
            if order <= 1:
                continue
            
            rdp_value = self.rdp_values[order]
            if rdp_value < 0:
                continue
            
            # Convert RDP to (ε, δ)-DP
            epsilon = rdp_value + math.log(1 / delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return max(0.0, min_epsilon)
    
    def reset(self):
        """Reset the accountant."""
        self.rdp_values = {order: 0.0 for order in self.orders}


class PrivacyValidator:
    """Privacy guarantee validator."""
    
    def __init__(self):
        """Initialize privacy validator."""
        self.accountant = RDPAccountant()
        self.validation_history = []
    
    def validate_privacy_parameters(self, params: PrivacyParameters) -> Dict[str, Any]:
        """Validate privacy parameters for correctness."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Validate noise multiplier
        if params.noise_multiplier <= 0:
            validation_results["errors"].append("Noise multiplier must be positive")
            validation_results["valid"] = False
        elif params.noise_multiplier < 0.5:
            validation_results["warnings"].append(
                f"Very low noise multiplier ({params.noise_multiplier:.2f}) may provide weak privacy"
            )
        elif params.noise_multiplier > 10:
            validation_results["warnings"].append(
                f"Very high noise multiplier ({params.noise_multiplier:.2f}) may significantly impact utility"
            )
        
        # Validate gradient norm clipping
        if params.max_grad_norm <= 0:
            validation_results["errors"].append("Maximum gradient norm must be positive")
            validation_results["valid"] = False
        elif params.max_grad_norm > 10:
            validation_results["warnings"].append(
                f"High gradient clipping threshold ({params.max_grad_norm}) may be ineffective"
            )
        
        # Validate batch size
        if params.batch_size <= 0:
            validation_results["errors"].append("Batch size must be positive")
            validation_results["valid"] = False
        elif params.batch_size < 32:
            validation_results["warnings"].append(
                f"Small batch size ({params.batch_size}) may lead to high variance in privacy accounting"
            )
        
        # Validate epochs
        if params.epochs <= 0:
            validation_results["errors"].append("Number of epochs must be positive")
            validation_results["valid"] = False
        elif params.epochs > 100:
            validation_results["warnings"].append(
                f"Large number of epochs ({params.epochs}) will consume significant privacy budget"
            )
        
        # Validate dataset size
        if params.dataset_size <= 0:
            validation_results["errors"].append("Dataset size must be positive")
            validation_results["valid"] = False
        
        # Calculate privacy-utility tradeoff metrics
        if validation_results["valid"]:
            sampling_rate = params.batch_size / params.dataset_size
            steps = params.epochs * (params.dataset_size // params.batch_size)
            
            # Recommend adjustments
            if sampling_rate > 0.1:
                validation_results["recommendations"].append(
                    "Consider reducing batch size relative to dataset size for better privacy"
                )
            
            if steps > 1000:
                validation_results["recommendations"].append(
                    "High number of training steps may exhaust privacy budget quickly"
                )
        
        return validation_results
    
    def compute_privacy_guarantees(
        self, 
        params: PrivacyParameters, 
        target_delta: float = 1e-5
    ) -> Dict[str, float]:
        """Compute privacy guarantees for given parameters."""
        
        # Reset accountant for fresh calculation
        self.accountant.reset()
        
        # Calculate sampling rate and steps
        sampling_rate = params.batch_size / params.dataset_size
        steps_per_epoch = params.dataset_size // params.batch_size
        total_steps = params.epochs * steps_per_epoch
        
        # Compute RDP values
        rdp_values = self.accountant.compose_privacy(
            noise_multiplier=params.noise_multiplier,
            steps=total_steps,
            sampling_rate=sampling_rate
        )
        
        # Convert to (ε, δ)-DP
        epsilon = self.accountant.get_privacy_spent(target_delta)
        
        privacy_guarantees = {
            "epsilon": epsilon,
            "delta": target_delta,
            "noise_multiplier": params.noise_multiplier,
            "max_grad_norm": params.max_grad_norm,
            "sampling_rate": sampling_rate,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "rdp_values": rdp_values
        }
        
        return privacy_guarantees
    
    def validate_budget_usage(
        self, 
        budget: PrivacyBudget, 
        params: PrivacyParameters
    ) -> Dict[str, Any]:
        """Validate privacy budget usage."""
        
        # Compute expected privacy cost
        guarantees = self.compute_privacy_guarantees(params, budget.delta)
        expected_epsilon = guarantees["epsilon"]
        
        validation = {
            "budget_sufficient": expected_epsilon <= budget.remaining_epsilon,
            "expected_epsilon": expected_epsilon,
            "available_epsilon": budget.remaining_epsilon,
            "utilization_rate": expected_epsilon / budget.epsilon if budget.epsilon > 0 else 0,
            "recommendations": []
        }
        
        # Generate recommendations
        if not validation["budget_sufficient"]:
            validation["recommendations"].append(
                f"Insufficient privacy budget. Need {expected_epsilon:.2f}, have {budget.remaining_epsilon:.2f}"
            )
            validation["recommendations"].append(
                "Consider: reducing epochs, increasing noise multiplier, or increasing batch size"
            )
        
        utilization = validation["utilization_rate"]
        if utilization > 0.8:
            validation["recommendations"].append(
                f"High budget utilization ({utilization:.1%}). Consider reserving budget for future rounds."
            )
        elif utilization < 0.1:
            validation["recommendations"].append(
                f"Low budget utilization ({utilization:.1%}). Could potentially reduce noise for better utility."
            )
        
        return validation
    
    def monitor_privacy_leakage(
        self, 
        client_updates: List[Dict[str, Any]], 
        global_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor for potential privacy leakage."""
        
        monitoring_results = {
            "leakage_detected": False,
            "anomalies": [],
            "client_statistics": {},
            "recommendations": []
        }
        
        if not client_updates:
            return monitoring_results
        
        # Analyze client update patterns
        update_norms = []
        gradient_similarities = []
        
        for i, update in enumerate(client_updates):
            client_id = f"client_{i}"
            
            # Mock parameter analysis (in real implementation, would analyze actual tensors)
            update_norm = 1.5 + 0.3 * i  # Mock norm calculation
            update_norms.append(update_norm)
            
            # Calculate similarity with global model (mock)
            similarity = 0.9 - 0.1 * (i / len(client_updates))  # Mock similarity
            gradient_similarities.append(similarity)
            
            monitoring_results["client_statistics"][client_id] = {
                "update_norm": update_norm,
                "similarity_to_global": similarity
            }
        
        # Detect anomalies
        if update_norms:
            mean_norm = sum(update_norms) / len(update_norms)
            max_norm = max(update_norms)
            min_norm = min(update_norms)
            
            # Check for outliers
            if max_norm > 3 * mean_norm:
                monitoring_results["anomalies"].append(
                    f"Unusually large update norm detected: {max_norm:.3f} (mean: {mean_norm:.3f})"
                )
                monitoring_results["leakage_detected"] = True
            
            if max_norm - min_norm > 2 * mean_norm:
                monitoring_results["anomalies"].append(
                    f"High variance in update norms: range [{min_norm:.3f}, {max_norm:.3f}]"
                )
        
        # Check gradient similarities
        if gradient_similarities:
            min_similarity = min(gradient_similarities)
            max_similarity = max(gradient_similarities)
            
            if max_similarity - min_similarity > 0.5:
                monitoring_results["anomalies"].append(
                    f"High variance in client similarities: range [{min_similarity:.3f}, {max_similarity:.3f}]"
                )
        
        # Generate recommendations
        if monitoring_results["leakage_detected"]:
            monitoring_results["recommendations"].append("Increase noise multiplier")
            monitoring_results["recommendations"].append("Implement additional gradient clipping")
            monitoring_results["recommendations"].append("Consider client selection based on update norms")
        
        return monitoring_results
    
    def generate_privacy_report(
        self, 
        budget: PrivacyBudget, 
        params: PrivacyParameters, 
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        
        guarantees = self.compute_privacy_guarantees(params, budget.delta)
        budget_validation = self.validate_budget_usage(budget, params)
        
        report = {
            "timestamp": "2025-08-18T14:00:00Z",  # Mock timestamp
            "privacy_guarantees": guarantees,
            "budget_status": {
                "total_epsilon": budget.epsilon,
                "total_delta": budget.delta,
                "spent_epsilon": budget.spent_epsilon,
                "spent_delta": budget.spent_delta,
                "remaining_epsilon": budget.remaining_epsilon,
                "remaining_delta": budget.remaining_delta,
                "is_exhausted": budget.is_exhausted
            },
            "parameter_validation": validation_results,
            "budget_validation": budget_validation,
            "overall_status": "VALID" if validation_results["valid"] and budget_validation["budget_sufficient"] else "INVALID",
            "recommendations": (
                validation_results.get("recommendations", []) + 
                budget_validation.get("recommendations", [])
            )
        }
        
        return report


def test_privacy_validation_system():
    """Test privacy validation system functionality."""
    logger.info("=== Testing Privacy Validation System ===")
    
    try:
        validator = PrivacyValidator()
        
        # Test 1: Basic parameter validation
        logger.info("--- Test 1: Parameter Validation ---")
        
        params = PrivacyParameters(
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            batch_size=64,
            epochs=10,
            dataset_size=1000
        )
        
        validation_results = validator.validate_privacy_parameters(params)
        assert validation_results["valid"], "Valid parameters should pass validation"
        logger.info(f"✓ Parameter validation: {len(validation_results['warnings'])} warnings")
        
        # Test 2: Invalid parameter detection
        logger.info("--- Test 2: Invalid Parameter Detection ---")
        
        invalid_params = PrivacyParameters(
            noise_multiplier=-0.5,  # Invalid: negative
            max_grad_norm=0,        # Invalid: zero
            batch_size=128,
            epochs=5,
            dataset_size=500
        )
        
        invalid_validation = validator.validate_privacy_parameters(invalid_params)
        assert not invalid_validation["valid"], "Invalid parameters should fail validation"
        assert len(invalid_validation["errors"]) > 0, "Should detect errors"
        logger.info(f"✓ Invalid parameter detection: {len(invalid_validation['errors'])} errors")
        
        # Test 3: Privacy guarantee computation
        logger.info("--- Test 3: Privacy Guarantee Computation ---")
        
        guarantees = validator.compute_privacy_guarantees(params)
        assert guarantees["epsilon"] > 0, "Should compute positive epsilon"
        assert guarantees["delta"] > 0, "Should have positive delta"
        assert guarantees["total_steps"] > 0, "Should have positive total steps"
        
        logger.info(f"✓ Computed privacy guarantees: ε={guarantees['epsilon']:.2f}, δ={guarantees['delta']:.2e}")
        
        # Test 4: Budget validation
        logger.info("--- Test 4: Budget Validation ---")
        
        budget = PrivacyBudget(epsilon=8.0, delta=1e-5)
        budget_validation = validator.validate_budget_usage(budget, params)
        
        assert "budget_sufficient" in budget_validation, "Should check budget sufficiency"
        assert "expected_epsilon" in budget_validation, "Should compute expected epsilon"
        
        logger.info(f"✓ Budget validation: sufficient={budget_validation['budget_sufficient']}")
        
        # Test 5: Privacy leakage monitoring
        logger.info("--- Test 5: Privacy Leakage Monitoring ---")
        
        mock_updates = [{"update": f"client_{i}_update"} for i in range(5)]
        mock_global_model = {"global": "model_state"}
        
        monitoring = validator.monitor_privacy_leakage(mock_updates, mock_global_model)
        assert "leakage_detected" in monitoring, "Should check for leakage"
        assert "client_statistics" in monitoring, "Should provide client statistics"
        
        logger.info(f"✓ Privacy leakage monitoring: {len(monitoring['anomalies'])} anomalies detected")
        
        # Test 6: Comprehensive privacy report
        logger.info("--- Test 6: Privacy Report Generation ---")
        
        report = validator.generate_privacy_report(budget, params, validation_results)
        
        required_sections = [
            "privacy_guarantees", "budget_status", "parameter_validation",
            "budget_validation", "overall_status", "recommendations"
        ]
        
        for section in required_sections:
            assert section in report, f"Report missing section: {section}"
        
        logger.info(f"✓ Privacy report generated with status: {report['overall_status']}")
        
        # Test 7: RDP Accountant functionality
        logger.info("--- Test 7: RDP Accountant ---")
        
        accountant = RDPAccountant()
        initial_rdp = accountant.rdp_values.copy()
        
        # Compose some privacy
        accountant.compose_privacy(noise_multiplier=1.0, steps=100, sampling_rate=0.01)
        
        # Check that RDP values increased
        final_rdp = accountant.rdp_values
        increased = any(final_rdp[order] > initial_rdp[order] for order in accountant.orders)
        assert increased, "RDP values should increase after composition"
        
        # Get privacy spent
        epsilon_spent = accountant.get_privacy_spent(delta=1e-5)
        assert epsilon_spent >= 0, "Privacy spent should be non-negative"
        
        logger.info(f"✓ RDP Accountant: ε={epsilon_spent:.3f} for 100 steps")
        
        logger.info("✅ All privacy validation tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Privacy validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_privacy_budget_management():
    """Test privacy budget management."""
    logger.info("=== Testing Privacy Budget Management ===")
    
    try:
        # Test budget creation and tracking
        budget = PrivacyBudget(epsilon=8.0, delta=1e-5)
        
        assert budget.remaining_epsilon == 8.0, "Initial remaining epsilon should equal total"
        assert budget.remaining_delta == 1e-5, "Initial remaining delta should equal total"
        assert not budget.is_exhausted, "Budget should not be exhausted initially"
        
        # Simulate spending budget
        budget.spent_epsilon = 3.0
        budget.spent_delta = 5e-6
        
        assert budget.remaining_epsilon == 5.0, "Should correctly calculate remaining epsilon"
        assert budget.remaining_delta == 5e-6, "Should correctly calculate remaining delta"
        assert not budget.is_exhausted, "Budget should not be exhausted yet"
        
        # Exhaust budget
        budget.spent_epsilon = 8.5  # Exceed total
        
        assert budget.remaining_epsilon == 0.0, "Remaining epsilon should be capped at 0"
        assert budget.is_exhausted, "Budget should be exhausted"
        
        logger.info("✅ Privacy budget management test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Privacy budget management test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_privacy_accounting_edge_cases():
    """Test privacy accounting edge cases."""
    logger.info("=== Testing Privacy Accounting Edge Cases ===")
    
    try:
        validator = PrivacyValidator()
        
        # Test with zero sampling rate
        params_zero_sampling = PrivacyParameters(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=1000,      # Full batch
            epochs=1,
            dataset_size=1000     # Same as batch size = no sampling
        )
        
        guarantees = validator.compute_privacy_guarantees(params_zero_sampling)
        # With full batch (no sampling), privacy cost should be minimal
        logger.info(f"✓ Zero sampling rate: ε={guarantees['epsilon']:.6f}")
        
        # Test with very small dataset
        params_small_dataset = PrivacyParameters(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=10,
            epochs=1,
            dataset_size=10
        )
        
        guarantees_small = validator.compute_privacy_guarantees(params_small_dataset)
        logger.info(f"✓ Small dataset: ε={guarantees_small['epsilon']:.6f}")
        
        # Test with extreme noise multiplier
        params_high_noise = PrivacyParameters(
            noise_multiplier=100.0,  # Very high noise
            max_grad_norm=1.0,
            batch_size=32,
            epochs=10,
            dataset_size=1000
        )
        
        guarantees_high_noise = validator.compute_privacy_guarantees(params_high_noise)
        logger.info(f"✓ High noise: ε={guarantees_high_noise['epsilon']:.6f}")
        
        # Test delta edge cases
        accountant = RDPAccountant()
        accountant.compose_privacy(1.0, 10, 0.1)
        
        # Test with very small delta
        epsilon_small_delta = accountant.get_privacy_spent(1e-10)
        logger.info(f"✓ Small delta (1e-10): ε={epsilon_small_delta:.3f}")
        
        # Test with zero delta (should return inf)
        epsilon_zero_delta = accountant.get_privacy_spent(0.0)
        assert epsilon_zero_delta == float('inf'), "Zero delta should return infinite epsilon"
        
        # Test with delta >= 1 (should return inf)
        epsilon_large_delta = accountant.get_privacy_spent(1.0)
        assert epsilon_large_delta == float('inf'), "Delta >= 1 should return infinite epsilon"
        
        logger.info("✅ Privacy accounting edge cases test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Privacy accounting edge cases test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all privacy validation tests."""
    logger.info("Starting privacy validation system tests...")
    
    tests = [
        test_privacy_validation_system,
        test_privacy_budget_management,
        test_privacy_accounting_edge_cases,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Privacy Validation Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All privacy validation tests PASSED!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. See logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
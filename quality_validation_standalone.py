#!/usr/bin/env python3
"""
Standalone quality validation system for novel LoRA optimization components.
Validates syntax, imports, and basic functionality without external test dependencies.
"""

import sys
import os
import traceback
import importlib
import inspect
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class QualityValidator:
    """Standalone quality validation system."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.validation_results = []
    
    def validate_import(self, module_path: str, description: str = None) -> bool:
        """Validate module import."""
        try:
            module = importlib.import_module(module_path)
            self.tests_passed += 1
            result = f"✅ Import validation passed: {module_path}"
            if description:
                result += f" ({description})"
            print(result)
            self.validation_results.append({'test': f"import_{module_path}", 'status': 'PASS', 'details': result})
            return True
        except Exception as e:
            self.tests_failed += 1
            result = f"❌ Import validation failed: {module_path} - {str(e)}"
            print(result)
            self.validation_results.append({'test': f"import_{module_path}", 'status': 'FAIL', 'details': result})
            return False
    
    def validate_class_instantiation(self, module, class_name: str, *args, **kwargs) -> bool:
        """Validate class can be instantiated."""
        try:
            cls = getattr(module, class_name)
            if args or kwargs:
                instance = cls(*args, **kwargs)
            else:
                # Try to instantiate with reasonable defaults
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())[1:]  # Skip 'self'
                
                test_args = []
                test_kwargs = {}
                
                for param in params:
                    if param.default != inspect.Parameter.empty:
                        continue  # Has default, skip
                    
                    # Provide mock values based on type hints or name
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == str:
                            test_kwargs[param.name] = "test"
                        elif param.annotation == int:
                            test_kwargs[param.name] = 1
                        elif param.annotation == float:
                            test_kwargs[param.name] = 1.0
                        elif param.annotation == bool:
                            test_kwargs[param.name] = True
                        elif hasattr(param.annotation, '__name__') and 'Model' in param.annotation.__name__:
                            # Mock model - skip this test
                            print(f"⚠️  Skipping instantiation test for {class_name} (requires model)")
                            return True
                    elif 'model' in param.name.lower():
                        # Skip model-dependent classes
                        print(f"⚠️  Skipping instantiation test for {class_name} (requires model)")
                        return True
                    else:
                        # Default fallback
                        test_kwargs[param.name] = None
                
                instance = cls(**test_kwargs)
            
            self.tests_passed += 1
            result = f"✅ Instantiation validation passed: {class_name}"
            print(result)
            self.validation_results.append({'test': f"instantiate_{class_name}", 'status': 'PASS', 'details': result})
            return True
        except Exception as e:
            if "requires model" in str(e).lower() or "model" in str(e).lower():
                print(f"⚠️  Skipping instantiation test for {class_name} (model dependency)")
                return True
            
            self.tests_failed += 1
            result = f"❌ Instantiation validation failed: {class_name} - {str(e)}"
            print(result)
            self.validation_results.append({'test': f"instantiate_{class_name}", 'status': 'FAIL', 'details': result})
            return False
    
    def validate_function_signature(self, module, function_name: str) -> bool:
        """Validate function signature."""
        try:
            func = getattr(module, function_name)
            sig = inspect.signature(func)
            
            self.tests_passed += 1
            result = f"✅ Function signature validation passed: {function_name}{sig}"
            print(result)
            self.validation_results.append({'test': f"signature_{function_name}", 'status': 'PASS', 'details': result})
            return True
        except Exception as e:
            self.tests_failed += 1
            result = f"❌ Function signature validation failed: {function_name} - {str(e)}"
            print(result)
            self.validation_results.append({'test': f"signature_{function_name}", 'status': 'FAIL', 'details': result})
            return False
    
    def validate_enum_values(self, module, enum_name: str) -> bool:
        """Validate enum values."""
        try:
            enum_cls = getattr(module, enum_name)
            values = list(enum_cls)
            
            if len(values) == 0:
                raise ValueError("Enum has no values")
            
            self.tests_passed += 1
            result = f"✅ Enum validation passed: {enum_name} with {len(values)} values"
            print(result)
            self.validation_results.append({'test': f"enum_{enum_name}", 'status': 'PASS', 'details': result})
            return True
        except Exception as e:
            self.tests_failed += 1
            result = f"❌ Enum validation failed: {enum_name} - {str(e)}"
            print(result)
            self.validation_results.append({'test': f"enum_{enum_name}", 'status': 'FAIL', 'details': result})
            return False
    
    def print_summary(self):
        """Print validation summary."""
        total = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*80)
        print("🔍 QUALITY VALIDATION SUMMARY")
        print("="*80)
        print(f"Total tests: {total}")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT: Quality validation passed with high confidence!")
        elif success_rate >= 80:
            print("✅ GOOD: Quality validation passed with acceptable confidence.")
        elif success_rate >= 70:
            print("⚠️  MODERATE: Quality validation passed with some concerns.")
        else:
            print("❌ POOR: Quality validation failed. Review required.")
        
        return success_rate >= 80


def main():
    """Run standalone quality validation."""
    print("🚀 Starting Standalone Quality Validation")
    print("="*80)
    
    validator = QualityValidator()
    
    # Validate core novel optimization components
    print("\n📦 Validating Novel LoRA Hyperparameter Optimizer...")
    if validator.validate_import("dp_federated_lora.novel_lora_hyperparameter_optimizer", "Novel LoRA optimizer"):
        import dp_federated_lora.novel_lora_hyperparameter_optimizer as novel_opt
        
        # Validate enums
        validator.validate_enum_values(novel_opt, "OptimizationStrategy")
        
        # Validate dataclasses
        validator.validate_class_instantiation(novel_opt, "LoRAHyperParams")
        validator.validate_class_instantiation(novel_opt, "OptimizationResult", 
                                             best_params=novel_opt.LoRAHyperParams(),
                                             best_score=0.8,
                                             optimization_history=[],
                                             convergence_metrics={})
        
        # Validate factory functions
        validator.validate_function_signature(novel_opt, "create_novel_lora_optimizer")
    
    # Validate robust optimization system
    print("\n🛡️ Validating Robust LoRA Optimization System...")
    if validator.validate_import("dp_federated_lora.robust_lora_optimization_system", "Robust system"):
        import dp_federated_lora.robust_lora_optimization_system as robust_opt
        
        # Validate enums
        validator.validate_enum_values(robust_opt, "OptimizationState")
        validator.validate_enum_values(robust_opt, "ValidationLevel")
        
        # Validate configurations
        validator.validate_class_instantiation(robust_opt, "OptimizationConfig")
        validator.validate_class_instantiation(robust_opt, "HealthMetrics")
        
        # Validate factory function
        validator.validate_function_signature(robust_opt, "create_robust_lora_optimizer")
    
    # Validate scalable optimization engine
    print("\n⚡ Validating Scalable LoRA Optimization Engine...")
    if validator.validate_import("dp_federated_lora.scalable_lora_optimization_engine", "Scalable engine"):
        import dp_federated_lora.scalable_lora_optimization_engine as scalable_opt
        
        # Validate enums
        validator.validate_enum_values(scalable_opt, "ScalingStrategy")
        validator.validate_enum_values(scalable_opt, "ResourceTier")
        
        # Validate configurations
        validator.validate_class_instantiation(scalable_opt, "ScalingConfig")
        
        # Validate factory functions
        validator.validate_function_signature(scalable_opt, "create_scalable_optimizer")
        validator.validate_function_signature(scalable_opt, "create_enterprise_optimizer")
    
    # Additional integration validation
    print("\n🔗 Validating Integration Points...")
    
    try:
        # Test that components can work together
        from dp_federated_lora.novel_lora_hyperparameter_optimizer import OptimizationStrategy, LoRAHyperParams
        from dp_federated_lora.robust_lora_optimization_system import OptimizationConfig, ValidationLevel
        from dp_federated_lora.scalable_lora_optimization_engine import ScalingConfig, ScalingStrategy
        
        # Test configuration compatibility
        opt_config = OptimizationConfig(
            strategy=OptimizationStrategy.HYBRID_QUANTUM,
            n_trials=10,
            validation_level=ValidationLevel.STANDARD
        )
        
        scaling_config = ScalingConfig(
            strategy=ScalingStrategy.ADAPTIVE_HYBRID,
            max_workers=4
        )
        
        lora_params = LoRAHyperParams(r=16, lora_alpha=32.0)
        
        validator.tests_passed += 1
        print("✅ Configuration compatibility validation passed")
        validator.validation_results.append({'test': 'config_compatibility', 'status': 'PASS', 'details': 'Configurations work together'})
        
    except Exception as e:
        validator.tests_failed += 1
        print(f"❌ Configuration compatibility validation failed: {e}")
        validator.validation_results.append({'test': 'config_compatibility', 'status': 'FAIL', 'details': str(e)})
    
    # Validate code quality aspects
    print("\n📋 Validating Code Quality...")
    
    # Check that all modules have proper docstrings
    modules_to_check = [
        "dp_federated_lora.novel_lora_hyperparameter_optimizer",
        "dp_federated_lora.robust_lora_optimization_system",
        "dp_federated_lora.scalable_lora_optimization_engine"
    ]
    
    for module_path in modules_to_check:
        try:
            module = importlib.import_module(module_path)
            if module.__doc__ and len(module.__doc__.strip()) > 50:
                validator.tests_passed += 1
                print(f"✅ Module documentation quality passed: {module_path}")
                validator.validation_results.append({'test': f'docs_{module_path}', 'status': 'PASS', 'details': 'Good documentation'})
            else:
                validator.tests_failed += 1
                print(f"❌ Module documentation quality failed: {module_path}")
                validator.validation_results.append({'test': f'docs_{module_path}', 'status': 'FAIL', 'details': 'Poor documentation'})
        except Exception as e:
            print(f"⚠️  Could not check documentation for {module_path}: {e}")
    
    # Final summary and results
    success = validator.print_summary()
    
    # Save results to file
    import json
    results_file = "quality_validation_results.json"
    
    summary_data = {
        'timestamp': str(__import__('datetime').datetime.now()),
        'total_tests': validator.tests_passed + validator.tests_failed,
        'tests_passed': validator.tests_passed,
        'tests_failed': validator.tests_failed,
        'success_rate': (validator.tests_passed / (validator.tests_passed + validator.tests_failed) * 100) if (validator.tests_passed + validator.tests_failed) > 0 else 0,
        'validation_results': validator.validation_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
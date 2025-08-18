#!/usr/bin/env python3
"""
Structure validation test for LoRA functionality.

This script validates the code structure and imports without requiring
external dependencies like PyTorch.
"""

import logging
import sys
import ast
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_lora_client_methods():
    """Validate that LoRA client has required methods."""
    logger.info("=== Validating LoRA Client Methods ===")
    
    client_file = Path("src/dp_federated_lora/client.py")
    if not client_file.exists():
        logger.error("❌ Client file not found")
        return False
    
    with open(client_file, 'r') as f:
        content = f.read()
    
    required_methods = [
        "extract_lora_parameters",
        "merge_lora_weights", 
        "get_lora_statistics",
        "adaptive_rank_selection",
        "update_lora_rank"
    ]
    
    found_methods = []
    for method in required_methods:
        if f"def {method}(" in content:
            found_methods.append(method)
            logger.info(f"✅ Found method: {method}")
        else:
            logger.error(f"❌ Missing method: {method}")
    
    success = len(found_methods) == len(required_methods)
    if success:
        logger.info("✅ All required LoRA client methods found")
    else:
        logger.error(f"❌ Missing {len(required_methods) - len(found_methods)} methods")
    
    return success


def validate_lora_aggregator():
    """Validate that LoRA aggregator is properly implemented."""
    logger.info("=== Validating LoRA Aggregator ===")
    
    aggregation_file = Path("src/dp_federated_lora/aggregation.py")
    if not aggregation_file.exists():
        logger.error("❌ Aggregation file not found")
        return False
    
    with open(aggregation_file, 'r') as f:
        content = f.read()
    
    # Check for LoRA aggregator class
    if "class LoRAAggregator" not in content:
        logger.error("❌ LoRAAggregator class not found")
        return False
    
    logger.info("✅ Found LoRAAggregator class")
    
    # Check for required methods
    required_methods = [
        "_aggregate_lora_matrices",
        "_validate_lora_consistency", 
        "_update_parameter_stats",
        "get_parameter_statistics"
    ]
    
    found_methods = []
    for method in required_methods:
        if f"def {method}(" in content:
            found_methods.append(method)
            logger.info(f"✅ Found method: {method}")
    
    # Check for LoRA aggregation method in enum
    config_file = Path("src/dp_federated_lora/config.py")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        if "LORA_FEDAVG" in config_content:
            logger.info("✅ Found LORA_FEDAVG aggregation method")
        else:
            logger.error("❌ Missing LORA_FEDAVG aggregation method")
            return False
    
    success = len(found_methods) >= 3  # At least most methods should be present
    if success:
        logger.info("✅ LoRA aggregator validation passed")
    else:
        logger.error("❌ LoRA aggregator validation failed")
    
    return success


def validate_import_structure():
    """Validate that imports are properly structured."""
    logger.info("=== Validating Import Structure ===")
    
    files_to_check = [
        "src/dp_federated_lora/client.py",
        "src/dp_federated_lora/aggregation.py",
        "src/dp_federated_lora/config.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ File not found: {file_path}")
            all_valid = False
            continue
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            logger.info(f"✅ Valid Python syntax: {file_path}")
            
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path}: {e}")
            all_valid = False
        except Exception as e:
            logger.error(f"❌ Error parsing {file_path}: {e}")
            all_valid = False
    
    return all_valid


def validate_code_quality():
    """Basic code quality checks."""
    logger.info("=== Validating Code Quality ===")
    
    client_file = Path("src/dp_federated_lora/client.py")
    if not client_file.exists():
        return False
    
    with open(client_file, 'r') as f:
        content = f.read()
    
    # Check for docstrings
    docstring_patterns = ['"""', "'''"]
    has_docstrings = any(pattern in content for pattern in docstring_patterns)
    
    if has_docstrings:
        logger.info("✅ Found docstrings")
    else:
        logger.warning("⚠️  Limited docstrings found")
    
    # Check for error handling
    if "try:" in content and "except" in content:
        logger.info("✅ Found error handling")
    else:
        logger.warning("⚠️  Limited error handling found")
    
    # Check for logging
    if "logger." in content:
        logger.info("✅ Found logging statements")
    else:
        logger.warning("⚠️  No logging statements found")
    
    return True


def validate_lora_specific_logic():
    """Validate LoRA-specific logic patterns."""
    logger.info("=== Validating LoRA-Specific Logic ===")
    
    client_file = Path("src/dp_federated_lora/client.py")
    aggregation_file = Path("src/dp_federated_lora/aggregation.py")
    
    validation_checks = []
    
    # Check client file for LoRA patterns
    if client_file.exists():
        with open(client_file, 'r') as f:
            client_content = f.read()
        
        # Check for LoRA parameter patterns
        lora_patterns = ["lora_A", "lora_B", "lora_embedding", "extract_lora"]
        found_patterns = [p for p in lora_patterns if p in client_content]
        
        if len(found_patterns) >= 3:
            logger.info(f"✅ Found LoRA patterns in client: {found_patterns}")
            validation_checks.append(True)
        else:
            logger.error(f"❌ Insufficient LoRA patterns in client: {found_patterns}")
            validation_checks.append(False)
    
    # Check aggregation file for LoRA-specific logic
    if aggregation_file.exists():
        with open(aggregation_file, 'r') as f:
            agg_content = f.read()
        
        # Check for LoRA aggregation logic
        agg_patterns = ["lora_A", "lora_B", "_aggregate_lora_matrices", "LoRAAggregator"]
        found_agg_patterns = [p for p in agg_patterns if p in agg_content]
        
        if len(found_agg_patterns) >= 3:
            logger.info(f"✅ Found LoRA aggregation patterns: {found_agg_patterns}")
            validation_checks.append(True)
        else:
            logger.error(f"❌ Insufficient LoRA aggregation patterns: {found_agg_patterns}")
            validation_checks.append(False)
    
    return all(validation_checks)


def main():
    """Run all structure validation tests."""
    logger.info("Starting LoRA structure validation tests...")
    
    tests = [
        validate_import_structure,
        validate_lora_client_methods,
        validate_lora_aggregator,
        validate_code_quality,
        validate_lora_specific_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Validation Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All LoRA structure validations PASSED!")
        return 0
    else:
        logger.warning("⚠️  Some validations failed. See logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
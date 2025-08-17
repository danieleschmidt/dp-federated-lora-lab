#!/usr/bin/env python3
"""
Research Validation Script for DP-Federated LoRA Lab

This script validates the core research functionality and demonstrates
autonomous breakthrough discovery capabilities.
"""

import os
import sys
import time
import json
import asyncio
import logging
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_core_imports():
    """Validate that core research components can be imported."""
    try:
        # Test numpy import
        import numpy as np
        logger.info("✅ NumPy imported successfully")
        
        # Test core research engine components
        from dp_federated_lora.research_engine import (
            AutonomousResearchEngine,
            NovelAlgorithmGenerator,
            ExperimentalFramework,
            ResearchValidator
        )
        logger.info("✅ Research engine components imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def validate_algorithm_generation():
    """Validate novel algorithm generation."""
    try:
        from dp_federated_lora.research_engine import NovelAlgorithmGenerator
        
        generator = NovelAlgorithmGenerator()
        hypothesis = generator.generate_novel_hypothesis()
        
        logger.info(f"✅ Generated hypothesis: {hypothesis.title}")
        logger.info(f"   Algorithm type: {hypothesis.algorithm_type.value}")
        logger.info(f"   Expected improvement: {hypothesis.expected_improvement:.1%}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Algorithm generation failed: {e}")
        return False

async def validate_experimental_framework():
    """Validate experimental framework."""
    try:
        from dp_federated_lora.research_engine import (
            NovelAlgorithmGenerator,
            ExperimentalFramework
        )
        
        generator = NovelAlgorithmGenerator()
        framework = ExperimentalFramework(output_dir="validation_results")
        
        # Generate a simple hypothesis
        hypothesis = generator.generate_novel_hypothesis()
        
        # Configure minimal dataset
        dataset_config = {
            "num_clients": 5,
            "data_distribution": "iid",
            "dataset_size": 100,
            "privacy_budget": 4.0
        }
        
        # Run minimal experiment
        logger.info("🧪 Running experimental validation...")
        results = await framework.conduct_experiment(
            hypothesis, dataset_config, num_runs=2
        )
        
        logger.info(f"✅ Experimental framework validation completed")
        logger.info(f"   Results generated: {len(results)} experiments")
        
        return True
    except Exception as e:
        logger.error(f"❌ Experimental framework validation failed: {e}")
        return False

async def validate_research_validation():
    """Validate research validation system."""
    try:
        from dp_federated_lora.research_engine import (
            NovelAlgorithmGenerator,
            ExperimentalFramework,
            ResearchValidator,
            ExperimentalResult
        )
        
        validator = ResearchValidator()
        generator = NovelAlgorithmGenerator()
        hypothesis = generator.generate_novel_hypothesis()
        
        # Create mock experimental results
        baseline_results = []
        proposed_results = []
        
        for i in range(3):
            # Baseline results
            baseline_result = ExperimentalResult(
                hypothesis_id=hypothesis.id,
                method=hypothesis.baseline_method,
                metrics={
                    "accuracy": 0.85 + 0.02 * (i - 1),
                    "f1_score": 0.82 + 0.02 * (i - 1),
                    "privacy_spent": 8.0 + 0.5 * (i - 1)
                },
                privacy_spent=8.0,
                runtime=300.0,
                resource_usage={"memory": 1.0, "cpu": 1.0},
                statistical_significance={},
                reproducible=True,
                experiment_id=f"baseline_{i}"
            )
            baseline_results.append(baseline_result)
            
            # Improved proposed results
            proposed_result = ExperimentalResult(
                hypothesis_id=hypothesis.id,
                method=hypothesis.proposed_method,
                metrics={
                    "accuracy": 0.90 + 0.02 * (i - 1),  # Better accuracy
                    "f1_score": 0.87 + 0.02 * (i - 1),  # Better F1
                    "privacy_spent": 6.0 + 0.3 * (i - 1)  # Better privacy
                },
                privacy_spent=6.0,
                runtime=280.0,
                resource_usage={"memory": 0.9, "cpu": 1.1},
                statistical_significance={},
                reproducible=True,
                experiment_id=f"proposed_{i}"
            )
            proposed_results.append(proposed_result)
        
        # Combine results
        all_results = baseline_results + proposed_results
        
        # Perform statistical analysis
        framework = ExperimentalFramework()
        statistical_analysis = framework._perform_statistical_analysis(
            baseline_results, proposed_results, hypothesis
        )
        
        # Validate findings
        breakthrough = validator.validate_research_findings(
            hypothesis, all_results, statistical_analysis
        )
        
        if breakthrough:
            logger.info("✅ Research validation successful - BREAKTHROUGH DETECTED!")
            logger.info(f"   Improvement factor: {breakthrough.improvement_factor:.1%}")
            logger.info(f"   Publication readiness: {breakthrough.publication_readiness:.1%}")
        else:
            logger.info("✅ Research validation completed (no breakthrough)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Research validation failed: {e}")
        return False

async def validate_autonomous_engine():
    """Validate autonomous research engine (quick test)."""
    try:
        from dp_federated_lora.research_engine import AutonomousResearchEngine
        
        # Create engine
        engine = AutonomousResearchEngine(output_dir="autonomous_validation")
        
        # Generate a few hypotheses quickly
        logger.info("🤖 Testing autonomous hypothesis generation...")
        
        for i in range(3):
            hypothesis = engine.algorithm_generator.generate_novel_hypothesis()
            engine.active_hypotheses.append(hypothesis)
            logger.info(f"   Generated: {hypothesis.title}")
        
        logger.info("✅ Autonomous engine validation completed")
        logger.info(f"   Hypotheses generated: {len(engine.active_hypotheses)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Autonomous engine validation failed: {e}")
        return False

def generate_validation_report():
    """Generate validation report."""
    report = {
        "validation_timestamp": time.time(),
        "validation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "validation_results": {},
        "status": "unknown"
    }
    
    return report

async def main():
    """Main validation function."""
    logger.info("🚀 Starting DP-Federated LoRA Research Validation")
    logger.info("=" * 60)
    
    report = generate_validation_report()
    validation_results = {}
    
    # Validate core imports
    logger.info("📦 Validating core imports...")
    validation_results["core_imports"] = validate_core_imports()
    
    # Validate algorithm generation
    logger.info("\n🧬 Validating algorithm generation...")
    validation_results["algorithm_generation"] = validate_algorithm_generation()
    
    # Validate experimental framework
    logger.info("\n🧪 Validating experimental framework...")
    validation_results["experimental_framework"] = await validate_experimental_framework()
    
    # Validate research validation
    logger.info("\n📊 Validating research validation system...")
    validation_results["research_validation"] = await validate_research_validation()
    
    # Validate autonomous engine
    logger.info("\n🤖 Validating autonomous research engine...")
    validation_results["autonomous_engine"] = await validate_autonomous_engine()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    for test_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL VALIDATIONS PASSED - RESEARCH SYSTEM READY!")
        report["status"] = "success"
    else:
        logger.info("⚠️  Some validations failed - see logs above")
        report["status"] = "partial_failure"
    
    # Save report
    report["validation_results"] = validation_results
    report["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests / total_tests * 100
    }
    
    report_path = Path("research_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Validation report saved to: {report_path}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
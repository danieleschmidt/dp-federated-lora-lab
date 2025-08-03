"""
Privacy-utility analysis example.

This example demonstrates how to analyze privacy-utility tradeoffs
in DP-Federated LoRA training experiments.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run privacy-utility analysis."""
    
    try:
        from dp_federated_lora import (
            UtilityMonitor,
            create_high_privacy_config,
            create_performance_config,
            create_default_config
        )
        
        print("🔍 Privacy-Utility Analysis Example")
        
        # Create utility monitor
        monitor = UtilityMonitor()
        
        # Simulate different privacy-utility experiments
        experiments = [
            {"name": "High Privacy", "epsilon": 1.0, "utility": 0.78},
            {"name": "Balanced", "epsilon": 4.0, "utility": 0.85}, 
            {"name": "Performance", "epsilon": 8.0, "utility": 0.89},
            {"name": "Low Privacy", "epsilon": 16.0, "utility": 0.92},
        ]
        
        print("\n📊 Recording experimental results...")
        
        # Record experimental data points
        for exp in experiments:
            monitor.record_utility_point({
                "epsilon": exp["epsilon"],
                "utility": exp["utility"],
                "accuracy": exp["utility"],
                "experiment": exp["name"]
            })
            print(f"   ✅ {exp['name']}: ε={exp['epsilon']}, utility={exp['utility']:.1%}")
        
        # Set baseline (non-private performance)
        baseline_utility = 0.95
        monitor.set_baseline_utility(baseline_utility)
        print(f"\n🎯 Baseline utility (non-private): {baseline_utility:.1%}")
        
        # Generate comprehensive report
        print("\n📋 Generating analysis report...")
        report = monitor.generate_report()
        
        # Display key findings
        print(f"\n🔍 Analysis Results:")
        print(f"   Total experiments: {report['summary']['total_experiments']}")
        print(f"   Epsilon range: {report['summary']['epsilon_range'][0]:.1f} - {report['summary']['epsilon_range'][1]:.1f}")
        print(f"   Utility range: {report['summary']['utility_range'][0]:.1%} - {report['summary']['utility_range'][1]:.1%}")
        print(f"   Privacy-utility correlation: {report['summary']['privacy_utility_correlation']:.3f}")
        
        # Optimal points
        best_efficiency = report['optimal_points']['best_efficiency']
        print(f"\n⭐ Best efficiency point:")
        print(f"   ε={best_efficiency['epsilon']:.1f}, utility={best_efficiency['utility']:.1%}")
        print(f"   Utility/Privacy ratio: {best_efficiency['ratio']:.2f}")
        
        max_utility = report['optimal_points']['max_utility']
        print(f"\n🎯 Maximum utility point:")
        print(f"   ε={max_utility['epsilon']:.1f}, utility={max_utility['utility']:.1%}")
        
        # Baseline comparison
        if 'baseline_comparison' in report:
            baseline = report['baseline_comparison']
            print(f"\n📉 Utility degradation vs baseline:")
            print(f"   Average degradation: {baseline['avg_degradation_percent']:.1f}%")
            print(f"   Best case degradation: {baseline['best_degradation_percent']:.1f}%")
        
        # Find optimal budget for target utility
        target_utility = 0.85
        optimal_epsilon = monitor.find_optimal_privacy_budget(target_utility, tolerance=0.02)
        
        if optimal_epsilon:
            print(f"\n🎯 For target utility {target_utility:.1%}:")
            print(f"   Recommended ε ≈ {optimal_epsilon:.1f}")
        
        # Save analysis results
        output_dir = Path("./privacy_analysis")
        output_dir.mkdir(exist_ok=True)
        
        analysis_file = output_dir / "privacy_utility_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Analysis saved to {analysis_file}")
        
        # Configuration recommendations
        print(f"\n🔧 Configuration Recommendations:")
        
        # High privacy scenario
        high_privacy_config = create_high_privacy_config()
        print(f"   High Privacy (ε≤{high_privacy_config.privacy.epsilon}):")
        print(f"     - Expected utility: ~78-82%")
        print(f"     - Recommended LoRA rank: {high_privacy_config.lora.r}")
        print(f"     - Rounds needed: {high_privacy_config.num_rounds}")
        
        # Balanced scenario  
        default_config = create_default_config()
        print(f"   Balanced (ε≤{default_config.privacy.epsilon}):")
        print(f"     - Expected utility: ~85-88%")
        print(f"     - Recommended LoRA rank: {default_config.lora.r}")
        print(f"     - Rounds needed: {default_config.num_rounds}")
        
        # Performance scenario
        perf_config = create_performance_config()
        print(f"   Performance (ε≤{perf_config.privacy.epsilon}):")
        print(f"     - Expected utility: ~89-92%")
        print(f"     - Recommended LoRA rank: {perf_config.lora.r}")
        print(f"     - Rounds needed: {perf_config.num_rounds}")
        
        print(f"\n✨ Privacy analysis completed!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"💡 Please install required packages first")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
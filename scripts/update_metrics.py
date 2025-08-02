#!/usr/bin/env python3
"""
Automated Metrics Update Script for dp-federated-lora-lab
Updates metrics with new values and calculates trends
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MetricsUpdater:
    """Updates project metrics and calculates trends"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.history_file = self.repo_path / "logs" / "metrics-history.json"
        
        # Ensure logs directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"Metrics file not found: {self.metrics_file}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load metrics file: {e}")
            return {}
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load metrics history"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.warning(f"Failed to load history file: {e}")
            return []
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save updated metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def save_history(self, history: List[Dict[str, Any]]) -> None:
        """Save metrics history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            logger.info(f"History saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def update_metric_value(self, metrics: Dict[str, Any], category: str, 
                          metric_name: str, new_value: float, 
                          target: Optional[float] = None) -> None:
        """Update a specific metric value"""
        try:
            if category not in metrics.get("metrics", {}):
                logger.warning(f"Category '{category}' not found in metrics")
                return
            
            if metric_name not in metrics["metrics"][category].get("metrics", {}):
                logger.warning(f"Metric '{metric_name}' not found in category '{category}'")
                return
            
            metric = metrics["metrics"][category]["metrics"][metric_name]
            old_value = metric.get("current", 0)
            
            # Update value
            metric["current"] = new_value
            if target is not None:
                metric["target"] = target
            
            # Calculate trend
            if new_value > old_value:
                trend = "increasing"
            elif new_value < old_value:
                trend = "decreasing"
            else:
                trend = "stable"
            
            metric["trend"] = trend
            metric["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Updated {category}.{metric_name}: {old_value} -> {new_value} ({trend})")
            
        except Exception as e:
            logger.error(f"Failed to update metric {category}.{metric_name}: {e}")
    
    def calculate_trend_analysis(self, history: List[Dict[str, Any]], 
                               category: str, metric_name: str) -> Dict[str, Any]:
        """Calculate trend analysis for a metric"""
        try:
            values = []
            timestamps = []
            
            for entry in history:
                if (category in entry.get("metrics", {}) and 
                    metric_name in entry["metrics"][category].get("metrics", {})):
                    
                    value = entry["metrics"][category]["metrics"][metric_name].get("current")
                    timestamp = entry.get("timestamp")
                    
                    if value is not None and timestamp:
                        values.append(float(value))
                        timestamps.append(timestamp)
            
            if len(values) < 2:
                return {"trend": "insufficient_data", "slope": 0, "r_squared": 0}
            
            # Simple linear regression for trend
            df = pd.DataFrame({"value": values, "timestamp": pd.to_datetime(timestamps)})
            df["time_numeric"] = df["timestamp"].astype(int) / 10**9  # Convert to seconds
            
            correlation = df["time_numeric"].corr(df["value"])
            
            # Calculate slope (simple approximation)
            if len(values) >= 2:
                slope = (values[-1] - values[0]) / len(values)
            else:
                slope = 0
            
            # Determine trend
            if abs(correlation) < 0.1:
                trend = "stable"
            elif correlation > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            return {
                "trend": trend,
                "slope": slope,
                "correlation": correlation,
                "data_points": len(values),
                "latest_value": values[-1] if values else 0,
                "earliest_value": values[0] if values else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trend for {category}.{metric_name}: {e}")
            return {"trend": "error", "slope": 0, "correlation": 0}
    
    def add_to_history(self, metrics: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add current metrics to history"""
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics.get("metrics", {})
        }
        
        history.append(history_entry)
        
        # Keep only last 100 entries to prevent file from growing too large
        if len(history) > 100:
            history = history[-100:]
        
        return history
    
    def generate_metrics_dashboard_data(self, metrics: Dict[str, Any], 
                                      history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data for metrics dashboard"""
        dashboard_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_metrics": 0,
                "healthy_metrics": 0,
                "warning_metrics": 0,
                "critical_metrics": 0
            },
            "categories": {},
            "trends": {}
        }
        
        try:
            for category, category_data in metrics.get("metrics", {}).items():
                if not isinstance(category_data, dict) or "metrics" not in category_data:
                    continue
                
                category_metrics = category_data["metrics"]
                dashboard_data["categories"][category] = {
                    "description": category_data.get("description", ""),
                    "metrics_count": len(category_metrics),
                    "metrics": {}
                }
                
                for metric_name, metric_data in category_metrics.items():
                    if not isinstance(metric_data, dict):
                        continue
                    
                    dashboard_data["summary"]["total_metrics"] += 1
                    
                    current = metric_data.get("current", 0)
                    target = metric_data.get("target")
                    
                    # Determine health status
                    if target is not None:
                        if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                            ratio = current / target if target != 0 else 1
                            if ratio >= 0.9:  # Within 90% of target
                                health = "healthy"
                                dashboard_data["summary"]["healthy_metrics"] += 1
                            elif ratio >= 0.7:  # Within 70% of target
                                health = "warning"
                                dashboard_data["summary"]["warning_metrics"] += 1
                            else:
                                health = "critical"
                                dashboard_data["summary"]["critical_metrics"] += 1
                        else:
                            health = "unknown"
                    else:
                        health = "no_target"
                        dashboard_data["summary"]["healthy_metrics"] += 1
                    
                    # Get trend analysis
                    trend_analysis = self.calculate_trend_analysis(history, category, metric_name)
                    
                    dashboard_data["categories"][category]["metrics"][metric_name] = {
                        "current": current,
                        "target": target,
                        "health": health,
                        "trend": metric_data.get("trend", "stable"),
                        "unit": metric_data.get("unit", "count"),
                        "description": metric_data.get("description", ""),
                        "last_updated": metric_data.get("last_updated", ""),
                        "trend_analysis": trend_analysis
                    }
                    
                    dashboard_data["trends"][f"{category}.{metric_name}"] = trend_analysis
        
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
        
        return dashboard_data
    
    def create_metrics_visualization(self, dashboard_data: Dict[str, Any], 
                                   output_dir: str = "reports") -> None:
        """Create visualizations for metrics"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create summary chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('DP-Federated LoRA Lab - Metrics Dashboard', fontsize=16, fontweight='bold')
            
            # Metrics health distribution
            summary = dashboard_data["summary"]
            health_data = [
                summary["healthy_metrics"],
                summary["warning_metrics"], 
                summary["critical_metrics"]
            ]
            health_labels = ["Healthy", "Warning", "Critical"]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            
            ax1.pie(health_data, labels=health_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Metrics Health Distribution')
            
            # Metrics by category
            categories = list(dashboard_data["categories"].keys())
            category_counts = [len(dashboard_data["categories"][cat]["metrics"]) for cat in categories]
            
            ax2.bar(categories, category_counts, color=sns.color_palette("husl", len(categories)))
            ax2.set_title('Metrics by Category')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Number of Metrics')
            ax2.tick_params(axis='x', rotation=45)
            
            # Trend distribution
            trends = [data.get("trend", "stable") for data in dashboard_data["trends"].values()]
            trend_counts = pd.Series(trends).value_counts()
            
            ax3.bar(trend_counts.index, trend_counts.values, color=["#3498db", "#e74c3c", "#2ecc71"])
            ax3.set_title('Trend Distribution')
            ax3.set_xlabel('Trend')
            ax3.set_ylabel('Count')
            
            # Sample metric over time (if history available)
            ax4.text(0.5, 0.5, f'Total Metrics: {summary["total_metrics"]}\\n'
                              f'Healthy: {summary["healthy_metrics"]}\\n'
                              f'Warning: {summary["warning_metrics"]}\\n'
                              f'Critical: {summary["critical_metrics"]}',
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Summary Statistics')
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / "metrics-dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics visualization saved to {output_path / 'metrics-dashboard.png'}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
    
    def run_update_cycle(self, updates: Optional[Dict[str, Any]] = None) -> None:
        """Run a complete metrics update cycle"""
        logger.info("Starting metrics update cycle...")
        
        # Load current metrics and history
        metrics = self.load_metrics()
        if not metrics:
            logger.error("No metrics found to update")
            return
        
        history = self.load_history()
        
        # Apply updates if provided
        if updates:
            for category, metric_updates in updates.items():
                for metric_name, update_data in metric_updates.items():
                    new_value = update_data.get("value")
                    target = update_data.get("target")
                    if new_value is not None:
                        self.update_metric_value(metrics, category, metric_name, new_value, target)
        
        # Add to history
        history = self.add_to_history(metrics, history)
        
        # Generate dashboard data
        dashboard_data = self.generate_metrics_dashboard_data(metrics, history)
        
        # Save everything
        self.save_metrics(metrics)
        self.save_history(history)
        
        # Save dashboard data
        dashboard_file = self.repo_path / "reports" / "dashboard-data.json"
        dashboard_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        # Create visualization
        self.create_metrics_visualization(dashboard_data)
        
        logger.info("Metrics update cycle completed successfully")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update project metrics")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--updates-file", help="JSON file with metric updates")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    updater = MetricsUpdater(args.repo_path)
    
    # Load updates if provided
    updates = None
    if args.updates_file:
        try:
            with open(args.updates_file, 'r') as f:
                updates = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load updates file: {e}")
            return
    
    # Run update cycle
    updater.run_update_cycle(updates)
    
    print("✅ Metrics update completed successfully")
    print(f"📊 Dashboard data saved to: {updater.repo_path}/reports/dashboard-data.json")
    
    if args.visualize:
        print(f"📈 Visualization saved to: {updater.repo_path}/reports/metrics-dashboard.png")


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "pandas", "seaborn"])
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    
    main()
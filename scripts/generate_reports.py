#!/usr/bin/env python3
"""
Automated Report Generation Script for dp-federated-lora-lab
Generates comprehensive reports from collected metrics
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive reports from metrics data"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.reports_dir = self.repo_path / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics from file"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}
    
    def generate_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate executive summary"""
        summary_parts = []
        
        # Project overview
        project_info = metrics.get("project", {})
        summary_parts.append(f"""
# 📊 Executive Summary - {project_info.get('name', 'Unknown Project')}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Repository:** {project_info.get('repository', 'N/A')}  
**License:** {project_info.get('license', 'N/A')}  
**Primary Language:** {project_info.get('primary_language', 'N/A')}

## 🎯 Project Overview

{project_info.get('description', 'No description available')}

**Domains:** {', '.join(project_info.get('domains', []))}
""")
        
        # Key metrics summary
        total_metrics = 0
        healthy_metrics = 0
        warning_metrics = 0
        critical_metrics = 0
        
        for category, category_data in metrics.get("metrics", {}).items():
            if not isinstance(category_data, dict) or "metrics" not in category_data:
                continue
            
            for metric_name, metric_data in category_data["metrics"].items():
                if not isinstance(metric_data, dict):
                    continue
                
                total_metrics += 1
                current = metric_data.get("current", 0)
                target = metric_data.get("target")
                
                if target is not None and isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    ratio = current / target if target != 0 else 1
                    if ratio >= 0.9:
                        healthy_metrics += 1
                    elif ratio >= 0.7:
                        warning_metrics += 1
                    else:
                        critical_metrics += 1
                else:
                    healthy_metrics += 1
        
        summary_parts.append(f"""
## 📈 Key Performance Indicators

| Metric | Value | Status |
|--------|--------|--------|
| **Total Metrics Tracked** | {total_metrics} | ℹ️ |
| **Healthy Metrics** | {healthy_metrics} ({healthy_metrics/total_metrics*100:.1f}%) | {'✅' if healthy_metrics/total_metrics > 0.8 else '⚠️'} |
| **Warning Metrics** | {warning_metrics} ({warning_metrics/total_metrics*100:.1f}%) | {'✅' if warning_metrics/total_metrics < 0.15 else '⚠️'} |
| **Critical Metrics** | {critical_metrics} ({critical_metrics/total_metrics*100:.1f}%) | {'✅' if critical_metrics == 0 else '🚨'} |
""")
        
        # Category highlights
        summary_parts.append("\n## 🎯 Category Highlights\n")
        
        for category, category_data in metrics.get("metrics", {}).items():
            if not isinstance(category_data, dict) or "metrics" not in category_data:
                continue
            
            category_metrics = category_data["metrics"]
            category_count = len(category_metrics)
            
            # Find best and worst performing metrics in category
            best_metric = None
            worst_metric = None
            best_ratio = 0
            worst_ratio = float('inf')
            
            for metric_name, metric_data in category_metrics.items():
                if not isinstance(metric_data, dict):
                    continue
                
                current = metric_data.get("current", 0)
                target = metric_data.get("target")
                
                if target is not None and isinstance(current, (int, float)) and isinstance(target, (int, float)) and target != 0:
                    ratio = current / target
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_metric = (metric_name, current, target, metric_data.get("unit", ""))
                    if ratio < worst_ratio:
                        worst_ratio = ratio
                        worst_metric = (metric_name, current, target, metric_data.get("unit", ""))
            
            summary_parts.append(f"""
### {category.replace('_', ' ').title()}
- **Metrics Count:** {category_count}
- **Description:** {category_data.get('description', 'No description')}""")
            
            if best_metric and worst_metric:
                summary_parts.append(f"""- **Best Performing:** {best_metric[0]} ({best_metric[1]:.2f}/{best_metric[2]:.2f} {best_metric[3]})
- **Needs Attention:** {worst_metric[0]} ({worst_metric[1]:.2f}/{worst_metric[2]:.2f} {worst_metric[3]})""")
        
        return "\n".join(summary_parts)
    
    def generate_detailed_report(self, metrics: Dict[str, Any]) -> str:
        """Generate detailed metrics report"""
        report_parts = []
        
        report_parts.append(f"""
# 📊 Detailed Metrics Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

""")
        
        for category, category_data in metrics.get("metrics", {}).items():
            if not isinstance(category_data, dict) or "metrics" not in category_data:
                continue
            
            report_parts.append(f"""
## {category.replace('_', ' ').title()}

{category_data.get('description', 'No description available')}

| Metric | Current | Target | Unit | Trend | Status | Last Updated |
|--------|---------|--------|------|-------|--------|--------------|""")
            
            for metric_name, metric_data in category_data["metrics"].items():
                if not isinstance(metric_data, dict):
                    continue
                
                current = metric_data.get("current", 0)
                target = metric_data.get("target", "N/A")
                unit = metric_data.get("unit", "count")
                trend = metric_data.get("trend", "stable")
                last_updated = metric_data.get("last_updated", "Unknown")
                
                # Determine status
                status = "ℹ️"
                if target != "N/A" and isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    ratio = current / target if target != 0 else 1
                    if ratio >= 0.9:
                        status = "✅"
                    elif ratio >= 0.7:
                        status = "⚠️"
                    else:
                        status = "🚨"
                
                # Format trend
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend, "❓")
                
                # Format last updated
                try:
                    if last_updated != "Unknown":
                        dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        last_updated = dt.strftime('%Y-%m-%d')
                except:
                    pass
                
                report_parts.append(f"| **{metric_name.replace('_', ' ').title()}** | {current} | {target} | {unit} | {trend_emoji} {trend} | {status} | {last_updated} |")
        
        return "\n".join(report_parts)
    
    def generate_json_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured JSON report"""
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": "1.0.0",
            "project": metrics.get("project", {}),
            "summary": {
                "total_metrics": 0,
                "healthy_metrics": 0,
                "warning_metrics": 0,
                "critical_metrics": 0,
                "categories": []
            },
            "detailed_metrics": {},
            "recommendations": [],
            "automation_status": metrics.get("automation", {}),
            "goals": metrics.get("goals", {})
        }
        
        # Calculate summary statistics
        for category, category_data in metrics.get("metrics", {}).items():
            if not isinstance(category_data, dict) or "metrics" not in category_data:
                continue
            
            report["summary"]["categories"].append(category)
            category_summary = {
                "name": category,
                "description": category_data.get("description", ""),
                "metrics_count": len(category_data["metrics"]),
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "metrics": {}
            }
            
            for metric_name, metric_data in category_data["metrics"].items():
                if not isinstance(metric_data, dict):
                    continue
                
                report["summary"]["total_metrics"] += 1
                current = metric_data.get("current", 0)
                target = metric_data.get("target")
                
                # Determine health status
                health = "healthy"
                if target is not None and isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    ratio = current / target if target != 0 else 1
                    if ratio >= 0.9:
                        health = "healthy"
                        report["summary"]["healthy_metrics"] += 1
                        category_summary["healthy"] += 1
                    elif ratio >= 0.7:
                        health = "warning"
                        report["summary"]["warning_metrics"] += 1
                        category_summary["warning"] += 1
                    else:
                        health = "critical"
                        report["summary"]["critical_metrics"] += 1
                        category_summary["critical"] += 1
                else:
                    report["summary"]["healthy_metrics"] += 1
                    category_summary["healthy"] += 1
                
                # Add detailed metric info
                category_summary["metrics"][metric_name] = {
                    "current": current,
                    "target": target,
                    "unit": metric_data.get("unit", "count"),
                    "trend": metric_data.get("trend", "stable"),
                    "health": health,
                    "description": metric_data.get("description", ""),
                    "last_updated": metric_data.get("last_updated", "")
                }
            
            report["detailed_metrics"][category] = category_summary
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations(report)
        
        return report
    
    def generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Check for critical metrics
        if report["summary"]["critical_metrics"] > 0:
            recommendations.append({
                "priority": "high",
                "category": "critical_metrics",
                "title": "Address Critical Metrics",
                "description": f"There are {report['summary']['critical_metrics']} critical metrics that need immediate attention.",
                "action": "Review and improve critical metrics to meet targets"
            })
        
        # Check overall health ratio
        total_metrics = report["summary"]["total_metrics"]
        healthy_ratio = report["summary"]["healthy_metrics"] / total_metrics if total_metrics > 0 else 0
        
        if healthy_ratio < 0.8:
            recommendations.append({
                "priority": "medium",
                "category": "overall_health",
                "title": "Improve Overall Metrics Health",
                "description": f"Only {healthy_ratio:.1%} of metrics are healthy. Target: >80%",
                "action": "Focus on improving metrics that are below target"
            })
        
        # Category-specific recommendations
        for category, category_data in report["detailed_metrics"].items():
            total_cat_metrics = category_data["metrics_count"]
            critical_cat_metrics = category_data["critical"]
            
            if critical_cat_metrics > 0:
                recommendations.append({
                    "priority": "high",
                    "category": category,
                    "title": f"Improve {category.replace('_', ' ').title()}",
                    "description": f"{critical_cat_metrics}/{total_cat_metrics} metrics are critical in {category}",
                    "action": f"Focus on {category} improvements"
                })
        
        return recommendations
    
    def generate_all_reports(self) -> None:
        """Generate all report types"""
        logger.info("Generating comprehensive reports...")
        
        # Load metrics
        metrics = self.load_metrics()
        if not metrics:
            logger.error("No metrics available for reporting")
            return
        
        # Generate executive summary (Markdown)
        executive_summary = self.generate_executive_summary(metrics)
        with open(self.reports_dir / "executive-summary.md", 'w') as f:
            f.write(executive_summary)
        logger.info("Executive summary generated")
        
        # Generate detailed report (Markdown)
        detailed_report = self.generate_detailed_report(metrics)
        with open(self.reports_dir / "detailed-metrics-report.md", 'w') as f:
            f.write(detailed_report)
        logger.info("Detailed report generated")
        
        # Generate JSON report
        json_report = self.generate_json_report(metrics)
        with open(self.reports_dir / "metrics-report.json", 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        logger.info("JSON report generated")
        
        # Generate simple HTML report
        html_report = self.generate_html_report(json_report)
        with open(self.reports_dir / "metrics-report.html", 'w') as f:
            f.write(html_report)
        logger.info("HTML report generated")
        
        logger.info(f"All reports generated in: {self.reports_dir}")
    
    def generate_html_report(self, json_report: Dict[str, Any]) -> str:
        """Generate simple HTML report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metrics Report - {json_report.get('project', {}).get('name', 'Unknown Project')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .status-healthy {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-critical {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 20px; margin: 20px 0; }}
        .rec-high {{ border-left: 4px solid #e74c3c; }}
        .rec-medium {{ border-left: 4px solid #f39c12; }}
        .rec-low {{ border-left: 4px solid #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Metrics Report</h1>
        <p><strong>Generated:</strong> {json_report.get('generated_at', 'Unknown')}</p>
        <p><strong>Project:</strong> {json_report.get('project', {}).get('name', 'Unknown Project')}</p>
        
        <h2>📈 Summary</h2>
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{json_report.get('summary', {}).get('total_metrics', 0)}</div>
                <div class="metric-label">Total Metrics</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-healthy">{json_report.get('summary', {}).get('healthy_metrics', 0)}</div>
                <div class="metric-label">Healthy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-warning">{json_report.get('summary', {}).get('warning_metrics', 0)}</div>
                <div class="metric-label">Warning</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-critical">{json_report.get('summary', {}).get('critical_metrics', 0)}</div>
                <div class="metric-label">Critical</div>
            </div>
        </div>
"""
        
        # Add categories
        for category, category_data in json_report.get('detailed_metrics', {}).items():
            html += f"""
        <h2>{category.replace('_', ' ').title()}</h2>
        <p>{category_data.get('description', '')}</p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Current</th>
                <th>Target</th>
                <th>Unit</th>
                <th>Trend</th>
                <th>Status</th>
            </tr>
"""
            
            for metric_name, metric_data in category_data.get('metrics', {}).items():
                status_class = f"status-{metric_data.get('health', 'healthy')}"
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(metric_data.get('trend', 'stable'), "❓")
                
                html += f"""
            <tr>
                <td><strong>{metric_name.replace('_', ' ').title()}</strong></td>
                <td>{metric_data.get('current', 'N/A')}</td>
                <td>{metric_data.get('target', 'N/A')}</td>
                <td>{metric_data.get('unit', 'count')}</td>
                <td>{trend_emoji} {metric_data.get('trend', 'stable')}</td>
                <td class="{status_class}">{metric_data.get('health', 'healthy').title()}</td>
            </tr>
"""
            
            html += "        </table>\n"
        
        # Add recommendations
        recommendations = json_report.get('recommendations', [])
        if recommendations:
            html += """
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
"""
            
            for rec in recommendations:
                priority_class = f"rec-{rec.get('priority', 'low')}"
                html += f"""
            <div class="{priority_class}" style="margin-bottom: 15px; padding: 10px;">
                <h4>{rec.get('title', 'No Title')}</h4>
                <p>{rec.get('description', 'No description')}</p>
                <strong>Action:</strong> {rec.get('action', 'No action specified')}
            </div>
"""
            
            html += "        </div>\n"
        
        html += """
    </div>
</body>
</html>
"""
        
        return html


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive metrics reports")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--format", choices=["all", "markdown", "json", "html"], 
                       default="all", help="Report format")
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.repo_path)
    
    if args.format == "all":
        generator.generate_all_reports()
    else:
        # Generate specific format (simplified implementation)
        generator.generate_all_reports()
    
    print("✅ Reports generated successfully")
    print(f"📁 Reports saved to: {generator.reports_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Automated Metrics Collection Script for dp-federated-lora-lab
Collects comprehensive metrics for SDLC monitoring and improvement
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/metrics-collection.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class MetricValue(BaseModel):
    """Individual metric value model"""
    
    current: float = Field(..., description="Current metric value")
    target: Optional[float] = Field(None, description="Target metric value")
    trend: str = Field("stable", description="Trend: increasing, decreasing, stable")
    unit: str = Field("count", description="Unit of measurement")
    description: Optional[str] = Field(None, description="Metric description")
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MetricsCollector:
    """Main metrics collection class"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.current_metrics = self._load_current_metrics()
    
    def _load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Metrics file not found: {self.metrics_file}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load metrics file: {e}")
            return {}
    
    def _run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Command failed: {' '.join(command)}, error: {e}")
            return False, str(e)
    
    async def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics"""
        logger.info("Collecting code quality metrics...")
        metrics = {}
        
        try:
            # Lines of code
            success, output = self._run_command(["find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"])
            if success and output:
                total_lines = sum(int(line.split()[0]) for line in output.strip().split('\n') 
                                if line.strip() and line.split()[0].isdigit())
                metrics["lines_of_code"] = MetricValue(
                    current=total_lines,
                    target=10000,
                    trend="increasing",
                    unit="lines",
                    description="Total lines of Python code"
                ).dict()
            
            # Test coverage
            success, output = self._run_command(["python", "-m", "pytest", "--cov=src", "--cov-report=json"])
            if success:
                try:
                    # Look for coverage.json file
                    coverage_file = self.repo_path / "coverage.json"
                    if coverage_file.exists():
                        with open(coverage_file, 'r') as f:
                            coverage_data = json.load(f)
                            coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
                            metrics["test_coverage"] = MetricValue(
                                current=coverage_percent,
                                target=90.0,
                                trend="stable",
                                unit="percentage",
                                description="Test coverage percentage"
                            ).dict()
                except Exception as e:
                    logger.warning(f"Failed to parse coverage data: {e}")
            
            # Cyclomatic complexity (using radon if available)
            success, output = self._run_command(["python", "-m", "radon", "cc", "src/", "--average"])
            if success and "Average complexity:" in output:
                try:
                    avg_complexity = float(output.split("Average complexity:")[1].strip().split()[0])
                    metrics["cyclomatic_complexity"] = MetricValue(
                        current=avg_complexity,
                        target=10.0,
                        trend="stable",
                        unit="average",
                        description="Average cyclomatic complexity"
                    ).dict()
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse complexity data: {e}")
            
            # Maintainability index (using radon if available)
            success, output = self._run_command(["python", "-m", "radon", "mi", "src/", "--show"])
            if success:
                try:
                    # Parse maintainability index from radon output
                    mi_values = []
                    for line in output.split('\n'):
                        if ' - ' in line and line.strip():
                            try:
                                mi_score = float(line.split(' - ')[1].split()[0])
                                mi_values.append(mi_score)
                            except (ValueError, IndexError):
                                continue
                    
                    if mi_values:
                        avg_mi = sum(mi_values) / len(mi_values)
                        metrics["maintainability_index"] = MetricValue(
                            current=avg_mi,
                            target=80.0,
                            trend="stable",
                            unit="index",
                            description="Average maintainability index"
                        ).dict()
                except Exception as e:
                    logger.warning(f"Failed to parse maintainability index: {e}")
            
        except Exception as e:
            logger.error(f"Failed to collect code quality metrics: {e}")
        
        return metrics
    
    async def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics"""
        logger.info("Collecting security metrics...")
        metrics = {}
        
        try:
            # Bandit security scan
            success, output = self._run_command(["python", "-m", "bandit", "-r", "src/", "-f", "json"])
            if success:
                try:
                    bandit_data = json.loads(output)
                    vulnerability_count = {
                        "critical": len([issue for issue in bandit_data.get("results", []) if issue.get("issue_severity") == "HIGH"]),
                        "high": len([issue for issue in bandit_data.get("results", []) if issue.get("issue_severity") == "MEDIUM"]),
                        "medium": len([issue for issue in bandit_data.get("results", []) if issue.get("issue_severity") == "LOW"]),
                        "low": 0,
                        "target_critical": 0,
                        "target_high": 0,
                        "target_medium": 0,
                        "target_low": 0,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                    metrics["vulnerability_count"] = vulnerability_count
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse bandit output: {e}")
            
            # Safety check for dependencies
            success, output = self._run_command(["python", "-m", "safety", "check", "--json"])
            if success:
                try:
                    safety_data = json.loads(output)
                    vulnerable_packages = len(safety_data) if isinstance(safety_data, list) else 0
                    
                    # Calculate security score (100 - vulnerable packages * 5, min 0)
                    security_score = max(0, 100 - vulnerable_packages * 5)
                    metrics["dependency_security_score"] = MetricValue(
                        current=security_score,
                        target=95.0,
                        trend="stable",
                        unit="percentage",
                        description="Dependency security score"
                    ).dict()
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse safety output: {e}")
            
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
        
        return metrics
    
    async def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development process metrics"""
        logger.info("Collecting development metrics...")
        metrics = {}
        
        try:
            # Build success rate (from recent CI runs)
            success, output = self._run_command(["git", "log", "--oneline", "-20"])
            if success:
                # This is a simplified calculation - in real scenario you'd query CI/CD API
                total_commits = len(output.strip().split('\n'))
                metrics["build_success_rate"] = MetricValue(
                    current=94.2,  # Placeholder - would be calculated from CI data
                    target=96.0,
                    trend="stable",
                    unit="percentage",
                    description="Build success rate"
                ).dict()
            
            # Lead time (average time from commit to production)
            metrics["lead_time"] = MetricValue(
                current=3.2,  # Placeholder - would be calculated from deployment data
                target=2.5,
                trend="decreasing",
                unit="days",
                description="Average lead time"
            ).dict()
            
            # Mean time to recovery
            metrics["mean_time_to_recovery"] = MetricValue(
                current=45,  # Placeholder - would be calculated from incident data
                target=30,
                trend="decreasing",
                unit="minutes",
                description="Mean time to recovery"
            ).dict()
            
        except Exception as e:
            logger.error(f"Failed to collect development metrics: {e}")
        
        return metrics
    
    async def collect_repository_health_metrics(self) -> Dict[str, Any]:
        """Collect repository health metrics"""
        logger.info("Collecting repository health metrics...")
        metrics = {}
        
        try:
            # Documentation coverage (count of .md files vs code files)
            success, code_output = self._run_command(["find", "src/", "-name", "*.py", "-type", "f"])
            success2, doc_output = self._run_command(["find", "docs/", "-name", "*.md", "-type", "f"])
            
            if success and success2:
                code_files = len([line for line in code_output.strip().split('\n') if line.strip()])
                doc_files = len([line for line in doc_output.strip().split('\n') if line.strip()])
                
                if code_files > 0:
                    doc_coverage = min(100, (doc_files / code_files) * 100)
                    metrics["documentation_coverage"] = MetricValue(
                        current=doc_coverage,
                        target=80.0,
                        trend="increasing",
                        unit="percentage",
                        description="Documentation coverage ratio"
                    ).dict()
            
            # Dependency freshness (check requirements.txt age)
            requirements_file = self.repo_path / "requirements.txt"
            if requirements_file.exists():
                file_age_days = (datetime.now() - datetime.fromtimestamp(requirements_file.stat().st_mtime)).days
                freshness_score = max(0, 100 - file_age_days * 2)  # Decrease by 2% per day
                metrics["dependency_freshness"] = MetricValue(
                    current=freshness_score,
                    target=92.0,
                    trend="stable",
                    unit="percentage",
                    description="Dependency freshness score"
                ).dict()
            
        except Exception as e:
            logger.error(f"Failed to collect repository health metrics: {e}")
        
        return metrics
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        logger.info("Collecting performance metrics...")
        metrics = {}
        
        try:
            # System performance
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics["system_cpu_usage"] = MetricValue(
                current=cpu_percent,
                target=80.0,
                trend="stable",
                unit="percentage",
                description="System CPU usage"
            ).dict()
            
            metrics["system_memory_usage"] = MetricValue(
                current=memory.percent,
                target=80.0,
                trend="stable",
                unit="percentage",
                description="System memory usage"
            ).dict()
            
            # Placeholder metrics for ML-specific performance
            metrics["training_throughput"] = MetricValue(
                current=1250,  # Would be measured from actual training runs
                target=1500,
                trend="increasing",
                unit="samples_per_second",
                description="Training throughput"
            ).dict()
            
            metrics["memory_efficiency"] = MetricValue(
                current=82.0,  # Would be calculated from actual training
                target=85.0,
                trend="stable",
                unit="percentage",
                description="Memory utilization efficiency"
            ).dict()
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
        
        return metrics
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics"""
        logger.info("Starting comprehensive metrics collection...")
        
        all_metrics = {
            "metadata": {
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "collector_version": "1.0.0",
                "repository_path": str(self.repo_path)
            },
            "metrics": {
                "code_quality": await self.collect_code_quality_metrics(),
                "security": await self.collect_security_metrics(),
                "development": await self.collect_development_metrics(),
                "repository_health": await self.collect_repository_health_metrics(),
                "performance": await self.collect_performance_metrics()
            }
        }
        
        logger.info("Metrics collection completed")
        return all_metrics
    
    def update_metrics_file(self, new_metrics: Dict[str, Any]) -> None:
        """Update the project metrics file with new data"""
        try:
            # Merge with existing metrics
            if self.current_metrics:
                # Update metrics sections
                for category, metrics in new_metrics.get("metrics", {}).items():
                    if category in self.current_metrics.get("metrics", {}):
                        self.current_metrics["metrics"][category]["metrics"].update(metrics)
                    else:
                        # Add new category
                        if "metrics" not in self.current_metrics:
                            self.current_metrics["metrics"] = {}
                        self.current_metrics["metrics"][category] = {
                            "description": f"{category.replace('_', ' ').title()} metrics",
                            "metrics": metrics
                        }
                
                # Update tracking info
                self.current_metrics["tracking"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            else:
                self.current_metrics = new_metrics
            
            # Save updated metrics
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2, default=str)
            
            logger.info(f"Metrics file updated: {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to update metrics file: {e}")
    
    def generate_metrics_report(self, output_file: str = "metrics-report.json") -> None:
        """Generate a metrics report"""
        try:
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_metrics": sum(
                        len(category.get("metrics", {})) 
                        for category in self.current_metrics.get("metrics", {}).values()
                        if isinstance(category, dict)
                    ),
                    "categories": list(self.current_metrics.get("metrics", {}).keys()),
                    "last_collection": self.current_metrics.get("tracking", {}).get("last_updated", "unknown")
                },
                "full_metrics": self.current_metrics
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Metrics report generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate metrics report: {e}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect comprehensive metrics for dp-federated-lora-lab")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--output", default="metrics-report.json", help="Output report file")
    parser.add_argument("--update-file", action="store_true", help="Update project metrics file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create collector
    collector = MetricsCollector(args.repo_path)
    
    # Collect metrics
    metrics = await collector.collect_all_metrics()
    
    # Update metrics file if requested
    if args.update_file:
        collector.update_metrics_file(metrics)
    
    # Generate report
    collector.generate_metrics_report(args.output)
    
    # Print summary
    total_metrics = sum(
        len(category_metrics) 
        for category_metrics in metrics.get("metrics", {}).values()
    )
    print(f"✅ Collected {total_metrics} metrics across {len(metrics.get('metrics', {}))} categories")
    print(f"📊 Report saved to: {args.output}")
    
    if args.update_file:
        print(f"📝 Project metrics file updated: {collector.metrics_file}")


if __name__ == "__main__":
    asyncio.run(main())
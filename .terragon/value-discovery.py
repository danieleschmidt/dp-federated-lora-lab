#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for SDLC Enhancement
Implements continuous value discovery with WSJF+ICE+TechnicalDebt scoring.
"""

import json
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
try:
    import yaml
except ImportError:
    # Simplified fallback if yaml not available
    class SimpleYAML:
        @staticmethod
        def safe_load(content):
            # Very basic YAML parser for our config structure
            lines = content.strip().split('\n')
            result = {}
            current_dict = result
            indent_stack = [result]
            
            for line in lines:
                if ':' in line and not line.strip().startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value:
                        # Try to convert numbers
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                    
                    current_dict[key] = value
            
            return result
    
    yaml = SimpleYAML()


@dataclass
class ValueItem:
    """Represents a discovered value item."""
    id: str
    title: str
    description: str
    category: str
    type: str
    files: List[str]
    estimated_effort: float  # hours
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    created_at: str
    priority: str
    dependencies: List[str]
    risk_level: float


class ValueDiscoveryEngine:
    """Continuous value discovery and prioritization engine."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = repo_path / "BACKLOG.md"
        
        # Load configuration
        with open(self.config_path) as f:
            content = f.read()
            self.config = yaml.safe_load(content)
        
        # Load existing metrics
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing value metrics."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {
            "executionHistory": [],
            "backlogMetrics": {
                "totalItems": 0,
                "averageAge": 0,
                "debtRatio": 0,
                "velocityTrend": "stable"
            },
            "discoveryStats": {
                "itemsDiscovered": 0,
                "itemsCompleted": 0,
                "netChange": 0
            }
        }
    
    def discover_value_items(self) -> List[ValueItem]:
        """Execute comprehensive value discovery."""
        items = []
        
        # 1. Git history analysis
        items.extend(self._discover_from_git_history())
        
        # 2. Static analysis
        items.extend(self._discover_from_static_analysis())
        
        # 3. Configuration analysis
        items.extend(self._discover_from_config_analysis())
        
        # 4. Test analysis
        items.extend(self._discover_from_test_analysis())
        
        # 5. Security analysis
        items.extend(self._discover_from_security_analysis())
        
        # Score and prioritize
        for item in items:
            self._calculate_scores(item)
        
        # Sort by composite score
        items.sort(key=lambda x: x.composite_score, reverse=True)
        
        return items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover items from git commit history."""
        items = []
        
        try:
            # Get recent commits with TODO/FIXME patterns
            result = subprocess.run([
                "git", "log", "--oneline", "-n", "50", "--grep=TODO\\|FIXME\\|HACK\\|temp"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.strip().split('\\n'):
                if line:
                    commit_hash = line.split()[0]
                    message = " ".join(line.split()[1:])
                    
                    items.append(ValueItem(
                        id=f"git-{commit_hash[:8]}",
                        title=f"Address technical debt in: {message[:50]}",
                        description=f"Commit message indicates technical debt: {message}",
                        category="technical_debt",
                        type="refactoring",
                        files=[],
                        estimated_effort=2.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        priority="medium",
                        dependencies=[],
                        risk_level=0.3
                    ))
        except subprocess.CalledProcessError:
            pass
        
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static analysis."""
        items = []
        
        # Import statement analysis
        init_file = self.repo_path / "src" / "dp_federated_lora" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            
            # Check for non-existent imports
            import_lines = re.findall(r"from \\.(\\w+) import", content)
            for module in import_lines:
                module_file = self.repo_path / "src" / "dp_federated_lora" / f"{module}.py"
                if not module_file.exists():
                    items.append(ValueItem(
                        id=f"missing-module-{module}",
                        title=f"Implement missing module: {module}",
                        description=f"Module {module} is imported but doesn't exist",
                        category="implementation",
                        type="feature_development",
                        files=[str(module_file)],
                        estimated_effort=8.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        priority="high",
                        dependencies=[],
                        risk_level=0.2
                    ))
        
        return items
    
    def _discover_from_config_analysis(self) -> List[ValueItem]:
        """Discover items from configuration analysis."""
        items = []
        
        # Check for missing GitHub Actions
        gh_actions_dir = self.repo_path / ".github" / "workflows"
        if not gh_actions_dir.exists():
            items.append(ValueItem(
                id="setup-github-actions",
                title="Implement GitHub Actions CI/CD workflows",
                description="Repository has workflow documentation but no actual GitHub Actions",
                category="cicd",
                type="automation",
                files=[".github/workflows/ci.yml", ".github/workflows/release.yml"],
                estimated_effort=4.0,
                wsjf_score=0,
                ice_score=0,
                technical_debt_score=0,
                composite_score=0,
                created_at=datetime.now(timezone.utc).isoformat(),
                priority="high",
                dependencies=[],
                risk_level=0.1
            ))
        
        # Check for mutation testing setup
        mutmut_config = self.repo_path / "pytest-mutmut.toml"
        if mutmut_config.exists():
            items.append(ValueItem(
                id="implement-mutation-testing",
                title="Set up mutation testing with mutmut",
                description="Configuration exists but mutation testing not integrated into CI",
                category="testing",
                type="quality_improvement",
                files=["pytest-mutmut.toml"],
                estimated_effort=3.0,
                wsjf_score=0,
                ice_score=0,
                technical_debt_score=0,
                composite_score=0,
                created_at=datetime.now(timezone.utc).isoformat(),
                priority="medium",
                dependencies=["setup-github-actions"],
                risk_level=0.2
            ))
        
        return items
    
    def _discover_from_test_analysis(self) -> List[ValueItem]:
        """Discover items from test analysis."""
        items = []
        
        # Check test coverage
        try:
            result = subprocess.run([
                "python3", "-m", "pytest", "--cov=src", "--cov-report=term-missing", "--tb=no", "-q"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Parse coverage output
            coverage_match = re.search(r"TOTAL\\s+(\\d+)%", result.stdout)
            if coverage_match:
                coverage = int(coverage_match.group(1))
                if coverage < 80:
                    items.append(ValueItem(
                        id="improve-test-coverage",
                        title=f"Improve test coverage from {coverage}% to 80%",
                        description=f"Current test coverage is {coverage}%, below target of 80%",
                        category="testing",
                        type="quality_improvement",
                        files=["tests/"],
                        estimated_effort=6.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        priority="medium",
                        dependencies=[],
                        risk_level=0.3
                    ))
        except subprocess.CalledProcessError:
            pass
        
        return items
    
    def _discover_from_security_analysis(self) -> List[ValueItem]:
        """Discover items from security analysis."""
        items = []
        
        try:
            # Run safety check
            result = subprocess.run([
                "python3", "-m", "safety", "check", "--json", "--ignore", "52365"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0 and result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    items.append(ValueItem(
                        id=f"security-{vuln['advisory'][:8]}",
                        title=f"Fix security vulnerability in {vuln['package_name']}",
                        description=f"Vulnerability: {vuln['advisory']}",
                        category="security",
                        type="security_fix",
                        files=["requirements.txt"],
                        estimated_effort=1.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        priority="high",
                        dependencies=[],
                        risk_level=0.1
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        return items
    
    def _calculate_scores(self, item: ValueItem) -> None:
        """Calculate WSJF, ICE, and composite scores for an item."""
        # WSJF Components
        user_business_value = self._score_user_impact(item)
        time_criticality = self._score_urgency(item)
        risk_reduction = self._score_risk_mitigation(item)
        opportunity_enablement = self._score_opportunity(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        
        job_size = item.estimated_effort
        item.wsjf_score = cost_of_delay / max(job_size, 0.5)
        
        # ICE Components
        impact = self._score_business_impact(item)
        confidence = self._score_execution_confidence(item)
        ease = self._score_implementation_ease(item)
        
        item.ice_score = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = self._calculate_debt_cost(item)
        debt_interest = self._calculate_debt_growth(item)
        hotspot_multiplier = self._get_churn_complexity(item)
        
        item.technical_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Composite Score with adaptive weights
        weights = self.config["scoring"]["weights"]
        
        normalized_wsjf = min(item.wsjf_score / 50, 1.0)
        normalized_ice = min(item.ice_score / 1000, 1.0)
        normalized_debt = min(item.technical_debt_score / 100, 1.0)
        
        item.composite_score = (
            weights["wsjf"] * normalized_wsjf * 100 +
            weights["ice"] * normalized_ice * 100 +
            weights["technicalDebt"] * normalized_debt * 100
        )
        
        # Apply boosts
        if item.category == "security":
            item.composite_score *= self.config["scoring"]["thresholds"]["securityBoost"]
        
        # Update priority based on score
        if item.composite_score > 80:
            item.priority = "high"
        elif item.composite_score > 50:
            item.priority = "medium"
        else:
            item.priority = "low"
    
    def _score_user_impact(self, item: ValueItem) -> float:
        """Score user business value impact (1-10)."""
        category_scores = {
            "security": 9,
            "performance": 7,
            "testing": 6,
            "cicd": 8,
            "technical_debt": 5,
            "implementation": 7,
            "documentation": 4
        }
        return category_scores.get(item.category, 5)
    
    def _score_urgency(self, item: ValueItem) -> float:
        """Score time criticality (1-10)."""
        if item.category == "security":
            return 9
        elif item.category == "cicd":
            return 7
        return 5
    
    def _score_risk_mitigation(self, item: ValueItem) -> float:
        """Score risk reduction (1-10)."""
        return (1 - item.risk_level) * 10
    
    def _score_opportunity(self, item: ValueItem) -> float:
        """Score opportunity enablement (1-10)."""
        if item.dependencies:
            return 8  # Enables other work
        return 4
    
    def _score_business_impact(self, item: ValueItem) -> float:
        """Score business impact for ICE (1-10)."""
        return self._score_user_impact(item)
    
    def _score_execution_confidence(self, item: ValueItem) -> float:
        """Score execution confidence for ICE (1-10)."""
        if item.estimated_effort <= 2:
            return 9
        elif item.estimated_effort <= 8:
            return 7
        return 5
    
    def _score_implementation_ease(self, item: ValueItem) -> float:
        """Score implementation ease for ICE (1-10)."""
        return 10 - min(item.estimated_effort, 8)
    
    def _calculate_debt_cost(self, item: ValueItem) -> float:
        """Calculate technical debt cost."""
        if item.category == "technical_debt":
            return item.estimated_effort * 2
        return item.estimated_effort * 0.5
    
    def _calculate_debt_growth(self, item: ValueItem) -> float:
        """Calculate debt growth cost."""
        if item.category == "technical_debt":
            return item.estimated_effort * 1.5
        return 0
    
    def _get_churn_complexity(self, item: ValueItem) -> float:
        """Get churn complexity multiplier."""
        # Simplified - would normally analyze git file change frequency
        return 1.0
    
    def generate_backlog(self, items: List[ValueItem]) -> None:
        """Generate BACKLOG.md with discovered items."""
        now = datetime.now(timezone.utc)
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Next Execution: Continuous

## 🎯 Next Best Value Item
"""
        
        if items:
            top_item = items[0]
            content += f"""**[{top_item.id.upper()}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.0f} | **Tech Debt**: {top_item.technical_debt_score:.1f}
- **Estimated Effort**: {top_item.estimated_effort} hours
- **Category**: {top_item.category.replace('_', ' ').title()}

"""
        
        content += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(items[:10], 1):
            content += f"| {i} | {item.id} | {item.title[:40]}... | {item.composite_score:.1f} | {item.category.title()} | {item.estimated_effort} |\\n"
        
        content += f"""

## 📈 Value Metrics
- **Items Discovered**: {len(items)}
- **High Priority Items**: {len([i for i in items if i.priority == 'high'])}
- **Security Items**: {len([i for i in items if i.category == 'security'])}
- **Technical Debt Items**: {len([i for i in items if i.category == 'technical_debt'])}

## 🔄 Discovery Sources
- **Static Analysis**: {len([i for i in items if 'static' in i.id])}%
- **Git History**: {len([i for i in items if i.id.startswith('git')])}%
- **Configuration**: {len([i for i in items if 'config' in i.category])}%
- **Security Scans**: {len([i for i in items if i.category == 'security'])}%
"""
        
        self.backlog_path.write_text(content)
    
    def save_metrics(self, items: List[ValueItem]) -> None:
        """Save updated metrics."""
        self.metrics["backlogMetrics"]["totalItems"] = len(items)
        self.metrics["discoveryStats"]["itemsDiscovered"] = len(items)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item for execution."""
        for item in items:
            # Check dependencies
            if not self._are_dependencies_met(item):
                continue
            
            # Check risk threshold
            if item.risk_level > self.config["scoring"]["thresholds"]["maxRisk"]:
                continue
            
            # Check minimum score threshold
            if item.composite_score < self.config["scoring"]["thresholds"]["minScore"]:
                continue
            
            return item
        
        return None
    
    def _are_dependencies_met(self, item: ValueItem) -> bool:
        """Check if item dependencies are met."""
        # Simplified - would check completed items
        return len(item.dependencies) == 0


def main():
    """Run value discovery."""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Running autonomous value discovery...")
    items = engine.discover_value_items()
    
    print(f"📊 Discovered {len(items)} value items")
    
    # Generate backlog
    engine.generate_backlog(items)
    print(f"📝 Generated backlog: {engine.backlog_path}")
    
    # Save metrics
    engine.save_metrics(items)
    print(f"💾 Saved metrics: {engine.metrics_path}")
    
    # Select next item
    next_item = engine.select_next_best_value(items)
    if next_item:
        print(f"🎯 Next best value: {next_item.title} (Score: {next_item.composite_score:.1f})")
    else:
        print("✅ No qualifying items found - repository in good state!")


if __name__ == "__main__":
    main()
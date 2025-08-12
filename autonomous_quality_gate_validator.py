#!/usr/bin/env python3
"""
Autonomous Quality Gate Validator - Standalone Implementation

Performs comprehensive quality validation without external dependencies.
"""

import json
import time
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    execution_time: float
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AutonomousQualityValidator:
    """Autonomous quality validation without external dependencies."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.quality_gates = []
        self.results = []
        
    def validate_project_structure(self) -> QualityGateResult:
        """Validate project structure and organization."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        score = 0.0
        
        # Check for essential directories
        essential_dirs = ['src', 'tests', 'docs', 'scripts']
        existing_dirs = []
        
        for dir_name in essential_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                existing_dirs.append(dir_name)
                score += 25  # 25 points per essential directory
        
        details["existing_directories"] = existing_dirs
        details["total_directories_checked"] = len(essential_dirs)
        
        # Check for configuration files
        config_files = ['pyproject.toml', 'requirements.txt', 'README.md', 'LICENSE']
        existing_configs = []
        
        for file_name in config_files:
            file_path = self.project_root / file_name
            if file_path.exists() and file_path.is_file():
                existing_configs.append(file_name)
        
        details["existing_config_files"] = existing_configs
        details["config_score"] = len(existing_configs) / len(config_files) * 100
        
        # Generate recommendations
        missing_dirs = set(essential_dirs) - set(existing_dirs)
        if missing_dirs:
            recommendations.append(f"Create missing directories: {', '.join(missing_dirs)}")
        
        missing_configs = set(config_files) - set(existing_configs)
        if missing_configs:
            recommendations.append(f"Add missing configuration files: {', '.join(missing_configs)}")
        
        # Overall score
        structure_score = score  # Directory score
        config_bonus = len(existing_configs) / len(config_files) * 20  # Config bonus
        final_score = min(100, structure_score + config_bonus)
        
        details["final_score"] = final_score
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="project_structure",
            passed=final_score >= 70,
            score=final_score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def validate_code_organization(self) -> QualityGateResult:
        """Validate code organization and module structure."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return QualityGateResult(
                gate_name="code_organization",
                passed=False,
                score=0.0,
                details={"error": "src directory not found"},
                recommendations=["Create src directory for source code"],
                execution_time=time.time() - start_time
            )
        
        # Count Python files
        python_files = list(src_dir.rglob("*.py"))
        details["python_files_count"] = len(python_files)
        
        # Check for __init__.py files
        init_files = list(src_dir.rglob("__init__.py"))
        details["init_files_count"] = len(init_files)
        
        # Check for main package
        main_package_dir = src_dir / "dp_federated_lora"
        if main_package_dir.exists():
            details["main_package_exists"] = True
            
            # Count modules in main package
            main_modules = list(main_package_dir.glob("*.py"))
            details["main_package_modules"] = len(main_modules)
            
            # Check for key modules
            key_modules = [
                "server.py", "client.py", "privacy.py", "aggregation.py",
                "monitoring.py", "exceptions.py", "config.py"
            ]
            
            existing_key_modules = []
            for module in key_modules:
                if (main_package_dir / module).exists():
                    existing_key_modules.append(module)
            
            details["existing_key_modules"] = existing_key_modules
            details["key_modules_ratio"] = len(existing_key_modules) / len(key_modules)
        else:
            details["main_package_exists"] = False
            recommendations.append("Create main package directory: src/dp_federated_lora/")
        
        # Calculate score
        score = 0
        
        if python_files:
            score += 30  # Has Python files
        
        if init_files:
            score += 20  # Has proper package structure
        
        if details.get("main_package_exists", False):
            score += 30  # Main package exists
            
            # Bonus for key modules
            module_bonus = details.get("key_modules_ratio", 0) * 20
            score += module_bonus
        
        details["calculated_score"] = score
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="code_organization",
            passed=score >= 60,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def validate_documentation_quality(self) -> QualityGateResult:
        """Validate documentation quality and completeness."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        
        # Check README.md
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            details["readme_length"] = len(readme_content)
            details["readme_lines"] = len(readme_content.split('\n'))
            
            # Check for key sections
            key_sections = [
                "installation", "usage", "example", "api", "contributing",
                "license", "overview", "features", "setup"
            ]
            
            found_sections = []
            for section in key_sections:
                if section.lower() in readme_content.lower():
                    found_sections.append(section)
            
            details["found_readme_sections"] = found_sections
            details["readme_completeness"] = len(found_sections) / len(key_sections)
        else:
            details["readme_exists"] = False
            recommendations.append("Create comprehensive README.md file")
        
        # Check docs directory
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.md"))
            details["doc_files_count"] = len(doc_files)
            details["doc_file_names"] = [f.name for f in doc_files[:10]]  # First 10
        else:
            details["docs_directory_exists"] = False
            recommendations.append("Create docs directory with documentation")
        
        # Check for docstrings in Python files
        src_dir = self.project_root / "src"
        python_files = list(src_dir.rglob("*.py")) if src_dir.exists() else []
        
        files_with_docstrings = 0
        total_functions = 0
        functions_with_docstrings = 0
        
        for py_file in python_files[:20]:  # Check first 20 files
            try:
                content = py_file.read_text()
                
                # Check for module docstring
                if '"""' in content and content.strip().startswith('"""'):
                    files_with_docstrings += 1
                
                # Count functions and their docstrings
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('def ') or line.strip().startswith('async def '):
                        total_functions += 1
                        
                        # Check if next non-empty line is a docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line:
                                if next_line.startswith('"""') or next_line.startswith("'''"):
                                    functions_with_docstrings += 1
                                break
                                
            except Exception:
                continue  # Skip files that can't be read
        
        details["files_with_docstrings"] = files_with_docstrings
        details["total_python_files_checked"] = len(python_files[:20])
        details["total_functions"] = total_functions
        details["functions_with_docstrings"] = functions_with_docstrings
        details["docstring_coverage"] = (functions_with_docstrings / total_functions * 100) if total_functions > 0 else 0
        
        # Calculate score
        score = 0
        
        # README score
        if readme_path.exists():
            readme_score = details.get("readme_completeness", 0) * 40
            score += readme_score
        
        # Docs directory score
        if docs_dir.exists():
            score += 20
        
        # Docstring coverage score
        docstring_score = details.get("docstring_coverage", 0) * 0.4  # Max 40 points
        score += docstring_score
        
        details["calculated_score"] = score
        
        # Generate recommendations
        if details.get("readme_completeness", 0) < 0.7:
            recommendations.append("Improve README.md completeness (add more key sections)")
        
        if details.get("docstring_coverage", 0) < 60:
            recommendations.append("Add docstrings to functions and classes")
        
        if not docs_dir.exists():
            recommendations.append("Create comprehensive documentation in docs/ directory")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="documentation_quality",
            passed=score >= 50,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def validate_test_structure(self) -> QualityGateResult:
        """Validate test structure and organization."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return QualityGateResult(
                gate_name="test_structure",
                passed=False,
                score=0.0,
                details={"error": "tests directory not found"},
                recommendations=["Create tests directory with test files"],
                execution_time=time.time() - start_time
            )
        
        # Count test files
        test_files = list(tests_dir.rglob("test_*.py"))
        details["test_files_count"] = len(test_files)
        
        # Check for test categories
        test_categories = ["unit", "integration", "e2e"]
        existing_categories = []
        
        for category in test_categories:
            category_dir = tests_dir / category
            if category_dir.exists():
                existing_categories.append(category)
                category_test_files = list(category_dir.glob("test_*.py"))
                details[f"{category}_test_files"] = len(category_test_files)
        
        details["existing_test_categories"] = existing_categories
        
        # Check for test configuration
        test_configs = ["conftest.py", "pytest.ini", "__init__.py"]
        existing_test_configs = []
        
        for config in test_configs:
            if (tests_dir / config).exists():
                existing_test_configs.append(config)
        
        details["existing_test_configs"] = existing_test_configs
        
        # Analyze test content (simplified)
        total_test_functions = 0
        for test_file in test_files[:10]:  # Check first 10 test files
            try:
                content = test_file.read_text()
                # Count test functions
                test_func_count = content.count("def test_")
                total_test_functions += test_func_count
            except Exception:
                continue
        
        details["total_test_functions"] = total_test_functions
        
        # Calculate score
        score = 0
        
        if test_files:
            score += 40  # Has test files
            
            # Bonus for number of test files
            if len(test_files) >= 5:
                score += 20
            elif len(test_files) >= 10:
                score += 30
        
        # Bonus for test categories
        score += len(existing_categories) * 10
        
        # Bonus for test configuration
        score += len(existing_test_configs) * 5
        
        # Bonus for test functions
        if total_test_functions >= 10:
            score += 15
        elif total_test_functions >= 20:
            score += 25
        
        details["calculated_score"] = score
        
        # Generate recommendations
        if len(test_files) < 5:
            recommendations.append("Add more test files to improve coverage")
        
        missing_categories = set(test_categories) - set(existing_categories)
        if missing_categories:
            recommendations.append(f"Add test categories: {', '.join(missing_categories)}")
        
        if "conftest.py" not in existing_test_configs:
            recommendations.append("Add conftest.py for shared test fixtures")
        
        if total_test_functions < 20:
            recommendations.append("Add more test functions to improve test coverage")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="test_structure",
            passed=score >= 50,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def validate_security_practices(self) -> QualityGateResult:
        """Validate security practices and configurations."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        
        # Check for security configuration files
        security_files = [
            ".gitignore", "SECURITY.md", ".github/workflows/security.yml",
            "bandit.yaml", ".pre-commit-config.yaml"
        ]
        
        existing_security_files = []
        for sec_file in security_files:
            file_path = self.project_root / sec_file
            if file_path.exists():
                existing_security_files.append(sec_file)
        
        details["existing_security_files"] = existing_security_files
        
        # Check .gitignore for sensitive patterns
        gitignore_path = self.project_root / ".gitignore"
        sensitive_patterns_found = []
        
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            
            sensitive_patterns = [
                "*.key", "*.pem", "*.env", "__pycache__", "*.pyc",
                ".venv", "venv/", "node_modules/", "*.log"
            ]
            
            for pattern in sensitive_patterns:
                if pattern in gitignore_content:
                    sensitive_patterns_found.append(pattern)
            
            details["gitignore_sensitive_patterns"] = sensitive_patterns_found
            details["gitignore_completeness"] = len(sensitive_patterns_found) / len(sensitive_patterns)
        
        # Scan for potential security issues in code
        src_dir = self.project_root / "src"
        security_issues = []
        
        if src_dir.exists():
            python_files = list(src_dir.rglob("*.py"))
            
            for py_file in python_files[:15]:  # Check first 15 files
                try:
                    content = py_file.read_text()
                    content_lower = content.lower()
                    
                    # Check for hardcoded secrets (simplified)
                    secret_indicators = [
                        "password = ", "api_key = ", "secret = ",
                        "token = ", "private_key = "
                    ]
                    
                    for indicator in secret_indicators:
                        if indicator in content_lower:
                            # Check if it's not a placeholder or variable
                            lines = content.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if indicator in line.lower():
                                    # Simple heuristic to avoid false positives
                                    if not any(placeholder in line.lower() for placeholder in 
                                             ["none", "null", "placeholder", "your_", "todo", "fixme"]):
                                        security_issues.append({
                                            "file": str(py_file.relative_to(self.project_root)),
                                            "line": line_num,
                                            "issue": f"Potential hardcoded {indicator.split('=')[0].strip()}"
                                        })
                    
                    # Check for insecure imports
                    insecure_imports = ["pickle", "subprocess", "os.system", "eval", "exec"]
                    for imp in insecure_imports:
                        if f"import {imp}" in content or f"from {imp}" in content:
                            security_issues.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "issue": f"Potentially insecure import: {imp}"
                            })
                    
                except Exception:
                    continue
        
        details["security_issues_found"] = len(security_issues)
        details["security_issues"] = security_issues[:5]  # First 5 issues
        
        # Calculate score
        score = 0
        
        # Security files score
        security_files_score = len(existing_security_files) / len(security_files) * 40
        score += security_files_score
        
        # Gitignore completeness score
        if gitignore_path.exists():
            gitignore_score = details.get("gitignore_completeness", 0) * 30
            score += gitignore_score
        
        # Security issues penalty
        if len(security_issues) == 0:
            score += 30  # No security issues found
        elif len(security_issues) <= 2:
            score += 15  # Few security issues
        else:
            score -= len(security_issues) * 2  # Penalty for many issues
        
        score = max(0, score)  # Ensure non-negative score
        details["calculated_score"] = score
        
        # Generate recommendations
        missing_security_files = set(security_files) - set(existing_security_files)
        if missing_security_files:
            recommendations.append(f"Add security files: {', '.join(missing_security_files)}")
        
        if details.get("gitignore_completeness", 0) < 0.8:
            recommendations.append("Improve .gitignore to include more sensitive file patterns")
        
        if len(security_issues) > 0:
            recommendations.append(f"Review and fix {len(security_issues)} potential security issues")
        
        if not any("security" in f for f in existing_security_files):
            recommendations.append("Add security scanning to CI/CD pipeline")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="security_practices",
            passed=score >= 60,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def validate_configuration_quality(self) -> QualityGateResult:
        """Validate configuration file quality."""
        start_time = time.time()
        
        details = {}
        recommendations = []
        
        # Check pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                content = pyproject_path.read_text()
                details["pyproject_exists"] = True
                details["pyproject_size"] = len(content)
                
                # Check for key sections
                key_sections = [
                    "[build-system]", "[project]", "[tool.black]",
                    "[tool.pytest]", "[tool.mypy]"
                ]
                
                found_sections = []
                for section in key_sections:
                    if section in content:
                        found_sections.append(section)
                
                details["pyproject_sections"] = found_sections
                details["pyproject_completeness"] = len(found_sections) / len(key_sections)
                
            except Exception as e:
                details["pyproject_error"] = str(e)
        else:
            details["pyproject_exists"] = False
            recommendations.append("Create pyproject.toml for project configuration")
        
        # Check requirements.txt
        requirements_path = self.project_root / "requirements.txt"
        if requirements_path.exists():
            try:
                content = requirements_path.read_text()
                requirements = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                details["requirements_count"] = len(requirements)
                details["requirements_sample"] = requirements[:5]  # First 5 requirements
                
                # Check for version pinning
                pinned_requirements = [req for req in requirements if '>=' in req or '==' in req or '~=' in req]
                details["pinned_requirements_ratio"] = len(pinned_requirements) / len(requirements) if requirements else 0
                
            except Exception as e:
                details["requirements_error"] = str(e)
        else:
            details["requirements_exists"] = False
            recommendations.append("Create requirements.txt with project dependencies")
        
        # Check for development configuration files
        dev_configs = [
            ".pre-commit-config.yaml", "tox.ini", "pytest.ini",
            ".flake8", ".pylintrc", "mypy.ini"
        ]
        
        existing_dev_configs = []
        for config in dev_configs:
            if (self.project_root / config).exists():
                existing_dev_configs.append(config)
        
        details["existing_dev_configs"] = existing_dev_configs
        details["dev_config_ratio"] = len(existing_dev_configs) / len(dev_configs)
        
        # Calculate score
        score = 0
        
        # pyproject.toml score
        if details.get("pyproject_exists", False):
            score += 30
            pyproject_bonus = details.get("pyproject_completeness", 0) * 20
            score += pyproject_bonus
        
        # requirements.txt score
        if requirements_path.exists():
            score += 20
            # Bonus for version pinning
            pinning_bonus = details.get("pinned_requirements_ratio", 0) * 10
            score += pinning_bonus
        
        # Development configuration score
        dev_config_score = details.get("dev_config_ratio", 0) * 30
        score += dev_config_score
        
        details["calculated_score"] = score
        
        # Generate recommendations
        if not details.get("pyproject_exists", False):
            recommendations.append("Create comprehensive pyproject.toml configuration")
        elif details.get("pyproject_completeness", 0) < 0.7:
            recommendations.append("Add missing sections to pyproject.toml")
        
        if not requirements_path.exists():
            recommendations.append("Create requirements.txt with pinned versions")
        elif details.get("pinned_requirements_ratio", 0) < 0.8:
            recommendations.append("Pin more dependency versions in requirements.txt")
        
        if details.get("dev_config_ratio", 0) < 0.5:
            recommendations.append("Add more development configuration files (pre-commit, linting, etc.)")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="configuration_quality",
            passed=score >= 60,
            score=score,
            details=details,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        print("🔬 Starting Autonomous Quality Gate Validation...")
        
        quality_gates = [
            self.validate_project_structure,
            self.validate_code_organization,
            self.validate_documentation_quality,
            self.validate_test_structure,
            self.validate_security_practices,
            self.validate_configuration_quality
        ]
        
        results = []
        total_score = 0
        passed_gates = 0
        
        for gate_func in quality_gates:
            try:
                print(f"  Running {gate_func.__name__}...")
                result = gate_func()
                results.append(result)
                
                total_score += result.score
                if result.passed:
                    passed_gates += 1
                    
                print(f"    ✅ {result.gate_name}: {result.score:.1f}% ({'PASSED' if result.passed else 'FAILED'})")
                
            except Exception as e:
                print(f"    ❌ {gate_func.__name__} failed: {e}")
                results.append(QualityGateResult(
                    gate_name=gate_func.__name__,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    recommendations=[f"Fix {gate_func.__name__} execution error"],
                    execution_time=0.0
                ))
        
        # Calculate overall metrics
        overall_score = total_score / len(quality_gates) if quality_gates else 0
        pass_rate = passed_gates / len(quality_gates) * 100 if quality_gates else 0
        
        # Collect all recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Generate summary
        summary = {
            "timestamp": time.time(),
            "overall_score": overall_score,
            "pass_rate": pass_rate,
            "gates_passed": passed_gates,
            "total_gates": len(quality_gates),
            "individual_results": [asdict(result) for result in results],
            "top_recommendations": all_recommendations[:10],  # Top 10 recommendations
            "quality_level": self._determine_quality_level(overall_score)
        }
        
        print(f"\n📊 Quality Gate Summary:")
        print(f"  Overall Score: {overall_score:.1f}%")
        print(f"  Gates Passed: {passed_gates}/{len(quality_gates)} ({pass_rate:.1f}%)")
        print(f"  Quality Level: {summary['quality_level']}")
        
        if all_recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(all_recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Save results
        results_file = self.project_root / "quality_gate_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        return summary
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "SATISFACTORY"
        elif score >= 60:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"


def main():
    """Main function to run quality gates."""
    validator = AutonomousQualityValidator()
    summary = validator.run_all_quality_gates()
    
    # Exit with appropriate code
    if summary["pass_rate"] >= 80:
        print("\n✅ Quality gates PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Quality gates FAILED!")
        print("Please address the recommendations above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
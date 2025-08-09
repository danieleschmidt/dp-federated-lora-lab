#!/usr/bin/env python3
"""
Basic Quality Gates Script for DP-Federated LoRA system.

This is a simplified version that works without external dependencies
and focuses on basic code quality and structure validation.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QualityGateResult(Enum):
    """Quality gate results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class BasicGateResult:
    """Basic quality gate result."""
    name: str
    result: QualityGateResult
    score: float
    message: str
    details: Dict[str, Any]
    execution_time: float = 0.0


class BasicQualityGateRunner:
    """Basic quality gate runner without external dependencies."""
    
    def __init__(self, project_root: Path):
        """Initialize quality gate runner."""
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.results: List[BasicGateResult] = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all basic quality gates."""
        self.logger.info("Starting basic quality gate execution")
        start_time = time.time()
        
        gates = [
            ("Project Structure", self.check_project_structure),
            ("Code Organization", self.check_code_organization),
            ("Basic Code Quality", self.check_basic_code_quality),
            ("Configuration Files", self.check_configuration_files),
            ("Documentation", self.check_documentation),
            ("Security Patterns", self.check_security_patterns)
        ]
        
        for gate_name, gate_func in gates:
            self.logger.info(f"Running: {gate_name}")
            gate_start = time.time()
            
            try:
                result = gate_func()
                result.execution_time = time.time() - gate_start
                self.results.append(result)
                
                status = "✓ PASSED" if result.result == QualityGateResult.PASSED else "✗ FAILED"
                self.logger.info(f"{gate_name}: {status} (score: {result.score:.2f})")
                
            except Exception as e:
                result = BasicGateResult(
                    name=gate_name,
                    result=QualityGateResult.FAILED,
                    score=0.0,
                    message=f"Execution failed: {str(e)}",
                    details={"error": str(e)},
                    execution_time=time.time() - gate_start
                )
                self.results.append(result)
                self.logger.error(f"{gate_name}: FAILED - {e}")
        
        total_time = time.time() - start_time
        summary = self.generate_summary()
        summary['total_execution_time'] = total_time
        
        return summary
    
    def check_project_structure(self) -> BasicGateResult:
        """Check essential project structure."""
        required_files = [
            "README.md",
            "pyproject.toml", 
            "requirements.txt",
            "src/dp_federated_lora/__init__.py",
            "src/dp_federated_lora/config.py",
            "src/dp_federated_lora/server.py",
            "src/dp_federated_lora/client.py"
        ]
        
        required_dirs = [
            "src",
            "src/dp_federated_lora",
            "tests",
            "scripts",
            "docs"
        ]
        
        missing_files = [f for f in required_files if not (self.project_root / f).exists()]
        missing_dirs = [d for d in required_dirs if not (self.project_root / d).exists()]
        
        total_items = len(required_files) + len(required_dirs)
        missing_items = len(missing_files) + len(missing_dirs)
        score = max(0.0, (total_items - missing_items) / total_items)
        
        if missing_items == 0:
            return BasicGateResult(
                name="Project Structure",
                result=QualityGateResult.PASSED,
                score=score,
                message="All essential files and directories present",
                details={
                    "required_files": len(required_files),
                    "required_dirs": len(required_dirs),
                    "missing_files": missing_files,
                    "missing_dirs": missing_dirs
                }
            )
        else:
            return BasicGateResult(
                name="Project Structure",
                result=QualityGateResult.FAILED,
                score=score,
                message=f"Missing {missing_items} essential items",
                details={
                    "missing_files": missing_files,
                    "missing_dirs": missing_dirs
                }
            )
    
    def check_code_organization(self) -> BasicGateResult:
        """Check code organization and module structure."""
        python_files = list(self.src_dir.rglob("*.py"))
        
        if not python_files:
            return BasicGateResult(
                name="Code Organization",
                result=QualityGateResult.FAILED,
                score=0.0,
                message="No Python files found in src directory",
                details={"python_files": 0}
            )
        
        # Check for proper __init__.py files
        init_files = [f for f in python_files if f.name == "__init__.py"]
        packages = list(set(f.parent for f in python_files if f.name != "__init__.py"))
        package_coverage = len(init_files) / max(1, len(packages))
        
        # Check for reasonable file sizes
        large_files = []
        total_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    total_lines += line_count
                    
                    if line_count > 1000:  # Files larger than 1000 lines
                        large_files.append((py_file.name, line_count))
            except Exception as e:
                self.logger.warning(f"Could not read {py_file}: {e}")
        
        # Calculate organization score
        size_penalty = min(0.3, len(large_files) * 0.1)
        org_score = min(1.0, package_coverage * 0.7 + 0.3 - size_penalty)
        
        if org_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Code organization is good"
        elif org_score >= 0.6:
            result = QualityGateResult.WARNING
            message = "Code organization needs minor improvements"
        else:
            result = QualityGateResult.FAILED
            message = "Code organization needs significant improvements"
        
        return BasicGateResult(
            name="Code Organization",
            result=result,
            score=org_score,
            message=message,
            details={
                "python_files": len(python_files),
                "init_files": len(init_files),
                "packages": len(packages),
                "package_coverage": package_coverage,
                "large_files": large_files,
                "total_lines": total_lines,
                "avg_file_size": total_lines / len(python_files) if python_files else 0
            }
        )
    
    def check_basic_code_quality(self) -> BasicGateResult:
        """Check basic code quality indicators."""
        python_files = list(self.src_dir.rglob("*.py"))
        
        quality_metrics = {
            "files_with_docstrings": 0,
            "functions_with_docstrings": 0,
            "total_functions": 0,
            "long_lines": 0,
            "total_lines": 0,
            "empty_files": 0,
            "files_with_imports": 0
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                    if not content.strip():
                        quality_metrics["empty_files"] += 1
                        continue
                    
                    quality_metrics["total_lines"] += len(lines)
                    
                    # Check for docstrings in file
                    if '"""' in content or "'''" in content:
                        quality_metrics["files_with_docstrings"] += 1
                    
                    # Check for imports
                    if any(line.strip().startswith(("import ", "from ")) for line in lines):
                        quality_metrics["files_with_imports"] += 1
                    
                    # Analyze line by line
                    in_function = False
                    for line in lines:
                        # Check line length
                        if len(line) > 120:
                            quality_metrics["long_lines"] += 1
                        
                        stripped = line.strip()
                        
                        # Count functions
                        if stripped.startswith("def "):
                            quality_metrics["total_functions"] += 1
                            in_function = True
                        elif stripped.startswith('"""') or stripped.startswith("'''"):
                            if in_function:
                                quality_metrics["functions_with_docstrings"] += 1
                                in_function = False
                        elif stripped and not stripped.startswith("#"):
                            in_function = False
                            
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Calculate quality score
        if len(python_files) == 0:
            return BasicGateResult(
                name="Basic Code Quality",
                result=QualityGateResult.FAILED,
                score=0.0,
                message="No Python files to analyze",
                details=quality_metrics
            )
        
        docstring_rate = quality_metrics["files_with_docstrings"] / len(python_files)
        function_doc_rate = (quality_metrics["functions_with_docstrings"] / 
                           max(1, quality_metrics["total_functions"]))
        
        long_line_penalty = min(0.2, quality_metrics["long_lines"] / quality_metrics["total_lines"] * 10)
        empty_file_penalty = min(0.2, quality_metrics["empty_files"] / len(python_files))
        
        quality_score = min(1.0, docstring_rate * 0.4 + function_doc_rate * 0.4 + 0.2 - 
                           long_line_penalty - empty_file_penalty)
        
        if quality_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Code quality is excellent"
        elif quality_score >= 0.6:
            result = QualityGateResult.WARNING
            message = "Code quality is acceptable"
        else:
            result = QualityGateResult.FAILED
            message = "Code quality needs improvement"
        
        return BasicGateResult(
            name="Basic Code Quality",
            result=result,
            score=quality_score,
            message=message,
            details=quality_metrics
        )
    
    def check_configuration_files(self) -> BasicGateResult:
        """Check essential configuration files."""
        config_files = {
            "pyproject.toml": True,
            "requirements.txt": True,
            "requirements-dev.txt": False,
            "pytest.ini": False,
            "tox.ini": False,
            ".gitignore": False
        }
        
        file_scores = {}
        total_score = 0
        required_files = 0
        
        for config_file, is_required in config_files.items():
            file_path = self.project_root / config_file
            if is_required:
                required_files += 1
            
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if content.strip():
                        score = 1.0
                        file_scores[config_file] = {"exists": True, "valid": True, "score": score}
                    else:
                        score = 0.5  # Exists but empty
                        file_scores[config_file] = {"exists": True, "valid": False, "score": score}
                except Exception as e:
                    score = 0.3  # Exists but unreadable
                    file_scores[config_file] = {"exists": True, "valid": False, "score": score, "error": str(e)}
            else:
                score = 0.0 if is_required else 0.8  # Missing required vs optional
                file_scores[config_file] = {"exists": False, "valid": False, "score": score}
            
            total_score += score
        
        avg_score = total_score / len(config_files)
        
        # Check if all required files are present and valid
        required_valid = all(
            file_scores[f]["exists"] and file_scores[f]["valid"]
            for f, required in config_files.items() if required
        )
        
        if required_valid and avg_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Configuration files are complete and valid"
        elif required_valid:
            result = QualityGateResult.WARNING
            message = "Required config files present, optional ones missing"
        else:
            result = QualityGateResult.FAILED
            message = "Missing or invalid required configuration files"
        
        return BasicGateResult(
            name="Configuration Files",
            result=result,
            score=avg_score,
            message=message,
            details={
                "file_scores": file_scores,
                "required_files": required_files,
                "avg_score": avg_score
            }
        )
    
    def check_documentation(self) -> BasicGateResult:
        """Check documentation completeness."""
        doc_files = {
            "README.md": 2.0,  # Most important
            "CONTRIBUTING.md": 1.0,
            "CHANGELOG.md": 1.0,
            "LICENSE": 1.0,
            "docs/": 1.0  # Directory
        }
        
        doc_scores = {}
        total_weighted_score = 0
        total_weight = sum(doc_files.values())
        
        for doc_item, weight in doc_files.items():
            doc_path = self.project_root / doc_item
            
            if doc_path.exists():
                if doc_path.is_file():
                    try:
                        content = doc_path.read_text(encoding='utf-8')
                        if len(content.strip()) > 100:  # Meaningful content
                            score = weight
                        elif len(content.strip()) > 10:  # Some content
                            score = weight * 0.5
                        else:  # Minimal content
                            score = weight * 0.2
                    except:
                        score = weight * 0.3
                else:  # Directory
                    # Check if directory has files
                    files_in_dir = list(doc_path.rglob("*"))
                    if len(files_in_dir) > 0:
                        score = weight
                    else:
                        score = weight * 0.3
                
                doc_scores[doc_item] = {"exists": True, "score": score}
                total_weighted_score += score
            else:
                doc_scores[doc_item] = {"exists": False, "score": 0}
        
        final_score = total_weighted_score / total_weight
        
        if final_score >= 0.8:
            result = QualityGateResult.PASSED
            message = "Documentation is comprehensive"
        elif final_score >= 0.5:
            result = QualityGateResult.WARNING
            message = "Documentation is adequate but could be improved"
        else:
            result = QualityGateResult.FAILED
            message = "Documentation is insufficient"
        
        return BasicGateResult(
            name="Documentation",
            result=result,
            score=final_score,
            message=message,
            details={
                "doc_scores": doc_scores,
                "total_weighted_score": total_weighted_score,
                "total_weight": total_weight
            }
        )
    
    def check_security_patterns(self) -> BasicGateResult:
        """Check for basic security anti-patterns."""
        security_patterns = [
            ("eval(", "Direct eval() usage"),
            ("exec(", "Direct exec() usage"),
            ("os.system(", "Shell command execution"),
            ("shell=True", "Shell execution enabled"),
            ("pickle.loads(", "Unsafe pickle deserialization"),
            ("yaml.load(", "Unsafe YAML loading"),
            ("subprocess.call(", "Direct subprocess call"),
            ("input(", "Direct input() usage"),
            ("raw_input(", "Direct raw_input() usage")
        ]
        
        security_issues = []
        files_scanned = 0
        
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    files_scanned += 1
                    
                    for pattern, description in security_patterns:
                        if pattern in content:
                            security_issues.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "pattern": pattern,
                                "description": description
                            })
            except Exception as e:
                self.logger.warning(f"Could not scan {py_file}: {e}")
        
        if files_scanned == 0:
            return BasicGateResult(
                name="Security Patterns",
                result=QualityGateResult.SKIPPED,
                score=0.0,
                message="No files to scan for security patterns",
                details={"files_scanned": 0}
            )
        
        # Calculate security score
        issue_penalty = min(1.0, len(security_issues) * 0.2)
        security_score = max(0.0, 1.0 - issue_penalty)
        
        if len(security_issues) == 0:
            result = QualityGateResult.PASSED
            message = "No obvious security anti-patterns detected"
        elif len(security_issues) <= 2:
            result = QualityGateResult.WARNING
            message = f"Found {len(security_issues)} potential security issues"
        else:
            result = QualityGateResult.FAILED
            message = f"Found {len(security_issues)} security anti-patterns"
        
        return BasicGateResult(
            name="Security Patterns",
            result=result,
            score=security_score,
            message=message,
            details={
                "files_scanned": files_scanned,
                "security_issues": security_issues,
                "issue_count": len(security_issues)
            }
        )
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate quality gates summary."""
        if not self.results:
            return {"overall_status": "NO_GATES_RUN"}
        
        passed = [r for r in self.results if r.result == QualityGateResult.PASSED]
        failed = [r for r in self.results if r.result == QualityGateResult.FAILED]
        warnings = [r for r in self.results if r.result == QualityGateResult.WARNING]
        
        avg_score = sum(r.score for r in self.results) / len(self.results)
        
        # Determine overall status
        if len(failed) == 0 and len(warnings) == 0:
            overall_status = "ALL_PASSED"
        elif len(failed) == 0:
            overall_status = "PASSED_WITH_WARNINGS"
        elif len(failed) <= 1:
            overall_status = "MINOR_FAILURES"
        else:
            overall_status = "MULTIPLE_FAILURES"
        
        return {
            "overall_status": overall_status,
            "overall_score": avg_score,
            "total_gates": len(self.results),
            "passed": len(passed),
            "failed": len(failed),
            "warnings": len(warnings),
            "gate_results": [
                {
                    "name": r.name,
                    "result": r.result.value,
                    "score": r.score,
                    "message": r.message,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ],
            "failed_gates": [r.name for r in failed]
        }


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    
    print("="*60)
    print("DP-FEDERATED LORA - BASIC QUALITY GATES")
    print("="*60)
    
    runner = BasicQualityGateRunner(project_root)
    summary = runner.run_all_gates()
    
    # Save results
    report_file = project_root / "basic_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Overall Score: {summary['overall_score']:.2f}/1.00")
    print(f"Gates Passed: {summary['passed']}/{summary['total_gates']}")
    print(f"Gates Failed: {summary['failed']}")
    print(f"Gates with Warnings: {summary['warnings']}")
    print(f"Execution Time: {summary.get('total_execution_time', 0):.2f}s")
    
    if summary['failed_gates']:
        print(f"\nFailed Gates: {', '.join(summary['failed_gates'])}")
    
    print(f"\nDetailed report: {report_file}")
    print("="*60)
    
    # Return appropriate exit code
    if summary['overall_status'] in ['MULTIPLE_FAILURES']:
        return 1
    elif summary['overall_status'] in ['MINOR_FAILURES', 'PASSED_WITH_WARNINGS']:
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
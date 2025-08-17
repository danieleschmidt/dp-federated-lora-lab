#!/usr/bin/env python3
"""
Quality Gates Validator for DP-Federated LoRA Research System

This module implements comprehensive quality gates to ensure research reproducibility,
statistical significance, and publication readiness.
"""

import os
import sys
import time
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityGateStatus(Enum):
    """Quality gate status levels."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class QualityGateType(Enum):
    """Types of quality gates."""
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    REPRODUCIBILITY = "reproducibility"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_type: QualityGateType
    gate_name: str
    status: QualityGateStatus
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class QualityReport:
    """Comprehensive quality report."""
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    gate_results: List[QualityGateResult]
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class CodeQualityValidator:
    """Validates code quality standards."""
    
    def __init__(self):
        self.min_score_threshold = 0.85
    
    async def validate_python_code(self, code_paths: List[Path]) -> QualityGateResult:
        """Validate Python code quality."""
        start_time = time.time()
        
        try:
            total_score = 0.0
            file_count = 0
            issues = []
            
            for code_path in code_paths:
                if code_path.suffix == '.py' and code_path.exists():
                    file_score, file_issues = self._analyze_python_file(code_path)
                    total_score += file_score
                    file_count += 1
                    issues.extend(file_issues)
            
            if file_count == 0:
                return QualityGateResult(
                    gate_type=QualityGateType.CODE_QUALITY,
                    gate_name="Python Code Quality",
                    status=QualityGateStatus.SKIPPED,
                    score=0.0,
                    message="No Python files found to analyze",
                    execution_time=time.time() - start_time
                )
            
            average_score = total_score / file_count
            status = QualityGateStatus.PASSED if average_score >= self.min_score_threshold else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_type=QualityGateType.CODE_QUALITY,
                gate_name="Python Code Quality",
                status=status,
                score=average_score,
                message=f"Code quality score: {average_score:.2f}/1.0",
                details={
                    "files_analyzed": file_count,
                    "issues_found": len(issues),
                    "issues": issues[:10]  # Top 10 issues
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.CODE_QUALITY,
                gate_name="Python Code Quality",
                status=QualityGateStatus.FAILED,
                score=0.0,
                message=f"Code quality validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _analyze_python_file(self, file_path: Path) -> Tuple[float, List[str]]:
        """Analyze a single Python file for quality metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            
            # Calculate basic metrics
            docstring_score = self._check_docstrings(content)
            complexity_score = self._check_complexity(content)
            style_score = self._check_style(content)
            length_score = self._check_line_lengths(lines)
            
            # Combine scores
            overall_score = (docstring_score + complexity_score + style_score + length_score) / 4
            
            issues = []
            if docstring_score < 0.7:
                issues.append(f"Missing or insufficient docstrings in {file_path.name}")
            if complexity_score < 0.7:
                issues.append(f"High code complexity in {file_path.name}")
            if style_score < 0.8:
                issues.append(f"Style issues in {file_path.name}")
            if length_score < 0.9:
                issues.append(f"Long lines in {file_path.name}")
            
            return overall_score, issues
            
        except Exception as e:
            return 0.0, [f"Error analyzing {file_path.name}: {str(e)}"]
    
    def _check_docstrings(self, content: str) -> float:
        """Check for docstring coverage."""
        lines = content.split('\n')
        
        # Count functions and classes
        functions = sum(1 for line in lines if line.strip().startswith('def '))
        classes = sum(1 for line in lines if line.strip().startswith('class '))
        
        # Count docstrings (simplified check)
        docstrings = content.count('"""') // 2 + content.count("'''") // 2
        
        if functions + classes == 0:
            return 1.0  # No functions or classes to document
        
        coverage = min(1.0, docstrings / (functions + classes))
        return coverage
    
    def _check_complexity(self, content: str) -> float:
        """Check code complexity (simplified McCabe complexity)."""
        complexity_keywords = ['if', 'elif', 'while', 'for', 'try', 'except', 'with']
        
        lines = content.split('\n')
        total_complexity = 0
        
        for line in lines:
            stripped = line.strip()
            for keyword in complexity_keywords:
                if stripped.startswith(keyword + ' ') or stripped.startswith(keyword + '('):
                    total_complexity += 1
        
        # Normalize by number of lines
        if len(lines) == 0:
            return 1.0
        
        complexity_ratio = total_complexity / len(lines)
        
        # Score inversely related to complexity
        if complexity_ratio < 0.1:
            return 1.0
        elif complexity_ratio < 0.2:
            return 0.8
        elif complexity_ratio < 0.3:
            return 0.6
        else:
            return 0.4
    
    def _check_style(self, content: str) -> float:
        """Check basic style guidelines."""
        lines = content.split('\n')
        style_issues = 0
        
        for line in lines:
            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                style_issues += 1
            
            # Check for tabs (prefer spaces)
            if '\t' in line:
                style_issues += 1
        
        if len(lines) == 0:
            return 1.0
        
        style_score = max(0.0, 1.0 - (style_issues / len(lines) * 2))
        return style_score
    
    def _check_line_lengths(self, lines: List[str]) -> float:
        """Check line length compliance."""
        long_lines = sum(1 for line in lines if len(line) > 100)
        
        if len(lines) == 0:
            return 1.0
        
        compliance_score = max(0.0, 1.0 - (long_lines / len(lines) * 2))
        return compliance_score

class SecurityValidator:
    """Validates security standards."""
    
    def __init__(self):
        self.security_patterns = [
            'password',
            'secret',
            'token',
            'api_key',
            'private_key',
            'credential'
        ]
    
    async def validate_security(self, code_paths: List[Path]) -> QualityGateResult:
        """Validate security practices."""
        start_time = time.time()
        
        try:
            security_issues = []
            files_scanned = 0
            
            for code_path in code_paths:
                if code_path.suffix in ['.py', '.json', '.yaml', '.yml'] and code_path.exists():
                    file_issues = self._scan_file_for_secrets(code_path)
                    security_issues.extend(file_issues)
                    files_scanned += 1
            
            # Calculate security score
            if files_scanned == 0:
                score = 1.0
                status = QualityGateStatus.SKIPPED
                message = "No files found to scan for security issues"
            elif len(security_issues) == 0:
                score = 1.0
                status = QualityGateStatus.PASSED
                message = "No security issues detected"
            else:
                # Score based on number of issues
                score = max(0.0, 1.0 - (len(security_issues) / files_scanned * 0.5))
                status = QualityGateStatus.FAILED if score < 0.8 else QualityGateStatus.WARNING
                message = f"Found {len(security_issues)} potential security issues"
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                gate_name="Security Scan",
                status=status,
                score=score,
                message=message,
                details={
                    "files_scanned": files_scanned,
                    "security_issues": security_issues
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                gate_name="Security Scan",
                status=QualityGateStatus.FAILED,
                score=0.0,
                message=f"Security validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _scan_file_for_secrets(self, file_path: Path) -> List[str]:
        """Scan file for potential secrets."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            issues = []
            
            for pattern in self.security_patterns:
                if pattern in content:
                    # Check if it's not in a comment or string that's clearly not a secret
                    if not self._is_false_positive(content, pattern):
                        issues.append(f"Potential secret '{pattern}' found in {file_path.name}")
            
            return issues
            
        except Exception as e:
            return [f"Error scanning {file_path.name}: {str(e)}"]
    
    def _is_false_positive(self, content: str, pattern: str) -> bool:
        """Check if a security pattern match is likely a false positive."""
        # Simple heuristics to reduce false positives
        false_positive_contexts = [
            f'"{pattern}"',  # In quotes as a string literal
            f"'{pattern}'",  # In single quotes
            f'# {pattern}',  # In comments
            f'def {pattern}',  # Function name
            f'class {pattern}',  # Class name
        ]
        
        return any(context in content for context in false_positive_contexts)

class ReproducibilityValidator:
    """Validates research reproducibility."""
    
    def __init__(self):
        self.required_files = [
            'requirements.txt',
            'README.md',
            'pyproject.toml'
        ]
    
    async def validate_reproducibility(self, project_root: Path) -> QualityGateResult:
        """Validate reproducibility requirements."""
        start_time = time.time()
        
        try:
            reproducibility_score = 0.0
            missing_files = []
            found_files = []
            
            # Check for required files
            for required_file in self.required_files:
                file_path = project_root / required_file
                if file_path.exists():
                    found_files.append(required_file)
                    reproducibility_score += 1.0 / len(self.required_files)
                else:
                    missing_files.append(required_file)
            
            # Check for additional reproducibility features
            bonus_features = {
                'Dockerfile': 0.1,
                'docker-compose.yml': 0.1,
                '.github/workflows': 0.1,
                'tests/': 0.2,
                'docs/': 0.1
            }
            
            for feature, bonus in bonus_features.items():
                feature_path = project_root / feature
                if feature_path.exists():
                    reproducibility_score += bonus
                    found_files.append(feature)
            
            # Normalize score to 0-1
            reproducibility_score = min(1.0, reproducibility_score)
            
            status = QualityGateStatus.PASSED if reproducibility_score >= 0.8 else QualityGateStatus.WARNING
            if reproducibility_score < 0.5:
                status = QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_type=QualityGateType.REPRODUCIBILITY,
                gate_name="Reproducibility Check",
                status=status,
                score=reproducibility_score,
                message=f"Reproducibility score: {reproducibility_score:.2f}/1.0",
                details={
                    "found_files": found_files,
                    "missing_files": missing_files,
                    "recommendations": self._get_reproducibility_recommendations(missing_files)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.REPRODUCIBILITY,
                gate_name="Reproducibility Check",
                status=QualityGateStatus.FAILED,
                score=0.0,
                message=f"Reproducibility validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _get_reproducibility_recommendations(self, missing_files: List[str]) -> List[str]:
        """Get recommendations for improving reproducibility."""
        recommendations = []
        
        if 'requirements.txt' in missing_files:
            recommendations.append("Add requirements.txt with exact dependency versions")
        
        if 'README.md' in missing_files:
            recommendations.append("Create comprehensive README with setup instructions")
        
        if 'pyproject.toml' in missing_files:
            recommendations.append("Add pyproject.toml for modern Python packaging")
        
        recommendations.extend([
            "Consider adding Dockerfile for containerized reproducibility",
            "Add comprehensive test suite for validation",
            "Include example usage and benchmarks",
            "Document hardware requirements and expected runtime"
        ])
        
        return recommendations

class DocumentationValidator:
    """Validates documentation quality."""
    
    def __init__(self):
        self.min_documentation_score = 0.7
    
    async def validate_documentation(self, project_root: Path) -> QualityGateResult:
        """Validate documentation quality."""
        start_time = time.time()
        
        try:
            documentation_score = 0.0
            documentation_issues = []
            
            # Check README.md
            readme_path = project_root / 'README.md'
            if readme_path.exists():
                readme_score = self._analyze_readme(readme_path)
                documentation_score += readme_score * 0.5  # README is 50% of score
            else:
                documentation_issues.append("Missing README.md file")
            
            # Check for additional documentation
            docs_dir = project_root / 'docs'
            if docs_dir.exists() and docs_dir.is_dir():
                docs_score = self._analyze_docs_directory(docs_dir)
                documentation_score += docs_score * 0.3  # Docs directory is 30% of score
            else:
                documentation_issues.append("Missing docs/ directory")
            
            # Check code comments and docstrings
            python_files = list(project_root.rglob('*.py'))
            if python_files:
                comment_score = self._analyze_code_documentation(python_files)
                documentation_score += comment_score * 0.2  # Comments are 20% of score
            
            status = QualityGateStatus.PASSED if documentation_score >= self.min_documentation_score else QualityGateStatus.WARNING
            if documentation_score < 0.4:
                status = QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION,
                gate_name="Documentation Quality",
                status=status,
                score=documentation_score,
                message=f"Documentation score: {documentation_score:.2f}/1.0",
                details={
                    "issues": documentation_issues,
                    "recommendations": self._get_documentation_recommendations(documentation_issues)
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION,
                gate_name="Documentation Quality",
                status=QualityGateStatus.FAILED,
                score=0.0,
                message=f"Documentation validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _analyze_readme(self, readme_path: Path) -> float:
        """Analyze README.md quality."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            score = 0.0
            
            # Check for essential sections
            essential_sections = [
                'overview',
                'installation',
                'usage',
                'example',
                'requirements',
                'getting started'
            ]
            
            content_lower = content.lower()
            for section in essential_sections:
                if section in content_lower:
                    score += 1.0 / len(essential_sections)
            
            # Bonus for length (indicates comprehensive documentation)
            if len(content) > 1000:
                score += 0.2
            
            # Bonus for code examples
            if '```' in content:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _analyze_docs_directory(self, docs_dir: Path) -> float:
        """Analyze docs directory quality."""
        try:
            doc_files = list(docs_dir.rglob('*.md'))
            
            if len(doc_files) == 0:
                return 0.0
            
            # Score based on number and variety of documentation files
            score = min(1.0, len(doc_files) / 5.0)  # Full score for 5+ doc files
            
            return score
            
        except Exception:
            return 0.0
    
    def _analyze_code_documentation(self, python_files: List[Path]) -> float:
        """Analyze code documentation quality."""
        if not python_files:
            return 1.0  # No code, no problem
        
        total_score = 0.0
        
        for py_file in python_files[:10]:  # Analyze up to 10 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count docstrings
                docstring_count = content.count('"""') // 2 + content.count("'''") // 2
                
                # Count functions and classes
                lines = content.split('\n')
                functions = sum(1 for line in lines if line.strip().startswith('def '))
                classes = sum(1 for line in lines if line.strip().startswith('class '))
                
                if functions + classes > 0:
                    file_score = min(1.0, docstring_count / (functions + classes))
                    total_score += file_score
                
            except Exception:
                continue
        
        return total_score / len(python_files) if python_files else 1.0
    
    def _get_documentation_recommendations(self, issues: List[str]) -> List[str]:
        """Get documentation improvement recommendations."""
        recommendations = [
            "Add comprehensive README with clear examples",
            "Include API documentation for all public functions",
            "Add tutorial and getting started guide",
            "Document configuration options and parameters",
            "Include troubleshooting section",
            "Add performance benchmarks and expected results"
        ]
        
        return recommendations

class QualityGatesOrchestrator:
    """Orchestrates all quality gate validations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.validators = {
            QualityGateType.CODE_QUALITY: CodeQualityValidator(),
            QualityGateType.SECURITY: SecurityValidator(),
            QualityGateType.REPRODUCIBILITY: ReproducibilityValidator(),
            QualityGateType.DOCUMENTATION: DocumentationValidator()
        }
    
    async def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report."""
        logger.info("Starting comprehensive quality gates validation")
        
        gate_results = []
        
        # Get all Python files for code analysis
        python_files = list(self.project_root.rglob('*.py'))
        
        # Get all relevant files for security scanning
        scan_files = []
        for pattern in ['*.py', '*.json', '*.yaml', '*.yml', '*.txt']:
            scan_files.extend(self.project_root.rglob(pattern))
        
        # Run quality gates
        tasks = [
            self.validators[QualityGateType.CODE_QUALITY].validate_python_code(python_files),
            self.validators[QualityGateType.SECURITY].validate_security(scan_files),
            self.validators[QualityGateType.REPRODUCIBILITY].validate_reproducibility(self.project_root),
            self.validators[QualityGateType.DOCUMENTATION].validate_documentation(self.project_root)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, QualityGateResult):
                gate_results.append(result)
                logger.info(f"{result.gate_name}: {result.status.value.upper()} (score: {result.score:.2f})")
            elif isinstance(result, Exception):
                logger.error(f"Quality gate failed with exception: {result}")
        
        # Calculate overall metrics
        total_gates = len(gate_results)
        passed_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.PASSED)
        failed_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.FAILED)
        warning_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.WARNING)
        
        # Calculate weighted overall score
        overall_score = sum(r.score for r in gate_results) / total_gates if total_gates > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results)
        
        quality_report = QualityReport(
            overall_score=overall_score,
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warning_gates=warning_gates,
            gate_results=gate_results,
            recommendations=recommendations
        )
        
        logger.info(f"Quality gates completed: {passed_gates}/{total_gates} passed, overall score: {overall_score:.2f}")
        return quality_report
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate improvement recommendations based on gate results."""
        recommendations = []
        
        for result in gate_results:
            if result.status in [QualityGateStatus.FAILED, QualityGateStatus.WARNING]:
                if 'recommendations' in result.details:
                    recommendations.extend(result.details['recommendations'])
                else:
                    recommendations.append(f"Improve {result.gate_name.lower()} (current score: {result.score:.2f})")
        
        # Add general recommendations
        if any(r.status == QualityGateStatus.FAILED for r in gate_results):
            recommendations.append("Address all failing quality gates before production deployment")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def save_quality_report(self, report: QualityReport, output_file: Path):
        """Save quality report to file."""
        try:
            report_data = asdict(report)
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Quality report saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")

async def main():
    """Main function for quality gates validation."""
    logger.info("🛡️ Starting Quality Gates Validation")
    
    project_root = Path('/root/repo')
    orchestrator = QualityGatesOrchestrator(project_root)
    
    try:
        # Run all quality gates
        quality_report = await orchestrator.run_all_quality_gates()
        
        # Save report
        output_file = project_root / 'quality_gates_report.json'
        await orchestrator.save_quality_report(quality_report, output_file)
        
        # Print summary
        print("\n" + "=" * 70)
        print("🛡️ QUALITY GATES VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Overall Score: {quality_report.overall_score:.2f}/1.0")
        print(f"Gates Passed: {quality_report.passed_gates}/{quality_report.total_gates}")
        print(f"Gates Failed: {quality_report.failed_gates}")
        print(f"Gates with Warnings: {quality_report.warning_gates}")
        
        print("\n📋 Gate Results:")
        for result in quality_report.gate_results:
            status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️", "skipped": "⏭️"}
            emoji = status_emoji.get(result.status.value, "❓")
            print(f"  {emoji} {result.gate_name}: {result.score:.2f} - {result.message}")
        
        if quality_report.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(quality_report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Determine overall result
        if quality_report.overall_score >= 0.8 and quality_report.failed_gates == 0:
            print("\n🎉 QUALITY GATES PASSED - READY FOR PRODUCTION!")
            return True
        elif quality_report.overall_score >= 0.6:
            print("\n⚠️ QUALITY GATES PARTIALLY PASSED - IMPROVEMENTS RECOMMENDED")
            return True
        else:
            print("\n❌ QUALITY GATES FAILED - CRITICAL ISSUES MUST BE ADDRESSED")
            return False
        
    except Exception as e:
        logger.error(f"Quality gates validation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
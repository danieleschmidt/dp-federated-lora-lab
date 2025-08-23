#!/usr/bin/env python3
"""
Syntax validation system that checks code quality without external dependencies.
Validates Python syntax, structure, and design patterns.
"""

import ast
import sys
import os
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Tuple


class PythonSyntaxValidator:
    """Validates Python syntax and code structure."""
    
    def __init__(self):
        self.validation_results = []
        self.total_files_checked = 0
        self.files_passed = 0
        self.files_failed = 0
        self.total_lines = 0
        self.issues_found = []
    
    def validate_file_syntax(self, file_path: Path) -> bool:
        """Validate syntax of a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines
            lines = content.count('\n') + 1
            self.total_lines += lines
            
            # Parse AST to validate syntax
            tree = ast.parse(content, filename=str(file_path))
            
            # Additional structure validation
            issues = self._validate_code_structure(tree, content, file_path)
            
            if issues:
                self.issues_found.extend(issues)
                for issue in issues:
                    print(f"⚠️  {file_path.name}: {issue}")
            
            self.files_passed += 1
            self.validation_results.append({
                'file': str(file_path),
                'status': 'PASS',
                'lines': lines,
                'issues': issues
            })
            
            print(f"✅ Syntax validation passed: {file_path.name} ({lines} lines)")
            return True
            
        except SyntaxError as e:
            self.files_failed += 1
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            print(f"❌ Syntax validation failed: {file_path.name} - {error_msg}")
            
            self.validation_results.append({
                'file': str(file_path),
                'status': 'FAIL',
                'error': error_msg,
                'line': e.lineno
            })
            return False
            
        except Exception as e:
            self.files_failed += 1
            error_msg = f"Validation error: {str(e)}"
            print(f"❌ Validation error: {file_path.name} - {error_msg}")
            
            self.validation_results.append({
                'file': str(file_path),
                'status': 'ERROR',
                'error': error_msg
            })
            return False
    
    def _validate_code_structure(self, tree: ast.AST, content: str, file_path: Path) -> List[str]:
        """Validate code structure and patterns."""
        issues = []
        
        # Check for module docstring
        if not ast.get_docstring(tree):
            issues.append("Missing module docstring")
        
        # Count classes and functions
        classes = 0
        functions = 0
        async_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes += 1
                # Check for class docstring
                if not ast.get_docstring(node):
                    issues.append(f"Class '{node.name}' missing docstring")
            
            elif isinstance(node, ast.FunctionDef):
                functions += 1
                # Check for function docstring (only for non-private functions)
                if not node.name.startswith('_') and not ast.get_docstring(node):
                    issues.append(f"Function '{node.name}' missing docstring")
            
            elif isinstance(node, ast.AsyncFunctionDef):
                async_functions += 1
                if not node.name.startswith('_') and not ast.get_docstring(node):
                    issues.append(f"Async function '{node.name}' missing docstring")
        
        # Check import organization
        imports_issues = self._check_import_organization(tree)
        issues.extend(imports_issues)
        
        # Check line length (basic check)
        long_lines = []
        for i, line in enumerate(content.split('\n'), 1):
            if len(line) > 120:  # Allow slightly longer than PEP 8
                long_lines.append(i)
        
        if long_lines:
            issues.append(f"Long lines found at: {', '.join(map(str, long_lines[:5]))}")
        
        return issues
    
    def _check_import_organization(self, tree: ast.AST) -> List[str]:
        """Check import statement organization."""
        issues = []
        
        imports = []
        from_imports = []
        
        # Collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                from_imports.append((module, node.lineno))
        
        # Check if stdlib imports come first (basic heuristic)
        stdlib_modules = {'os', 'sys', 'json', 'time', 'asyncio', 'threading', 'multiprocessing', 'collections'}
        
        stdlib_found = False
        third_party_found = False
        
        for module_name, line_no in imports + from_imports:
            root_module = module_name.split('.')[0]
            
            if root_module in stdlib_modules:
                if third_party_found:
                    issues.append("Standard library imports should come before third-party imports")
                    break
                stdlib_found = True
            else:
                third_party_found = True
        
        return issues


def validate_novel_lora_system():
    """Validate the novel LoRA optimization system files."""
    print("🚀 Starting Novel LoRA System Syntax Validation")
    print("="*80)
    
    validator = PythonSyntaxValidator()
    
    # Files to validate
    src_dir = Path("src/dp_federated_lora")
    files_to_check = [
        "novel_lora_hyperparameter_optimizer.py",
        "robust_lora_optimization_system.py", 
        "scalable_lora_optimization_engine.py"
    ]
    
    # Also check test file
    test_files = [
        Path("tests/test_novel_lora_optimization.py")
    ]
    
    validation_passed = True
    
    print("\n📁 Validating Core Implementation Files...")
    for filename in files_to_check:
        file_path = src_dir / filename
        
        if file_path.exists():
            validator.total_files_checked += 1
            if not validator.validate_file_syntax(file_path):
                validation_passed = False
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n🧪 Validating Test Files...")
    for test_file in test_files:
        if test_file.exists():
            validator.total_files_checked += 1
            if not validator.validate_file_syntax(test_file):
                validation_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    # Additional code quality checks
    print("\n📊 Performing Code Quality Analysis...")
    
    quality_metrics = {
        'total_files': validator.total_files_checked,
        'files_passed': validator.files_passed,
        'files_failed': validator.files_failed,
        'total_lines': validator.total_lines,
        'total_issues': len(validator.issues_found),
        'avg_lines_per_file': validator.total_lines / validator.total_files_checked if validator.total_files_checked > 0 else 0
    }
    
    print(f"📄 Total files checked: {quality_metrics['total_files']}")
    print(f"📝 Total lines of code: {quality_metrics['total_lines']}")
    print(f"📊 Average lines per file: {quality_metrics['avg_lines_per_file']:.1f}")
    print(f"⚠️  Total issues found: {quality_metrics['total_issues']}")
    
    # Categorize issues
    issue_categories = {
        'missing_docstrings': 0,
        'long_lines': 0,
        'import_order': 0,
        'other': 0
    }
    
    for issue in validator.issues_found:
        if 'docstring' in issue.lower():
            issue_categories['missing_docstrings'] += 1
        elif 'long lines' in issue.lower():
            issue_categories['long_lines'] += 1
        elif 'import' in issue.lower():
            issue_categories['import_order'] += 1
        else:
            issue_categories['other'] += 1
    
    print("\n📋 Issue Categories:")
    for category, count in issue_categories.items():
        if count > 0:
            print(f"  • {category.replace('_', ' ').title()}: {count}")
    
    # Calculate quality score
    if validator.total_files_checked > 0:
        syntax_score = (validator.files_passed / validator.total_files_checked) * 100
        quality_penalty = min(len(validator.issues_found) * 2, 20)  # Max 20% penalty
        overall_score = max(0, syntax_score - quality_penalty)
    else:
        syntax_score = 0
        overall_score = 0
    
    print("\n" + "="*80)
    print("📊 SYNTAX VALIDATION SUMMARY")
    print("="*80)
    print(f"✅ Files passed: {validator.files_passed}/{validator.total_files_checked}")
    print(f"❌ Files failed: {validator.files_failed}")
    print(f"📈 Syntax success rate: {syntax_score:.1f}%")
    print(f"🎯 Overall quality score: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print("🎉 EXCELLENT: Code quality is outstanding!")
        status = "EXCELLENT"
    elif overall_score >= 80:
        print("✅ GOOD: Code quality is solid with minor improvements needed.")
        status = "GOOD"
    elif overall_score >= 70:
        print("⚠️  ACCEPTABLE: Code quality is acceptable but needs improvement.")
        status = "ACCEPTABLE"
    else:
        print("❌ NEEDS WORK: Code quality requires significant improvement.")
        status = "NEEDS_WORK"
        validation_passed = False
    
    # Save detailed results
    results = {
        'timestamp': str(__import__('datetime').datetime.now()),
        'validation_status': status,
        'quality_metrics': quality_metrics,
        'syntax_score': syntax_score,
        'overall_score': overall_score,
        'issue_categories': issue_categories,
        'validation_results': validator.validation_results,
        'issues_found': validator.issues_found
    }
    
    with open('syntax_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: syntax_validation_results.json")
    
    return validation_passed


def main():
    """Run syntax validation."""
    try:
        success = validate_novel_lora_system()
        
        if success:
            print("\n🎉 Syntax validation completed successfully!")
            return 0
        else:
            print("\n❌ Syntax validation failed. Please review the issues above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
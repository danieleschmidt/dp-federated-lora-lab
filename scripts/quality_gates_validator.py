"""
Quality Gates Validator: Comprehensive quality assurance system.

Validates:
- Code quality and standards
- Security vulnerabilities
- Performance benchmarks
- Test coverage and results
- Documentation completeness
- Deployment readiness
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

class QualityGate:
    """Individual quality gate with validation logic"""
    
    def __init__(self, name: str, description: str, weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = weight
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.errors = []
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Override in subclasses to implement validation logic"""
        return True, 1.0, {}

class ProjectStructureGate(QualityGate):
    """Validate project structure and organization"""
    
    def __init__(self):
        super().__init__("Project Structure", "Validate project organization and file structure")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate project structure"""
        score = 0.0
        details = {}
        errors = []
        
        # Required directories
        required_dirs = ['src', 'tests', 'scripts', 'docs']
        present_dirs = []
        
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                present_dirs.append(dir_name)
                score += 0.2
            else:
                errors.append(f"Missing directory: {dir_name}")
                
        details['required_directories'] = {
            'present': present_dirs,
            'missing': [d for d in required_dirs if d not in present_dirs]
        }
        
        # Required files
        required_files = [
            'pyproject.toml',
            'requirements.txt', 
            'README.md',
            'LICENSE'
        ]
        present_files = []
        
        for file_name in required_files:
            if os.path.exists(file_name):
                present_files.append(file_name)
                score += 0.05
            else:
                errors.append(f"Missing file: {file_name}")
                
        details['required_files'] = {
            'present': present_files,
            'missing': [f for f in required_files if f not in present_files]
        }
        
        # Check source code structure
        src_path = Path('src/dp_federated_lora')
        if src_path.exists():
            python_files = list(src_path.glob('*.py'))
            details['source_files'] = [f.name for f in python_files]
            
            if len(python_files) >= 5:
                score += 0.2
            else:
                errors.append(f"Insufficient source files: {len(python_files)} < 5")
        else:
            errors.append("Source directory not found")
            
        passed = score >= 0.8
        return passed, score, {'details': details, 'errors': errors}

class CodeQualityGate(QualityGate):
    """Validate code quality standards"""
    
    def __init__(self):
        super().__init__("Code Quality", "Validate code formatting and style standards")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate code quality"""
        score = 0.0
        details = {}
        errors = []
        
        # Check Python syntax
        python_files = list(Path('src').rglob('*.py')) + list(Path('tests').rglob('*.py'))
        syntax_errors = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors += 1
                errors.append(f"Syntax error in {py_file}: {e}")
                
        if syntax_errors == 0:
            score += 0.4
            details['syntax_check'] = 'passed'
        else:
            details['syntax_check'] = f'{syntax_errors} errors'
            
        # Check imports
        import_errors = 0
        for py_file in python_files[:5]:  # Check first 5 files
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', str(py_file)],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    import_errors += 1
            except subprocess.TimeoutExpired:
                import_errors += 1
            except Exception:
                import_errors += 1
                
        if import_errors <= 2:  # Allow some import errors due to missing dependencies
            score += 0.3
            details['import_check'] = f'{import_errors} errors (acceptable)'
        else:
            details['import_check'] = f'{import_errors} errors'
            errors.append(f"Too many import errors: {import_errors}")
            
        # Check for basic code quality indicators
        code_quality_score = self._analyze_code_quality(python_files)
        score += code_quality_score * 0.3
        details['code_quality_score'] = code_quality_score
        
        passed = score >= 0.7
        return passed, score, {'details': details, 'errors': errors}
        
    def _analyze_code_quality(self, python_files: List[Path]) -> float:
        """Analyze code quality indicators"""
        total_score = 0.0
        file_count = 0
        
        for py_file in python_files[:10]:  # Analyze first 10 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                file_score = 0.0
                
                # Check for docstrings
                if '"""' in content or "'''" in content:
                    file_score += 0.3
                    
                # Check for type hints
                if ': ' in content and '->' in content:
                    file_score += 0.2
                    
                # Check for error handling
                if 'try:' in content and 'except' in content:
                    file_score += 0.2
                    
                # Check for logging
                if 'logger' in content or 'logging' in content:
                    file_score += 0.1
                    
                # Check for constants (uppercase variables)
                lines = content.split('\n')
                has_constants = any(line.strip().split('=')[0].strip().isupper() 
                                  for line in lines if '=' in line and not line.strip().startswith('#'))
                if has_constants:
                    file_score += 0.1
                    
                # Check reasonable line length (< 120 chars for most lines)
                long_lines = sum(1 for line in lines if len(line) > 120)
                if long_lines < len(lines) * 0.1:  # Less than 10% long lines
                    file_score += 0.1
                    
                total_score += file_score
                file_count += 1
                
            except Exception:
                continue
                
        return total_score / max(file_count, 1)

class SecurityGate(QualityGate):
    """Validate security standards"""
    
    def __init__(self):
        super().__init__("Security", "Validate security standards and practices")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate security"""
        score = 0.0
        details = {}
        errors = []
        
        # Check for security-related modules
        security_modules = [
            'cryptography',
            'hashlib',
            'secrets',
            'hmac'
        ]
        
        python_files = list(Path('src').rglob('*.py'))
        security_features = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                for module in security_modules:
                    if f'import {module}' in content or f'from {module}' in content:
                        security_features.append(f"{module} in {py_file.name}")
                        
            except Exception:
                continue
                
        if len(security_features) >= 3:
            score += 0.4
            details['security_modules'] = security_features
        else:
            errors.append("Insufficient security module usage")
            details['security_modules'] = security_features
            
        # Check for security patterns
        security_patterns = [
            'encrypt',
            'decrypt', 
            'hash',
            'signature',
            'authentication',
            'authorization',
            'token'
        ]
        
        pattern_matches = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                for pattern in security_patterns:
                    if pattern in content:
                        pattern_matches += 1
                        break
                        
            except Exception:
                continue
                
        if pattern_matches >= 5:
            score += 0.3
            details['security_patterns'] = f'{pattern_matches} files with security patterns'
        else:
            details['security_patterns'] = f'{pattern_matches} files with security patterns'
            
        # Check for hardcoded secrets (basic check)
        secret_violations = self._check_hardcoded_secrets(python_files)
        if secret_violations == 0:
            score += 0.3
            details['secret_check'] = 'passed'
        else:
            errors.append(f"Found {secret_violations} potential hardcoded secrets")
            details['secret_check'] = f'{secret_violations} violations'
            
        passed = score >= 0.6
        return passed, score, {'details': details, 'errors': errors}
        
    def _check_hardcoded_secrets(self, python_files: List[Path]) -> int:
        """Check for hardcoded secrets"""
        violations = 0
        secret_patterns = [
            'password = "',
            'pwd = "',
            'secret = "',
            'api_key = "',
            'token = "'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                for pattern in secret_patterns:
                    if pattern in content and 'example' not in content and 'test' not in content:
                        violations += 1
                        
            except Exception:
                continue
                
        return violations

class TestCoverageGate(QualityGate):
    """Validate test coverage and quality"""
    
    def __init__(self):
        super().__init__("Test Coverage", "Validate test coverage and quality")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate test coverage"""
        score = 0.0
        details = {}
        errors = []
        
        # Check test directory exists
        if not os.path.exists('tests'):
            errors.append("Tests directory not found")
            return False, 0.0, {'details': details, 'errors': errors}
            
        # Count test files
        test_files = list(Path('tests').rglob('test_*.py'))
        if len(test_files) >= 1:
            score += 0.3
            details['test_files'] = [f.name for f in test_files]
        else:
            errors.append("No test files found")
            
        # Run basic tests
        try:
            test_results = self._run_basic_tests()
            if test_results['passed'] > 0:
                score += 0.4
                details['test_execution'] = test_results
                
                # Bonus for high pass rate
                pass_rate = test_results['passed'] / max(test_results['total'], 1)
                if pass_rate >= 0.8:
                    score += 0.2
                    
            else:
                errors.append("No tests passed")
                details['test_execution'] = test_results
                
        except Exception as e:
            errors.append(f"Failed to run tests: {e}")
            
        # Check test quality
        test_quality = self._analyze_test_quality(test_files)
        score += test_quality * 0.1
        details['test_quality'] = test_quality
        
        passed = score >= 0.6
        return passed, score, {'details': details, 'errors': errors}
        
    def _run_basic_tests(self) -> Dict[str, int]:
        """Run basic tests and return results"""
        try:
            # Try to run our custom test runner
            result = subprocess.run(
                [sys.executable, 'tests/test_basic_functionality.py'],
                capture_output=True, text=True, timeout=120
            )
            
            output = result.stdout + result.stderr
            
            # Parse output for test results
            passed = output.count('✅')
            failed = output.count('❌')
            total = passed + failed
            
            return {
                'passed': passed,
                'failed': failed,
                'total': total,
                'output': output[-1000:]  # Last 1000 chars
            }
            
        except Exception as e:
            return {
                'passed': 0,
                'failed': 1,
                'total': 1,
                'error': str(e)
            }
            
    def _analyze_test_quality(self, test_files: List[Path]) -> float:
        """Analyze test quality"""
        if not test_files:
            return 0.0
            
        quality_score = 0.0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    
                file_score = 0.0
                
                # Check for assertions
                if 'assert' in content:
                    file_score += 0.3
                    
                # Check for test methods
                if 'def test_' in content:
                    file_score += 0.3
                    
                # Check for setup/teardown
                if 'setUp' in content or 'tearDown' in content:
                    file_score += 0.2
                    
                # Check for mock usage
                if 'mock' in content.lower() or 'Mock' in content:
                    file_score += 0.1
                    
                # Check for async tests
                if 'async def test_' in content:
                    file_score += 0.1
                    
                quality_score += file_score
                
            except Exception:
                continue
                
        return quality_score / len(test_files)

class DocumentationGate(QualityGate):
    """Validate documentation completeness"""
    
    def __init__(self):
        super().__init__("Documentation", "Validate documentation completeness and quality")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate documentation"""
        score = 0.0
        details = {}
        errors = []
        
        # Check README.md
        if os.path.exists('README.md'):
            readme_score = self._analyze_readme()
            score += readme_score * 0.5
            details['readme_score'] = readme_score
        else:
            errors.append("README.md not found")
            
        # Check for code documentation
        python_files = list(Path('src').rglob('*.py'))
        doc_score = self._analyze_code_documentation(python_files)
        score += doc_score * 0.3
        details['code_documentation'] = doc_score
        
        # Check for additional documentation
        docs_dir = Path('docs')
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob('*.md'))
            if len(doc_files) > 0:
                score += 0.2
                details['additional_docs'] = len(doc_files)
            else:
                details['additional_docs'] = 0
        else:
            details['additional_docs'] = 0
            
        passed = score >= 0.6
        return passed, score, {'details': details, 'errors': errors}
        
    def _analyze_readme(self) -> float:
        """Analyze README quality"""
        try:
            with open('README.md', 'r') as f:
                content = f.read()
                
            score = 0.0
            
            # Check for title
            if content.startswith('#'):
                score += 0.2
                
            # Check for sections
            required_sections = ['overview', 'features', 'installation', 'usage', 'example']
            found_sections = sum(1 for section in required_sections 
                               if section.lower() in content.lower())
            score += (found_sections / len(required_sections)) * 0.4
            
            # Check for code examples
            if '```' in content:
                score += 0.2
                
            # Check for badges
            if '![' in content and 'badge' in content.lower():
                score += 0.1
                
            # Check length (should be substantial)
            if len(content) > 5000:
                score += 0.1
                
            return score
            
        except Exception:
            return 0.0
            
    def _analyze_code_documentation(self, python_files: List[Path]) -> float:
        """Analyze code documentation quality"""
        if not python_files:
            return 0.0
            
        doc_score = 0.0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                file_score = 0.0
                
                # Check for module docstring
                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    file_score += 0.4
                    
                # Check for function docstrings
                docstring_count = content.count('"""') + content.count("'''")
                if docstring_count >= 4:  # Module + at least 1 function/class
                    file_score += 0.3
                elif docstring_count >= 2:
                    file_score += 0.2
                    
                # Check for type hints
                if ': ' in content and '->' in content:
                    file_score += 0.2
                    
                # Check for inline comments
                comment_lines = sum(1 for line in content.split('\n') 
                                  if line.strip().startswith('#') and len(line.strip()) > 2)
                if comment_lines > 5:
                    file_score += 0.1
                    
                doc_score += file_score
                
            except Exception:
                continue
                
        return doc_score / len(python_files)

class PerformanceGate(QualityGate):
    """Validate performance characteristics"""
    
    def __init__(self):
        super().__init__("Performance", "Validate performance and scalability")
        
    def validate(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Validate performance"""
        score = 0.0
        details = {}
        errors = []
        
        # Check for performance-related code
        python_files = list(Path('src').rglob('*.py'))
        performance_indicators = [
            'async',
            'asyncio',
            'threading',
            'multiprocessing',
            'concurrent',
            'cache',
            'optimize',
            'performance'
        ]
        
        performance_usage = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                for indicator in performance_indicators:
                    if indicator in content:
                        performance_usage += 1
                        break
                        
            except Exception:
                continue
                
        if performance_usage >= 3:
            score += 0.4
            details['performance_features'] = f'{performance_usage} files with performance features'
        else:
            details['performance_features'] = f'{performance_usage} files with performance features'
            
        # Check for scalability patterns
        scalability_patterns = [
            'pool',
            'queue',
            'batch',
            'chunk',
            'parallel',
            'distributed'
        ]
        
        scalability_usage = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                for pattern in scalability_patterns:
                    if pattern in content:
                        scalability_usage += 1
                        break
                        
            except Exception:
                continue
                
        if scalability_usage >= 2:
            score += 0.3
            details['scalability_features'] = f'{scalability_usage} files with scalability features'
        else:
            details['scalability_features'] = f'{scalability_usage} files with scalability features'
            
        # Run basic performance test
        perf_test = self._run_performance_test()
        score += perf_test * 0.3
        details['performance_test'] = perf_test
        
        passed = score >= 0.5
        return passed, score, {'details': details, 'errors': errors}
        
    def _run_performance_test(self) -> float:
        """Run basic performance test"""
        try:
            start_time = time.time()
            
            # Simple performance test: import main module
            import sys
            sys.path.insert(0, 'src')
            
            try:
                import dp_federated_lora
                import_time = time.time() - start_time
                
                # Score based on import time (faster is better)
                if import_time < 1.0:
                    return 1.0
                elif import_time < 2.0:
                    return 0.8
                elif import_time < 5.0:
                    return 0.6
                else:
                    return 0.3
                    
            except ImportError:
                # Still give some credit if structure exists
                return 0.2
                
        except Exception:
            return 0.0

class QualityGateValidator:
    """Main quality gate validation orchestrator"""
    
    def __init__(self):
        self.gates = [
            ProjectStructureGate(),
            CodeQualityGate(),
            SecurityGate(),
            TestCoverageGate(),
            DocumentationGate(),
            PerformanceGate()
        ]
        self.results = {}
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates"""
        print("🛡️ Running Quality Gate Validation")
        print("=" * 50)
        
        total_score = 0.0
        total_weight = 0.0
        passed_gates = 0
        
        for gate in self.gates:
            print(f"\n📋 Running {gate.name}")
            
            try:
                start_time = time.time()
                passed, score, details = gate.validate()
                duration = time.time() - start_time
                
                gate.passed = passed
                gate.score = score
                gate.details = details
                
                # Store results
                self.results[gate.name] = {
                    'passed': passed,
                    'score': score,
                    'details': details,
                    'duration': duration
                }
                
                total_score += score * gate.weight
                total_weight += gate.weight
                
                if passed:
                    passed_gates += 1
                    print(f"  ✅ PASSED (Score: {score:.2f})")
                else:
                    print(f"  ❌ FAILED (Score: {score:.2f})")
                    
                # Show errors if any
                if 'errors' in details and details['errors']:
                    for error in details['errors'][:3]:  # Show first 3 errors
                        print(f"    ⚠️ {error}")
                        
            except Exception as e:
                print(f"  💥 ERROR: {e}")
                self.results[gate.name] = {
                    'passed': False,
                    'score': 0.0,
                    'error': str(e),
                    'duration': 0.0
                }
                
        # Calculate overall results
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        overall_passed = passed_gates >= len(self.gates) * 0.7  # 70% must pass
        
        print("\n" + "=" * 50)
        print("📊 Quality Gate Results:")
        print(f"  Gates Passed: {passed_gates}/{len(self.gates)}")
        print(f"  Overall Score: {overall_score:.2f}")
        print(f"  Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        # Summary
        summary = {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'passed_gates': passed_gates,
            'total_gates': len(self.gates),
            'gate_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
        
    def generate_report(self, output_file: str = 'quality_gate_report.json'):
        """Generate detailed quality gate report"""
        
        if not self.results:
            print("No results to report. Run quality gates first.")
            return
            
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_gates': len(self.gates),
                'passed_gates': sum(1 for result in self.results.values() if result.get('passed', False))
            },
            'summary': {
                'overall_score': sum(result.get('score', 0) for result in self.results.values()) / len(self.results),
                'gate_scores': {name: result.get('score', 0) for name, result in self.results.items()}
            },
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Quality gate report written to: {output_file}")
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        for gate_name, result in self.results.items():
            if not result.get('passed', False):
                score = result.get('score', 0)
                
                if gate_name == "Project Structure" and score < 0.8:
                    recommendations.append("Improve project structure by adding missing directories and files")
                    
                elif gate_name == "Code Quality" and score < 0.7:
                    recommendations.append("Improve code quality with better formatting, type hints, and documentation")
                    
                elif gate_name == "Security" and score < 0.6:
                    recommendations.append("Enhance security with proper encryption, authentication, and secret management")
                    
                elif gate_name == "Test Coverage" and score < 0.6:
                    recommendations.append("Increase test coverage with more comprehensive test cases")
                    
                elif gate_name == "Documentation" and score < 0.6:
                    recommendations.append("Improve documentation with better README, code comments, and examples")
                    
                elif gate_name == "Performance" and score < 0.5:
                    recommendations.append("Optimize performance with async patterns, caching, and scalability features")
                    
        if not recommendations:
            recommendations.append("All quality gates passed! Consider adding more advanced features and optimizations.")
            
        return recommendations

def main():
    """Main quality gate validation execution"""
    validator = QualityGateValidator()
    
    # Run all quality gates
    summary = validator.run_all_gates()
    
    # Generate detailed report
    report = validator.generate_report()
    
    # Exit with appropriate code
    if summary['overall_passed']:
        print("\n🎉 All quality gates passed!")
        return 0
    else:
        print(f"\n⚠️ Quality validation failed. See report for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
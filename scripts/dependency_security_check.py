#!/usr/bin/env python3
"""
Dependency Security Checker

Scans dependencies for known vulnerabilities and security issues.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


class DependencySecurityChecker:
    """Check dependencies for security vulnerabilities."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.vulnerability_db = self._load_vulnerability_db()
    
    def _load_vulnerability_db(self) -> Dict[str, Any]:
        """Load known vulnerability database."""
        # Simplified vulnerability database
        return {
            "pillow": {
                "vulnerable_versions": ["< 9.0.0"],
                "cve": ["CVE-2022-22817", "CVE-2022-22816"],
                "severity": "HIGH",
                "description": "Path traversal and buffer overflow vulnerabilities"
            },
            "requests": {
                "vulnerable_versions": ["< 2.20.0"],
                "cve": ["CVE-2018-18074"],
                "severity": "MEDIUM",
                "description": "Session fixation vulnerability"
            },
            "pyyaml": {
                "vulnerable_versions": ["< 5.4"],
                "cve": ["CVE-2020-14343", "CVE-2020-1747"],
                "severity": "HIGH",
                "description": "Arbitrary code execution via unsafe loading"
            },
            "jinja2": {
                "vulnerable_versions": ["< 2.11.3"],
                "cve": ["CVE-2020-28493"],
                "severity": "MEDIUM",
                "description": "Regular expression denial of service"
            }
        }
    
    def check_requirements_file(self, requirements_path: Path) -> List[Dict[str, Any]]:
        """Check requirements file for vulnerable packages."""
        vulnerabilities = []
        
        if not requirements_path.exists():
            return vulnerabilities
        
        try:
            content = requirements_path.read_text()
            lines = content.strip().split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse package name and version
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                package_name = package_name.lower()
                
                if package_name in self.vulnerability_db:
                    vuln_info = self.vulnerability_db[package_name]
                    
                    vulnerability = {
                        "file": str(requirements_path),
                        "line": line_num,
                        "package": package_name,
                        "current_spec": line,
                        "vulnerability": vuln_info,
                        "recommendation": f"Update {package_name} to a secure version"
                    }
                    vulnerabilities.append(vulnerability)
        
        except Exception as e:
            print(f"Error reading {requirements_path}: {e}")
        
        return vulnerabilities
    
    def run_safety_check(self) -> List[Dict[str, Any]]:
        """Run safety check if available."""
        vulnerabilities = []
        
        try:
            # Try to run safety check
            result = subprocess.run(
                ["python", "-m", "safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse safety output
                if result.stdout.strip():
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerabilities.append({
                            "source": "safety",
                            "package": vuln.get("package", "unknown"),
                            "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                            "affected_versions": vuln.get("affected_versions", "unknown"),
                            "description": vuln.get("advisory", "No description available")
                        })
        
        except Exception:
            # Safety not available, skip
            pass
        
        return vulnerabilities
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency security report."""
        report = {
            "timestamp": time.time(),
            "vulnerabilities": [],
            "recommendations": [],
            "summary": {
                "total_vulnerabilities": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0
            }
        }
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements-dev.txt"]
        for req_file in req_files:
            req_path = self.project_root / req_file
            vulns = self.check_requirements_file(req_path)
            report["vulnerabilities"].extend(vulns)
        
        # Run safety check
        safety_vulns = self.run_safety_check()
        report["vulnerabilities"].extend(safety_vulns)
        
        # Calculate summary
        report["summary"]["total_vulnerabilities"] = len(report["vulnerabilities"])
        
        for vuln in report["vulnerabilities"]:
            severity = "MEDIUM"  # Default
            if "vulnerability" in vuln:
                severity = vuln["vulnerability"].get("severity", "MEDIUM")
            elif "advisory" in vuln:
                # Heuristic severity detection
                advisory = vuln["advisory"].lower()
                if any(word in advisory for word in ["critical", "execute", "rce"]):
                    severity = "HIGH"
                elif any(word in advisory for word in ["denial", "dos"]):
                    severity = "MEDIUM"
            
            if severity == "HIGH":
                report["summary"]["high_severity"] += 1
            elif severity == "MEDIUM":
                report["summary"]["medium_severity"] += 1
            else:
                report["summary"]["low_severity"] += 1
        
        # Generate recommendations
        if report["vulnerabilities"]:
            report["recommendations"] = [
                "Update vulnerable packages to secure versions",
                "Pin package versions in requirements files",
                "Implement automated dependency scanning in CI/CD",
                "Regular security audits of dependencies",
                "Use virtual environments to isolate dependencies"
            ]
        else:
            report["recommendations"] = [
                "Continue regular dependency security monitoring",
                "Keep dependencies up to date",
                "Monitor security advisories for used packages"
            ]
        
        return report
    
    def fix_vulnerabilities(self) -> List[str]:
        """Attempt to fix known vulnerabilities automatically."""
        fixes_applied = []
        
        # This is a simplified fix mechanism
        # In practice, this would need more sophisticated version resolution
        
        req_path = self.project_root / "requirements.txt"
        if req_path.exists():
            try:
                content = req_path.read_text()
                lines = content.strip().split('\n')
                modified = False
                
                new_lines = []
                for line in lines:
                    original_line = line
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        new_lines.append(original_line)
                        continue
                    
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip().lower()
                    
                    # Apply fixes for known vulnerabilities
                    if package_name == "pillow" and "==" in line:
                        version = line.split('==')[1].strip()
                        if version.startswith(('6.', '7.', '8.')):
                            new_line = f"pillow>=9.0.0"
                            new_lines.append(new_line)
                            fixes_applied.append(f"Updated pillow from {version} to >=9.0.0")
                            modified = True
                        else:
                            new_lines.append(original_line)
                    elif package_name == "pyyaml" and "==" in line:
                        version = line.split('==')[1].strip()
                        if version.startswith(('3.', '4.', '5.0', '5.1', '5.2', '5.3')):
                            new_line = f"pyyaml>=5.4.1"
                            new_lines.append(new_line)
                            fixes_applied.append(f"Updated PyYAML from {version} to >=5.4.1")
                            modified = True
                        else:
                            new_lines.append(original_line)
                    else:
                        new_lines.append(original_line)
                
                if modified:
                    req_path.write_text('\n'.join(new_lines))
                    fixes_applied.append("Updated requirements.txt with security fixes")
            
            except Exception as e:
                fixes_applied.append(f"Error applying fixes: {e}")
        
        return fixes_applied


def main():
    """Main function for standalone execution."""
    checker = DependencySecurityChecker()
    
    print("🔍 Running dependency security check...")
    
    # Generate report
    report = checker.generate_security_report()
    
    # Save report
    report_path = Path("dependency_security_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📊 Found {report['summary']['total_vulnerabilities']} vulnerabilities")
    print(f"   High severity: {report['summary']['high_severity']}")
    print(f"   Medium severity: {report['summary']['medium_severity']}")
    print(f"   Low severity: {report['summary']['low_severity']}")
    
    # Apply automatic fixes
    if report["vulnerabilities"]:
        print("\n🔧 Attempting automatic fixes...")
        fixes = checker.fix_vulnerabilities()
        for fix in fixes:
            print(f"   ✅ {fix}")
    
    print(f"\n📁 Report saved: {report_path}")
    
    return report['summary']['total_vulnerabilities'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

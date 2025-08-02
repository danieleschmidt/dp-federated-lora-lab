#!/usr/bin/env python3
"""
License header checker for dp-federated-lora-lab.

This script ensures all Python files contain proper license headers
for compliance with open source licensing requirements.
"""

import sys
import re
from pathlib import Path
from typing import List, Optional

# Expected license header template
LICENSE_HEADER = '''"""
Copyright (c) 2025 Terragon Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""'''

EXCLUDE_PATTERNS = [
    r'__init__\.py$',
    r'test_.*\.py$',
    r'conftest\.py$',
    r'setup\.py$',
    r'.*_test\.py$',
]

def has_license_header(file_path: Path) -> bool:
    """Check if a Python file has a license header."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Look for copyright notice in the first 50 lines
        lines = content.split('\n')[:50]
        first_50_lines = '\n'.join(lines).lower()
        
        # Check for key license terms
        required_terms = [
            'copyright',
            'permission is hereby granted',
            'mit license' or 'software is provided "as is"'
        ]
        
        return any(term in first_50_lines for term in required_terms[:2])
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def should_exclude(file_path: Path) -> bool:
    """Check if file should be excluded from license header check."""
    file_str = str(file_path)
    return any(re.search(pattern, file_str) for pattern in EXCLUDE_PATTERNS)

def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files that need license headers."""
    python_files = []
    
    for py_file in root_dir.rglob("*.py"):
        if not should_exclude(py_file):
            python_files.append(py_file)
    
    return python_files

def main() -> int:
    """Main license header checker."""
    root_dir = Path(".")
    python_files = find_python_files(root_dir)
    
    missing_license = []
    
    for py_file in python_files:
        if not has_license_header(py_file):
            missing_license.append(py_file)
    
    if missing_license:
        print("❌ Files missing license headers:")
        for file_path in missing_license:
            print(f"  - {file_path}")
        print("\nTo fix: Add MIT license header to the top of each file.")
        print("Use the template in scripts/license_header_template.py")
        return 1
    
    print(f"✅ All {len(python_files)} Python files have proper license headers.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
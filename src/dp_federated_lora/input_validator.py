"""
Comprehensive Input Validation Module

Provides secure input validation and sanitization.
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Common validation patterns
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$',
        'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        'alphanumeric': r'^[a-zA-Z0-9]+$',
        'filename': r'^[a-zA-Z0-9._-]+$',
        'path': r'^[a-zA-Z0-9._/-]+$'
    }
    
    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',               # XSS
        r'vbscript:',                # XSS
        r'onload\s*=',               # XSS
        r'onerror\s*=',              # XSS
        r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL injection
        r'\.\./|\.\.\\',           # Path traversal
        r'(exec|eval|system|passthru)\s*\(',  # Code injection
        r'(\${|<%|<\?)',             # Template injection
    ]
    
    @classmethod
    def validate_string(cls, value: str, pattern: str = None, max_length: int = None, 
                       min_length: int = None, allow_empty: bool = True) -> bool:
        """Validate string input."""
        if not isinstance(value, str):
            return False
        
        if not allow_empty and not value.strip():
            return False
        
        if min_length and len(value) < min_length:
            return False
        
        if max_length and len(value) > max_length:
            return False
        
        # Check for dangerous patterns
        for dangerous_pattern in cls.DANGEROUS_PATTERNS:
            if re.search(dangerous_pattern, value, re.IGNORECASE):
                return False
        
        # Check specific pattern if provided
        if pattern and pattern in cls.PATTERNS:
            return bool(re.match(cls.PATTERNS[pattern], value))
        elif pattern:
            return bool(re.match(pattern, value))
        
        return True
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return ""
        
        # HTML escape
        value = html.escape(value)
        
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        return value
    
    @classmethod
    def validate_json(cls, value: str) -> bool:
        """Validate JSON input."""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: List[str] = None) -> bool:
        """Validate URL input."""
        if not cls.validate_string(url, 'url'):
            return False
        
        try:
            parsed = urlparse(url)
            if allowed_schemes:
                return parsed.scheme in allowed_schemes
            return parsed.scheme in ['http', 'https']
        except Exception:
            return False
    
    @classmethod
    def validate_file_path(cls, path: str, allowed_extensions: List[str] = None) -> bool:
        """Validate file path input."""
        # Check for path traversal
        if '..' in path or path.startswith('/'):
            return False
        
        # Check for dangerous characters
        if any(char in path for char in ['<', '>', '|', ':', '*', '?', '"']):
            return False
        
        # Check file extension if specified
        if allowed_extensions:
            ext = path.split('.')[-1].lower()
            return ext in [e.lower() for e in allowed_extensions]
        
        return True
    
    @classmethod
    def validate_integer(cls, value: Union[int, str], min_val: int = None, 
                        max_val: int = None) -> bool:
        """Validate integer input."""
        try:
            int_val = int(value)
            
            if min_val is not None and int_val < min_val:
                return False
            
            if max_val is not None and int_val > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def validate_float(cls, value: Union[float, str], min_val: float = None, 
                      max_val: float = None) -> bool:
        """Validate float input."""
        try:
            float_val = float(value)
            
            if min_val is not None and float_val < min_val:
                return False
            
            if max_val is not None and float_val > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def validate_request_data(cls, data: Dict[str, Any], schema: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """Validate request data against schema."""
        errors = []
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Required field check
            if rules.get('required', False) and value is None:
                errors.append(f"Field '{field}' is required")
                continue
            
            if value is None:
                continue
            
            # Type validation
            expected_type = rules.get('type', str)
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
                continue
            
            # String validation
            if expected_type == str:
                max_length = rules.get('max_length')
                min_length = rules.get('min_length')
                pattern = rules.get('pattern')
                
                if not cls.validate_string(value, pattern, max_length, min_length):
                    errors.append(f"Field '{field}' failed validation")
            
            # Integer validation
            elif expected_type == int:
                min_val = rules.get('min_value')
                max_val = rules.get('max_value')
                
                if not cls.validate_integer(value, min_val, max_val):
                    errors.append(f"Field '{field}' is out of valid range")
            
            # Float validation
            elif expected_type == float:
                min_val = rules.get('min_value')
                max_val = rules.get('max_value')
                
                if not cls.validate_float(value, min_val, max_val):
                    errors.append(f"Field '{field}' is out of valid range")
        
        return len(errors) == 0, errors


# Rate limiting helper
class RateLimiter:
    """Simple rate limiting implementation."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, identifier: str, max_requests: int = 100, 
                   time_window: int = 3600) -> bool:
        """Check if request is allowed under rate limiting."""
        import time
        
        current_time = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < time_window
        ]
        
        # Check rate limit
        if len(self.requests[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True

"""
Security Middleware and Headers Implementation

Provides comprehensive security headers and middleware.
"""

from typing import Dict, Callable, Any
import time
import hashlib
import secrets


class SecurityMiddleware:
    """Comprehensive security middleware."""
    
    def __init__(self, app: Any = None):
        self.app = app
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin"
        }
        
        # Rate limiting storage
        self.rate_limit_storage = {}
        
        # Security event logging
        self.security_events = []
    
    def add_security_headers(self, response: Any) -> Any:
        """Add security headers to response."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add nonce for CSP if needed
        nonce = secrets.token_urlsafe(16)
        response.headers["X-CSP-Nonce"] = nonce
        
        return response
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check request rate limiting."""
        current_time = time.time()
        
        if identifier not in self.rate_limit_storage:
            self.rate_limit_storage[identifier] = []
        
        # Clean old entries
        self.rate_limit_storage[identifier] = [
            timestamp for timestamp in self.rate_limit_storage[identifier]
            if current_time - timestamp < window_seconds
        ]
        
        # Check limit
        if len(self.rate_limit_storage[identifier]) >= max_requests:
            self.log_security_event("RATE_LIMIT_EXCEEDED", {"identifier": identifier})
            return False
        
        # Add current request
        self.rate_limit_storage[identifier].append(current_time)
        return True
    
    def validate_request_size(self, content_length: int, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate request content size."""
        if content_length > max_size:
            self.log_security_event("REQUEST_TOO_LARGE", {"size": content_length})
            return False
        return True
    
    def detect_suspicious_patterns(self, data: str) -> bool:
        """Detect suspicious patterns in request data."""
        suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'union.*select',
            r'drop.*table',
            r'\.\./.*etc/passwd',
            r'cmd\.exe',
            r'powershell',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                self.log_security_event("SUSPICIOUS_PATTERN", {"pattern": pattern})
                return True
        
        return False
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
            "event_id": secrets.token_hex(8)
        }
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security monitoring report."""
        event_counts = {}
        recent_events = [
            event for event in self.security_events
            if time.time() - event["timestamp"] < 86400  # Last 24 hours
        ]
        
        for event in recent_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(recent_events),
            "event_types": event_counts,
            "recent_events": recent_events[-10:],  # Last 10 events
            "rate_limit_active_ips": len(self.rate_limit_storage)
        }
    
    def process_request(self, request: Any) -> tuple[bool, str]:
        """Process incoming request through security checks."""
        # Check rate limiting
        client_ip = getattr(request, 'remote_addr', 'unknown')
        if not self.check_rate_limit(client_ip):
            return False, "Rate limit exceeded"
        
        # Check request size
        content_length = getattr(request, 'content_length', 0) or 0
        if not self.validate_request_size(content_length):
            return False, "Request too large"
        
        # Check for suspicious patterns
        request_data = str(getattr(request, 'data', ''))
        if self.detect_suspicious_patterns(request_data):
            return False, "Suspicious request blocked"
        
        return True, "OK"


# CSRF Protection
class CSRFProtection:
    """CSRF (Cross-Site Request Forgery) protection."""
    
    def __init__(self):
        self.tokens = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session."""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = {
            "token": token,
            "timestamp": time.time()
        }
        return token
    
    def validate_token(self, session_id: str, provided_token: str) -> bool:
        """Validate CSRF token."""
        if session_id not in self.tokens:
            return False
        
        stored_data = self.tokens[session_id]
        
        # Check token expiry (1 hour)
        if time.time() - stored_data["timestamp"] > 3600:
            del self.tokens[session_id]
            return False
        
        return secrets.compare_digest(stored_data["token"], provided_token)
    
    def cleanup_expired_tokens(self) -> None:
        """Clean up expired CSRF tokens."""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, data in self.tokens.items()
            if current_time - data["timestamp"] > 3600
        ]
        
        for session_id in expired_sessions:
            del self.tokens[session_id]


# Global instances
security_middleware = SecurityMiddleware()
csrf_protection = CSRFProtection()

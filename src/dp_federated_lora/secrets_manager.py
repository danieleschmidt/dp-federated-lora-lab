"""
Secure Secrets Management Utility

Provides secure access to environment variables and secrets.
"""

import os
import base64
import secrets
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecureSecretsManager:
    """Secure secrets management with encryption."""
    
    def __init__(self):
        self._encryption_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption key from environment."""
        key_material = os.getenv("ENCRYPTION_KEY")
        if not key_material:
            # Generate a new key if none exists
            key = Fernet.generate_key()
            print(f"Generated new encryption key: {key.decode()}")
            print("Please save this key to your .env file as ENCRYPTION_KEY")
            self._encryption_key = key
        else:
            try:
                self._encryption_key = key_material.encode()
                # Validate key
                Fernet(self._encryption_key)
            except Exception:
                # Invalid key, generate new one
                self._encryption_key = Fernet.generate_key()
    
    def get_secret(self, key: str, default: Optional[str] = None, required: bool = True) -> Optional[str]:
        """Get secret from environment with validation."""
        value = os.getenv(key, default)
        
        if required and not value:
            raise ValueError(f"Required secret '{key}' not found in environment")
        
        if value and key.upper() in ["PASSWORD", "SECRET", "KEY", "TOKEN"]:
            if len(value) < 8:
                raise ValueError(f"Secret '{key}' is too short (minimum 8 characters)")
        
        return value
    
    def generate_secure_key(self, length: int = 32) -> str:
        """Generate cryptographically secure key."""
        return secrets.token_urlsafe(length)
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt secret value."""
        if not self._encryption_key:
            raise ValueError("Encryption key not available")
        
        fernet = Fernet(self._encryption_key)
        encrypted = fernet.encrypt(secret.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt secret value."""
        if not self._encryption_key:
            raise ValueError("Encryption key not available")
        
        fernet = Fernet(self._encryption_key)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_secret.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate all required secrets are present."""
        required_secrets = [
            "SECRET_KEY",
            "DATABASE_URL",
            "API_KEY"
        ]
        
        validation_results = {}
        for secret in required_secrets:
            try:
                value = self.get_secret(secret, required=True)
                validation_results[secret] = bool(value)
            except ValueError:
                validation_results[secret] = False
        
        return validation_results


# Global instance
secrets_manager = SecureSecretsManager()

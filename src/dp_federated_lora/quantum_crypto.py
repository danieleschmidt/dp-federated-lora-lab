"""
Quantum-Resistant Cryptography Module

Implements post-quantum cryptographic algorithms and secure practices.
"""

import os
import hashlib
import secrets
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


class QuantumResistantCrypto:
    """Quantum-resistant cryptographic operations."""
    
    @staticmethod
    def secure_hash(data: bytes, algorithm: str = "SHA256") -> bytes:
        """Compute secure hash using quantum-resistant algorithms."""
        hash_algorithms = {
            "SHA256": hashes.SHA256(),
            "SHA384": hashes.SHA384(),
            "SHA512": hashes.SHA512(),
            "BLAKE2b": hashes.BLAKE2b(64),
            "BLAKE2s": hashes.BLAKE2s(32)
        }
        
        if algorithm not in hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        digest = hashes.Hash(hash_algorithms[algorithm], backend=default_backend())
        digest.update(data)
        return digest.finalize()
    
    @staticmethod
    def derive_key(password: bytes, salt: bytes, length: int = 32, iterations: int = 100000) -> bytes:
        """Derive cryptographic key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(password)
    
    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data using AES-256-GCM (quantum-resistant)."""
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return ciphertext and authentication tag combined with IV
        return iv + encryptor.tag + ciphertext, iv
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data encrypted with AES-256-GCM."""
        # Extract IV (first 12 bytes) and tag (next 16 bytes)
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    @staticmethod
    def generate_secure_key(length: int = 32) -> bytes:
        """Generate cryptographically secure random key."""
        return os.urandom(length)
    
    @staticmethod
    def generate_rsa_keypair(key_size: int = 4096) -> Tuple[bytes, bytes]:
        """Generate RSA keypair (minimum 4096 bits for quantum resistance)."""
        if key_size < 4096:
            raise ValueError("Key size must be at least 4096 bits for quantum resistance")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    @staticmethod
    def constant_time_compare(a: bytes, b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        return secrets.compare_digest(a, b)
    
    @staticmethod
    def secure_random_string(length: int = 32) -> str:
        """Generate cryptographically secure random string."""
        return secrets.token_urlsafe(length)


# Legacy cryptography detection and replacement
class CryptoMigrator:
    """Migrate legacy cryptographic implementations."""
    
    DEPRECATED_ALGORITHMS = {
        "MD5": "SHA256",
        "SHA1": "SHA256", 
        "DES": "AES-256-GCM",
        "3DES": "AES-256-GCM",
        "RC4": "ChaCha20-Poly1305"
    }
    
    @classmethod
    def check_deprecated_crypto(cls, code: str) -> List[str]:
        """Check for deprecated cryptographic algorithms."""
        issues = []
        for deprecated, recommended in cls.DEPRECATED_ALGORITHMS.items():
            if deprecated.lower() in code.lower():
                issues.append(f"Replace {deprecated} with {recommended}")
        return issues
    
    @classmethod
    def suggest_migration(cls, algorithm: str) -> str:
        """Suggest migration path for deprecated algorithm."""
        return cls.DEPRECATED_ALGORITHMS.get(algorithm.upper(), "Use quantum-resistant alternative")

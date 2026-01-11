"""
Encryption utilities for securing QRZ API keys.
"""

import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import config


def get_encryption_key() -> bytes:
    """
    Get or derive the encryption key from environment variable.

    Uses PBKDF2 to derive a proper Fernet key from the secret.
    Salt is configurable via ENCRYPTION_SALT environment variable for per-deployment security.
    """
    secret = config.ENCRYPTION_KEY
    salt = config.ENCRYPTION_SALT.encode()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key


def encrypt_api_key(api_key: str) -> str:
    """Encrypt a QRZ API key for storage."""
    fernet = Fernet(get_encryption_key())
    encrypted = fernet.encrypt(api_key.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt a stored QRZ API key."""
    fernet = Fernet(get_encryption_key())
    encrypted = base64.urlsafe_b64decode(encrypted_key.encode())
    decrypted = fernet.decrypt(encrypted)
    return decrypted.decode()

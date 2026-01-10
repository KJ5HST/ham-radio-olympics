"""
Authentication and session management for Ham Radio Olympics.
"""

import hashlib
import secrets
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from database import get_db
from config import config

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
SESSION_COOKIE_NAME = config.SESSION_COOKIE_NAME


@dataclass
class User:
    """Authenticated user."""
    callsign: str
    email: Optional[str]
    has_qrz_key: bool
    has_lotw_creds: bool = False
    is_admin: bool = False
    is_referee: bool = False


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=config.BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def _verify_sha256_password(password: str, stored_hash: str) -> bool:
    """Verify a password against legacy SHA-256 hash (for migration)."""
    try:
        salt, hash_value = stored_hash.split(":", 1)
        hash_input = f"{salt}{password}".encode()
        return hashlib.sha256(hash_input).hexdigest() == hash_value
    except ValueError:
        return False


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash. Supports both bcrypt and legacy SHA-256."""
    # Try bcrypt first (new format starts with $2b$)
    if stored_hash.startswith('$2b$') or stored_hash.startswith('$2a$'):
        try:
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        except Exception:
            return False

    # Fall back to legacy SHA-256 format (salt:hash)
    return _verify_sha256_password(password, stored_hash)


def _upgrade_password_hash(callsign: str, password: str) -> None:
    """Upgrade a legacy SHA-256 password to bcrypt."""
    new_hash = hash_password(password)
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET password_hash = ? WHERE callsign = ?",
            (new_hash, callsign)
        )
    logger.info(f"Upgraded password hash for {callsign} from SHA-256 to bcrypt")


def _is_account_locked(callsign: str) -> bool:
    """Check if account is locked due to failed login attempts."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT locked_until FROM competitors WHERE callsign = ?",
            (callsign,)
        )
        row = cursor.fetchone()
        if row and row["locked_until"]:
            locked_until = datetime.fromisoformat(row["locked_until"])
            if datetime.utcnow() < locked_until:
                return True
            # Lockout expired, clear it
            _clear_lockout(callsign)
    return False


def _record_failed_login(callsign: str) -> bool:
    """Record a failed login attempt. Returns True if account is now locked."""
    with get_db() as conn:
        # Increment failed attempts
        conn.execute(
            "UPDATE competitors SET failed_login_attempts = failed_login_attempts + 1 WHERE callsign = ?",
            (callsign,)
        )
        # Check if we need to lock
        cursor = conn.execute(
            "SELECT failed_login_attempts FROM competitors WHERE callsign = ?",
            (callsign,)
        )
        row = cursor.fetchone()
        if row and row["failed_login_attempts"] >= config.LOCKOUT_ATTEMPTS:
            locked_until = datetime.utcnow() + timedelta(minutes=config.LOCKOUT_DURATION_MINUTES)
            conn.execute(
                "UPDATE competitors SET locked_until = ? WHERE callsign = ?",
                (locked_until.isoformat(), callsign)
            )
            logger.warning(f"Account locked due to failed attempts: {callsign}")
            return True
    return False


def _clear_lockout(callsign: str) -> None:
    """Clear lockout after successful login or lockout expiry."""
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET failed_login_attempts = 0, locked_until = NULL WHERE callsign = ?",
            (callsign,)
        )


def create_session(callsign: str) -> str:
    """Create a new session for a user."""
    session_id = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(days=config.SESSION_DURATION_DAYS)

    with get_db() as conn:
        # Clean up expired sessions for this user
        conn.execute(
            "DELETE FROM sessions WHERE callsign = ? OR expires_at < ?",
            (callsign, now.isoformat())
        )
        # Create new session
        conn.execute(
            "INSERT INTO sessions (id, callsign, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, callsign, now.isoformat(), expires.isoformat())
        )

    return session_id


def get_session_user(session_id: Optional[str]) -> Optional[User]:
    """Get the user for a session, or None if invalid/expired/disabled."""
    if not session_id:
        return None

    with get_db() as conn:
        now = datetime.utcnow().isoformat()
        cursor = conn.execute("""
            SELECT c.callsign, c.email, c.qrz_api_key_encrypted,
                   c.lotw_username_encrypted, c.is_admin, c.is_referee
            FROM sessions s
            JOIN competitors c ON s.callsign = c.callsign
            WHERE s.id = ? AND s.expires_at > ? AND c.is_disabled = 0
        """, (session_id, now))
        row = cursor.fetchone()

        if row:
            return User(
                callsign=row["callsign"],
                email=row["email"],
                has_qrz_key=bool(row["qrz_api_key_encrypted"]),
                has_lotw_creds=bool(row["lotw_username_encrypted"]),
                is_admin=bool(row["is_admin"]),
                is_referee=bool(row["is_referee"])
            )

    return None


def delete_session(session_id: str) -> None:
    """Delete a session (logout)."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def register_user(
    callsign: str,
    password: str,
    email: Optional[str] = None,
    qrz_api_key_encrypted: Optional[str] = None,
    lotw_username_encrypted: Optional[str] = None,
    lotw_password_encrypted: Optional[str] = None
) -> bool:
    """Register a new user. Returns True if successful."""
    callsign = callsign.upper().strip()
    password_hash = hash_password(password)
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Check if user already exists
        existing = conn.execute(
            "SELECT 1 FROM competitors WHERE callsign = ?", (callsign,)
        ).fetchone()

        if existing:
            return False

        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, email, qrz_api_key_encrypted,
                                    lotw_username_encrypted, lotw_password_encrypted, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (callsign, password_hash, email, qrz_api_key_encrypted,
              lotw_username_encrypted, lotw_password_encrypted, now))

    logger.info(f"Registered new user: {callsign}")
    return True


def authenticate_user(callsign: str, password: str) -> Optional[str]:
    """Authenticate a user.

    Returns:
        session_id if successful
        'disabled' if account disabled
        'locked' if account is locked
        None if invalid credentials
    """
    callsign = callsign.upper().strip()

    # Check if account is locked first
    if _is_account_locked(callsign):
        logger.warning(f"Login attempt for locked account: {callsign}")
        return "locked"

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT password_hash, is_disabled FROM competitors WHERE callsign = ?",
            (callsign,)
        )
        row = cursor.fetchone()

        if row:
            stored_hash = row["password_hash"]
            is_legacy = not (stored_hash.startswith('$2b$') or stored_hash.startswith('$2a$'))

            if verify_password(password, stored_hash):
                if row["is_disabled"]:
                    logger.warning(f"Login attempt for disabled account: {callsign}")
                    return "disabled"

                # Clear any failed login attempts on successful login
                _clear_lockout(callsign)

                # Upgrade legacy SHA-256 hash to bcrypt on successful login
                if is_legacy:
                    _upgrade_password_hash(callsign, password)

                logger.info(f"Successful login: {callsign}")
                return create_session(callsign)
            else:
                logger.warning(f"Failed login attempt for: {callsign}")
                # Record failed attempt and potentially lock account
                if _record_failed_login(callsign):
                    return "locked"

    return None


def update_user_email(callsign: str, email: str) -> None:
    """Update user's email."""
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET email = ? WHERE callsign = ?",
            (email, callsign)
        )


def update_user_password(callsign: str, new_password: str) -> None:
    """Update user's password."""
    password_hash = hash_password(new_password)
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET password_hash = ? WHERE callsign = ?",
            (password_hash, callsign)
        )

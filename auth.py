"""
Authentication and session management for Ham Radio Olympics.
"""

import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from database import get_db


SESSION_DURATION_DAYS = 30
SESSION_COOKIE_NAME = "hro_session"


@dataclass
class User:
    """Authenticated user."""
    callsign: str
    email: Optional[str]
    has_qrz_key: bool
    is_admin: bool = False
    is_referee: bool = False


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hash_input = f"{salt}{password}".encode()
    password_hash = hashlib.sha256(hash_input).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt, hash_value = stored_hash.split(":", 1)
        hash_input = f"{salt}{password}".encode()
        return hashlib.sha256(hash_input).hexdigest() == hash_value
    except ValueError:
        return False


def create_session(callsign: str) -> str:
    """Create a new session for a user."""
    session_id = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(days=SESSION_DURATION_DAYS)

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
            SELECT c.callsign, c.email, c.qrz_api_key_encrypted, c.is_admin, c.is_referee
            FROM sessions s
            JOIN contestants c ON s.callsign = c.callsign
            WHERE s.id = ? AND s.expires_at > ? AND c.is_disabled = 0
        """, (session_id, now))
        row = cursor.fetchone()

        if row:
            return User(
                callsign=row["callsign"],
                email=row["email"],
                has_qrz_key=bool(row["qrz_api_key_encrypted"]),
                is_admin=bool(row["is_admin"]),
                is_referee=bool(row["is_referee"])
            )

    return None


def delete_session(session_id: str) -> None:
    """Delete a session (logout)."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def register_user(callsign: str, password: str, email: Optional[str] = None, qrz_api_key_encrypted: Optional[str] = None) -> bool:
    """Register a new user. Returns True if successful."""
    callsign = callsign.upper().strip()
    password_hash = hash_password(password)
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Check if user already exists
        existing = conn.execute(
            "SELECT 1 FROM contestants WHERE callsign = ?", (callsign,)
        ).fetchone()

        if existing:
            return False

        conn.execute("""
            INSERT INTO contestants (callsign, password_hash, email, qrz_api_key_encrypted, registered_at)
            VALUES (?, ?, ?, ?, ?)
        """, (callsign, password_hash, email, qrz_api_key_encrypted, now))

    return True


def authenticate_user(callsign: str, password: str) -> Optional[str]:
    """Authenticate a user. Returns session_id if successful, 'disabled' if account disabled, None otherwise."""
    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT password_hash, is_disabled FROM contestants WHERE callsign = ?",
            (callsign,)
        )
        row = cursor.fetchone()

        if row and verify_password(password, row["password_hash"]):
            if row["is_disabled"]:
                return "disabled"
            return create_session(callsign)

    return None


def update_user_email(callsign: str, email: str) -> None:
    """Update user's email."""
    with get_db() as conn:
        conn.execute(
            "UPDATE contestants SET email = ? WHERE callsign = ?",
            (email, callsign)
        )


def update_user_password(callsign: str, new_password: str) -> None:
    """Update user's password."""
    password_hash = hash_password(new_password)
    with get_db() as conn:
        conn.execute(
            "UPDATE contestants SET password_hash = ? WHERE callsign = ?",
            (password_hash, callsign)
        )

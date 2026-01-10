"""
Tests for authentication and user management.
"""

import pytest
from auth import (
    hash_password, verify_password, create_session, get_session_user,
    delete_session, register_user, authenticate_user, update_user_email,
    update_user_password
)
from database import get_db, reset_db


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()
    yield


class TestPasswordHashing:
    """Test password hashing functions."""

    def test_hash_password_creates_hash(self):
        """Test that hash_password creates a bcrypt hash."""
        password = "testpassword123"
        hashed = hash_password(password)

        # Bcrypt hashes start with $2b$ or $2a$
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert len(hashed) == 60  # Standard bcrypt hash length

    def test_hash_password_unique_salts(self):
        """Test that each hash has a unique salt."""
        password = "samepassword"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "mysecretpass"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "mysecretpass"
        hashed = hash_password(password)

        assert verify_password("wrongpassword", hashed) is False

    def test_verify_password_invalid_hash(self):
        """Test verifying with invalid hash format."""
        assert verify_password("password", "invalidhash") is False

    def test_verify_legacy_sha256_password(self):
        """Test verifying a legacy SHA-256 password."""
        import hashlib
        import secrets

        password = "legacypassword"
        # Create legacy format: salt:sha256_hash
        salt = secrets.token_hex(16)
        hash_input = f"{salt}{password}".encode()
        legacy_hash = f"{salt}:{hashlib.sha256(hash_input).hexdigest()}"

        # Should verify successfully
        assert verify_password(password, legacy_hash) is True
        assert verify_password("wrongpassword", legacy_hash) is False


class TestUserRegistration:
    """Test user registration."""

    def test_register_new_user(self):
        """Test registering a new user."""
        result = register_user("W1ABC", "password123", "test@example.com")
        assert result is True

        # Verify user exists in database
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM competitors WHERE callsign = ?", ("W1ABC",)
            )
            user = cursor.fetchone()
            assert user is not None
            assert user["email"] == "test@example.com"

    def test_register_duplicate_user(self):
        """Test registering a duplicate user fails."""
        register_user("K2DEF", "password123")
        result = register_user("K2DEF", "anotherpass")

        assert result is False

    def test_register_normalizes_callsign(self):
        """Test that callsign is normalized to uppercase."""
        register_user("w3xyz", "password123")

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM competitors WHERE callsign = ?", ("W3XYZ",)
            )
            assert cursor.fetchone() is not None

    def test_register_without_email(self):
        """Test registering without email."""
        result = register_user("N4TEST", "password123")
        assert result is True


class TestAuthentication:
    """Test user authentication."""

    def test_authenticate_valid_credentials(self):
        """Test authentication with valid credentials."""
        register_user("W1ABC", "password123")
        session_id = authenticate_user("W1ABC", "password123")

        assert session_id is not None
        assert len(session_id) > 20

    def test_authenticate_invalid_password(self):
        """Test authentication with wrong password."""
        register_user("W1ABC", "password123")
        session_id = authenticate_user("W1ABC", "wrongpassword")

        assert session_id is None

    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user."""
        session_id = authenticate_user("NOEXIST", "password123")
        assert session_id is None

    def test_authenticate_case_insensitive_callsign(self):
        """Test callsign is case-insensitive."""
        register_user("W1ABC", "password123")
        session_id = authenticate_user("w1abc", "password123")

        assert session_id is not None


class TestSessions:
    """Test session management."""

    def test_create_session(self):
        """Test creating a session."""
        register_user("W1ABC", "password123")
        session_id = create_session("W1ABC")

        assert session_id is not None
        assert len(session_id) > 20

    def test_get_session_user_valid(self):
        """Test getting user from valid session."""
        register_user("W1ABC", "password123", "test@example.com")
        session_id = create_session("W1ABC")

        user = get_session_user(session_id)

        assert user is not None
        assert user.callsign == "W1ABC"
        assert user.email == "test@example.com"

    def test_get_session_user_invalid(self):
        """Test getting user from invalid session."""
        user = get_session_user("invalid-session-id")
        assert user is None

    def test_get_session_user_none(self):
        """Test getting user with None session."""
        user = get_session_user(None)
        assert user is None

    def test_delete_session(self):
        """Test deleting a session."""
        register_user("W1ABC", "password123")
        session_id = create_session("W1ABC")

        # Verify session works
        assert get_session_user(session_id) is not None

        # Delete session
        delete_session(session_id)

        # Verify session no longer works
        assert get_session_user(session_id) is None


class TestUserUpdates:
    """Test user update functions."""

    def test_update_email(self):
        """Test updating user email."""
        register_user("W1ABC", "password123", "old@example.com")
        update_user_email("W1ABC", "new@example.com")

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT email FROM competitors WHERE callsign = ?", ("W1ABC",)
            )
            assert cursor.fetchone()["email"] == "new@example.com"

    def test_update_password(self):
        """Test updating user password."""
        register_user("W1ABC", "oldpassword")
        update_user_password("W1ABC", "newpassword")

        # Old password should fail
        assert authenticate_user("W1ABC", "oldpassword") is None
        # New password should work
        assert authenticate_user("W1ABC", "newpassword") is not None


class TestAdminRole:
    """Test admin role functionality."""

    def test_new_users_are_not_admin_by_default(self):
        """Test that new users are not admin by default."""
        register_user("W1ABC", "password123")
        session_id = create_session("W1ABC")
        user = get_session_user(session_id)

        assert user is not None
        assert user.is_admin is False

    def test_admin_status_in_database(self):
        """Test that is_admin defaults to 0 in database."""
        register_user("W1ABC", "password123")
        register_user("W2DEF", "password123")

        with get_db() as conn:
            # Both users should not be admin
            cursor = conn.execute(
                "SELECT is_admin FROM competitors WHERE callsign = ?", ("W1ABC",)
            )
            assert cursor.fetchone()["is_admin"] == 0

            cursor = conn.execute(
                "SELECT is_admin FROM competitors WHERE callsign = ?", ("W2DEF",)
            )
            assert cursor.fetchone()["is_admin"] == 0

    def test_admin_can_be_set_via_database(self):
        """Test that admin can be set via database update."""
        register_user("W1ADM", "password123")

        with get_db() as conn:
            conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = ?", ("W1ADM",))

        session_id = create_session("W1ADM")
        user = get_session_user(session_id)

        assert user is not None
        assert user.is_admin is True

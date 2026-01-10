"""
Tests for email service - TDD: Tests written before implementation.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Create temp file for test database BEFORE importing app modules
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["DATABASE_PATH"] = _test_db_path


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    from database import reset_db
    reset_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    """Cleanup database file after all tests."""
    yield
    if os.path.exists(_test_db_path):
        os.remove(_test_db_path)


class TestEmailServiceInterface:
    """Test email service interface and backends."""

    def test_email_service_exists(self):
        """Test that email_service module exists."""
        import email_service
        assert email_service is not None

    def test_send_email_function_exists(self):
        """Test that send_email function exists."""
        from email_service import send_email
        assert callable(send_email)

    def test_send_email_is_async(self):
        """Test that send_email is an async function."""
        import asyncio
        from email_service import send_email
        assert asyncio.iscoroutinefunction(send_email)

    @pytest.mark.asyncio
    async def test_send_email_accepts_required_params(self):
        """Test send_email accepts to, subject, and body parameters."""
        from email_service import send_email

        # Should not raise - just testing interface
        with patch('email_service._get_backend') as mock_backend:
            mock_backend.return_value = AsyncMock()
            await send_email(
                to="test@example.com",
                subject="Test Subject",
                body="Test body"
            )

    @pytest.mark.asyncio
    async def test_send_email_accepts_html_body(self):
        """Test send_email accepts optional html_body parameter."""
        from email_service import send_email

        with patch('email_service._get_backend') as mock_backend:
            mock_backend.return_value = AsyncMock()
            await send_email(
                to="test@example.com",
                subject="Test Subject",
                body="Plain text",
                html_body="<p>HTML body</p>"
            )


class TestConsoleBackend:
    """Test console email backend (for development/testing)."""

    @pytest.mark.asyncio
    async def test_console_backend_logs_email(self, capfd):
        """Test console backend prints email to stdout."""
        import email_service

        with patch.object(email_service.config, 'EMAIL_BACKEND', 'console'):
            await email_service.send_email(
                to="test@example.com",
                subject="Test Subject",
                body="Test body content"
            )

        captured = capfd.readouterr()
        assert "test@example.com" in captured.out
        assert "Test Subject" in captured.out
        assert "Test body content" in captured.out

    @pytest.mark.asyncio
    async def test_console_backend_returns_success(self):
        """Test console backend returns True."""
        import email_service

        with patch.object(email_service.config, 'EMAIL_BACKEND', 'console'):
            result = await email_service.send_email(
                to="test@example.com",
                subject="Test",
                body="Body"
            )
        assert result is True


class TestSMTPBackend:
    """Test SMTP email backend."""

    @pytest.mark.asyncio
    async def test_smtp_backend_connects_to_server(self):
        """Test SMTP backend attempts to connect to configured server."""
        import email_service
        import aiosmtplib

        with patch.object(email_service.config, 'EMAIL_BACKEND', 'smtp'):
            with patch.object(email_service.config, 'SMTP_HOST', 'smtp.test.com'):
                with patch.object(email_service.config, 'SMTP_PORT', 587):
                    with patch.object(aiosmtplib, 'send', new_callable=AsyncMock) as mock_send:
                        mock_send.return_value = ({}, "OK")
                        await email_service.send_email(
                            to="test@example.com",
                            subject="Test",
                            body="Body"
                        )
                        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_smtp_backend_uses_configured_credentials(self):
        """Test SMTP backend uses configured username and password."""
        import email_service
        import aiosmtplib

        with patch.object(email_service.config, 'EMAIL_BACKEND', 'smtp'):
            with patch.object(email_service.config, 'SMTP_HOST', 'smtp.test.com'):
                with patch.object(email_service.config, 'SMTP_USER', 'testuser'):
                    with patch.object(email_service.config, 'SMTP_PASSWORD', 'testpass'):
                        with patch.object(aiosmtplib, 'send', new_callable=AsyncMock) as mock_send:
                            mock_send.return_value = ({}, "OK")
                            await email_service.send_email(
                                to="test@example.com",
                                subject="Test",
                                body="Body"
                            )
                            # Check credentials were passed
                            call_kwargs = mock_send.call_args[1]
                            assert call_kwargs.get('username') == 'testuser'
                            assert call_kwargs.get('password') == 'testpass'

    @pytest.mark.asyncio
    async def test_smtp_backend_handles_connection_error(self):
        """Test SMTP backend handles connection errors gracefully."""
        import email_service
        import aiosmtplib

        with patch.object(email_service.config, 'EMAIL_BACKEND', 'smtp'):
            with patch.object(aiosmtplib, 'send', new_callable=AsyncMock) as mock_send:
                mock_send.side_effect = Exception("Connection refused")
                result = await email_service.send_email(
                    to="test@example.com",
                    subject="Test",
                    body="Body"
                )
                assert result is False


class TestEmailTemplates:
    """Test email template rendering."""

    def test_render_password_reset_template(self):
        """Test password reset email template renders correctly."""
        from email_service import render_email_template

        html = render_email_template(
            "password_reset",
            reset_url="https://example.com/reset/abc123",
            callsign="W1ABC"
        )

        assert "W1ABC" in html
        assert "https://example.com/reset/abc123" in html
        assert "reset" in html.lower()

    def test_render_welcome_template(self):
        """Test welcome email template renders correctly."""
        from email_service import render_email_template

        html = render_email_template(
            "welcome",
            callsign="W1ABC"
        )

        assert "W1ABC" in html
        assert "welcome" in html.lower()

    def test_render_template_escapes_html(self):
        """Test template escapes HTML in user input."""
        from email_service import render_email_template

        html = render_email_template(
            "welcome",
            callsign="<script>alert('xss')</script>"
        )

        assert "<script>" not in html
        assert "&lt;script&gt;" in html or "script" not in html


class TestPasswordResetTokens:
    """Test password reset token management."""

    def test_create_reset_token(self):
        """Test creating a password reset token."""
        from auth import register_user
        from email_service import create_password_reset_token

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        assert token is not None
        assert len(token) > 20

    def test_create_reset_token_stores_in_database(self):
        """Test reset token is stored in database."""
        from auth import register_user
        from email_service import create_password_reset_token
        from database import get_db

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM password_reset_tokens WHERE token = ?",
                (token,)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["callsign"] == "W1ABC"

    def test_validate_reset_token_valid(self):
        """Test validating a valid reset token."""
        from auth import register_user
        from email_service import create_password_reset_token, validate_reset_token

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        callsign = validate_reset_token(token)
        assert callsign == "W1ABC"

    def test_validate_reset_token_invalid(self):
        """Test validating an invalid reset token."""
        from email_service import validate_reset_token

        callsign = validate_reset_token("invalid-token-12345")
        assert callsign is None

    def test_validate_reset_token_expired(self):
        """Test validating an expired reset token."""
        from auth import register_user
        from email_service import create_password_reset_token, validate_reset_token
        from database import get_db
        from datetime import datetime, timedelta

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        # Expire the token manually
        with get_db() as conn:
            past = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            conn.execute(
                "UPDATE password_reset_tokens SET expires_at = ? WHERE token = ?",
                (past, token)
            )

        callsign = validate_reset_token(token)
        assert callsign is None

    def test_validate_reset_token_already_used(self):
        """Test validating an already-used reset token."""
        from auth import register_user
        from email_service import create_password_reset_token, validate_reset_token, mark_token_used

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        # Mark as used
        mark_token_used(token)

        callsign = validate_reset_token(token)
        assert callsign is None

    def test_reset_token_expires_after_1_hour(self):
        """Test reset tokens expire after 1 hour by default."""
        from auth import register_user
        from email_service import create_password_reset_token
        from database import get_db
        from datetime import datetime, timedelta

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT expires_at FROM password_reset_tokens WHERE token = ?",
                (token,)
            )
            row = cursor.fetchone()
            expires_at = datetime.fromisoformat(row["expires_at"])
            now = datetime.utcnow()

            # Should expire roughly 1 hour from now (allow 5 min tolerance)
            diff = expires_at - now
            assert timedelta(minutes=55) < diff < timedelta(minutes=65)


class TestPasswordResetEndpoints:
    """Test password reset API endpoints."""

    def test_forgot_password_page_exists(self, client):
        """Test forgot password page is accessible."""
        response = client.get("/forgot-password")
        assert response.status_code == 200
        assert "forgot" in response.text.lower() or "reset" in response.text.lower()

    def test_forgot_password_requires_email_or_callsign(self, client):
        """Test forgot password requires callsign."""
        response = client.post("/forgot-password", json={})
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_forgot_password_sends_email(self, client):
        """Test forgot password sends reset email."""
        from auth import register_user

        register_user("W1RST", "password123", "test@example.com")

        with patch('main.send_password_reset_email', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            response = client.post("/forgot-password", json={
                "callsign": "W1RST"
            })

            # Should succeed and send email
            assert response.status_code in [200, 303]
            mock_send.assert_called_once()

    def test_forgot_password_no_email_on_file(self, client):
        """Test forgot password handles users without email."""
        from auth import register_user

        register_user("W1NOE", "password123")  # No email

        response = client.post("/forgot-password", json={
            "callsign": "W1NOE"
        })

        # Should return error about no email
        assert response.status_code in [400, 200]

    def test_reset_password_page_with_valid_token(self, client):
        """Test reset password page with valid token."""
        from auth import register_user
        from email_service import create_password_reset_token

        register_user("W1ABC", "password123", "test@example.com")
        token = create_password_reset_token("W1ABC")

        response = client.get(f"/reset-password/{token}")
        assert response.status_code == 200
        assert "password" in response.text.lower()

    def test_reset_password_page_with_invalid_token(self, client):
        """Test reset password page with invalid token."""
        response = client.get("/reset-password/invalid-token-12345")
        assert response.status_code in [400, 404]

    def test_reset_password_changes_password(self, client):
        """Test reset password actually changes the password."""
        from auth import register_user, authenticate_user
        from email_service import create_password_reset_token

        register_user("W1CHG", "oldpassword", "test@example.com")
        token = create_password_reset_token("W1CHG")

        response = client.post(f"/reset-password/{token}", data={
            "password": "newpassword123",
            "confirm_password": "newpassword123"
        })

        assert response.status_code in [200, 303]

        # Old password should fail
        assert authenticate_user("W1CHG", "oldpassword") is None
        # New password should work
        assert authenticate_user("W1CHG", "newpassword123") is not None

    def test_reset_password_requires_matching_passwords(self, client):
        """Test reset password requires matching passwords."""
        from auth import register_user
        from email_service import create_password_reset_token

        register_user("W1MAT", "password123", "test@example.com")
        token = create_password_reset_token("W1MAT")

        response = client.post(f"/reset-password/{token}", data={
            "password": "newpassword123",
            "confirm_password": "differentpassword"
        })

        assert response.status_code in [400, 422]

    def test_reset_password_invalidates_token(self, client):
        """Test reset password invalidates token after use."""
        from auth import register_user
        from email_service import create_password_reset_token, validate_reset_token

        register_user("W1INV", "password123", "test@example.com")
        token = create_password_reset_token("W1INV")

        # Use the token
        client.post(f"/reset-password/{token}", data={
            "password": "newpassword123",
            "confirm_password": "newpassword123"
        })

        # Token should now be invalid
        assert validate_reset_token(token) is None


@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)

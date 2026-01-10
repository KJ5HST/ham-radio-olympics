"""
Tests for centralized configuration.
"""

import os
import pytest


class TestConfigDefaults:
    """Test config default values."""

    def test_session_duration_default(self):
        """Test default session duration is 30 days."""
        from config import Config
        assert Config.SESSION_DURATION_DAYS == 30

    def test_session_cookie_name(self):
        """Test session cookie name."""
        from config import Config
        assert Config.SESSION_COOKIE_NAME == "hro_session"

    def test_csrf_cookie_name(self):
        """Test CSRF cookie name."""
        from config import Config
        assert Config.CSRF_COOKIE_NAME == "hro_csrf"

    def test_bcrypt_rounds_default(self):
        """Test default bcrypt rounds is 12."""
        from config import Config
        assert Config.BCRYPT_ROUNDS == 12

    def test_lockout_attempts_default(self):
        """Test default lockout attempts is 5."""
        from config import Config
        assert Config.LOCKOUT_ATTEMPTS == 5

    def test_lockout_duration_default(self):
        """Test default lockout duration is 15 minutes."""
        from config import Config
        assert Config.LOCKOUT_DURATION_MINUTES == 15

    def test_password_min_length_default(self):
        """Test default password minimum length is 8."""
        from config import Config
        assert Config.PASSWORD_MIN_LENGTH == 8

    def test_sync_interval_default(self):
        """Test default sync interval is 3600 seconds."""
        from config import Config
        assert Config.SYNC_INTERVAL_SECONDS == 3600

    def test_max_qsos_display_default(self):
        """Test default max QSOs display is 50."""
        from config import Config
        assert Config.MAX_QSOS_DISPLAY == 50

    def test_max_medals_display_default(self):
        """Test default max medals display is 10."""
        from config import Config
        assert Config.MAX_MEDALS_DISPLAY == 10

    def test_medal_points_defaults(self):
        """Test default medal point values."""
        from config import Config
        assert Config.MEDAL_POINTS["gold"] == 3
        assert Config.MEDAL_POINTS["silver"] == 2
        assert Config.MEDAL_POINTS["bronze"] == 1

    def test_email_backend_default(self):
        """Test default email backend is console."""
        from config import Config
        assert Config.EMAIL_BACKEND == "console"

    def test_smtp_port_default(self):
        """Test default SMTP port is 587."""
        from config import Config
        assert Config.SMTP_PORT == 587


class TestConfigEnvironmentOverrides:
    """Test config values can be overridden via environment."""

    def test_lockout_attempts_from_env(self):
        """Test LOCKOUT_ATTEMPTS can be set from environment."""
        # Save original
        original = os.environ.get("LOCKOUT_ATTEMPTS")
        try:
            os.environ["LOCKOUT_ATTEMPTS"] = "10"
            # Need to reload config to pick up new env var
            import importlib
            import config
            importlib.reload(config)
            assert config.Config.LOCKOUT_ATTEMPTS == 10
        finally:
            # Restore original
            if original:
                os.environ["LOCKOUT_ATTEMPTS"] = original
            else:
                os.environ.pop("LOCKOUT_ATTEMPTS", None)
            import importlib
            import config
            importlib.reload(config)

    def test_lockout_duration_from_env(self):
        """Test LOCKOUT_DURATION_MINUTES can be set from environment."""
        original = os.environ.get("LOCKOUT_DURATION_MINUTES")
        try:
            os.environ["LOCKOUT_DURATION_MINUTES"] = "30"
            import importlib
            import config
            importlib.reload(config)
            assert config.Config.LOCKOUT_DURATION_MINUTES == 30
        finally:
            if original:
                os.environ["LOCKOUT_DURATION_MINUTES"] = original
            else:
                os.environ.pop("LOCKOUT_DURATION_MINUTES", None)
            import importlib
            import config
            importlib.reload(config)

    def test_sync_interval_from_env(self):
        """Test SYNC_INTERVAL can be set from environment."""
        original = os.environ.get("SYNC_INTERVAL")
        try:
            os.environ["SYNC_INTERVAL"] = "7200"
            import importlib
            import config
            importlib.reload(config)
            assert config.Config.SYNC_INTERVAL_SECONDS == 7200
        finally:
            if original:
                os.environ["SYNC_INTERVAL"] = original
            else:
                os.environ.pop("SYNC_INTERVAL", None)
            import importlib
            import config
            importlib.reload(config)


class TestConfigInstance:
    """Test the global config instance."""

    def test_config_instance_exists(self):
        """Test global config instance is available."""
        from config import config
        assert config is not None

    def test_config_instance_has_attributes(self):
        """Test config instance has expected attributes."""
        from config import config
        assert hasattr(config, 'SESSION_DURATION_DAYS')
        assert hasattr(config, 'LOCKOUT_ATTEMPTS')
        assert hasattr(config, 'BCRYPT_ROUNDS')
        assert hasattr(config, 'TESTING')

    def test_testing_flag_in_test_mode(self):
        """Test TESTING flag is True when TESTING env var is set."""
        from config import config
        # TESTING env var is set by test runner
        assert config.TESTING is True

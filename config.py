"""
Centralized configuration for Ham Radio Olympics.

All configurable values are loaded from environment variables with sensible defaults.
"""

import os


def _require_env(name: str, test_default: str) -> str:
    """
    Get a required environment variable.

    In testing mode, returns a test default. In production, raises an error if not set.
    """
    value = os.getenv(name)
    if value:
        return value

    # Allow test defaults only in testing mode
    if os.getenv("TESTING"):
        return test_default

    raise ValueError(
        f"Required environment variable {name} is not set. "
        f"Set {name} in your environment or Fly.io secrets."
    )


class Config:
    """Application configuration loaded from environment variables."""

    # Site theming
    SITE_THEME: str = os.getenv("SITE_THEME", "olympics")  # olympics, coolcontest
    SITE_NAME: str = os.getenv("SITE_NAME", "Ham Radio Olympics")
    SITE_TAGLINE: str = os.getenv("SITE_TAGLINE", "Compete. Connect. Conquer the Airwaves.")

    # Base URL for the app (used in email links, etc.)
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8080")

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "ham_olympics.db")

    # Session settings
    SESSION_DURATION_DAYS: int = int(os.getenv("SESSION_DURATION_DAYS", "30"))
    SESSION_COOKIE_NAME: str = "hro_session"
    CSRF_COOKIE_NAME: str = "hro_csrf"
    # Set to False for local development (HTTP), True in production (HTTPS)
    # Defaults to False in testing mode (TestClient uses HTTP)
    SECURE_COOKIES: bool = os.getenv(
        "SECURE_COOKIES",
        "false" if os.getenv("TESTING") else "true"
    ).lower() in ("true", "1", "yes")

    # Security - these are required in production
    ADMIN_KEY: str = _require_env("ADMIN_KEY", "test-admin-key")
    ENCRYPTION_KEY: str = _require_env("ENCRYPTION_KEY", "test-encryption-key")
    # Salt for key derivation - required in production, unique per deployment
    ENCRYPTION_SALT: str = _require_env("ENCRYPTION_SALT", "test-encryption-salt")
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))

    # Account lockout
    LOCKOUT_ATTEMPTS: int = int(os.getenv("LOCKOUT_ATTEMPTS", "5"))
    LOCKOUT_DURATION_MINUTES: int = int(os.getenv("LOCKOUT_DURATION_MINUTES", "15"))

    # Password requirements
    PASSWORD_MIN_LENGTH: int = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))

    # Rate limiting
    RATE_LIMIT_SIGNUP: str = os.getenv("RATE_LIMIT_SIGNUP", "5/minute")
    RATE_LIMIT_LOGIN: str = os.getenv("RATE_LIMIT_LOGIN", "10/minute")

    # Sync settings
    SYNC_INTERVAL_SECONDS: int = int(os.getenv("SYNC_INTERVAL", "3600"))

    # Display limits
    MAX_QSOS_DISPLAY: int = int(os.getenv("MAX_QSOS_DISPLAY", "50"))
    MAX_MEDALS_DISPLAY: int = int(os.getenv("MAX_MEDALS_DISPLAY", "10"))

    # Scoring
    MEDAL_POINTS: dict = {
        "gold": int(os.getenv("MEDAL_POINTS_GOLD", "3")),
        "silver": int(os.getenv("MEDAL_POINTS_SILVER", "2")),
        "bronze": int(os.getenv("MEDAL_POINTS_BRONZE", "1")),
    }

    # QRZ XML API credentials (for callsign lookups)
    QRZ_USERNAME: str = os.getenv("QRZ_USERNAME", "")
    QRZ_PASSWORD: str = os.getenv("QRZ_PASSWORD", "")

    # Email settings
    EMAIL_BACKEND: str = os.getenv("EMAIL_BACKEND", "console")  # console, smtp, or resend
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "noreply@hamradio-olympics.com")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "")  # Email for error notifications

    # File uploads (resources)
    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        os.path.join(os.path.dirname(os.getenv("DATABASE_PATH", "ham_olympics.db")), "uploads", "resources")
    )
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 10 MB
    ALLOWED_UPLOAD_EXTENSIONS: set = {
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".txt", ".md", ".csv", ".adi", ".adif",
        ".doc", ".docx", ".xls", ".xlsx", ".zip"
    }

    # Testing mode
    TESTING: bool = bool(os.getenv("TESTING", ""))


# Global config instance
config = Config()

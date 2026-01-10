"""
Centralized configuration for Ham Radio Olympics.

All configurable values are loaded from environment variables with sensible defaults.
"""

import os


class Config:
    """Application configuration loaded from environment variables."""

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "ham_olympics.db")

    # Session settings
    SESSION_DURATION_DAYS: int = int(os.getenv("SESSION_DURATION_DAYS", "30"))
    SESSION_COOKIE_NAME: str = "hro_session"
    CSRF_COOKIE_NAME: str = "hro_csrf"

    # Security
    ADMIN_KEY: str = os.getenv("ADMIN_KEY", "admin-secret-change-me")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")
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

    # Email settings (for future use)
    EMAIL_BACKEND: str = os.getenv("EMAIL_BACKEND", "console")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "noreply@hamradio-olympics.com")

    # Testing mode
    TESTING: bool = bool(os.getenv("TESTING", ""))


# Global config instance
config = Config()

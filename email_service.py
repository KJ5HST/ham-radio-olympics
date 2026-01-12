"""
Email service for Ham Radio Olympics.

Supports multiple backends:
- console: Print emails to stdout (for development/testing)
- smtp: Send via SMTP server
"""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional
from html import escape

from config import config
from database import get_db

logger = logging.getLogger(__name__)

# Token expiry time
RESET_TOKEN_EXPIRY_HOURS = 1


async def send_email(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """
    Send an email using the configured backend.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Plain text body
        html_body: Optional HTML body

    Returns:
        True if email was sent successfully, False otherwise
    """
    backend = _get_backend()
    return await backend(to, subject, body, html_body)


def _get_backend():
    """Get the email backend function based on config."""
    backend_name = config.EMAIL_BACKEND

    if backend_name == "console":
        return _console_backend
    elif backend_name == "smtp":
        return _smtp_backend
    elif backend_name == "resend":
        return _resend_backend
    else:
        logger.warning(f"Unknown email backend '{backend_name}', using console")
        return _console_backend


async def _console_backend(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """Console backend - prints email to stdout."""
    print("=" * 60)
    print("EMAIL (Console Backend)")
    print("=" * 60)
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print("-" * 60)
    print(body)
    if html_body:
        print("-" * 60)
        print("HTML Body:")
        print(html_body)
    print("=" * 60)
    return True


async def _smtp_backend(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """SMTP backend - sends email via SMTP server."""
    try:
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Create message
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
        else:
            msg = MIMEText(body, "plain")

        msg["Subject"] = subject
        msg["From"] = config.EMAIL_FROM
        msg["To"] = to

        # Send via SMTP
        await aiosmtplib.send(
            msg,
            hostname=config.SMTP_HOST,
            port=config.SMTP_PORT,
            username=config.SMTP_USER if config.SMTP_USER else None,
            password=config.SMTP_PASSWORD if config.SMTP_PASSWORD else None,
            start_tls=True
        )

        logger.info(f"Email sent to {to}: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {to}: {e}")
        return False


async def _resend_backend(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """Resend backend - sends email via Resend API."""
    try:
        import resend

        resend.api_key = config.RESEND_API_KEY

        params = {
            "from": config.EMAIL_FROM,
            "to": [to],
            "subject": subject,
            "text": body,
        }
        if html_body:
            params["html"] = html_body

        resend.Emails.send(params)

        logger.info(f"Email sent to {to}: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {to}: {e}")
        return False


def render_email_template(template_name: str, **kwargs) -> str:
    """
    Render an email template with the given variables.

    Args:
        template_name: Name of the template (e.g., "password_reset", "welcome")
        **kwargs: Variables to pass to the template

    Returns:
        Rendered HTML string
    """
    # Escape all string values to prevent XSS
    safe_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            safe_kwargs[key] = escape(value)
        else:
            safe_kwargs[key] = value

    if template_name == "password_reset":
        return _render_password_reset_template(**safe_kwargs)
    elif template_name == "welcome":
        return _render_welcome_template(**safe_kwargs)
    elif template_name == "medal_notification":
        return _render_medal_notification_template(**safe_kwargs)
    elif template_name == "match_reminder":
        return _render_match_reminder_template(**safe_kwargs)
    elif template_name == "email_verification":
        return _render_email_verification_template(**safe_kwargs)
    else:
        raise ValueError(f"Unknown template: {template_name}")


def _render_password_reset_template(reset_url: str, callsign: str) -> str:
    """Render password reset email template."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Password Reset - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Password Reset Request</h1>
    <p>Hello {callsign},</p>
    <p>We received a request to reset your password for your Ham Radio Olympics account.</p>
    <p>Click the button below to reset your password:</p>
    <p style="text-align: center; margin: 30px 0;">
        <a href="{reset_url}" style="background-color: #4299e1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Reset Password
        </a>
    </p>
    <p>Or copy and paste this link into your browser:</p>
    <p style="word-break: break-all; color: #4299e1;">{reset_url}</p>
    <p>This link will expire in 1 hour.</p>
    <p>If you didn't request this reset, you can safely ignore this email.</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">Ham Radio Olympics</p>
</body>
</html>
"""


def _render_welcome_template(callsign: str) -> str:
    """Render welcome email template."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Welcome to Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Welcome to Ham Radio Olympics!</h1>
    <p>Hello {callsign},</p>
    <p>Welcome to the Ham Radio Olympics! Your account has been successfully created.</p>
    <p>You can now:</p>
    <ul>
        <li>Connect your QRZ.com or LoTW account to sync your QSOs</li>
        <li>Enter sports and compete for medals</li>
        <li>Track your progress on the leaderboard</li>
    </ul>
    <p>Good luck and 73!</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">Ham Radio Olympics</p>
</body>
</html>
"""


def create_password_reset_token(callsign: str) -> str:
    """
    Create a password reset token for a user.

    Args:
        callsign: The user's callsign

    Returns:
        The generated token
    """
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(hours=RESET_TOKEN_EXPIRY_HOURS)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO password_reset_tokens (token, callsign, created_at, expires_at, used)
            VALUES (?, ?, ?, ?, 0)
        """, (token, callsign.upper(), now.isoformat(), expires.isoformat()))

    logger.info(f"Created password reset token for {callsign}")
    return token


def validate_reset_token(token: str) -> Optional[str]:
    """
    Validate a password reset token.

    Args:
        token: The token to validate

    Returns:
        The callsign if valid, None otherwise
    """
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT callsign, expires_at, used
            FROM password_reset_tokens
            WHERE token = ?
        """, (token,))
        row = cursor.fetchone()

        if not row:
            return None

        # Check if used
        if row["used"]:
            return None

        # Check if expired
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires_at:
            return None

        return row["callsign"]


def mark_token_used(token: str) -> None:
    """Mark a password reset token as used."""
    with get_db() as conn:
        conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE token = ?",
            (token,)
        )


async def send_password_reset_email(callsign: str, email: str, reset_url: str) -> bool:
    """
    Send a password reset email.

    Args:
        callsign: User's callsign
        email: User's email address
        reset_url: The password reset URL

    Returns:
        True if email sent successfully
    """
    html_body = render_email_template("password_reset", reset_url=reset_url, callsign=callsign)
    plain_body = f"""
Password Reset Request

Hello {callsign},

We received a request to reset your password for your Ham Radio Olympics account.

Click this link to reset your password:
{reset_url}

This link will expire in 1 hour.

If you didn't request this reset, you can safely ignore this email.

Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject="Password Reset - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )


def _render_medal_notification_template(
    callsign: str,
    sport_name: str,
    match_name: str,
    medal_type: str,
    competition: str,
    points: int
) -> str:
    """Render medal notification email template."""
    medal_colors = {
        "gold": "#FFD700",
        "silver": "#C0C0C0",
        "bronze": "#CD7F32"
    }
    medal_color = medal_colors.get(medal_type.lower(), "#4299e1")
    medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}.get(medal_type.lower(), "🏅")

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medal Earned - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Congratulations! {medal_emoji}</h1>
    <p>Hello {callsign},</p>
    <p>You've earned a medal in the Ham Radio Olympics!</p>
    <div style="background-color: #f7fafc; border-left: 4px solid {medal_color}; padding: 20px; margin: 20px 0;">
        <p style="margin: 0; font-size: 18px;"><strong>{medal_type.title()} Medal</strong></p>
        <p style="margin: 10px 0 0 0; color: #4a5568;">
            {competition} Competition<br>
            {sport_name} - {match_name}
        </p>
        <p style="margin: 10px 0 0 0; color: #2d3748; font-size: 16px;">
            <strong>+{points} points</strong>
        </p>
    </div>
    <p>Keep up the great work!</p>
    <p>73,<br>Ham Radio Olympics</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">Ham Radio Olympics</p>
</body>
</html>
"""


def _render_match_reminder_template(
    callsign: str,
    sport_name: str,
    match_name: str,
    target_value: str,
    start_date: str,
    end_date: str
) -> str:
    """Render match reminder email template."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Match Starting Soon - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Match Starting Soon! 📻</h1>
    <p>Hello {callsign},</p>
    <p>A new match is about to begin in the Ham Radio Olympics!</p>
    <div style="background-color: #ebf8ff; border-left: 4px solid #4299e1; padding: 20px; margin: 20px 0;">
        <p style="margin: 0; font-size: 18px;"><strong>{sport_name}</strong></p>
        <p style="margin: 10px 0 0 0; color: #4a5568;">
            Match: {match_name}<br>
            Target: {target_value}
        </p>
        <p style="margin: 10px 0 0 0; color: #2d3748;">
            <strong>{start_date} - {end_date}</strong>
        </p>
    </div>
    <p>Get your rig ready and good luck!</p>
    <p>73,<br>Ham Radio Olympics</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">Ham Radio Olympics</p>
</body>
</html>
"""


def _render_email_verification_template(callsign: str, verification_url: str) -> str:
    """Render email verification template."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Verify Your Email - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Verify Your Email Address</h1>
    <p>Hello {callsign},</p>
    <p>Please verify your email address by clicking the button below:</p>
    <p style="text-align: center; margin: 30px 0;">
        <a href="{verification_url}" style="background-color: #48bb78; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Verify Email
        </a>
    </p>
    <p>Or copy and paste this link into your browser:</p>
    <p style="word-break: break-all; color: #4299e1;">{verification_url}</p>
    <p>This link will expire in 24 hours.</p>
    <p>If you didn't create an account, you can safely ignore this email.</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">Ham Radio Olympics</p>
</body>
</html>
"""


async def send_welcome_email(callsign: str, email: str) -> bool:
    """
    Send a welcome email to a new user.

    Args:
        callsign: User's callsign
        email: User's email address

    Returns:
        True if email sent successfully
    """
    html_body = render_email_template("welcome", callsign=callsign)
    plain_body = f"""
Welcome to Ham Radio Olympics!

Hello {callsign},

Welcome to the Ham Radio Olympics! Your account has been successfully created.

You can now:
- Connect your QRZ.com or LoTW account to sync your QSOs
- Enter sports and compete for medals
- Track your progress on the leaderboard

Good luck and 73!

Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject="Welcome to Ham Radio Olympics!",
        body=plain_body.strip(),
        html_body=html_body
    )


async def send_medal_notification_email(
    callsign: str,
    email: str,
    sport_name: str,
    match_name: str,
    medal_type: str,
    competition: str,
    points: int
) -> bool:
    """
    Send a medal notification email.

    Args:
        callsign: User's callsign
        email: User's email address
        sport_name: Name of the sport
        match_name: Name of the match
        medal_type: gold, silver, or bronze
        competition: Distance or Cool Factor
        points: Points earned

    Returns:
        True if email sent successfully
    """
    medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}.get(medal_type.lower(), "🏅")

    html_body = render_email_template(
        "medal_notification",
        callsign=callsign,
        sport_name=sport_name,
        match_name=match_name,
        medal_type=medal_type,
        competition=competition,
        points=points
    )
    plain_body = f"""
Congratulations! {medal_emoji}

Hello {callsign},

You've earned a medal in the Ham Radio Olympics!

{medal_type.title()} Medal - {competition} Competition
{sport_name} - {match_name}
+{points} points

Keep up the great work!

73,
Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject=f"{medal_emoji} You earned a {medal_type} medal! - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )


async def send_match_reminder_email(
    callsign: str,
    email: str,
    sport_name: str,
    match_name: str,
    target_value: str,
    start_date: str,
    end_date: str
) -> bool:
    """
    Send a match reminder email.

    Args:
        callsign: User's callsign
        email: User's email address
        sport_name: Name of the sport
        match_name: Name of the match
        target_value: Target for the match
        start_date: Match start date
        end_date: Match end date

    Returns:
        True if email sent successfully
    """
    html_body = render_email_template(
        "match_reminder",
        callsign=callsign,
        sport_name=sport_name,
        match_name=match_name,
        target_value=target_value,
        start_date=start_date,
        end_date=end_date
    )
    plain_body = f"""
Match Starting Soon!

Hello {callsign},

A new match is about to begin in the Ham Radio Olympics!

{sport_name}
Match: {match_name}
Target: {target_value}
{start_date} - {end_date}

Get your rig ready and good luck!

73,
Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject=f"Match Starting: {sport_name} - {match_name}",
        body=plain_body.strip(),
        html_body=html_body
    )


# Email verification token expiry
VERIFICATION_TOKEN_EXPIRY_HOURS = 24


def create_email_verification_token(callsign: str) -> str:
    """
    Create an email verification token for a user.

    Args:
        callsign: The user's callsign

    Returns:
        The generated token
    """
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(hours=VERIFICATION_TOKEN_EXPIRY_HOURS)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO email_verification_tokens (token, callsign, created_at, expires_at, used)
            VALUES (?, ?, ?, ?, 0)
        """, (token, callsign.upper(), now.isoformat(), expires.isoformat()))

    logger.info(f"Created email verification token for {callsign}")
    return token


def validate_email_verification_token(token: str) -> Optional[str]:
    """
    Validate an email verification token.

    Args:
        token: The token to validate

    Returns:
        The callsign if valid, None otherwise
    """
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT callsign, expires_at, used
            FROM email_verification_tokens
            WHERE token = ?
        """, (token,))
        row = cursor.fetchone()

        if not row:
            return None

        if row["used"]:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires_at:
            return None

        return row["callsign"]


def mark_email_verification_token_used(token: str) -> None:
    """Mark an email verification token as used."""
    with get_db() as conn:
        conn.execute(
            "UPDATE email_verification_tokens SET used = 1 WHERE token = ?",
            (token,)
        )


async def send_email_verification(callsign: str, email: str, verification_url: str) -> bool:
    """
    Send an email verification email.

    Args:
        callsign: User's callsign
        email: User's email address
        verification_url: The verification URL

    Returns:
        True if email sent successfully
    """
    html_body = render_email_template(
        "email_verification",
        callsign=callsign,
        verification_url=verification_url
    )
    plain_body = f"""
Verify Your Email Address

Hello {callsign},

Please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

If you didn't create an account, you can safely ignore this email.

Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject="Verify Your Email - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )

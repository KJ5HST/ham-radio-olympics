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

# Base URL for the app (used in email links)
APP_BASE_URL = config.APP_BASE_URL


def _get_email_footer_html() -> str:
    """Get the HTML footer with unsubscribe link for all emails."""
    return f"""
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 30px 0;">
    <p style="color: #718096; font-size: 12px;">
        Ham Radio Olympics<br>
        <a href="{APP_BASE_URL}/settings#notifications" style="color: #718096;">Manage email preferences</a> |
        <a href="{APP_BASE_URL}/settings#notifications" style="color: #718096;">Unsubscribe</a>
    </p>
"""


def _get_email_footer_text() -> str:
    """Get the plain text footer with unsubscribe info for all emails."""
    return f"""
---
Ham Radio Olympics
Manage email preferences: {APP_BASE_URL}/settings#notifications
To unsubscribe, visit your settings page and disable email notifications.
"""


async def send_email(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    skip_bcc: bool = False
) -> bool:
    """
    Send an email using the configured backend.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Plain text body
        html_body: Optional HTML body
        skip_bcc: If True, don't BCC admin even if enabled (for admin error emails)

    Returns:
        True if email was sent successfully, False otherwise
    """
    # Check if admin BCC is enabled
    bcc = None
    if not skip_bcc:
        try:
            from database import get_setting
            if get_setting("admin_bcc_emails") == "1":
                # Use ADMIN_EMAIL env var, or fall back to first admin user's email
                admin_email = config.ADMIN_EMAIL
                if not admin_email:
                    admin_email = _get_admin_email_fallback()
                if admin_email and admin_email != to:
                    bcc = admin_email
        except Exception:
            pass  # Don't fail email send if we can't check setting

    backend = _get_backend()
    return await backend(to, subject, body, html_body, bcc)


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


def _get_admin_email_fallback() -> Optional[str]:
    """Get the email address of the first admin user as a BCC fallback."""
    try:
        row = get_db().execute(
            "SELECT email FROM competitors WHERE is_admin = 1 AND email IS NOT NULL "
            "AND email != '' AND email_verified = 1 LIMIT 1"
        ).fetchone()
        return row["email"] if row else None
    except Exception:
        return None


async def _console_backend(
    to: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    bcc: Optional[str] = None
) -> bool:
    """Console backend - prints email to stdout."""
    print("=" * 60)
    print("EMAIL (Console Backend)")
    print("=" * 60)
    print(f"To: {to}")
    if bcc:
        print(f"Bcc: {bcc}")
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
    html_body: Optional[str] = None,
    bcc: Optional[str] = None
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
        if bcc:
            msg["Bcc"] = bcc

        # Build recipient list
        recipients = [to]
        if bcc:
            recipients.append(bcc)

        # Send via SMTP
        await aiosmtplib.send(
            msg,
            hostname=config.SMTP_HOST,
            port=config.SMTP_PORT,
            username=config.SMTP_USER if config.SMTP_USER else None,
            password=config.SMTP_PASSWORD if config.SMTP_PASSWORD else None,
            start_tls=True,
            recipients=recipients
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
    html_body: Optional[str] = None,
    bcc: Optional[str] = None
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
        if bcc:
            params["bcc"] = [bcc]

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
    elif template_name == "record_notification":
        return _render_record_notification_template(**safe_kwargs)
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
    {_get_email_footer_html()}
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
    {_get_email_footer_html()}
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
    """Render medal notification email template (single medal - legacy)."""
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
    {_get_email_footer_html()}
</body>
</html>
"""


def _render_medals_summary_template(callsign: str, medals: list, total_points: int) -> str:
    """Render consolidated medals notification email template."""
    medal_colors = {
        "gold": "#FFD700",
        "silver": "#C0C0C0",
        "bronze": "#CD7F32"
    }
    medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}

    medals_html = ""
    for m in medals:
        color = medal_colors.get(m["medal_type"].lower(), "#4299e1")
        emoji = medal_emoji.get(m["medal_type"].lower(), "🏅")
        medals_html += f"""
        <div style="background-color: #f7fafc; border-left: 4px solid {color}; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 16px;"><strong>{emoji} {m["medal_type"].title()} Medal</strong></p>
            <p style="margin: 5px 0 0 0; color: #4a5568;">
                {m["competition"]} - {m["sport_name"]}<br>
                Target: {m["match_name"]}
            </p>
            <p style="margin: 5px 0 0 0; color: #2d3748;">
                <strong>+{m["points"]} points</strong>
            </p>
        </div>
        """

    medal_count = len(medals)
    title_text = "medal" if medal_count == 1 else "medals"

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medals Earned - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Congratulations! 🎉</h1>
    <p>Hello {callsign},</p>
    <p>You've earned <strong>{medal_count} {title_text}</strong> in the Ham Radio Olympics!</p>
    {medals_html}
    <div style="background-color: #ebf8ff; padding: 15px; margin: 20px 0; border-radius: 4px;">
        <p style="margin: 0; font-size: 18px; color: #2d3748;">
            <strong>Total from these medals: +{total_points} points</strong>
        </p>
    </div>
    <p>Keep up the great work!</p>
    <p>73,<br>Ham Radio Olympics</p>
    {_get_email_footer_html()}
</body>
</html>
"""


def _render_push_disabled_template(callsign: str, reason: str) -> str:
    """Render push notifications disabled email template."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Push Notifications Disabled - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Push Notifications Disabled</h1>
    <p>Hello {callsign},</p>
    <p>Your push notification subscription has been automatically disabled because we could no longer deliver notifications to your device.</p>
    <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0;">
        <p style="margin: 0; color: #92400e;">
            <strong>Why did this happen?</strong><br>
            {reason}
        </p>
    </div>
    <h3 style="color: #2d3748;">How to Re-enable Push Notifications</h3>
    <ol style="color: #4a5568; line-height: 1.8;">
        <li>Log in to Ham Radio Olympics</li>
        <li>Go to your <strong>Settings</strong> page</li>
        <li>Scroll to the <strong>Push Notifications</strong> section</li>
        <li>Click <strong>"Enable Push Notifications"</strong></li>
        <li>When your browser asks for permission, click <strong>Allow</strong></li>
    </ol>
    <p style="margin-top: 20px;">
        <a href="{APP_BASE_URL}/settings#notifications" style="background-color: #4299e1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Go to Settings
        </a>
    </p>
    <p style="color: #718096; margin-top: 20px;">
        If you no longer wish to receive push notifications, no action is needed.
    </p>
    <p>73,<br>Ham Radio Olympics</p>
    {_get_email_footer_html()}
</body>
</html>
"""


def _render_admin_error_template(error_type: str, error_details: str, context: dict) -> str:
    """Render admin error notification email template."""
    context_html = ""
    if context:
        context_items = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in context.items()])
        context_html = f"""
        <div style="background-color: #f7fafc; padding: 15px; margin: 10px 0; border-radius: 4px;">
            <p style="margin: 0 0 10px 0;"><strong>Context:</strong></p>
            <ul style="margin: 0; padding-left: 20px;">{context_items}</ul>
        </div>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Error Alert - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #c53030;">⚠️ Error Alert</h1>
    <p>An error occurred in the Ham Radio Olympics system.</p>
    <div style="background-color: #fed7d7; border-left: 4px solid #c53030; padding: 15px; margin: 20px 0;">
        <p style="margin: 0; font-size: 16px;"><strong>Error Type:</strong> {error_type}</p>
        <p style="margin: 10px 0 0 0; color: #4a5568;">
            {error_details}
        </p>
    </div>
    {context_html}
    <p style="color: #718096; font-size: 12px;">
        This is an automated error notification from Ham Radio Olympics.
    </p>
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
    {_get_email_footer_html()}
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
    {_get_email_footer_html()}
</body>
</html>
"""


def _render_record_notification_template(
    callsign: str,
    record_type: str,
    value: str,
    sport_name: str,
    previous_holder: str = None,
    previous_value: str = None
) -> str:
    """Render record notification email template."""
    record_labels = {
        "longest_distance": "Longest Distance",
        "highest_cool_factor": "Highest Cool Factor",
        "lowest_power": "Lowest Power DX"
    }
    record_label = record_labels.get(record_type, record_type)

    previous_section = ""
    if previous_holder and previous_value:
        previous_section = f"""
        <p style="margin: 10px 0 0 0; color: #718096; font-size: 14px;">
            Previous record: {previous_value} by {previous_holder}
        </p>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>New Record - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">New World Record!</h1>
    <p>Hello {callsign},</p>
    <p>Congratulations! You've set a new world record in the Ham Radio Olympics!</p>
    <div style="background-color: #faf5ff; border-left: 4px solid #9f7aea; padding: 20px; margin: 20px 0;">
        <p style="margin: 0; font-size: 18px;"><strong>{record_label}</strong></p>
        <p style="margin: 10px 0 0 0; color: #4a5568;">
            Sport: {sport_name}
        </p>
        <p style="margin: 10px 0 0 0; color: #2d3748; font-size: 24px;">
            <strong>{value}</strong>
        </p>
        {previous_section}
    </div>
    <p>Your achievement has been recorded in the Ham Radio Olympics history books!</p>
    <p>73,<br>Ham Radio Olympics</p>
    {_get_email_footer_html()}
</body>
</html>
"""


async def send_record_notification_email(
    callsign: str,
    email: str,
    record_type: str,
    value: str,
    sport_name: str,
    previous_holder: str = None,
    previous_value: str = None
) -> bool:
    """
    Send a record notification email.

    Args:
        callsign: User's callsign
        email: User's email address
        record_type: Type of record (longest_distance, highest_cool_factor, lowest_power)
        value: The record value
        sport_name: Name of the sport
        previous_holder: Previous record holder's callsign (optional)
        previous_value: Previous record value (optional)

    Returns:
        True if email sent successfully
    """
    record_labels = {
        "longest_distance": "Longest Distance",
        "highest_cool_factor": "Highest Cool Factor",
        "lowest_power": "Lowest Power DX"
    }
    record_label = record_labels.get(record_type, record_type)

    html_body = render_email_template(
        "record_notification",
        callsign=callsign,
        record_type=record_type,
        value=value,
        sport_name=sport_name,
        previous_holder=previous_holder,
        previous_value=previous_value
    )

    previous_text = ""
    if previous_holder and previous_value:
        previous_text = f"\nPrevious record: {previous_value} by {previous_holder}"

    plain_body = f"""
New World Record!

Hello {callsign},

Congratulations! You've set a new world record in the Ham Radio Olympics!

{record_label}
Sport: {sport_name}
Value: {value}{previous_text}

Your achievement has been recorded in the Ham Radio Olympics history books!

73,
Ham Radio Olympics
"""

    return await send_email(
        to=email,
        subject=f"New World Record! {record_label} - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )


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


async def send_push_disabled_email(
    callsign: str,
    email: str,
    reason: str = "Your browser or device reported that the notification subscription is no longer valid. This can happen if you cleared your browser data, disabled notifications in your browser settings, or if the subscription expired."
) -> bool:
    """
    Send an email notifying user their push subscription was disabled.

    Args:
        callsign: User's callsign
        email: User's email address
        reason: Explanation of why push was disabled

    Returns:
        True if email sent successfully
    """
    html_body = _render_push_disabled_template(callsign, reason)

    plain_body = f"""
Push Notifications Disabled

Hello {callsign},

Your push notification subscription has been automatically disabled because we could no longer deliver notifications to your device.

Why did this happen?
{reason}

How to Re-enable Push Notifications:
1. Log in to Ham Radio Olympics
2. Go to your Settings page
3. Scroll to the Push Notifications section
4. Click "Enable Push Notifications"
5. When your browser asks for permission, click Allow

Go to Settings: {APP_BASE_URL}/settings#notifications

If you no longer wish to receive push notifications, no action is needed.

73,
Ham Radio Olympics
{_get_email_footer_text()}
"""

    return await send_email(
        to=email,
        subject="Push Notifications Disabled - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )


async def send_admin_error_email(
    error_type: str,
    error_details: str,
    context: dict = None
) -> bool:
    """
    Send an error notification email to the admin.

    Args:
        error_type: Type/category of error (e.g., "Push Notification Failure")
        error_details: Detailed error message
        context: Optional dict with additional context

    Returns:
        True if email sent successfully
    """
    admin_email = config.ADMIN_EMAIL
    if not admin_email:
        logger.warning("ADMIN_EMAIL not configured, cannot send error notification")
        return False

    html_body = _render_admin_error_template(error_type, error_details, context or {})

    context_text = ""
    if context:
        context_text = "\n".join([f"  - {k}: {v}" for k, v in context.items()])
        context_text = f"\n\nContext:\n{context_text}"

    plain_body = f"""
Error Alert - Ham Radio Olympics

An error occurred in the Ham Radio Olympics system.

Error Type: {error_type}
Details: {error_details}{context_text}

This is an automated error notification.
"""

    return await send_email(
        to=admin_email,
        subject=f"⚠️ Error: {error_type} - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body,
        skip_bcc=True  # Don't BCC admin on admin emails
    )


async def send_medals_summary_email(
    callsign: str,
    email: str,
    medals: list,
    total_points: int
) -> bool:
    """
    Send a consolidated medals notification email.

    Args:
        callsign: User's callsign
        email: User's email address
        medals: List of medal dicts with keys: medal_type, competition, sport_name, match_name, points
        total_points: Total points from all medals

    Returns:
        True if email sent successfully
    """
    if not medals:
        return True

    medal_count = len(medals)
    medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}

    # Find the best medal for the subject line
    best_medal = None
    for medal_type in ["gold", "silver", "bronze"]:
        for m in medals:
            if m["medal_type"].lower() == medal_type:
                best_medal = medal_type
                break
        if best_medal:
            break

    emoji = medal_emoji.get(best_medal, "🏅") if best_medal else "🎉"

    html_body = _render_medals_summary_template(callsign, medals, total_points)

    # Build plain text version
    medals_text = ""
    for m in medals:
        e = medal_emoji.get(m["medal_type"].lower(), "🏅")
        medals_text += f"\n{e} {m['medal_type'].title()} Medal - {m['competition']}\n"
        medals_text += f"   {m['sport_name']} - {m['match_name']}\n"
        medals_text += f"   +{m['points']} points\n"

    title_text = "medal" if medal_count == 1 else "medals"
    plain_body = f"""
Congratulations! 🎉

Hello {callsign},

You've earned {medal_count} {title_text} in the Ham Radio Olympics!
{medals_text}
Total from these medals: +{total_points} points

Keep up the great work!

73,
Ham Radio Olympics
{_get_email_footer_text()}
"""

    if medal_count == 1:
        subject = f"{emoji} You earned a {best_medal} medal! - Ham Radio Olympics"
    else:
        subject = f"{emoji} You earned {medal_count} medals! - Ham Radio Olympics"

    return await send_email(
        to=email,
        subject=subject,
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


async def notify_new_medals() -> dict:
    """
    Send notifications for new medals that haven't been notified yet.

    Consolidates all medals per person into a single email to avoid spam.

    Returns:
        Dict with counts of notifications sent and any errors
    """
    from database import get_db

    results = {"sent": 0, "skipped": 0, "errors": 0, "medals_notified": 0}

    with get_db() as conn:
        # Find medals with actual medals (gold/silver/bronze) that haven't been notified
        # Only notify users who have email notifications enabled and a verified email
        cursor = conn.execute("""
            SELECT m.id, m.callsign, m.role, m.qso_race_medal, m.cool_factor_medal,
                   m.total_points, mat.target_value, s.name as sport_name,
                   c.email, c.email_notifications_enabled, c.email_verified,
                   COALESCE(c.email_medal_notifications, 1) as email_medal_notifications
            FROM medals m
            JOIN matches mat ON m.match_id = mat.id
            JOIN sports s ON mat.sport_id = s.id
            JOIN competitors c ON m.callsign = c.callsign
            WHERE m.notified_at IS NULL
              AND (m.qso_race_medal IS NOT NULL OR m.cool_factor_medal IS NOT NULL)
            ORDER BY m.callsign, s.name, mat.target_value
        """)
        all_medals = cursor.fetchall()

        # Group medals by callsign
        medals_by_person = {}
        medal_ids_by_person = {}
        for medal in all_medals:
            medal = dict(medal)
            callsign = medal["callsign"]

            if callsign not in medals_by_person:
                medals_by_person[callsign] = {
                    "email": medal["email"],
                    "email_verified": medal["email_verified"],
                    "email_notifications_enabled": medal["email_notifications_enabled"],
                    "email_medal_notifications": medal["email_medal_notifications"],
                    "medals": []
                }
                medal_ids_by_person[callsign] = []

            medal_ids_by_person[callsign].append(medal["id"])

            # Build medal entries for this person
            if medal["qso_race_medal"]:
                points = {"gold": 3, "silver": 2, "bronze": 1}.get(medal["qso_race_medal"], 0)
                medals_by_person[callsign]["medals"].append({
                    "medal_type": medal["qso_race_medal"],
                    "competition": "QSO Race",
                    "sport_name": medal["sport_name"],
                    "match_name": medal["target_value"],
                    "points": points
                })

            if medal["cool_factor_medal"]:
                points = {"gold": 3, "silver": 2, "bronze": 1}.get(medal["cool_factor_medal"], 0)
                medals_by_person[callsign]["medals"].append({
                    "medal_type": medal["cool_factor_medal"],
                    "competition": "Cool Factor",
                    "sport_name": medal["sport_name"],
                    "match_name": medal["target_value"],
                    "points": points
                })

        # Send one consolidated email per person
        now = datetime.utcnow().isoformat()
        for callsign, data in medals_by_person.items():
            medal_ids = medal_ids_by_person[callsign]

            # Skip if no email, notifications disabled, or medal notifications disabled
            if not data["email"] or not data["email_notifications_enabled"] or not data["email_medal_notifications"]:
                results["skipped"] += len(data["medals"])
                # Mark all as notified to prevent future attempts
                for medal_id in medal_ids:
                    conn.execute(
                        "UPDATE medals SET notified_at = ? WHERE id = ?",
                        (now, medal_id)
                    )
                continue

            # Skip unverified emails
            if not data["email_verified"]:
                results["skipped"] += len(data["medals"])
                for medal_id in medal_ids:
                    conn.execute(
                        "UPDATE medals SET notified_at = ? WHERE id = ?",
                        (now, medal_id)
                    )
                continue

            # Daily digest: at most 1 medal notification email per user per 24 hours
            today_start = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            already_notified = conn.execute(
                "SELECT COUNT(*) FROM medals "
                "WHERE callsign = ? AND notified_at > ? "
                "AND (qso_race_medal IS NOT NULL OR cool_factor_medal IS NOT NULL)",
                (callsign, today_start)
            ).fetchone()[0]
            if already_notified > 0:
                results["skipped"] += len(data["medals"])
                logger.info(f"Medal email already sent in last 24h for {callsign}, skipping until next digest window")
                continue

            # Calculate total points
            total_points = sum(m["points"] for m in data["medals"])

            # Send consolidated email
            try:
                success = await send_medals_summary_email(
                    callsign=callsign,
                    email=data["email"],
                    medals=data["medals"],
                    total_points=total_points
                )
                if success:
                    results["sent"] += 1
                    results["medals_notified"] += len(data["medals"])
                else:
                    results["errors"] += 1
                    # Notify admin of email failure
                    await send_admin_error_email(
                        error_type="Medal Email Failure",
                        error_details=f"Failed to send medal notification email to {callsign}",
                        context={
                            "callsign": callsign,
                            "email": data["email"],
                            "medal_count": len(data["medals"])
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to send medal notification to {callsign}: {e}")
                results["errors"] += 1
                # Notify admin of error
                await send_admin_error_email(
                    error_type="Medal Email Error",
                    error_details=str(e),
                    context={
                        "callsign": callsign,
                        "email": data["email"],
                        "medal_count": len(data["medals"])
                    }
                )

            # Mark all medals as notified regardless of success
            # (to prevent repeated attempts on persistent failures)
            for medal_id in medal_ids:
                conn.execute(
                    "UPDATE medals SET notified_at = ? WHERE id = ?",
                    (now, medal_id)
                )

    logger.info(f"Medal notifications: sent={results['sent']}, medals={results['medals_notified']}, skipped={results['skipped']}, errors={results['errors']}")
    return results


def _render_weekly_digest_template(callsign: str, matches: list) -> str:
    """Render weekly match digest email template."""
    matches_html = ""
    for match in matches:
        matches_html += f"""
        <div style="background-color: #ebf8ff; border-left: 4px solid #4299e1; padding: 15px; margin: 10px 0;">
            <p style="margin: 0; font-size: 16px;"><strong>{match['sport_name']}</strong></p>
            <p style="margin: 5px 0 0 0; color: #4a5568;">
                Target: {match['target_value']}<br>
                {match['start_date']} - {match['end_date']}
            </p>
        </div>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Upcoming Matches - Ham Radio Olympics</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2d3748;">Upcoming Matches This Week</h1>
    <p>Hello {callsign},</p>
    <p>Here are the upcoming matches in the Ham Radio Olympics:</p>
    {matches_html}
    <p>Get your rig ready and good luck!</p>
    <p>73,<br>Ham Radio Olympics</p>
    {_get_email_footer_html()}
</body>
</html>
"""


async def send_weekly_match_digest(days_ahead: int = 7) -> dict:
    """
    Send weekly digest of upcoming matches.

    Only sends to users who haven't received a digest in the past 7 days.

    Args:
        days_ahead: Number of days ahead to look for matches

    Returns:
        Dict with counts of digests sent and any errors
    """
    from database import get_db

    results = {"sent": 0, "skipped": 0, "errors": 0}
    now = datetime.utcnow()
    one_week_ago = now - timedelta(days=7)
    end_window = now + timedelta(days=days_ahead)

    with get_db() as conn:
        # Get all upcoming matches in the next X days
        cursor = conn.execute("""
            SELECT m.id, m.target_value, m.start_date, m.end_date,
                   s.id as sport_id, s.name as sport_name
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE o.is_active = 1
              AND m.start_date >= ?
              AND m.start_date <= ?
            ORDER BY m.start_date
        """, (now.isoformat(), end_window.isoformat()))
        all_matches = [dict(row) for row in cursor.fetchall()]

        if not all_matches:
            logger.info("No upcoming matches to notify about")
            return results

        # Get competitors who:
        # - Have email verified and match reminders enabled
        # - Haven't received a digest in the past week
        cursor = conn.execute("""
            SELECT c.callsign, c.email
            FROM competitors c
            WHERE c.email IS NOT NULL
              AND c.email_verified = 1
              AND c.email_notifications_enabled = 1
              AND COALESCE(c.email_match_reminders, 1) = 1
              AND (c.last_match_digest_at IS NULL OR c.last_match_digest_at < ?)
        """, (one_week_ago.isoformat(),))
        competitors = cursor.fetchall()

        for competitor in competitors:
            competitor = dict(competitor)
            callsign = competitor["callsign"]

            # Get sports this competitor is entered in
            cursor = conn.execute(
                "SELECT sport_id FROM sport_entries WHERE callsign = ?",
                (callsign,)
            )
            entered_sports = {row[0] for row in cursor.fetchall()}

            # Filter matches to only those the competitor is entered in
            relevant_matches = [
                m for m in all_matches if m["sport_id"] in entered_sports
            ]

            if not relevant_matches:
                results["skipped"] += 1
                continue

            # Send digest email
            try:
                html_body = _render_weekly_digest_template(callsign, relevant_matches)

                matches_text = "\n".join([
                    f"- {m['sport_name']}: {m['target_value']} ({m['start_date'][:10]} - {m['end_date'][:10]})"
                    for m in relevant_matches
                ])

                plain_body = f"""
Upcoming Matches This Week

Hello {callsign},

Here are the upcoming matches in the Ham Radio Olympics:

{matches_text}

Get your rig ready and good luck!

73,
Ham Radio Olympics
{_get_email_footer_text()}
"""

                success = await send_email(
                    to=competitor["email"],
                    subject=f"Upcoming Matches This Week - Ham Radio Olympics",
                    body=plain_body.strip(),
                    html_body=html_body
                )

                if success:
                    results["sent"] += 1
                    # Update last_match_digest_at
                    conn.execute(
                        "UPDATE competitors SET last_match_digest_at = ? WHERE callsign = ?",
                        (now.isoformat(), callsign)
                    )
                else:
                    results["errors"] += 1

            except Exception as e:
                logger.error(f"Failed to send weekly digest to {callsign}: {e}")
                results["errors"] += 1

    logger.info(f"Weekly digest: sent={results['sent']}, skipped={results['skipped']}, errors={results['errors']}")
    return results


async def send_match_reminders(hours_before: int = 168) -> dict:
    """
    Send weekly match digest emails.

    This is now an alias for send_weekly_match_digest for backwards compatibility.
    Default is 168 hours (7 days) ahead.

    Args:
        hours_before: Hours before to look for matches (default 168 = 1 week)

    Returns:
        Dict with counts of reminders sent and any errors
    """
    days_ahead = hours_before // 24 if hours_before >= 24 else 7
    return await send_weekly_match_digest(days_ahead=days_ahead)

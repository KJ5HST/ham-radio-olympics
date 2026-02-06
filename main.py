"""
Ham Radio Olympics - Main FastAPI Application
"""

import asyncio
import json
import logging
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException, Depends, UploadFile, File, Cookie, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, EmailStr, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from database import init_db, get_db, seed_example_olympiad
from crypto import encrypt_api_key
from sync import sync_competitor, sync_competitor_with_key, sync_competitor_lotw, sync_competitor_lotw_stored, sync_all_competitors, recompute_sport_matches
from qrz_client import verify_api_key
from lotw_client import verify_lotw_credentials
from scoring import recompute_match_medals, compute_team_standings, is_mode_allowed
from dxcc import get_country_name, get_continent_name, get_all_continents, get_all_countries
from auth import (
    register_user, authenticate_user, get_session_user, delete_session,
    update_user_email, update_user_password, hash_password, User, SESSION_COOKIE_NAME
)
from config import config
from email_service import (
    create_password_reset_token, validate_reset_token, mark_token_used,
    send_password_reset_email
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from log messages."""

    # Patterns for sensitive query parameters (password, api key, etc.)
    SENSITIVE_PATTERNS = [
        (re.compile(r'([?&]password=)[^&\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'([?&]api_key=)[^&\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'([?&]key=)[^&\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


# Apply filter to httpx logger to redact passwords from URLs
httpx_logger = logging.getLogger("httpx")
httpx_logger.addFilter(SensitiveDataFilter())

# Rate limiter - disabled in test mode
def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key, return empty string in test mode to disable."""
    if config.TESTING:
        return ""  # Disable rate limiting in tests
    return get_remote_address(request)

limiter = Limiter(key_func=get_rate_limit_key)

# Background task handle
_sync_task = None
_sync_process = None


def get_sync_script_path():
    """Get the path to the sync script."""
    return os.path.join(os.path.dirname(__file__), "scripts", "run_sync.py")


async def run_sync_subprocess():
    """Run sync as a subprocess, completely independent of the event loop."""
    global _sync_process
    script_path = get_sync_script_path()

    try:
        # Use asyncio subprocess to spawn without blocking
        _sync_process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(__file__)
        )
        stdout, stderr = await _sync_process.communicate()

        if _sync_process.returncode == 0:
            logger.info(f"Background sync completed successfully")
            if stdout:
                for line in stdout.decode().strip().split('\n')[-5:]:  # Last 5 lines
                    logger.debug(f"Sync output: {line}")
        else:
            logger.error(f"Background sync failed with code {_sync_process.returncode}")
            if stderr:
                logger.error(f"Sync stderr: {stderr.decode()}")
    except Exception as e:
        logger.exception(f"Failed to run sync subprocess: {e}")
    finally:
        _sync_process = None


def is_sync_paused() -> bool:
    """Check if auto-sync is paused (stored in database)."""
    with get_db() as conn:
        cursor = conn.execute("SELECT value FROM settings WHERE key = 'sync_paused'")
        row = cursor.fetchone()
        return row is not None and row["value"] == "1"


def set_sync_paused(paused: bool) -> None:
    """Set the auto-sync paused state in database."""
    from datetime import datetime
    with get_db() as conn:
        conn.execute(
            """INSERT INTO settings (key, value, updated_at) VALUES ('sync_paused', ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            ("1" if paused else "0", datetime.utcnow().isoformat(),
             "1" if paused else "0", datetime.utcnow().isoformat())
        )
        conn.commit()


async def background_sync():
    """Background task that syncs all competitors periodically via subprocess."""
    while True:
        await asyncio.sleep(config.SYNC_INTERVAL_SECONDS)
        if is_sync_paused():
            logger.info("Auto-sync is paused, skipping")
            continue
        try:
            await run_sync_subprocess()
        except Exception as e:
            # Log error but keep running
            logger.exception(f"Background sync failed: {e}")


_pota_spot_task = None
POTA_SPOT_CHECK_INTERVAL = 30 * 60  # 30 minutes


async def background_pota_spot_check():
    """Background task that checks POTA spots every 30 minutes."""
    from notifications import check_pota_spots_and_notify

    # Run immediately on startup
    try:
        results = await check_pota_spots_and_notify()
        logger.info(f"POTA spot check (startup): {results['sports_with_spots']} sports, "
                   f"{results['notifications_sent']} push, {results['discord_sent']} Discord")
    except Exception as e:
        logger.exception(f"POTA spot check (startup) failed: {e}")

    # Then run every 30 minutes
    while True:
        await asyncio.sleep(POTA_SPOT_CHECK_INTERVAL)
        try:
            results = await check_pota_spots_and_notify()
            if results["discord_sent"] or results["notifications_sent"]:
                logger.info(f"POTA spot check: {results['sports_with_spots']} sports, "
                           f"{results['notifications_sent']} push, {results['discord_sent']} Discord")
        except Exception as e:
            logger.exception(f"POTA spot check failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    from database import backfill_records
    global _sync_task
    init_db()
    seed_example_olympiad()  # Seeds example data on fresh deployments
    backfill_records()  # Backfill records for existing QSOs if needed

    # Sync on wake if it's been more than an hour since the last sync (non-blocking subprocess)
    async def startup_sync_if_needed():
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    "SELECT MAX(last_sync_at) as last_sync FROM competitors WHERE last_sync_at IS NOT NULL"
                )
                row = cursor.fetchone()
                if row and row["last_sync"]:
                    last_sync = datetime.fromisoformat(row["last_sync"].replace("Z", "+00:00"))
                    if last_sync.tzinfo:
                        last_sync = last_sync.replace(tzinfo=None)
                    hours_since_sync = (datetime.utcnow() - last_sync).total_seconds() / 3600
                    if hours_since_sync >= 1:
                        logger.info(f"Last sync was {hours_since_sync:.1f} hours ago, triggering startup sync")
                        await run_sync_subprocess()
                else:
                    # No syncs yet, trigger one
                    logger.info("No previous sync found, triggering startup sync")
                    await run_sync_subprocess()
        except Exception as e:
            logger.exception(f"Startup sync check failed: {e}")

    # Run startup sync in background (non-blocking)
    asyncio.create_task(startup_sync_if_needed())

    # Start background tasks
    global _pota_spot_task
    _sync_task = asyncio.create_task(background_sync())
    _pota_spot_task = asyncio.create_task(background_pota_spot_check())
    yield
    # Cancel background tasks on shutdown
    if _sync_task:
        _sync_task.cancel()
    if _pota_spot_task:
        _pota_spot_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass
    # Terminate any running sync subprocess
    if _sync_process:
        try:
            _sync_process.terminate()
            await asyncio.wait_for(_sync_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _sync_process.kill()
        except Exception:
            pass


# Initialize app
app = FastAPI(
    title="Ham Radio Olympics",
    description="""
Olympic-style amateur radio competition platform.

## Features
- **Sports & Matches**: Organize competitions with configurable targets (continents, parks, grids, etc.)
- **Medal System**: Gold, Silver, Bronze medals for QSO Race and Cool Factor competitions
- **QSO Sync**: Automatic sync from QRZ Logbook and LoTW
- **Teams**: Team competitions with multiple scoring methods
- **Records**: Track personal bests and world records

## Authentication
- Web UI uses session cookies
- API endpoints use session cookies or require login
- Admin endpoints require `X-Admin-Key` header

## Rate Limits
- Most API endpoints: 60 requests/minute
- Auth endpoints: 3-5 requests/minute
    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "public", "description": "Public endpoints (no auth required)"},
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "user", "description": "User dashboard and profile"},
        {"name": "sports", "description": "Sports and matches"},
        {"name": "api", "description": "JSON API endpoints"},
        {"name": "admin", "description": "Admin management endpoints"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Templates
templates = Jinja2Templates(directory="templates")


def get_site_config():
    """Get current site config from database with fallback to env vars."""
    from database import get_setting
    return {
        "theme": get_setting("site_theme") or config.SITE_THEME,
        "name": get_setting("site_name") or config.SITE_NAME,
        "tagline": get_setting("site_tagline") or config.SITE_TAGLINE,
        "public_results": get_setting("public_results") == "1",
    }


templates.env.globals["get_site_config"] = get_site_config

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve service worker from root for proper scope
@app.get("/sw.js")
async def service_worker():
    """Serve service worker from root path for full scope."""
    from starlette.responses import FileResponse
    return FileResponse(
        "static/sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"}
    )


# CSRF token management
CSRF_COOKIE_NAME = config.CSRF_COOKIE_NAME


def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return secrets.token_urlsafe(32)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """CSRF protection middleware."""
    # Get or create CSRF token
    csrf_token = request.cookies.get(CSRF_COOKIE_NAME)
    is_new_session = csrf_token is None

    if not csrf_token:
        csrf_token = generate_csrf_token()

    request.state.csrf_token = csrf_token

    # Skip CSRF validation in test mode
    if config.TESTING:
        response = await call_next(request)
        return response

    # Validate CSRF for all POSTs except JSON API calls
    # JSON requires CORS preflight and cannot be submitted via simple HTML forms
    # Skip validation for new sessions (no cookie yet) to allow first form submission
    # Skip validation for logout (low-risk, doesn't modify data)
    csrf_exempt_paths = ["/logout"]
    if request.method == "POST" and not is_new_session and request.url.path not in csrf_exempt_paths:
        content_type = request.headers.get("content-type", "")

        # JSON API calls are exempt (require CORS preflight, can't be forged via HTML forms)
        # All other content types (form, text/plain, empty) require CSRF validation
        # Non-form content types will fail token parsing and be rejected
        is_json_api = "application/json" in content_type

        if not is_json_api:
            # Read body and cache it for later use by the endpoint
            body = await request.body()
            # Parse form data manually to check CSRF token
            from urllib.parse import parse_qs
            form_data = parse_qs(body.decode())
            form_token = form_data.get("csrf_token", [None])[0]

            if not form_token or form_token != csrf_token:
                logger.warning(f"CSRF validation failed for {request.url.path}")
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed"}
                )

            # Create a new request with the cached body so endpoint can read it
            async def receive():
                return {"type": "http.request", "body": body}
            request = Request(request.scope, receive)
            request.state.csrf_token = csrf_token

    response = await call_next(request)

    # Set CSRF cookie if not present
    if is_new_session:
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=csrf_token,
            httponly=True,
            samesite="lax",
            secure=config.SECURE_COOKIES,
            max_age=30 * 24 * 60 * 60  # 30 days
        )

    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Control referrer information
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions policy (disable unnecessary features)
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none';"
    )

    # HSTS - only on production (not localhost)
    host = request.headers.get("host", "")
    if not host.startswith("localhost") and not host.startswith("127.0.0.1"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


def get_csrf_token(request: Request) -> str:
    """Get CSRF token for the request."""
    return getattr(request.state, 'csrf_token', generate_csrf_token())


def format_date(value: str, include_time: bool = False, home_grid: str = None, time_display: str = "utc") -> str:
    """Format ISO date string to readable format.

    Args:
        value: ISO date/datetime string
        include_time: Whether to include time in output
        home_grid: User's home grid square for timezone calculation
        time_display: 'utc' or 'local'
    """
    if not value:
        return "-"
    try:
        # Parse ISO format
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Remove timezone info for manipulation
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)

        tz_str = ""
        if include_time:
            if time_display == "local" and home_grid:
                offset = grid_to_timezone_offset(home_grid, dt)
                dt = dt + timedelta(hours=offset)
                sign = "+" if offset >= 0 else ""
                tz_str = f" (UTC{sign}{offset})"
            else:
                tz_str = " UTC"
            return dt.strftime("%b %-d, %Y %-I:%M %p") + tz_str
        return dt.strftime("%b %-d, %Y")
    except (ValueError, AttributeError):
        return value


def format_datetime(value: str, home_grid: str = None, time_display: str = "utc") -> str:
    """Format ISO datetime string with time.

    Args:
        value: ISO datetime string
        home_grid: User's home grid square for timezone calculation
        time_display: 'utc' or 'local'
    """
    return format_date(value, include_time=True, home_grid=home_grid, time_display=time_display)


templates.env.filters["format_date"] = format_date
templates.env.filters["format_datetime"] = format_datetime


def format_datetime_short(value: str) -> str:
    """Format datetime as mm/dd/yy hh:mm UTC for compact display."""
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt.strftime("%m/%d/%y %H:%M") + "z"
    except (ValueError, AttributeError):
        return value[:16] if len(value) >= 16 else value


templates.env.filters["format_datetime_short"] = format_datetime_short


def get_country_flag(dxcc_code) -> str:
    """Get country flag emoji from DXCC code, wrapped in a span with country name tooltip."""
    from markupsafe import Markup
    from dxcc import get_country_name
    from callsign_lookup import get_country_flag as _get_flag

    if not dxcc_code:
        return ""
    try:
        country = get_country_name(int(dxcc_code))
        if country:
            flag = _get_flag(country)
            if flag:
                # Escape country name for HTML attribute
                safe_country = country.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                return Markup(f'<span title="{safe_country}" style="cursor: help;">{flag}</span>')
    except (ValueError, TypeError):
        pass
    return ""


templates.env.filters["country_flag"] = get_country_flag


def competitor_display(first_name, callsign) -> str:
    """Format competitor name as 'FirstName (CALLSIGN)' or just 'CALLSIGN'."""
    if first_name:
        return f"{first_name} ({callsign})"
    return callsign


templates.env.filters["competitor_display"] = competitor_display


def pota_tooltip(value: str) -> str:
    """Wrap POTA park references with tooltip markup.

    POTA references follow format: XX-NNNN (2+ letters, dash, 4+ digits)
    Examples: K-0001, US-12607, VE-0123, DL-0456
    """
    import html
    if not value:
        return value
    # Escape the entire value first to prevent XSS
    escaped_value = html.escape(str(value))
    # Pattern matches POTA park references (3+ digits to handle non-zero-padded IDs)
    pattern = r'\b([A-Z]{1,3}-\d{3,})\b'

    def replace_pota(match):
        ref = match.group(1)
        # ref is already escaped since it comes from escaped_value
        return f'<span class="pota-ref" data-pota="{ref}">{ref}</span>'

    result = re.sub(pattern, replace_pota, escaped_value)
    return Markup(result)


templates.env.filters["pota_tooltip"] = pota_tooltip


def format_distance(value, unit: str = "km") -> str:
    """Format distance with unit conversion.

    Args:
        value: Distance in kilometers
        unit: 'km' or 'mi'
    """
    if value is None:
        return ""
    try:
        km = float(value)
        if unit == "mi":
            miles = km * 0.621371
            return f"{miles:,.0f} mi"
        return f"{km:,.0f} km"
    except (ValueError, TypeError):
        return str(value)


templates.env.filters["format_distance"] = format_distance


def grid_to_timezone_offset(grid: str, dt: datetime = None) -> int:
    """Calculate timezone offset (hours from UTC) based on grid square longitude.

    Uses simplified calculation: each 20 degrees = ~1.33 hours offset.
    Grid squares are 2 degrees wide for the first 2 chars (field).

    If dt is provided, adjusts for daylight saving time in applicable regions.
    """
    if not grid or len(grid) < 2:
        return 0

    try:
        # First char (A-R) represents 20-degree longitude zones
        lon_field = ord(grid[0].upper()) - ord('A')  # 0-17
        # Calculate center longitude: -180 + (field * 20) + 10
        lon = -180 + (lon_field * 20) + 10

        # Second char (A-R) represents 10-degree latitude zones
        lat_field = ord(grid[1].upper()) - ord('A')  # 0-17
        lat = -90 + (lat_field * 10) + 5

        # Timezone is roughly longitude / 15
        offset = round(lon / 15)

        # Check for DST in applicable regions
        if dt:
            # Continental US/Canada (lon -130 to -50, lat 25 to 60)
            # Most observe DST from second Sunday in March to first Sunday in November
            if -130 <= lon <= -50 and 25 <= lat <= 60:
                if _is_us_dst(dt):
                    offset += 1
            # Western Europe (lon -10 to 30, lat 35 to 70)
            # Observes DST from last Sunday in March to last Sunday in October
            elif -10 <= lon <= 30 and 35 <= lat <= 70:
                if _is_eu_dst(dt):
                    offset += 1

        return offset
    except (ValueError, IndexError):
        return 0


def _is_us_dst(dt: datetime) -> bool:
    """Check if US daylight saving time is in effect for the given date.

    DST runs from second Sunday in March to first Sunday in November.
    """
    year = dt.year
    # Second Sunday in March
    march_start = datetime(year, 3, 8)  # Earliest possible second Sunday
    while march_start.weekday() != 6:  # Find Sunday
        march_start += timedelta(days=1)

    # First Sunday in November
    nov_end = datetime(year, 11, 1)
    while nov_end.weekday() != 6:
        nov_end += timedelta(days=1)

    return march_start <= dt.replace(tzinfo=None) < nov_end


def _is_eu_dst(dt: datetime) -> bool:
    """Check if EU daylight saving time is in effect for the given date.

    DST runs from last Sunday in March to last Sunday in October.
    """
    year = dt.year
    # Last Sunday in March
    march_end = datetime(year, 3, 31)
    while march_end.weekday() != 6:
        march_end -= timedelta(days=1)

    # Last Sunday in October
    oct_end = datetime(year, 10, 31)
    while oct_end.weekday() != 6:
        oct_end -= timedelta(days=1)

    return march_end <= dt.replace(tzinfo=None) < oct_end


def format_time_local(value, home_grid: str = None, time_display: str = "utc") -> str:
    """Format datetime with optional local time conversion.

    Args:
        value: ISO datetime string (UTC)
        home_grid: User's home grid square for timezone calculation
        time_display: 'utc' or 'local'
    """
    if not value:
        return ""

    try:
        if isinstance(value, str):
            # Handle ISO format with or without T
            value = value.replace('T', ' ').split('.')[0]
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        else:
            dt = value

        if time_display == "local" and home_grid:
            offset = grid_to_timezone_offset(home_grid, dt)
            dt = dt + timedelta(hours=offset)
            sign = "+" if offset >= 0 else ""
            tz_str = f" (UTC{sign}{offset})"
        else:
            tz_str = " UTC"

        return dt.strftime("%b %-d, %Y %-I:%M %p") + tz_str
    except (ValueError, AttributeError):
        return str(value)


templates.env.filters["format_time_local"] = format_time_local


# Admin key from environment
ADMIN_KEY = config.ADMIN_KEY


# Pydantic models
def is_valid_callsign_format(callsign: str) -> bool:
    """
    Validate callsign format using regex patterns for international amateur callsigns.

    Patterns cover:
    - US: W1AW, K2ABC, N3XYZ, KD5DX, WA1ABC, etc.
    - International: VE3ABC, G4XYZ, DL1ABC, JA1ABC, VK2ABC, etc.

    General format: [prefix][digit][suffix]
    - Prefix: 1-3 letters/numbers (starts with letter for most countries)
    - Digit: 1 digit (required)
    - Suffix: 1-4 letters
    """
    callsign = callsign.upper().strip()

    # Pattern: 1-3 char prefix, 1 digit, 1-4 letter suffix
    # Examples: W1A, K2AB, N3ABC, KD5DX, WA1ABCD, VE3XYZ, G4ABC, DL1ABC
    pattern = r'^[A-Z]{1,2}[0-9][A-Z]{1,4}$|^[A-Z][0-9][A-Z]{1,4}$|^[0-9][A-Z][0-9][A-Z]{1,4}$|^[A-Z]{1,2}[0-9]{1,2}[A-Z]{1,4}$'

    return bool(re.match(pattern, callsign))


class UserSignup(BaseModel):
    callsign: str
    password: str
    qrz_api_key: Optional[str] = None
    lotw_username: Optional[str] = None
    lotw_password: Optional[str] = None
    store_credentials: bool = True  # Whether to store credentials for auto-sync
    email: Optional[EmailStr] = None

    @field_validator("callsign")
    @classmethod
    def validate_callsign(cls, v):
        v = v.upper().strip()
        if not is_valid_callsign_format(v):
            raise ValueError("Invalid callsign format")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserLogin(BaseModel):
    callsign: str
    password: str


class OlympiadCreate(BaseModel):
    name: str
    start_date: str
    end_date: str
    qualifying_qsos: int = 0

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
        return v

    @field_validator('qualifying_qsos')
    @classmethod
    def validate_qualifying_qsos(cls, v):
        if v < 0:
            raise ValueError('qualifying_qsos must be non-negative')
        return v


class SportCreate(BaseModel):
    name: str
    description: Optional[str] = None
    target_type: str
    work_enabled: bool = True
    activate_enabled: bool = False
    separate_pools: bool = False
    allowed_modes: Optional[str] = None

    @field_validator('target_type')
    @classmethod
    def validate_target_type(cls, v):
        valid_types = {'continent', 'country', 'park', 'call', 'grid', 'any', 'pota'}
        if v not in valid_types:
            raise ValueError(f'target_type must be one of: {", ".join(valid_types)}')
        return v


class MatchCreate(BaseModel):
    start_date: str
    end_date: str
    target_value: str
    target_type: Optional[str] = None  # Override sport's target_type if set
    allowed_modes: Optional[str] = None
    max_power_w: Optional[int] = None
    confirmation_deadline: Optional[str] = None  # Deadline for QSO confirmations

    @field_validator('start_date', 'end_date', 'confirmation_deadline')
    @classmethod
    def validate_date_format(cls, v):
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)')
        return v

    @field_validator('target_type')
    @classmethod
    def validate_target_type(cls, v):
        if v is not None:
            valid_types = {'continent', 'country', 'park', 'call', 'grid', 'any', 'pota'}
            if v not in valid_types:
                raise ValueError(f'Invalid target_type. Must be one of: {", ".join(sorted(valid_types))}')
        return v

    @field_validator('max_power_w')
    @classmethod
    def validate_max_power(cls, v):
        if v is not None and v <= 0:
            raise ValueError('max_power_w must be positive')
        return v


class QSODisqualifyRequest(BaseModel):
    """Request to disqualify a QSO from a sport."""
    reason: str

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v):
        if not v or not v.strip():
            raise ValueError('Reason is required')
        if len(v.strip()) < 10:
            raise ValueError('Reason must be at least 10 characters')
        return v.strip()


class QSORefuteRequest(BaseModel):
    """Request to refute a QSO disqualification."""
    refutation: str

    @field_validator('refutation')
    @classmethod
    def validate_refutation(cls, v):
        if not v or not v.strip():
            raise ValueError('Refutation is required')
        if len(v.strip()) < 10:
            raise ValueError('Refutation must be at least 10 characters')
        return v.strip()


class QSORequalifyRequest(BaseModel):
    """Request to requalify a disqualified QSO."""
    reason: str

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v):
        if not v or not v.strip():
            raise ValueError('Reason is required')
        if len(v.strip()) < 10:
            raise ValueError('Reason must be at least 10 characters')
        return v.strip()


# Admin authentication dependency
def verify_admin(request: Request):
    """Verify admin access via key or logged-in admin user."""
    # Check admin key from header only (not query params for security)
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key and admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user and user.is_admin:
        return True
    # For browser requests, redirect to login instead of JSON error
    accept = request.headers.get("Accept", "")
    if "text/html" in accept:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"}
        )
    raise HTTPException(status_code=403, detail="Admin access required")


def is_referee_for_sport(callsign: str, sport_id: int) -> bool:
    """Check if a user is assigned as referee for a specific sport."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT 1 FROM referee_assignments WHERE callsign = ? AND sport_id = ?",
            (callsign, sport_id)
        )
        return cursor.fetchone() is not None


def verify_admin_or_sport_referee(request: Request, sport_id: int):
    """Verify admin access or referee assignment for the sport."""
    # Check admin key from header only (not query params for security)
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key and admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin or referee for this sport
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user:
        if user.is_admin:
            return True
        if user.is_referee and is_referee_for_sport(user.callsign, sport_id):
            return True
    # For browser requests, redirect to login instead of JSON error
    accept = request.headers.get("Accept", "")
    if "text/html" in accept:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"}
        )
    raise HTTPException(status_code=403, detail="Admin or referee access required")


def verify_admin_or_referee(request: Request):
    """Verify admin access or referee role."""
    # Check admin key from header only (not query params for security)
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key and admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin or referee
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user and (user.is_admin or user.is_referee):
        return True
    # For browser requests, redirect to login instead of JSON error
    accept = request.headers.get("Accept", "")
    if "text/html" in accept:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"}
        )
    raise HTTPException(status_code=403, detail="Admin or referee access required")


def format_target_display(target_value: str, target_type: str) -> str:
    """Format target value with human-readable name based on target type."""
    if target_type == "continent":
        name = get_continent_name(target_value)
        if name:
            return f"{name} ({target_value})"
    elif target_type == "country":
        try:
            name = get_country_name(int(target_value))
            if name:
                return f"{name} ({target_value})"
        except (ValueError, TypeError):
            pass
    return target_value


# User authentication dependency
def get_current_user(request: Request) -> Optional[User]:
    """Get current user from session cookie."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    return get_session_user(session_id)


def require_user(request: Request) -> User:
    """Require authenticated user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def is_public_results_enabled() -> bool:
    """Check if public results viewing is enabled."""
    from database import get_setting
    return get_setting("public_results") == "1"


def optional_user(request: Request) -> Optional[User]:
    """Get current user if logged in, otherwise return None."""
    return get_current_user(request)


def require_user_or_public(request: Request) -> Optional[User]:
    """Require authenticated user OR public results must be enabled."""
    user = get_current_user(request)
    if user:
        return user
    if is_public_results_enabled():
        return None
    raise HTTPException(status_code=401, detail="Not authenticated")


def get_display_prefs(user: Optional[User]) -> dict:
    """Get display preferences for a user. Returns defaults if no user or not found."""
    if not user:
        return {"distance_unit": "km", "time_display": "utc", "home_grid": None}
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT distance_unit, time_display, home_grid FROM competitors WHERE callsign = ?",
            (user.callsign,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
    return {"distance_unit": "km", "time_display": "utc", "home_grid": None}


def get_client_ip(request: Request) -> Optional[str]:
    """Get real client IP, handling proxy headers from Fly.io."""
    # Check X-Forwarded-For header (set by Fly.io proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
        # The first one is the original client
        return forwarded_for.split(",")[0].strip()
    # Fall back to direct client IP
    return request.client.host if request.client else None


# ============================================================
# PUBLIC ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page with active Olympiad info."""
    user = get_current_user(request)
    if not user and not is_public_results_enabled():
        return RedirectResponse(url="/login", status_code=303)
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

        sports = []
        if olympiad:
            cursor = conn.execute(
                "SELECT * FROM sports WHERE olympiad_id = ?",
                (olympiad["id"],)
            )
            sports = [dict(row) for row in cursor.fetchall()]

            # Fetch matches for each sport
            for sport in sports:
                cursor = conn.execute(
                    "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
                    (sport["id"],)
                )
                sport["matches"] = [dict(row) for row in cursor.fetchall()]
                # Add target_display to each match
                for match in sport["matches"]:
                    match["target_display"] = format_target_display(match["target_value"], sport["target_type"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "olympiad": dict(olympiad) if olympiad else None,
        "sports": sports,
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}




@app.get("/olympiad")
async def get_olympiad(user: User = Depends(require_user)):
    """Get current active Olympiad details."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

        if not olympiad:
            raise HTTPException(status_code=404, detail="No active Olympiad")

        return dict(olympiad)


@app.get("/olympiad/sports")
async def get_olympiad_sports(user: User = Depends(require_user)):
    """List all Sports in the active Olympiad."""
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

        if not olympiad:
            raise HTTPException(status_code=404, detail="No active Olympiad")

        cursor = conn.execute(
            "SELECT * FROM sports WHERE olympiad_id = ?",
            (olympiad["id"],)
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/olympiad/sport/{sport_id}", response_class=HTMLResponse)
async def get_sport(request: Request, sport_id: int, page: int = 1, user: Optional[User] = Depends(require_user_or_public)):
    """Get Sport details and standings."""
    MATCHES_PER_PAGE = 50

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Get total match count for pagination
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM matches WHERE sport_id = ?",
            (sport_id,)
        )
        total_matches = cursor.fetchone()["count"]
        total_pages = (total_matches + MATCHES_PER_PAGE - 1) // MATCHES_PER_PAGE

        # Clamp page to valid range
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        offset = (page - 1) * MATCHES_PER_PAGE

        # Get paginated matches
        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date LIMIT ? OFFSET ?",
            (sport_id, MATCHES_PER_PAGE, offset)
        )
        matches = [dict(row) for row in cursor.fetchall()]

        # Add target_display to matches
        sport_dict = dict(sport)
        for match in matches:
            match["target_display"] = format_target_display(match["target_value"], sport_dict["target_type"])
            match["user_logged"] = False  # Default

        # Check if current user has confirmed QSOs for each match
        if user:
            from scoring import matches_target, is_mode_allowed
            match_ids = [m["id"] for m in matches]
            if match_ids:
                placeholders = ",".join("?" * len(match_ids))
                # Get user's confirmed QSOs that could apply to these matches
                cursor = conn.execute(f"""
                    SELECT q.*, ma.id as match_id, ma.target_value, ma.start_date, ma.end_date,
                           COALESCE(ma.allowed_modes, s.allowed_modes) as match_allowed_modes
                    FROM qsos q
                    CROSS JOIN matches ma
                    JOIN sports s ON ma.sport_id = s.id
                    WHERE q.competitor_callsign = ?
                      AND q.is_confirmed = 1
                      AND ma.id IN ({placeholders})
                      AND q.qso_datetime_utc >= ma.start_date
                      AND q.qso_datetime_utc <= ma.end_date || ' 23:59:59'
                """, [user.callsign] + match_ids)
                user_qsos = cursor.fetchall()

                # Check which matches the user has logged
                logged_matches = set()
                for qso in user_qsos:
                    qso_dict = dict(qso)
                    match_id = qso_dict["match_id"]
                    target_value = qso_dict["target_value"]
                    allowed_modes = qso_dict["match_allowed_modes"]

                    # Check mode is allowed
                    if not is_mode_allowed(qso_dict.get("mode"), allowed_modes):
                        continue

                    # Check if QSO matches target
                    matches_work = sport_dict["work_enabled"] and matches_target(qso_dict, sport_dict["target_type"], target_value, "work")
                    matches_activate = sport_dict["activate_enabled"] and matches_target(qso_dict, sport_dict["target_type"], target_value, "activate")
                    if matches_work or matches_activate:
                        logged_matches.add(match_id)

                # Update matches with logged status
                for match in matches:
                    match["user_logged"] = match["id"] in logged_matches

        # Get cumulative standings with competitor names
        cursor = conn.execute("""
            SELECT m.callsign, m.role, c.first_name,
                   SUM(m.total_points) as total_points,
                   SUM(CASE WHEN m.qso_race_medal = 'gold' THEN 1 ELSE 0 END +
                       CASE WHEN m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold,
                   SUM(CASE WHEN m.qso_race_medal = 'silver' THEN 1 ELSE 0 END +
                       CASE WHEN m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver,
                   SUM(CASE WHEN m.qso_race_medal = 'bronze' THEN 1 ELSE 0 END +
                       CASE WHEN m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            LEFT JOIN competitors c ON m.callsign = c.callsign
            WHERE ma.sport_id = ?
            GROUP BY m.callsign, m.role, c.first_name
            ORDER BY total_points DESC, gold DESC, silver DESC, bronze DESC, m.callsign
        """, (sport_id,))
        standings = [dict(row) for row in cursor.fetchall()]

        # Get detailed medals per match for each competitor (for expandable rows)
        cursor = conn.execute("""
            SELECT m.callsign, m.match_id, ma.target_value, ma.start_date, ma.end_date,
                   m.qso_race_medal, m.qso_race_claim_time,
                   m.cool_factor_medal, m.cool_factor_value,
                   m.pota_bonus, m.total_points,
                   COALESCE(ma.allowed_modes, s.allowed_modes) as allowed_modes
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            WHERE ma.sport_id = ?
            ORDER BY ma.start_date DESC
        """, (sport_id,))
        medal_rows = cursor.fetchall()

        # Get all confirmed QSOs for competitors with medals in this sport
        callsigns = list(set(row["callsign"] for row in medal_rows))
        qsos_by_callsign = {}
        if callsigns:
            from scoring import matches_target, is_mode_allowed
            placeholders = ",".join("?" * len(callsigns))
            cursor = conn.execute(f"""
                SELECT q.competitor_callsign, q.dx_callsign, q.qso_datetime_utc,
                       q.mode, q.tx_power_w, q.distance_km, q.cool_factor,
                       q.dx_grid, q.dx_sig_info, q.my_sig_info, q.dx_dxcc, q.my_dxcc,
                       q.my_grid
                FROM qsos q
                WHERE q.competitor_callsign IN ({placeholders})
                  AND q.is_confirmed = 1
                ORDER BY q.qso_datetime_utc ASC
            """, callsigns)
            all_qsos = [dict(row) for row in cursor.fetchall()]

            # Index QSOs by callsign for faster lookup
            for qso in all_qsos:
                cs = qso["competitor_callsign"]
                if cs not in qsos_by_callsign:
                    qsos_by_callsign[cs] = []
                qsos_by_callsign[cs].append(qso)

        medal_details = {}
        for row in medal_rows:
            callsign = row["callsign"]
            match_id = row["match_id"]
            target_value = row["target_value"]
            start_date = row["start_date"]
            end_date = row["end_date"]
            qso_medal = row["qso_race_medal"]
            cf_medal = row["cool_factor_medal"]
            allowed_modes = row["allowed_modes"]

            # Find QSOs that match this medal's target and date range
            start_date_cmp = start_date[:10] if start_date else None
            end_date_cmp = end_date[:10] if end_date else None
            matching_qsos = []

            for qso in qsos_by_callsign.get(callsign, []):
                qso_date = qso["qso_datetime_utc"][:10] if qso["qso_datetime_utc"] else None
                if qso_date and start_date_cmp and end_date_cmp and start_date_cmp <= qso_date <= end_date_cmp:
                    # Check if QSO mode is allowed for this sport/match
                    if not is_mode_allowed(qso.get("mode"), allowed_modes):
                        continue
                    # Check if QSO matches target (respecting sport's enabled modes)
                    matches_work = sport_dict["work_enabled"] and matches_target(qso, sport_dict["target_type"], target_value, "work")
                    matches_activate = sport_dict["activate_enabled"] and matches_target(qso, sport_dict["target_type"], target_value, "activate")
                    if matches_work or matches_activate:
                        matching_qsos.append(qso)

            # Build the specific medal-winning QSOs list (not all matching QSOs)
            medal_qsos = []

            # For QSO Race medal: the earliest QSO
            if qso_medal and matching_qsos:
                earliest = min(matching_qsos, key=lambda q: q["qso_datetime_utc"])
                medal_qsos.append({
                    "dx_callsign": earliest["dx_callsign"],
                    "time": earliest["qso_datetime_utc"],
                    "mode": earliest["mode"],
                    "power": earliest["tx_power_w"],
                    "distance": earliest["distance_km"],
                    "cool_factor": earliest["cool_factor"],
                    "medal_type": "QSO Race",
                    "my_park": earliest.get("my_sig_info"),
                    "my_grid": earliest.get("my_grid"),
                })

            # For Cool Factor medal: the QSO with highest cool_factor
            # Always add as separate row even if same QSO won both medals
            if cf_medal and matching_qsos:
                qsos_with_cf = [q for q in matching_qsos if q["cool_factor"] and q["cool_factor"] > 0]
                if qsos_with_cf:
                    best_cf = max(qsos_with_cf, key=lambda q: q["cool_factor"])
                    medal_qsos.append({
                        "dx_callsign": best_cf["dx_callsign"],
                        "time": best_cf["qso_datetime_utc"],
                        "mode": best_cf["mode"],
                        "power": best_cf["tx_power_w"],
                        "distance": best_cf["distance_km"],
                        "cool_factor": best_cf["cool_factor"],
                        "medal_type": "Cool Factor",
                        "my_park": best_cf.get("my_sig_info"),
                        "my_grid": best_cf.get("my_grid"),
                    })

            if callsign not in medal_details:
                medal_details[callsign] = []
            medal_details[callsign].append({
                "match_id": match_id,
                "target": format_target_display(target_value, sport_dict["target_type"]),
                "target_value": target_value,
                "qso_medal": qso_medal,
                "qso_time": row["qso_race_claim_time"],
                "cf_medal": cf_medal,
                "cf_value": row["cool_factor_value"],
                "pota_bonus": row["pota_bonus"],
                "points": row["total_points"],
                "qsos": medal_qsos
            })

        # Check if user has entered this sport
        is_entered = False
        if user:
            cursor = conn.execute(
                "SELECT id FROM sport_entries WHERE callsign = ? AND sport_id = ?",
                (user.callsign, sport_id)
            )
            is_entered = cursor.fetchone() is not None

        # Get entry count
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM sport_entries WHERE sport_id = ?",
            (sport_id,)
        )
        entry_count = cursor.fetchone()["count"]

        # Get currently active matches (by date) for POTA spot indicators
        # This is separate from paginated matches to ensure spots always show
        active_matches = []
        if sport["target_type"] == "park":
            now = datetime.utcnow().strftime("%Y-%m-%d")
            cursor = conn.execute("""
                SELECT id, target_value FROM matches
                WHERE sport_id = ? AND start_date <= ? AND end_date >= ?
            """, (sport_id, now, now))
            active_matches = [dict(row) for row in cursor.fetchall()]
            for match in active_matches:
                match["target_display"] = format_target_display(match["target_value"], sport["target_type"])

        return templates.TemplateResponse("sport.html", {
            "request": request,
            "user": user,
            "sport": dict(sport),
            "matches": matches,
            "active_matches": active_matches,
            "standings": standings,
            "medal_details": medal_details,
            "is_entered": is_entered,
            "entry_count": entry_count,
            "page": page,
            "total_pages": total_pages,
            "total_matches": total_matches,
            "display_prefs": get_display_prefs(user),
        })


@app.get("/olympiad/sport/{sport_id}/participants", response_class=HTMLResponse)
async def get_sport_participants(request: Request, sport_id: int, user: User = Depends(require_user)):
    """View all participants in a Sport."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Get all competitors entered in this sport with their medal counts and points
        cursor = conn.execute("""
            SELECT c.callsign, c.first_name, c.last_name,
                   COALESCE(SUM(m.total_points), 0) as total_points,
                   SUM(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                   SUM(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                   SUM(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count
            FROM sport_entries se
            JOIN competitors c ON se.callsign = c.callsign
            LEFT JOIN medals m ON m.callsign = c.callsign AND m.match_id IN (
                SELECT id FROM matches WHERE sport_id = ?
            )
            WHERE se.sport_id = ?
            GROUP BY c.callsign, c.first_name, c.last_name
            ORDER BY total_points DESC, c.callsign
        """, (sport_id, sport_id))
        participants = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("sport_participants.html", {
        "request": request,
        "user": user,
        "sport": dict(sport),
        "participants": participants,
    })


@app.get("/olympiad/sport/{sport_id}/matches")
async def get_sport_matches(sport_id: int, user: Optional[User] = Depends(require_user_or_public)):
    """List all Matches in a Sport."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
            (sport_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/olympiad/sport/{sport_id}/match/{match_id}", response_class=HTMLResponse)
async def get_match(request: Request, sport_id: int, match_id: int, user: Optional[User] = Depends(require_user_or_public)):
    """Get Match details and leaderboard."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE id = ? AND sport_id = ?",
            (match_id, sport_id)
        )
        match = cursor.fetchone()

        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        # Get medal results with competitor names
        cursor = conn.execute("""
            SELECT m.*, c.first_name
            FROM medals m
            LEFT JOIN competitors c ON m.callsign = c.callsign
            WHERE m.match_id = ?
            ORDER BY m.total_points DESC, m.qso_race_claim_time ASC
        """, (match_id,))
        medals = [dict(row) for row in cursor.fetchall()]

        # Get sport info for display
        cursor = conn.execute("SELECT name, target_type, work_enabled, activate_enabled, allowed_modes FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

        # Format target display
        match_dict = dict(match)
        target_display = format_target_display(match_dict["target_value"], sport["target_type"]) if sport else match_dict["target_value"]

        # Check if current user is referee for this sport
        is_sport_referee = False
        if user:
            is_sport_referee = user.is_admin or (user.is_referee and is_referee_for_sport(user.callsign, sport_id))

        # Get confirmed QSOs for each competitor in this match (for expandable rows)
        # Include all fields needed for target matching, plus disqualification status
        cursor = conn.execute("""
            SELECT q.id, q.competitor_callsign, q.dx_callsign, q.qso_datetime_utc,
                   q.mode, q.tx_power_w, q.distance_km, q.cool_factor, q.is_confirmed,
                   q.dx_grid, q.dx_sig_info, q.my_sig_info, q.dx_dxcc, q.my_dxcc, q.my_grid,
                   dq.status as dq_status
            FROM qsos q
            LEFT JOIN qso_disqualifications dq ON q.id = dq.qso_id AND dq.sport_id = ?
            WHERE q.competitor_callsign IN (SELECT callsign FROM medals WHERE match_id = ?)
              AND datetime(q.qso_datetime_utc) >= datetime(?)
              AND datetime(q.qso_datetime_utc) <= datetime(?)
              AND q.is_confirmed = 1
            ORDER BY q.qso_datetime_utc ASC
        """, (sport_id, match_id, match_dict["start_date"], match_dict["end_date"]))

        # Filter QSOs to only those matching the target
        # Use match's target_type override if set, otherwise use sport's target_type
        from scoring import matches_target
        target_type = match_dict.get("target_type") or (sport["target_type"] if sport else None)
        target_value = match_dict["target_value"]

        # Collect all matching QSOs per competitor, then pick only medal-winning ones
        all_qsos_by_callsign = {}
        # Parse allowed_modes for filtering
        allowed_modes = None
        if sport and sport["allowed_modes"]:
            allowed_modes = [m.strip().upper() for m in sport["allowed_modes"].split(",")]

        for row in cursor.fetchall():
            qso = dict(row)

            # Filter by allowed_modes if specified
            if allowed_modes:
                qso_mode = (qso.get("mode") or "").upper()
                if qso_mode not in allowed_modes:
                    continue

            # Check if QSO matches target (only check modes that are enabled for this sport)
            if target_type and target_value:
                matches_work = sport["work_enabled"] and matches_target(qso, target_type, target_value, "work")
                matches_activate = sport["activate_enabled"] and matches_target(qso, target_type, target_value, "activate")
                if not matches_work and not matches_activate:
                    continue

            callsign = row["competitor_callsign"]
            if callsign not in all_qsos_by_callsign:
                all_qsos_by_callsign[callsign] = []
            all_qsos_by_callsign[callsign].append({
                "id": row["id"],  # QSO ID for disqualification actions
                "dx_callsign": row["dx_callsign"],
                "time": row["qso_datetime_utc"],
                "mode": row["mode"],
                "power": row["tx_power_w"],
                "distance": row["distance_km"],
                "cool_factor": row["cool_factor"],
                "confirmed": row["is_confirmed"],
                "grid": row["dx_grid"],
                "park": row["dx_sig_info"],
                "my_park": row["my_sig_info"],  # Competitor was activating from this park
                "my_grid": row["my_grid"],  # Competitor's grid for local time display
                "dq_status": row["dq_status"],  # Disqualification status
            })

        # Now pick only the medal-winning QSOs for each competitor:
        # 1. First confirmed QSO (QSO Race) - already sorted by time ASC
        # 2. Best cool factor QSO (Cool Factor medal)
        qso_details = {}
        for callsign, qsos in all_qsos_by_callsign.items():
            # Filter to only non-disqualified QSOs for medal determination
            valid_qsos = [q for q in qsos if q["dq_status"] != "disqualified"]
            if not valid_qsos:
                continue

            medal_qsos = []
            qso_ids_added = set()

            # First QSO (QSO Race winner) - list is already sorted by time
            first_qso = valid_qsos[0]
            first_qso["medal_type"] = "QSO Race"
            medal_qsos.append(first_qso)
            qso_ids_added.add(first_qso["id"])

            # Best cool factor QSO - find the one with highest cool_factor value
            cf_qsos = [q for q in valid_qsos if q["cool_factor"] is not None and q["cool_factor"] > 0]
            if cf_qsos:
                best_cf_qso = max(cf_qsos, key=lambda q: q["cool_factor"])
                if best_cf_qso["id"] not in qso_ids_added:
                    best_cf_qso["medal_type"] = "Cool Factor"
                    medal_qsos.append(best_cf_qso)
                else:
                    # Same QSO wins both - update the label
                    first_qso["medal_type"] = "QSO Race + Cool Factor"

            qso_details[callsign] = medal_qsos

        display_prefs = get_display_prefs(user)
        return templates.TemplateResponse("match.html", {
            "request": request,
            "user": user,
            "match": match_dict,
            "target_display": target_display,
            "target_type": target_type,
            "target_value": match_dict["target_value"],
            "sport_name": sport["name"] if sport else "Unknown",
            "sport_id": sport_id,
            "medals": medals,
            "qso_details": qso_details,
            "display_prefs": display_prefs,
            "is_sport_referee": is_sport_referee,
        })


@app.post("/sport/{sport_id}/enter")
async def enter_sport(sport_id: int, user: User = Depends(require_user)):
    """Opt into a sport."""
    import sqlite3
    with get_db() as conn:
        # Verify sport exists and belongs to an active Olympiad
        cursor = conn.execute("""
            SELECT s.id FROM sports s
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE s.id = ? AND o.is_active = 1
        """, (sport_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Sport not found or Olympiad not active")

        # Check if already entered
        cursor = conn.execute(
            "SELECT id FROM sport_entries WHERE callsign = ? AND sport_id = ?",
            (user.callsign, sport_id)
        )
        if cursor.fetchone():
            return {"message": "Already entered"}

        # Enter the sport
        now = datetime.utcnow().isoformat()
        try:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                (user.callsign, sport_id, now)
            )
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to enter sport {sport_id} for {user.callsign}: {e}")
            raise HTTPException(status_code=500, detail="Database error - please contact support")

    # Recompute medals for this sport
    recompute_sport_matches(sport_id)

    return {"message": "Entered sport successfully"}


@app.post("/sport/{sport_id}/leave")
async def leave_sport(sport_id: int, user: User = Depends(require_user)):
    """Opt out of a sport."""
    with get_db() as conn:
        # Verify sport belongs to an active Olympiad
        cursor = conn.execute("""
            SELECT s.id FROM sports s
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE s.id = ? AND o.is_active = 1
        """, (sport_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Sport not found or Olympiad not active")

        conn.execute(
            "DELETE FROM sport_entries WHERE callsign = ? AND sport_id = ?",
            (user.callsign, sport_id)
        )

    # Recompute medals for this sport
    recompute_sport_matches(sport_id)

    return {"message": "Left sport"}


@app.get("/records", response_class=HTMLResponse)
async def get_records(request: Request, user: Optional[User] = Depends(require_user_or_public)):
    """Get world records page. Records are pre-computed during sync."""
    import json
    from database import get_setting

    with get_db() as conn:
        # Global records (sport_id IS NULL, callsign IS NULL) with holder names and match info
        cursor = conn.execute("""
            SELECT r.record_type, r.value, r.qso_id, r.achieved_at, r.match_id,
                   q.competitor_callsign as holder, q.dx_callsign, q.mode, q.band,
                   q.tx_power_w, q.distance_km, q.cool_factor, q.qso_datetime_utc,
                   q.my_sig_info,
                   c.first_name as holder_first_name,
                   m.target_value,
                   s.id as linked_sport_id, s.name as sport_name
            FROM records r
            LEFT JOIN qsos q ON r.qso_id = q.id
            LEFT JOIN competitors c ON q.competitor_callsign = c.callsign
            LEFT JOIN matches m ON r.match_id = m.id
            LEFT JOIN sports s ON m.sport_id = s.id
            WHERE r.callsign IS NULL AND r.sport_id IS NULL
            ORDER BY r.record_type
        """)
        world_records = [dict(row) for row in cursor.fetchall()]

    # Load pre-computed data from cache
    triathlon_json = get_setting("cache_triathlon_leaders")
    triathlon_leaders = json.loads(triathlon_json) if triathlon_json else []

    mentions_json = get_setting("cache_honorable_mentions")
    honorable_mentions = json.loads(mentions_json) if mentions_json else {}

    distance_json = get_setting("cache_distance_records")
    distance_records = json.loads(distance_json) if distance_json else []

    cf_json = get_setting("cache_cool_factor_records")
    cool_factor_records = json.loads(cf_json) if cf_json else []

    display_prefs = get_display_prefs(user)
    return templates.TemplateResponse("records.html", {
        "request": request,
        "user": user,
        "world_records": world_records,
        "distance_records": distance_records,
        "cool_factor_records": cool_factor_records,
        "triathlon_leaders": triathlon_leaders,
        "honorable_mentions": honorable_mentions,
        "display_prefs": display_prefs,
    })


@app.get("/medals", response_class=HTMLResponse)
async def get_medals_page(request: Request, user: Optional[User] = Depends(require_user_or_public)):
    """Get medal standings page showing all competitors sorted by medal count."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT
                m.callsign,
                c.first_name,
                SUM(CASE WHEN m.qso_race_medal = 'gold' THEN 1 ELSE 0 END) +
                SUM(CASE WHEN m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                SUM(CASE WHEN m.qso_race_medal = 'silver' THEN 1 ELSE 0 END) +
                SUM(CASE WHEN m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                SUM(CASE WHEN m.qso_race_medal = 'bronze' THEN 1 ELSE 0 END) +
                SUM(CASE WHEN m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count,
                SUM(CASE WHEN m.qso_race_medal IS NOT NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN m.cool_factor_medal IS NOT NULL THEN 1 ELSE 0 END) as total_medals,
                SUM(m.total_points) as total_points
            FROM medals m
            JOIN competitors c ON m.callsign = c.callsign
            WHERE m.qualified = 1
            GROUP BY m.callsign
            HAVING total_medals > 0
            ORDER BY total_medals DESC, gold_count DESC, silver_count DESC, bronze_count DESC
        """)
        medal_standings = [dict(row) for row in cursor.fetchall()]

        # Get detailed medal info for each competitor (for expandable rows)
        callsigns = [s["callsign"] for s in medal_standings]
        medal_details = {}
        if callsigns:
            placeholders = ",".join("?" * len(callsigns))
            cursor = conn.execute(f"""
                SELECT m.callsign, m.match_id, m.qso_race_medal, m.cool_factor_medal,
                       m.cool_factor_value, m.pota_bonus, m.total_points,
                       ma.target_value, ma.start_date, ma.end_date,
                       s.id as sport_id, s.name as sport_name, s.target_type,
                       s.work_enabled, s.activate_enabled,
                       COALESCE(ma.allowed_modes, s.allowed_modes) as allowed_modes
                FROM medals m
                JOIN matches ma ON m.match_id = ma.id
                JOIN sports s ON ma.sport_id = s.id
                WHERE m.callsign IN ({placeholders})
                  AND (m.qso_race_medal IS NOT NULL OR m.cool_factor_medal IS NOT NULL)
                ORDER BY ma.start_date DESC
            """, callsigns)
            medal_rows = cursor.fetchall()

            # Get QSOs for these competitors
            from scoring import matches_target, is_mode_allowed
            cursor = conn.execute(f"""
                SELECT q.competitor_callsign, q.dx_callsign, q.qso_datetime_utc,
                       q.mode, q.tx_power_w, q.distance_km, q.cool_factor,
                       q.dx_sig_info, q.my_sig_info, q.dx_dxcc, q.my_dxcc, q.my_grid
                FROM qsos q
                WHERE q.competitor_callsign IN ({placeholders})
                  AND q.is_confirmed = 1
                ORDER BY q.qso_datetime_utc ASC
            """, callsigns)
            all_qsos = [dict(row) for row in cursor.fetchall()]

            # Index QSOs by callsign
            qsos_by_callsign = {}
            for qso in all_qsos:
                cs = qso["competitor_callsign"]
                if cs not in qsos_by_callsign:
                    qsos_by_callsign[cs] = []
                qsos_by_callsign[cs].append(qso)

            # Build medal details with the specific QSOs that earned medals
            for row in medal_rows:
                callsign = row["callsign"]
                if callsign not in medal_details:
                    medal_details[callsign] = []

                # Find QSOs matching this medal's target
                start_date = row["start_date"][:10] if row["start_date"] else None
                end_date = row["end_date"][:10] if row["end_date"] else None
                target_value = row["target_value"]
                target_type = row["target_type"]
                work_enabled = row["work_enabled"]
                activate_enabled = row["activate_enabled"]
                allowed_modes = row["allowed_modes"]

                all_matching = []
                for qso in qsos_by_callsign.get(callsign, []):
                    qso_date = qso["qso_datetime_utc"][:10] if qso["qso_datetime_utc"] else None
                    if qso_date and start_date and end_date and start_date <= qso_date <= end_date:
                        # Check if QSO mode is allowed for this sport/match
                        if not is_mode_allowed(qso.get("mode"), allowed_modes):
                            continue
                        matches_work = work_enabled and matches_target(qso, target_type, target_value, "work")
                        matches_activate = activate_enabled and matches_target(qso, target_type, target_value, "activate")
                        if matches_work or matches_activate:
                            all_matching.append(qso)

                # Only show the QSO(s) that actually earned the medal
                medal_qsos = []
                # QSO Race medal: earliest QSO
                if row["qso_race_medal"] and all_matching:
                    earliest = min(all_matching, key=lambda q: q["qso_datetime_utc"] or "")
                    medal_qsos.append({
                        "dx_callsign": earliest["dx_callsign"],
                        "time": earliest["qso_datetime_utc"],
                        "mode": earliest["mode"],
                        "power": earliest["tx_power_w"],
                        "distance": earliest["distance_km"],
                        "cool_factor": earliest["cool_factor"],
                        "medal_type": "QSO Race",
                        "my_park": earliest.get("my_sig_info"),  # Activation indicator
                        "my_grid": earliest.get("my_grid"),  # Competitor's grid for local time
                    })
                # Cool Factor medal: highest cool_factor QSO
                if row["cool_factor_medal"] and all_matching:
                    best_cf = max((q for q in all_matching if q.get("cool_factor")),
                                  key=lambda q: q["cool_factor"] or 0, default=None)
                    if best_cf:
                        medal_qsos.append({
                            "dx_callsign": best_cf["dx_callsign"],
                            "time": best_cf["qso_datetime_utc"],
                            "mode": best_cf["mode"],
                            "power": best_cf["tx_power_w"],
                            "distance": best_cf["distance_km"],
                            "cool_factor": best_cf["cool_factor"],
                            "medal_type": "Cool Factor",
                            "my_park": best_cf.get("my_sig_info"),  # Activation indicator
                            "my_grid": best_cf.get("my_grid"),  # Competitor's grid for local time
                        })

                medal_details[callsign].append({
                    "match_id": row["match_id"],
                    "sport_id": row["sport_id"],
                    "sport_name": row["sport_name"],
                    "target": format_target_display(target_value, target_type),
                    "qso_medal": row["qso_race_medal"],
                    "cf_medal": row["cool_factor_medal"],
                    "cf_value": row["cool_factor_value"],
                    "points": row["total_points"],
                    "qsos": medal_qsos
                })

        # Get medal standings by sport (grouped by role to match sport page)
        cursor = conn.execute("""
            SELECT
                s.id as sport_id,
                s.name as sport_name,
                m.callsign,
                m.role,
                c.first_name,
                SUM(CASE WHEN m.qso_race_medal = 'gold' THEN 1 ELSE 0 END +
                    CASE WHEN m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                SUM(CASE WHEN m.qso_race_medal = 'silver' THEN 1 ELSE 0 END +
                    CASE WHEN m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                SUM(CASE WHEN m.qso_race_medal = 'bronze' THEN 1 ELSE 0 END +
                    CASE WHEN m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count,
                SUM(m.total_points) as total_points
            FROM medals m
            LEFT JOIN competitors c ON m.callsign = c.callsign
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            GROUP BY s.id, m.callsign, m.role, c.first_name
            ORDER BY s.name, total_points DESC
        """)
        sport_standings_raw = [dict(row) for row in cursor.fetchall()]

        # Group by sport
        sport_standings = {}
        for row in sport_standings_raw:
            sport_id = row["sport_id"]
            if sport_id not in sport_standings:
                sport_standings[sport_id] = {
                    "name": row["sport_name"],
                    "competitors": []
                }
            sport_standings[sport_id]["competitors"].append(row)

    display_prefs = get_display_prefs(user)
    return templates.TemplateResponse("medals.html", {
        "request": request,
        "user": user,
        "medal_standings": medal_standings,
        "medal_details": medal_details,
        "sport_standings": sport_standings,
        "display_prefs": display_prefs,
    })


@app.get("/competitor/{callsign}", response_class=HTMLResponse)
async def get_competitor(
    request: Request,
    callsign: str,
    user: Optional[User] = Depends(require_user_or_public),
    band: Optional[str] = None,
    mode: Optional[str] = None,
    confirmed: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    dx_call: Optional[str] = None,
    distance_min: Optional[str] = None,
    distance_max: Optional[str] = None,
    cf_min: Optional[str] = None,
    cf_max: Optional[str] = None,
    power_min: Optional[str] = None,
    power_max: Optional[str] = None,
    pota: Optional[str] = None,
    page: int = 1
):
    """Get competitor's QSOs, medals, and personal bests."""
    callsign = callsign.upper()
    is_own_profile = user and (callsign == user.callsign)
    per_page = 50

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT callsign, first_name, registered_at, last_sync_at, qrz_api_key_encrypted FROM competitors WHERE callsign = ?",
            (callsign,)
        )
        competitor = cursor.fetchone()

        if not competitor:
            raise HTTPException(status_code=404, detail="Competitor not found")

        competitor_dict = dict(competitor)
        has_qrz_key = bool(competitor_dict.pop("qrz_api_key_encrypted", None))

        # Build QSO query with filters
        base_where = "WHERE competitor_callsign = ?"
        qso_params = [callsign]

        if band:
            base_where += " AND band = ?"
            qso_params.append(band)
        if mode:
            base_where += " AND mode = ?"
            qso_params.append(mode)
        if confirmed and confirmed.isdigit():
            base_where += " AND is_confirmed = ?"
            qso_params.append(int(confirmed))

        # Date range filters
        if date_from:
            base_where += " AND date(qso_datetime_utc) >= ?"
            qso_params.append(date_from)
        if date_to:
            base_where += " AND date(qso_datetime_utc) <= ?"
            qso_params.append(date_to)

        # DX callsign search (case-insensitive, supports partial match)
        if dx_call:
            base_where += " AND UPPER(dx_callsign) LIKE ?"
            qso_params.append(f"%{dx_call.upper()}%")

        # Distance range filters
        if distance_min:
            try:
                base_where += " AND distance_km >= ?"
                qso_params.append(float(distance_min))
            except ValueError:
                pass
        if distance_max:
            try:
                base_where += " AND distance_km <= ?"
                qso_params.append(float(distance_max))
            except ValueError:
                pass

        # Cool Factor range filters
        if cf_min:
            try:
                base_where += " AND cool_factor >= ?"
                qso_params.append(float(cf_min))
            except ValueError:
                pass
        if cf_max:
            try:
                base_where += " AND cool_factor <= ?"
                qso_params.append(float(cf_max))
            except ValueError:
                pass

        # Power range filters
        if power_min:
            try:
                base_where += " AND tx_power_w >= ?"
                qso_params.append(float(power_min))
            except ValueError:
                pass
        if power_max:
            try:
                base_where += " AND tx_power_w <= ?"
                qso_params.append(float(power_max))
            except ValueError:
                pass

        # POTA filter (has park reference)
        if pota == "1":
            base_where += " AND (dx_sig_info IS NOT NULL OR my_sig_info IS NOT NULL)"
        elif pota == "0":
            base_where += " AND dx_sig_info IS NULL AND my_sig_info IS NULL"

        # Get total QSO count for pagination
        cursor = conn.execute(f"SELECT COUNT(*) FROM qsos {base_where}", qso_params)
        total_qsos = cursor.fetchone()[0]
        total_pages = (total_qsos + per_page - 1) // per_page if total_qsos > 0 else 1
        page = max(1, min(page, total_pages))
        offset = (page - 1) * per_page

        # Get QSOs with pagination
        cursor = conn.execute(f"""
            SELECT * FROM qsos
            {base_where}
            ORDER BY qso_datetime_utc DESC
            LIMIT ? OFFSET ?
        """, qso_params + [per_page, offset])
        qsos = [dict(row) for row in cursor.fetchall()]

        # Add country names and disqualification info to QSOs
        for qso in qsos:
            if qso.get("dx_dxcc"):
                qso["dx_country"] = get_country_name(qso["dx_dxcc"])
            # Get disqualification info for this QSO (sport-specific)
            cursor = conn.execute("""
                SELECT dq.sport_id, s.name as sport_name, dq.status
                FROM qso_disqualifications dq
                JOIN sports s ON dq.sport_id = s.id
                WHERE dq.qso_id = ?
            """, (qso["id"],))
            dq_rows = cursor.fetchall()
            qso["disqualifications"] = [dict(r) for r in dq_rows]
            qso["has_disqualified"] = any(d["status"] == "disqualified" for d in qso["disqualifications"])
            qso["has_refuted"] = any(d["status"] == "refuted" for d in qso["disqualifications"])

        # Get medals with qualifying QSO info
        # Use subqueries with MIN(id) to avoid cartesian product when multiple QSOs have same timestamp
        cursor = conn.execute("""
            SELECT m.*, ma.target_value, s.id as sport_id, s.name as sport_name, s.target_type,
                   qr.dx_callsign as qso_race_dx,
                   qc.dx_callsign as cool_factor_dx
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            LEFT JOIN qsos qr ON qr.id = (
                SELECT MIN(id) FROM qsos
                WHERE competitor_callsign = m.callsign
                AND qso_datetime_utc = m.qso_race_claim_time
            )
            LEFT JOIN qsos qc ON qc.id = (
                SELECT MIN(id) FROM qsos
                WHERE competitor_callsign = m.callsign
                AND qso_datetime_utc = m.cool_factor_claim_time
            )
            WHERE m.callsign = ?
            ORDER BY ma.start_date DESC
        """, (callsign,))
        medals = [dict(row) for row in cursor.fetchall()]

        # Add target_display
        for medal in medals:
            medal["target_display"] = format_target_display(medal["target_value"], medal["target_type"])

        # Get personal bests (from competitions)
        cursor = conn.execute("""
            SELECT * FROM records
            WHERE callsign = ?
        """, (callsign,))
        personal_bests = [dict(row) for row in cursor.fetchall()]

        # Get all-time personal bests (from all QSOs, regardless of competition or confirmation)
        cursor = conn.execute("""
            SELECT
                MAX(distance_km) as longest_distance,
                MAX(cool_factor) as highest_cool_factor,
                MIN(CASE WHEN tx_power_w > 0 THEN tx_power_w END) as lowest_power
            FROM qsos
            WHERE competitor_callsign = ? AND distance_km IS NOT NULL
        """, (callsign,))
        row = cursor.fetchone()
        alltime_bests = []
        if row:
            if row["longest_distance"]:
                # Get the QSO details for longest distance
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, distance_km
                    FROM qsos WHERE competitor_callsign = ? AND distance_km = ? LIMIT 1
                """, (callsign, row["longest_distance"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "longest_distance",
                        "value": row["longest_distance"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })
            if row["highest_cool_factor"]:
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, cool_factor
                    FROM qsos WHERE competitor_callsign = ? AND cool_factor = ? LIMIT 1
                """, (callsign, row["highest_cool_factor"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "highest_cool_factor",
                        "value": row["highest_cool_factor"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })
            if row["lowest_power"]:
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, tx_power_w
                    FROM qsos WHERE competitor_callsign = ? AND tx_power_w = ? LIMIT 1
                """, (callsign, row["lowest_power"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "lowest_power",
                        "value": row["lowest_power"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })

        # Get sports the competitor is entered in with points per sport
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name,
                   COALESCE(SUM(m.total_points), 0) as sport_points,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN m.id END) as gold_count,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN m.id END) as silver_count,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN m.id END) as bronze_count
            FROM sport_entries se
            JOIN sports s ON se.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            LEFT JOIN matches ma ON ma.sport_id = s.id
            LEFT JOIN medals m ON m.match_id = ma.id AND m.callsign = ?
            WHERE se.callsign = ?
            GROUP BY s.id, s.name, o.name
            ORDER BY o.name, s.name
        """, (callsign, callsign))
        sport_entries = [dict(row) for row in cursor.fetchall()]

        # Calculate medal summary (count each medal type separately)
        gold = sum((1 if m["qso_race_medal"] == "gold" else 0) + (1 if m["cool_factor_medal"] == "gold" else 0) for m in medals)
        silver = sum((1 if m["qso_race_medal"] == "silver" else 0) + (1 if m["cool_factor_medal"] == "silver" else 0) for m in medals)
        bronze = sum((1 if m["qso_race_medal"] == "bronze" else 0) + (1 if m["cool_factor_medal"] == "bronze" else 0) for m in medals)
        total_points = sum(m["total_points"] for m in medals)

        # Can sync if: viewing own profile with QRZ key, OR admin/referee viewing someone with QRZ key
        can_sync = has_qrz_key and (is_own_profile or (user and (user.is_admin or user.is_referee)))

        # Get logged-in user's display preferences (used for displaying distances/times)
        if user:
            cursor = conn.execute(
                "SELECT distance_unit, time_display, home_grid FROM competitors WHERE callsign = ?",
                (user.callsign,)
            )
            user_prefs = cursor.fetchone()
            if user_prefs:
                # Add display preferences to competitor_dict so template can use them
                competitor_dict["distance_unit"] = user_prefs["distance_unit"]
                competitor_dict["time_display"] = user_prefs["time_display"]
                competitor_dict["home_grid"] = user_prefs["home_grid"]

        # Get team info if viewing own profile
        user_team = None
        team_member_count = 0
        is_team_captain = False
        pending_team_invites = []
        if is_own_profile:
            user_team = get_user_team(conn, callsign)
            if user_team:
                cursor = conn.execute("SELECT COUNT(*) FROM team_members WHERE team_id = ?", (user_team["id"],))
                team_member_count = cursor.fetchone()[0]
                is_team_captain = user_team["captain_callsign"] == callsign
            else:
                # Get pending team invites
                cursor = conn.execute("""
                    SELECT ti.*, t.name as team_name
                    FROM team_invites ti
                    JOIN teams t ON ti.team_id = t.id
                    WHERE ti.callsign = ? AND ti.invite_type = 'invite'
                    ORDER BY ti.created_at DESC
                """, (callsign,))
                pending_team_invites = cursor.fetchall()

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "competitor": competitor_dict,
            "medal_summary": {
                "gold": gold,
                "silver": silver,
                "bronze": bronze,
                "total_points": total_points,
            },
            "qsos": qsos,
            "medals": medals,
            "personal_bests": personal_bests,
            "alltime_bests": alltime_bests,
            "sport_entries": sport_entries,
            "is_own_profile": is_own_profile,
            "can_sync": can_sync,
            "pagination": {
                "page": page,
                "total_pages": total_pages,
                "total_items": total_qsos,
                "per_page": per_page,
            },
            "user_team": user_team,
            "team_member_count": team_member_count,
            "is_team_captain": is_team_captain,
            "pending_team_invites": pending_team_invites,
        })


@app.get("/sync", response_class=HTMLResponse)
async def sync_page(request: Request, callsign: Optional[str] = None):
    """Sync page - shows sync results."""
    if callsign:
        # Single competitor sync runs inline (smaller operation)
        result = await sync_competitor(callsign)
        return templates.TemplateResponse("sync.html", {
            "request": request,
            "result": result,
            "callsign": callsign,
        })
    else:
        # Full sync runs as subprocess to avoid blocking
        asyncio.create_task(run_sync_subprocess())
        return templates.TemplateResponse("sync.html", {
            "request": request,
            "result": {"message": "Full sync started in background"},
            "callsign": None,
        })


@app.post("/sync")
async def trigger_sync(callsign: Optional[str] = None):
    """Trigger QRZ sync (API). Syncs single competitor or all if no callsign provided."""
    from notifications import send_pending_push_disabled_emails, check_pota_spots_and_notify

    if callsign:
        # Single competitor sync runs inline (smaller operation)
        result = await sync_competitor(callsign)
        # Send any pending push disabled notification emails
        await send_pending_push_disabled_emails()
        # Check POTA spots and notify (non-blocking, fire and forget)
        asyncio.create_task(check_pota_spots_and_notify())
        return result
    else:
        # Full sync runs as subprocess to avoid blocking
        asyncio.create_task(run_sync_subprocess())
        return {"message": "Full sync started in background"}


class QRZSyncRequest(BaseModel):
    api_key: str


class LoTWSyncRequest(BaseModel):
    username: str
    password: str


class CreateTeamRequest(BaseModel):
    name: str
    description: Optional[str] = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Team name is required')
        if len(v.strip()) < 2:
            raise ValueError('Team name must be at least 2 characters')
        if len(v.strip()) > 100:
            raise ValueError('Team name must be 100 characters or less')
        return v.strip()


class UpdateTeamRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Team name cannot be empty')
            if len(v.strip()) < 2:
                raise ValueError('Team name must be at least 2 characters')
            if len(v.strip()) > 100:
                raise ValueError('Team name must be 100 characters or less')
            return v.strip()
        return v


@app.post("/sync/qrz")
async def trigger_qrz_sync_with_key(request: Request, sync_data: QRZSyncRequest):
    """Trigger QRZ sync for the current user using provided API key."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await sync_competitor_with_key(user.callsign, sync_data.api_key)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/sync/lotw")
async def trigger_lotw_sync(request: Request, sync_data: LoTWSyncRequest):
    """Trigger LoTW sync for the current user using provided credentials."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await sync_competitor_lotw(user.callsign, sync_data.username, sync_data.password)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/sync/lotw/stored")
async def trigger_lotw_sync_stored(request: Request):
    """Trigger LoTW sync for the current user using stored credentials."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await sync_competitor_lotw_stored(user.callsign)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ============================================================
# QSO RESET ENDPOINTS
# ============================================================


class QSOResetRequest(BaseModel):
    """Request body for QSO reset with provided credentials."""
    qrz_api_key: Optional[str] = None
    lotw_username: Optional[str] = None
    lotw_password: Optional[str] = None


@app.post("/qsos/reset/preflight")
async def qso_reset_preflight(request: Request):
    """
    Check what credentials a competitor has stored.

    Returns whether they can reset (have at least one credential source).
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    with get_db() as conn:
        cursor = conn.execute(
            """SELECT qrz_api_key_encrypted, lotw_username_encrypted, lotw_password_encrypted
               FROM competitors WHERE callsign = ?""",
            (user.callsign,)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Competitor not found")

    has_qrz = bool(row["qrz_api_key_encrypted"])
    has_lotw = bool(row["lotw_username_encrypted"] and row["lotw_password_encrypted"])

    return {
        "has_qrz": has_qrz,
        "has_lotw": has_lotw,
        "can_reset": has_qrz or has_lotw
    }


@app.post("/qsos/reset")
async def qso_reset(request: Request):
    """
    Reset QSOs for the logged-in competitor using stored credentials.

    Deletes all QSOs and medals, then triggers a full sync.
    Requires at least one stored credential (QRZ or LoTW).
    """
    from sync import delete_competitor_qsos

    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Check stored credentials
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT qrz_api_key_encrypted, lotw_username_encrypted, lotw_password_encrypted
               FROM competitors WHERE callsign = ?""",
            (user.callsign,)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Competitor not found")

    has_qrz = bool(row["qrz_api_key_encrypted"])
    has_lotw = bool(row["lotw_username_encrypted"] and row["lotw_password_encrypted"])

    if not has_qrz and not has_lotw:
        raise HTTPException(
            status_code=400,
            detail="No stored credentials. Please provide QRZ API key or LoTW credentials."
        )

    # Delete all QSOs and medals
    deleted_count = delete_competitor_qsos(user.callsign)

    # Sync from stored credentials
    results = {"deleted_qsos": deleted_count}

    if has_qrz:
        qrz_result = await sync_competitor(user.callsign)
        results["qrz_sync"] = qrz_result

    if has_lotw:
        lotw_result = await sync_competitor_lotw_stored(user.callsign)
        results["lotw_sync"] = lotw_result

    return results


@app.post("/qsos/reset-with-key")
async def qso_reset_with_key(request: Request, reset_data: QSOResetRequest):
    """
    Reset QSOs with provided credentials (when none are stored).

    Validates credentials, stores them encrypted, deletes QSOs, and triggers full sync.
    At least one credential source (QRZ key OR LoTW username+password) is required.
    """
    from sync import delete_competitor_qsos

    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    has_qrz = reset_data.qrz_api_key and reset_data.qrz_api_key.strip()
    has_lotw = reset_data.lotw_username and reset_data.lotw_password

    if not has_qrz and not has_lotw:
        raise HTTPException(
            status_code=400,
            detail="Please provide QRZ API key and/or LoTW credentials"
        )

    # Validate credentials before deleting anything
    if has_qrz:
        is_valid = await verify_api_key(reset_data.qrz_api_key, user.callsign)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid QRZ API key. Please verify your API key is correct."
            )

    if has_lotw:
        is_valid = await verify_lotw_credentials(
            reset_data.lotw_username, reset_data.lotw_password, user.callsign
        )
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid LoTW credentials or username does not match callsign."
            )

    # Store credentials
    with get_db() as conn:
        if has_qrz:
            encrypted_key = encrypt_api_key(reset_data.qrz_api_key)
            conn.execute(
                "UPDATE competitors SET qrz_api_key_encrypted = ? WHERE callsign = ?",
                (encrypted_key, user.callsign)
            )
        if has_lotw:
            encrypted_username = encrypt_api_key(reset_data.lotw_username)
            encrypted_password = encrypt_api_key(reset_data.lotw_password)
            conn.execute(
                """UPDATE competitors SET
                   lotw_username_encrypted = ?, lotw_password_encrypted = ?
                   WHERE callsign = ?""",
                (encrypted_username, encrypted_password, user.callsign)
            )
        conn.commit()

    # Delete all QSOs and medals
    deleted_count = delete_competitor_qsos(user.callsign)

    # Sync from provided credentials
    results = {"deleted_qsos": deleted_count}

    if has_qrz:
        qrz_result = await sync_competitor_with_key(user.callsign, reset_data.qrz_api_key)
        results["qrz_sync"] = qrz_result

    if has_lotw:
        lotw_result = await sync_competitor_lotw(
            user.callsign, reset_data.lotw_username, reset_data.lotw_password
        )
        results["lotw_sync"] = lotw_result

    return results


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Signup page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
@limiter.limit("5/minute")
async def signup(request: Request, signup_data: UserSignup):
    """Create a new user account."""
    callsign = signup_data.callsign.upper()

    # Must provide either QRZ API key or LoTW credentials
    has_qrz = signup_data.qrz_api_key and signup_data.qrz_api_key.strip()
    has_lotw = signup_data.lotw_username and signup_data.lotw_password

    if not has_qrz and not has_lotw:
        raise HTTPException(status_code=400, detail="Please provide QRZ API key and/or LoTW credentials")

    encrypted_qrz_key = None
    encrypted_lotw_username = None
    encrypted_lotw_password = None

    # Verify QRZ API key if provided
    if has_qrz:
        is_valid = await verify_api_key(signup_data.qrz_api_key, callsign)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid QRZ API key. Please verify your API key is correct and your QRZ XML subscription is active. "
                       "You can find your API key in QRZ.com -> My Logbook -> Settings -> API tab."
            )
        # Store if user wants auto-sync
        if signup_data.store_credentials:
            encrypted_qrz_key = encrypt_api_key(signup_data.qrz_api_key)

    # Verify LoTW credentials if provided
    if has_lotw:
        is_valid = await verify_lotw_credentials(signup_data.lotw_username, signup_data.lotw_password, callsign)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid LoTW credentials or username does not match callsign.")
        # Store if user wants auto-sync
        if signup_data.store_credentials:
            encrypted_lotw_username = encrypt_api_key(signup_data.lotw_username)
            encrypted_lotw_password = encrypt_api_key(signup_data.lotw_password)

    success = register_user(
        callsign,
        signup_data.password,
        signup_data.email,
        encrypted_qrz_key,
        encrypted_lotw_username,
        encrypted_lotw_password
    )
    if not success:
        raise HTTPException(status_code=400, detail="Callsign already registered")

    # Initial sync using provided credentials (even if not stored)
    if has_qrz:
        try:
            await sync_competitor_with_key(callsign, signup_data.qrz_api_key)
        except Exception:
            pass  # Don't fail signup if sync fails

    if has_lotw:
        try:
            await sync_competitor_lotw(callsign, signup_data.lotw_username, signup_data.lotw_password)
        except Exception:
            pass  # Don't fail signup if sync fails

    # Send welcome email if email was provided
    if signup_data.email:
        from email_service import send_welcome_email, create_email_verification_token, send_email_verification
        try:
            await send_welcome_email(callsign, signup_data.email)
            # Also send verification email
            token = create_email_verification_token(callsign)
            base_url = str(request.base_url).rstrip("/")
            verification_url = f"{base_url}/verify-email/{token}"
            await send_email_verification(callsign, signup_data.email, verification_url)
        except Exception:
            pass  # Don't fail signup if email fails

    # Send Discord notification for new signup
    try:
        from notifications import discord_notify_signup
        discord_notify_signup(callsign)
    except Exception:
        pass  # Don't fail signup if Discord notification fails

    # Auto-login after signup
    session_id = authenticate_user(callsign, signup_data.password)

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax",
        secure=config.SECURE_COOKIES
    )
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, login_data: UserLogin, background_tasks: BackgroundTasks):
    """Authenticate user and create session."""
    from audit import log_action

    result = authenticate_user(login_data.callsign, login_data.password)

    if result == "disabled":
        raise HTTPException(status_code=403, detail="Account has been disabled")

    if result == "locked":
        raise HTTPException(status_code=423, detail="Account is temporarily locked due to too many failed login attempts. Please try again later.")

    if not result:
        raise HTTPException(status_code=401, detail="Invalid callsign or password")

    # Log successful login
    log_action(
        actor_callsign=login_data.callsign.upper(),
        action="login",
        target_type="session",
        target_id=login_data.callsign.upper(),
        ip_address=get_client_ip(request)
    )

    # Trigger background sync for the user's QSOs
    callsign = login_data.callsign.upper()
    background_tasks.add_task(sync_competitor, callsign)
    background_tasks.add_task(sync_competitor_lotw_stored, callsign)

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=result,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax",
        secure=config.SECURE_COOKIES
    )
    # Rotate CSRF token on login to prevent session fixation
    new_csrf = generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=new_csrf,
        httponly=True,
        samesite="lax",
        secure=config.SECURE_COOKIES,
        max_age=30 * 24 * 60 * 60
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    """Logout and delete session."""
    from audit import log_action

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_current_user(request)

    if session_id:
        delete_session(session_id)

    # Log logout
    if user:
        log_action(
            actor_callsign=user.callsign,
            action="logout",
            target_type="session",
            target_id=user.callsign,
            ip_address=get_client_ip(request)
        )

    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    # Rotate CSRF token on logout
    new_csrf = generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=new_csrf,
        httponly=True,
        samesite="lax",
        secure=config.SECURE_COOKIES,
        max_age=30 * 24 * 60 * 60
    )
    return response


# ============================================================
# PASSWORD RESET
# ============================================================

@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    """Forgot password page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("forgot_password.html", {
        "request": request,
        "csrf_token": get_csrf_token(request)
    })


class ForgotPasswordForm(BaseModel):
    callsign: str


@app.post("/forgot-password")
@limiter.limit("3/minute")
async def forgot_password(request: Request, form_data: ForgotPasswordForm):
    """Handle forgot password request."""
    callsign = form_data.callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT email FROM competitors WHERE callsign = ?",
            (callsign,)
        )
        row = cursor.fetchone()

        if not row:
            # Don't reveal if user exists
            return templates.TemplateResponse("forgot_password.html", {
                "request": request,
                "csrf_token": get_csrf_token(request),
                "message": "If an account exists with that callsign, a password reset email will be sent."
            })

        email = row["email"]
        if not email:
            raise HTTPException(
                status_code=400,
                detail="No email address on file for this account. Please contact an administrator."
            )

        # Create reset token
        token = create_password_reset_token(callsign)
        reset_url = f"{request.base_url}reset-password/{token}"

        # Send email
        await send_password_reset_email(callsign, email, reset_url)

        return templates.TemplateResponse("forgot_password.html", {
            "request": request,
            "csrf_token": get_csrf_token(request),
            "message": "If an account exists with that callsign, a password reset email will be sent."
        })


@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str):
    """Reset password page."""
    callsign = validate_reset_token(token)
    if not callsign:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": token,
        "callsign": callsign,
        "csrf_token": get_csrf_token(request)
    })


@app.post("/reset-password/{token}")
@limiter.limit("5/minute")
async def reset_password(
    request: Request,
    token: str,
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Handle password reset."""
    callsign = validate_reset_token(token)
    if not callsign:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if len(password) < config.PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {config.PASSWORD_MIN_LENGTH} characters"
        )

    # Update password
    update_user_password(callsign, password)

    # Mark token as used
    mark_token_used(token)

    # Redirect to login with success message
    return RedirectResponse(url="/login?reset=success", status_code=303)


@app.get("/verify-email/{token}")
async def verify_email(request: Request, token: str):
    """Verify user's email address."""
    from email_service import validate_email_verification_token, mark_email_verification_token_used

    callsign = validate_email_verification_token(token)
    if not callsign:
        return templates.TemplateResponse("message.html", {
            "request": request,
            "title": "Verification Failed",
            "message": "Invalid or expired verification link. Please request a new verification email from your settings.",
            "type": "error"
        })

    # Mark email as verified
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET email_verified = 1 WHERE callsign = ?",
            (callsign,)
        )

    # Mark token as used
    mark_email_verification_token_used(token)

    return templates.TemplateResponse("message.html", {
        "request": request,
        "title": "Email Verified",
        "message": "Your email address has been verified successfully!",
        "type": "success"
    })


@app.post("/settings/resend-verification")
@limiter.limit("3/minute")
async def resend_verification(request: Request, user: User = Depends(require_user)):
    """Resend email verification."""
    from email_service import create_email_verification_token, send_email_verification

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT email, email_verified FROM competitors WHERE callsign = ?",
            (user.callsign,)
        )
        row = cursor.fetchone()

        if not row or not row["email"]:
            return RedirectResponse(
                url="/settings?error=Please+add+an+email+address+first",
                status_code=303
            )

        if row["email_verified"]:
            return RedirectResponse(
                url="/settings?error=Your+email+is+already+verified",
                status_code=303
            )

        # Create and send verification email
        token = create_email_verification_token(user.callsign)
        base_url = str(request.base_url).rstrip("/")
        verification_url = f"{base_url}/verify-email/{token}"

        await send_email_verification(user.callsign, row["email"], verification_url)

    return RedirectResponse(
        url="/settings?updated=verification_sent",
        status_code=303
    )


# ============================================================
# USER DASHBOARD & SETTINGS
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    user: User = Depends(require_user),
    band: Optional[str] = None,
    mode: Optional[str] = None,
    confirmed: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    dx_call: Optional[str] = None,
    distance_min: Optional[str] = None,
    distance_max: Optional[str] = None,
    cf_min: Optional[str] = None,
    cf_max: Optional[str] = None,
    power_min: Optional[str] = None,
    power_max: Optional[str] = None,
    pota: Optional[str] = None,
    page: int = 1
):
    """User dashboard with stats, QSOs, and medals."""
    per_page = 50
    offset = (page - 1) * per_page

    with get_db() as conn:
        # Build QSO query with filters
        base_where = "WHERE competitor_callsign = ?"
        qso_params = [user.callsign]

        if band:
            base_where += " AND band = ?"
            qso_params.append(band)
        if mode:
            base_where += " AND mode = ?"
            qso_params.append(mode)
        if confirmed and confirmed.isdigit():
            base_where += " AND is_confirmed = ?"
            qso_params.append(int(confirmed))

        # Date range filters
        if date_from:
            base_where += " AND date(qso_datetime_utc) >= ?"
            qso_params.append(date_from)
        if date_to:
            base_where += " AND date(qso_datetime_utc) <= ?"
            qso_params.append(date_to)

        # DX callsign search (case-insensitive, supports partial match)
        if dx_call:
            base_where += " AND UPPER(dx_callsign) LIKE ?"
            qso_params.append(f"%{dx_call.upper()}%")

        # Distance range filters
        if distance_min:
            try:
                base_where += " AND distance_km >= ?"
                qso_params.append(float(distance_min))
            except ValueError:
                pass
        if distance_max:
            try:
                base_where += " AND distance_km <= ?"
                qso_params.append(float(distance_max))
            except ValueError:
                pass

        # Cool Factor range filters
        if cf_min:
            try:
                base_where += " AND cool_factor >= ?"
                qso_params.append(float(cf_min))
            except ValueError:
                pass
        if cf_max:
            try:
                base_where += " AND cool_factor <= ?"
                qso_params.append(float(cf_max))
            except ValueError:
                pass

        # Power range filters
        if power_min:
            try:
                base_where += " AND tx_power_w >= ?"
                qso_params.append(float(power_min))
            except ValueError:
                pass
        if power_max:
            try:
                base_where += " AND tx_power_w <= ?"
                qso_params.append(float(power_max))
            except ValueError:
                pass

        # POTA filter (has park reference)
        if pota == "1":
            base_where += " AND (dx_sig_info IS NOT NULL OR my_sig_info IS NOT NULL)"
        elif pota == "0":
            base_where += " AND dx_sig_info IS NULL AND my_sig_info IS NULL"

        # Get total count for pagination
        count_query = f"SELECT COUNT(*) FROM qsos {base_where}"
        cursor = conn.execute(count_query, qso_params)
        total_qsos = cursor.fetchone()[0]
        total_pages = (total_qsos + per_page - 1) // per_page if total_qsos > 0 else 1

        # Ensure page is within bounds
        page = max(1, min(page, total_pages))

        qso_query = f"SELECT * FROM qsos {base_where} ORDER BY qso_datetime_utc DESC LIMIT ? OFFSET ?"
        cursor = conn.execute(qso_query, qso_params + [per_page, offset])
        qsos = [dict(row) for row in cursor.fetchall()]

        # Add country names to QSOs
        for qso in qsos:
            if qso.get("dx_dxcc"):
                qso["dx_country"] = get_country_name(qso["dx_dxcc"])

        # Get medals
        cursor = conn.execute("""
            SELECT m.*, ma.target_value, s.id as sport_id, s.name as sport_name, s.target_type
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            WHERE m.callsign = ?
            ORDER BY ma.start_date DESC
        """, (user.callsign,))
        medals = [dict(row) for row in cursor.fetchall()]

        # Add target_display
        for medal in medals:
            medal["target_display"] = format_target_display(medal["target_value"], medal["target_type"])

        # Get personal bests
        cursor = conn.execute("""
            SELECT * FROM records WHERE callsign = ?
        """, (user.callsign,))
        personal_bests = [dict(row) for row in cursor.fetchall()]

        # Get all-time personal bests (from all QSOs, regardless of competition or confirmation)
        cursor = conn.execute("""
            SELECT
                MAX(distance_km) as longest_distance,
                MAX(cool_factor) as highest_cool_factor,
                MIN(CASE WHEN tx_power_w > 0 THEN tx_power_w END) as lowest_power
            FROM qsos
            WHERE competitor_callsign = ? AND distance_km IS NOT NULL
        """, (user.callsign,))
        row = cursor.fetchone()
        alltime_bests = []
        if row:
            if row["longest_distance"]:
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, distance_km
                    FROM qsos WHERE competitor_callsign = ? AND distance_km = ? LIMIT 1
                """, (user.callsign, row["longest_distance"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "longest_distance",
                        "value": row["longest_distance"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })
            if row["highest_cool_factor"]:
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, cool_factor
                    FROM qsos WHERE competitor_callsign = ? AND cool_factor = ? LIMIT 1
                """, (user.callsign, row["highest_cool_factor"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "highest_cool_factor",
                        "value": row["highest_cool_factor"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })
            if row["lowest_power"]:
                cursor = conn.execute("""
                    SELECT dx_callsign, qso_datetime_utc, tx_power_w
                    FROM qsos WHERE competitor_callsign = ? AND tx_power_w = ? LIMIT 1
                """, (user.callsign, row["lowest_power"]))
                qso = cursor.fetchone()
                if qso:
                    alltime_bests.append({
                        "record_type": "lowest_power",
                        "value": row["lowest_power"],
                        "achieved_at": qso["qso_datetime_utc"],
                        "dx_callsign": qso["dx_callsign"]
                    })

        # Calculate medal summary (count each medal type separately)
        gold = sum((1 if m["qso_race_medal"] == "gold" else 0) + (1 if m["cool_factor_medal"] == "gold" else 0) for m in medals)
        silver = sum((1 if m["qso_race_medal"] == "silver" else 0) + (1 if m["cool_factor_medal"] == "silver" else 0) for m in medals)
        bronze = sum((1 if m["qso_race_medal"] == "bronze" else 0) + (1 if m["cool_factor_medal"] == "bronze" else 0) for m in medals)
        total_points = sum(m["total_points"] for m in medals)

        # Get competitor info including display preferences
        cursor = conn.execute(
            """SELECT first_name, registered_at, last_sync_at,
                      distance_unit, time_display, home_grid
               FROM competitors WHERE callsign = ?""",
            (user.callsign,)
        )
        competitor_info = dict(cursor.fetchone())

        # Get sports the competitor is entered in with points per sport
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name,
                   COALESCE(SUM(m.total_points), 0) as sport_points,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN m.id END) as gold_count,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN m.id END) as silver_count,
                   COUNT(DISTINCT CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN m.id END) as bronze_count
            FROM sport_entries se
            JOIN sports s ON se.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            LEFT JOIN matches ma ON ma.sport_id = s.id
            LEFT JOIN medals m ON m.match_id = ma.id AND m.callsign = ?
            WHERE se.callsign = ?
            GROUP BY s.id, s.name, o.name
            ORDER BY o.name, s.name
        """, (user.callsign, user.callsign))
        sport_entries = [dict(row) for row in cursor.fetchall()]

        # Get user's team if any
        user_team = get_user_team(conn, user.callsign)
        team_member_count = 0
        is_team_captain = False
        pending_team_invites = []
        if user_team:
            cursor = conn.execute("SELECT COUNT(*) FROM team_members WHERE team_id = ?", (user_team["id"],))
            team_member_count = cursor.fetchone()[0]
            is_team_captain = user_team["captain_callsign"] == user.callsign
        else:
            # Get pending team invites for user
            cursor = conn.execute("""
                SELECT ti.*, t.name as team_name
                FROM team_invites ti
                JOIN teams t ON ti.team_id = t.id
                WHERE ti.callsign = ? AND ti.invite_type = 'invite'
                ORDER BY ti.created_at DESC
            """, (user.callsign,))
            pending_team_invites = cursor.fetchall()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "competitor": {"callsign": user.callsign, **competitor_info},
        "qsos": qsos,
        "medals": medals,
        "personal_bests": personal_bests,
        "alltime_bests": alltime_bests,
        "sport_entries": sport_entries,
        "medal_summary": {
            "gold": gold,
            "silver": silver,
            "bronze": bronze,
            "total_points": total_points,
        },
        "is_own_profile": True,
        "can_sync": user.has_qrz_key,
        "pagination": {
            "page": page,
            "total_pages": total_pages,
            "total_items": total_qsos,
            "per_page": per_page,
        },
        "user_team": user_team,
        "team_member_count": team_member_count,
        "is_team_captain": is_team_captain,
        "pending_team_invites": pending_team_invites,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(require_user)):
    """Account settings page."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT email, email_verified, registered_at, first_name, last_name,
                      email_notifications_enabled, email_medal_notifications,
                      email_match_reminders, email_record_notifications,
                      distance_unit, time_display, home_grid
               FROM competitors WHERE callsign = ?""",
            (user.callsign,)
        )
        account = dict(cursor.fetchone())

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "account": account,
        "csrf_token": get_csrf_token(request),
    })


@app.post("/settings/name")
async def update_name(
    request: Request,
    first_name: str = Form(""),
    last_name: str = Form(""),
    user: User = Depends(require_user)
):
    """Update user name."""
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET first_name = ?, last_name = ? WHERE callsign = ?",
            (first_name.strip() or None, last_name.strip() or None, user.callsign)
        )
    return RedirectResponse(url="/settings?updated=name", status_code=303)


class EmailUpdate(BaseModel):
    email: EmailStr


@app.post("/settings/email")
async def update_email(request: Request, email: str = Form(...), user: User = Depends(require_user)):
    """Update user email."""
    # Validate email format
    try:
        validated = EmailUpdate(email=email)
    except Exception:
        return templates.TemplateResponse(
            "settings.html",
            {"request": request, "user": user, "error": "Invalid email format"}
        )
    update_user_email(user.callsign, validated.email)
    return RedirectResponse(url="/settings?updated=email", status_code=303)


@app.post("/settings/password")
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    user: User = Depends(require_user)
):
    """Change user password."""
    from audit import log_action
    from urllib.parse import quote

    # Verify current password
    session_id = authenticate_user(user.callsign, current_password)
    if not session_id or session_id in ('disabled', 'locked'):
        return RedirectResponse(
            url="/settings?error=" + quote("Current password is incorrect"),
            status_code=303
        )

    if len(new_password) < 8:
        return RedirectResponse(
            url="/settings?error=" + quote("New password must be at least 8 characters"),
            status_code=303
        )

    update_user_password(user.callsign, new_password)

    # Log password change
    log_action(
        actor_callsign=user.callsign,
        action="password_change",
        target_type="competitor",
        target_id=user.callsign,
        ip_address=get_client_ip(request)
    )

    return RedirectResponse(url="/settings?updated=password", status_code=303)


@app.post("/settings/qrz-key")
async def update_qrz_key(
    request: Request,
    qrz_api_key: str = Form(...),
    user: User = Depends(require_user)
):
    """Update QRZ API key."""
    encrypted_key = encrypt_api_key(qrz_api_key)
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET qrz_api_key_encrypted = ? WHERE callsign = ?",
            (encrypted_key, user.callsign)
        )
    return RedirectResponse(url="/settings?updated=qrz", status_code=303)


@app.delete("/settings/qrz-key")
async def remove_qrz_key(request: Request, user: User = Depends(require_user)):
    """Remove stored QRZ API key."""
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET qrz_api_key_encrypted = NULL WHERE callsign = ?",
            (user.callsign,)
        )
    return {"message": "QRZ API key removed"}


@app.post("/settings/lotw")
async def update_lotw_credentials(
    request: Request,
    lotw_username: str = Form(...),
    lotw_password: str = Form(...),
    user: User = Depends(require_user)
):
    """Update LoTW credentials."""
    encrypted_username = encrypt_api_key(lotw_username)
    encrypted_password = encrypt_api_key(lotw_password)
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET lotw_username_encrypted = ?, lotw_password_encrypted = ? WHERE callsign = ?",
            (encrypted_username, encrypted_password, user.callsign)
        )
    return RedirectResponse(url="/settings?updated=lotw", status_code=303)


@app.delete("/settings/lotw")
async def remove_lotw_credentials(request: Request, user: User = Depends(require_user)):
    """Remove stored LoTW credentials."""
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET lotw_username_encrypted = NULL, lotw_password_encrypted = NULL WHERE callsign = ?",
            (user.callsign,)
        )
    return {"message": "LoTW credentials removed"}


@app.post("/settings/notifications")
async def update_notification_preferences(
    request: Request,
    email_notifications_enabled: bool = Form(False),
    email_medal_notifications: bool = Form(False),
    email_match_reminders: bool = Form(False),
    email_record_notifications: bool = Form(False),
    user: User = Depends(require_user)
):
    """Update email notification preferences."""
    with get_db() as conn:
        conn.execute(
            """UPDATE competitors SET
               email_notifications_enabled = ?,
               email_medal_notifications = ?,
               email_match_reminders = ?,
               email_record_notifications = ?
               WHERE callsign = ?""",
            (
                1 if email_notifications_enabled else 0,
                1 if email_medal_notifications else 0,
                1 if email_match_reminders else 0,
                1 if email_record_notifications else 0,
                user.callsign
            )
        )
    return RedirectResponse(url="/settings?updated=notifications", status_code=303)


@app.post("/settings/display")
async def update_display_preferences(
    request: Request,
    distance_unit: str = Form("km"),
    time_display: str = Form("utc"),
    home_grid: str = Form(""),
    user: User = Depends(require_user)
):
    """Update display preferences (distance units, time format, home grid)."""
    # Validate inputs
    if distance_unit not in ("km", "mi"):
        distance_unit = "km"
    if time_display not in ("utc", "local"):
        time_display = "utc"
    # Validate and normalize home_grid
    home_grid = home_grid.strip().upper() if home_grid else None
    if home_grid and (len(home_grid) < 4 or len(home_grid) > 6):
        home_grid = None  # Invalid grid, clear it

    with get_db() as conn:
        conn.execute(
            """UPDATE competitors SET
               distance_unit = ?,
               time_display = ?,
               home_grid = ?
               WHERE callsign = ?""",
            (distance_unit, time_display, home_grid, user.callsign)
        )
    return RedirectResponse(url="/settings?updated=display", status_code=303)


# ============================================================
# PUSH NOTIFICATION API
# ============================================================

@app.get("/push/vapid-public-key")
async def get_vapid_public_key():
    """Get the VAPID public key for push subscription."""
    from notifications import get_vapid_public_key, is_push_configured

    if not is_push_configured():
        raise HTTPException(status_code=503, detail="Push notifications not configured")

    return {"publicKey": get_vapid_public_key()}


@app.post("/push/subscribe")
async def subscribe_push(
    request: Request,
    user: User = Depends(require_user)
):
    """Subscribe to push notifications."""
    from notifications import save_subscription, is_push_configured

    if not is_push_configured():
        raise HTTPException(status_code=503, detail="Push notifications not configured")

    try:
        data = await request.json()
        endpoint = data.get("endpoint")
        keys = data.get("keys", {})
        p256dh = keys.get("p256dh")
        auth = keys.get("auth")

        if not endpoint or not p256dh or not auth:
            raise HTTPException(status_code=400, detail="Invalid subscription data")

        user_agent = request.headers.get("User-Agent")

        if save_subscription(user.callsign, endpoint, p256dh, auth, user_agent):
            return {"message": "Subscribed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save subscription")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")


@app.delete("/push/unsubscribe")
async def unsubscribe_push(
    request: Request,
    user: User = Depends(require_user)
):
    """Unsubscribe from push notifications."""
    from notifications import remove_subscription

    try:
        data = await request.json()
        endpoint = data.get("endpoint")

        if not endpoint:
            raise HTTPException(status_code=400, detail="Endpoint required")

        remove_subscription(endpoint)
        return {"message": "Unsubscribed successfully"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")


@app.delete("/push/unsubscribe-all")
async def unsubscribe_all_push(user: User = Depends(require_user)):
    """Remove all push subscriptions for the current user."""
    from notifications import remove_all_subscriptions

    remove_all_subscriptions(user.callsign)
    return {"message": "All subscriptions removed"}


@app.get("/push/preferences")
async def get_push_preferences(user: User = Depends(require_user)):
    """Get push notification preferences."""
    from notifications import get_notification_preferences, get_subscriptions

    prefs = get_notification_preferences(user.callsign)
    subscriptions = get_subscriptions(user.callsign)

    return {
        "preferences": prefs,
        "subscription_count": len(subscriptions)
    }


@app.put("/push/preferences")
async def update_push_preferences(
    request: Request,
    user: User = Depends(require_user)
):
    """Update push notification preferences."""
    from notifications import update_notification_preferences

    try:
        data = await request.json()
        preferences = {
            "medal_changes": data.get("medal_changes", True),
            "new_confirmations": data.get("new_confirmations", True),
            "record_broken": data.get("record_broken", True),
            "match_reminders": data.get("match_reminders", True)
        }

        if update_notification_preferences(user.callsign, preferences):
            return {"message": "Preferences updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")


@app.post("/push/test")
async def send_test_notification(user: User = Depends(require_user)):
    """Send a test push notification to the current user."""
    from notifications import send_notification, NotificationPayload, get_subscriptions, send_pending_push_disabled_emails

    subscriptions = get_subscriptions(user.callsign)
    if not subscriptions:
        raise HTTPException(status_code=400, detail="No push subscriptions found")

    payload = NotificationPayload(
        title="Test Notification",
        body="Push notifications are working!",
        tag="test",
        url="/settings"
    )

    count = send_notification(user.callsign, payload)

    # Send any pending push disabled notification emails
    await send_pending_push_disabled_emails()

    return {"message": f"Test notification sent to {count} device(s)"}


# ============================================================
# QSO FILTER API (for AJAX filtering)
# ============================================================

@app.get("/api/qsos")
async def get_filtered_qsos(
    request: Request,
    user: Optional[User] = Depends(require_user_or_public),
    callsign: Optional[str] = None,
    band: Optional[str] = None,
    mode: Optional[str] = None,
    confirmed: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    dx_call: Optional[str] = None,
    distance_min: Optional[str] = None,
    distance_max: Optional[str] = None,
    cf_min: Optional[str] = None,
    cf_max: Optional[str] = None,
    power_min: Optional[str] = None,
    power_max: Optional[str] = None,
    pota: Optional[str] = None,
    page: int = 1
):
    """Get filtered QSOs as JSON for AJAX requests."""
    # Use provided callsign or default to logged-in user
    if not callsign and not user:
        raise HTTPException(status_code=400, detail="Callsign required")
    target_callsign = (callsign or user.callsign).upper()
    per_page = 50

    with get_db() as conn:
        # Build QSO query with filters
        base_where = "WHERE competitor_callsign = ?"
        qso_params = [target_callsign]

        if band:
            base_where += " AND band = ?"
            qso_params.append(band)
        if mode:
            base_where += " AND mode = ?"
            qso_params.append(mode)
        if confirmed and confirmed.isdigit():
            base_where += " AND is_confirmed = ?"
            qso_params.append(int(confirmed))

        # Date range filters
        if date_from:
            base_where += " AND date(qso_datetime_utc) >= ?"
            qso_params.append(date_from)
        if date_to:
            base_where += " AND date(qso_datetime_utc) <= ?"
            qso_params.append(date_to)

        # DX callsign search
        if dx_call:
            base_where += " AND UPPER(dx_callsign) LIKE ?"
            qso_params.append(f"%{dx_call.upper()}%")

        # Distance range filters
        if distance_min:
            try:
                base_where += " AND distance_km >= ?"
                qso_params.append(float(distance_min))
            except ValueError:
                pass
        if distance_max:
            try:
                base_where += " AND distance_km <= ?"
                qso_params.append(float(distance_max))
            except ValueError:
                pass

        # Cool Factor range filters
        if cf_min:
            try:
                base_where += " AND cool_factor >= ?"
                qso_params.append(float(cf_min))
            except ValueError:
                pass
        if cf_max:
            try:
                base_where += " AND cool_factor <= ?"
                qso_params.append(float(cf_max))
            except ValueError:
                pass

        # Power range filters
        if power_min:
            try:
                base_where += " AND tx_power_w >= ?"
                qso_params.append(float(power_min))
            except ValueError:
                pass
        if power_max:
            try:
                base_where += " AND tx_power_w <= ?"
                qso_params.append(float(power_max))
            except ValueError:
                pass

        # POTA filter
        if pota == "1":
            base_where += " AND (dx_sig_info IS NOT NULL OR my_sig_info IS NOT NULL)"
        elif pota == "0":
            base_where += " AND dx_sig_info IS NULL AND my_sig_info IS NULL"

        # Get competitor display preferences for formatting
        prefs_cursor = conn.execute("""
            SELECT home_grid, time_display, distance_unit
            FROM competitors WHERE callsign = ?
        """, (target_callsign,))
        prefs_row = prefs_cursor.fetchone()
        home_grid = prefs_row["home_grid"] if prefs_row else None
        time_display = prefs_row["time_display"] if prefs_row else "utc"
        distance_unit = prefs_row["distance_unit"] if prefs_row else "km"

        # Get total count for pagination
        cursor = conn.execute(f"SELECT COUNT(*) FROM qsos {base_where}", qso_params)
        total_qsos = cursor.fetchone()[0]
        total_pages = (total_qsos + per_page - 1) // per_page if total_qsos > 0 else 1
        page = max(1, min(page, total_pages))
        offset = (page - 1) * per_page

        # Get QSOs with pagination
        cursor = conn.execute(f"""
            SELECT id, dx_callsign, qso_datetime_utc, band, mode,
                   dx_grid, dx_dxcc, my_grid, tx_power_w, distance_km,
                   cool_factor, is_confirmed, dx_sig_info, my_sig_info
            FROM qsos {base_where}
            ORDER BY qso_datetime_utc DESC
            LIMIT ? OFFSET ?
        """, qso_params + [per_page, offset])
        qsos = []
        for row in cursor.fetchall():
            qso = dict(row)
            if qso.get("dx_dxcc"):
                qso["dx_country"] = get_country_name(qso["dx_dxcc"])
                qso["country_flag"] = get_country_flag(qso["dx_dxcc"])
            # Format datetime using competitor's preferences
            qso_grid = qso.get("my_grid") or home_grid
            qso["qso_datetime_formatted"] = format_time_local(
                qso.get("qso_datetime_utc"), qso_grid, time_display or "utc"
            )
            # Format distance using competitor's preferences
            if qso.get("distance_km"):
                if distance_unit == "mi":
                    qso["distance_formatted"] = f"{int(qso['distance_km'] * 0.621371)} mi"
                else:
                    qso["distance_formatted"] = f"{int(qso['distance_km'])} km"
            else:
                qso["distance_formatted"] = "-"
            qsos.append(qso)

    return {
        "qsos": qsos,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total_qsos,
            "total_pages": total_pages
        }
    }


# ============================================================
# EXPORT ENDPOINTS
# ============================================================

def format_adif_field(field_name: str, value: str) -> str:
    """Format a single ADIF field in <NAME:LENGTH>VALUE format."""
    if not value:
        return ""
    value_str = str(value)
    return f"<{field_name}:{len(value_str)}>{value_str}"


def qsos_to_adif(qsos: list, callsign: str) -> str:
    """
    Convert a list of QSOs to ADIF format.

    ADIF spec: https://adif.org/314/ADIF_314.htm
    Key fields:
    - CALL: Contacted station's callsign
    - QSO_DATE: Date in YYYYMMDD format
    - TIME_ON: Time in HHMMSS or HHMM format
    - BAND: Band (e.g., "20m")
    - MODE: Mode (e.g., "SSB", "FT8")
    - GRIDSQUARE: Contacted station's grid square
    - DXCC: DXCC entity number
    - TX_PWR: Transmit power in watts
    - MY_GRIDSQUARE: Operator's grid square
    - MY_SIG_INFO: Operator's special interest activity info (e.g., park ID)
    - SIG_INFO: Contacted station's special interest activity info
    """
    from datetime import datetime

    lines = []

    # ADIF header
    lines.append(f"ADIF Export from Ham Radio Olympics for {callsign}")
    lines.append(format_adif_field("ADIF_VER", "3.1.4"))
    lines.append(format_adif_field("PROGRAMID", "Ham Radio Olympics"))
    lines.append(format_adif_field("PROGRAMVERSION", "1.0"))
    lines.append(format_adif_field("CREATED_TIMESTAMP", datetime.utcnow().strftime("%Y%m%d %H%M%S")))
    lines.append("<EOH>")
    lines.append("")

    # QSO records
    for qso in qsos:
        record_parts = []

        # Parse datetime (format: 2025-01-15T12:00:00 or similar)
        qso_dt = qso.get("qso_datetime_utc", "")
        if qso_dt:
            try:
                # Handle ISO format with T separator
                if "T" in qso_dt:
                    dt = datetime.fromisoformat(qso_dt.replace("Z", "+00:00"))
                else:
                    dt = datetime.strptime(qso_dt[:19], "%Y-%m-%d %H:%M:%S")
                record_parts.append(format_adif_field("QSO_DATE", dt.strftime("%Y%m%d")))
                record_parts.append(format_adif_field("TIME_ON", dt.strftime("%H%M%S")))
            except (ValueError, AttributeError):
                pass

        # Required CALL field
        if qso.get("dx_callsign"):
            record_parts.append(format_adif_field("CALL", qso["dx_callsign"]))

        # Band and mode
        if qso.get("band"):
            # ADIF band should be lowercase without 'm' suffix sometimes,
            # but common practice is to include it (e.g., "20m", "40m")
            record_parts.append(format_adif_field("BAND", qso["band"].upper().replace("M", "m")))
        if qso.get("mode"):
            record_parts.append(format_adif_field("MODE", qso["mode"].upper()))

        # Grid squares
        if qso.get("dx_grid"):
            record_parts.append(format_adif_field("GRIDSQUARE", qso["dx_grid"].upper()))
        if qso.get("my_grid"):
            record_parts.append(format_adif_field("MY_GRIDSQUARE", qso["my_grid"].upper()))

        # DXCC entities
        if qso.get("dx_dxcc"):
            record_parts.append(format_adif_field("DXCC", str(qso["dx_dxcc"])))
        if qso.get("my_dxcc"):
            record_parts.append(format_adif_field("MY_DXCC", str(qso["my_dxcc"])))

        # Power
        if qso.get("tx_power_w"):
            record_parts.append(format_adif_field("TX_PWR", str(int(qso["tx_power_w"]))))

        # POTA/Special interest info
        if qso.get("dx_sig_info"):
            record_parts.append(format_adif_field("SIG", "POTA"))
            record_parts.append(format_adif_field("SIG_INFO", qso["dx_sig_info"]))
        if qso.get("my_sig_info"):
            record_parts.append(format_adif_field("MY_SIG", "POTA"))
            record_parts.append(format_adif_field("MY_SIG_INFO", qso["my_sig_info"]))

        # Station callsign (the competitor)
        record_parts.append(format_adif_field("STATION_CALLSIGN", callsign))

        # End of record
        record_parts.append("<EOR>")

        lines.append(" ".join(part for part in record_parts if part))

    return "\n".join(lines)


@app.get("/export/qsos")
async def export_qsos(
    request: Request,
    user: User = Depends(require_user),
    format: str = "csv"
):
    """Export user's QSOs as CSV or ADIF."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    with get_db() as conn:
        cursor = conn.execute("""
            SELECT dx_callsign, qso_datetime_utc, band, mode,
                   dx_grid, dx_dxcc, my_grid, my_dxcc,
                   tx_power_w, dx_sig_info, my_sig_info,
                   is_confirmed, cool_factor, distance_km
            FROM qsos
            WHERE competitor_callsign = ?
            ORDER BY qso_datetime_utc DESC
        """, (user.callsign,))
        qsos = [dict(row) for row in cursor.fetchall()]

    if format.lower() == "adif":
        # Generate ADIF
        content = qsos_to_adif(qsos, user.callsign)
        return StreamingResponse(
            iter([content]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={user.callsign}_qsos.adi"}
        )
    else:
        # Generate CSV (default)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["callsign", "datetime_utc", "band", "mode",
                         "grid", "dxcc", "power_w", "distance_km",
                         "cool_factor", "pota_park", "confirmed"])

        for qso in qsos:
            park = qso.get("dx_sig_info") or qso.get("my_sig_info") or ""
            writer.writerow([
                qso["dx_callsign"],
                qso["qso_datetime_utc"],
                qso["band"] or "",
                qso["mode"] or "",
                qso["dx_grid"] or "",
                qso["dx_dxcc"] or "",
                qso["tx_power_w"] or "",
                qso["distance_km"] or "",
                qso["cool_factor"] or "",
                park,
                "Yes" if qso["is_confirmed"] else "No"
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={user.callsign}_qsos.csv"}
        )


@app.get("/export/medals")
async def export_medals(request: Request, user: User = Depends(require_user)):
    """Export user's medals as CSV."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    with get_db() as conn:
        cursor = conn.execute("""
            SELECT m.qso_race_medal, m.cool_factor_medal, m.total_points,
                   s.name as sport_name, ma.start_date, ma.end_date
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            WHERE m.callsign = ?
            ORDER BY ma.start_date DESC
        """, (user.callsign,))
        medals = cursor.fetchall()

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["sport", "qso_race_medal", "cool_factor_medal", "total_points",
                     "start_date", "end_date"])

    for medal in medals:
        writer.writerow([
            medal["sport_name"],
            medal["qso_race_medal"] or "",
            medal["cool_factor_medal"] or "",
            medal["total_points"],
            medal["start_date"],
            medal["end_date"]
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={user.callsign}_medals.csv"}
    )


# ============================================================
# PDF EXPORT ENDPOINTS
# ============================================================

@app.get("/olympiad.pdf", tags=["public"])
async def get_olympiad_pdf(request: Request, refresh: bool = False):
    """
    Download the cached olympiad standings PDF.

    This PDF is automatically regenerated when medals or records change.
    No authentication required - this is public standings data.

    Query params:
        refresh: If true, force regeneration of the PDF
    """
    from starlette.responses import Response
    from pdf_export import get_cached_pdf, regenerate_active_olympiad_pdf

    # Get active olympiad
    with get_db() as conn:
        cursor = conn.execute("SELECT id, name FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

    if not olympiad:
        raise HTTPException(status_code=404, detail="No active olympiad found")

    # Force regeneration if requested
    if refresh:
        regenerate_active_olympiad_pdf()

    # Try to get cached PDF, regenerate if not exists
    pdf_bytes = get_cached_pdf(olympiad["id"])
    if not pdf_bytes:
        # Generate it now
        regenerate_active_olympiad_pdf()
        pdf_bytes = get_cached_pdf(olympiad["id"])

    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    filename = f"{olympiad['name'].replace(' ', '_')}_Standings.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/export/pdf/olympiad")
async def export_pdf_olympiad(
    request: Request,
    user: User = Depends(require_user),
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
):
    """Export active olympiad as PDF for offline reference."""
    from starlette.responses import Response
    from pdf_export import generate_olympiad_pdf

    # Get active olympiad
    with get_db() as conn:
        cursor = conn.execute("SELECT id, name FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

    if not olympiad:
        raise HTTPException(status_code=404, detail="No active olympiad found")

    try:
        pdf_bytes = generate_olympiad_pdf(
            olympiad_id=olympiad["id"],
            callsign=user.callsign,
            top_n=top_n,
            include_qsos=include_qsos,
            include_records=include_records
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    filename = f"{user.callsign}_olympiad_{olympiad['name'].replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/export/pdf/sport/{sport_id}")
async def export_pdf_sport(
    request: Request,
    sport_id: int,
    user: User = Depends(require_user),
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
):
    """Export single sport as PDF for offline reference."""
    from starlette.responses import Response
    from pdf_export import generate_sport_pdf

    # Verify sport exists
    with get_db() as conn:
        cursor = conn.execute("SELECT id, name FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

    if not sport:
        raise HTTPException(status_code=404, detail="Sport not found")

    try:
        pdf_bytes = generate_sport_pdf(
            sport_id=sport_id,
            callsign=user.callsign,
            top_n=top_n,
            include_qsos=include_qsos,
            include_records=include_records
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    filename = f"{user.callsign}_sport_{sport['name'].replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/export/pdf/my-sports")
async def export_pdf_my_sports(
    request: Request,
    user: User = Depends(require_user),
    top_n: int = 10,
    include_qsos: bool = False,
    include_records: bool = False
):
    """Export competitor's entered sports as PDF for offline reference."""
    from starlette.responses import Response
    from pdf_export import generate_my_sports_pdf

    pdf_bytes = generate_my_sports_pdf(
        callsign=user.callsign,
        top_n=top_n,
        include_qsos=include_qsos,
        include_records=include_records
    )

    filename = f"{user.callsign}_my_sports.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================
# REFEREE ENDPOINTS
# ============================================================

def verify_referee_access(request: Request):
    """Verify the user is a referee."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if not user or not user.is_referee:
        raise HTTPException(status_code=403, detail="Referee access required")
    return user


@app.get("/referee", response_class=HTMLResponse)
async def referee_dashboard(request: Request):
    """Referee dashboard showing assigned sports."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if not user or not user.is_referee:
        raise HTTPException(status_code=403, detail="Referee access required")

    with get_db() as conn:
        # Get sports assigned to this referee with counts
        cursor = conn.execute("""
            SELECT s.id, s.name, s.target_type, o.name as olympiad_name,
                   (SELECT COUNT(*) FROM matches m WHERE m.sport_id = s.id) as match_count,
                   (SELECT COUNT(*) FROM sport_entries se WHERE se.sport_id = s.id) as participant_count
            FROM referee_assignments ra
            JOIN sports s ON ra.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE ra.callsign = ?
            ORDER BY o.start_date DESC, s.name
        """, (user.callsign,))
        assigned_sports = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("referee/dashboard.html", {
        "request": request,
        "user": user,
        "assigned_sports": assigned_sports,
    })


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, _: bool = Depends(verify_admin)):
    """Admin dashboard."""
    from audit import get_audit_logs

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads ORDER BY start_date DESC")
        olympiads = [dict(row) for row in cursor.fetchall()]

        cursor = conn.execute("SELECT COUNT(*) as count FROM competitors")
        competitor_count = cursor.fetchone()["count"]

    # Get recent audit logs
    recent_audit = get_audit_logs(limit=10)

    return templates.TemplateResponse("admin/dashboard.html", {
        "request": request,
        "user": get_current_user(request),
        "olympiads": olympiads,
        "competitor_count": competitor_count,
        "recent_audit": recent_audit,
        "sync_paused": is_sync_paused(),
    })


@app.get("/admin/audit-log", response_class=HTMLResponse)
async def admin_audit_log(
    request: Request,
    _: bool = Depends(verify_admin),
    action: Optional[str] = None,
    page: int = 1
):
    """View audit log."""
    from audit import get_audit_logs, get_audit_log_count

    per_page = 50
    offset = (page - 1) * per_page

    logs = get_audit_logs(limit=per_page, offset=offset, action=action)
    total = get_audit_log_count(action=action)

    user = get_current_user(request)
    return templates.TemplateResponse("admin/audit_log.html", {
        "request": request,
        "user": user,
        "logs": logs,
        "total": total,
        "page": page,
        "per_page": per_page,
        "action_filter": action,
        "display_prefs": get_display_prefs(user),
    })


@app.get("/admin/backup")
async def admin_backup(request: Request, _: bool = Depends(verify_admin)):
    """Download database backup."""
    from audit import log_action
    from starlette.responses import FileResponse
    from datetime import datetime

    user = get_current_user(request)
    db_path = os.environ.get("DATABASE_PATH", "ham_radio_olympics.db")

    # Log the backup action
    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="database_backup",
        details="Database backup downloaded",
        ip_address=get_client_ip(request)
    )

    # Generate filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ham_radio_olympics_backup_{timestamp}.db"

    return FileResponse(
        path=db_path,
        filename=filename,
        media_type="application/octet-stream"
    )


class BulkCallsignsRequest(BaseModel):
    callsigns: List[str]


@app.post("/admin/competitors/bulk-disable")
async def bulk_disable_competitors(
    request: Request,
    data: BulkCallsignsRequest,
    _: bool = Depends(verify_admin)
):
    """Bulk disable multiple competitors."""
    from audit import log_action

    user = get_current_user(request)
    callsigns = [c.upper() for c in data.callsigns]

    if not callsigns:
        return {"message": "No callsigns provided", "disabled": 0}

    with get_db() as conn:
        placeholders = ",".join("?" * len(callsigns))
        conn.execute(
            f"UPDATE competitors SET is_disabled = 1 WHERE callsign IN ({placeholders})",
            callsigns
        )
        # Also delete their sessions
        conn.execute(
            f"DELETE FROM sessions WHERE callsign IN ({placeholders})",
            callsigns
        )

    # Log the action
    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="bulk_disable",
        target_type="competitors",
        details=f"Disabled {len(callsigns)} competitors: {', '.join(callsigns)}",
        ip_address=get_client_ip(request)
    )

    return {"message": f"Disabled {len(callsigns)} competitors", "disabled": len(callsigns)}


@app.post("/admin/competitors/bulk-enable")
async def bulk_enable_competitors(
    request: Request,
    data: BulkCallsignsRequest,
    _: bool = Depends(verify_admin)
):
    """Bulk enable multiple competitors."""
    from audit import log_action

    user = get_current_user(request)
    callsigns = [c.upper() for c in data.callsigns]

    if not callsigns:
        return {"message": "No callsigns provided", "enabled": 0}

    with get_db() as conn:
        placeholders = ",".join("?" * len(callsigns))
        conn.execute(
            f"UPDATE competitors SET is_disabled = 0 WHERE callsign IN ({placeholders})",
            callsigns
        )

    # Log the action
    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="bulk_enable",
        target_type="competitors",
        details=f"Enabled {len(callsigns)} competitors: {', '.join(callsigns)}",
        ip_address=get_client_ip(request)
    )

    return {"message": f"Enabled {len(callsigns)} competitors", "enabled": len(callsigns)}


@app.post("/admin/competitors/bulk-delete")
async def bulk_delete_competitors(
    request: Request,
    data: BulkCallsignsRequest,
    _: bool = Depends(verify_admin)
):
    """Bulk delete multiple competitors."""
    from audit import log_action

    user = get_current_user(request)
    callsigns = [c.upper() for c in data.callsigns]

    if not callsigns:
        return {"message": "No callsigns provided", "deleted": 0}

    with get_db() as conn:
        placeholders = ",".join("?" * len(callsigns))
        # Delete related data first
        conn.execute(f"DELETE FROM qsos WHERE competitor_callsign IN ({placeholders})", callsigns)
        conn.execute(f"DELETE FROM medals WHERE callsign IN ({placeholders})", callsigns)
        conn.execute(f"DELETE FROM sport_entries WHERE callsign IN ({placeholders})", callsigns)
        conn.execute(f"DELETE FROM sessions WHERE callsign IN ({placeholders})", callsigns)
        conn.execute(f"DELETE FROM competitors WHERE callsign IN ({placeholders})", callsigns)

    # Log the action
    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="bulk_delete",
        target_type="competitors",
        details=f"Deleted {len(callsigns)} competitors: {', '.join(callsigns)}",
        ip_address=get_client_ip(request)
    )

    return {"message": f"Deleted {len(callsigns)} competitors", "deleted": len(callsigns)}


@app.get("/admin/olympiads", response_class=HTMLResponse)
async def admin_olympiads(request: Request, _: bool = Depends(verify_admin)):
    """List/create Olympiads."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads ORDER BY start_date DESC")
        olympiads = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("admin/olympiads.html", {
        "request": request,
        "user": get_current_user(request),
        "olympiads": olympiads,
    })


@app.post("/admin/olympiad")
async def create_olympiad(olympiad: OlympiadCreate, _: bool = Depends(verify_admin)):
    """Create a new Olympiad."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
            VALUES (?, ?, ?, ?, 0)
        """, (olympiad.name, olympiad.start_date, olympiad.end_date, olympiad.qualifying_qsos))

        return {"id": cursor.lastrowid, "message": "Olympiad created"}


@app.post("/admin/check-park-anomalies")
async def admin_check_park_anomalies(request: Request, _: bool = Depends(verify_admin)):
    """Run park anomaly check and redirect back with results."""
    from sync import check_and_disqualify_park_anomalies
    from starlette.responses import RedirectResponse
    stats = check_and_disqualify_park_anomalies()
    # Store results in a flash message or just redirect
    return RedirectResponse(
        url=f"/admin/park-anomalies?checked=1&found={stats['anomalies_found']}&dq={stats['qsos_disqualified']}",
        status_code=303
    )


@app.get("/admin/park-anomalies")
async def admin_get_park_anomalies(
    request: Request,
    checked: int = 0,
    found: int = 0,
    dq: int = 0,
    _: bool = Depends(verify_admin)
):
    """View all QSOs with park reference anomalies (already disqualified by SYSTEM)."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT c.comment, c.created_at, d.qso_id, d.sport_id, d.status,
                   q.competitor_callsign, q.dx_callsign, q.my_sig_info, q.dx_sig_info, q.qso_datetime_utc,
                   s.name as sport_name
            FROM qso_disqualification_comments c
            JOIN qso_disqualifications d ON c.disqualification_id = d.id
            JOIN qsos q ON d.qso_id = q.id
            JOIN sports s ON d.sport_id = s.id
            WHERE c.author_callsign = 'SYSTEM'
            ORDER BY c.created_at DESC
        """)
        anomalies = [dict(row) for row in cursor.fetchall()]

    # Group by QSO but keep sport details for RQ action
    qso_map = {}
    for a in anomalies:
        qso_id = a["qso_id"]
        if qso_id not in qso_map:
            qso_map[qso_id] = {
                "qso_id": qso_id,
                "competitor": a["competitor_callsign"],
                "dx_call": a["dx_callsign"],
                "qso_date": a["qso_datetime_utc"][:10] if a["qso_datetime_utc"] else "",
                "my_sig_info": a["my_sig_info"],
                "dx_sig_info": a["dx_sig_info"],
                "comment": a["comment"].replace("Auto-flagged: Park reference format anomaly. ", ""),
                "created_at": a["created_at"],
                "sports": [],  # List of {sport_id, sport_name, status}
            }
        qso_map[qso_id]["sports"].append({
            "sport_id": a["sport_id"],
            "sport_name": a["sport_name"],
            "status": a["status"],
        })

    grouped = list(qso_map.values())

    # Count stats
    total_dq = sum(1 for a in grouped for s in a["sports"] if s["status"] == "disqualified")
    total_rq = sum(1 for a in grouped for s in a["sports"] if s["status"] == "requalified")

    return templates.TemplateResponse("admin/park_anomalies.html", {
        "request": request,
        "user": get_current_user(request),
        "anomalies": grouped,
        "total_count": len(grouped),
        "total_dq": total_dq,
        "total_rq": total_rq,
        "checked": checked,
        "found": found,
        "dq": dq,
    })


@app.get("/admin/qrz-debug/{callsign}")
async def admin_qrz_debug(callsign: str, _: bool = Depends(verify_admin)):
    """
    Debug endpoint to see raw ADIF fields from QRZ for a competitor.
    Useful for diagnosing why park IDs might not be coming through.
    """
    from crypto import decrypt_api_key
    from qrz_client import fetch_qsos, parse_qrz_response, parse_adif, QRZAPIError
    import httpx
    import binascii
    from cryptography.fernet import InvalidToken

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT qrz_api_key_encrypted FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        competitor = cursor.fetchone()

        if not competitor:
            raise HTTPException(status_code=404, detail=f"Competitor {callsign} not found")

        if not competitor["qrz_api_key_encrypted"]:
            raise HTTPException(status_code=400, detail="QRZ API key not configured for this competitor")

        try:
            api_key = decrypt_api_key(competitor["qrz_api_key_encrypted"])
        except (InvalidToken, binascii.Error, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to decrypt API key: {e}")

    # Fetch raw ADIF from QRZ (just first page, up to 50 recent QSOs)
    async with httpx.AsyncClient() as client:
        data = {
            "KEY": api_key,
            "ACTION": "FETCH",
            "OPTION": "TYPE:ADIF,MAX:50",
        }

        try:
            response = await client.post(
                "https://logbook.qrz.com/api",
                data=data,
                headers={"User-Agent": "HamRadioOlympics/1.0"},
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"QRZ API error: {e}")

        result = parse_qrz_response(response.text)

        if result.get("RESULT") == "AUTH":
            raise HTTPException(status_code=401, detail="QRZ authentication failed")

        if result.get("RESULT") == "FAIL":
            reason = result.get("REASON", "Unknown error")
            raise HTTPException(status_code=400, detail=f"QRZ error: {reason}")

        adif_data = result.get("ADIF", "")
        if not adif_data:
            return {"message": "No QSOs found", "qsos": []}

        raw_qsos = parse_adif(adif_data)

    # Return simplified view of each QSO's raw ADIF fields
    # Highlight park-related fields
    debug_qsos = []
    park_keywords = ['SIG', 'POTA', 'WWFF', 'PARK', 'NOTES', 'COMMENT']

    for raw_qso in raw_qsos[:20]:  # Limit to 20 most recent
        # Categorize fields
        park_fields = {k: v for k, v in raw_qso.items()
                       if any(kw in k.upper() for kw in park_keywords)}
        key_fields = {
            "CALL": raw_qso.get("CALL"),
            "QSO_DATE": raw_qso.get("QSO_DATE"),
            "TIME_ON": raw_qso.get("TIME_ON"),
            "BAND": raw_qso.get("BAND"),
            "MODE": raw_qso.get("MODE"),
        }
        debug_qsos.append({
            "key_fields": key_fields,
            "park_fields": park_fields,
            "all_fields": list(raw_qso.keys()),
        })

    return {
        "callsign": callsign.upper(),
        "qso_count": len(raw_qsos),
        "qsos": debug_qsos,
        "note": "park_fields shows fields containing SIG, POTA, WWFF, PARK, NOTES, or COMMENT"
    }


@app.get("/admin/olympiad/{olympiad_id}")
async def get_olympiad_admin(olympiad_id: int, _: bool = Depends(verify_admin)):
    """Get Olympiad for editing."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads WHERE id = ?", (olympiad_id,))
        olympiad = cursor.fetchone()

        if not olympiad:
            raise HTTPException(status_code=404, detail="Olympiad not found")

        return dict(olympiad)


@app.put("/admin/olympiad/{olympiad_id}")
async def update_olympiad(olympiad_id: int, olympiad: OlympiadCreate, _: bool = Depends(verify_admin)):
    """Update an Olympiad."""
    with get_db() as conn:
        conn.execute("""
            UPDATE olympiads
            SET name = ?, start_date = ?, end_date = ?, qualifying_qsos = ?
            WHERE id = ?
        """, (olympiad.name, olympiad.start_date, olympiad.end_date, olympiad.qualifying_qsos, olympiad_id))

    return {"message": "Olympiad updated"}


@app.delete("/admin/olympiad/{olympiad_id}")
async def delete_olympiad(olympiad_id: int, _: bool = Depends(verify_admin)):
    """Delete an Olympiad."""
    with get_db() as conn:
        conn.execute("DELETE FROM olympiads WHERE id = ?", (olympiad_id,))

    return {"message": "Olympiad deleted"}


@app.post("/admin/olympiad/{olympiad_id}/activate")
async def activate_olympiad(olympiad_id: int, _: bool = Depends(verify_admin)):
    """Set an Olympiad as active (deactivates others)."""
    with get_db() as conn:
        conn.execute("UPDATE olympiads SET is_active = 0")
        conn.execute("UPDATE olympiads SET is_active = 1 WHERE id = ?", (olympiad_id,))

    return {"message": "Olympiad activated"}


@app.post("/admin/olympiad/{olympiad_id}/deactivate")
async def deactivate_olympiad(olympiad_id: int, _: bool = Depends(verify_admin)):
    """Deactivate an Olympiad (pause it)."""
    with get_db() as conn:
        conn.execute("UPDATE olympiads SET is_active = 0 WHERE id = ?", (olympiad_id,))

    return {"message": "Olympiad deactivated"}


@app.get("/admin/olympiad/{olympiad_id}/sports", response_class=HTMLResponse)
async def admin_olympiad_sports(request: Request, olympiad_id: int, _: bool = Depends(verify_admin)):
    """List Sports for an Olympiad."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads WHERE id = ?", (olympiad_id,))
        olympiad = cursor.fetchone()
        if not olympiad:
            raise HTTPException(status_code=404, detail="Olympiad not found")

        cursor = conn.execute(
            "SELECT * FROM sports WHERE olympiad_id = ?",
            (olympiad_id,)
        )
        sports = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("admin/sports.html", {
        "request": request,
        "user": get_current_user(request),
        "olympiad": dict(olympiad),
        "sports": sports,
    })


@app.post("/admin/olympiad/{olympiad_id}/sport")
async def create_sport(olympiad_id: int, sport: SportCreate, _: bool = Depends(verify_admin)):
    """Create a new Sport."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO sports (olympiad_id, name, description, target_type,
                              work_enabled, activate_enabled, separate_pools, allowed_modes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            olympiad_id, sport.name, sport.description, sport.target_type,
            1 if sport.work_enabled else 0,
            1 if sport.activate_enabled else 0,
            1 if sport.separate_pools else 0,
            sport.allowed_modes,
        ))

        return {"id": cursor.lastrowid, "message": "Sport created"}


@app.get("/admin/sport/{sport_id}")
async def get_sport_admin(sport_id: int, _: bool = Depends(verify_admin)):
    """Get Sport for editing."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        return dict(sport)


@app.put("/admin/sport/{sport_id}")
async def update_sport(request: Request, sport_id: int, sport: SportCreate):
    """Update a Sport. Admins or assigned referees can update."""
    verify_admin_or_sport_referee(request, sport_id)
    with get_db() as conn:
        conn.execute("""
            UPDATE sports
            SET name = ?, description = ?, target_type = ?,
                work_enabled = ?, activate_enabled = ?, separate_pools = ?, allowed_modes = ?
            WHERE id = ?
        """, (
            sport.name, sport.description, sport.target_type,
            1 if sport.work_enabled else 0,
            1 if sport.activate_enabled else 0,
            1 if sport.separate_pools else 0,
            sport.allowed_modes,
            sport_id,
        ))

    return {"message": "Sport updated"}


@app.delete("/admin/sport/{sport_id}")
async def delete_sport(request: Request, sport_id: int):
    """Delete a Sport. Admins or assigned referees can delete."""
    verify_admin_or_sport_referee(request, sport_id)
    with get_db() as conn:
        conn.execute("DELETE FROM sports WHERE id = ?", (sport_id,))

    return {"message": "Sport deleted"}


@app.post("/admin/sport/{sport_id}/duplicate")
async def duplicate_sport(sport_id: int, _: bool = Depends(verify_admin)):
    """Duplicate a Sport and all its Matches."""
    with get_db() as conn:
        # Get original sport
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Create new sport with " (copy)" appended
        cursor = conn.execute("""
            INSERT INTO sports (olympiad_id, name, description, target_type,
                              work_enabled, activate_enabled, separate_pools, allowed_modes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sport["olympiad_id"],
            sport["name"] + " (copy)",
            sport["description"],
            sport["target_type"],
            sport["work_enabled"],
            sport["activate_enabled"],
            sport["separate_pools"],
            sport["allowed_modes"],
        ))
        new_sport_id = cursor.lastrowid

        # Get all matches from original sport
        cursor = conn.execute("SELECT * FROM matches WHERE sport_id = ?", (sport_id,))
        matches = cursor.fetchall()

        # Duplicate each match
        for match in matches:
            conn.execute("""
                INSERT INTO matches (sport_id, start_date, end_date, target_value, target_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                new_sport_id,
                match["start_date"],
                match["end_date"],
                match["target_value"],
                match["target_type"],
            ))

    return {
        "message": f"Sport duplicated with {len(matches)} matches",
        "new_sport_id": new_sport_id
    }


@app.get("/admin/sport/{sport_id}/matches", response_class=HTMLResponse)
async def admin_sport_matches(request: Request, sport_id: int):
    """List Matches for a Sport. Admins or assigned referees can view."""
    verify_admin_or_sport_referee(request, sport_id)
    from dxcc import get_all_continents, get_all_countries

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
            (sport_id,)
        )
        matches = [dict(row) for row in cursor.fetchall()]

    # Get target options based on sport type
    sport_dict = dict(sport)
    target_options = []
    if sport_dict["target_type"] == "continent":
        target_options = get_all_continents()
    elif sport_dict["target_type"] == "country":
        target_options = get_all_countries()

    # Add target_display to each match
    for match in matches:
        match["target_display"] = format_target_display(match["target_value"], sport_dict["target_type"])

    user = get_current_user(request)
    return templates.TemplateResponse("admin/matches.html", {
        "request": request,
        "user": user,
        "sport": sport_dict,
        "matches": matches,
        "target_options": target_options,
        "display_prefs": get_display_prefs(user),
    })


@app.get("/admin/sport/{sport_id}/competitors", response_class=HTMLResponse)
async def admin_sport_competitors(request: Request, sport_id: int):
    """List competitors for a Sport. Admins or assigned referees can view."""
    verify_admin_or_sport_referee(request, sport_id)

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Get all competitors entered in this sport with their medal counts
        cursor = conn.execute("""
            SELECT c.callsign, c.first_name, c.registered_at, c.is_disabled,
                   COUNT(DISTINCT m.id) as medal_count,
                   SUM(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                   SUM(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                   SUM(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count
            FROM sport_entries se
            JOIN competitors c ON se.callsign = c.callsign
            LEFT JOIN medals m ON m.callsign = c.callsign AND m.match_id IN (
                SELECT id FROM matches WHERE sport_id = ?
            )
            WHERE se.sport_id = ?
            GROUP BY c.callsign, c.first_name
            ORDER BY c.callsign
        """, (sport_id, sport_id))
        competitors = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("admin/sport_competitors.html", {
        "request": request,
        "user": get_current_user(request),
        "sport": dict(sport),
        "competitors": competitors,
    })


@app.post("/admin/sport/{sport_id}/competitor/{callsign}/disqualify")
async def disqualify_from_sport(request: Request, sport_id: int, callsign: str):
    """Disqualify a competitor from a specific sport. Admins or assigned referees can disqualify."""
    verify_admin_or_sport_referee(request, sport_id)
    callsign = callsign.upper()

    with get_db() as conn:
        # Check competitor exists and is entered in sport
        cursor = conn.execute(
            "SELECT 1 FROM sport_entries WHERE callsign = ? AND sport_id = ?",
            (callsign, sport_id)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found in this sport")

        # Remove from sport
        conn.execute(
            "DELETE FROM sport_entries WHERE callsign = ? AND sport_id = ?",
            (callsign, sport_id)
        )

        # Delete medals for this sport's matches
        conn.execute("""
            DELETE FROM medals WHERE callsign = ? AND match_id IN (
                SELECT id FROM matches WHERE sport_id = ?
            )
        """, (callsign, sport_id))

    # Recompute medals for affected matches
    recompute_sport_matches(sport_id)

    return {"message": f"Competitor {callsign} disqualified from sport"}


@app.post("/admin/sport/{sport_id}/match")
async def create_match(request: Request, sport_id: int, match: MatchCreate):
    """Create a new Match. Admins or assigned referees can create."""
    verify_admin_or_sport_referee(request, sport_id)

    # Validate start_date < end_date
    if match.start_date > match.end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    with get_db() as conn:
        # Get Olympiad date bounds for this sport
        cursor = conn.execute("""
            SELECT o.start_date, o.end_date, o.name
            FROM olympiads o
            JOIN sports s ON s.olympiad_id = o.id
            WHERE s.id = ?
        """, (sport_id,))
        olympiad = cursor.fetchone()

        if not olympiad:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Validate match dates fall within Olympiad bounds
        if match.start_date < olympiad["start_date"]:
            raise HTTPException(
                status_code=400,
                detail=f"Match start date cannot be before Olympiad start ({olympiad['start_date']})"
            )
        if match.end_date > olympiad["end_date"]:
            raise HTTPException(
                status_code=400,
                detail=f"Match end date cannot be after Olympiad end ({olympiad['end_date']})"
            )

        cursor = conn.execute("""
            INSERT INTO matches (sport_id, start_date, end_date, target_value, allowed_modes, max_power_w, target_type, confirmation_deadline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (sport_id, match.start_date, match.end_date, match.target_value, match.allowed_modes, match.max_power_w, match.target_type, match.confirmation_deadline))

        return {"id": cursor.lastrowid, "message": "Match created"}


@app.get("/admin/match/{match_id}")
async def get_match_admin(request: Request, match_id: int):
    """Get Match for editing. Admins or assigned referees can view."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        match = cursor.fetchone()

        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        # Verify access after we know the sport_id
        verify_admin_or_sport_referee(request, match["sport_id"])

        return dict(match)


@app.put("/admin/match/{match_id}")
async def update_match(request: Request, match_id: int, match: MatchCreate):
    """Update a Match. Admins or assigned referees can update."""
    # Validate start_date < end_date
    if match.start_date > match.end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    with get_db() as conn:
        # Get sport_id and Olympiad bounds for permission check and validation
        cursor = conn.execute("""
            SELECT m.sport_id, o.start_date as olympiad_start, o.end_date as olympiad_end
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE m.id = ?
        """, (match_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Match not found")

        verify_admin_or_sport_referee(request, existing["sport_id"])

        # Validate match dates fall within Olympiad bounds
        if match.start_date < existing["olympiad_start"]:
            raise HTTPException(
                status_code=400,
                detail=f"Match start date cannot be before Olympiad start ({existing['olympiad_start']})"
            )
        if match.end_date > existing["olympiad_end"]:
            raise HTTPException(
                status_code=400,
                detail=f"Match end date cannot be after Olympiad end ({existing['olympiad_end']})"
            )

        conn.execute("""
            UPDATE matches
            SET start_date = ?, end_date = ?, target_value = ?, allowed_modes = ?, max_power_w = ?, target_type = ?, confirmation_deadline = ?
            WHERE id = ?
        """, (match.start_date, match.end_date, match.target_value, match.allowed_modes, match.max_power_w, match.target_type, match.confirmation_deadline, match_id))

    # Recompute medals for this match (run in thread pool to avoid blocking/locks)
    await asyncio.to_thread(recompute_match_medals, match_id)

    return {"message": "Match updated"}


@app.delete("/admin/match/{match_id}")
async def delete_match(request: Request, match_id: int):
    """Delete a Match. Admins or assigned referees can delete."""
    with get_db() as conn:
        # Get sport_id for permission check
        cursor = conn.execute("SELECT sport_id FROM matches WHERE id = ?", (match_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Match not found")

        verify_admin_or_sport_referee(request, existing["sport_id"])

        conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))

    return {"message": "Match deleted"}


@app.get("/admin/competitors", response_class=HTMLResponse)
async def admin_competitors(
    request: Request,
    _: bool = Depends(verify_admin),
    search: Optional[str] = None,
    page: int = 1
):
    """List all competitors."""
    per_page = 50
    offset = (page - 1) * per_page

    with get_db() as conn:
        # Build query with optional search
        base_where = "WHERE callsign LIKE ?" if search else ""
        params = [f"%{search}%"] if search else []

        # Get total count
        count_query = f"SELECT COUNT(*) FROM competitors {base_where}"
        cursor = conn.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        page = max(1, min(page, total_pages))

        # Get paginated results
        query = f"""
            SELECT callsign, first_name, registered_at, last_sync_at, is_disabled, is_admin, is_referee
            FROM competitors
            {base_where}
            ORDER BY registered_at DESC
            LIMIT ? OFFSET ?
        """
        cursor = conn.execute(query, params + [per_page, offset])
        competitors = [dict(row) for row in cursor.fetchall()]

        # Get referee assignments in a single query to avoid N+1
        referee_callsigns = [c["callsign"] for c in competitors if c["is_referee"]]
        referee_assignments = {}
        if referee_callsigns:
            placeholders = ",".join("?" * len(referee_callsigns))
            cursor = conn.execute(f"""
                SELECT ra.callsign, s.id, s.name FROM referee_assignments ra
                JOIN sports s ON ra.sport_id = s.id
                WHERE ra.callsign IN ({placeholders})
            """, referee_callsigns)
            for row in cursor.fetchall():
                callsign = row["callsign"]
                if callsign not in referee_assignments:
                    referee_assignments[callsign] = []
                referee_assignments[callsign].append({"id": row["id"], "name": row["name"]})

        # Attach assignments to competitors
        for c in competitors:
            c["assigned_sports"] = referee_assignments.get(c["callsign"], [])

        # Get all sports for assignment dropdown
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name
            FROM sports s
            JOIN olympiads o ON s.olympiad_id = o.id
            ORDER BY o.id DESC, s.name
        """)
        all_sports = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("admin/competitors.html", {
        "request": request,
        "user": get_current_user(request),
        "competitors": competitors,
        "all_sports": all_sports,
        "pagination": {
            "page": page,
            "total_pages": total_pages,
            "total_items": total_count,
            "per_page": per_page,
        },
    })


@app.get("/admin/export/competitors")
async def admin_export_competitors(request: Request, _: bool = Depends(verify_admin)):
    """Export all competitors as CSV."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    with get_db() as conn:
        cursor = conn.execute("""
            SELECT callsign, email, registered_at, last_sync_at, is_disabled, is_admin, is_referee
            FROM competitors
            ORDER BY registered_at DESC
        """)
        competitors = cursor.fetchall()

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["callsign", "email", "registered_at", "last_sync_at",
                     "is_disabled", "is_admin", "is_referee"])

    for comp in competitors:
        writer.writerow([
            comp["callsign"],
            comp["email"] or "",
            comp["registered_at"],
            comp["last_sync_at"] or "",
            "Yes" if comp["is_disabled"] else "No",
            "Yes" if comp["is_admin"] else "No",
            "Yes" if comp["is_referee"] else "No"
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=competitors.csv"}
    )


@app.get("/admin/export/standings/{olympiad_id}")
async def admin_export_standings(olympiad_id: int, _: bool = Depends(verify_admin)):
    """Export olympiad standings as CSV."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    with get_db() as conn:
        # Verify olympiad exists
        cursor = conn.execute("SELECT name FROM olympiads WHERE id = ?", (olympiad_id,))
        olympiad = cursor.fetchone()
        if not olympiad:
            raise HTTPException(status_code=404, detail="Olympiad not found")

        # Get standings with medal counts and points
        cursor = conn.execute("""
            SELECT
                c.callsign,
                COALESCE(SUM(m.total_points), 0) as total_points,
                COUNT(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 END) as gold_count,
                COUNT(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 END) as silver_count,
                COUNT(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 END) as bronze_count
            FROM competitors c
            LEFT JOIN medals m ON c.callsign = m.callsign
            LEFT JOIN matches ma ON m.match_id = ma.id
            LEFT JOIN sports s ON ma.sport_id = s.id AND s.olympiad_id = ?
            GROUP BY c.callsign
            HAVING total_points > 0
            ORDER BY total_points DESC, gold_count DESC, silver_count DESC, bronze_count DESC
        """, (olympiad_id,))
        standings = cursor.fetchall()

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["rank", "callsign", "total_points", "gold", "silver", "bronze"])

    for rank, standing in enumerate(standings, 1):
        writer.writerow([
            rank,
            standing["callsign"],
            standing["total_points"],
            standing["gold_count"],
            standing["silver_count"],
            standing["bronze_count"]
        ])

    output.seek(0)
    olympiad_name = olympiad["name"].replace(" ", "_")
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={olympiad_name}_standings.csv"}
    )


@app.post("/admin/competitor/{callsign}/disable")
async def disable_competitor(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Disable a competitor's account."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_disabled = 1 WHERE callsign = ?", (callsign,))
        # Also delete their sessions so they're logged out
        conn.execute("DELETE FROM sessions WHERE callsign = ?", (callsign,))

    # Log the action
    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="competitor_disabled",
        target_type="competitor",
        target_id=callsign,
        details=f"Competitor {callsign} disabled by admin",
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} disabled"}


@app.post("/admin/competitor/{callsign}/enable")
async def enable_competitor(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Enable a competitor's account."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_disabled = 0 WHERE callsign = ?", (callsign,))

    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="competitor_enabled",
        target_type="competitor",
        target_id=callsign,
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} enabled"}


@app.post("/admin/competitor/{callsign}/set-admin")
async def set_admin_role(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Grant admin role to a competitor."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = ?", (callsign,))

    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="admin_role_granted",
        target_type="competitor",
        target_id=callsign,
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} is now an admin"}


@app.post("/admin/competitor/{callsign}/remove-admin")
async def remove_admin_role(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Remove admin role from a competitor."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_admin = 0 WHERE callsign = ?", (callsign,))

    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="admin_role_revoked",
        target_type="competitor",
        target_id=callsign,
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} is no longer an admin"}


@app.post("/admin/competitor/{callsign}/set-referee")
async def set_referee_role(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Grant referee role to a competitor."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_referee = 1 WHERE callsign = ?", (callsign,))

    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="referee_role_granted",
        target_type="competitor",
        target_id=callsign,
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} is now a referee"}


@app.post("/admin/competitor/{callsign}/remove-referee")
async def remove_referee_role(request: Request, callsign: str, _: bool = Depends(verify_admin)):
    """Remove referee role from a competitor."""
    from audit import log_action

    callsign = callsign.upper()
    user = get_current_user(request)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_referee = 0 WHERE callsign = ?", (callsign,))
        # Also remove all referee assignments
        conn.execute("DELETE FROM referee_assignments WHERE callsign = ?", (callsign,))

    log_action(
        actor_callsign=user.callsign if user else "unknown",
        action="referee_role_revoked",
        target_type="competitor",
        target_id=callsign,
        ip_address=get_client_ip(request)
    )

    return {"message": f"Competitor {callsign} is no longer a referee"}


@app.post("/admin/competitor/{callsign}/assign-sport/{sport_id}")
async def assign_referee_to_sport(callsign: str, sport_id: int, _: bool = Depends(verify_admin)):
    """Assign a referee to a sport."""
    callsign = callsign.upper()
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        # Check competitor exists and is a referee
        cursor = conn.execute("SELECT is_referee FROM competitors WHERE callsign = ?", (callsign,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Competitor not found")
        if not row["is_referee"]:
            raise HTTPException(status_code=400, detail="Competitor is not a referee")

        # Check sport exists
        cursor = conn.execute("SELECT 1 FROM sports WHERE id = ?", (sport_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Sport not found")

        # Check if already assigned
        cursor = conn.execute(
            "SELECT 1 FROM referee_assignments WHERE callsign = ? AND sport_id = ?",
            (callsign, sport_id)
        )
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Referee already assigned to this sport")

        conn.execute(
            "INSERT INTO referee_assignments (callsign, sport_id, assigned_at) VALUES (?, ?, ?)",
            (callsign, sport_id, now)
        )

    return {"message": f"Referee {callsign} assigned to sport {sport_id}"}


@app.delete("/admin/competitor/{callsign}/assign-sport/{sport_id}")
async def remove_referee_from_sport(callsign: str, sport_id: int, _: bool = Depends(verify_admin)):
    """Remove a referee from a sport."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM referee_assignments WHERE callsign = ? AND sport_id = ?",
            (callsign, sport_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assignment not found")

    return {"message": f"Referee {callsign} removed from sport {sport_id}"}


@app.post("/admin/competitor/{callsign}/disqualify")
async def disqualify_competitor(callsign: str, _: bool = Depends(verify_admin_or_referee)):
    """Disqualify a competitor from the active competition. Admins or referees can disqualify."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")

        # Get active olympiad
        cursor = conn.execute("SELECT id FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()
        if not olympiad:
            raise HTTPException(status_code=400, detail="No active competition")

        # Get all sports in active olympiad
        cursor = conn.execute("SELECT id FROM sports WHERE olympiad_id = ?", (olympiad["id"],))
        sport_ids = [row["id"] for row in cursor.fetchall()]

        # Remove from all sports
        removed_count = 0
        for sport_id in sport_ids:
            cursor = conn.execute(
                "DELETE FROM sport_entries WHERE callsign = ? AND sport_id = ?",
                (callsign, sport_id)
            )
            removed_count += cursor.rowcount

            # Delete medals for this sport
            conn.execute("""
                DELETE FROM medals WHERE callsign = ? AND match_id IN (
                    SELECT id FROM matches WHERE sport_id = ?
                )
            """, (callsign, sport_id))

    # Recompute medals for affected matches (run in thread pool to avoid blocking)
    from sync import recompute_all_active_matches
    await asyncio.to_thread(recompute_all_active_matches)

    return {"message": f"Competitor {callsign} disqualified from {removed_count} sport(s)"}


@app.post("/admin/competitor/{callsign}/reset-password")
async def reset_competitor_password(callsign: str, _: bool = Depends(verify_admin)):
    """Reset a competitor's password to a random value."""
    callsign = callsign.upper()

    # Generate a random password
    new_password = secrets.token_urlsafe(12)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")

        # Hash and store new password
        password_hash = hash_password(new_password)
        conn.execute(
            "UPDATE competitors SET password_hash = ? WHERE callsign = ?",
            (password_hash, callsign)
        )
        # Invalidate all sessions
        conn.execute("DELETE FROM sessions WHERE callsign = ?", (callsign,))

    return {"message": f"Password reset for {callsign}", "new_password": new_password}


@app.delete("/admin/competitor/{callsign}")
async def delete_competitor(callsign: str, _: bool = Depends(verify_admin)):
    """Delete a competitor."""
    with get_db() as conn:
        conn.execute("DELETE FROM competitors WHERE callsign = ?", (callsign.upper(),))

    return {"message": f"Competitor {callsign.upper()} deleted"}


@app.post("/admin/competitor/{callsign}/reset-qsos")
async def admin_reset_competitor_qsos(
    request: Request,
    callsign: str,
    _: bool = Depends(verify_admin)
):
    """
    Admin endpoint to reset a competitor's QSOs.

    Deletes all QSOs and medals for the specified competitor, then triggers
    a full sync using their stored credentials.
    """
    from sync import delete_competitor_qsos
    from audit import log_action

    callsign = callsign.upper()

    # Verify competitor exists and has stored credentials
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT qrz_api_key_encrypted, lotw_username_encrypted, lotw_password_encrypted
               FROM competitors WHERE callsign = ?""",
            (callsign,)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Competitor not found")

    has_qrz = bool(row["qrz_api_key_encrypted"])
    has_lotw = bool(row["lotw_username_encrypted"] and row["lotw_password_encrypted"])

    if not has_qrz and not has_lotw:
        raise HTTPException(
            status_code=400,
            detail=f"Competitor {callsign} has no stored credentials. Cannot sync after reset."
        )

    # Delete all QSOs and medals
    deleted_count = delete_competitor_qsos(callsign)

    # Log the action
    admin_user = get_current_user(request)
    log_action(
        admin_user.callsign if admin_user else "ADMIN",
        "admin_reset_qsos",
        f"Reset QSOs for {callsign}. Deleted {deleted_count} QSOs."
    )

    # Sync from stored credentials
    results = {"deleted_qsos": deleted_count, "callsign": callsign}

    if has_qrz:
        qrz_result = await sync_competitor(callsign)
        results["qrz_sync"] = qrz_result

    if has_lotw:
        lotw_result = await sync_competitor_lotw_stored(callsign)
        results["lotw_sync"] = lotw_result

    return results


@app.post("/admin/competitor/{callsign}/sync")
async def admin_sync_competitor(
    request: Request,
    callsign: str,
    _: bool = Depends(verify_admin)
):
    """
    Admin endpoint to trigger an incremental sync for a competitor.

    Uses their stored credentials for a normal incremental sync.
    """
    callsign = callsign.upper()

    # Verify competitor exists
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT qrz_api_key_encrypted, lotw_username_encrypted, lotw_password_encrypted
               FROM competitors WHERE callsign = ?""",
            (callsign,)
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Competitor not found")

    has_qrz = bool(row["qrz_api_key_encrypted"])
    has_lotw = bool(row["lotw_username_encrypted"] and row["lotw_password_encrypted"])

    if not has_qrz and not has_lotw:
        raise HTTPException(
            status_code=400,
            detail=f"Competitor {callsign} has no stored credentials."
        )

    results = {"callsign": callsign}

    if has_qrz:
        qrz_result = await sync_competitor(callsign)
        results["qrz_sync"] = qrz_result

    if has_lotw:
        lotw_result = await sync_competitor_lotw_stored(callsign)
        results["lotw_sync"] = lotw_result

    return results


# ============================================================
# ADMIN SETTINGS ENDPOINTS
# ============================================================

@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request, _: bool = Depends(verify_admin)):
    """Admin settings page."""
    from database import get_setting

    # Check if QRZ is configured (don't decrypt password, just check if it exists)
    qrz_username = get_setting("qrz_username", decrypt=True)
    qrz_configured = qrz_username is not None and get_setting("qrz_password") is not None

    # Get current theme settings (from DB, falling back to config)
    current_theme = get_setting("site_theme") or config.SITE_THEME
    site_name = get_setting("site_name") or config.SITE_NAME
    site_tagline = get_setting("site_tagline") or config.SITE_TAGLINE

    # Get public results setting
    public_results = get_setting("public_results") == "1"

    # Get email settings
    admin_bcc_emails = get_setting("admin_bcc_emails") == "1"
    admin_email = config.ADMIN_EMAIL

    # Get Discord settings
    discord_webhook_url = get_setting("discord_webhook_url")
    discord_configured = bool(discord_webhook_url)

    # Discord notification options (default to enabled)
    discord_notify_signups = get_setting("discord_notify_signups") != "0"
    discord_notify_records = get_setting("discord_notify_records") != "0"
    discord_notify_medals = get_setting("discord_notify_medals") != "0"
    discord_notify_pota = get_setting("discord_notify_pota") != "0"
    discord_pota_interval = int(get_setting("discord_pota_interval") or "30")

    return templates.TemplateResponse("admin/settings.html", {
        "request": request,
        "user": get_current_user(request),
        "qrz_username": qrz_username,
        "qrz_configured": qrz_configured,
        "current_theme": current_theme,
        "site_name": site_name,
        "site_tagline": site_tagline,
        "public_results": public_results,
        "admin_bcc_emails": admin_bcc_emails,
        "admin_email": admin_email,
        "discord_webhook_url": discord_webhook_url,
        "discord_configured": discord_configured,
        "discord_notify_signups": discord_notify_signups,
        "discord_notify_records": discord_notify_records,
        "discord_notify_medals": discord_notify_medals,
        "discord_notify_pota": discord_notify_pota,
        "discord_pota_interval": discord_pota_interval,
    })


@app.post("/admin/settings/qrz")
async def update_qrz_settings(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update QRZ API credentials."""
    from database import set_setting, get_setting
    from callsign_lookup import _qrz_session_key, _qrz_session_expires
    import callsign_lookup

    data = await request.json()
    username = data.get("username", "").strip().upper()
    password = data.get("password")

    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    # If password not provided, keep existing
    if not password:
        existing_password = get_setting("qrz_password", decrypt=True)
        if not existing_password:
            raise HTTPException(status_code=400, detail="Password is required")
        password = existing_password

    # Store encrypted
    set_setting("qrz_username", username, encrypt=True)
    set_setting("qrz_password", password, encrypt=True)

    # Clear cached session to force re-auth with new credentials
    callsign_lookup._qrz_session_key = None
    callsign_lookup._qrz_session_expires = None

    return {"message": "QRZ credentials saved successfully"}


@app.post("/admin/settings/qrz/test")
async def test_qrz_connection(_: bool = Depends(verify_admin)):
    """Test QRZ API connection with stored credentials."""
    from callsign_lookup import _get_qrz_session, lookup_callsign_qrz
    import callsign_lookup

    # Clear cached session to force fresh auth
    callsign_lookup._qrz_session_key = None
    callsign_lookup._qrz_session_expires = None

    # Try to get a session
    session = await _get_qrz_session()
    if not session:
        raise HTTPException(status_code=400, detail="Failed to authenticate with QRZ. Check credentials.")

    # Try a test lookup
    info = await lookup_callsign_qrz("W1AW")  # ARRL HQ callsign
    if info:
        return {"message": f"Connection successful! Test lookup: {info.first_name} ({info.country})"}
    else:
        return {"message": "Authenticated successfully, but test lookup returned no data"}


@app.delete("/admin/settings/qrz")
async def clear_qrz_settings(_: bool = Depends(verify_admin)):
    """Clear QRZ API credentials."""
    from database import set_setting
    import callsign_lookup

    set_setting("qrz_username", None)
    set_setting("qrz_password", None)

    # Clear cached session
    callsign_lookup._qrz_session_key = None
    callsign_lookup._qrz_session_expires = None

    return {"message": "QRZ credentials cleared"}


@app.post("/admin/settings/theme")
async def update_theme_settings(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update site theme settings."""
    from database import set_setting

    data = await request.json()
    theme = data.get("theme", "olympics")
    name = data.get("name", "").strip()
    tagline = data.get("tagline", "").strip()

    # Validate theme
    valid_themes = ["olympics", "coolcontest", "neon", "midnight"]
    if theme not in valid_themes:
        raise HTTPException(status_code=400, detail=f"Invalid theme. Must be one of: {', '.join(valid_themes)}")

    # Save settings
    set_setting("site_theme", theme)
    if name:
        set_setting("site_name", name)
    if tagline:
        set_setting("site_tagline", tagline)

    return {"message": "Theme settings saved successfully"}


@app.post("/admin/settings/public-results")
async def update_public_results(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update public results setting."""
    from database import set_setting

    data = await request.json()
    enabled = data.get("enabled", False)

    set_setting("public_results", "1" if enabled else "0")

    return {"message": f"Public results {'enabled' if enabled else 'disabled'}"}


@app.post("/admin/settings/email")
async def update_email_settings(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update email notification settings."""
    from database import set_setting

    data = await request.json()
    admin_bcc = data.get("admin_bcc_emails", False)

    set_setting("admin_bcc_emails", "1" if admin_bcc else "0")

    status = "enabled" if admin_bcc else "disabled"
    return {"message": f"Admin BCC on emails {status}"}


@app.post("/admin/settings/discord")
async def update_discord_settings(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update Discord webhook URL."""
    from database import set_setting

    data = await request.json()
    webhook_url = data.get("webhook_url", "").strip()

    if webhook_url and not webhook_url.startswith("https://discord.com/api/webhooks/"):
        raise HTTPException(status_code=400, detail="Invalid Discord webhook URL format")

    if webhook_url:
        set_setting("discord_webhook_url", webhook_url)
        return {"message": "Discord webhook URL saved successfully"}
    else:
        set_setting("discord_webhook_url", None)
        return {"message": "Discord webhook URL cleared"}


@app.post("/admin/settings/discord/test")
async def test_discord_webhook(_: bool = Depends(verify_admin)):
    """Test Discord webhook connection."""
    from notifications import test_discord_webhook

    result = test_discord_webhook()
    if result["success"]:
        return {"message": result["message"]}
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.delete("/admin/settings/discord")
async def clear_discord_settings(_: bool = Depends(verify_admin)):
    """Clear Discord webhook URL."""
    from database import set_setting

    set_setting("discord_webhook_url", None)
    return {"message": "Discord webhook URL cleared"}


@app.post("/admin/settings/discord/options")
async def update_discord_options(
    request: Request,
    _: bool = Depends(verify_admin)
):
    """Update Discord notification options."""
    from database import set_setting

    data = await request.json()

    # Save each option
    set_setting("discord_notify_signups", "1" if data.get("notify_signups", True) else "0")
    set_setting("discord_notify_records", "1" if data.get("notify_records", True) else "0")
    set_setting("discord_notify_medals", "1" if data.get("notify_medals", True) else "0")
    set_setting("discord_notify_pota", "1" if data.get("notify_pota", True) else "0")

    # Save POTA interval (default 30 minutes)
    pota_interval = data.get("pota_interval", 30)
    if pota_interval not in [15, 30, 60, 120, 240]:
        pota_interval = 30
    set_setting("discord_pota_interval", str(pota_interval))

    return {"message": "Discord notification options saved"}


# ============================================================
# ADMIN PUSH NOTIFICATION ENDPOINTS
# ============================================================

@app.post("/admin/notifications/send-reminders")
async def admin_send_match_reminders(_: bool = Depends(verify_admin)):
    """Send match reminder notifications for upcoming matches."""
    from notifications import send_match_reminders

    results = await asyncio.to_thread(send_match_reminders)

    return {
        "message": "Match reminders processed",
        "matches_checked": results["checked"],
        "reminders_sent": results["sent"],
        "reminders_skipped": results["skipped"]
    }


@app.get("/admin/notifications/check-pota-spots")
async def admin_check_pota_spots(_: bool = Depends(verify_admin)):
    """Check POTA spots for active matches and send notifications."""
    from notifications import check_pota_spots_and_notify

    results = await check_pota_spots_and_notify()

    return {
        "message": "POTA spots checked",
        "spots_checked": results["spots_checked"],
        "matches_with_spots": results["matches_with_spots"],
        "notifications_sent": results["notifications_sent"],
        "errors": results["errors"]
    }


@app.post("/admin/notifications/cleanup")
async def admin_cleanup_notifications(_: bool = Depends(verify_admin)):
    """Cleanup old notification records and stale subscriptions."""
    from notifications import cleanup_old_notifications, cleanup_stale_subscriptions

    old_deleted = await asyncio.to_thread(cleanup_old_notifications, 30)
    stale_deleted = await asyncio.to_thread(cleanup_stale_subscriptions, 90)

    return {
        "message": "Notification cleanup complete",
        "old_notifications_deleted": old_deleted,
        "stale_subscriptions_deleted": stale_deleted
    }


@app.get("/admin/notifications/stats")
async def admin_notification_stats(_: bool = Depends(verify_admin)):
    """Get push notification statistics."""
    with get_db() as conn:
        # Count subscriptions
        sub_count = conn.execute("SELECT COUNT(*) FROM push_subscriptions").fetchone()[0]

        # Count subscribers (unique callsigns)
        subscriber_count = conn.execute(
            "SELECT COUNT(DISTINCT callsign) FROM push_subscriptions"
        ).fetchone()[0]

        # Count sent notifications
        sent_count = conn.execute("SELECT COUNT(*) FROM sent_notifications").fetchone()[0]

        # Count recent notifications (last 24 hours)
        recent_count = conn.execute("""
            SELECT COUNT(*) FROM sent_notifications
            WHERE sent_at > datetime('now', '-24 hours')
        """).fetchone()[0]

    return {
        "total_subscriptions": sub_count,
        "unique_subscribers": subscriber_count,
        "total_sent": sent_count,
        "sent_last_24h": recent_count
    }


@app.post("/admin/recompute-records")
async def admin_recompute_records(_: bool = Depends(verify_admin)):
    """Recompute all medals and world records from match-qualifying QSOs."""
    from scoring import recompute_all_records
    from sync import recompute_all_active_matches
    from notifications import send_pending_push_disabled_emails

    # Run in thread pool to avoid blocking the event loop
    await asyncio.to_thread(recompute_all_active_matches)
    await asyncio.to_thread(recompute_all_records)

    # Send any pending push disabled notification emails
    await send_pending_push_disabled_emails()

    return {"message": "Medals and world records recomputed successfully"}


@app.post("/admin/merge-duplicates")
async def admin_merge_duplicates(_: bool = Depends(verify_admin)):
    """Merge duplicate QSOs from multiple sources (QRZ + LoTW)."""
    from sync import merge_duplicate_qsos

    result = merge_duplicate_qsos()

    if result["duplicate_groups"] > 0:
        return {"message": f"Merged {result['duplicate_groups']} duplicate groups, deleted {result['qsos_deleted']} duplicate QSOs"}
    else:
        return {"message": "No duplicate QSOs found"}


@app.post("/admin/send-medal-notifications")
async def admin_send_medal_notifications(_: bool = Depends(verify_admin)):
    """Send email notifications for new medals."""
    from email_service import notify_new_medals

    result = await notify_new_medals()

    if result["sent"] > 0:
        return {"message": f"Sent {result['sent']} medal notifications, skipped {result['skipped']}, errors {result['errors']}"}
    elif result["skipped"] > 0:
        return {"message": f"No notifications sent, skipped {result['skipped']} (no email or disabled)"}
    else:
        return {"message": "No new medals to notify"}


@app.post("/admin/send-match-reminders")
async def admin_send_match_reminders(_: bool = Depends(verify_admin), hours: int = 24):
    """Send email reminders for matches starting soon."""
    from email_service import send_match_reminders

    result = await send_match_reminders(hours_before=hours)

    if result["sent"] > 0:
        return {"message": f"Sent {result['sent']} match reminders, errors {result['errors']}"}
    else:
        return {"message": "No upcoming matches to remind about"}


@app.post("/admin/toggle-auto-sync")
async def admin_toggle_auto_sync(_: bool = Depends(verify_admin)):
    """Toggle the auto-sync on or off."""
    current = is_sync_paused()
    set_sync_paused(not current)
    new_state = not current
    status = "paused" if new_state else "running"
    logger.info(f"Auto-sync toggled to: {status}")
    return {"paused": new_state, "message": f"Auto-sync is now {status}"}


@app.get("/admin/auto-sync-status")
async def admin_auto_sync_status(_: bool = Depends(verify_admin)):
    """Get the current auto-sync status."""
    return {"paused": is_sync_paused()}


@app.get("/admin/teams", response_class=HTMLResponse)
async def admin_teams_page(
    request: Request,
    _: bool = Depends(verify_admin),
    page: int = 1,
    search: str = ""
):
    """Admin teams management page."""
    per_page = 20
    offset = (max(1, page) - 1) * per_page

    with get_db() as conn:
        # Build query
        base_where = "WHERE 1=1"
        params = []

        if search:
            base_where += " AND (t.name LIKE ? OR t.captain_callsign LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        # Get total count
        cursor = conn.execute(f"SELECT COUNT(*) FROM teams t {base_where}", params)
        total = cursor.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        # Get teams with member counts
        cursor = conn.execute(f"""
            SELECT t.*,
                   COUNT(DISTINCT tm.callsign) as member_count,
                   COALESCE(SUM(m.total_points), 0) as total_points
            FROM teams t
            LEFT JOIN team_members tm ON t.id = tm.team_id
            LEFT JOIN medals m ON tm.callsign = m.callsign
            {base_where}
            GROUP BY t.id
            ORDER BY t.created_at DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset])
        teams = cursor.fetchall()

        # Get all competitors for adding to teams
        cursor = conn.execute("""
            SELECT c.callsign, c.first_name, c.last_name
            FROM competitors c
            WHERE c.is_disabled = 0
            ORDER BY c.callsign
        """)
        competitors = cursor.fetchall()

    return templates.TemplateResponse("admin/teams.html", {
        "request": request,
        "user": get_current_user(request),
        "teams": teams,
        "competitors": competitors,
        "search": search,
        "csrf_token": request.state.csrf_token,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages
        }
    })


@app.post("/admin/team")
async def admin_create_team(
    request: Request,
    _: bool = Depends(verify_admin),
    name: str = Form(...),
    description: str = Form(""),
    captain_callsign: str = Form(...)
):
    """Admin create team with specified captain."""
    captain_callsign = captain_callsign.upper().strip()

    with get_db() as conn:
        # Validate captain exists
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (captain_callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=400, detail="Captain callsign not found")

        # Check if team name is taken
        cursor = conn.execute("SELECT 1 FROM teams WHERE name = ?", (name,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Team name already exists")

        # Check if captain is already on a team
        cursor = conn.execute("SELECT t.name FROM teams t JOIN team_members tm ON t.id = tm.team_id WHERE tm.callsign = ?", (captain_callsign,))
        existing = cursor.fetchone()
        if existing:
            raise HTTPException(status_code=400, detail=f"{captain_callsign} is already on team '{existing['name']}'")

        # Create team
        now = datetime.utcnow().isoformat()
        cursor = conn.execute("""
            INSERT INTO teams (name, description, captain_callsign, created_at, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (name.strip(), description.strip() or None, captain_callsign, now))
        team_id = cursor.lastrowid

        # Add captain as member
        conn.execute("""
            INSERT INTO team_members (team_id, callsign, joined_at)
            VALUES (?, ?, ?)
        """, (team_id, captain_callsign, now))

        # Audit log
        user = get_current_user(request)
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'admin_create_team', 'team', ?, ?)
        """, (now, user.callsign if user else None, str(team_id), f"Created team: {name}"))

    # Recompute team standings
    recompute_all_team_standings()

    return RedirectResponse(url="/admin/teams", status_code=303)


@app.delete("/admin/team/{team_id}")
async def admin_delete_team(request: Request, team_id: int, _: bool = Depends(verify_admin)):
    """Admin delete team."""
    with get_db() as conn:
        cursor = conn.execute("SELECT name FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))

        # Audit log
        user = get_current_user(request)
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'admin_delete_team', 'team', ?, ?)
        """, (now, user.callsign if user else None, str(team_id), f"Deleted team: {team['name']}"))

    return {"message": "Team deleted"}


@app.post("/admin/team/{team_id}/add/{callsign}")
async def admin_add_team_member(request: Request, team_id: int, callsign: str, _: bool = Depends(verify_admin)):
    """Admin add member to team."""
    callsign = callsign.upper().strip()

    with get_db() as conn:
        # Check team exists
        cursor = conn.execute("SELECT name FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check user exists
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")

        # Check if already on a team
        cursor = conn.execute("""
            SELECT t.name FROM teams t
            JOIN team_members tm ON t.id = tm.team_id
            WHERE tm.callsign = ?
        """, (callsign,))
        existing = cursor.fetchone()
        if existing:
            raise HTTPException(status_code=400, detail=f"{callsign} is already on team '{existing['name']}'")

        # Add member
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO team_members (team_id, callsign, joined_at)
            VALUES (?, ?, ?)
        """, (team_id, callsign, now))

        # Audit log
        user = get_current_user(request)
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'admin_add_team_member', 'team', ?, ?)
        """, (now, user.callsign if user else None, str(team_id), f"Added {callsign} to team"))

    # Recompute team standings
    recompute_all_team_standings()

    return {"message": f"{callsign} added to team"}


@app.post("/admin/team/{team_id}/remove/{callsign}")
async def admin_remove_team_member(request: Request, team_id: int, callsign: str, _: bool = Depends(verify_admin)):
    """Admin remove member from team."""
    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute("SELECT name, captain_callsign FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check member exists on team
        cursor = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Member not on this team")

        # If removing captain, need to reassign or fail
        if team["captain_callsign"] == callsign:
            raise HTTPException(status_code=400, detail="Cannot remove captain. Transfer captaincy first.")

        # Remove member
        conn.execute("DELETE FROM team_members WHERE team_id = ? AND callsign = ?", (team_id, callsign))

        # Audit log
        user = get_current_user(request)
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'admin_remove_team_member', 'team', ?, ?)
        """, (now, user.callsign if user else None, str(team_id), f"Removed {callsign} from team"))

    # Recompute team standings
    recompute_all_team_standings()

    return {"message": f"{callsign} removed from team"}


@app.post("/admin/team/{team_id}/transfer/{callsign}")
async def admin_transfer_captaincy(request: Request, team_id: int, callsign: str, _: bool = Depends(verify_admin)):
    """Admin transfer team captaincy."""
    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute("SELECT name, captain_callsign FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check new captain is on team
        cursor = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Member not on this team")

        conn.execute("UPDATE teams SET captain_callsign = ? WHERE id = ?", (callsign, team_id))

        # Audit log
        user = get_current_user(request)
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'admin_transfer_captaincy', 'team', ?, ?)
        """, (now, user.callsign if user else None, str(team_id), f"Transferred captaincy from {team['captain_callsign']} to {callsign}"))

    return {"message": f"Captaincy transferred to {callsign}"}


# ============================================================
# TEAM ENDPOINTS
# ============================================================

def get_user_team(conn, callsign: str):
    """Get the team a user belongs to, if any."""
    cursor = conn.execute("""
        SELECT t.*, tm.joined_at as member_joined_at
        FROM teams t
        JOIN team_members tm ON t.id = tm.team_id
        WHERE tm.callsign = ?
    """, (callsign,))
    return cursor.fetchone()


def recompute_all_team_standings():
    """Recompute team standings for all sports and matches."""
    with get_db() as conn:
        sports = conn.execute("SELECT id FROM sports").fetchall()
        for sport in sports:
            sport_id = sport[0]
            compute_team_standings(sport_id)
            # Also compute per-match standings
            matches = conn.execute("SELECT id FROM matches WHERE sport_id = ?", (sport_id,)).fetchall()
            for match in matches:
                compute_team_standings(sport_id, match[0])


def get_team_members(conn, team_id: int):
    """Get all members of a team with their stats."""
    cursor = conn.execute("""
        SELECT c.callsign, c.first_name, c.last_name, tm.joined_at,
               t.captain_callsign,
               COALESCE(SUM(m.total_points), 0) as total_points,
               COUNT(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 END) as gold,
               COUNT(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 END) as silver,
               COUNT(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 END) as bronze
        FROM team_members tm
        JOIN competitors c ON tm.callsign = c.callsign
        JOIN teams t ON tm.team_id = t.id
        LEFT JOIN medals m ON c.callsign = m.callsign
        WHERE tm.team_id = ?
        GROUP BY c.callsign
        ORDER BY total_points DESC, c.callsign
    """, (team_id,))
    return cursor.fetchall()


@app.get("/teams", response_class=HTMLResponse)
async def teams_list_page(request: Request, page: int = 1, per_page: int = 20, user: Optional[User] = Depends(require_user_or_public)):
    """List all active teams."""
    offset = (max(1, page) - 1) * per_page

    with get_db() as conn:
        # Get total count
        cursor = conn.execute("SELECT COUNT(*) FROM teams WHERE is_active = 1")
        total = cursor.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        # Get teams with member counts and total points
        cursor = conn.execute("""
            SELECT t.*,
                   c.first_name as captain_first_name,
                   COUNT(DISTINCT tm.callsign) as member_count,
                   COALESCE(SUM(m.total_points), 0) as total_points,
                   COUNT(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 END) as gold,
                   COUNT(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 END) as silver,
                   COUNT(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 END) as bronze
            FROM teams t
            LEFT JOIN competitors c ON t.captain_callsign = c.callsign
            LEFT JOIN team_members tm ON t.id = tm.team_id
            LEFT JOIN medals m ON tm.callsign = m.callsign
            WHERE t.is_active = 1
            GROUP BY t.id
            ORDER BY total_points DESC, t.name
            LIMIT ? OFFSET ?
        """, (per_page, offset))
        teams = cursor.fetchall()

        # Get user's current team if any
        user_team = None
        if user:
            user_team = get_user_team(conn, user.callsign)

        # Get per-sport medal breakdown for each team (for collapsible details)
        team_ids = [t["id"] for t in teams]
        team_sport_medals = {}
        if team_ids:
            placeholders = ",".join("?" * len(team_ids))
            cursor = conn.execute(f"""
                SELECT tm.team_id, s.id as sport_id, s.name as sport_name,
                       tm.total_points, tm.gold_count, tm.silver_count, tm.bronze_count, tm.medal
                FROM team_medals tm
                JOIN sports s ON tm.sport_id = s.id
                WHERE tm.team_id IN ({placeholders})
                  AND tm.match_id IS NULL
                  AND tm.calculation_method = 'normalized'
                  AND (tm.gold_count > 0 OR tm.silver_count > 0 OR tm.bronze_count > 0)
                ORDER BY tm.total_points DESC
            """, team_ids)
            for row in cursor.fetchall():
                team_id = row["team_id"]
                if team_id not in team_sport_medals:
                    team_sport_medals[team_id] = []
                team_sport_medals[team_id].append(dict(row))

        # Get medal-winning QSOs for each team's sports
        team_medal_qsos = {}
        if team_ids:
            from scoring import matches_target, is_mode_allowed

            # Get team members for each team
            cursor = conn.execute(f"""
                SELECT tm.team_id, tm.callsign, c.first_name
                FROM team_members tm
                JOIN competitors c ON tm.callsign = c.callsign
                WHERE tm.team_id IN ({placeholders})
            """, team_ids)
            team_members_map = {}
            all_member_callsigns = set()
            for row in cursor.fetchall():
                tid = row["team_id"]
                if tid not in team_members_map:
                    team_members_map[tid] = {}
                team_members_map[tid][row["callsign"]] = row["first_name"]
                all_member_callsigns.add(row["callsign"])

            if all_member_callsigns:
                member_placeholders = ",".join("?" * len(all_member_callsigns))
                member_list = list(all_member_callsigns)

                # Get medals for team members
                cursor = conn.execute(f"""
                    SELECT m.callsign, m.match_id, m.qso_race_medal, m.cool_factor_medal,
                           ma.target_value, ma.start_date, ma.end_date,
                           s.id as sport_id, s.target_type, s.work_enabled, s.activate_enabled,
                           COALESCE(ma.allowed_modes, s.allowed_modes) as allowed_modes
                    FROM medals m
                    JOIN matches ma ON m.match_id = ma.id
                    JOIN sports s ON ma.sport_id = s.id
                    WHERE m.callsign IN ({member_placeholders})
                      AND (m.qso_race_medal IS NOT NULL OR m.cool_factor_medal IS NOT NULL)
                """, member_list)
                member_medals = cursor.fetchall()

                # Get QSOs for team members
                cursor = conn.execute(f"""
                    SELECT q.competitor_callsign, q.dx_callsign, q.qso_datetime_utc,
                           q.mode, q.tx_power_w, q.distance_km, q.cool_factor,
                           q.dx_sig_info, q.my_sig_info, q.dx_dxcc, q.my_dxcc, q.my_grid
                    FROM qsos q
                    WHERE q.competitor_callsign IN ({member_placeholders})
                      AND q.is_confirmed = 1
                    ORDER BY q.qso_datetime_utc ASC
                """, member_list)
                all_qsos = [dict(row) for row in cursor.fetchall()]

                # Index QSOs by callsign
                qsos_by_callsign = {}
                for qso in all_qsos:
                    cs = qso["competitor_callsign"]
                    if cs not in qsos_by_callsign:
                        qsos_by_callsign[cs] = []
                    qsos_by_callsign[cs].append(qso)

                # Build medal QSOs per team per sport
                for row in member_medals:
                    callsign = row["callsign"]
                    sport_id = row["sport_id"]
                    start_date = row["start_date"][:10] if row["start_date"] else None
                    end_date = row["end_date"][:10] if row["end_date"] else None
                    target_value = row["target_value"]
                    target_type = row["target_type"]
                    work_enabled = row["work_enabled"]
                    activate_enabled = row["activate_enabled"]
                    allowed_modes = row["allowed_modes"]

                    # Find which team this member belongs to
                    member_team_id = None
                    member_first_name = None
                    for tid, members in team_members_map.items():
                        if callsign in members:
                            member_team_id = tid
                            member_first_name = members[callsign]
                            break

                    if not member_team_id:
                        continue

                    # Find matching QSOs
                    matching_qsos = []
                    for qso in qsos_by_callsign.get(callsign, []):
                        qso_date = qso["qso_datetime_utc"][:10] if qso["qso_datetime_utc"] else None
                        if qso_date and start_date and end_date and start_date <= qso_date <= end_date:
                            if not is_mode_allowed(qso.get("mode"), allowed_modes):
                                continue
                            matches_work = work_enabled and matches_target(qso, target_type, target_value, "work")
                            matches_activate = activate_enabled and matches_target(qso, target_type, target_value, "activate")
                            if matches_work or matches_activate:
                                matching_qsos.append(qso)

                    # Build medal QSOs
                    if member_team_id not in team_medal_qsos:
                        team_medal_qsos[member_team_id] = {}
                    if sport_id not in team_medal_qsos[member_team_id]:
                        team_medal_qsos[member_team_id][sport_id] = []

                    # QSO Race medal
                    if row["qso_race_medal"] and matching_qsos:
                        earliest = min(matching_qsos, key=lambda q: q["qso_datetime_utc"] or "")
                        team_medal_qsos[member_team_id][sport_id].append({
                            "callsign": callsign,
                            "first_name": member_first_name,
                            "dx_callsign": earliest["dx_callsign"],
                            "time": earliest["qso_datetime_utc"],
                            "mode": earliest["mode"],
                            "power": earliest["tx_power_w"],
                            "distance": earliest["distance_km"],
                            "cool_factor": earliest["cool_factor"],
                            "medal_type": "Race",
                            "medal": row["qso_race_medal"],
                        })

                    # Cool Factor medal
                    if row["cool_factor_medal"] and matching_qsos:
                        best_cf = max((q for q in matching_qsos if q.get("cool_factor")),
                                      key=lambda q: q["cool_factor"] or 0, default=None)
                        if best_cf:
                            team_medal_qsos[member_team_id][sport_id].append({
                                "callsign": callsign,
                                "first_name": member_first_name,
                                "dx_callsign": best_cf["dx_callsign"],
                                "time": best_cf["qso_datetime_utc"],
                                "mode": best_cf["mode"],
                                "power": best_cf["tx_power_w"],
                                "distance": best_cf["distance_km"],
                                "cool_factor": best_cf["cool_factor"],
                                "medal_type": "CF",
                                "medal": row["cool_factor_medal"],
                            })

        display_prefs = get_display_prefs(user)

        return templates.TemplateResponse("teams.html", {
            "request": request,
            "teams": teams,
            "team_sport_medals": team_sport_medals,
            "team_medal_qsos": team_medal_qsos,
            "display_prefs": display_prefs,
            "user_team": user_team,
            "user": user,
            "csrf_token": request.state.csrf_token,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages
            }
        })


@app.get("/team/{team_id}", response_class=HTMLResponse)
async def team_profile_page(request: Request, team_id: int, user: Optional[User] = Depends(require_user_or_public)):
    """Team profile page."""

    with get_db() as conn:
        # Get team
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Get members with stats
        members = get_team_members(conn, team_id)

        # Get captain info
        cursor = conn.execute(
            "SELECT callsign, first_name, last_name FROM competitors WHERE callsign = ?",
            (team["captain_callsign"],)
        )
        captain = cursor.fetchone()

        # Get team standings per sport (using normalized calculation)
        cursor = conn.execute("""
            SELECT tm.*, s.name as sport_name
            FROM team_medals tm
            JOIN sports s ON tm.sport_id = s.id
            WHERE tm.team_id = ? AND tm.match_id IS NULL AND tm.calculation_method = 'normalized'
            ORDER BY tm.total_points DESC
        """, (team_id,))
        sport_standings = cursor.fetchall()

        # Check if current user is a member or captain
        is_member = False
        is_captain = False
        has_pending_request = False
        if user:
            cursor = conn.execute(
                "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
                (team_id, user.callsign)
            )
            is_member = cursor.fetchone() is not None
            is_captain = team["captain_callsign"] == user.callsign

            # Check if user has pending request
            cursor = conn.execute(
                "SELECT 1 FROM team_invites WHERE team_id = ? AND callsign = ? AND invite_type = 'request'",
                (team_id, user.callsign)
            )
            has_pending_request = cursor.fetchone() is not None

        # Get pending requests (for captain)
        pending_requests = []
        pending_invites = []
        if is_captain:
            cursor = conn.execute("""
                SELECT ti.*, c.first_name, c.last_name
                FROM team_invites ti
                JOIN competitors c ON ti.callsign = c.callsign
                WHERE ti.team_id = ? AND ti.invite_type = 'request'
                ORDER BY ti.created_at DESC
            """, (team_id,))
            pending_requests = cursor.fetchall()

            cursor = conn.execute("""
                SELECT ti.*, c.first_name, c.last_name
                FROM team_invites ti
                JOIN competitors c ON ti.callsign = c.callsign
                WHERE ti.team_id = ? AND ti.invite_type = 'invite'
                ORDER BY ti.created_at DESC
            """, (team_id,))
            pending_invites = cursor.fetchall()

        # Get all competitors for invite dropdown (captain only)
        available_competitors = []
        if is_captain:
            cursor = conn.execute("""
                SELECT c.callsign, c.first_name, c.last_name
                FROM competitors c
                WHERE c.is_disabled = 0
                AND c.callsign NOT IN (SELECT callsign FROM team_members)
                AND c.callsign NOT IN (SELECT callsign FROM team_invites WHERE team_id = ?)
                ORDER BY c.callsign
            """, (team_id,))
            available_competitors = cursor.fetchall()

        return templates.TemplateResponse("team_profile.html", {
            "request": request,
            "team": team,
            "captain": captain,
            "members": members,
            "sport_standings": sport_standings,
            "is_member": is_member,
            "is_captain": is_captain,
            "has_pending_request": has_pending_request,
            "pending_requests": pending_requests,
            "pending_invites": pending_invites,
            "available_competitors": available_competitors,
            "user": user,
            "csrf_token": request.state.csrf_token
        })


@app.post("/team")
async def create_team(request: Request, data: CreateTeamRequest, user: User = Depends(require_user)):
    """Create a new team. Creator becomes captain."""

    with get_db() as conn:
        # Check if user is already on a team
        existing_team = get_user_team(conn, user.callsign)
        if existing_team:
            raise HTTPException(status_code=400, detail="You are already on a team. Leave your current team first.")

        # Check if team name is taken
        cursor = conn.execute("SELECT 1 FROM teams WHERE name = ?", (data.name,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Team name already exists")

        # Create team
        now = datetime.utcnow().isoformat()
        cursor = conn.execute("""
            INSERT INTO teams (name, description, captain_callsign, created_at, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (data.name, data.description, user.callsign, now))
        team_id = cursor.lastrowid

        # Add creator as member
        conn.execute("""
            INSERT INTO team_members (team_id, callsign, joined_at)
            VALUES (?, ?, ?)
        """, (team_id, user.callsign, now))

        # Log action
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'create_team', 'team', ?, ?)
        """, (now, user.callsign, str(team_id), f"Created team: {data.name}"))

    # Recompute team standings
    recompute_all_team_standings()

    return {"message": "Team created successfully", "team_id": team_id}


@app.put("/team/{team_id}")
async def update_team(request: Request, team_id: int, data: UpdateTeamRequest, user: User = Depends(require_user)):
    """Update team info. Captain only."""

    with get_db() as conn:
        # Check team exists and user is captain
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can update team info")

        # Check for name collision
        if data.name and data.name != team["name"]:
            cursor = conn.execute("SELECT 1 FROM teams WHERE name = ? AND id != ?", (data.name, team_id))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Team name already exists")

        # Update
        updates = []
        params = []
        if data.name:
            updates.append("name = ?")
            params.append(data.name)
        if data.description is not None:
            updates.append("description = ?")
            params.append(data.description)

        if updates:
            params.append(team_id)
            conn.execute(f"UPDATE teams SET {', '.join(updates)} WHERE id = ?", params)

        return {"message": "Team updated successfully"}


@app.delete("/team/{team_id}")
async def delete_team(request: Request, team_id: int, user: User = Depends(require_user)):
    """Delete team. Captain only."""

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can delete the team")

        # Delete team (cascade will remove members and medals)
        conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))

        # Log action
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'delete_team', 'team', ?, ?)
        """, (now, user.callsign, str(team_id), f"Deleted team: {team['name']}"))

        return {"message": "Team deleted successfully"}


@app.post("/team/{team_id}/request")
async def request_to_join_team(request: Request, team_id: int, user: User = Depends(require_user)):
    """Request to join a team. Captain must approve."""

    with get_db() as conn:
        # Check team exists and is active
        cursor = conn.execute("SELECT * FROM teams WHERE id = ? AND is_active = 1", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check user isn't already on a team
        existing_team = get_user_team(conn, user.callsign)
        if existing_team:
            if existing_team["id"] == team_id:
                raise HTTPException(status_code=400, detail="You are already on this team")
            raise HTTPException(status_code=400, detail="You are already on a team. Leave your current team first.")

        # Check for existing invite (auto-accept) or request
        cursor = conn.execute(
            "SELECT invite_type FROM team_invites WHERE team_id = ? AND callsign = ?",
            (team_id, user.callsign)
        )
        existing = cursor.fetchone()
        if existing:
            if existing["invite_type"] == "invite":
                # Auto-accept the invite
                now = datetime.utcnow().isoformat()
                conn.execute("DELETE FROM team_invites WHERE team_id = ? AND callsign = ?", (team_id, user.callsign))
                conn.execute("""
                    INSERT INTO team_members (team_id, callsign, joined_at)
                    VALUES (?, ?, ?)
                """, (team_id, user.callsign, now))
                return {"message": f"You have joined {team['name']}"}
            else:
                raise HTTPException(status_code=400, detail="You already have a pending request to join this team")

        # Create join request
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO team_invites (team_id, callsign, invite_type, created_at)
            VALUES (?, ?, 'request', ?)
        """, (team_id, user.callsign, now))

        return {"message": f"Request sent to join {team['name']}. The captain must approve."}


@app.post("/team/{team_id}/invite/{callsign}")
async def invite_to_team(request: Request, team_id: int, callsign: str, user: User = Depends(require_user)):
    """Invite a user to join the team. Captain only."""

    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can invite members")

        # Check invitee exists
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")

        # Check invitee isn't already on a team
        existing_team = get_user_team(conn, callsign)
        if existing_team:
            raise HTTPException(status_code=400, detail=f"{callsign} is already on a team")

        # Check for existing request (auto-approve) or invite
        cursor = conn.execute(
            "SELECT invite_type FROM team_invites WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        existing = cursor.fetchone()
        if existing:
            if existing["invite_type"] == "request":
                # Auto-approve the request
                now = datetime.utcnow().isoformat()
                conn.execute("DELETE FROM team_invites WHERE team_id = ? AND callsign = ?", (team_id, callsign))
                conn.execute("""
                    INSERT INTO team_members (team_id, callsign, joined_at)
                    VALUES (?, ?, ?)
                """, (team_id, callsign, now))
                return {"message": f"{callsign} has been added to the team"}
            else:
                raise HTTPException(status_code=400, detail=f"{callsign} already has a pending invite")

        # Create invite
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO team_invites (team_id, callsign, invite_type, created_at)
            VALUES (?, ?, 'invite', ?)
        """, (team_id, callsign, now))

        return {"message": f"Invite sent to {callsign}"}


@app.post("/team/{team_id}/approve/{callsign}")
async def approve_join_request(request: Request, team_id: int, callsign: str, user: User = Depends(require_user)):
    """Approve a join request. Captain only."""

    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can approve requests")

        # Check request exists
        cursor = conn.execute(
            "SELECT 1 FROM team_invites WHERE team_id = ? AND callsign = ? AND invite_type = 'request'",
            (team_id, callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="No pending request from this user")

        # Check they're still not on a team
        existing_team = get_user_team(conn, callsign)
        if existing_team:
            conn.execute("DELETE FROM team_invites WHERE team_id = ? AND callsign = ?", (team_id, callsign))
            raise HTTPException(status_code=400, detail=f"{callsign} has already joined another team")

        # Approve - add to team
        now = datetime.utcnow().isoformat()
        conn.execute("DELETE FROM team_invites WHERE team_id = ? AND callsign = ?", (team_id, callsign))
        conn.execute("""
            INSERT INTO team_members (team_id, callsign, joined_at)
            VALUES (?, ?, ?)
        """, (team_id, callsign, now))

    # Recompute team standings
    recompute_all_team_standings()

    return {"message": f"{callsign} has been added to the team"}


@app.post("/team/{team_id}/decline/{callsign}")
async def decline_join_request(request: Request, team_id: int, callsign: str, user: User = Depends(require_user)):
    """Decline a join request or cancel an invite. Captain only."""

    callsign = callsign.upper().strip()

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can decline requests")

        # Delete the invite/request
        cursor = conn.execute(
            "DELETE FROM team_invites WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No pending request or invite found")

        return {"message": "Request declined"}


@app.post("/team/{team_id}/accept")
async def accept_team_invite(request: Request, team_id: int, user: User = Depends(require_user)):
    """Accept an invite to join a team."""

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check invite exists
        cursor = conn.execute(
            "SELECT 1 FROM team_invites WHERE team_id = ? AND callsign = ? AND invite_type = 'invite'",
            (team_id, user.callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="No pending invite for this team")

        # Check user isn't already on a team
        existing_team = get_user_team(conn, user.callsign)
        if existing_team:
            conn.execute("DELETE FROM team_invites WHERE callsign = ?", (user.callsign,))
            raise HTTPException(status_code=400, detail="You are already on a team")

        # Accept - add to team
        now = datetime.utcnow().isoformat()
        conn.execute("DELETE FROM team_invites WHERE team_id = ? AND callsign = ?", (team_id, user.callsign))
        conn.execute("""
            INSERT INTO team_members (team_id, callsign, joined_at)
            VALUES (?, ?, ?)
        """, (team_id, user.callsign, now))

        return {"message": f"You have joined {team['name']}"}


@app.post("/team/{team_id}/reject")
async def reject_team_invite(request: Request, team_id: int, user: User = Depends(require_user)):
    """Reject an invite to join a team."""

    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM team_invites WHERE team_id = ? AND callsign = ? AND invite_type = 'invite'",
            (team_id, user.callsign)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="No pending invite for this team")

        return {"message": "Invite declined"}


@app.post("/team/{team_id}/leave")
async def leave_team(request: Request, team_id: int, user: User = Depends(require_user)):
    """Leave a team."""

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # Check user is on this team
        cursor = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, user.callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=400, detail="You are not on this team")

        # Captain cannot leave - must transfer or delete team
        if team["captain_callsign"] == user.callsign:
            raise HTTPException(
                status_code=400,
                detail="Captain cannot leave the team. Transfer captaincy or delete the team."
            )

        # Leave team
        conn.execute(
            "DELETE FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, user.callsign)
        )

        # Recompute team standings
        recompute_all_team_standings()

        # Log action
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'leave_team', 'team', ?, ?)
        """, (now, user.callsign, str(team_id), f"Left team: {team['name']}"))

        return {"message": f"You have left {team['name']}"}


@app.post("/team/{team_id}/remove/{callsign}")
async def remove_team_member(request: Request, team_id: int, callsign: str, user: User = Depends(require_user)):
    """Remove a member from the team. Captain only."""

    callsign = callsign.upper()

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can remove members")

        # Cannot remove captain
        if callsign == team["captain_callsign"]:
            raise HTTPException(status_code=400, detail="Cannot remove the captain")

        # Check member exists
        cursor = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Member not found on this team")

        # Remove member
        conn.execute(
            "DELETE FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )

        # Recompute team standings
        recompute_all_team_standings()

        # Log action
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'remove_team_member', 'team', ?, ?)
        """, (now, user.callsign, str(team_id), f"Removed {callsign} from team"))

        return {"message": f"{callsign} has been removed from the team"}


@app.post("/team/{team_id}/transfer/{callsign}")
async def transfer_captaincy(request: Request, team_id: int, callsign: str, user: User = Depends(require_user)):
    """Transfer team captaincy to another member. Captain only."""

    callsign = callsign.upper()

    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if team["captain_callsign"] != user.callsign and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the captain can transfer captaincy")

        # Check new captain is a member
        cursor = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND callsign = ?",
            (team_id, callsign)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Member not found on this team")

        # Transfer captaincy
        conn.execute("UPDATE teams SET captain_callsign = ? WHERE id = ?", (callsign, team_id))

        # Log action
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details)
            VALUES (?, ?, 'transfer_captaincy', 'team', ?, ?)
        """, (now, user.callsign, str(team_id), f"Transferred captaincy to {callsign}"))

        return {"message": f"Captaincy transferred to {callsign}"}


# ============================================================
# API v1 ENDPOINTS
# ============================================================

def require_api_auth(request: Request) -> User:
    """Require authentication for API endpoints, return JSON error if not authenticated."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@app.get("/api/v1/health")
@limiter.limit("60/minute")
async def api_health(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


def _normalize_park_reference(ref: str) -> str:
    """Normalize a POTA park reference to standard format (zero-padded to 4 digits)."""
    import re
    if not ref:
        return ref
    ref = ref.upper().strip()
    match = re.match(r'^([A-Z]{1,3})-(\d+)$', ref)
    if match:
        prefix = match.group(1)
        number = match.group(2).zfill(4)
        return f"{prefix}-{number}"
    return ref


@app.get("/api/v1/park/{reference}")
@limiter.limit("100/minute")
async def api_park_lookup(request: Request, reference: str):
    """Look up POTA park info with caching."""
    import httpx
    from datetime import timedelta

    original_reference = reference.upper().strip()
    reference = _normalize_park_reference(reference)
    was_normalized = (original_reference != reference)

    # Also detect if the park ID was likely normalized previously (zero-padded to 4 digits)
    # Only for 3-digit numbers (100-999) that were padded - e.g., US-0303 was probably US-303
    # Don't flag US-0001 through US-0099 since those are legitimate canonical IDs
    import re
    likely_normalized_match = re.match(r'^([A-Z]{1,3})-0([1-9]\d{2})$', reference)
    if likely_normalized_match and not was_normalized:
        # Show what the original likely was (e.g., US-303 for US-0303)
        prefix = likely_normalized_match.group(1)
        number = likely_normalized_match.group(2)
        original_reference = f"{prefix}-{number}"
        was_normalized = True

    # Check cache first (valid for 30 days)
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT name, location, grid, cached_at FROM pota_parks WHERE reference = ?",
            (reference,)
        )
        cached = cursor.fetchone()

        if cached:
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if datetime.utcnow() - cached_at < timedelta(days=30):
                return {
                    "reference": reference,
                    "original": original_reference if was_normalized else None,
                    "name": cached["name"],
                    "location": cached["location"],
                    "grid": cached["grid"],
                    "cached": True
                }

    # Fetch from POTA API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.pota.app/park/{reference}",
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                # POTA API returns empty/null for non-existent parks
                if not data or not isinstance(data, dict):
                    raise HTTPException(status_code=404, detail="Park not found")
                name = data.get("name", "Unknown")
                location = data.get("locationDesc", "")
                grid = data.get("grid", "")

                # Cache the result
                with get_db() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (reference, name, location, grid, datetime.utcnow().isoformat()))

                return {
                    "reference": reference,
                    "original": original_reference if was_normalized else None,
                    "name": name,
                    "location": location,
                    "grid": grid,
                    "cached": False
                }
            else:
                raise HTTPException(status_code=404, detail="Park not found")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Unable to reach POTA API")


@app.get("/api/v1/spots")
@limiter.limit("30/minute")
async def api_all_spots(request: Request, sport_id: Optional[int] = None):
    """Get all current POTA spots (returns list of park references with spots).

    If sport_id is provided, filters spots to only include modes allowed by that sport.
    Also returns spot details (activator, mode) for tooltip display.
    """
    import httpx
    from scoring import is_mode_allowed

    # Get allowed_modes for the sport if specified
    allowed_modes = None
    if sport_id:
        with get_db() as conn:
            sport = conn.execute(
                "SELECT allowed_modes FROM sports WHERE id = ?", (sport_id,)
            ).fetchone()
            if sport:
                allowed_modes = sport["allowed_modes"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pota.app/spot/activator",
                timeout=10.0
            )
            if response.status_code == 200:
                all_spots = response.json()
                if not all_spots:
                    return {"parks_with_spots": [], "spot_details": {}}

                # Filter by allowed modes if specified
                if allowed_modes:
                    all_spots = [
                        spot for spot in all_spots
                        if is_mode_allowed(spot.get("mode", ""), allowed_modes)
                    ]

                # Build park -> spot details mapping for tooltips
                # Group by park, showing activator and mode
                spot_details = {}
                for spot in all_spots:
                    park = spot.get("reference")
                    if not park:
                        continue
                    if park not in spot_details:
                        spot_details[park] = []
                    spot_details[park].append({
                        "activator": spot.get("activator", "Unknown"),
                        "mode": spot.get("mode", "?"),
                        "frequency": spot.get("frequency", "")
                    })

                parks_with_spots = list(spot_details.keys())

                return {"parks_with_spots": parks_with_spots, "spot_details": spot_details}
            else:
                return {"parks_with_spots": [], "spot_details": {}}
    except httpx.RequestError:
        return {"parks_with_spots": [], "spot_details": {}}


@app.get("/api/v1/park/{reference}/spots")
@limiter.limit("60/minute")
async def api_park_spots(request: Request, reference: str):
    """Get current POTA spots for a specific park."""
    import httpx

    reference = reference.upper().strip()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pota.app/spot/activator",
                timeout=10.0
            )
            if response.status_code == 200:
                all_spots = response.json()
                if not all_spots:
                    return {"spots": []}

                # Filter spots for this park
                park_spots = [
                    {
                        "activator": spot.get("activator"),
                        "frequency": spot.get("frequency"),
                        "mode": spot.get("mode"),
                        "spotter": spot.get("spotter"),
                        "spot_time": spot.get("spotTime"),
                        "comments": spot.get("comments"),
                        "qso_count": spot.get("count"),
                        "expires_in": spot.get("expire")
                    }
                    for spot in all_spots
                    if spot.get("reference") == reference
                ]

                return {"spots": park_spots, "reference": reference}
            else:
                return {"spots": [], "reference": reference}
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Unable to reach POTA API")


@app.get("/api/v1/me")
@limiter.limit("60/minute")
async def api_me(request: Request, user: User = Depends(require_api_auth)):
    """Get current user information."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT callsign, email, is_admin, registered_at FROM competitors WHERE callsign = ?",
            (user.callsign,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "callsign": row["callsign"],
            "email": row["email"],
            "is_admin": bool(row["is_admin"]),
            "created_at": row["registered_at"]
        }


@app.get("/api/v1/standings")
@limiter.limit("60/minute")
async def api_standings(request: Request, olympiad_id: Optional[int] = None):
    """Get standings for the current or specified olympiad."""
    with get_db() as conn:
        # Get olympiad
        if olympiad_id:
            cursor = conn.execute("SELECT * FROM olympiads WHERE id = ?", (olympiad_id,))
        else:
            cursor = conn.execute("SELECT * FROM olympiads WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
        olympiad = cursor.fetchone()

        if not olympiad:
            return {"standings": [], "olympiad": None}

        # Get standings with medal counts
        cursor = conn.execute("""
            SELECT c.callsign,
                   (SELECT COUNT(*) FROM qsos WHERE competitor_callsign = c.callsign) as total_qsos,
                   COALESCE(SUM(md.total_points), 0) as points,
                   COUNT(CASE WHEN md.qso_race_medal = 'gold' OR md.cool_factor_medal = 'gold' THEN 1 END) as gold,
                   COUNT(CASE WHEN md.qso_race_medal = 'silver' OR md.cool_factor_medal = 'silver' THEN 1 END) as silver,
                   COUNT(CASE WHEN md.qso_race_medal = 'bronze' OR md.cool_factor_medal = 'bronze' THEN 1 END) as bronze
            FROM competitors c
            LEFT JOIN medals md ON c.callsign = md.callsign
            LEFT JOIN matches mt ON md.match_id = mt.id
            LEFT JOIN sports s ON mt.sport_id = s.id AND s.olympiad_id = ?
            WHERE c.is_disabled = 0
            GROUP BY c.callsign
            HAVING points > 0
            ORDER BY points DESC, gold DESC, silver DESC, bronze DESC
        """, (olympiad["id"],))
        standings = [dict(row) for row in cursor.fetchall()]

        return {
            "standings": standings,
            "olympiad": {
                "id": olympiad["id"],
                "name": olympiad["name"],
                "start_date": olympiad["start_date"],
                "end_date": olympiad["end_date"]
            }
        }


@app.get("/api/v1/qsos")
@limiter.limit("60/minute")
async def api_qsos(
    request: Request,
    user: User = Depends(require_api_auth),
    page: int = 1,
    per_page: int = 50,
    band: Optional[str] = None,
    mode: Optional[str] = None,
    confirmed: Optional[bool] = None
):
    """Get QSOs for the current user."""
    offset = (max(1, page) - 1) * per_page

    with get_db() as conn:
        # Build query with filters
        query = "SELECT * FROM qsos WHERE competitor_callsign = ?"
        params = [user.callsign]

        if band:
            query += " AND band = ?"
            params.append(band)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if confirmed is not None:
            query += " AND is_confirmed = ?"
            params.append(1 if confirmed else 0)

        # Get total count
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        cursor = conn.execute(count_query, params)
        total = cursor.fetchone()[0]

        # Get paginated results
        query += " ORDER BY qso_datetime_utc DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        cursor = conn.execute(query, params)
        qsos = [dict(row) for row in cursor.fetchall()]

        return {
            "qsos": qsos,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }


@app.get("/api/v1/medals")
@limiter.limit("60/minute")
async def api_medals(request: Request, user: User = Depends(require_api_auth), olympiad_id: Optional[int] = None):
    """Get medals for the current user."""
    with get_db() as conn:
        # Get medals with match and sport info
        query = """
            SELECT m.*, s.name as sport_name, s.target_type, mt.start_date, o.name as olympiad_name
            FROM medals m
            JOIN matches mt ON m.match_id = mt.id
            JOIN sports s ON mt.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE m.callsign = ?
        """
        params = [user.callsign]

        if olympiad_id:
            query += " AND s.olympiad_id = ?"
            params.append(olympiad_id)

        query += " ORDER BY mt.start_date DESC"
        cursor = conn.execute(query, params)
        medals = [dict(row) for row in cursor.fetchall()]

        return {"medals": medals}


@app.get("/api/v1/sports")
@limiter.limit("60/minute")
async def api_sports(request: Request, olympiad_id: Optional[int] = None):
    """Get sports for the current or specified olympiad."""
    with get_db() as conn:
        if olympiad_id:
            cursor = conn.execute("SELECT * FROM sports WHERE olympiad_id = ?", (olympiad_id,))
        else:
            # Get active olympiad's sports
            cursor = conn.execute("""
                SELECT s.* FROM sports s
                JOIN olympiads o ON s.olympiad_id = o.id
                WHERE o.is_active = 1
            """)

        sports = [dict(row) for row in cursor.fetchall()]
        return {"sports": sports}


# ============================================================
# QSO DISQUALIFICATION ENDPOINTS
# ============================================================


def log_audit(actor_callsign: Optional[str], action: str, target_type: str, target_id: str, details: str, ip_address: Optional[str] = None):
    """Helper to log an audit entry."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), actor_callsign, action, target_type, target_id, details, ip_address))


@app.post("/referee/sport/{sport_id}/qso/{qso_id}/disqualify")
@limiter.limit("30/minute")
async def disqualify_qso(
    request: Request,
    sport_id: int,
    qso_id: int,
    data: QSODisqualifyRequest
):
    """
    Disqualify a QSO from a specific sport.

    Requires admin access or referee assignment for the sport.
    The QSO will not count toward medals in this sport until requalified.
    """
    verify_admin_or_sport_referee(request, sport_id)
    user = get_current_user(request)
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Verify QSO exists
        cursor = conn.execute("SELECT competitor_callsign FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            raise HTTPException(status_code=404, detail="QSO not found")

        # Verify sport exists
        cursor = conn.execute("SELECT name FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Check if already disqualified
        cursor = conn.execute(
            "SELECT id, status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
            (qso_id, sport_id)
        )
        existing = cursor.fetchone()

        if existing:
            if existing["status"] == "disqualified":
                raise HTTPException(status_code=400, detail="QSO is already disqualified for this sport")
            # Update existing record
            conn.execute(
                "UPDATE qso_disqualifications SET status = 'disqualified', updated_at = ? WHERE id = ?",
                (now, existing["id"])
            )
            dq_id = existing["id"]
        else:
            # Create new disqualification
            cursor = conn.execute(
                "INSERT INTO qso_disqualifications (qso_id, sport_id, status, created_at, updated_at) VALUES (?, ?, 'disqualified', ?, ?)",
                (qso_id, sport_id, now, now)
            )
            dq_id = cursor.lastrowid

        # Add comment
        conn.execute(
            "INSERT INTO qso_disqualification_comments (disqualification_id, author_callsign, comment_type, comment, created_at) VALUES (?, ?, 'disqualify', ?, ?)",
            (dq_id, user.callsign if user else "ADMIN", data.reason, now)
        )

    # Log audit
    log_audit(
        actor_callsign=user.callsign if user else None,
        action="qso_disqualified",
        target_type="qso",
        target_id=str(qso_id),
        details=f"Sport: {sport['name']}, Reason: {data.reason}",
        ip_address=request.client.host if request.client else None
    )

    # Recompute medals for affected matches in this sport (run in thread pool to avoid blocking/locks)
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM matches WHERE sport_id = ?", (sport_id,))
        match_ids = [row["id"] for row in cursor.fetchall()]

    for match_id in match_ids:
        await asyncio.to_thread(recompute_match_medals, match_id)

    return {
        "message": "QSO disqualified",
        "qso_id": qso_id,
        "sport_id": sport_id,
        "competitor": qso["competitor_callsign"]
    }


@app.post("/qso/{qso_id}/sport/{sport_id}/refute")
@limiter.limit("10/minute")
async def refute_qso_disqualification(
    request: Request,
    qso_id: int,
    sport_id: int,
    data: QSORefuteRequest
):
    """
    Submit a refutation for a disqualified QSO.

    Only the QSO owner can refute.
    The referee must review and decide whether to requalify or maintain the disqualification.
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")

    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Verify QSO exists and belongs to user
        cursor = conn.execute("SELECT competitor_callsign FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            raise HTTPException(status_code=404, detail="QSO not found")

        if qso["competitor_callsign"].upper() != user.callsign.upper():
            raise HTTPException(status_code=403, detail="You can only refute your own QSOs")

        # Check disqualification exists and is in disqualified status
        cursor = conn.execute(
            "SELECT id, status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
            (qso_id, sport_id)
        )
        dq = cursor.fetchone()

        if not dq:
            raise HTTPException(status_code=404, detail="No disqualification found for this QSO in this sport")

        if dq["status"] != "disqualified":
            raise HTTPException(status_code=400, detail=f"QSO is not currently disqualified (status: {dq['status']})")

        # Update status to refuted
        conn.execute(
            "UPDATE qso_disqualifications SET status = 'refuted', updated_at = ? WHERE id = ?",
            (now, dq["id"])
        )

        # Add refutation comment
        conn.execute(
            "INSERT INTO qso_disqualification_comments (disqualification_id, author_callsign, comment_type, comment, created_at) VALUES (?, ?, 'refute', ?, ?)",
            (dq["id"], user.callsign, data.refutation, now)
        )

    # Log audit
    log_audit(
        actor_callsign=user.callsign,
        action="qso_refuted",
        target_type="qso",
        target_id=str(qso_id),
        details=f"Sport ID: {sport_id}, Refutation: {data.refutation}",
        ip_address=request.client.host if request.client else None
    )

    return {
        "message": "Refutation submitted",
        "qso_id": qso_id,
        "sport_id": sport_id,
        "status": "refuted"
    }


@app.post("/referee/sport/{sport_id}/qso/{qso_id}/requalify")
@limiter.limit("30/minute")
async def requalify_qso(
    request: Request,
    sport_id: int,
    qso_id: int,
    data: QSORequalifyRequest
):
    """
    Requalify a previously disqualified QSO.

    Requires admin access or referee assignment for the sport.
    The QSO will count toward medals again in this sport.
    """
    verify_admin_or_sport_referee(request, sport_id)
    user = get_current_user(request)
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Verify QSO exists
        cursor = conn.execute("SELECT competitor_callsign FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            raise HTTPException(status_code=404, detail="QSO not found")

        # Verify sport exists
        cursor = conn.execute("SELECT name FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()
        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Check disqualification exists
        cursor = conn.execute(
            "SELECT id, status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
            (qso_id, sport_id)
        )
        dq = cursor.fetchone()

        if not dq:
            raise HTTPException(status_code=404, detail="No disqualification found for this QSO in this sport")

        if dq["status"] == "requalified":
            raise HTTPException(status_code=400, detail="QSO is already requalified")

        # Update status to requalified
        conn.execute(
            "UPDATE qso_disqualifications SET status = 'requalified', updated_at = ? WHERE id = ?",
            (now, dq["id"])
        )

        # Add requalify comment
        conn.execute(
            "INSERT INTO qso_disqualification_comments (disqualification_id, author_callsign, comment_type, comment, created_at) VALUES (?, ?, 'requalify', ?, ?)",
            (dq["id"], user.callsign if user else "ADMIN", data.reason, now)
        )

    # Log audit
    log_audit(
        actor_callsign=user.callsign if user else None,
        action="qso_requalified",
        target_type="qso",
        target_id=str(qso_id),
        details=f"Sport: {sport['name']}, Reason: {data.reason}",
        ip_address=request.client.host if request.client else None
    )

    # Recompute medals for affected matches in this sport (run in thread pool to avoid blocking/locks)
    with get_db() as conn:
        cursor = conn.execute("SELECT id FROM matches WHERE sport_id = ?", (sport_id,))
        match_ids = [row["id"] for row in cursor.fetchall()]

    for match_id in match_ids:
        await asyncio.to_thread(recompute_match_medals, match_id)

    return {
        "message": "QSO requalified",
        "qso_id": qso_id,
        "sport_id": sport_id,
        "competitor": qso["competitor_callsign"]
    }


@app.get("/qso/{qso_id}/disqualifications")
@limiter.limit("60/minute")
async def get_qso_disqualifications(request: Request, qso_id: int):
    """
    Get disqualification history for a QSO.

    Public endpoint - transparency is important for fair competition.
    Returns all sports where this QSO has been disqualified/refuted/requalified,
    along with the full comment history.
    """
    with get_db() as conn:
        # Verify QSO exists
        cursor = conn.execute("SELECT competitor_callsign, dx_callsign, qso_datetime_utc FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            raise HTTPException(status_code=404, detail="QSO not found")

        # Get all disqualifications for this QSO
        cursor = conn.execute("""
            SELECT dq.id, dq.sport_id, s.name as sport_name, dq.status, dq.created_at, dq.updated_at
            FROM qso_disqualifications dq
            JOIN sports s ON dq.sport_id = s.id
            WHERE dq.qso_id = ?
            ORDER BY dq.created_at DESC
        """, (qso_id,))
        disqualifications = [dict(row) for row in cursor.fetchall()]

        # Get comments for each disqualification
        for dq in disqualifications:
            cursor = conn.execute("""
                SELECT author_callsign, comment_type, comment, created_at
                FROM qso_disqualification_comments
                WHERE disqualification_id = ?
                ORDER BY created_at ASC
            """, (dq["id"],))
            dq["comments"] = [dict(row) for row in cursor.fetchall()]
            del dq["id"]  # Remove internal ID from response

    return {
        "qso_id": qso_id,
        "competitor_callsign": qso["competitor_callsign"],
        "dx_callsign": qso["dx_callsign"],
        "qso_datetime_utc": qso["qso_datetime_utc"],
        "disqualifications": disqualifications
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    # Get the detail message from the exception
    detail = getattr(exc, 'detail', None) or "An error occurred"

    accept_header = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    # Treat as API request if: API path, admin path, sync path, accepts JSON, or sends JSON body
    is_api_request = (
        request.url.path.startswith("/api/") or
        request.url.path.startswith("/admin/") or
        request.url.path.startswith("/sync") or
        request.url.path.startswith("/qsos/") or
        request.url.path.startswith("/sport/") or
        request.url.path.startswith("/olympiad/sports") or
        "application/json" in accept_header or
        "application/json" in content_type
    )

    # For 401 errors on non-API requests, redirect to login page
    if exc.status_code == 401 and not is_api_request:
        return RedirectResponse(url="/login", status_code=303)

    if is_api_request:
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    # For browser requests to pages that don't exist, show nice error page
    if exc.status_code == 404:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_code": 404,
            "error_title": "Page Not Found",
            "error_message": "The page you're looking for doesn't exist."
        }, status_code=404)

    # Other errors
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": exc.status_code,
        "error_title": "Error",
        "error_message": detail
    }, status_code=exc.status_code)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    if request.url.path.startswith("/admin/") or request.headers.get("accept") == "application/json":
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": 500,
        "error_title": "Server Error",
        "error_message": "Something went wrong. Please try again later."
    }, status_code=500)


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

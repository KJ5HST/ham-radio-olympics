"""
Ham Radio Olympics - Main FastAPI Application
"""

import asyncio
import logging
import os
import re
import secrets
from datetime import datetime
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

# Rate limiter - disabled in test mode
def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key, return empty string in test mode to disable."""
    if config.TESTING:
        return ""  # Disable rate limiting in tests
    return get_remote_address(request)

limiter = Limiter(key_func=get_rate_limit_key)

# Background task handle
_sync_task = None


async def background_sync():
    """Background task that syncs all competitors periodically."""
    while True:
        await asyncio.sleep(config.SYNC_INTERVAL_SECONDS)
        try:
            await sync_all_competitors()
        except Exception as e:
            # Log error but keep running
            logger.exception(f"Background sync failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    from database import backfill_records
    global _sync_task
    init_db()
    seed_example_olympiad()  # Seeds example data on fresh deployments
    backfill_records()  # Backfill records for existing QSOs if needed
    # Start background sync task
    _sync_task = asyncio.create_task(background_sync())
    yield
    # Cancel background task on shutdown
    if _sync_task:
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
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

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


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


def format_date(value: str, include_time: bool = False) -> str:
    """Format ISO date string to readable format."""
    if not value:
        return "-"
    try:
        # Parse ISO format
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if include_time:
            return dt.strftime("%b %-d, %Y %-I:%M %p") + " UTC"
        return dt.strftime("%b %-d, %Y")
    except (ValueError, AttributeError):
        return value


def format_datetime(value: str) -> str:
    """Format ISO datetime string with time."""
    return format_date(value, include_time=True)


templates.env.filters["format_date"] = format_date
templates.env.filters["format_datetime"] = format_datetime


def get_country_flag(dxcc_code) -> str:
    """Get country flag emoji from DXCC code."""
    from dxcc import get_country_name
    from callsign_lookup import get_country_flag as _get_flag

    if not dxcc_code:
        return ""
    try:
        country = get_country_name(int(dxcc_code))
        if country:
            return _get_flag(country)
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
    # Pattern matches POTA park references
    pattern = r'\b([A-Z]{1,3}-\d{4,})\b'

    def replace_pota(match):
        ref = match.group(1)
        # ref is already escaped since it comes from escaped_value
        return f'<span class="pota-ref" data-pota="{ref}">{ref}</span>'

    result = re.sub(pattern, replace_pota, escaped_value)
    return Markup(result)


templates.env.filters["pota_tooltip"] = pota_tooltip

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
        valid_types = {'continent', 'country', 'park', 'call', 'grid', 'any'}
        if v not in valid_types:
            raise ValueError(f'target_type must be one of: {", ".join(valid_types)}')
        return v


class MatchCreate(BaseModel):
    start_date: str
    end_date: str
    target_value: str
    allowed_modes: Optional[str] = None
    max_power_w: Optional[int] = None

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)')
        return v

    @field_validator('max_power_w')
    @classmethod
    def validate_max_power(cls, v):
        if v is not None and v <= 0:
            raise ValueError('max_power_w must be positive')
        return v


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
    if not user:
        return RedirectResponse(url="/signup", status_code=303)
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
async def get_sport(request: Request, sport_id: int, page: int = 1, user: User = Depends(require_user)):
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
            ORDER BY total_points DESC
        """, (sport_id,))
        standings = [dict(row) for row in cursor.fetchall()]

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

        return templates.TemplateResponse("sport.html", {
            "request": request,
            "user": user,
            "sport": dict(sport),
            "matches": matches,
            "standings": standings,
            "is_entered": is_entered,
            "entry_count": entry_count,
            "page": page,
            "total_pages": total_pages,
            "total_matches": total_matches,
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
async def get_sport_matches(sport_id: int, user: User = Depends(require_user)):
    """List all Matches in a Sport."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
            (sport_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/olympiad/sport/{sport_id}/match/{match_id}", response_class=HTMLResponse)
async def get_match(request: Request, sport_id: int, match_id: int, user: User = Depends(require_user)):
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
        cursor = conn.execute("SELECT name, target_type FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

        # Format target display
        match_dict = dict(match)
        target_display = format_target_display(match_dict["target_value"], sport["target_type"]) if sport else match_dict["target_value"]

        return templates.TemplateResponse("match.html", {
            "request": request,
            "user": user,
            "match": match_dict,
            "target_display": target_display,
            "target_type": sport["target_type"] if sport else None,
            "target_value": match_dict["target_value"],
            "sport_name": sport["name"] if sport else "Unknown",
            "sport_id": sport_id,
            "medals": medals,
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
async def get_records(request: Request, user: User = Depends(require_user)):
    """Get world records page."""
    with get_db() as conn:
        # Global records (sport_id IS NULL, callsign IS NULL) with holder names and match info
        cursor = conn.execute("""
            SELECT r.record_type, r.value, r.qso_id, r.achieved_at, r.match_id,
                   q.competitor_callsign as holder, q.dx_callsign, q.mode, q.band,
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

        # Per-mode records from medal-qualifying QSOs only
        # Join medals with QSOs to find actual QSOs that earned medals
        # Then rank by distance/cool_factor per mode
        # Note: We also fetch allowed_modes to filter in Python (SQLite lacks good string functions)
        cursor = conn.execute("""
            WITH medal_competitor_matches AS (
                -- Get all competitor/match pairs that have medals
                SELECT DISTINCT med.callsign, med.match_id
                FROM medals med
                WHERE med.qualified = 1
            ),
            matching_qsos AS (
                -- Find QSOs from medal-holding competitors during their matches
                -- that match the target (for parks, check sig_info)
                SELECT q.id, q.mode, q.distance_km, q.cool_factor, q.competitor_callsign,
                       q.qso_datetime_utc, c.first_name, m.id as match_id, m.target_value,
                       s.id as sport_id, s.name as sport_name, s.target_type,
                       COALESCE(m.allowed_modes, s.allowed_modes) as allowed_modes
                FROM medal_competitor_matches mcm
                INNER JOIN matches m ON mcm.match_id = m.id
                INNER JOIN sports s ON m.sport_id = s.id
                INNER JOIN qsos q ON q.competitor_callsign = mcm.callsign
                                 AND q.qso_datetime_utc >= m.start_date
                                 AND q.qso_datetime_utc <= m.end_date
                                 AND q.is_confirmed = 1
                LEFT JOIN competitors c ON q.competitor_callsign = c.callsign
                WHERE q.mode IS NOT NULL AND q.mode != ''
                  -- For parks, verify the QSO matches the target
                  AND (s.target_type != 'park'
                       OR q.dx_sig_info = m.target_value
                       OR q.my_sig_info = m.target_value)
            ),
            ranked_qsos AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY mode ORDER BY distance_km DESC) as dist_rank,
                       ROW_NUMBER() OVER (PARTITION BY mode ORDER BY cool_factor DESC) as cf_rank
                FROM matching_qsos
            )
            SELECT * FROM ranked_qsos
            WHERE dist_rank <= 3 OR cf_rank <= 3
            ORDER BY mode, dist_rank
        """)
        candidate_qsos = cursor.fetchall()

        # Import mode filtering function
        from scoring import is_mode_allowed

        # Build mode records (filtering by allowed_modes for each sport/match)
        mode_best = {}
        for qso in candidate_qsos:
            qso_dict = dict(qso)
            mode = qso_dict['mode']

            # Skip QSOs whose mode isn't allowed for the sport/match
            allowed_modes = qso_dict.get('allowed_modes')
            if not is_mode_allowed(mode, allowed_modes):
                continue

            if mode not in mode_best:
                mode_best[mode] = {'max_distance': 0, 'max_cool_factor': 0}

            if qso_dict['distance_km'] and qso_dict['distance_km'] > mode_best[mode]['max_distance']:
                mode_best[mode]['max_distance'] = qso_dict['distance_km']
                mode_best[mode]['distance_qso'] = qso_dict

            if qso_dict['cool_factor'] and qso_dict['cool_factor'] > mode_best[mode]['max_cool_factor']:
                mode_best[mode]['max_cool_factor'] = qso_dict['cool_factor']
                mode_best[mode]['cool_factor_qso'] = qso_dict

        # Build separate lists for distance and cool factor records
        distance_records = []
        cool_factor_records = []

        for mode, data in mode_best.items():
            if 'distance_qso' in data and data['max_distance']:
                d = data['distance_qso']
                distance_records.append({
                    'mode': mode,
                    'value': data['max_distance'],
                    'holder': d['competitor_callsign'],
                    'holder_name': d['first_name'],
                    'match_id': d['match_id'],
                    'sport_id': d['sport_id'],
                    'sport_name': d['sport_name'],
                    'target': d['target_value'],
                    'date': d['qso_datetime_utc'],
                })

            if 'cool_factor_qso' in data and data['max_cool_factor']:
                cf = data['cool_factor_qso']
                cool_factor_records.append({
                    'mode': mode,
                    'value': data['max_cool_factor'],
                    'holder': cf['competitor_callsign'],
                    'holder_name': cf['first_name'],
                    'match_id': cf['match_id'],
                    'sport_id': cf['sport_id'],
                    'sport_name': cf['sport_name'],
                    'target': cf['target_value'],
                    'date': cf['qso_datetime_utc'],
                })

        # Sort each by date (newest first)
        distance_records.sort(key=lambda x: x.get('date') or '', reverse=True)
        cool_factor_records.sort(key=lambda x: x.get('date') or '', reverse=True)

        return templates.TemplateResponse("records.html", {
            "request": request,
            "user": user,
            "world_records": world_records,
            "distance_records": distance_records,
            "cool_factor_records": cool_factor_records,
        })


@app.get("/medals", response_class=HTMLResponse)
async def get_medals_page(request: Request, user: User = Depends(require_user)):
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

    return templates.TemplateResponse("medals.html", {
        "request": request,
        "user": user,
        "medal_standings": medal_standings
    })


@app.get("/competitor/{callsign}", response_class=HTMLResponse)
async def get_competitor(
    request: Request,
    callsign: str,
    user: User = Depends(require_user),
    band: Optional[str] = None,
    mode: Optional[str] = None,
    confirmed: Optional[str] = None,
    page: int = 1
):
    """Get competitor's QSOs, medals, and personal bests."""
    callsign = callsign.upper()
    is_own_profile = (callsign == user.callsign)
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

        # Add country names to QSOs
        for qso in qsos:
            if qso.get("dx_dxcc"):
                qso["dx_country"] = get_country_name(qso["dx_dxcc"])

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
        can_sync = has_qrz_key and (is_own_profile or user.is_admin or user.is_referee)

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
        result = await sync_competitor(callsign)
    else:
        result = await sync_all_competitors()

    return templates.TemplateResponse("sync.html", {
        "request": request,
        "result": result,
        "callsign": callsign,
    })


@app.post("/sync")
async def trigger_sync(callsign: Optional[str] = None):
    """Trigger QRZ sync (API). Syncs single competitor or all if no callsign provided."""
    if callsign:
        result = await sync_competitor(callsign)
    else:
        result = await sync_all_competitors()

    return result


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
        details="Successful login",
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
            details="User logged out",
            ip_address=get_client_ip(request)
        )

    response = RedirectResponse(url="/signup", status_code=303)
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

        # Get competitor info
        cursor = conn.execute(
            "SELECT first_name, registered_at, last_sync_at FROM competitors WHERE callsign = ?",
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
                      email_match_reminders, email_record_notifications
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
        details="Password changed via settings",
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


# ============================================================
# EXPORT ENDPOINTS
# ============================================================

@app.get("/export/qsos")
async def export_qsos(request: Request, user: User = Depends(require_user)):
    """Export user's QSOs as CSV."""
    import csv
    import io
    from starlette.responses import StreamingResponse

    with get_db() as conn:
        cursor = conn.execute("""
            SELECT dx_callsign, qso_datetime_utc, band, mode,
                   dx_grid, dx_dxcc, is_confirmed, cool_factor
            FROM qsos
            WHERE competitor_callsign = ?
            ORDER BY qso_datetime_utc DESC
        """, (user.callsign,))
        qsos = cursor.fetchall()

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["callsign", "datetime_utc", "band", "mode",
                     "grid", "dxcc", "confirmed", "cool_factor"])

    for qso in qsos:
        writer.writerow([
            qso["dx_callsign"],
            qso["qso_datetime_utc"],
            qso["band"],
            qso["mode"],
            qso["dx_grid"] or "",
            qso["dx_dxcc"] or "",
            "Yes" if qso["is_confirmed"] else "No",
            qso["cool_factor"] or ""
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

    return templates.TemplateResponse("admin/audit_log.html", {
        "request": request,
        "user": get_current_user(request),
        "logs": logs,
        "total": total,
        "page": page,
        "per_page": per_page,
        "action_filter": action,
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

    return templates.TemplateResponse("admin/matches.html", {
        "request": request,
        "user": get_current_user(request),
        "sport": sport_dict,
        "matches": matches,
        "target_options": target_options,
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
            INSERT INTO matches (sport_id, start_date, end_date, target_value, allowed_modes, max_power_w)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sport_id, match.start_date, match.end_date, match.target_value, match.allowed_modes, match.max_power_w))

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
            SET start_date = ?, end_date = ?, target_value = ?, allowed_modes = ?, max_power_w = ?
            WHERE id = ?
        """, (match.start_date, match.end_date, match.target_value, match.allowed_modes, match.max_power_w, match_id))

    # Recompute medals for this match (outside connection to avoid lock)
    recompute_match_medals(match_id)

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
async def enable_competitor(callsign: str, _: bool = Depends(verify_admin)):
    """Enable a competitor's account."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_disabled = 0 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} enabled"}


@app.post("/admin/competitor/{callsign}/set-admin")
async def set_admin_role(callsign: str, _: bool = Depends(verify_admin)):
    """Grant admin role to a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is now an admin"}


@app.post("/admin/competitor/{callsign}/remove-admin")
async def remove_admin_role(callsign: str, _: bool = Depends(verify_admin)):
    """Remove admin role from a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_admin = 0 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is no longer an admin"}


@app.post("/admin/competitor/{callsign}/set-referee")
async def set_referee_role(callsign: str, _: bool = Depends(verify_admin)):
    """Grant referee role to a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_referee = 1 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is now a referee"}


@app.post("/admin/competitor/{callsign}/remove-referee")
async def remove_referee_role(callsign: str, _: bool = Depends(verify_admin)):
    """Remove referee role from a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM competitors WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE competitors SET is_referee = 0 WHERE callsign = ?", (callsign,))
        # Also remove all referee assignments
        conn.execute("DELETE FROM referee_assignments WHERE callsign = ?", (callsign,))

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

    # Recompute medals for affected matches
    from sync import recompute_all_active_matches
    recompute_all_active_matches()

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

    return templates.TemplateResponse("admin/settings.html", {
        "request": request,
        "user": get_current_user(request),
        "qrz_username": qrz_username,
        "qrz_configured": qrz_configured,
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


@app.post("/admin/recompute-records")
async def admin_recompute_records(_: bool = Depends(verify_admin)):
    """Recompute all medals and world records from match-qualifying QSOs."""
    from scoring import recompute_all_records
    from sync import recompute_all_active_matches

    # First recompute medals for all matches
    recompute_all_active_matches()

    # Then recompute world records
    recompute_all_records()

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
async def teams_list_page(request: Request, page: int = 1, per_page: int = 20, user: User = Depends(require_user)):
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
                   COUNT(DISTINCT tm.callsign) as member_count,
                   COALESCE(SUM(m.total_points), 0) as total_points,
                   COUNT(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 END) as gold,
                   COUNT(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 END) as silver,
                   COUNT(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 END) as bronze
            FROM teams t
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

        return templates.TemplateResponse("teams.html", {
            "request": request,
            "teams": teams,
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
async def team_profile_page(request: Request, team_id: int, user: User = Depends(require_user)):
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


@app.get("/api/v1/park/{reference}")
@limiter.limit("100/minute")
async def api_park_lookup(request: Request, reference: str):
    """Look up POTA park info with caching."""
    import httpx
    from datetime import timedelta

    reference = reference.upper().strip()

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
async def api_all_spots(request: Request):
    """Get all current POTA spots (returns list of park references with spots)."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pota.app/spot/activator",
                timeout=10.0
            )
            if response.status_code == 200:
                all_spots = response.json()
                if not all_spots:
                    return {"parks_with_spots": []}

                # Get unique park references that have spots
                parks_with_spots = list(set(
                    spot.get("reference")
                    for spot in all_spots
                    if spot.get("reference")
                ))

                return {"parks_with_spots": parks_with_spots}
            else:
                return {"parks_with_spots": []}
    except httpx.RequestError:
        return {"parks_with_spots": []}


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

"""
Ham Radio Olympics - Main FastAPI Application
"""

import asyncio
import os
import re
import secrets
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException, Depends, UploadFile, File, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from database import init_db, get_db
from crypto import encrypt_api_key
from sync import sync_contestant, sync_all_contestants, recompute_sport_matches
from qrz_client import verify_api_key
from scoring import recompute_match_medals
from dxcc import get_country_name, get_continent_name, get_all_continents, get_all_countries
from auth import (
    register_user, authenticate_user, get_session_user, delete_session,
    update_user_email, update_user_password, hash_password, User, SESSION_COOKIE_NAME
)

# Sync interval in seconds (1 hour)
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", 3600))

# Background task handle
_sync_task = None


async def background_sync():
    """Background task that syncs all contestants periodically."""
    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        try:
            await sync_all_contestants()
        except Exception:
            # Log error but keep running
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global _sync_task
    init_db()
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
    description="Olympic-style amateur radio competition",
    version="1.0.0",
    lifespan=lifespan,
)

# Templates
templates = Jinja2Templates(directory="templates")


def format_date(value: str, include_time: bool = False) -> str:
    """Format ISO date string to readable format."""
    if not value:
        return "-"
    try:
        # Parse ISO format
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if include_time:
            return dt.strftime("%b %-d, %Y %-I:%M %p")
        return dt.strftime("%b %-d, %Y")
    except (ValueError, AttributeError):
        return value


def format_datetime(value: str) -> str:
    """Format ISO datetime string with time."""
    return format_date(value, include_time=True)


templates.env.filters["format_date"] = format_date
templates.env.filters["format_datetime"] = format_datetime

# Admin key from environment
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin-secret-change-me")


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
    qrz_api_key: str
    email: Optional[str] = None

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

    @field_validator("qrz_api_key")
    @classmethod
    def validate_qrz_api_key(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("QRZ API key is required")
        return v


class UserLogin(BaseModel):
    callsign: str
    password: str


class OlympiadCreate(BaseModel):
    name: str
    start_date: str
    end_date: str
    qualifying_qsos: int = 0


class SportCreate(BaseModel):
    name: str
    description: Optional[str] = None
    target_type: str
    work_enabled: bool = True
    activate_enabled: bool = False
    separate_pools: bool = False


class MatchCreate(BaseModel):
    start_date: str
    end_date: str
    target_value: str


# Admin authentication dependency
def verify_admin(request: Request):
    """Verify admin access via key or logged-in admin user."""
    # Check admin key first
    admin_key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
    if admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user and user.is_admin:
        return True
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
    # Check admin key first
    admin_key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
    if admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin or referee for this sport
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user:
        if user.is_admin:
            return True
        if user.is_referee and is_referee_for_sport(user.callsign, sport_id):
            return True
    raise HTTPException(status_code=403, detail="Admin or referee access required")


def verify_admin_or_referee(request: Request):
    """Verify admin access or referee role."""
    # Check admin key first
    admin_key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
    if admin_key == ADMIN_KEY:
        return True
    # Check if logged-in user is admin or referee
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_session_user(session_id)
    if user and (user.is_admin or user.is_referee):
        return True
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
async def get_olympiad():
    """Get current active Olympiad details."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads WHERE is_active = 1")
        olympiad = cursor.fetchone()

        if not olympiad:
            raise HTTPException(status_code=404, detail="No active Olympiad")

        return dict(olympiad)


@app.get("/olympiad/sports")
async def get_olympiad_sports():
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
async def get_sport(request: Request, sport_id: int):
    """Get Sport details and standings."""
    user = get_current_user(request)
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM sports WHERE id = ?", (sport_id,))
        sport = cursor.fetchone()

        if not sport:
            raise HTTPException(status_code=404, detail="Sport not found")

        # Get all matches
        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
            (sport_id,)
        )
        matches = [dict(row) for row in cursor.fetchall()]

        # Add target_display to matches
        sport_dict = dict(sport)
        for match in matches:
            match["target_display"] = format_target_display(match["target_value"], sport_dict["target_type"])

        # Get cumulative standings
        cursor = conn.execute("""
            SELECT callsign, role,
                   SUM(total_points) as total_points,
                   SUM(CASE WHEN distance_medal = 'gold' THEN 1 ELSE 0 END +
                       CASE WHEN cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold,
                   SUM(CASE WHEN distance_medal = 'silver' THEN 1 ELSE 0 END +
                       CASE WHEN cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver,
                   SUM(CASE WHEN distance_medal = 'bronze' THEN 1 ELSE 0 END +
                       CASE WHEN cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            WHERE ma.sport_id = ?
            GROUP BY callsign, role
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
        })


@app.get("/olympiad/sport/{sport_id}/matches")
async def get_sport_matches(sport_id: int):
    """List all Matches in a Sport."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE sport_id = ? ORDER BY start_date",
            (sport_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/olympiad/sport/{sport_id}/match/{match_id}", response_class=HTMLResponse)
async def get_match(request: Request, sport_id: int, match_id: int):
    """Get Match details and leaderboard."""
    user = get_current_user(request)
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE id = ? AND sport_id = ?",
            (match_id, sport_id)
        )
        match = cursor.fetchone()

        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        # Get medal results
        cursor = conn.execute("""
            SELECT * FROM medals
            WHERE match_id = ?
            ORDER BY total_points DESC, distance_claim_time ASC
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
            "sport_name": sport["name"] if sport else "Unknown",
            "sport_id": sport_id,
            "medals": medals,
        })


@app.post("/sport/{sport_id}/enter")
async def enter_sport(sport_id: int, user: User = Depends(require_user)):
    """Opt into a sport."""
    with get_db() as conn:
        # Verify sport exists
        cursor = conn.execute("SELECT id FROM sports WHERE id = ?", (sport_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Sport not found")

        # Check if already entered
        cursor = conn.execute(
            "SELECT id FROM sport_entries WHERE callsign = ? AND sport_id = ?",
            (user.callsign, sport_id)
        )
        if cursor.fetchone():
            return {"message": "Already entered"}

        # Enter the sport
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
            (user.callsign, sport_id, now)
        )

    # Recompute medals for this sport
    recompute_sport_matches(sport_id)

    return {"message": "Entered sport successfully"}


@app.post("/sport/{sport_id}/leave")
async def leave_sport(sport_id: int, user: User = Depends(require_user)):
    """Opt out of a sport."""
    with get_db() as conn:
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
        # Global records (sport_id IS NULL, callsign IS NULL)
        cursor = conn.execute("""
            SELECT r.*, q.contestant_callsign as holder
            FROM records r
            LEFT JOIN qsos q ON r.qso_id = q.id
            WHERE r.callsign IS NULL AND r.sport_id IS NULL
            ORDER BY r.record_type
        """)
        world_records = [dict(row) for row in cursor.fetchall()]

        return templates.TemplateResponse("records.html", {
            "request": request,
            "user": user,
            "world_records": world_records
        })


@app.get("/contestant/{callsign}", response_class=HTMLResponse)
async def get_contestant(request: Request, callsign: str, user: User = Depends(require_user)):
    """Get contestant's QSOs, medals, and personal bests."""
    callsign = callsign.upper()
    is_own_profile = (callsign == user.callsign)

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT callsign, registered_at, last_sync_at, qrz_api_key_encrypted FROM contestants WHERE callsign = ?",
            (callsign,)
        )
        contestant = cursor.fetchone()

        if not contestant:
            raise HTTPException(status_code=404, detail="Contestant not found")

        contestant_dict = dict(contestant)
        has_qrz_key = bool(contestant_dict.pop("qrz_api_key_encrypted", None))

        # Get QSOs
        cursor = conn.execute("""
            SELECT * FROM qsos
            WHERE contestant_callsign = ?
            ORDER BY qso_datetime_utc DESC
            LIMIT 50
        """, (callsign,))
        qsos = [dict(row) for row in cursor.fetchall()]

        # Get medals
        cursor = conn.execute("""
            SELECT m.*, ma.target_value, s.name as sport_name, s.target_type
            FROM medals m
            JOIN matches ma ON m.match_id = ma.id
            JOIN sports s ON ma.sport_id = s.id
            WHERE m.callsign = ?
            ORDER BY ma.start_date DESC
        """, (callsign,))
        medals = [dict(row) for row in cursor.fetchall()]

        # Add target_display
        for medal in medals:
            medal["target_display"] = format_target_display(medal["target_value"], medal["target_type"])

        # Get personal bests
        cursor = conn.execute("""
            SELECT * FROM records
            WHERE callsign = ?
        """, (callsign,))
        personal_bests = [dict(row) for row in cursor.fetchall()]

        # Get sports the competitor is entered in with points per sport
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name,
                   COALESCE(SUM(m.total_points), 0) as sport_points,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN m.id END) as gold_count,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN m.id END) as silver_count,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN m.id END) as bronze_count
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
        gold = sum((1 if m["distance_medal"] == "gold" else 0) + (1 if m["cool_factor_medal"] == "gold" else 0) for m in medals)
        silver = sum((1 if m["distance_medal"] == "silver" else 0) + (1 if m["cool_factor_medal"] == "silver" else 0) for m in medals)
        bronze = sum((1 if m["distance_medal"] == "bronze" else 0) + (1 if m["cool_factor_medal"] == "bronze" else 0) for m in medals)
        total_points = sum(m["total_points"] for m in medals)

        # Can sync if: viewing own profile with QRZ key, OR admin/referee viewing someone with QRZ key
        can_sync = has_qrz_key and (is_own_profile or user.is_admin or user.is_referee)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "contestant": contestant_dict,
            "medal_summary": {
                "gold": gold,
                "silver": silver,
                "bronze": bronze,
                "total_points": total_points,
            },
            "qsos": qsos,
            "medals": medals,
            "personal_bests": personal_bests,
            "sport_entries": sport_entries,
            "is_own_profile": is_own_profile,
            "can_sync": can_sync,
        })


@app.get("/sync", response_class=HTMLResponse)
async def sync_page(request: Request, callsign: Optional[str] = None):
    """Sync page - shows sync results."""
    if callsign:
        result = await sync_contestant(callsign)
    else:
        result = await sync_all_contestants()

    return templates.TemplateResponse("sync.html", {
        "request": request,
        "result": result,
        "callsign": callsign,
    })


@app.post("/sync")
async def trigger_sync(callsign: Optional[str] = None):
    """Trigger QRZ sync (API). Syncs single contestant or all if no callsign provided."""
    if callsign:
        result = await sync_contestant(callsign)
    else:
        result = await sync_all_contestants()

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
async def signup(signup_data: UserSignup):
    """Create a new user account."""
    callsign = signup_data.callsign.upper()

    # Verify QRZ API key is valid AND belongs to the callsign being registered
    is_valid = await verify_api_key(signup_data.qrz_api_key, callsign)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid QRZ API key or key does not match callsign.")

    # Encrypt the API key
    encrypted_key = encrypt_api_key(signup_data.qrz_api_key)

    success = register_user(callsign, signup_data.password, signup_data.email, encrypted_key)
    if not success:
        raise HTTPException(status_code=400, detail="Callsign already registered")

    # Sync QRZ data if API key was provided
    if signup_data.qrz_api_key:
        try:
            await sync_contestant(callsign)
        except Exception:
            pass  # Don't fail signup if sync fails

    # Auto-login after signup
    session_id = authenticate_user(callsign, signup_data.password)

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax"
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
async def login(login_data: UserLogin):
    """Authenticate user and create session."""
    result = authenticate_user(login_data.callsign, login_data.password)

    if result == "disabled":
        raise HTTPException(status_code=403, detail="Account has been disabled")

    if not result:
        raise HTTPException(status_code=401, detail="Invalid callsign or password")

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=result,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax"
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    """Logout and delete session."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        delete_session(session_id)

    response = RedirectResponse(url="/signup", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


# ============================================================
# USER DASHBOARD & SETTINGS
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request, user: User = Depends(require_user)):
    """User dashboard with stats, QSOs, and medals."""
    with get_db() as conn:
        # Get QSOs
        cursor = conn.execute("""
            SELECT * FROM qsos
            WHERE contestant_callsign = ?
            ORDER BY qso_datetime_utc DESC
            LIMIT 50
        """, (user.callsign,))
        qsos = [dict(row) for row in cursor.fetchall()]

        # Get medals
        cursor = conn.execute("""
            SELECT m.*, ma.target_value, s.name as sport_name, s.target_type
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

        # Calculate medal summary (count each medal type separately)
        gold = sum((1 if m["distance_medal"] == "gold" else 0) + (1 if m["cool_factor_medal"] == "gold" else 0) for m in medals)
        silver = sum((1 if m["distance_medal"] == "silver" else 0) + (1 if m["cool_factor_medal"] == "silver" else 0) for m in medals)
        bronze = sum((1 if m["distance_medal"] == "bronze" else 0) + (1 if m["cool_factor_medal"] == "bronze" else 0) for m in medals)
        total_points = sum(m["total_points"] for m in medals)

        # Get contestant info
        cursor = conn.execute(
            "SELECT registered_at, last_sync_at FROM contestants WHERE callsign = ?",
            (user.callsign,)
        )
        contestant_info = dict(cursor.fetchone())

        # Get sports the competitor is entered in with points per sport
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name,
                   COALESCE(SUM(m.total_points), 0) as sport_points,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN m.id END) as gold_count,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN m.id END) as silver_count,
                   COUNT(DISTINCT CASE WHEN m.distance_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN m.id END) as bronze_count
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

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "contestant": {"callsign": user.callsign, **contestant_info},
        "qsos": qsos,
        "medals": medals,
        "personal_bests": personal_bests,
        "sport_entries": sport_entries,
        "medal_summary": {
            "gold": gold,
            "silver": silver,
            "bronze": bronze,
            "total_points": total_points,
        },
        "is_own_profile": True,
        "can_sync": user.has_qrz_key,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(require_user)):
    """Account settings page."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT email, registered_at FROM contestants WHERE callsign = ?",
            (user.callsign,)
        )
        account = dict(cursor.fetchone())

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "account": account,
    })


@app.post("/settings/email")
async def update_email(request: Request, email: str = Form(...), user: User = Depends(require_user)):
    """Update user email."""
    update_user_email(user.callsign, email)
    return RedirectResponse(url="/settings?updated=email", status_code=303)


@app.post("/settings/password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    user: User = Depends(require_user)
):
    """Change user password."""
    # Verify current password
    session_id = authenticate_user(user.callsign, current_password)
    if not session_id:
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    update_user_password(user.callsign, new_password)
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
            "UPDATE contestants SET qrz_api_key_encrypted = ? WHERE callsign = ?",
            (encrypted_key, user.callsign)
        )
    return RedirectResponse(url="/settings?updated=qrz", status_code=303)


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, _: bool = Depends(verify_admin)):
    """Admin dashboard."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM olympiads ORDER BY start_date DESC")
        olympiads = [dict(row) for row in cursor.fetchall()]

        cursor = conn.execute("SELECT COUNT(*) as count FROM contestants")
        contestant_count = cursor.fetchone()["count"]

    return templates.TemplateResponse("admin/dashboard.html", {
        "request": request,
        "user": get_current_user(request),
        "olympiads": olympiads,
        "contestant_count": contestant_count,
    })


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
                              work_enabled, activate_enabled, separate_pools)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            olympiad_id, sport.name, sport.description, sport.target_type,
            1 if sport.work_enabled else 0,
            1 if sport.activate_enabled else 0,
            1 if sport.separate_pools else 0,
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
                work_enabled = ?, activate_enabled = ?, separate_pools = ?
            WHERE id = ?
        """, (
            sport.name, sport.description, sport.target_type,
            1 if sport.work_enabled else 0,
            1 if sport.activate_enabled else 0,
            1 if sport.separate_pools else 0,
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
            SELECT c.callsign, c.registered_at, c.is_disabled,
                   COUNT(DISTINCT m.id) as medal_count,
                   SUM(CASE WHEN m.distance_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 ELSE 0 END) as gold_count,
                   SUM(CASE WHEN m.distance_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 ELSE 0 END) as silver_count,
                   SUM(CASE WHEN m.distance_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 ELSE 0 END) as bronze_count
            FROM sport_entries se
            JOIN contestants c ON se.callsign = c.callsign
            LEFT JOIN medals m ON m.callsign = c.callsign AND m.match_id IN (
                SELECT id FROM matches WHERE sport_id = ?
            )
            WHERE se.sport_id = ?
            GROUP BY c.callsign
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
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO matches (sport_id, start_date, end_date, target_value)
            VALUES (?, ?, ?, ?)
        """, (sport_id, match.start_date, match.end_date, match.target_value))

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
    with get_db() as conn:
        # Get sport_id for permission check
        cursor = conn.execute("SELECT sport_id FROM matches WHERE id = ?", (match_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Match not found")

        verify_admin_or_sport_referee(request, existing["sport_id"])

        conn.execute("""
            UPDATE matches
            SET start_date = ?, end_date = ?, target_value = ?
            WHERE id = ?
        """, (match.start_date, match.end_date, match.target_value, match_id))

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


@app.get("/admin/contestants", response_class=HTMLResponse)
async def admin_contestants(request: Request, _: bool = Depends(verify_admin)):
    """List all competitors."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT callsign, registered_at, last_sync_at, is_disabled, is_admin, is_referee
            FROM contestants
            ORDER BY registered_at DESC
        """)
        contestants = [dict(row) for row in cursor.fetchall()]

        # Get referee assignments for each referee
        for c in contestants:
            if c["is_referee"]:
                cursor = conn.execute("""
                    SELECT s.id, s.name FROM referee_assignments ra
                    JOIN sports s ON ra.sport_id = s.id
                    WHERE ra.callsign = ?
                """, (c["callsign"],))
                c["assigned_sports"] = [dict(row) for row in cursor.fetchall()]
            else:
                c["assigned_sports"] = []

        # Get all sports for assignment dropdown
        cursor = conn.execute("""
            SELECT s.id, s.name, o.name as olympiad_name
            FROM sports s
            JOIN olympiads o ON s.olympiad_id = o.id
            ORDER BY o.id DESC, s.name
        """)
        all_sports = [dict(row) for row in cursor.fetchall()]

    return templates.TemplateResponse("admin/contestants.html", {
        "request": request,
        "user": get_current_user(request),
        "contestants": contestants,
        "all_sports": all_sports,
    })


@app.post("/admin/contestant/{callsign}/disable")
async def disable_contestant(callsign: str, _: bool = Depends(verify_admin)):
    """Disable a competitor's account."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_disabled = 1 WHERE callsign = ?", (callsign,))
        # Also delete their sessions so they're logged out
        conn.execute("DELETE FROM sessions WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} disabled"}


@app.post("/admin/contestant/{callsign}/enable")
async def enable_contestant(callsign: str, _: bool = Depends(verify_admin)):
    """Enable a competitor's account."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_disabled = 0 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} enabled"}


@app.post("/admin/contestant/{callsign}/set-admin")
async def set_admin_role(callsign: str, _: bool = Depends(verify_admin)):
    """Grant admin role to a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_admin = 1 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is now an admin"}


@app.post("/admin/contestant/{callsign}/remove-admin")
async def remove_admin_role(callsign: str, _: bool = Depends(verify_admin)):
    """Remove admin role from a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_admin = 0 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is no longer an admin"}


@app.post("/admin/contestant/{callsign}/set-referee")
async def set_referee_role(callsign: str, _: bool = Depends(verify_admin)):
    """Grant referee role to a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_referee = 1 WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is now a referee"}


@app.post("/admin/contestant/{callsign}/remove-referee")
async def remove_referee_role(callsign: str, _: bool = Depends(verify_admin)):
    """Remove referee role from a competitor."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")
        conn.execute("UPDATE contestants SET is_referee = 0 WHERE callsign = ?", (callsign,))
        # Also remove all referee assignments
        conn.execute("DELETE FROM referee_assignments WHERE callsign = ?", (callsign,))

    return {"message": f"Competitor {callsign} is no longer a referee"}


@app.post("/admin/contestant/{callsign}/assign-sport/{sport_id}")
async def assign_referee_to_sport(callsign: str, sport_id: int, _: bool = Depends(verify_admin)):
    """Assign a referee to a sport."""
    callsign = callsign.upper()
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        # Check competitor exists and is a referee
        cursor = conn.execute("SELECT is_referee FROM contestants WHERE callsign = ?", (callsign,))
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


@app.delete("/admin/contestant/{callsign}/assign-sport/{sport_id}")
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


@app.post("/admin/contestant/{callsign}/disqualify")
async def disqualify_contestant(callsign: str, _: bool = Depends(verify_admin_or_referee)):
    """Disqualify a competitor from the active competition. Admins or referees can disqualify."""
    callsign = callsign.upper()
    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
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


@app.post("/admin/contestant/{callsign}/reset-password")
async def reset_contestant_password(callsign: str, _: bool = Depends(verify_admin)):
    """Reset a competitor's password to a random value."""
    callsign = callsign.upper()

    # Generate a random password
    new_password = secrets.token_urlsafe(12)

    with get_db() as conn:
        cursor = conn.execute("SELECT 1 FROM contestants WHERE callsign = ?", (callsign,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Competitor not found")

        # Hash and store new password
        password_hash = hash_password(new_password)
        conn.execute(
            "UPDATE contestants SET password_hash = ? WHERE callsign = ?",
            (password_hash, callsign)
        )
        # Invalidate all sessions
        conn.execute("DELETE FROM sessions WHERE callsign = ?", (callsign,))

    return {"message": f"Password reset for {callsign}", "new_password": new_password}


@app.delete("/admin/contestant/{callsign}")
async def delete_contestant(callsign: str, _: bool = Depends(verify_admin)):
    """Delete a competitor."""
    with get_db() as conn:
        conn.execute("DELETE FROM contestants WHERE callsign = ?", (callsign.upper(),))

    return {"message": f"Competitor {callsign.upper()} deleted"}


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

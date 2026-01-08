"""
Database setup and models for Ham Radio Olympics.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

DATABASE_PATH = os.getenv("DATABASE_PATH", "ham_olympics.db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        conn.executescript("""
            -- Contestants table
            CREATE TABLE IF NOT EXISTS contestants (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                qrz_api_key_encrypted TEXT,
                registered_at TEXT NOT NULL,
                last_sync_at TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_referee INTEGER NOT NULL DEFAULT 0,
                is_disabled INTEGER NOT NULL DEFAULT 0
            );

            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                callsign TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign) ON DELETE CASCADE
            );

            -- Olympiads table
            CREATE TABLE IF NOT EXISTS olympiads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );

            -- Sports table
            CREATE TABLE IF NOT EXISTS sports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL CHECK (target_type IN ('continent', 'country', 'park', 'call', 'grid')),
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id) ON DELETE CASCADE
            );

            -- Matches table
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_value TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE
            );

            -- QSOs table
            CREATE TABLE IF NOT EXISTS qsos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contestant_callsign TEXT NOT NULL,
                dx_callsign TEXT NOT NULL,
                qso_datetime_utc TEXT NOT NULL,
                band TEXT,
                mode TEXT,
                my_dxcc INTEGER,
                my_grid TEXT,
                my_sig_info TEXT,
                dx_dxcc INTEGER,
                dx_grid TEXT,
                dx_sig_info TEXT,
                distance_km REAL,
                tx_power_w REAL,
                cool_factor REAL,
                is_confirmed INTEGER NOT NULL DEFAULT 0,
                qrz_logid TEXT,
                FOREIGN KEY (contestant_callsign) REFERENCES contestants(callsign) ON DELETE CASCADE
            );

            -- Medals table
            CREATE TABLE IF NOT EXISTS medals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                callsign TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('work', 'activate', 'combined')),
                qualified INTEGER NOT NULL DEFAULT 1,
                distance_medal TEXT CHECK (distance_medal IN ('gold', 'silver', 'bronze', NULL)),
                distance_claim_time TEXT,
                cool_factor_medal TEXT CHECK (cool_factor_medal IN ('gold', 'silver', 'bronze', NULL)),
                cool_factor_value REAL,
                cool_factor_claim_time TEXT,
                pota_bonus INTEGER NOT NULL DEFAULT 0,
                total_points INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign) ON DELETE CASCADE,
                UNIQUE (match_id, callsign, role)
            );

            -- Sport entries table (opt-in)
            CREATE TABLE IF NOT EXISTS sport_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                entered_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign) ON DELETE CASCADE,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                UNIQUE (callsign, sport_id)
            );

            -- Referee assignments table
            CREATE TABLE IF NOT EXISTS referee_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign) ON DELETE CASCADE,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                UNIQUE (callsign, sport_id)
            );

            -- Records table
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport_id INTEGER,
                callsign TEXT,
                record_type TEXT NOT NULL CHECK (record_type IN ('longest_distance', 'highest_cool_factor', 'lowest_power')),
                value REAL NOT NULL,
                qso_id INTEGER,
                achieved_at TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign) ON DELETE CASCADE,
                FOREIGN KEY (qso_id) REFERENCES qsos(id) ON DELETE SET NULL
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_qsos_contestant ON qsos(contestant_callsign);
            CREATE INDEX IF NOT EXISTS idx_qsos_datetime ON qsos(qso_datetime_utc);
            CREATE INDEX IF NOT EXISTS idx_qsos_confirmed ON qsos(is_confirmed);
            CREATE INDEX IF NOT EXISTS idx_medals_match ON medals(match_id);
            CREATE INDEX IF NOT EXISTS idx_medals_callsign ON medals(callsign);
            CREATE INDEX IF NOT EXISTS idx_sports_olympiad ON sports(olympiad_id);
            CREATE INDEX IF NOT EXISTS idx_matches_sport ON matches(sport_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_callsign ON sessions(callsign);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_sport_entries_callsign ON sport_entries(callsign);
            CREATE INDEX IF NOT EXISTS idx_sport_entries_sport ON sport_entries(sport_id);
            CREATE INDEX IF NOT EXISTS idx_referee_assignments_callsign ON referee_assignments(callsign);
            CREATE INDEX IF NOT EXISTS idx_referee_assignments_sport ON referee_assignments(sport_id);
        """)


def reset_db():
    """Reset the database (for testing)."""
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    init_db()

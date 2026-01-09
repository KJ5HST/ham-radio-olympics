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
        # Migration: rename contestants -> competitors if needed
        # Check what tables exist
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        has_contestants = 'contestants' in tables
        has_competitors = 'competitors' in tables

        if has_contestants and has_competitors:
            # Both tables exist - drop the old empty contestants table (partial migration state)
            conn.execute("DROP TABLE contestants")
            conn.commit()
        elif has_contestants and not has_competitors:
            # Need to migrate
            conn.execute("ALTER TABLE contestants RENAME TO competitors")
            conn.commit()

        # Check if qsos table has old column name
        cursor = conn.execute("PRAGMA table_info(qsos)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'contestant_callsign' in columns:
            conn.execute("ALTER TABLE qsos RENAME COLUMN contestant_callsign TO competitor_callsign")
            conn.commit()

        # Drop old index if it exists
        conn.execute("DROP INDEX IF EXISTS idx_qsos_contestant")
        conn.commit()

        conn.executescript("""
            -- Competitors table
            CREATE TABLE IF NOT EXISTS competitors (
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
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
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
                competitor_callsign TEXT NOT NULL,
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
                FOREIGN KEY (competitor_callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
            );

            -- Medals table
            CREATE TABLE IF NOT EXISTS medals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                callsign TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('work', 'activate', 'combined')),
                qualified INTEGER NOT NULL DEFAULT 1,
                qso_race_medal TEXT CHECK (qso_race_medal IN ('gold', 'silver', 'bronze', NULL)),
                qso_race_claim_time TEXT,
                cool_factor_medal TEXT CHECK (cool_factor_medal IN ('gold', 'silver', 'bronze', NULL)),
                cool_factor_value REAL,
                cool_factor_claim_time TEXT,
                pota_bonus INTEGER NOT NULL DEFAULT 0,
                total_points INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                UNIQUE (match_id, callsign, role)
            );

            -- Sport entries table (opt-in)
            CREATE TABLE IF NOT EXISTS sport_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                entered_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                UNIQUE (callsign, sport_id)
            );

            -- Referee assignments table
            CREATE TABLE IF NOT EXISTS referee_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
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
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                FOREIGN KEY (qso_id) REFERENCES qsos(id) ON DELETE SET NULL
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_qsos_competitor ON qsos(competitor_callsign);
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


def seed_example_olympiad():
    """
    Seed the database with an example 2026 Radio Olympics.

    This creates a complete olympiad with 3 sports demonstrating different configurations:
    - Continental DX: Monthly continent targets (work mode only)
    - Country DX: Monthly country targets (work mode only)
    - National Park POTA: All US National Parks (work + activate, separate pools)

    Only runs if no olympiad exists yet.
    """
    with get_db() as conn:
        # Check if olympiad already exists
        existing = conn.execute("SELECT COUNT(*) FROM olympiads").fetchone()[0]
        if existing > 0:
            return  # Don't seed if data exists

        # Create the olympiad
        conn.execute("""
            INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
            VALUES ('2026 Radio Olympics', '2026-01-01', '2026-12-01', 0, 1)
        """)

        # Sport 1: Continental DX of the Year
        conn.execute("""
            INSERT INTO sports (olympiad_id, name, description, target_type, work_enabled, activate_enabled, separate_pools)
            VALUES (1, 'Continental DX of the Year',
                    'Work all continents (1 per month). Points for quick QSO and power factor (QRP). Bonus for a successful POTA hunt in that continent.',
                    'continent', 1, 0, 0)
        """)

        # Sport 2: Country DX of the Year
        conn.execute("""
            INSERT INTO sports (olympiad_id, name, description, target_type, work_enabled, activate_enabled, separate_pools)
            VALUES (1, 'Country DX of the Year',
                    'Work each country. Points for quick QSO and power factor (QRP). Bonus for a successful POTA hunt in that country.',
                    'country', 1, 0, 0)
        """)

        # Sport 3: National Park POTA Challenge
        conn.execute("""
            INSERT INTO sports (olympiad_id, name, description, target_type, work_enabled, activate_enabled, separate_pools)
            VALUES (1, 'National Park POTA Challenge',
                    'Activate and Hunt POTA! Points are awarded for both so park-to-park is your biggest score. Separate medals for hunters and activators.',
                    'park', 1, 1, 1)
        """)

        # Continental DX matches (Sport 1) - one continent per month
        continent_matches = [
            ('2026-01-01T00:00:00', '2026-02-28T23:59:00', 'NA'),  # North America
            ('2026-03-01T00:00:00', '2026-03-31T23:59:00', 'SA'),  # South America
            ('2026-04-01T00:00:00', '2026-04-30T23:59:00', 'OC'),  # Oceania
            ('2026-05-01T00:00:00', '2026-05-31T23:59:00', 'EU'),  # Europe
            ('2026-06-01T00:00:00', '2026-06-30T23:59:00', 'AF'),  # Africa
            ('2026-07-01T00:00:00', '2026-07-31T23:59:00', 'AS'),  # Asia
            ('2026-08-01T00:00:00', '2026-08-31T23:59:00', 'AN'),  # Antarctica
        ]
        for start, end, target in continent_matches:
            conn.execute(
                "INSERT INTO matches (sport_id, start_date, end_date, target_value) VALUES (1, ?, ?, ?)",
                (start, end, target)
            )

        # Country DX matches (Sport 2) - different countries each month
        # DXCC codes: 291=USA, 224=Poland, 108=Belgium, 503=Czech, 221=Norway, 45=Austria, 422=Liechtenstein, 1=Canada, 223=Sweden, 163=Finland, 339=Japan
        country_matches = [
            ('2026-01-01T00:00:00', '2026-01-31T23:59:00', '291'),  # USA
            ('2026-02-01T00:00:00', '2026-02-28T23:59:00', '224'),  # Poland
            ('2026-03-01T00:00:00', '2026-03-31T23:59:00', '108'),  # Belgium
            ('2026-04-01T00:00:00', '2026-04-30T23:59:00', '503'),  # Czech Republic
            ('2026-05-01T00:00:00', '2026-05-31T23:59:00', '221'),  # Norway
            ('2026-06-01T00:00:00', '2026-06-30T23:59:00', '45'),   # Austria
            ('2026-07-01T00:00:00', '2026-07-31T23:59:00', '422'),  # Liechtenstein
            ('2026-08-01T00:00:00', '2026-08-31T23:59:00', '1'),    # Canada
            ('2026-09-01T00:00:00', '2026-09-30T23:59:00', '223'),  # Sweden
            ('2026-10-01T00:00:00', '2026-10-31T23:59:00', '163'),  # Finland
            ('2026-11-01T00:00:00', '2026-11-30T23:59:00', '339'),  # Japan
        ]
        for start, end, target in country_matches:
            conn.execute(
                "INSERT INTO matches (sport_id, start_date, end_date, target_value) VALUES (2, ?, ?, ?)",
                (start, end, target)
            )

        # National Park POTA matches (Sport 3) - US National Parks spanning full year
        national_parks = [
            'US-0001', 'US-0004', 'US-0005', 'US-0006', 'US-0007', 'US-0008', 'US-0009',
            'US-0010', 'US-0011', 'US-0012', 'US-0014', 'US-0017', 'US-0018', 'US-0020',
            'US-0021', 'US-0022', 'US-0023', 'US-0024', 'US-0026', 'US-0028', 'US-0030',
            'US-0031', 'US-0032', 'US-0033', 'US-0034', 'US-0035', 'US-0036', 'US-0037',
            'US-0038', 'US-0039', 'US-0041', 'US-0044', 'US-0045', 'US-0046', 'US-0047',
            'US-0048', 'US-0049', 'US-0050', 'US-0051', 'US-0052', 'US-0055', 'US-0056',
            'US-0057', 'US-0058', 'US-0059', 'US-0060', 'US-0063', 'US-0064', 'US-0065',
            'US-0067', 'US-0068', 'US-0069', 'US-0070', 'US-0071', 'US-0072', 'US-0640',
            'US-0765', 'US-0779', 'US-0696', 'US-0970', 'US-12607',
        ]
        for park in national_parks:
            conn.execute(
                "INSERT INTO matches (sport_id, start_date, end_date, target_value) VALUES (3, ?, ?, ?)",
                ('2026-01-01T00:00:00', '2026-12-01T23:59:59', park)
            )

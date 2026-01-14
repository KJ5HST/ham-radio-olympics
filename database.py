"""
Database setup and models for Ham Radio Olympics.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

# Import config for DATABASE_PATH - note: config must not import database to avoid circular imports
# We use a function to get the path so tests can override it before imports
def _get_database_path():
    return os.getenv("DATABASE_PATH", "ham_olympics.db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(_get_database_path())
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


@contextmanager
def get_db_exclusive():
    """Context manager with IMMEDIATE transaction for exclusive write access.

    Use this for operations that read-then-write where concurrent modifications
    could cause race conditions (e.g., medal recomputation).
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
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

        # Migration: Fix sessions table foreign key (only if needed)
        # Check if sessions table has FK pointing to old 'contestants' table
        if 'sessions' in tables:
            fk_info = conn.execute("PRAGMA foreign_key_list(sessions)").fetchall()
            needs_sessions_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_sessions_migration:
                conn.execute("DROP TABLE sessions")
                conn.commit()

        # Migration: Fix qsos table foreign key from contestants to competitors
        if 'qsos' in tables:
            # Check if qsos FK still points to contestants by checking sqlite_master
            fk_info = conn.execute("PRAGMA foreign_key_list(qsos)").fetchall()
            needs_qsos_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_qsos_migration:
                # Recreate qsos table with correct foreign key
                conn.execute("ALTER TABLE qsos RENAME TO qsos_old")
                conn.execute("""
                    CREATE TABLE qsos (
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
                    )
                """)
                conn.execute("""
                    INSERT INTO qsos SELECT * FROM qsos_old
                """)
                conn.execute("DROP TABLE qsos_old")
                conn.commit()

        # Migration: Fix medals table foreign key from contestants to competitors
        if 'medals' in tables:
            fk_info = conn.execute("PRAGMA foreign_key_list(medals)").fetchall()
            needs_medals_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_medals_migration:
                conn.execute("ALTER TABLE medals RENAME TO medals_old")
                conn.execute("""
                    CREATE TABLE medals (
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
                    )
                """)
                conn.execute("INSERT INTO medals SELECT * FROM medals_old")
                conn.execute("DROP TABLE medals_old")
                conn.commit()

        # Migration: Fix sport_entries table foreign key
        if 'sport_entries' in tables:
            fk_info = conn.execute("PRAGMA foreign_key_list(sport_entries)").fetchall()
            needs_sport_entries_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_sport_entries_migration:
                conn.execute("ALTER TABLE sport_entries RENAME TO sport_entries_old")
                conn.execute("""
                    CREATE TABLE sport_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        callsign TEXT NOT NULL,
                        sport_id INTEGER NOT NULL,
                        entered_at TEXT NOT NULL,
                        FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                        FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                        UNIQUE (callsign, sport_id)
                    )
                """)
                conn.execute("INSERT INTO sport_entries SELECT * FROM sport_entries_old")
                conn.execute("DROP TABLE sport_entries_old")
                conn.commit()

        # Migration: Fix referee_assignments table foreign key
        if 'referee_assignments' in tables:
            fk_info = conn.execute("PRAGMA foreign_key_list(referee_assignments)").fetchall()
            needs_referee_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_referee_migration:
                conn.execute("ALTER TABLE referee_assignments RENAME TO referee_assignments_old")
                conn.execute("""
                    CREATE TABLE referee_assignments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        callsign TEXT NOT NULL,
                        sport_id INTEGER NOT NULL,
                        assigned_at TEXT NOT NULL,
                        FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                        FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                        UNIQUE (callsign, sport_id)
                    )
                """)
                conn.execute("INSERT INTO referee_assignments SELECT * FROM referee_assignments_old")
                conn.execute("DROP TABLE referee_assignments_old")
                conn.commit()

        # Migration: Fix records table foreign key
        if 'records' in tables:
            fk_info = conn.execute("PRAGMA foreign_key_list(records)").fetchall()
            needs_records_migration = any(fk[2] == 'contestants' for fk in fk_info)
            if needs_records_migration:
                conn.execute("ALTER TABLE records RENAME TO records_old")
                conn.execute("""
                    CREATE TABLE records (
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
                    )
                """)
                conn.execute("INSERT INTO records SELECT * FROM records_old")
                conn.execute("DROP TABLE records_old")
                conn.commit()

        # Migration: add LoTW credential columns if they don't exist (only for existing tables)
        if has_competitors:
            cursor = conn.execute("PRAGMA table_info(competitors)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'lotw_username_encrypted' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN lotw_username_encrypted TEXT")
                conn.execute("ALTER TABLE competitors ADD COLUMN lotw_password_encrypted TEXT")
                conn.commit()
            # Migration: add lockout columns
            if 'failed_login_attempts' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0")
                conn.execute("ALTER TABLE competitors ADD COLUMN locked_until TEXT")
                conn.commit()
            # Migration: add email_verified column
            if 'email_verified' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
                conn.commit()
            # Migration: add name columns
            if 'first_name' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN first_name TEXT")
                conn.execute("ALTER TABLE competitors ADD COLUMN last_name TEXT")
                conn.commit()

        # Migration: add match_id column to records table
        if 'records' in tables:
            cursor = conn.execute("PRAGMA table_info(records)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'match_id' not in columns:
                conn.execute("ALTER TABLE records ADD COLUMN match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE")
                conn.commit()

        # Migration: add allowed_modes column to sports table
        if 'sports' in tables:
            cursor = conn.execute("PRAGMA table_info(sports)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'allowed_modes' not in columns:
                conn.execute("ALTER TABLE sports ADD COLUMN allowed_modes TEXT")
                conn.commit()

        # Migration: add allowed_modes column to matches table
        if 'matches' in tables:
            cursor = conn.execute("PRAGMA table_info(matches)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'allowed_modes' not in columns:
                conn.execute("ALTER TABLE matches ADD COLUMN allowed_modes TEXT")
                conn.commit()

        # Migration: add max_power_w column to matches table
        if 'matches' in tables:
            cursor = conn.execute("PRAGMA table_info(matches)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'max_power_w' not in columns:
                conn.execute("ALTER TABLE matches ADD COLUMN max_power_w INTEGER")
                conn.commit()

        # Migration: add email_notifications_enabled to competitors
        if 'competitors' in tables:
            cursor = conn.execute("PRAGMA table_info(competitors)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'email_notifications_enabled' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN email_notifications_enabled INTEGER NOT NULL DEFAULT 1")
                conn.commit()

        # Migration: add notified_at to medals
        if 'medals' in tables:
            cursor = conn.execute("PRAGMA table_info(medals)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'notified_at' not in columns:
                conn.execute("ALTER TABLE medals ADD COLUMN notified_at TEXT")
                conn.commit()

        # Migration: add granular email notification preferences
        if 'competitors' in tables:
            cursor = conn.execute("PRAGMA table_info(competitors)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'email_medal_notifications' not in columns:
                conn.execute("ALTER TABLE competitors ADD COLUMN email_medal_notifications INTEGER NOT NULL DEFAULT 1")
                conn.execute("ALTER TABLE competitors ADD COLUMN email_match_reminders INTEGER NOT NULL DEFAULT 1")
                conn.execute("ALTER TABLE competitors ADD COLUMN email_record_notifications INTEGER NOT NULL DEFAULT 1")
                conn.execute("ALTER TABLE competitors ADD COLUMN last_match_digest_at TEXT")
                conn.commit()

        # Migration: update sports table to support 'any' target_type
        # SQLite doesn't support modifying CHECK constraints, so we need to recreate the table
        if 'sports' in tables:
            # Check current CHECK constraint by looking at table definition
            table_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='sports'"
            ).fetchone()
            if table_sql and "'any'" not in table_sql[0]:
                # Need to recreate table with updated CHECK constraint
                # Disable foreign keys to allow dropping referenced table
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("ALTER TABLE sports RENAME TO sports_old")
                conn.execute("""
                    CREATE TABLE sports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        olympiad_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        target_type TEXT NOT NULL CHECK (target_type IN ('continent', 'country', 'park', 'call', 'grid', 'any')),
                        work_enabled INTEGER NOT NULL DEFAULT 1,
                        activate_enabled INTEGER NOT NULL DEFAULT 0,
                        separate_pools INTEGER NOT NULL DEFAULT 0,
                        allowed_modes TEXT,
                        FOREIGN KEY (olympiad_id) REFERENCES olympiads(id) ON DELETE CASCADE
                    )
                """)
                conn.execute("INSERT INTO sports SELECT * FROM sports_old")
                conn.execute("DROP TABLE sports_old")
                # Recreate index
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sports_olympiad ON sports(olympiad_id)")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.commit()

        conn.executescript("""
            -- Competitors table
            CREATE TABLE IF NOT EXISTS competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                email_verified INTEGER NOT NULL DEFAULT 0,
                email_notifications_enabled INTEGER NOT NULL DEFAULT 1,
                email_medal_notifications INTEGER NOT NULL DEFAULT 1,
                email_match_reminders INTEGER NOT NULL DEFAULT 1,
                email_record_notifications INTEGER NOT NULL DEFAULT 1,
                last_match_digest_at TEXT,
                first_name TEXT,
                last_name TEXT,
                qrz_api_key_encrypted TEXT,
                lotw_username_encrypted TEXT,
                lotw_password_encrypted TEXT,
                registered_at TEXT NOT NULL,
                last_sync_at TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_referee INTEGER NOT NULL DEFAULT 0,
                is_disabled INTEGER NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            );

            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                callsign TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
            );

            -- Password reset tokens table
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                callsign TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
            );

            -- Email verification tokens table
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                token TEXT PRIMARY KEY,
                callsign TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
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
                target_type TEXT NOT NULL CHECK (target_type IN ('continent', 'country', 'park', 'call', 'grid', 'any')),
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                allowed_modes TEXT,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id) ON DELETE CASCADE
            );

            -- Matches table
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_value TEXT NOT NULL,
                allowed_modes TEXT,
                max_power_w INTEGER,
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
                notified_at TEXT,
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
                match_id INTEGER,
                callsign TEXT,
                record_type TEXT NOT NULL CHECK (record_type IN ('longest_distance', 'highest_cool_factor', 'lowest_power')),
                value REAL NOT NULL,
                qso_id INTEGER,
                achieved_at TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                FOREIGN KEY (qso_id) REFERENCES qsos(id) ON DELETE SET NULL
            );

            -- Audit log table
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                actor_callsign TEXT,
                action TEXT NOT NULL,
                target_type TEXT,
                target_id TEXT,
                details TEXT,
                ip_address TEXT
            );

            -- Callsign cache table (for name/country lookups)
            CREATE TABLE IF NOT EXISTS callsign_cache (
                callsign TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                country TEXT,
                dxcc INTEGER,
                grid TEXT,
                cached_at TEXT NOT NULL
            );

            -- Settings table (key-value store for admin configuration)
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                is_encrypted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            -- POTA park cache table
            CREATE TABLE IF NOT EXISTS pota_parks (
                reference TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                grid TEXT,
                cached_at TEXT NOT NULL
            );

            -- Teams table
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                captain_callsign TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (captain_callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
            );

            -- Team members table
            CREATE TABLE IF NOT EXISTS team_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                callsign TEXT NOT NULL UNIQUE,
                joined_at TEXT NOT NULL,
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE
            );

            -- Team invites/requests table
            CREATE TABLE IF NOT EXISTS team_invites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                callsign TEXT NOT NULL,
                invite_type TEXT NOT NULL CHECK (invite_type IN ('invite', 'request')),
                created_at TEXT NOT NULL,
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (callsign) REFERENCES competitors(callsign) ON DELETE CASCADE,
                UNIQUE(team_id, callsign)
            );

            -- Team medals table
            CREATE TABLE IF NOT EXISTS team_medals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                match_id INTEGER,
                sport_id INTEGER NOT NULL,
                calculation_method TEXT NOT NULL CHECK (calculation_method IN ('normalized', 'top_n', 'average', 'sum')),
                total_points REAL NOT NULL,
                member_count INTEGER NOT NULL,
                gold_count INTEGER NOT NULL DEFAULT 0,
                silver_count INTEGER NOT NULL DEFAULT 0,
                bronze_count INTEGER NOT NULL DEFAULT 0,
                medal TEXT CHECK (medal IN ('gold', 'silver', 'bronze', NULL)),
                computed_at TEXT NOT NULL,
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
                FOREIGN KEY (sport_id) REFERENCES sports(id) ON DELETE CASCADE,
                UNIQUE(team_id, match_id, sport_id, calculation_method)
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
            CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log(actor_callsign);
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
            CREATE INDEX IF NOT EXISTS idx_teams_captain ON teams(captain_callsign);
            CREATE INDEX IF NOT EXISTS idx_team_members_team ON team_members(team_id);
            CREATE INDEX IF NOT EXISTS idx_team_members_callsign ON team_members(callsign);
            CREATE INDEX IF NOT EXISTS idx_team_medals_team ON team_medals(team_id);
            CREATE INDEX IF NOT EXISTS idx_team_medals_sport ON team_medals(sport_id);
            CREATE INDEX IF NOT EXISTS idx_team_medals_match ON team_medals(match_id);
            CREATE INDEX IF NOT EXISTS idx_team_invites_team ON team_invites(team_id);
            CREATE INDEX IF NOT EXISTS idx_team_invites_callsign ON team_invites(callsign);

            -- Unique constraint to prevent duplicate QSOs
            CREATE UNIQUE INDEX IF NOT EXISTS idx_qsos_unique
            ON qsos(competitor_callsign, dx_callsign, qso_datetime_utc);

            -- Performance index for records lookups
            CREATE INDEX IF NOT EXISTS idx_records_callsign ON records(callsign);
        """)


def backfill_records():
    """
    Backfill records table from existing QSOs.

    This ensures personal bests and world records exist for QSOs
    that were synced before the records feature was added.
    Only runs if there are confirmed QSOs but no records.

    Uses recompute_all_records() to ensure only QSOs that fall within
    match date ranges are considered for records.
    """
    with get_db() as conn:
        # Check if backfill is needed
        qso_count = conn.execute(
            "SELECT COUNT(*) FROM qsos WHERE is_confirmed = 1 AND distance_km IS NOT NULL"
        ).fetchone()[0]
        record_count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

        if qso_count == 0 or record_count > 0:
            return  # No QSOs or records already exist

    # Use recompute_all_records which properly filters by match date ranges
    from scoring import recompute_all_records
    recompute_all_records()


def reset_db():
    """Reset the database (for testing)."""
    db_path = _get_database_path()
    if os.path.exists(db_path):
        os.remove(db_path)
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


def get_setting(key: str, default: Optional[str] = None, decrypt: bool = False) -> Optional[str]:
    """
    Get a setting value from the database.

    Args:
        key: Setting key
        default: Default value if not found
        decrypt: If True, decrypt the value (for sensitive settings)

    Returns:
        Setting value or default
    """
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT value, is_encrypted FROM settings WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()

        if not row or row["value"] is None:
            return default

        value = row["value"]
        if row["is_encrypted"] and decrypt:
            from crypto import decrypt_api_key
            try:
                value = decrypt_api_key(value)
            except Exception:
                return default

        return value


def set_setting(key: str, value: Optional[str], encrypt: bool = False) -> None:
    """
    Set a setting value in the database.

    Args:
        key: Setting key
        value: Setting value (None to delete)
        encrypt: If True, encrypt the value before storing
    """
    with get_db() as conn:
        if value is None:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            return

        stored_value = value
        if encrypt and value:
            from crypto import encrypt_api_key
            stored_value = encrypt_api_key(value)

        conn.execute("""
            INSERT INTO settings (key, value, is_encrypted, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                is_encrypted = excluded.is_encrypted,
                updated_at = excluded.updated_at
        """, (key, stored_value, 1 if encrypt else 0, datetime.utcnow().isoformat()))

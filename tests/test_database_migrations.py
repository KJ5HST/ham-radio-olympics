"""
Tests for database migrations - ensuring backward compatibility with old schemas.
"""

import os
import tempfile
import pytest
import sqlite3

# Set test mode BEFORE importing app modules
os.environ["TESTING"] = "1"


class TestDatabaseMigrations:
    """Test database migration paths for backward compatibility."""

    def _create_db_with_schema(self, schema_sql):
        """Create a temp database with the given schema."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.close()
        return db_path

    def test_migrate_contestants_to_competitors(self):
        """Test migration from 'contestants' to 'competitors' table."""
        # Create database with old 'contestants' table
        old_schema = """
            CREATE TABLE contestants (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                qrz_api_key_encrypted TEXT,
                registered_at TEXT NOT NULL,
                last_sync_at TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_referee INTEGER NOT NULL DEFAULT 0,
                is_disabled INTEGER NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            );
            INSERT INTO contestants (callsign, password_hash, registered_at)
            VALUES ('W1TEST', 'hash123', '2024-01-01');
        """
        db_path = self._create_db_with_schema(old_schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            # Need to reimport to pick up new db path
            import importlib
            import database
            importlib.reload(database)

            # Run init_db which should migrate
            database.init_db()

            # Verify competitors table exists and contestants doesn't
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert 'competitors' in tables
            assert 'contestants' not in tables

            # Verify data was preserved
            cursor = conn.execute("SELECT callsign FROM competitors WHERE callsign = 'W1TEST'")
            assert cursor.fetchone() is not None
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_both_tables_exist(self):
        """Test migration when both contestants and competitors exist (partial migration state)."""
        # Create database with both tables (partial migration state)
        schema = """
            CREATE TABLE contestants (
                callsign TEXT PRIMARY KEY
            );
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1BOTH', 'hash', '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert 'competitors' in tables
            assert 'contestants' not in tables  # Old empty table should be dropped
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_qsos_column_rename(self):
        """Test migration of qsos.contestant_callsign to competitor_callsign."""
        # Scenario: competitors table exists but qsos still has old column name 'contestant_callsign'
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE qsos (
                id INTEGER PRIMARY KEY,
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
                FOREIGN KEY (contestant_callsign) REFERENCES competitors(callsign)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1COL', 'hash', '2024-01-01');
            INSERT INTO qsos (contestant_callsign, dx_callsign, qso_datetime_utc)
            VALUES ('W1COL', 'W2DX', '2024-01-01T12:00:00');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA table_info(qsos)")
            columns = [row[1] for row in cursor.fetchall()]
            assert 'competitor_callsign' in columns
            assert 'contestant_callsign' not in columns
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_qsos_foreign_key(self):
        """Test migration of qsos foreign key from contestants to competitors."""
        # Scenario: competitors table already exists but qsos FK still points to 'contestants'
        # (This happens if table was manually renamed but FK not updated)
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE qsos (
                id INTEGER PRIMARY KEY,
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
                FOREIGN KEY (competitor_callsign) REFERENCES contestants(callsign)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1FK', 'hash', '2024-01-01');
            INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc)
            VALUES ('W1FK', 'W2DX', '2024-01-01T12:00:00');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            fk_info = conn.execute("PRAGMA foreign_key_list(qsos)").fetchall()
            # Verify FK now points to competitors
            assert any(fk[2] == 'competitors' for fk in fk_info)
            assert not any(fk[2] == 'contestants' for fk in fk_info)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_medals_foreign_key(self):
        """Test migration of medals foreign key from contestants to competitors."""
        # Scenario: competitors table already exists but medals FK still points to 'contestants'
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE olympiads (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sports (
                id INTEGER PRIMARY KEY,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL,
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id)
            );
            CREATE TABLE matches (
                id INTEGER PRIMARY KEY,
                sport_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_value TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id)
            );
            CREATE TABLE medals (
                id INTEGER PRIMARY KEY,
                match_id INTEGER NOT NULL,
                callsign TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('work', 'activate', 'combined')),
                qualified INTEGER NOT NULL DEFAULT 1,
                qso_race_medal TEXT,
                qso_race_claim_time TEXT,
                cool_factor_medal TEXT,
                cool_factor_value REAL,
                cool_factor_claim_time TEXT,
                pota_bonus INTEGER NOT NULL DEFAULT 0,
                total_points INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (match_id) REFERENCES matches(id),
                FOREIGN KEY (callsign) REFERENCES contestants(callsign)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1MED', 'hash', '2024-01-01');
            INSERT INTO olympiads (name, start_date, end_date, is_active) VALUES ('Test', '2024-01-01', '2024-12-31', 1);
            INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'Sport', 'country');
            INSERT INTO matches (sport_id, start_date, end_date, target_value) VALUES (1, '2024-01-01', '2024-01-31', '291');
            INSERT INTO medals (match_id, callsign, role) VALUES (1, 'W1MED', 'work');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            fk_info = conn.execute("PRAGMA foreign_key_list(medals)").fetchall()
            # Verify FK now points to competitors
            assert any(fk[2] == 'competitors' for fk in fk_info)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_add_lotw_columns(self):
        """Test adding LoTW credential columns to existing competitors table."""
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                qrz_api_key_encrypted TEXT,
                registered_at TEXT NOT NULL,
                last_sync_at TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_referee INTEGER NOT NULL DEFAULT 0,
                is_disabled INTEGER NOT NULL DEFAULT 0,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1LOTW', 'hash', '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA table_info(competitors)")
            columns = [row[1] for row in cursor.fetchall()]
            assert 'lotw_username_encrypted' in columns
            assert 'lotw_password_encrypted' in columns
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_add_lockout_columns(self):
        """Test adding lockout columns to existing competitors table."""
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                qrz_api_key_encrypted TEXT,
                lotw_username_encrypted TEXT,
                lotw_password_encrypted TEXT,
                registered_at TEXT NOT NULL,
                last_sync_at TEXT,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_referee INTEGER NOT NULL DEFAULT 0,
                is_disabled INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1LOCK', 'hash', '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA table_info(competitors)")
            columns = [row[1] for row in cursor.fetchall()]
            assert 'failed_login_attempts' in columns
            assert 'locked_until' in columns
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_sport_entries_foreign_key(self):
        """Test migration of sport_entries foreign key."""
        # Scenario: competitors table exists but sport_entries FK still points to 'contestants'
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE olympiads (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sports (
                id INTEGER PRIMARY KEY,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL,
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id)
            );
            CREATE TABLE sport_entries (
                id INTEGER PRIMARY KEY,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                entered_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign),
                FOREIGN KEY (sport_id) REFERENCES sports(id)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1SE', 'hash', '2024-01-01');
            INSERT INTO olympiads (name, start_date, end_date, is_active) VALUES ('Test', '2024-01-01', '2024-12-31', 1);
            INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'Sport', 'country');
            INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES ('W1SE', 1, '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            fk_info = conn.execute("PRAGMA foreign_key_list(sport_entries)").fetchall()
            assert any(fk[2] == 'competitors' for fk in fk_info)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_referee_assignments_foreign_key(self):
        """Test migration of referee_assignments foreign key."""
        # Scenario: competitors table exists but referee_assignments FK still points to 'contestants'
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE olympiads (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sports (
                id INTEGER PRIMARY KEY,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL,
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id)
            );
            CREATE TABLE referee_assignments (
                id INTEGER PRIMARY KEY,
                callsign TEXT NOT NULL,
                sport_id INTEGER NOT NULL,
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (callsign) REFERENCES contestants(callsign),
                FOREIGN KEY (sport_id) REFERENCES sports(id)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1REF', 'hash', '2024-01-01');
            INSERT INTO olympiads (name, start_date, end_date, is_active) VALUES ('Test', '2024-01-01', '2024-12-31', 1);
            INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'Sport', 'country');
            INSERT INTO referee_assignments (callsign, sport_id, assigned_at) VALUES ('W1REF', 1, '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            fk_info = conn.execute("PRAGMA foreign_key_list(referee_assignments)").fetchall()
            assert any(fk[2] == 'competitors' for fk in fk_info)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_migrate_records_foreign_key(self):
        """Test migration of records foreign key."""
        # Scenario: competitors table exists but records FK still points to 'contestants'
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE olympiads (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sports (
                id INTEGER PRIMARY KEY,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL,
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id)
            );
            CREATE TABLE qsos (
                id INTEGER PRIMARY KEY,
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
                FOREIGN KEY (competitor_callsign) REFERENCES competitors(callsign)
            );
            CREATE TABLE records (
                id INTEGER PRIMARY KEY,
                sport_id INTEGER,
                callsign TEXT,
                record_type TEXT NOT NULL CHECK (record_type IN ('longest_distance', 'highest_cool_factor', 'lowest_power')),
                value REAL NOT NULL,
                qso_id INTEGER,
                achieved_at TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id),
                FOREIGN KEY (callsign) REFERENCES contestants(callsign),
                FOREIGN KEY (qso_id) REFERENCES qsos(id)
            );
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES ('W1REC', 'hash', '2024-01-01');
            INSERT INTO olympiads (name, start_date, end_date, is_active) VALUES ('Test', '2024-01-01', '2024-12-31', 1);
            INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'Sport', 'country');
            INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc) VALUES ('W1REC', 'W2DX', '2024-01-01');
            INSERT INTO records (sport_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (1, 'W1REC', 'longest_distance', 1000.0, 1, '2024-01-01');
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            database.init_db()

            conn = sqlite3.connect(db_path)
            fk_info = conn.execute("PRAGMA foreign_key_list(records)").fetchall()
            assert any(fk[2] == 'competitors' for fk in fk_info)
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_seed_example_olympiad_skips_when_exists(self):
        """Test seed_example_olympiad doesn't seed when olympiad already exists."""
        # Create database with existing olympiad
        schema = """
            CREATE TABLE competitors (
                callsign TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
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
            CREATE TABLE olympiads (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                qualifying_qsos INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sports (
                id INTEGER PRIMARY KEY,
                olympiad_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                target_type TEXT NOT NULL,
                work_enabled INTEGER NOT NULL DEFAULT 1,
                activate_enabled INTEGER NOT NULL DEFAULT 0,
                separate_pools INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (olympiad_id) REFERENCES olympiads(id)
            );
            CREATE TABLE matches (
                id INTEGER PRIMARY KEY,
                sport_id INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                target_value TEXT NOT NULL,
                FOREIGN KEY (sport_id) REFERENCES sports(id)
            );
            -- Insert existing olympiad to trigger early return
            INSERT INTO olympiads (id, name, start_date, end_date, is_active)
            VALUES (99, 'Existing Olympics', '2025-01-01', '2025-12-31', 1);
        """
        db_path = self._create_db_with_schema(schema)

        try:
            os.environ["DATABASE_PATH"] = db_path
            import importlib
            import database
            importlib.reload(database)

            # Call seed_example_olympiad - should return early without adding data
            database.seed_example_olympiad()

            # Verify only the one olympiad exists (not the demo one)
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM olympiads")
            count = cursor.fetchone()[0]
            assert count == 1  # Only the existing one

            cursor = conn.execute("SELECT name FROM olympiads")
            name = cursor.fetchone()[0]
            assert name == "Existing Olympics"  # Not the demo olympiad
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

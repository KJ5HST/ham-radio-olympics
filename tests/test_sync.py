"""
Tests for sync functionality.
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, AsyncMock

# Create temp file for test database
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)

os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"

from database import reset_db, get_db
from crypto import encrypt_api_key
from sync import sync_competitor, sync_all_competitors, _upsert_qso, recompute_all_active_matches
from qrz_client import QSOData


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()
    yield


@pytest.fixture
def registered_competitor():
    """Create a registered competitor with password and QRZ API key."""
    from auth import hash_password
    with get_db() as conn:
        encrypted_key = encrypt_api_key("test-api-key")
        password_hash = hash_password("password123")
        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
            VALUES (?, ?, ?, ?)
        """, ("W1TEST", password_hash, encrypted_key, datetime.utcnow().isoformat()))
    return "W1TEST"


@pytest.fixture
def setup_olympiad():
    """Create an active Olympiad with Sport and Match."""
    with get_db() as conn:
        # Create Olympiad
        conn.execute("""
            INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, ("Test Olympics", "2026-01-01", "2026-12-31", 0))

        # Create Sport
        conn.execute("""
            INSERT INTO sports (olympiad_id, name, target_type, work_enabled, activate_enabled, separate_pools)
            VALUES (1, 'DX Challenge', 'continent', 1, 0, 0)
        """)

        # Create Match
        conn.execute("""
            INSERT INTO matches (sport_id, start_date, end_date, target_value)
            VALUES (1, '2026-01-01T00:00:00', '2026-01-31T23:59:59', 'EU')
        """)


class TestUpsertQSO:
    """Test QSO insert/update logic."""

    def test_insert_new_qso(self, registered_competitor):
        """Test inserting a new QSO."""
        qso = QSOData(
            dx_callsign="DL1ABC",
            qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
            band="20M",
            mode="SSB",
            tx_power=5.0,
            my_dxcc=291,
            my_grid="EM12",
            my_sig_info=None,
            dx_dxcc=230,
            dx_grid="JN58",
            dx_sig_info=None,
            is_confirmed=True,
            qrz_logid="12345",
        )

        with get_db() as conn:
            result = _upsert_qso(conn, "W1TEST", qso)

        assert result == "new"

        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM qsos WHERE competitor_callsign = ?", ("W1TEST",))
            qsos = cursor.fetchall()
            assert len(qsos) == 1
            assert qsos[0]["dx_callsign"] == "DL1ABC"
            assert qsos[0]["is_confirmed"] == 1

    def test_update_existing_qso(self, registered_competitor):
        """Test updating confirmation status of existing QSO."""
        # Insert unconfirmed QSO
        qso = QSOData(
            dx_callsign="DL1ABC",
            qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
            band="20M",
            mode="SSB",
            tx_power=5.0,
            my_dxcc=291,
            my_grid="EM12",
            my_sig_info=None,
            dx_dxcc=230,
            dx_grid="JN58",
            dx_sig_info=None,
            is_confirmed=False,
            qrz_logid="12345",
        )

        with get_db() as conn:
            _upsert_qso(conn, "W1TEST", qso)

        # Now update to confirmed
        qso.is_confirmed = True
        with get_db() as conn:
            result = _upsert_qso(conn, "W1TEST", qso)

        assert result == "updated"

        with get_db() as conn:
            cursor = conn.execute("SELECT is_confirmed FROM qsos WHERE qrz_logid = ?", ("12345",))
            row = cursor.fetchone()
            assert row["is_confirmed"] == 1

    def test_no_update_if_unchanged(self, registered_competitor):
        """Test no update if QSO hasn't changed."""
        qso = QSOData(
            dx_callsign="DL1ABC",
            qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
            band="20M",
            mode="SSB",
            tx_power=5.0,
            my_dxcc=291,
            my_grid="EM12",
            my_sig_info=None,
            dx_dxcc=230,
            dx_grid="JN58",
            dx_sig_info=None,
            is_confirmed=True,
            qrz_logid="12345",
        )

        with get_db() as conn:
            _upsert_qso(conn, "W1TEST", qso)
            result = _upsert_qso(conn, "W1TEST", qso)

        assert result is None

    def test_distance_calculated(self, registered_competitor):
        """Test that distance is calculated from grids."""
        qso = QSOData(
            dx_callsign="DL1ABC",
            qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
            band="20M",
            mode="SSB",
            tx_power=5.0,
            my_dxcc=291,
            my_grid="EM12",
            my_sig_info=None,
            dx_dxcc=230,
            dx_grid="JN58",
            dx_sig_info=None,
            is_confirmed=True,
            qrz_logid="12345",
        )

        with get_db() as conn:
            _upsert_qso(conn, "W1TEST", qso)

        with get_db() as conn:
            cursor = conn.execute("SELECT distance_km, cool_factor FROM qsos WHERE qrz_logid = ?", ("12345",))
            row = cursor.fetchone()
            assert row["distance_km"] is not None
            assert row["distance_km"] > 8000  # EM12 to JN58 is ~8500km
            assert row["cool_factor"] is not None
            assert row["cool_factor"] > 1600  # 8500/5 = 1700


class TestSyncCompetitor:
    """Test competitor sync."""

    def test_sync_nonexistent_competitor(self):
        """Test syncing a competitor that doesn't exist."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("NOTEXIST")
        )
        assert "error" in result
        assert "not found" in result["error"]

    def test_sync_no_api_key_configured(self):
        """Test syncing competitor without QRZ API key."""
        from auth import hash_password
        # Create competitor without API key
        with get_db() as conn:
            password_hash = hash_password("password123")
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, registered_at)
                VALUES (?, ?, ?)
            """, ("W1NOKEY", password_hash, datetime.utcnow().isoformat()))

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1NOKEY")
        )
        assert "error" in result
        assert "not configured" in result["error"]

    @patch('sync.fetch_qsos')
    def test_sync_empty_logbook(self, mock_fetch, registered_competitor):
        """Test syncing when QRZ returns no QSOs."""
        mock_fetch.return_value = []

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1TEST")
        )
        assert "message" in result
        assert "No QSOs found" in result["message"]

    @patch('sync.fetch_qsos')
    def test_sync_with_qsos(self, mock_fetch, registered_competitor):
        """Test syncing with QSO data."""
        mock_fetch.return_value = [
            QSOData(
                dx_callsign="DL1ABC",
                qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
                band="20M",
                mode="SSB",
                tx_power=5.0,
                my_dxcc=291,
                my_grid="EM12",
                my_sig_info=None,
                dx_dxcc=230,
                dx_grid="JN58",
                dx_sig_info=None,
                is_confirmed=True,
                qrz_logid="12345",
            )
        ]

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1TEST")
        )

        assert result["callsign"] == "W1TEST"
        assert result["new_qsos"] == 1
        assert result["total_fetched"] == 1


class TestRecomputeMatches:
    """Test medal recomputation."""

    def test_recompute_with_no_matches(self):
        """Test recompute with no active matches."""
        # Should not raise
        recompute_all_active_matches()

    def test_recompute_with_active_match(self, setup_olympiad, registered_competitor):
        """Test recompute updates medals."""
        # Opt competitor into sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1TEST", 1, "2026-01-01T00:00:00")
            )

        # Add a QSO that matches the EU target
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed, distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8500.0, 1700.0)
            """, ("W1TEST", "DL1ABC", "2026-01-15T12:00:00", 5.0, "EM12", "JN58", 230))

        recompute_all_active_matches()

        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM medals WHERE callsign = ?", ("W1TEST",))
            medals = cursor.fetchall()
            assert len(medals) == 1
            assert medals[0]["qso_race_medal"] == "gold"
            assert medals[0]["cool_factor_medal"] == "gold"


class TestSyncEdgeCases:
    """Test edge cases in sync functionality."""

    def test_sync_with_decrypt_error(self):
        """Test sync handles decryption errors."""
        from auth import hash_password
        with get_db() as conn:
            # Insert competitor with invalid encrypted key
            password_hash = hash_password("password123")
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("W1BAD", password_hash, "invalid_encrypted_data", datetime.utcnow().isoformat()))

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1BAD")
        )
        assert "error" in result
        assert "decrypt" in result["error"].lower() or "Failed" in result["error"]

    @patch('sync.fetch_qsos')
    def test_sync_with_api_error(self, mock_fetch, registered_competitor):
        """Test sync handles API errors."""
        from qrz_client import QRZAPIError
        mock_fetch.side_effect = QRZAPIError("API connection failed")

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1TEST")
        )
        assert "error" in result
        assert "API connection failed" in result["error"]

    @patch('sync.fetch_qsos')
    def test_sync_updates_existing_qso(self, mock_fetch, registered_competitor):
        """Test updating an existing QSO confirmation status."""
        # First sync with unconfirmed QSO
        mock_fetch.return_value = [
            QSOData(
                dx_callsign="DL1ABC",
                qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
                band="20M",
                mode="SSB",
                tx_power=5.0,
                my_dxcc=291,
                my_grid="EM12",
                my_sig_info=None,
                dx_dxcc=230,
                dx_grid="JN58",
                dx_sig_info=None,
                is_confirmed=False,
                qrz_logid="12345",
            )
        ]

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1TEST")
        )
        assert result["new_qsos"] == 1

        # Second sync with same QSO now confirmed
        mock_fetch.return_value = [
            QSOData(
                dx_callsign="DL1ABC",
                qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
                band="20M",
                mode="SSB",
                tx_power=5.0,
                my_dxcc=291,
                my_grid="EM12",
                my_sig_info=None,
                dx_dxcc=230,
                dx_grid="JN58",
                dx_sig_info=None,
                is_confirmed=True,
                qrz_logid="12345",
            )
        ]

        result = asyncio.get_event_loop().run_until_complete(
            sync_competitor("W1TEST")
        )
        assert result["updated_qsos"] == 1

    def test_upsert_qso_with_invalid_grid(self, registered_competitor):
        """Test QSO with invalid grid format."""
        qso = QSOData(
            dx_callsign="DL1ABC",
            qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
            band="20M",
            mode="SSB",
            tx_power=5.0,
            my_dxcc=291,
            my_grid="INVALID",  # Invalid grid
            my_sig_info=None,
            dx_dxcc=230,
            dx_grid="ALSO_BAD",  # Invalid grid
            dx_sig_info=None,
            is_confirmed=True,
            qrz_logid="99999",
        )

        with get_db() as conn:
            result = _upsert_qso(conn, "W1TEST", qso)

        assert result == "new"

        # Verify QSO was saved but without distance/cool_factor
        with get_db() as conn:
            cursor = conn.execute("SELECT distance_km, cool_factor FROM qsos WHERE qrz_logid = ?", ("99999",))
            row = cursor.fetchone()
            assert row["distance_km"] is None
            assert row["cool_factor"] is None


class TestSyncAllCompetitors:
    """Test sync_all_competitors function."""

    @patch('sync.fetch_qsos')
    def test_sync_all_competitors(self, mock_fetch):
        """Test syncing all competitors."""
        from auth import hash_password
        # Register two competitors
        with get_db() as conn:
            encrypted_key = encrypt_api_key("test-api-key")
            password_hash = hash_password("password123")
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("W1TEST", password_hash, encrypted_key, datetime.utcnow().isoformat()))
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("K2TEST", password_hash, encrypted_key, datetime.utcnow().isoformat()))

        mock_fetch.return_value = [
            QSOData(
                dx_callsign="DL1ABC",
                qso_datetime=datetime(2026, 1, 15, 12, 0, 0),
                band="20M",
                mode="SSB",
                tx_power=5.0,
                my_dxcc=291,
                my_grid="EM12",
                my_sig_info=None,
                dx_dxcc=230,
                dx_grid="JN58",
                dx_sig_info=None,
                is_confirmed=True,
                qrz_logid="12345",
            )
        ]

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_all_competitors()
        )

        assert result["competitors_synced"] == 2
        assert result["total_new_qsos"] == 2
        assert result["errors"] == []

    @patch('sync.fetch_qsos')
    def test_sync_all_with_errors(self, mock_fetch):
        """Test sync_all handles individual errors."""
        from qrz_client import QRZAPIError
        from auth import hash_password

        # Register one competitor
        with get_db() as conn:
            encrypted_key = encrypt_api_key("test-api-key")
            password_hash = hash_password("password123")
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("W1ERR", password_hash, encrypted_key, datetime.utcnow().isoformat()))

        mock_fetch.side_effect = QRZAPIError("API error")

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            sync_all_competitors()
        )

        assert result["competitors_synced"] == 1
        assert len(result["errors"]) == 1

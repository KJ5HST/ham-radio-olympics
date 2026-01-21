"""
Tests for QSO reset functionality.
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, AsyncMock

# Create temp file for test database
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)

# Set test database and admin key BEFORE importing app modules
os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"
os.environ["TESTING"] = "1"

from fastapi.testclient import TestClient
from main import app
from database import reset_db, get_db
from auth import create_session, SESSION_COOKIE_NAME, hash_password
from crypto import encrypt_api_key
from sync import delete_competitor_qsos


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    """Cleanup database file after all tests."""
    yield
    if os.path.exists(_test_db_path):
        os.remove(_test_db_path)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def admin_headers():
    """Admin authentication headers."""
    return {"X-Admin-Key": "test-admin-key"}


@pytest.fixture
def registered_competitor():
    """Create a registered competitor with QRZ API key stored."""
    with get_db() as conn:
        encrypted_key = encrypt_api_key("test-api-key")
        password_hash = hash_password("password123")
        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
            VALUES (?, ?, ?, ?)
        """, ("W1TEST", password_hash, encrypted_key, datetime.utcnow().isoformat()))
    return "W1TEST"


@pytest.fixture
def registered_competitor_no_creds():
    """Create a registered competitor WITHOUT stored credentials."""
    with get_db() as conn:
        password_hash = hash_password("password123")
        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES (?, ?, ?)
        """, ("W1NOCRED", password_hash, datetime.utcnow().isoformat()))
    return "W1NOCRED"


@pytest.fixture
def setup_olympiad():
    """Create an active Olympiad with Sport and Match for medal tests."""
    with get_db() as conn:
        # Create Olympiad
        conn.execute("""
            INSERT INTO olympiads (id, name, start_date, end_date, qualifying_qsos, is_active)
            VALUES (1, 'Test Olympics', '2026-01-01', '2026-12-31', 0, 1)
        """)
        # Create Sport
        conn.execute("""
            INSERT INTO sports (id, olympiad_id, name, target_type, work_enabled, activate_enabled, separate_pools)
            VALUES (1, 1, 'DX Challenge', 'continent', 1, 0, 0)
        """)
        # Create Match
        conn.execute("""
            INSERT INTO matches (id, sport_id, start_date, end_date, target_value)
            VALUES (1, 1, '2026-01-01T00:00:00', '2026-01-31T23:59:59', 'EU')
        """)


@pytest.fixture
def competitor_with_qsos(registered_competitor, setup_olympiad):
    """Create a competitor with some QSOs and medals."""
    with get_db() as conn:
        # Add some QSOs
        for i in range(5):
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, is_confirmed)
                VALUES (?, ?, ?, ?)
            """, (registered_competitor, f"DL{i}ABC", f"2026-01-15T12:0{i}:00", 1))

        # Add a medal (now match_id=1 exists)
        conn.execute("""
            INSERT INTO medals (callsign, match_id, role, qso_race_medal, total_points)
            VALUES (?, ?, ?, ?, ?)
        """, (registered_competitor, 1, "work", "gold", 3))

    return registered_competitor


@pytest.fixture
def auth_cookies(registered_competitor):
    """Get auth cookies for logged-in user."""
    session_id = create_session(registered_competitor)
    return {SESSION_COOKIE_NAME: session_id}


@pytest.fixture
def auth_cookies_no_creds(registered_competitor_no_creds):
    """Get auth cookies for logged-in user without credentials."""
    session_id = create_session(registered_competitor_no_creds)
    return {SESSION_COOKIE_NAME: session_id}


class TestDeleteCompetitorQsos:
    """Test the delete_competitor_qsos helper function."""

    def test_delete_qsos_returns_count(self, competitor_with_qsos):
        """Test that delete returns count of deleted QSOs."""
        count = delete_competitor_qsos(competitor_with_qsos)
        assert count == 5

    def test_delete_qsos_removes_all_qsos(self, competitor_with_qsos):
        """Test that all QSOs are deleted."""
        delete_competitor_qsos(competitor_with_qsos)

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM qsos WHERE competitor_callsign = ?",
                (competitor_with_qsos,)
            )
            assert cursor.fetchone()[0] == 0

    def test_delete_qsos_removes_medals(self, competitor_with_qsos):
        """Test that medals are deleted."""
        delete_competitor_qsos(competitor_with_qsos)

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM medals WHERE callsign = ?",
                (competitor_with_qsos,)
            )
            assert cursor.fetchone()[0] == 0

    def test_delete_qsos_clears_last_sync(self, competitor_with_qsos):
        """Test that last_sync_at is cleared."""
        # First set a last_sync_at value
        with get_db() as conn:
            conn.execute(
                "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
                (datetime.utcnow().isoformat(), competitor_with_qsos)
            )

        delete_competitor_qsos(competitor_with_qsos)

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT last_sync_at FROM competitors WHERE callsign = ?",
                (competitor_with_qsos,)
            )
            assert cursor.fetchone()[0] is None

    def test_delete_qsos_case_insensitive(self, competitor_with_qsos):
        """Test that deletion works with lowercase callsign."""
        count = delete_competitor_qsos(competitor_with_qsos.lower())
        assert count == 5

    def test_delete_qsos_returns_zero_for_no_qsos(self, registered_competitor):
        """Test that delete returns 0 when no QSOs exist."""
        count = delete_competitor_qsos(registered_competitor)
        assert count == 0


class TestPreflightEndpoint:
    """Test the /qsos/reset/preflight endpoint."""

    def test_preflight_requires_auth(self, client):
        """Test that preflight requires authentication."""
        # Clear cookies to ensure no auth
        client.cookies.clear()
        response = client.post("/qsos/reset/preflight")
        # Endpoint should return 401 for unauthenticated requests
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"

    def test_preflight_returns_credentials_status(self, client, auth_cookies, registered_competitor):
        """Test preflight returns credential status for user with stored creds."""
        response = client.post("/qsos/reset/preflight", cookies=auth_cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["has_qrz"] is True
        assert data["has_lotw"] is False
        assert data["can_reset"] is True

    def test_preflight_no_credentials(self, client, auth_cookies_no_creds):
        """Test preflight for user with no stored credentials."""
        response = client.post("/qsos/reset/preflight", cookies=auth_cookies_no_creds)
        assert response.status_code == 200
        data = response.json()
        assert data["has_qrz"] is False
        assert data["has_lotw"] is False
        assert data["can_reset"] is False

    def test_preflight_with_lotw_credentials(self, client, registered_competitor, auth_cookies):
        """Test preflight with LoTW credentials stored."""
        # Add LoTW credentials
        with get_db() as conn:
            conn.execute(
                """UPDATE competitors SET
                   lotw_username_encrypted = ?, lotw_password_encrypted = ?
                   WHERE callsign = ?""",
                (encrypt_api_key("W1TEST"), encrypt_api_key("password"), registered_competitor)
            )

        response = client.post("/qsos/reset/preflight", cookies=auth_cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["has_qrz"] is True
        assert data["has_lotw"] is True
        assert data["can_reset"] is True


class TestResetEndpoint:
    """Test the /qsos/reset endpoint."""

    def test_reset_requires_auth(self, client):
        """Test that reset requires authentication."""
        response = client.post("/qsos/reset")
        assert response.status_code == 401

    def test_reset_requires_stored_credentials(self, client, auth_cookies_no_creds):
        """Test that reset fails without stored credentials."""
        response = client.post("/qsos/reset", cookies=auth_cookies_no_creds)
        assert response.status_code == 400
        assert "No stored credentials" in response.json()["detail"]

    @patch('main.sync_competitor', new_callable=AsyncMock)
    def test_reset_deletes_qsos_and_syncs(self, mock_sync, client, competitor_with_qsos, auth_cookies):
        """Test that reset deletes QSOs and triggers sync."""
        mock_sync.return_value = {"new_qsos": 10, "updated_qsos": 0}

        response = client.post("/qsos/reset", cookies=auth_cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_qsos"] == 5
        assert "qrz_sync" in data
        mock_sync.assert_called_once()


class TestResetWithKeyEndpoint:
    """Test the /qsos/reset-with-key endpoint."""

    def test_reset_with_key_requires_auth(self, client):
        """Test that endpoint requires authentication."""
        response = client.post("/qsos/reset-with-key", json={"qrz_api_key": "test"})
        assert response.status_code == 401

    def test_reset_with_key_requires_credentials(self, client, auth_cookies):
        """Test that at least one credential is required."""
        response = client.post("/qsos/reset-with-key", json={}, cookies=auth_cookies)
        assert response.status_code == 400
        assert "Please provide" in response.json()["detail"]

    @patch('main.verify_api_key', new_callable=AsyncMock)
    @patch('main.sync_competitor_with_key', new_callable=AsyncMock)
    def test_reset_with_qrz_key(self, mock_sync, mock_verify, client, competitor_with_qsos, auth_cookies):
        """Test reset with QRZ API key validates and syncs."""
        mock_verify.return_value = True
        mock_sync.return_value = {"new_qsos": 10, "updated_qsos": 0}

        response = client.post(
            "/qsos/reset-with-key",
            json={"qrz_api_key": "new-api-key"},
            cookies=auth_cookies
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_qsos"] == 5
        mock_verify.assert_called_once()
        mock_sync.assert_called_once()

    @patch('main.verify_api_key', new_callable=AsyncMock)
    def test_reset_with_invalid_key_fails(self, mock_verify, client, auth_cookies):
        """Test that invalid API key is rejected before deletion."""
        mock_verify.return_value = False

        response = client.post(
            "/qsos/reset-with-key",
            json={"qrz_api_key": "invalid-key"},
            cookies=auth_cookies
        )
        assert response.status_code == 400
        assert "Invalid QRZ API key" in response.json()["detail"]


class TestAdminResetEndpoint:
    """Test the /admin/competitor/{callsign}/reset-qsos endpoint."""

    def test_admin_reset_requires_admin(self, client, registered_competitor):
        """Test that endpoint requires admin authentication."""
        response = client.post(f"/admin/competitor/{registered_competitor}/reset-qsos")
        assert response.status_code in [401, 403]

    def test_admin_reset_404_for_unknown_user(self, client, admin_headers):
        """Test 404 for unknown competitor."""
        response = client.post(
            "/admin/competitor/UNKNOWN/reset-qsos",
            headers=admin_headers
        )
        assert response.status_code == 404

    def test_admin_reset_fails_without_stored_creds(self, client, admin_headers, registered_competitor_no_creds):
        """Test that admin reset fails if competitor has no stored credentials."""
        response = client.post(
            f"/admin/competitor/{registered_competitor_no_creds}/reset-qsos",
            headers=admin_headers
        )
        assert response.status_code == 400
        assert "no stored credentials" in response.json()["detail"]

    @patch('main.sync_competitor', new_callable=AsyncMock)
    def test_admin_reset_deletes_and_syncs(self, mock_sync, client, admin_headers, competitor_with_qsos):
        """Test admin reset deletes QSOs and syncs."""
        mock_sync.return_value = {"new_qsos": 15, "updated_qsos": 0}

        response = client.post(
            f"/admin/competitor/{competitor_with_qsos}/reset-qsos",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_qsos"] == 5
        assert data["callsign"] == competitor_with_qsos
        mock_sync.assert_called_once()


class TestAdminSyncEndpoint:
    """Test the /admin/competitor/{callsign}/sync endpoint."""

    def test_admin_sync_requires_admin(self, client, registered_competitor):
        """Test that endpoint requires admin authentication."""
        response = client.post(f"/admin/competitor/{registered_competitor}/sync")
        assert response.status_code in [401, 403]

    def test_admin_sync_404_for_unknown_user(self, client, admin_headers):
        """Test 404 for unknown competitor."""
        response = client.post(
            "/admin/competitor/UNKNOWN/sync",
            headers=admin_headers
        )
        assert response.status_code == 404

    def test_admin_sync_fails_without_stored_creds(self, client, admin_headers, registered_competitor_no_creds):
        """Test that admin sync fails if competitor has no stored credentials."""
        response = client.post(
            f"/admin/competitor/{registered_competitor_no_creds}/sync",
            headers=admin_headers
        )
        assert response.status_code == 400
        assert "no stored credentials" in response.json()["detail"]

    @patch('main.sync_competitor', new_callable=AsyncMock)
    def test_admin_sync_triggers_sync(self, mock_sync, client, admin_headers, registered_competitor):
        """Test admin sync triggers incremental sync."""
        mock_sync.return_value = {"new_qsos": 5, "updated_qsos": 2}

        response = client.post(
            f"/admin/competitor/{registered_competitor}/sync",
            headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["callsign"] == registered_competitor
        assert "qrz_sync" in data
        mock_sync.assert_called_once_with(registered_competitor)

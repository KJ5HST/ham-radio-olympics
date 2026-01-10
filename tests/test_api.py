"""
Tests for API endpoints.
"""

import pytest
import os
import tempfile
from datetime import datetime

# Create temp file for test database
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)

# Set test database and admin key BEFORE importing app modules
os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"

from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.datastructures import Headers, QueryParams
from main import app, verify_admin_or_referee, format_target_display
from database import reset_db, init_db, get_db
from auth import create_session, SESSION_COOKIE_NAME


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()
    yield
    # Cleanup after test


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


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def signup_user(client, callsign, password="password123", qrz_api_key="test-api-key"):
    """Helper to create a user via signup."""
    return client.post("/signup", json={
        "callsign": callsign,
        "password": password,
        "qrz_api_key": qrz_api_key
    })


class TestRegistration:
    """Test competitor registration via signup."""

    def test_register_new_competitor(self, client):
        """Test registering a new competitor."""
        response = signup_user(client, "W1ABC")
        # Signup redirects on success
        assert response.status_code in [200, 303]

    def test_register_duplicate_rejected(self, client):
        """Test duplicate registration is rejected."""
        signup_user(client, "W1ABC")
        client.cookies.clear()

        response = client.post("/signup", json={
            "callsign": "W1ABC",
            "password": "password456",
            "qrz_api_key": "test-api-key"
        })

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_register_normalizes_callsign(self, client):
        """Test callsign is normalized to uppercase."""
        response = client.post("/signup", json={
            "callsign": "w1abc",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        # Check user was created with uppercase callsign
        assert response.status_code in [200, 303]

        with get_db() as conn:
            cursor = conn.execute("SELECT callsign FROM competitors WHERE callsign = 'W1ABC'")
            assert cursor.fetchone() is not None

    def test_register_invalid_callsign_rejected(self, client):
        """Test invalid callsign format is rejected."""
        response = client.post("/signup", json={
            "callsign": "A",  # Too short
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })

        assert response.status_code == 422  # Validation error

    def test_signup_calls_sync_competitor_with_api_key(self, client):
        """Test that signup syncs QRZ data when API key is provided."""
        from unittest.mock import patch, AsyncMock

        with patch('main.sync_competitor_with_key', new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {"synced": 0}
            response = client.post("/signup", json={
                "callsign": "W1SYN",
                "password": "password123",
                "qrz_api_key": "test-api-key"
            })
            assert response.status_code in [200, 303]
            mock_sync.assert_called_once_with("W1SYN", "test-api-key")

    def test_signup_requires_qrz_or_lotw(self, client):
        """Test that signup requires either QRZ API key or LoTW credentials."""
        response = client.post("/signup", json={
            "callsign": "W1NOS",
            "password": "password123"
            # No QRZ API key or LoTW credentials
        })
        assert response.status_code == 400
        assert "QRZ API key and/or LoTW" in response.json()["detail"]

    def test_signup_succeeds_if_sync_fails(self, client):
        """Test that signup succeeds even if QRZ sync fails."""
        from unittest.mock import patch, AsyncMock

        with patch('main.sync_competitor_with_key', new_callable=AsyncMock) as mock_sync:
            mock_sync.side_effect = Exception("QRZ API error")
            response = client.post("/signup", json={
                "callsign": "W1ERR",
                "password": "password123",
                "qrz_api_key": "test-api-key"
            })
            # Signup should still succeed
            assert response.status_code in [200, 303]

            # User should still be created
            with get_db() as conn:
                cursor = conn.execute("SELECT callsign FROM competitors WHERE callsign = 'W1ERR'")
                assert cursor.fetchone() is not None

    def test_signup_rejects_invalid_qrz_key(self, client, mock_qrz_verify):
        """Test that signup fails when QRZ API key verification fails."""
        mock_qrz_verify.return_value = False
        response = client.post("/signup", json={
            "callsign": "W1BAD",
            "password": "password123",
            "qrz_api_key": "invalid-api-key"
        })
        assert response.status_code == 400
        assert "Invalid QRZ API key" in response.json()["detail"]


class TestCallsignValidation:
    """Test callsign format validation."""

    def test_valid_us_callsigns(self):
        """Test valid US amateur callsigns."""
        from main import is_valid_callsign_format

        valid_callsigns = [
            "W1AW", "K2ABC", "N3XYZ", "KD5DX", "WA1ABC",
            "WB4SON", "KC2ABC", "KG7ABC", "W1A", "K2AB",
            "N3A", "AA1A", "AB2CD", "KK6ABC"
        ]
        for call in valid_callsigns:
            assert is_valid_callsign_format(call), f"{call} should be valid"

    def test_valid_international_callsigns(self):
        """Test valid international amateur callsigns."""
        from main import is_valid_callsign_format

        valid_callsigns = [
            "VE3ABC", "G4XYZ", "DL1ABC", "JA1ABC", "VK2ABC",
            "ZL1ABC", "F5ABC", "I1ABC", "EA3ABC"
        ]
        for call in valid_callsigns:
            assert is_valid_callsign_format(call), f"{call} should be valid"

    def test_invalid_callsigns(self):
        """Test invalid callsign formats."""
        from main import is_valid_callsign_format

        invalid_callsigns = [
            "123", "ABC", "A1", "ABCDEFGHIJ",
            "W1", "1ABC", "W-1ABC", "W1ABC!", "", "A"
        ]
        for call in invalid_callsigns:
            assert not is_valid_callsign_format(call), f"{call} should be invalid"

    def test_signup_rejects_invalid_format(self, client):
        """Test that signup rejects invalid callsign format."""
        response = client.post("/signup", json={
            "callsign": "123INVALID",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        assert response.status_code == 422  # Validation error


class TestOlympiadEndpoints:
    """Test Olympiad-related endpoints."""

    def test_get_olympiad_no_active(self, client):
        """Test getting Olympiad when none active."""
        response = client.get("/olympiad")
        assert response.status_code == 404

    def test_get_olympiad_with_active(self, client, admin_headers):
        """Test getting active Olympiad."""
        # Create and activate an Olympiad
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)

        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        response = client.get("/olympiad")
        assert response.status_code == 200
        assert response.json()["name"] == "2026 Olympics"

    def test_get_olympiad_sports_empty(self, client, admin_headers):
        """Test getting Sports when none exist."""
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        response = client.get("/olympiad/sports")
        assert response.status_code == 200
        assert response.json() == []


class TestAdminEndpoints:
    """Test admin endpoints."""

    def test_admin_requires_auth(self, client):
        """Test admin endpoints require authentication."""
        response = client.get("/admin")
        assert response.status_code == 403

    def test_admin_with_valid_key(self, client, admin_headers):
        """Test admin dashboard returns HTML page."""
        response = client.get("/admin", headers=admin_headers)
        assert response.status_code == 200
        assert "Admin Dashboard" in response.text
        assert "Olympiads" in response.text

    def test_admin_with_query_param(self, client):
        """Test admin key via query parameter returns HTML page."""
        response = client.get("/admin?admin_key=test-admin-key")
        assert response.status_code == 200
        assert "Admin Dashboard" in response.text

    def test_admin_with_logged_in_admin_user(self, client):
        """Test admin access via logged-in admin user."""
        from database import get_db
        # Create admin user
        client.post("/signup", json={"callsign": "W1ADM", "password": "password123", "qrz_api_key": "test-api-key"})
        with get_db() as conn:
            conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = 'W1ADM'")
        # Login
        client.post("/login", json={"callsign": "W1ADM", "password": "password123"})
        # Access admin without key
        response = client.get("/admin")
        assert response.status_code == 200
        assert "Admin Dashboard" in response.text

    def test_create_olympiad(self, client, admin_headers):
        """Test creating an Olympiad."""
        response = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 3
        }, headers=admin_headers)

        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_update_olympiad(self, client, admin_headers):
        """Test updating an Olympiad."""
        client.post("/admin/olympiad", json={
            "name": "Original",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)

        response = client.put("/admin/olympiad/1", json={
            "name": "Updated",
            "start_date": "2026-02-01",
            "end_date": "2026-11-30",
            "qualifying_qsos": 5
        }, headers=admin_headers)

        assert response.status_code == 200

        # Verify update
        get_response = client.get("/admin/olympiad/1", headers=admin_headers)
        data = get_response.json()
        assert data["name"] == "Updated"
        assert data["qualifying_qsos"] == 5

    def test_delete_olympiad(self, client, admin_headers):
        """Test deleting an Olympiad."""
        client.post("/admin/olympiad", json={
            "name": "To Delete",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)

        response = client.delete("/admin/olympiad/1", headers=admin_headers)
        assert response.status_code == 200

        # Verify deleted
        get_response = client.get("/admin/olympiad/1", headers=admin_headers)
        assert get_response.status_code == 404

    def test_activate_olympiad(self, client, admin_headers):
        """Test activating an Olympiad deactivates others."""
        # Create two Olympiads
        client.post("/admin/olympiad", json={
            "name": "First",
            "start_date": "2026-01-01",
            "end_date": "2026-06-30",
            "qualifying_qsos": 0
        }, headers=admin_headers)

        client.post("/admin/olympiad", json={
            "name": "Second",
            "start_date": "2026-07-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)

        # Activate first
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        first = client.get("/admin/olympiad/1", headers=admin_headers).json()
        assert first["is_active"] == 1

        # Activate second
        client.post("/admin/olympiad/2/activate", headers=admin_headers)

        first = client.get("/admin/olympiad/1", headers=admin_headers).json()
        second = client.get("/admin/olympiad/2", headers=admin_headers).json()

        assert first["is_active"] == 0
        assert second["is_active"] == 1

    def test_deactivate_olympiad(self, client, admin_headers):
        """Test deactivating (pausing) an Olympiad."""
        # Create and activate an Olympiad
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        # Verify it's active
        olympiad = client.get("/admin/olympiad/1", headers=admin_headers).json()
        assert olympiad["is_active"] == 1

        # Deactivate it
        response = client.post("/admin/olympiad/1/deactivate", headers=admin_headers)
        assert response.status_code == 200

        # Verify it's now inactive
        olympiad = client.get("/admin/olympiad/1", headers=admin_headers).json()
        assert olympiad["is_active"] == 0


class TestSportEndpoints:
    """Test Sport management endpoints."""

    @pytest.fixture
    def olympiad(self, client, admin_headers):
        """Create an Olympiad for testing."""
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        return 1

    def test_create_sport(self, client, admin_headers, olympiad):
        """Test creating a Sport."""
        response = client.post(f"/admin/olympiad/{olympiad}/sport", json={
            "name": "DX Challenge",
            "description": "Work DX stations",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_get_sport(self, client, admin_headers, olympiad):
        """Test getting Sport details."""
        client.post(f"/admin/olympiad/{olympiad}/sport", json={
            "name": "POTA Championship",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)

        response = client.get("/admin/sport/1", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "POTA Championship"
        assert data["target_type"] == "park"
        assert data["separate_pools"] == 1


class TestMatchEndpoints:
    """Test Match management endpoints."""

    @pytest.fixture
    def sport(self, client, admin_headers):
        """Create Olympiad and Sport for testing."""
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        return 1

    def test_create_match(self, client, admin_headers, sport):
        """Test creating a Match."""
        response = client.post(f"/admin/sport/{sport}/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_create_match_with_gap(self, client, admin_headers, sport):
        """Test creating Matches with a gap between them."""
        # January
        client.post(f"/admin/sport/{sport}/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # Skip February (for Hamvention prep or whatever)

        # March
        response = client.post(f"/admin/sport/{sport}/match", json={
            "start_date": "2026-03-01T00:00:00",
            "end_date": "2026-03-31T23:59:59",
            "target_value": "AS"
        }, headers=admin_headers)

        assert response.status_code == 200

        # Get all matches - endpoint returns HTML, check it contains both targets
        matches_page = client.get(f"/admin/sport/{sport}/matches", headers=admin_headers)
        assert matches_page.status_code == 200
        assert "EU" in matches_page.text
        assert "AS" in matches_page.text

    def test_get_match_details(self, client, admin_headers, sport):
        """Test getting Match details."""
        client.post(f"/admin/sport/{sport}/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        response = client.get("/admin/match/1", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["target_value"] == "EU"

    def test_match_page_shows_country_name(self, client, admin_headers):
        """Test match page displays country name with DXCC code."""
        # Create olympiad and sport with country target type
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "Country DX",
            "target_type": "country",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        # Create match targeting USA (291)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "291"
        }, headers=admin_headers)

        # View match page (need to be logged in)
        signup_user(client, "W1TEST")
        response = client.get("/olympiad/sport/1/match/1")
        assert response.status_code == 200
        assert "United States" in response.text
        assert "291" in response.text


class TestCompetitorEndpoints:
    """Test competitor-related endpoints."""

    def test_get_competitor(self, client):
        """Test getting competitor details page."""
        # Register
        signup_user(client, "K1ABC")

        response = client.get("/competitor/K1ABC")
        assert response.status_code == 200
        assert "K1ABC" in response.text
        assert "Gold Medals" in response.text
        assert "Total Points" in response.text

    def test_get_competitor_not_found(self, client):
        """Test getting non-existent competitor."""
        # Must be logged in to access competitor pages
        signup_user(client, "W1TST")
        response = client.get("/competitor/UNKNOWN")
        assert response.status_code == 404

    def test_get_competitor_requires_auth(self, client):
        """Test competitor page requires authentication."""
        response = client.get("/competitor/ANYONE", follow_redirects=False)
        assert response.status_code == 401

    def test_delete_competitor(self, client, admin_headers):
        """Test admin can delete competitor."""
        # Create the competitor to delete
        signup_user(client, "W2XYZ")
        client.post("/logout")

        # Sign up as different user to verify deletion
        signup_user(client, "W2CHK")

        response = client.delete("/admin/competitor/W2XYZ", headers=admin_headers)
        assert response.status_code == 200

        # Verify deleted - need to be logged in to check
        get_response = client.get("/competitor/W2XYZ")
        assert get_response.status_code == 404

    def test_disable_competitor(self, client, admin_headers):
        """Test admin can disable a competitor's account."""
        signup_user(client, "W1DIS")
        client.post("/logout")

        response = client.post("/admin/competitor/W1DIS/disable", headers=admin_headers)
        assert response.status_code == 200
        assert "disabled" in response.json()["message"]

    def test_disable_competitor_not_found(self, client, admin_headers):
        """Test disabling non-existent competitor."""
        response = client.post("/admin/competitor/NOTEXIST/disable", headers=admin_headers)
        assert response.status_code == 404

    def test_enable_competitor(self, client, admin_headers):
        """Test admin can enable a disabled competitor's account."""
        signup_user(client, "W1ENA")
        client.post("/logout")

        # Disable first
        client.post("/admin/competitor/W1ENA/disable", headers=admin_headers)

        # Then enable
        response = client.post("/admin/competitor/W1ENA/enable", headers=admin_headers)
        assert response.status_code == 200
        assert "enabled" in response.json()["message"]

    def test_enable_competitor_not_found(self, client, admin_headers):
        """Test enabling non-existent competitor."""
        response = client.post("/admin/competitor/NOTEXIST/enable", headers=admin_headers)
        assert response.status_code == 404

    def test_reset_password(self, client, admin_headers):
        """Test admin can reset a competitor's password."""
        signup_user(client, "W1RST")
        client.post("/logout")

        response = client.post("/admin/competitor/W1RST/reset-password", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "new_password" in data
        assert len(data["new_password"]) > 8

        # Verify old password no longer works
        login_response = client.post("/login", json={
            "callsign": "W1RST",
            "password": "password123"
        })
        assert login_response.status_code == 401

        # Verify new password works
        login_response = client.post("/login", json={
            "callsign": "W1RST",
            "password": data["new_password"]
        })
        assert login_response.status_code in [200, 303]

    def test_reset_password_not_found(self, client, admin_headers):
        """Test resetting password for non-existent competitor."""
        response = client.post("/admin/competitor/NOTEXIST/reset-password", headers=admin_headers)
        assert response.status_code == 404

    def test_reset_password_invalidates_sessions(self, client, admin_headers):
        """Test that resetting password logs out the user."""
        signup_user(client, "W1SES")
        # User is now logged in with a session

        # Get their session cookie
        session_cookie = client.cookies.get("hro_session")
        assert session_cookie is not None

        # Admin resets password
        client.post("/admin/competitor/W1SES/reset-password", headers=admin_headers)

        # Session should be invalidated - try to access a protected API endpoint
        client.cookies.clear()
        client.cookies.set("hro_session", session_cookie)
        response = client.post("/sport/1/enter")
        # Should fail with 401 since session is invalid
        assert response.status_code == 401

    def test_disqualify_competitor(self, client, admin_headers):
        """Test admin can disqualify a competitor from competition."""
        # Setup competition
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create competitor and enter sport
        signup_user(client, "W1DQ")
        client.post("/sport/1/enter")
        client.post("/logout")

        # Disqualify
        response = client.post("/admin/competitor/W1DQ/disqualify", headers=admin_headers)
        assert response.status_code == 200
        assert "disqualified" in response.json()["message"]

    def test_disqualify_competitor_not_found(self, client, admin_headers):
        """Test disqualifying non-existent competitor."""
        # Need active olympiad for disqualify
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        response = client.post("/admin/competitor/NOTEXIST/disqualify", headers=admin_headers)
        assert response.status_code == 404

    def test_disqualify_competitor_no_active_olympiad(self, client, admin_headers):
        """Test disqualifying when no active competition."""
        signup_user(client, "W1NODQ")
        client.post("/logout")

        response = client.post("/admin/competitor/W1NODQ/disqualify", headers=admin_headers)
        assert response.status_code == 400
        assert "No active competition" in response.json()["detail"]

    def test_set_admin_role(self, client, admin_headers):
        """Test setting admin role."""
        signup_user(client, "W1ADM")
        client.post("/logout")

        response = client.post("/admin/competitor/W1ADM/set-admin", headers=admin_headers)
        assert response.status_code == 200
        assert "admin" in response.json()["message"]

    def test_set_admin_role_not_found(self, client, admin_headers):
        """Test setting admin role for non-existent user."""
        response = client.post("/admin/competitor/NOTEXIST/set-admin", headers=admin_headers)
        assert response.status_code == 404

    def test_remove_admin_role(self, client, admin_headers):
        """Test removing admin role."""
        signup_user(client, "W1RADM")
        client.post("/logout")

        client.post("/admin/competitor/W1RADM/set-admin", headers=admin_headers)
        response = client.post("/admin/competitor/W1RADM/remove-admin", headers=admin_headers)
        assert response.status_code == 200
        assert "no longer an admin" in response.json()["message"]

    def test_remove_admin_role_not_found(self, client, admin_headers):
        """Test removing admin role for non-existent user."""
        response = client.post("/admin/competitor/NOTEXIST/remove-admin", headers=admin_headers)
        assert response.status_code == 404

    def test_set_referee_role(self, client, admin_headers):
        """Test setting referee role."""
        signup_user(client, "W1REF")
        client.post("/logout")

        response = client.post("/admin/competitor/W1REF/set-referee", headers=admin_headers)
        assert response.status_code == 200
        assert "referee" in response.json()["message"]

    def test_set_referee_role_not_found(self, client, admin_headers):
        """Test setting referee role for non-existent user."""
        response = client.post("/admin/competitor/NOTEXIST/set-referee", headers=admin_headers)
        assert response.status_code == 404

    def test_remove_referee_role(self, client, admin_headers):
        """Test removing referee role."""
        signup_user(client, "W1RREF")
        client.post("/logout")

        client.post("/admin/competitor/W1RREF/set-referee", headers=admin_headers)
        response = client.post("/admin/competitor/W1RREF/remove-referee", headers=admin_headers)
        assert response.status_code == 200
        assert "no longer a referee" in response.json()["message"]

    def test_remove_referee_role_not_found(self, client, admin_headers):
        """Test removing referee role for non-existent user."""
        response = client.post("/admin/competitor/NOTEXIST/remove-referee", headers=admin_headers)
        assert response.status_code == 404

    def test_assign_referee_to_sport(self, client, admin_headers):
        """Test assigning a referee to a sport."""
        # Create sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create referee
        signup_user(client, "W1ARF")
        client.post("/logout")
        client.post("/admin/competitor/W1ARF/set-referee", headers=admin_headers)

        # Assign to sport
        response = client.post("/admin/competitor/W1ARF/assign-sport/1", headers=admin_headers)
        assert response.status_code == 200
        assert "assigned" in response.json()["message"]

    def test_assign_referee_not_found(self, client, admin_headers):
        """Test assigning non-existent user."""
        response = client.post("/admin/competitor/NOTEXIST/assign-sport/1", headers=admin_headers)
        assert response.status_code == 404

    def test_assign_non_referee_to_sport(self, client, admin_headers):
        """Test assigning non-referee to sport."""
        signup_user(client, "W1NREF")
        client.post("/logout")

        # Create sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        response = client.post("/admin/competitor/W1NREF/assign-sport/1", headers=admin_headers)
        assert response.status_code == 400
        assert "not a referee" in response.json()["detail"]

    def test_assign_referee_sport_not_found(self, client, admin_headers):
        """Test assigning referee to non-existent sport."""
        signup_user(client, "W1ASNF")
        client.post("/logout")
        client.post("/admin/competitor/W1ASNF/set-referee", headers=admin_headers)

        response = client.post("/admin/competitor/W1ASNF/assign-sport/999", headers=admin_headers)
        assert response.status_code == 404
        assert "Sport not found" in response.json()["detail"]

    def test_assign_referee_already_assigned(self, client, admin_headers):
        """Test assigning referee already assigned to sport."""
        # Create sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create and assign referee
        signup_user(client, "W1DUP")
        client.post("/logout")
        client.post("/admin/competitor/W1DUP/set-referee", headers=admin_headers)
        client.post("/admin/competitor/W1DUP/assign-sport/1", headers=admin_headers)

        # Try to assign again
        response = client.post("/admin/competitor/W1DUP/assign-sport/1", headers=admin_headers)
        assert response.status_code == 400
        assert "already assigned" in response.json()["detail"]

    def test_remove_referee_from_sport(self, client, admin_headers):
        """Test removing referee from sport."""
        # Create sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create and assign referee
        signup_user(client, "W1REMR")
        client.post("/logout")
        client.post("/admin/competitor/W1REMR/set-referee", headers=admin_headers)
        client.post("/admin/competitor/W1REMR/assign-sport/1", headers=admin_headers)

        # Remove
        response = client.delete("/admin/competitor/W1REMR/assign-sport/1", headers=admin_headers)
        assert response.status_code == 200
        assert "removed" in response.json()["message"]

    def test_remove_referee_assignment_not_found(self, client, admin_headers):
        """Test removing non-existent assignment."""
        response = client.delete("/admin/competitor/NOTEXIST/assign-sport/1", headers=admin_headers)
        assert response.status_code == 404


class TestRefereeAccess:
    """Test referee access to sport/match management."""

    @pytest.fixture
    def referee_client(self, client, admin_headers):
        """Create a referee user and sport setup."""
        # Create olympiad and sport
        resp = client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        assert resp.status_code == 200, f"Create olympiad failed: {resp.text}"

        resp = client.post("/admin/olympiad/1/activate", headers=admin_headers)
        assert resp.status_code == 200, f"Activate olympiad failed: {resp.text}"

        resp = client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        assert resp.status_code == 200, f"Create sport failed: {resp.text}"

        resp = client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        assert resp.status_code == 200, f"Create match failed: {resp.text}"

        # Create referee and assign to sport
        resp = signup_user(client, "W1REF")
        assert resp.status_code in [200, 303], f"Signup failed: {resp.text}"

        resp = client.post("/logout")
        assert resp.status_code == 200, f"Logout failed: {resp.text}"

        resp = client.post("/admin/competitor/W1REF/set-referee", headers=admin_headers)
        assert resp.status_code == 200, f"Set referee failed: {resp.text}"

        resp = client.post("/admin/competitor/W1REF/assign-sport/1", headers=admin_headers)
        assert resp.status_code == 200, f"Assign sport failed: {resp.text}"

        # Login as referee
        resp = client.post("/login", json={"callsign": "W1REF", "password": "password123"})
        assert resp.status_code == 200, f"Login failed: {resp.text}"

        return client

    def test_referee_can_view_sport_matches(self, referee_client):
        """Test referee can view matches for assigned sport."""
        response = referee_client.get("/admin/sport/1/matches")
        assert response.status_code == 200
        assert "DX Challenge" in response.text

    def test_referee_can_update_sport(self, referee_client):
        """Test referee can update assigned sport."""
        response = referee_client.put("/admin/sport/1", json={
            "name": "DX Challenge Updated",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        })
        assert response.status_code == 200

    def test_referee_can_create_match(self, referee_client):
        """Test referee can create match for assigned sport."""
        response = referee_client.post("/admin/sport/1/match", json={
            "start_date": "2026-02-01T00:00:00",
            "end_date": "2026-02-28T23:59:59",
            "target_value": "AF"
        })
        assert response.status_code == 200

    def test_referee_can_update_match(self, referee_client):
        """Test referee can update match in assigned sport."""
        response = referee_client.put("/admin/match/1", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-06-30T23:59:59",
            "target_value": "EU"
        })
        assert response.status_code == 200

    def test_referee_can_get_match(self, referee_client):
        """Test referee can get match in assigned sport."""
        response = referee_client.get("/admin/match/1")
        assert response.status_code == 200

    def test_referee_can_delete_match(self, referee_client, admin_headers):
        """Test referee can delete match in assigned sport."""
        # Create another match to delete
        referee_client.post("/admin/sport/1/match", json={
            "start_date": "2026-03-01T00:00:00",
            "end_date": "2026-03-31T23:59:59",
            "target_value": "SA"
        })
        response = referee_client.delete("/admin/match/2")
        assert response.status_code == 200

    def test_referee_can_disqualify(self, referee_client, admin_headers):
        """Test referee can disqualify competitors."""
        signup_user(referee_client, "W1DQ")
        referee_client.post("/logout")
        # Login as referee again
        referee_client.post("/login", json={"callsign": "W1REF", "password": "password123"})

        response = referee_client.post("/admin/competitor/W1DQ/disqualify")
        assert response.status_code == 200

    def test_referee_cannot_access_unassigned_sport(self, referee_client, admin_headers):
        """Test referee cannot access sport they are not assigned to."""
        # Create second sport (olympiad already set up by referee_client fixture)
        referee_client.post("/admin/olympiad/1/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)

        # referee_client is already logged in as W1REF who is assigned to sport 1
        # Try to access sport 2 (not assigned)
        response = referee_client.get("/admin/sport/2/matches")
        assert response.status_code == 403

    def test_admin_via_session_can_access_sport(self, client, admin_headers):
        """Test admin via session (not header) can access sport endpoints."""
        # Create olympiad and sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create admin user, promote to admin via API, then login
        signup_user(client, "W1ADM")
        client.post("/logout")
        client.post("/admin/competitor/W1ADM/set-admin", headers=admin_headers)
        client.post("/login", json={"callsign": "W1ADM", "password": "password123"})

        # Access sport without admin header (just session cookie)
        response = client.get("/admin/sport/1/matches")
        assert response.status_code == 200

    def test_admin_via_session_can_disqualify(self, client, admin_headers):
        """Test admin via session (not header) can disqualify."""
        # Create active olympiad
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create competitor to disqualify and enter them in sport
        signup_user(client, "W1DQ")
        client.post("/sport/1/enter")
        client.post("/logout")

        # Create user, promote to admin via API, then login
        signup_user(client, "W1ADM")
        client.post("/logout")
        client.post("/admin/competitor/W1ADM/set-admin", headers=admin_headers)
        client.post("/login", json={"callsign": "W1ADM", "password": "password123"})

        # Disqualify without admin header (just session cookie) - explicitly no headers
        response = client.post("/admin/competitor/W1DQ/disqualify", headers={})
        assert response.status_code == 200

    def test_admin_competitors_page_with_non_referee(self, client, admin_headers):
        """Test admin competitors page shows non-referee competitors with empty assignments."""
        # Create non-referee competitor
        signup_user(client, "W1NON")
        client.post("/logout")

        # Get competitors page as admin
        response = client.get("/admin/competitors", headers=admin_headers)
        assert response.status_code == 200
        assert "W1NON" in response.text
        assert "Competitor" in response.text  # Role should show as Competitor

    def test_verify_admin_or_referee_with_referee_session(self, client, admin_headers):
        """Test verify_admin_or_referee passes for referee via session."""
        # Create active olympiad
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create referee
        signup_user(client, "W1RFC")
        client.post("/logout")
        client.post("/admin/competitor/W1RFC/set-referee", headers=admin_headers)

        # Create user to disqualify
        signup_user(client, "W1DQC")
        client.post("/logout")

        # Login as referee
        client.post("/login", json={"callsign": "W1RFC", "password": "password123"})

        # Try disqualify - this should use referee session (not admin header)
        response = client.post("/admin/competitor/W1DQC/disqualify", headers={})
        assert response.status_code == 200

    def test_referee_update_nonexistent_match(self, referee_client):
        """Test referee trying to update non-existent match."""
        response = referee_client.put("/admin/match/999", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-06-30T23:59:59",
            "target_value": "EU"
        })
        assert response.status_code == 404

    def test_referee_delete_nonexistent_match(self, referee_client):
        """Test referee trying to delete non-existent match."""
        response = referee_client.delete("/admin/match/999")
        assert response.status_code == 404

    def test_admin_competitors_page_shows_referee_assignments(self, client, admin_headers):
        """Test admin competitors page shows referee sport assignments."""
        # Create olympiad and sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Create referee and assign to sport
        signup_user(client, "W1RFB")
        client.post("/logout")
        client.post("/admin/competitor/W1RFB/set-referee", headers=admin_headers)
        client.post("/admin/competitor/W1RFB/assign-sport/1", headers=admin_headers)

        # Get competitors page as admin
        response = client.get("/admin/competitors", headers=admin_headers)
        assert response.status_code == 200
        assert "W1RFB" in response.text
        assert "Referee" in response.text
        assert "DX Challenge" in response.text

    def test_sport_competitors_page(self, referee_client, admin_headers):
        """Test viewing competitors for a sport."""
        # Create a competitor and enter them in the sport
        signup_user(referee_client, "W1COMP")
        referee_client.post("/sport/1/enter")
        referee_client.post("/logout")

        # Login as referee and view competitors
        referee_client.post("/login", json={"callsign": "W1REF", "password": "password123"})
        response = referee_client.get("/admin/sport/1/competitors")
        assert response.status_code == 200
        assert "W1COMP" in response.text
        assert "Competitors in DX Challenge" in response.text

    def test_sport_competitors_page_empty(self, referee_client):
        """Test viewing competitors for a sport with no entries."""
        response = referee_client.get("/admin/sport/1/competitors")
        assert response.status_code == 200
        assert "No competitors have entered this sport yet" in response.text

    def test_disqualify_from_sport(self, referee_client, admin_headers):
        """Test disqualifying a competitor from a specific sport."""
        # Create a competitor and enter them in the sport
        signup_user(referee_client, "W1DQS")
        referee_client.post("/sport/1/enter")
        referee_client.post("/logout")

        # Login as referee and disqualify
        referee_client.post("/login", json={"callsign": "W1REF", "password": "password123"})
        response = referee_client.post("/admin/sport/1/competitor/W1DQS/disqualify")
        assert response.status_code == 200
        assert "disqualified" in response.json()["message"]

        # Verify competitor is removed from sport
        response = referee_client.get("/admin/sport/1/competitors")
        assert "W1DQS" not in response.text

    def test_disqualify_from_sport_not_found(self, referee_client):
        """Test disqualifying a competitor not in the sport."""
        response = referee_client.post("/admin/sport/1/competitor/NOTHERE/disqualify")
        assert response.status_code == 404
        assert "not found in this sport" in response.json()["detail"]

    def test_sport_competitors_page_not_found(self, client, admin_headers):
        """Test viewing competitors for a non-existent sport."""
        response = client.get("/admin/sport/999/competitors", headers=admin_headers)
        assert response.status_code == 404

    def test_referee_cannot_view_unassigned_sport_competitors(self, referee_client, admin_headers):
        """Test referee cannot view competitors for a sport they're not assigned to."""
        # Create second sport
        referee_client.post("/admin/olympiad/1/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)

        # Try to view competitors for sport 2 (not assigned)
        response = referee_client.get("/admin/sport/2/competitors")
        assert response.status_code == 403


class TestVerifyAdminOrReferee:
    """Unit tests for verify_admin_or_referee function."""

    def test_verify_with_referee_session_directly(self):
        """Test verify_admin_or_referee with a referee session cookie."""
        from auth import register_user

        # Create referee user
        register_user("W1UNIT", "password123", qrz_api_key_encrypted="test")

        # Make user a referee
        with get_db() as conn:
            conn.execute("UPDATE competitors SET is_referee = 1 WHERE callsign = ?", ("W1UNIT",))

        # Create a session
        session_id = create_session("W1UNIT")

        # Create a mock request with the session cookie
        scope = {
            "type": "http",
            "headers": [],
            "query_string": b"",
            "method": "POST",
            "path": "/admin/competitor/TEST/disqualify"
        }
        request = Request(scope)
        # Manually set cookies on the request (simulating cookie parsing)
        request._cookies = {SESSION_COOKIE_NAME: session_id}

        # Call the function directly
        result = verify_admin_or_referee(request)
        assert result is True

    def test_verify_rejects_regular_user(self):
        """Test verify_admin_or_referee rejects a user who is neither admin nor referee."""
        from auth import register_user
        from fastapi import HTTPException

        # Create regular user (not admin, not referee)
        register_user("W1REG", "password123", qrz_api_key_encrypted="test")

        # Create a session
        session_id = create_session("W1REG")

        # Create a mock request with the session cookie
        scope = {
            "type": "http",
            "headers": [],
            "query_string": b"",
            "method": "POST",
            "path": "/admin/competitor/TEST/disqualify"
        }
        request = Request(scope)
        request._cookies = {SESSION_COOKIE_NAME: session_id}

        # Call the function - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_or_referee(request)
        assert exc_info.value.status_code == 403


class TestRecordsEndpoint:
    """Test records endpoint."""

    def test_get_records_requires_auth(self, client):
        """Test records page requires authentication."""
        response = client.get("/records", follow_redirects=False)
        assert response.status_code == 401

    def test_get_records_empty(self, client):
        """Test getting records page when none exist."""
        signup_user(client, "W1REC")
        response = client.get("/records")
        assert response.status_code == 200
        assert "World Records" in response.text
        assert "No world records have been set yet" in response.text

    def test_get_records_with_data(self, client, admin_headers):
        """Test getting records page with actual records."""
        # Set up olympiad, sport, match
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # Register competitor and add a QSO that would set a record
        signup_user(client, "W1TEST")

        # Insert a QSO directly into DB to set a record
        from database import get_db
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc,
                    band, mode, tx_power_w, my_grid, dx_grid, my_dxcc, dx_dxcc,
                    distance_km, is_confirmed)
                VALUES ('W1TEST', 'DL1ABC', '2026-01-15 12:00:00',
                    '20M', 'SSB', 5.0, 'EM12', 'JN58', 291, 230,
                    8500.0, 1)
            """)
            qso_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # Insert a world record
            conn.execute("""
                INSERT INTO records (record_type, value, qso_id, achieved_at)
                VALUES ('longest_distance', 8500.0, ?, '2026-01-15 12:00:00')
            """, (qso_id,))

        response = client.get("/records")
        assert response.status_code == 200
        assert "World Records" in response.text
        assert "W1TEST" in response.text


class TestSportEntry:
    """Test sport opt-in/opt-out endpoints."""

    @pytest.fixture
    def setup_sport(self, client, admin_headers):
        """Set up a sport for testing."""
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        return 1  # sport_id

    def test_enter_sport_requires_auth(self, client, setup_sport):
        """Test entering sport requires authentication."""
        response = client.post("/sport/1/enter")
        assert response.status_code == 401

    def test_enter_sport_success(self, client, admin_headers, setup_sport):
        """Test successfully entering a sport."""
        signup_user(client, "W1ENT")
        response = client.post("/sport/1/enter")
        assert response.status_code == 200
        assert response.json()["message"] == "Entered sport successfully"

    def test_enter_sport_already_entered(self, client, admin_headers, setup_sport):
        """Test entering sport when already entered."""
        signup_user(client, "W1DBL")
        client.post("/sport/1/enter")
        # Try to enter again
        response = client.post("/sport/1/enter")
        assert response.status_code == 200
        assert response.json()["message"] == "Already entered"

    def test_enter_sport_not_found(self, client, admin_headers, setup_sport):
        """Test entering non-existent sport."""
        signup_user(client, "W1NOSP")
        response = client.post("/sport/999/enter")
        assert response.status_code == 404

    def test_leave_sport_success(self, client, admin_headers, setup_sport):
        """Test successfully leaving a sport."""
        signup_user(client, "W1LVE")
        client.post("/sport/1/enter")
        response = client.post("/sport/1/leave")
        assert response.status_code == 200
        assert response.json()["message"] == "Left sport"

    def test_leave_sport_requires_auth(self, client, setup_sport):
        """Test leaving sport requires authentication."""
        response = client.post("/sport/1/leave")
        assert response.status_code == 401

    def test_sport_page_shows_entry_status(self, client, admin_headers, setup_sport):
        """Test sport page shows entry status and participant count."""
        signup_user(client, "W1VIEW")
        # Check before entering
        response = client.get("/olympiad/sport/1")
        assert response.status_code == 200
        assert "Participants:" in response.text
        assert "Participate" in response.text

        # Enter the sport
        client.post("/sport/1/enter")
        response = client.get("/olympiad/sport/1")
        assert "You are entered" in response.text
        assert "Leave Sport" in response.text

    def test_enter_sport_recomputes_medals(self, client, admin_headers):
        """Test entering sport triggers medal recomputation."""
        from database import get_db
        # Create olympiad, sport, and match
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # Create user and add a matching QSO
        signup_user(client, "W1MED")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8500.0, 1700.0)
            """, ("W1MED", "DL1ABC", "2026-06-15T12:00:00", 5.0, "EM12", "JN58", 230))

        # Enter sport - should trigger medal recomputation
        client.post("/sport/1/enter")

        # Check medal was created
        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM medals WHERE callsign = 'W1MED'")
            medal = cursor.fetchone()
        assert medal is not None
        assert medal["qso_race_medal"] == "gold"


class TestMultipleSports:
    """Test single QSO matching multiple Sports."""

    @pytest.fixture
    def setup_multi_sport(self, client, admin_headers):
        """Set up Olympiad with multiple Sports."""
        # Create Olympiad
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)

        # Sport 1: DX Challenge (continent)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

        # Sport 2: POTA Championship (park)
        client.post("/admin/olympiad/1/sport", json={
            "name": "POTA Championship",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)

        # Create matches
        # DX Challenge January - target EU
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # POTA Week 1 - target K-0001
        client.post("/admin/sport/2/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-07T23:59:59",
            "target_value": "K-0001"
        }, headers=admin_headers)

    def test_sports_are_independent(self, client, admin_headers, setup_multi_sport):
        """Test that Sports have independent leaderboards."""
        sports = client.get("/olympiad/sports").json()

        assert len(sports) == 2
        assert sports[0]["name"] == "DX Challenge"
        assert sports[1]["name"] == "POTA Championship"

        # Each Sport has its own matches
        dx_matches = client.get("/olympiad/sport/1/matches").json()
        pota_matches = client.get("/olympiad/sport/2/matches").json()

        assert len(dx_matches) == 1
        assert dx_matches[0]["target_value"] == "EU"

        assert len(pota_matches) == 1
        assert pota_matches[0]["target_value"] == "K-0001"


class TestLandingPage:
    """Test landing page endpoint."""

    @pytest.fixture
    def logged_in_client(self, client):
        """Create a logged-in client."""
        client.post("/signup", json={
            "callsign": "W1LAND",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        return client

    def test_landing_page_redirects_to_signup_when_not_logged_in(self, client):
        """Test landing page redirects to signup for unauthenticated users."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/signup"

    def test_landing_page_no_olympiad(self, logged_in_client):
        """Test landing page with no active olympiad."""
        response = logged_in_client.get("/")
        assert response.status_code == 200
        assert "Ham Radio Olympics" in response.text

    def test_landing_page_with_olympiad(self, logged_in_client, admin_headers):
        """Test landing page with active olympiad."""
        logged_in_client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)

        response = logged_in_client.get("/")
        assert response.status_code == 200
        assert "2026 Olympics" in response.text


class TestSportEndpointsPublic:
    """Test public sport endpoints."""

    @pytest.fixture
    def setup_sport(self, client, admin_headers):
        """Create olympiad with sport and match."""
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

    def test_get_sport_details(self, client, admin_headers, setup_sport):
        """Test getting sport details page."""
        response = client.get("/olympiad/sport/1")
        assert response.status_code == 200
        assert "DX Challenge" in response.text
        assert "Matches" in response.text
        assert "EU" in response.text

    def test_get_sport_not_found(self, client):
        """Test getting non-existent sport."""
        response = client.get("/olympiad/sport/999")
        assert response.status_code == 404

    def test_get_match_details(self, client, admin_headers, setup_sport):
        """Test getting match details page with readable dates."""
        response = client.get("/olympiad/sport/1/match/1")
        assert response.status_code == 200
        assert "EU" in response.text
        assert "DX Challenge" in response.text
        assert "Target:" in response.text
        # Dates should be in readable format, not ISO
        assert "Jan 1, 2026" in response.text
        assert "Jan 31, 2026" in response.text

    def test_get_match_not_found(self, client, admin_headers, setup_sport):
        """Test getting non-existent match."""
        response = client.get("/olympiad/sport/1/match/999")
        assert response.status_code == 404

    def test_get_olympiad_sports_no_active(self, client):
        """Test getting sports when no active olympiad."""
        response = client.get("/olympiad/sports")
        assert response.status_code == 404
        assert "No active Olympiad" in response.json()["detail"]


class TestSyncEndpoint:
    """Test sync endpoint."""

    def test_sync_single_competitor(self, client):
        """Test syncing a single competitor."""
        # Register first
        signup_user(client, "W1TEST")
        response = client.post("/sync?callsign=W1TEST")
        assert response.status_code == 200

    def test_sync_all_competitors(self, client):
        """Test syncing all competitors."""
        response = client.post("/sync")
        assert response.status_code == 200

    def test_sync_page_single_competitor(self, client):
        """Test sync page for single competitor."""
        signup_user(client, "W1SYNC")
        response = client.get("/sync?callsign=W1SYNC")
        assert response.status_code == 200
        assert "Sync Results" in response.text
        assert "W1SYNC" in response.text

    def test_sync_page_all_competitors(self, client):
        """Test sync page for all competitors."""
        response = client.get("/sync")
        assert response.status_code == 200
        assert "Sync Results" in response.text
        assert "Syncing all competitors" in response.text

    def test_sync_page_nonexistent_competitor(self, client):
        """Test sync page for nonexistent competitor shows error."""
        response = client.get("/sync?callsign=NOTEXIST")
        assert response.status_code == 200
        assert "Error" in response.text
        assert "not found" in response.text


class TestAdminHTMLEndpoints:
    """Test admin HTML page endpoints."""

    def test_admin_olympiads_page(self, client, admin_headers):
        """Test admin olympiads HTML page."""
        response = client.get("/admin/olympiads", headers=admin_headers)
        assert response.status_code == 200
        assert "Olympiads" in response.text

    def test_admin_sports_page(self, client, admin_headers):
        """Test admin sports HTML page."""
        # Create olympiad first
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        response = client.get("/admin/olympiad/1/sports", headers=admin_headers)
        assert response.status_code == 200
        assert "Sports" in response.text

    def test_admin_sports_page_olympiad_not_found(self, client, admin_headers):
        """Test admin sports page with nonexistent olympiad."""
        response = client.get("/admin/olympiad/999/sports", headers=admin_headers)
        assert response.status_code == 404

    def test_admin_matches_page(self, client, admin_headers):
        """Test admin matches HTML page."""
        # Create olympiad and sport first
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        response = client.get("/admin/sport/1/matches", headers=admin_headers)
        assert response.status_code == 200
        assert "Matches" in response.text

    def test_admin_matches_page_with_country_target_type(self, client, admin_headers):
        """Test admin matches page shows country dropdown for country target_type."""
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DXCC",
            "target_type": "country",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        response = client.get("/admin/sport/1/matches", headers=admin_headers)
        assert response.status_code == 200
        # Should have dropdown with country options
        assert "<select" in response.text
        assert "United States" in response.text
        assert "291" in response.text  # USA DXCC code

    def test_admin_matches_page_sport_not_found(self, client, admin_headers):
        """Test admin matches page with nonexistent sport."""
        response = client.get("/admin/sport/999/matches", headers=admin_headers)
        assert response.status_code == 404

    def test_admin_competitors_page(self, client, admin_headers):
        """Test admin competitors HTML page."""
        response = client.get("/admin/competitors", headers=admin_headers)
        assert response.status_code == 200
        assert "Competitors" in response.text

    def test_admin_olympiads_page_has_edit_buttons(self, client, admin_headers):
        """Test admin olympiads page has Edit buttons."""
        # Create olympiad
        client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        response = client.get("/admin/olympiads", headers=admin_headers)
        assert response.status_code == 200
        assert "Edit" in response.text
        assert "editOlympiad" in response.text

    def test_admin_sports_page_has_edit_buttons(self, client, admin_headers):
        """Test admin sports page has Edit buttons."""
        # Create olympiad and sport
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        response = client.get("/admin/olympiad/1/sports", headers=admin_headers)
        assert response.status_code == 200
        assert "Edit" in response.text
        assert "editSport" in response.text

    def test_admin_matches_page_has_edit_buttons(self, client, admin_headers):
        """Test admin matches page has Edit buttons."""
        # Create olympiad, sport, and match
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        response = client.get("/admin/sport/1/matches", headers=admin_headers)
        assert response.status_code == 200
        assert "Edit" in response.text
        assert "editMatch" in response.text


class TestAdminSportCRUD:
    """Test admin sport CRUD operations."""

    @pytest.fixture
    def setup_olympiad(self, client, admin_headers):
        """Create an olympiad."""
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)

    def test_get_sport_not_found(self, client, admin_headers):
        """Test getting nonexistent sport."""
        response = client.get("/admin/sport/999", headers=admin_headers)
        assert response.status_code == 404

    def test_update_sport(self, client, admin_headers, setup_olympiad):
        """Test updating a sport."""
        # Create sport
        client.post("/admin/olympiad/1/sport", json={
            "name": "Original",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        # Update it
        response = client.put("/admin/sport/1", json={
            "name": "Updated",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)
        assert response.status_code == 200
        # Verify
        sport = client.get("/admin/sport/1", headers=admin_headers).json()
        assert sport["name"] == "Updated"
        assert sport["target_type"] == "park"

    def test_delete_sport(self, client, admin_headers, setup_olympiad):
        """Test deleting a sport."""
        # Create sport
        client.post("/admin/olympiad/1/sport", json={
            "name": "ToDelete",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        # Delete it
        response = client.delete("/admin/sport/1", headers=admin_headers)
        assert response.status_code == 200
        # Verify deleted
        get_resp = client.get("/admin/sport/1", headers=admin_headers)
        assert get_resp.status_code == 404


class TestAdminMatchCRUD:
    """Test admin match CRUD operations."""

    @pytest.fixture
    def setup_sport(self, client, admin_headers):
        """Create olympiad and sport."""
        client.post("/admin/olympiad", json={
            "name": "Test",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)

    def test_get_match_not_found(self, client, admin_headers):
        """Test getting nonexistent match."""
        response = client.get("/admin/match/999", headers=admin_headers)
        assert response.status_code == 404

    def test_update_match(self, client, admin_headers, setup_sport):
        """Test updating a match."""
        # Create match
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        # Update it
        response = client.put("/admin/match/1", json={
            "start_date": "2026-02-01T00:00:00",
            "end_date": "2026-02-28T23:59:59",
            "target_value": "AS"
        }, headers=admin_headers)
        assert response.status_code == 200
        # Verify
        match = client.get("/admin/match/1", headers=admin_headers).json()
        assert match["target_value"] == "AS"

    def test_delete_match(self, client, admin_headers, setup_sport):
        """Test deleting a match."""
        # Create match
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        # Delete it
        response = client.delete("/admin/match/1", headers=admin_headers)
        assert response.status_code == 200
        # Verify deleted
        get_resp = client.get("/admin/match/1", headers=admin_headers)
        assert get_resp.status_code == 404


class TestDateFormatting:
    """Test date formatting filters."""

    def test_format_date_empty(self):
        """Test format_date with empty value."""
        from main import format_date
        assert format_date("") == "-"
        assert format_date(None) == "-"

    def test_format_date_invalid(self):
        """Test format_date with invalid date string."""
        from main import format_date
        assert format_date("not-a-date") == "not-a-date"
        assert format_date("invalid") == "invalid"


class TestLifespan:
    """Test application lifespan startup and shutdown."""

    def test_lifespan_startup(self):
        """Test that lifespan startup initializes the database (covers main.py:25-26)."""
        # Use context manager to trigger lifespan events
        with TestClient(app) as client:
            # If we get here, the lifespan startup completed successfully
            response = client.get("/")
            assert response.status_code == 200


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_signup_page(self, client):
        """Test signup page renders."""
        response = client.get("/signup")
        assert response.status_code == 200
        assert "Create Account" in response.text

    def test_signup_page_redirects_when_logged_in(self, client):
        """Test signup page redirects if already logged in."""
        # Sign up and stay logged in
        client.post("/signup", json={
            "callsign": "W1RED",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        # Now try to access signup again
        response = client.get("/signup", follow_redirects=False)
        assert response.status_code == 303
        assert "/dashboard" in response.headers["location"]

    def test_login_page_redirects_when_logged_in(self, client):
        """Test login page redirects if already logged in."""
        # Sign up and stay logged in
        client.post("/signup", json={
            "callsign": "W1LGR",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        # Now try to access login
        response = client.get("/login", follow_redirects=False)
        assert response.status_code == 303
        assert "/dashboard" in response.headers["location"]

    def test_signup_success(self, client):
        """Test successful signup."""
        response = client.post("/signup", json={
            "callsign": "W1NEW",
            "password": "password123",
            "email": "test@example.com",
            "qrz_api_key": "test-api-key"
        }, follow_redirects=False)
        assert response.status_code == 303
        assert "hro_session" in response.cookies

    def test_signup_duplicate(self, client):
        """Test signup with duplicate callsign."""
        client.post("/signup", json={
            "callsign": "W1DUP",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        response = client.post("/signup", json={
            "callsign": "W1DUP",
            "password": "password456",
            "qrz_api_key": "test-api-key"
        })
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_signup_short_password(self, client):
        """Test signup with short password."""
        response = client.post("/signup", json={
            "callsign": "W1SHT",
            "password": "short",
            "qrz_api_key": "test-api-key"
        })
        assert response.status_code == 422

    def test_signup_empty_qrz_api_key(self, client):
        """Test signup with empty QRZ API key and no LoTW credentials."""
        response = client.post("/signup", json={
            "callsign": "W1NOK",
            "password": "password123",
            "qrz_api_key": "   "  # Whitespace only
        })
        assert response.status_code == 400
        assert "QRZ API key and/or LoTW" in response.json()["detail"]

    def test_login_page(self, client):
        """Test login page renders."""
        response = client.get("/login")
        assert response.status_code == 200
        assert "Log In" in response.text

    def test_login_success(self, client):
        """Test successful login."""
        # First signup
        client.post("/signup", json={
            "callsign": "W1LOG",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        # Clear cookies
        client.cookies.clear()
        # Then login
        response = client.post("/login", json={
            "callsign": "W1LOG",
            "password": "password123"
        }, follow_redirects=False)
        assert response.status_code == 303
        assert "hro_session" in response.cookies

    def test_login_wrong_password(self, client):
        """Test login with wrong password."""
        client.post("/signup", json={
            "callsign": "W1WRO",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        client.cookies.clear()
        response = client.post("/login", json={
            "callsign": "W1WRO",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    def test_login_disabled_account(self, client, admin_headers):
        """Test login with disabled account."""
        client.post("/signup", json={
            "callsign": "W1DIS",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        client.cookies.clear()
        # Disable the account
        client.post("/admin/competitor/W1DIS/disable", headers=admin_headers)
        # Try to login
        response = client.post("/login", json={
            "callsign": "W1DIS",
            "password": "password123"
        })
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]

    def test_logout(self, client):
        """Test logout."""
        # Signup to get session
        client.post("/signup", json={
            "callsign": "W1LGO",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        # Logout
        response = client.post("/logout", follow_redirects=False)
        assert response.status_code == 303
        # Cookie should be deleted
        assert response.cookies.get("hro_session") is None or response.cookies.get("hro_session") == ""


class TestUserDashboard:
    """Test user dashboard endpoints."""

    @pytest.fixture
    def logged_in_client(self, client):
        """Create a logged-in client."""
        client.post("/signup", json={
            "callsign": "W1DASH",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        return client

    def test_dashboard_requires_auth(self, client):
        """Test dashboard requires authentication."""
        response = client.get("/dashboard")
        assert response.status_code == 401

    def test_dashboard_renders(self, logged_in_client):
        """Test dashboard renders for logged-in user."""
        response = logged_in_client.get("/dashboard")
        assert response.status_code == 200
        assert "Welcome, W1DASH" in response.text
        assert "Gold Medals" in response.text

    def test_dashboard_shows_country_name_for_medals(self, logged_in_client, admin_headers):
        """Test dashboard shows country name for country-target medals."""
        from database import get_db
        # Create country-target sport and match
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "Country DX",
            "target_type": "country",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        logged_in_client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "291"
        }, headers=admin_headers)

        # Enter sport and add medal directly
        logged_in_client.post("/sport/1/enter")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO medals (match_id, callsign, role, qso_race_medal, total_points)
                VALUES (1, 'W1DASH', 'combined', 'gold', 3)
            """)

        response = logged_in_client.get("/dashboard")
        assert response.status_code == 200
        assert "United States" in response.text
        assert "291" in response.text

    def test_dashboard_shows_target_for_non_country_medals(self, logged_in_client, admin_headers):
        """Test dashboard shows raw target for non-country targets."""
        from database import get_db
        # Create continent-target sport and match
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "Continent DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        logged_in_client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # Enter sport and add medal directly
        logged_in_client.post("/sport/1/enter")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO medals (match_id, callsign, role, qso_race_medal, total_points)
                VALUES (1, 'W1DASH', 'combined', 'gold', 3)
            """)

        response = logged_in_client.get("/dashboard")
        assert response.status_code == 200
        assert "EU" in response.text

    def test_competitor_page_shows_country_name_for_medals(self, logged_in_client, admin_headers):
        """Test competitor page shows country name for country-target medals."""
        from database import get_db
        # Create country-target sport and match
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "Country DX",
            "target_type": "country",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        logged_in_client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "291"
        }, headers=admin_headers)

        # Enter sport and add medal directly
        logged_in_client.post("/sport/1/enter")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO medals (match_id, callsign, role, qso_race_medal, total_points)
                VALUES (1, 'W1DASH', 'combined', 'gold', 3)
            """)

        response = logged_in_client.get("/competitor/W1DASH")
        assert response.status_code == 200
        assert "United States" in response.text
        assert "291" in response.text

    def test_competitor_page_shows_target_for_non_country_medals(self, logged_in_client, admin_headers):
        """Test competitor page shows raw target for non-country targets."""
        from database import get_db
        # Create continent-target sport and match
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "Continent DX",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        logged_in_client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-12-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        # Enter sport and add medal directly
        logged_in_client.post("/sport/1/enter")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO medals (match_id, callsign, role, qso_race_medal, total_points)
                VALUES (1, 'W1DASH', 'combined', 'gold', 3)
            """)

        response = logged_in_client.get("/competitor/W1DASH")
        assert response.status_code == 200
        assert "EU" in response.text

    def test_settings_requires_auth(self, client):
        """Test settings requires authentication."""
        response = client.get("/settings")
        assert response.status_code == 401

    def test_settings_renders(self, logged_in_client):
        """Test settings renders for logged-in user."""
        response = logged_in_client.get("/settings")
        assert response.status_code == 200
        assert "Account Settings" in response.text
        assert "W1DASH" in response.text

    def test_nav_shows_logout_when_logged_in(self, logged_in_client):
        """Test nav shows logout link when logged in."""
        response = logged_in_client.get("/dashboard")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text
        assert "Log In" not in response.text

    def test_nav_shows_login_when_logged_out(self, client):
        """Test nav shows login/signup when not logged in."""
        response = client.get("/signup")
        assert response.status_code == 200
        assert "Sign Up" in response.text
        assert "Log In" in response.text
        assert "Log Out" not in response.text

    def test_home_shows_logout_when_logged_in(self, logged_in_client, admin_headers):
        """Test home page shows logout when logged in."""
        # Create active olympiad so Get Started section would appear
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test", "start_date": "2026-01-01",
            "end_date": "2026-12-31", "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        response = logged_in_client.get("/")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text
        assert "Get Started" not in response.text

    def test_records_shows_logout_when_logged_in(self, logged_in_client):
        """Test records page shows logout when logged in."""
        response = logged_in_client.get("/records")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text

    def test_competitor_page_shows_logout_when_logged_in(self, logged_in_client):
        """Test competitor page shows logout when logged in."""
        response = logged_in_client.get("/competitor/W1DASH")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text

    def test_competitor_page_requires_auth_when_logged_out(self, client):
        """Test competitor page requires authentication when logged out."""
        # First create a competitor
        client.post("/signup", json={"callsign": "W1TEST", "password": "password123", "qrz_api_key": "test-api-key"})
        client.post("/logout")  # Log out
        response = client.get("/competitor/W1TEST", follow_redirects=False)
        assert response.status_code == 401

    def test_sport_page_shows_logout_when_logged_in(self, logged_in_client, admin_headers):
        """Test sport page shows logout when logged in."""
        # Create olympiad and sport
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test", "start_date": "2026-01-01",
            "end_date": "2026-12-31", "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "DX", "target_type": "continent",
            "work_enabled": True, "activate_enabled": False, "separate_pools": False
        }, headers=admin_headers)
        response = logged_in_client.get("/olympiad/sport/1")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text

    def test_match_page_shows_logout_when_logged_in(self, logged_in_client, admin_headers):
        """Test match page shows logout when logged in."""
        # Create olympiad, sport, and match
        logged_in_client.post("/admin/olympiad", json={
            "name": "Test", "start_date": "2026-01-01",
            "end_date": "2026-12-31", "qualifying_qsos": 0
        }, headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/activate", headers=admin_headers)
        logged_in_client.post("/admin/olympiad/1/sport", json={
            "name": "DX", "target_type": "continent",
            "work_enabled": True, "activate_enabled": False, "separate_pools": False
        }, headers=admin_headers)
        logged_in_client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00", "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        response = logged_in_client.get("/olympiad/sport/1/match/1")
        assert response.status_code == 200
        assert "Log Out" in response.text
        assert "Sign Up" not in response.text


class TestUserSettings:
    """Test user settings update endpoints."""

    @pytest.fixture
    def logged_in_client(self, client):
        """Create a logged-in client."""
        client.post("/signup", json={
            "callsign": "W1SET",
            "password": "password123",
            "qrz_api_key": "test-api-key"
        })
        return client

    def test_update_email(self, logged_in_client):
        """Test updating email."""
        response = logged_in_client.post("/settings/email",
            data={"email": "newemail@example.com"},
            follow_redirects=False)
        assert response.status_code == 303
        assert "updated=email" in response.headers["location"]

    def test_change_password(self, logged_in_client):
        """Test changing password."""
        response = logged_in_client.post("/settings/password",
            data={
                "current_password": "password123",
                "new_password": "newpassword456"
            },
            follow_redirects=False)
        assert response.status_code == 303
        assert "updated=password" in response.headers["location"]

    def test_change_password_wrong_current(self, logged_in_client):
        """Test changing password with wrong current password."""
        response = logged_in_client.post("/settings/password",
            data={
                "current_password": "wrongpassword",
                "new_password": "newpassword456"
            })
        assert response.status_code == 400

    def test_change_password_too_short(self, logged_in_client):
        """Test changing password to short password."""
        response = logged_in_client.post("/settings/password",
            data={
                "current_password": "password123",
                "new_password": "short"
            })
        assert response.status_code == 400

    def test_update_qrz_key(self, logged_in_client):
        """Test updating QRZ API key."""
        response = logged_in_client.post("/settings/qrz-key",
            data={"qrz_api_key": "my-api-key-123"},
            follow_redirects=False)
        assert response.status_code == 303
        assert "updated=qrz" in response.headers["location"]


class TestBackgroundSync:
    """Test background sync functionality."""

    @pytest.mark.asyncio
    async def test_background_sync_calls_sync_all(self):
        """Test background_sync calls sync_all_competitors after sleep."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from main import background_sync

        with patch('main.sync_all_competitors', new_callable=AsyncMock) as mock_sync:
            with patch('main.SYNC_INTERVAL', 0.01):  # Very short interval for testing
                # Run background_sync briefly then cancel
                task = asyncio.create_task(background_sync())
                await asyncio.sleep(0.05)  # Let it run a few cycles
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Should have been called at least once
            assert mock_sync.call_count >= 1

    @pytest.mark.asyncio
    async def test_background_sync_handles_exceptions(self):
        """Test background_sync continues after exceptions."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from main import background_sync

        call_count = 0

        async def mock_sync_with_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            # Second call succeeds

        with patch('main.sync_all_competitors', side_effect=mock_sync_with_error):
            with patch('main.SYNC_INTERVAL', 0.01):
                task = asyncio.create_task(background_sync())
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Should have continued after the exception
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_lifespan_starts_and_cancels_sync_task(self):
        """Test lifespan properly manages background sync task."""
        import asyncio
        from unittest.mock import patch, AsyncMock, MagicMock
        import main

        with patch('main.init_db'):
            with patch('main.background_sync', new_callable=AsyncMock) as mock_bg:
                # Make background_sync return a coroutine that sleeps forever
                async def long_running():
                    await asyncio.sleep(3600)
                mock_bg.return_value = long_running()

                # Enter lifespan
                async with main.lifespan(MagicMock()):
                    # Task should be created
                    assert main._sync_task is not None

                # After exiting, task should be cancelled
                assert main._sync_task.cancelled() or main._sync_task.done()


class TestFormatTargetDisplay:
    """Test format_target_display helper function."""

    def test_continent_target(self):
        """Test continent target formatting."""
        result = format_target_display("EU", "continent")
        assert result == "Europe (EU)"

    def test_country_target(self):
        """Test country target formatting."""
        result = format_target_display("291", "country")
        assert result == "United States (291)"

    def test_country_target_invalid_value(self):
        """Test country target with non-numeric value."""
        result = format_target_display("invalid", "country")
        assert result == "invalid"

    def test_park_target(self):
        """Test park target just returns the value."""
        result = format_target_display("K-0001", "park")
        assert result == "K-0001"

    def test_grid_target(self):
        """Test grid target just returns the value."""
        result = format_target_display("FN31", "grid")
        assert result == "FN31"

    def test_call_target(self):
        """Test call target just returns the value."""
        result = format_target_display("W1AW", "call")
        assert result == "W1AW"

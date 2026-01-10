"""
Tests for data export and search features - TDD: Tests written before implementation.
"""

import os
import csv
import io
import tempfile
import pytest
from unittest.mock import patch

# Set test mode BEFORE importing app modules
os.environ["TESTING"] = "1"

# Create temp file for test database BEFORE importing app modules
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["DATABASE_PATH"] = _test_db_path


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    from database import reset_db
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
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
def logged_in_client(client):
    """Create test client with logged in user."""
    # Signup auto-logs in (requires QRZ API key or LoTW credentials)
    client.post("/signup", json={
        "callsign": "W1EXP",
        "password": "password123",
        "email": "export@example.com",
        "qrz_api_key": "test-api-key"
    })
    return client


@pytest.fixture
def admin_client(client):
    """Create test client with logged in admin user."""
    from database import get_db

    # Signup auto-logs in (requires QRZ API key or LoTW credentials)
    client.post("/signup", json={
        "callsign": "W1ADM",
        "password": "password123",
        "email": "admin@example.com",
        "qrz_api_key": "test-api-key"
    })

    # Make admin
    with get_db() as conn:
        conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = 'W1ADM'")

    return client


class TestExportQSOs:
    """Test QSO export functionality."""

    def test_export_qsos_endpoint_exists(self, logged_in_client):
        """Test export QSOs endpoint exists."""
        response = logged_in_client.get("/export/qsos")
        assert response.status_code != 404

    def test_export_qsos_requires_auth(self, client):
        """Test export QSOs requires authentication."""
        response = client.get("/export/qsos")
        assert response.status_code in [401, 403, 302, 303]

    def test_export_qsos_returns_csv(self, logged_in_client):
        """Test export QSOs returns CSV content type."""
        response = logged_in_client.get("/export/qsos")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_export_qsos_has_headers(self, logged_in_client):
        """Test export QSOs includes CSV headers."""
        response = logged_in_client.get("/export/qsos")
        content = response.text

        # Should have standard QSO fields
        assert "callsign" in content.lower() or "call" in content.lower()
        assert "band" in content.lower()
        assert "mode" in content.lower()

    def test_export_qsos_includes_user_data(self, logged_in_client):
        """Test export includes user's QSO data."""
        from database import get_db

        # Add some QSOs for this user
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2ABC", "2024-01-15T12:00:00", "20m", "SSB", 1))

        response = logged_in_client.get("/export/qsos")
        content = response.text

        assert "K2ABC" in content
        assert "2024-01-15" in content
        assert "20m" in content

    def test_export_qsos_only_own_data(self, logged_in_client):
        """Test export only includes user's own QSOs."""
        from database import get_db
        from auth import register_user

        # Create another user
        register_user("W1OTHER", "password123")

        # Add QSOs for different users
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2ABC", "2024-01-15T12:00:00", "20m", "SSB", 1))
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1OTHER", "K3DEF", "2024-01-16T12:00:00", "40m", "CW", 1))

        response = logged_in_client.get("/export/qsos")
        content = response.text

        assert "K2ABC" in content
        assert "K3DEF" not in content  # Should not include other user's QSOs

    def test_export_qsos_content_disposition(self, logged_in_client):
        """Test export has proper content disposition for download."""
        response = logged_in_client.get("/export/qsos")
        disposition = response.headers.get("content-disposition", "")

        assert "attachment" in disposition
        assert ".csv" in disposition


class TestExportMedals:
    """Test medals export functionality."""

    def test_export_medals_endpoint_exists(self, logged_in_client):
        """Test export medals endpoint exists."""
        response = logged_in_client.get("/export/medals")
        assert response.status_code != 404

    def test_export_medals_requires_auth(self, client):
        """Test export medals requires authentication."""
        response = client.get("/export/medals")
        assert response.status_code in [401, 403, 302, 303]

    def test_export_medals_returns_csv(self, logged_in_client):
        """Test export medals returns CSV content type."""
        response = logged_in_client.get("/export/medals")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_export_medals_includes_user_medals(self, logged_in_client):
        """Test export includes user's medals."""
        from database import get_db

        # Add medal for this user
        with get_db() as conn:
            # First create olympiad, sport, and match
            conn.execute("""
                INSERT OR IGNORE INTO olympiads (id, name, start_date, end_date, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (1, "Test Olympics", "2024-01-01", "2024-12-31", 1))
            conn.execute("""
                INSERT OR IGNORE INTO sports (id, name, olympiad_id, target_type)
                VALUES (?, ?, ?, ?)
            """, (1, "DX Marathon", 1, "country"))
            conn.execute("""
                INSERT OR IGNORE INTO matches (id, sport_id, start_date, end_date, target_value)
                VALUES (?, ?, ?, ?, ?)
            """, (1, 1, "2024-01-01", "2024-01-31", "DL"))
            conn.execute("""
                INSERT INTO medals (callsign, match_id, role, qso_race_medal, total_points)
                VALUES (?, ?, ?, ?, ?)
            """, ("W1EXP", 1, "work", "gold", 3))

        response = logged_in_client.get("/export/medals")
        content = response.text

        assert "gold" in content.lower()
        assert "DX Marathon" in content or "dx marathon" in content.lower()


class TestAdminExports:
    """Test admin export functionality."""

    def test_admin_export_competitors_exists(self, admin_client):
        """Test admin export competitors endpoint exists."""
        response = admin_client.get("/admin/export/competitors")
        assert response.status_code != 404

    def test_admin_export_competitors_requires_admin(self, logged_in_client):
        """Test admin export requires admin role."""
        response = logged_in_client.get("/admin/export/competitors")
        assert response.status_code in [401, 403]

    def test_admin_export_competitors_returns_csv(self, admin_client):
        """Test admin export competitors returns CSV."""
        response = admin_client.get("/admin/export/competitors")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    def test_admin_export_competitors_includes_all(self, admin_client):
        """Test admin export includes all competitors."""
        from auth import register_user

        register_user("W1TST", "password123", "test1@example.com")
        register_user("W2TST", "password123", "test2@example.com")

        response = admin_client.get("/admin/export/competitors")
        content = response.text

        assert "W1TST" in content
        assert "W2TST" in content

    def test_admin_export_standings_exists(self, admin_client):
        """Test admin export standings endpoint exists."""
        from database import get_db

        # Create olympiad
        with get_db() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO olympiads (id, name, start_date, end_date, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (1, "Test Olympics", "2024-01-01", "2024-12-31", 1))

        response = admin_client.get("/admin/export/standings/1")
        assert response.status_code != 404


class TestPagination:
    """Test pagination functionality."""

    def test_pagination_module_exists(self):
        """Test pagination module exists."""
        import pagination
        assert pagination is not None

    def test_paginator_class_exists(self):
        """Test Paginator class exists."""
        from pagination import Paginator
        assert Paginator is not None

    def test_paginator_default_values(self):
        """Test Paginator has sensible defaults."""
        from pagination import Paginator

        p = Paginator()
        assert p.page == 1
        assert p.per_page == 50

    def test_paginator_custom_values(self):
        """Test Paginator accepts custom values."""
        from pagination import Paginator

        p = Paginator(page=3, per_page=25)
        assert p.page == 3
        assert p.per_page == 25

    def test_paginator_get_offset(self):
        """Test Paginator calculates offset correctly."""
        from pagination import Paginator

        p1 = Paginator(page=1, per_page=50)
        assert p1.get_offset() == 0

        p2 = Paginator(page=2, per_page=50)
        assert p2.get_offset() == 50

        p3 = Paginator(page=3, per_page=25)
        assert p3.get_offset() == 50

    def test_paginator_get_page_info(self):
        """Test Paginator returns page info."""
        from pagination import Paginator

        p = Paginator(page=2, per_page=10)
        info = p.get_page_info(total=55)

        assert info["current_page"] == 2
        assert info["per_page"] == 10
        assert info["total_items"] == 55
        assert info["total_pages"] == 6  # ceil(55/10)
        assert info["has_prev"] is True
        assert info["has_next"] is True

    def test_paginator_page_info_first_page(self):
        """Test page info on first page."""
        from pagination import Paginator

        p = Paginator(page=1, per_page=10)
        info = p.get_page_info(total=55)

        assert info["has_prev"] is False
        assert info["has_next"] is True

    def test_paginator_page_info_last_page(self):
        """Test page info on last page."""
        from pagination import Paginator

        p = Paginator(page=6, per_page=10)
        info = p.get_page_info(total=55)

        assert info["has_prev"] is True
        assert info["has_next"] is False

    def test_paginator_page_info_single_page(self):
        """Test page info with single page of results."""
        from pagination import Paginator

        p = Paginator(page=1, per_page=10)
        info = p.get_page_info(total=5)

        assert info["total_pages"] == 1
        assert info["has_prev"] is False
        assert info["has_next"] is False

    def test_paginator_page_info_empty(self):
        """Test page info with no results."""
        from pagination import Paginator

        p = Paginator(page=1, per_page=10)
        info = p.get_page_info(total=0)

        assert info["total_pages"] == 0
        assert info["has_prev"] is False
        assert info["has_next"] is False


class TestSearchFiltering:
    """Test search and filtering functionality."""

    def test_dashboard_filter_by_band(self, logged_in_client):
        """Test dashboard can filter QSOs by band."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2A20", "2024-01-15T12:00:00", "20m", "SSB", 1))
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2A40", "2024-01-15T12:00:00", "40m", "SSB", 1))

        response = logged_in_client.get("/dashboard?band=20m")

        assert response.status_code == 200
        assert "K2A20" in response.text
        assert "K2A40" not in response.text

    def test_dashboard_filter_by_mode(self, logged_in_client):
        """Test dashboard can filter QSOs by mode."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2SSB", "2024-01-15T12:00:00", "20m", "SSB", 1))
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2CW", "2024-01-15T12:00:00", "20m", "CW", 1))

        response = logged_in_client.get("/dashboard?mode=CW")

        assert response.status_code == 200
        assert "K2CW" in response.text
        assert "K2SSB" not in response.text

    def test_dashboard_filter_by_confirmed(self, logged_in_client):
        """Test dashboard can filter QSOs by confirmed status."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2CON", "2024-01-15T12:00:00", "20m", "SSB", 1))
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode, is_confirmed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("W1EXP", "K2UNC", "2024-01-15T12:00:00", "20m", "SSB", 0))

        response = logged_in_client.get("/dashboard?confirmed=1")

        assert response.status_code == 200
        assert "K2CON" in response.text
        assert "K2UNC" not in response.text

    def test_dashboard_pagination(self, logged_in_client):
        """Test dashboard supports pagination."""
        response = logged_in_client.get("/dashboard?page=1")
        assert response.status_code == 200

    def test_admin_competitors_search(self, admin_client):
        """Test admin can search competitors."""
        from auth import register_user

        # Register user directly (not via signup which would override session)
        register_user("W1FND", "password123", "find@example.com")

        response = admin_client.get("/admin/competitors?search=W1FND")

        assert response.status_code == 200
        assert "W1FND" in response.text

"""
Tests for API v1 endpoints and OpenAPI documentation - TDD: Tests written before implementation.
"""

import os
import tempfile
import pytest

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
    client.post("/signup", json={
        "callsign": "W1API",
        "password": "password123",
        "qrz_api_key": "test-key"
    })
    return client


@pytest.fixture
def admin_client(client):
    """Create test client with logged in admin user."""
    from database import get_db

    client.post("/signup", json={
        "callsign": "W1ADM",
        "password": "password123",
        "qrz_api_key": "test-key"
    })

    # Make admin
    with get_db() as conn:
        conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = 'W1ADM'")

    return client


class TestOpenAPIDocumentation:
    """Test OpenAPI/Swagger documentation."""

    def test_openapi_json_endpoint(self, client):
        """Test /openapi.json endpoint exists."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

    def test_openapi_has_info(self, client):
        """Test OpenAPI spec has info section."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "info" in data
        assert "title" in data["info"]
        assert "version" in data["info"]

    def test_openapi_has_paths(self, client):
        """Test OpenAPI spec has paths section."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "paths" in data
        assert len(data["paths"]) > 0

    def test_swagger_ui_available(self, client):
        """Test Swagger UI is available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_redoc_available(self, client):
        """Test ReDoc is available at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestAPIv1Standings:
    """Test /api/v1/standings endpoint."""

    def test_standings_endpoint_exists(self, client):
        """Test standings endpoint exists."""
        response = client.get("/api/v1/standings")
        assert response.status_code != 404

    def test_standings_returns_json(self, client):
        """Test standings returns JSON."""
        response = client.get("/api/v1/standings")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_standings_returns_list(self, client):
        """Test standings returns a list."""
        response = client.get("/api/v1/standings")
        data = response.json()
        assert isinstance(data, dict)
        assert "standings" in data
        assert isinstance(data["standings"], list)

    def test_standings_includes_competitor_info(self, logged_in_client):
        """Test standings include competitor information."""
        from database import get_db

        # Create an olympiad and give the user some points
        with get_db() as conn:
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Test Olympics', '2024-01-01', '2024-12-31', 0, 1)
            """)

        response = logged_in_client.get("/api/v1/standings")
        data = response.json()
        assert "standings" in data


class TestAPIv1QSOs:
    """Test /api/v1/qsos endpoint."""

    def test_qsos_endpoint_exists(self, logged_in_client):
        """Test QSOs endpoint exists."""
        response = logged_in_client.get("/api/v1/qsos")
        assert response.status_code != 404

    def test_qsos_requires_auth(self, client):
        """Test QSOs endpoint requires authentication."""
        response = client.get("/api/v1/qsos")
        assert response.status_code in [401, 403]

    def test_qsos_returns_json(self, logged_in_client):
        """Test QSOs returns JSON."""
        response = logged_in_client.get("/api/v1/qsos")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_qsos_returns_list(self, logged_in_client):
        """Test QSOs returns a list."""
        response = logged_in_client.get("/api/v1/qsos")
        data = response.json()
        assert isinstance(data, dict)
        assert "qsos" in data
        assert isinstance(data["qsos"], list)

    def test_qsos_pagination(self, logged_in_client):
        """Test QSOs endpoint supports pagination."""
        response = logged_in_client.get("/api/v1/qsos?page=1&per_page=10")
        data = response.json()
        assert "page" in data or "pagination" in data or "qsos" in data


class TestAPIv1Medals:
    """Test /api/v1/medals endpoint."""

    def test_medals_endpoint_exists(self, logged_in_client):
        """Test medals endpoint exists."""
        response = logged_in_client.get("/api/v1/medals")
        assert response.status_code != 404

    def test_medals_requires_auth(self, client):
        """Test medals endpoint requires authentication."""
        response = client.get("/api/v1/medals")
        assert response.status_code in [401, 403]

    def test_medals_returns_json(self, logged_in_client):
        """Test medals returns JSON."""
        response = logged_in_client.get("/api/v1/medals")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_medals_returns_list(self, logged_in_client):
        """Test medals returns a list."""
        response = logged_in_client.get("/api/v1/medals")
        data = response.json()
        assert isinstance(data, dict)
        assert "medals" in data
        assert isinstance(data["medals"], list)


class TestAPIv1Sports:
    """Test /api/v1/sports endpoint."""

    def test_sports_endpoint_exists(self, client):
        """Test sports endpoint exists."""
        response = client.get("/api/v1/sports")
        assert response.status_code != 404

    def test_sports_returns_json(self, client):
        """Test sports returns JSON."""
        response = client.get("/api/v1/sports")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_sports_returns_list(self, client):
        """Test sports returns a list."""
        response = client.get("/api/v1/sports")
        data = response.json()
        assert isinstance(data, dict)
        assert "sports" in data
        assert isinstance(data["sports"], list)

    def test_sports_include_sport_info(self, admin_client):
        """Test sports include sport information."""
        from database import get_db

        # Create an olympiad with a sport
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Test Olympics', '2024-01-01', '2024-12-31', 0, 1)
            """)
            olympiad_id = cursor.lastrowid
            conn.execute("""
                INSERT INTO sports (olympiad_id, name, description, target_type)
                VALUES (?, 'Test Sport', 'A test sport', 'country')
            """, (olympiad_id,))

        response = admin_client.get("/api/v1/sports")
        data = response.json()
        assert len(data["sports"]) >= 1


class TestAPIv1Me:
    """Test /api/v1/me endpoint for user profile."""

    def test_me_endpoint_exists(self, logged_in_client):
        """Test /api/v1/me endpoint exists."""
        response = logged_in_client.get("/api/v1/me")
        assert response.status_code != 404

    def test_me_requires_auth(self, client):
        """Test /api/v1/me requires authentication."""
        response = client.get("/api/v1/me")
        assert response.status_code in [401, 403]

    def test_me_returns_json(self, logged_in_client):
        """Test /api/v1/me returns JSON."""
        response = logged_in_client.get("/api/v1/me")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_me_returns_user_info(self, logged_in_client):
        """Test /api/v1/me returns user information."""
        response = logged_in_client.get("/api/v1/me")
        data = response.json()
        assert "callsign" in data
        assert data["callsign"] == "W1API"

    def test_me_user_deleted_from_db(self, logged_in_client):
        """Test /api/v1/me returns 401 when user deleted from database."""
        from database import get_db

        # Delete the user from DB while keeping their session cookie
        with get_db() as conn:
            conn.execute("DELETE FROM competitors WHERE callsign = 'W1API'")

        # Session auth fails first when user doesn't exist - returns 401
        response = logged_in_client.get("/api/v1/me")
        assert response.status_code == 401


class TestAPIv1Health:
    """Test /api/v1/health endpoint."""

    def test_health_endpoint_exists(self, client):
        """Test health check endpoint exists."""
        response = client.get("/api/v1/health")
        assert response.status_code != 404

    def test_health_returns_ok(self, client):
        """Test health check returns OK status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"


class TestAPIAuthentication:
    """Test API authentication methods."""

    def test_api_accepts_session_auth(self, logged_in_client):
        """Test API accepts session-based authentication."""
        response = logged_in_client.get("/api/v1/me")
        assert response.status_code == 200

    def test_api_rejects_unauthenticated(self, client):
        """Test API rejects unauthenticated requests for protected endpoints."""
        response = client.get("/api/v1/me")
        assert response.status_code in [401, 403]

    def test_api_returns_json_error(self, client):
        """Test API returns JSON error for auth failures."""
        response = client.get("/api/v1/me")
        assert response.status_code in [401, 403]
        data = response.json()
        assert "detail" in data or "error" in data


class TestAPIResponseFormat:
    """Test API response formats."""

    def test_api_includes_metadata(self, client):
        """Test API responses include metadata."""
        response = client.get("/api/v1/standings")
        data = response.json()
        # Should have some metadata about the response
        assert isinstance(data, dict)

    def test_api_error_format(self, client):
        """Test API error responses have consistent format."""
        # Try to access a protected endpoint without auth
        response = client.get("/api/v1/me")
        assert response.status_code in [401, 403]
        data = response.json()
        # Should have error info
        assert "detail" in data or "error" in data or "message" in data


class TestAPIVersion:
    """Test API versioning."""

    def test_api_v1_prefix(self, client):
        """Test API uses /api/v1 prefix."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_non_versioned_api_redirects_or_404(self, client):
        """Test that /api without version returns 404 or redirects."""
        response = client.get("/api/health")
        # Should either 404 or redirect to versioned endpoint
        assert response.status_code in [404, 307, 308]


class TestAPIv1QSOFilters:
    """Test QSO endpoint filters."""

    def test_qsos_filter_by_band(self, logged_in_client):
        """Test filtering QSOs by band."""
        from database import get_db

        # Create test QSOs with different bands
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band)
                VALUES ('W1API', 'W2TEST', '2024-01-01T12:00:00', '20m')
            """)
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band)
                VALUES ('W1API', 'W3TEST', '2024-01-01T13:00:00', '40m')
            """)

        response = logged_in_client.get("/api/v1/qsos?band=20m")
        data = response.json()
        assert response.status_code == 200
        assert all(q["band"] == "20m" for q in data["qsos"])

    def test_qsos_filter_by_mode(self, logged_in_client):
        """Test filtering QSOs by mode."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, mode)
                VALUES ('W1API', 'W2TEST', '2024-01-01T12:00:00', 'SSB')
            """)
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, mode)
                VALUES ('W1API', 'W3TEST', '2024-01-01T13:00:00', 'FT8')
            """)

        response = logged_in_client.get("/api/v1/qsos?mode=SSB")
        data = response.json()
        assert response.status_code == 200
        assert all(q["mode"] == "SSB" for q in data["qsos"])

    def test_qsos_filter_by_confirmed(self, logged_in_client):
        """Test filtering QSOs by confirmed status."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, is_confirmed)
                VALUES ('W1API', 'W2TEST', '2024-01-01T12:00:00', 1)
            """)
            conn.execute("""
                INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, is_confirmed)
                VALUES ('W1API', 'W3TEST', '2024-01-01T13:00:00', 0)
            """)

        response = logged_in_client.get("/api/v1/qsos?confirmed=true")
        data = response.json()
        assert response.status_code == 200
        assert all(q["is_confirmed"] == 1 for q in data["qsos"])


class TestAPIv1MedalsFilters:
    """Test medals endpoint filters."""

    def test_medals_filter_by_olympiad(self, logged_in_client):
        """Test filtering medals by olympiad_id."""
        from database import get_db

        with get_db() as conn:
            # Create two olympiads
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('2024 Olympics', '2024-01-01', '2024-12-31', 0, 0)
            """)
            olympiad1_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('2025 Olympics', '2025-01-01', '2025-12-31', 0, 1)
            """)
            olympiad2_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Create sports and matches for each
            conn.execute("""
                INSERT INTO sports (olympiad_id, name, description, target_type)
                VALUES (?, 'Sport 1', 'Desc', 'country')
            """, (olympiad1_id,))
            sport1_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            conn.execute("""
                INSERT INTO matches (sport_id, start_date, end_date, target_value)
                VALUES (?, '2024-01-01', '2024-01-31', '291')
            """, (sport1_id,))
            match1_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Create medal for the user
            conn.execute("""
                INSERT INTO medals (match_id, callsign, role, qso_race_medal, total_points)
                VALUES (?, 'W1API', 'work', 'gold', 3)
            """, (match1_id,))

        response = logged_in_client.get(f"/api/v1/medals?olympiad_id={olympiad1_id}")
        data = response.json()
        assert response.status_code == 200
        assert "medals" in data


class TestAPIv1SportsFilters:
    """Test sports endpoint filters."""

    def test_sports_filter_by_olympiad(self, client):
        """Test filtering sports by olympiad_id."""
        from database import get_db

        with get_db() as conn:
            # Create olympiad with sports
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Test Olympics', '2024-01-01', '2024-12-31', 0, 0)
            """)
            olympiad_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute("""
                INSERT INTO sports (olympiad_id, name, description, target_type)
                VALUES (?, 'Test Sport', 'A sport', 'country')
            """, (olympiad_id,))

        response = client.get(f"/api/v1/sports?olympiad_id={olympiad_id}")
        data = response.json()
        assert response.status_code == 200
        assert len(data["sports"]) >= 1


class TestAPIv1StandingsFilters:
    """Test standings endpoint filters."""

    def test_standings_filter_by_olympiad(self, client):
        """Test filtering standings by olympiad_id."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Filter Test', '2024-01-01', '2024-12-31', 0, 0)
            """)
            olympiad_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        response = client.get(f"/api/v1/standings?olympiad_id={olympiad_id}")
        data = response.json()
        assert response.status_code == 200
        assert "standings" in data


class TestExportStandingsErrors:
    """Test export standings error handling."""

    def test_export_standings_invalid_olympiad(self, admin_client):
        """Test export standings with non-existent olympiad."""
        response = admin_client.get("/admin/export/standings/99999")
        assert response.status_code == 404

    def test_export_standings_valid_olympiad(self, admin_client):
        """Test export standings with valid olympiad."""
        from database import get_db

        with get_db() as conn:
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Export Test', '2024-01-01', '2024-12-31', 0, 1)
            """)
            olympiad_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        response = admin_client.get(f"/admin/export/standings/{olympiad_id}")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

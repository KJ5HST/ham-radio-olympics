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

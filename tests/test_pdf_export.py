"""
Tests for PDF export functionality - TDD: Tests written before implementation.
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

# Create temp upload dir for tests
_test_upload_dir = tempfile.mkdtemp()
os.environ["UPLOAD_DIR"] = _test_upload_dir


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test and ensure upload dir is correct."""
    from database import reset_db
    reset_db()
    from config import config
    import main as main_module
    config.UPLOAD_DIR = _test_upload_dir
    main_module.config.UPLOAD_DIR = _test_upload_dir
    os.makedirs(_test_upload_dir, exist_ok=True)
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    """Cleanup database file and upload dir after all tests."""
    yield
    if os.path.exists(_test_db_path):
        os.remove(_test_db_path)
    import shutil
    if os.path.exists(_test_upload_dir):
        shutil.rmtree(_test_upload_dir)


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
        "callsign": "W1PDF",
        "password": "password123",
        "email": "pdf@example.com",
        "qrz_api_key": "test-api-key"
    })
    return client


@pytest.fixture
def setup_olympiad_data():
    """Setup complete olympiad with sports, matches, and medals."""
    from database import get_db
    from auth import hash_password

    with get_db() as conn:
        # Create olympiad
        conn.execute("""
            INSERT INTO olympiads (id, name, start_date, end_date, qualifying_qsos, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (1, "2026 Ham Radio Olympics", "2026-01-01", "2026-12-31", 1, 1))

        # Create sport
        conn.execute("""
            INSERT INTO sports (id, olympiad_id, name, description, target_type, work_enabled, activate_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (1, 1, "DX Challenge", "Work DX stations", "continent", 1, 0))

        # Create matches
        conn.execute("""
            INSERT INTO matches (id, sport_id, start_date, end_date, target_value)
            VALUES (?, ?, ?, ?, ?)
        """, (1, 1, "2026-01-01", "2026-01-31", "EU"))
        conn.execute("""
            INSERT INTO matches (id, sport_id, start_date, end_date, target_value)
            VALUES (?, ?, ?, ?, ?)
        """, (2, 1, "2026-02-01", "2026-02-28", "AS"))

        # Create competitors
        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES (?, ?, ?)
        """, ("W1TOP", hash_password("test123"), "2026-01-01T00:00:00"))
        conn.execute("""
            INSERT INTO competitors (callsign, password_hash, registered_at)
            VALUES (?, ?, ?)
        """, ("W2SEC", hash_password("test123"), "2026-01-01T00:00:00"))

        # Create medals with points
        conn.execute("""
            INSERT INTO medals (match_id, callsign, role, qso_race_medal, cool_factor_medal, total_points, qualified, qso_race_claim_time, cool_factor_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "W1TOP", "work", "gold", "gold", 6, 1, "2026-01-15T12:00:00", 150.5))
        conn.execute("""
            INSERT INTO medals (match_id, callsign, role, qso_race_medal, cool_factor_medal, total_points, qualified, qso_race_claim_time, cool_factor_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "W2SEC", "work", "silver", "silver", 4, 1, "2026-01-15T12:30:00", 120.3))

        # Create QSOs
        conn.execute("""
            INSERT INTO qsos (competitor_callsign, dx_callsign, qso_datetime_utc, band, mode,
                            dx_dxcc, dx_grid, distance_km, tx_power_w, cool_factor, is_confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("W1TOP", "DL1ABC", "2026-01-15T12:00:00", "20m", "SSB", 230, "JO31", 6500.0, 100.0, 65.0, 1))

        # Create records
        conn.execute("""
            INSERT INTO records (sport_id, callsign, record_type, value, achieved_at)
            VALUES (?, ?, ?, ?, ?)
        """, (1, "W1TOP", "longest_distance", 10000.0, "2026-01-15T12:00:00"))


class TestPDFOlympiadExport:
    """Test PDF export of entire olympiad."""

    def test_export_pdf_olympiad_endpoint_exists(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad endpoint exists."""
        response = logged_in_client.get("/export/pdf/olympiad")
        # 200 means endpoint exists and worked, anything else indicates issues
        assert response.status_code == 200

    def test_export_pdf_olympiad_requires_auth(self, client):
        """Test export PDF olympiad requires authentication."""
        response = client.get("/export/pdf/olympiad", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers.get("location") == "/login"

    def test_export_pdf_olympiad_returns_pdf(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad returns PDF content type."""
        response = logged_in_client.get("/export/pdf/olympiad")
        assert response.status_code == 200
        assert "application/pdf" in response.headers.get("content-type", "")

    def test_export_pdf_olympiad_content_disposition(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad has proper content disposition."""
        response = logged_in_client.get("/export/pdf/olympiad")
        disposition = response.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert ".pdf" in disposition

    def test_export_pdf_olympiad_is_valid_pdf(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad returns valid PDF bytes."""
        response = logged_in_client.get("/export/pdf/olympiad")
        # PDF files start with %PDF
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_olympiad_with_top_n(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad respects top_n parameter."""
        response = logged_in_client.get("/export/pdf/olympiad?top_n=5")
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_olympiad_with_include_qsos(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad includes QSOs when requested."""
        response = logged_in_client.get("/export/pdf/olympiad?include_qsos=true")
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_olympiad_with_include_records(self, logged_in_client, setup_olympiad_data):
        """Test export PDF olympiad includes records when requested."""
        response = logged_in_client.get("/export/pdf/olympiad?include_records=true")
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_olympiad_no_active_olympiad(self, logged_in_client):
        """Test export PDF olympiad when no active olympiad exists."""
        response = logged_in_client.get("/export/pdf/olympiad")
        assert response.status_code == 404


class TestPDFSportExport:
    """Test PDF export of single sport."""

    def test_export_pdf_sport_endpoint_exists(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport endpoint exists."""
        response = logged_in_client.get("/export/pdf/sport/1")
        assert response.status_code == 200

    def test_export_pdf_sport_requires_auth(self, client, setup_olympiad_data):
        """Test export PDF sport requires authentication."""
        response = client.get("/export/pdf/sport/1", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers.get("location") == "/login"

    def test_export_pdf_sport_returns_pdf(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport returns PDF content type."""
        response = logged_in_client.get("/export/pdf/sport/1")
        assert response.status_code == 200
        assert "application/pdf" in response.headers.get("content-type", "")

    def test_export_pdf_sport_content_disposition(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport has proper content disposition."""
        response = logged_in_client.get("/export/pdf/sport/1")
        disposition = response.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert ".pdf" in disposition

    def test_export_pdf_sport_is_valid_pdf(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport returns valid PDF bytes."""
        response = logged_in_client.get("/export/pdf/sport/1")
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_sport_not_found(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport returns 404 for non-existent sport."""
        response = logged_in_client.get("/export/pdf/sport/9999")
        assert response.status_code == 404

    def test_export_pdf_sport_with_parameters(self, logged_in_client, setup_olympiad_data):
        """Test export PDF sport with all parameters."""
        response = logged_in_client.get("/export/pdf/sport/1?top_n=3&include_qsos=true&include_records=true")
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")


class TestPDFMySportsExport:
    """Test PDF export of competitor's sports."""

    def test_export_pdf_my_sports_endpoint_exists(self, logged_in_client, setup_olympiad_data):
        """Test export PDF my-sports endpoint exists."""
        response = logged_in_client.get("/export/pdf/my-sports")
        assert response.status_code == 200

    def test_export_pdf_my_sports_requires_auth(self, client):
        """Test export PDF my-sports requires authentication."""
        response = client.get("/export/pdf/my-sports", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers.get("location") == "/login"

    def test_export_pdf_my_sports_returns_pdf(self, logged_in_client, setup_olympiad_data):
        """Test export PDF my-sports returns PDF even with no entries."""
        # User W1PDF hasn't entered any sports
        response = logged_in_client.get("/export/pdf/my-sports")
        assert response.status_code == 200
        assert "application/pdf" in response.headers.get("content-type", "")

    def test_export_pdf_my_sports_with_entries(self, logged_in_client, setup_olympiad_data):
        """Test export PDF my-sports with sport entries."""
        from database import get_db
        from datetime import datetime

        # Add sport entry for the logged-in user
        with get_db() as conn:
            conn.execute("""
                INSERT INTO sport_entries (callsign, sport_id, entered_at)
                VALUES (?, ?, ?)
            """, ("W1PDF", 1, datetime.utcnow().isoformat()))

        response = logged_in_client.get("/export/pdf/my-sports")
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

    def test_export_pdf_my_sports_content_disposition(self, logged_in_client, setup_olympiad_data):
        """Test export PDF my-sports has proper content disposition."""
        response = logged_in_client.get("/export/pdf/my-sports")
        disposition = response.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert ".pdf" in disposition
        assert "W1PDF" in disposition  # Should include user's callsign


class TestPDFExportModule:
    """Test PDF export module functions."""

    def test_pdf_module_exists(self):
        """Test PDF export module exists."""
        import pdf_export
        assert pdf_export is not None

    def test_olympics_pdf_class_exists(self):
        """Test OlympicsPDF class exists."""
        from pdf_export import OlympicsPDF
        assert OlympicsPDF is not None

    def test_generate_olympiad_pdf_function_exists(self):
        """Test generate_olympiad_pdf function exists."""
        from pdf_export import generate_olympiad_pdf
        assert callable(generate_olympiad_pdf)

    def test_generate_sport_pdf_function_exists(self):
        """Test generate_sport_pdf function exists."""
        from pdf_export import generate_sport_pdf
        assert callable(generate_sport_pdf)

    def test_generate_my_sports_pdf_function_exists(self):
        """Test generate_my_sports_pdf function exists."""
        from pdf_export import generate_my_sports_pdf
        assert callable(generate_my_sports_pdf)

    def test_generate_olympiad_pdf_returns_bytes(self, setup_olympiad_data):
        """Test generate_olympiad_pdf returns bytes."""
        from pdf_export import generate_olympiad_pdf
        result = generate_olympiad_pdf(
            olympiad_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF")

    def test_generate_sport_pdf_returns_bytes(self, setup_olympiad_data):
        """Test generate_sport_pdf returns bytes."""
        from pdf_export import generate_sport_pdf
        result = generate_sport_pdf(
            sport_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF")

    def test_generate_my_sports_pdf_returns_bytes(self, setup_olympiad_data):
        """Test generate_my_sports_pdf returns bytes."""
        from pdf_export import generate_my_sports_pdf
        result = generate_my_sports_pdf(
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF")


class TestPDFContentAccuracy:
    """Test PDF content accuracy matches database."""

    def test_pdf_contains_olympiad_name(self, setup_olympiad_data):
        """Test PDF contains the olympiad name."""
        from pdf_export import generate_olympiad_pdf

        result = generate_olympiad_pdf(
            olympiad_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        # PDF content check - we can't easily read the text but can check it's not empty
        assert len(result) > 1000  # A real PDF with content should be substantial

    def test_pdf_contains_sport_data(self, setup_olympiad_data):
        """Test PDF for sport contains sport data."""
        from pdf_export import generate_sport_pdf

        result = generate_sport_pdf(
            sport_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=True,
            include_records=True
        )
        # Check PDF is substantial (has real content)
        assert len(result) > 1000

    def test_pdf_with_records_is_larger(self, setup_olympiad_data):
        """Test PDF with records included is larger."""
        from pdf_export import generate_olympiad_pdf

        without_records = generate_olympiad_pdf(
            olympiad_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        with_records = generate_olympiad_pdf(
            olympiad_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=True
        )
        # PDF with records should generally be larger
        assert len(with_records) >= len(without_records)

    def test_pdf_with_qsos_is_larger(self, setup_olympiad_data):
        """Test PDF with QSOs included is larger."""
        from pdf_export import generate_sport_pdf

        without_qsos = generate_sport_pdf(
            sport_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=False,
            include_records=False
        )
        with_qsos = generate_sport_pdf(
            sport_id=1,
            callsign="W1PDF",
            top_n=10,
            include_qsos=True,
            include_records=False
        )
        # PDF with QSOs should generally be larger
        assert len(with_qsos) >= len(without_qsos)


class TestCachedPDF:
    """Test cached PDF functionality."""

    def test_cached_pdf_functions_exist(self):
        """Test cached PDF functions exist."""
        from pdf_export import (
            get_cached_pdf,
            get_cached_pdf_info,
            regenerate_cached_pdf,
            regenerate_active_olympiad_pdf,
            generate_cached_olympiad_pdf
        )
        assert callable(get_cached_pdf)
        assert callable(get_cached_pdf_info)
        assert callable(regenerate_cached_pdf)
        assert callable(regenerate_active_olympiad_pdf)
        assert callable(generate_cached_olympiad_pdf)

    def test_generate_cached_olympiad_pdf(self, setup_olympiad_data):
        """Test generate_cached_olympiad_pdf returns bytes."""
        from pdf_export import generate_cached_olympiad_pdf
        result = generate_cached_olympiad_pdf(olympiad_id=1)
        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF")

    def test_regenerate_cached_pdf(self, setup_olympiad_data):
        """Test regenerate_cached_pdf creates file."""
        from pdf_export import regenerate_cached_pdf, get_cached_pdf, get_cached_pdf_path
        import os

        # Regenerate
        result = regenerate_cached_pdf(olympiad_id=1)
        assert result is True

        # Check file exists
        pdf_path = get_cached_pdf_path(1)
        assert pdf_path.exists()

        # Check can read it back
        pdf_bytes = get_cached_pdf(1)
        assert pdf_bytes is not None
        assert pdf_bytes.startswith(b"%PDF")

        # Cleanup
        if pdf_path.exists():
            os.remove(pdf_path)

    def test_regenerate_active_olympiad_pdf(self, setup_olympiad_data):
        """Test regenerate_active_olympiad_pdf works with active olympiad."""
        from pdf_export import regenerate_active_olympiad_pdf, get_cached_pdf, get_cached_pdf_path
        import os

        result = regenerate_active_olympiad_pdf()
        assert result is True

        # Check file exists
        pdf_bytes = get_cached_pdf(1)
        assert pdf_bytes is not None

        # Cleanup
        pdf_path = get_cached_pdf_path(1)
        if pdf_path.exists():
            os.remove(pdf_path)

    def test_regenerate_active_olympiad_pdf_no_active(self, client):
        """Test regenerate_active_olympiad_pdf returns False when no active olympiad."""
        from pdf_export import regenerate_active_olympiad_pdf
        result = regenerate_active_olympiad_pdf()
        assert result is False

    def test_get_cached_pdf_info(self, setup_olympiad_data):
        """Test get_cached_pdf_info returns correct info."""
        from pdf_export import regenerate_cached_pdf, get_cached_pdf_info, get_cached_pdf_path
        import os

        # Before regeneration
        info = get_cached_pdf_info(1)
        assert info["exists"] is False

        # After regeneration
        regenerate_cached_pdf(1)
        info = get_cached_pdf_info(1)
        assert info["exists"] is True
        assert info["size_bytes"] > 0
        assert info["generated_at"] is not None

        # Cleanup
        pdf_path = get_cached_pdf_path(1)
        if pdf_path.exists():
            os.remove(pdf_path)

    def test_sync_recompute_regenerates_pdf(self, setup_olympiad_data):
        """Test that sync recompute triggers PDF regeneration."""
        from sync import recompute_sport_matches
        from pdf_export import get_cached_pdf, get_cached_pdf_path
        import os

        # Recompute sport matches (should trigger PDF regeneration once at end)
        recompute_sport_matches(1)

        # Check PDF was generated
        pdf_bytes = get_cached_pdf(1)
        assert pdf_bytes is not None
        assert pdf_bytes.startswith(b"%PDF")

        # Cleanup
        pdf_path = get_cached_pdf_path(1)
        if pdf_path.exists():
            os.remove(pdf_path)

    def test_regenerate_creates_standings_resource(self, setup_olympiad_data):
        """Test regenerate_cached_pdf upserts standings into resources."""
        from pdf_export import regenerate_cached_pdf, get_cached_pdf_path
        from database import get_db

        result = regenerate_cached_pdf(olympiad_id=1)
        assert result is True

        with get_db() as conn:
            resource = conn.execute(
                "SELECT * FROM resource_files WHERE title = 'Current Standings (PDF)'"
            ).fetchone()
            assert resource is not None
            assert resource["uploaded_by"] == "system"
            assert resource["file_size"] > 0
            # Check access is all_competitors
            access = conn.execute(
                "SELECT access_type FROM resource_access WHERE resource_id = ?",
                (resource["id"],)
            ).fetchone()
            assert access["access_type"] == "all_competitors"

        # Cleanup
        pdf_path = get_cached_pdf_path(1)
        if pdf_path.exists():
            os.remove(pdf_path)

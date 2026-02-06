"""
Tests for admin resource file distribution — TDD: Tests written before implementation.
"""

import io
import os
import tempfile
import pytest
from datetime import datetime

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
    # Patch UPLOAD_DIR on BOTH config objects (config module singleton AND main module's import)
    # In full test suite these can be different object instances
    from config import config
    import main as main_module
    config.UPLOAD_DIR = _test_upload_dir
    main_module.config.UPLOAD_DIR = _test_upload_dir
    os.makedirs(_test_upload_dir, exist_ok=True)
    # Clean upload dir between tests
    for f in os.listdir(_test_upload_dir):
        fpath = os.path.join(_test_upload_dir, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Cleanup database and upload dir after all tests."""
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
def admin_client(client):
    """Create test client with logged in admin user."""
    from database import get_db
    client.post("/signup", json={
        "callsign": "W1ADM",
        "password": "password123",
        "email": "admin@example.com",
        "qrz_api_key": "test-api-key"
    })
    with get_db() as conn:
        conn.execute("UPDATE competitors SET is_admin = 1 WHERE callsign = 'W1ADM'")
    return client


@pytest.fixture
def regular_client():
    """Create test client with logged in regular user."""
    from fastapi.testclient import TestClient
    from main import app
    c = TestClient(app)
    c.post("/signup", json={
        "callsign": "W2USR",
        "password": "password123",
        "email": "user@example.com",
        "qrz_api_key": "test-api-key"
    })
    return c


@pytest.fixture
def referee_client():
    """Create test client with logged in referee user."""
    from fastapi.testclient import TestClient
    from main import app
    from database import get_db
    c = TestClient(app)
    c.post("/signup", json={
        "callsign": "W3REF",
        "password": "password123",
        "email": "ref@example.com",
        "qrz_api_key": "test-api-key"
    })
    with get_db() as conn:
        conn.execute("UPDATE competitors SET is_referee = 1 WHERE callsign = 'W3REF'")
    return c


def _make_pdf_bytes(size=1024):
    """Create a fake PDF file of given size."""
    return b"%PDF-1.4 " + b"x" * (size - 9)


def _upload_file(admin_client, title="Test File", description="A test file",
                 filename="test.pdf", content=None, access_types=None,
                 sport_ids=None, callsigns=None):
    """Helper to upload a file via the admin endpoint."""
    if content is None:
        content = _make_pdf_bytes()
    if access_types is None:
        access_types = ["public"]

    # Build multipart fields — httpx requires all fields as list of tuples
    # with files as (field, (filename, content, content_type)) tuples
    fields = [
        ("title", (None, title)),
        ("description", (None, description)),
    ]
    for at in access_types:
        fields.append(("access_types", (None, at)))
    if sport_ids:
        for sid in sport_ids:
            fields.append(("sport_ids", (None, str(sid))))
    if callsigns:
        fields.append(("callsigns", (None, callsigns)))
    fields.append(("file", (filename, content, "application/pdf")))

    return admin_client.post(
        "/admin/resources/upload",
        files=fields,
    )


class TestResourceUpload:
    """Test resource file upload."""

    def test_admin_can_upload_file(self, admin_client):
        """Admin can upload a file successfully."""
        resp = _upload_file(admin_client)
        assert resp.status_code in (200, 303), resp.text

    def test_non_admin_rejected(self, regular_client):
        """Non-admin users cannot upload files."""
        resp = regular_client.post(
            "/admin/resources/upload",
            files=[
                ("title", (None, "Hack")),
                ("access_types", (None, "public")),
                ("file", ("test.pdf", b"data", "application/pdf")),
            ],
        )
        assert resp.status_code in (403, 303)

    def test_oversized_file_rejected(self, admin_client):
        """Files exceeding MAX_UPLOAD_SIZE are rejected."""
        from config import config
        big = b"x" * (config.MAX_UPLOAD_SIZE + 1)
        resp = _upload_file(admin_client, content=big)
        assert resp.status_code == 400

    def test_bad_extension_rejected(self, admin_client):
        """Files with disallowed extensions are rejected."""
        resp = _upload_file(admin_client, filename="evil.exe", content=b"MZ" + b"\x00" * 100)
        assert resp.status_code == 400

    def test_title_required(self, admin_client):
        """Upload without title is rejected."""
        resp = admin_client.post(
            "/admin/resources/upload",
            files=[
                ("title", (None, "")),
                ("access_types", (None, "public")),
                ("file", ("test.pdf", _make_pdf_bytes(), "application/pdf")),
            ],
        )
        assert resp.status_code in (400, 422)

    def test_file_stored_on_disk(self, admin_client):
        """Uploaded file is saved to disk with correct extension."""
        _upload_file(admin_client)
        from database import get_db
        with get_db() as conn:
            rf = conn.execute("SELECT stored_filename FROM resource_files").fetchone()
        assert rf is not None
        assert rf["stored_filename"].endswith(".pdf")
        file_path = os.path.join(_test_upload_dir, rf["stored_filename"])
        assert os.path.exists(file_path), f"File not found at {file_path}"

    def test_db_rows_created(self, admin_client):
        """Upload creates resource_files and resource_access rows."""
        from database import get_db
        _upload_file(admin_client, access_types=["public", "all_competitors"])

        with get_db() as conn:
            rf = conn.execute("SELECT * FROM resource_files").fetchall()
            assert len(rf) == 1
            assert rf[0]["title"] == "Test File"
            assert rf[0]["uploaded_by"] == "W1ADM"

            ra = conn.execute("SELECT * FROM resource_access WHERE resource_id = ?", (rf[0]["id"],)).fetchall()
            assert len(ra) == 2
            types = {r["access_type"] for r in ra}
            assert types == {"public", "all_competitors"}

    def test_audit_log_entry(self, admin_client):
        """Upload creates an audit log entry."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            logs = conn.execute(
                "SELECT * FROM audit_log WHERE action = 'resource_upload'"
            ).fetchall()
            assert len(logs) == 1
            assert logs[0]["actor_callsign"] == "W1ADM"

    def test_sport_access_rule(self, admin_client):
        """Upload with sport access creates sport rule in resource_access."""
        from database import get_db
        # Create an olympiad and sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active) VALUES ('Test', '2026-01-01', '2026-12-31', 0, 1)"
            )
            conn.execute(
                "INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'DX Challenge', 'continent')"
            )

        _upload_file(admin_client, access_types=["sport"], sport_ids=[1])

        with get_db() as conn:
            ra = conn.execute("SELECT * FROM resource_access WHERE access_type = 'sport'").fetchall()
            assert len(ra) == 1
            assert ra[0]["access_value"] == "1"

    def test_individual_access_rule(self, admin_client):
        """Upload with individual callsigns creates individual rules."""
        from database import get_db
        _upload_file(admin_client, access_types=["individual"], callsigns="W2USR, W3ABC")

        with get_db() as conn:
            ra = conn.execute("SELECT * FROM resource_access WHERE access_type = 'individual' ORDER BY access_value").fetchall()
            assert len(ra) == 2
            assert ra[0]["access_value"] == "W2USR"
            assert ra[1]["access_value"] == "W3ABC"


class TestResourceDelete:
    """Test resource file deletion."""

    def test_admin_can_delete(self, admin_client):
        """Admin can delete a resource."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()
        resp = admin_client.delete(f"/admin/resources/{rf['id']}")
        assert resp.status_code == 200

    def test_non_admin_rejected(self, admin_client, regular_client):
        """Non-admin cannot delete a resource."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()
        resp = regular_client.delete(f"/admin/resources/{rf['id']}")
        assert resp.status_code in (403, 303)

    def test_404_for_missing(self, admin_client):
        """Deleting non-existent resource returns 404."""
        resp = admin_client.delete("/admin/resources/9999")
        assert resp.status_code == 404

    def test_disk_file_removed(self, admin_client):
        """Deleting a resource removes the file from disk."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT * FROM resource_files").fetchone()
            stored = rf["stored_filename"]
        file_path = os.path.join(_test_upload_dir, stored)
        assert os.path.exists(file_path), f"File not found at {file_path} before delete"

        admin_client.delete(f"/admin/resources/{rf['id']}")
        assert not os.path.exists(file_path)

    def test_db_rows_removed(self, admin_client):
        """Deleting a resource removes DB rows."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()
        admin_client.delete(f"/admin/resources/{rf['id']}")

        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) FROM resource_files").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM resource_access").fetchone()[0] == 0

    def test_audit_log_entry(self, admin_client):
        """Deleting a resource creates audit log entry."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()
        admin_client.delete(f"/admin/resources/{rf['id']}")

        with get_db() as conn:
            logs = conn.execute(
                "SELECT * FROM audit_log WHERE action = 'resource_delete'"
            ).fetchall()
            assert len(logs) == 1

    def test_handles_missing_disk_file(self, admin_client):
        """Deletion succeeds even if disk file is already gone."""
        from database import get_db
        _upload_file(admin_client)
        with get_db() as conn:
            rf = conn.execute("SELECT * FROM resource_files").fetchone()
        # Remove file manually
        file_path = os.path.join(_test_upload_dir, rf["stored_filename"])
        if os.path.exists(file_path):
            os.remove(file_path)

        resp = admin_client.delete(f"/admin/resources/{rf['id']}")
        assert resp.status_code == 200


class TestResourceAdminPage:
    """Test admin resources management page."""

    def test_loads_for_admin(self, admin_client):
        """Admin resources page loads for admin."""
        resp = admin_client.get("/admin/resources")
        assert resp.status_code == 200
        assert "Resources" in resp.text

    def test_rejected_for_non_admin(self, regular_client):
        """Admin resources page rejected for non-admin."""
        resp = regular_client.get("/admin/resources")
        assert resp.status_code in (403, 303)

    def test_lists_files(self, admin_client):
        """Admin resources page lists uploaded files."""
        _upload_file(admin_client, title="My Contest Rules")
        resp = admin_client.get("/admin/resources")
        assert resp.status_code == 200
        assert "My Contest Rules" in resp.text

    def test_shows_access_summary(self, admin_client):
        """Admin resources page shows access summary."""
        _upload_file(admin_client, title="Public Doc", access_types=["public"])
        resp = admin_client.get("/admin/resources")
        assert "Public" in resp.text

    def test_dashboard_has_resources_link(self, admin_client):
        """Admin dashboard has link to resources."""
        resp = admin_client.get("/admin")
        assert resp.status_code == 200
        assert "/admin/resources" in resp.text


class TestResourceAccessControl:
    """Test resource access control logic."""

    def _create_resource_with_access(self, admin_client, access_types, sport_ids=None, callsigns=None):
        """Helper to create a resource and return its ID."""
        from database import get_db
        _upload_file(admin_client, access_types=access_types, sport_ids=sport_ids, callsigns=callsigns)
        with get_db() as conn:
            return conn.execute("SELECT id FROM resource_files ORDER BY id DESC LIMIT 1").fetchone()["id"]

    def test_public_file_anonymous_access(self, admin_client, client):
        """Public file accessible by anonymous user."""
        rid = self._create_resource_with_access(admin_client, ["public"])
        # Use a fresh client (no login)
        from fastapi.testclient import TestClient
        from main import app
        anon = TestClient(app)
        resp = anon.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_all_competitors_logged_in(self, admin_client, regular_client):
        """all_competitors: logged-in user can access."""
        rid = self._create_resource_with_access(admin_client, ["all_competitors"])
        # Need a separate regular user client
        from database import get_db
        # regular_client is already logged in as W2USR (via fixture creation)
        # But we need W2USR to exist — create via fresh signup
        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W5REG",
            "password": "password123",
            "email": "w5@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_all_competitors_anonymous_denied(self, admin_client):
        """all_competitors: anonymous user denied."""
        rid = self._create_resource_with_access(admin_client, ["all_competitors"])
        from fastapi.testclient import TestClient
        from main import app
        anon = TestClient(app)
        resp = anon.get(f"/resources/{rid}/download", follow_redirects=False)
        assert resp.status_code in (303, 403)

    def test_admin_only_admin_yes(self, admin_client):
        """admin: admin user can access."""
        rid = self._create_resource_with_access(admin_client, ["admin"])
        resp = admin_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_admin_only_regular_no(self, admin_client):
        """admin: regular user cannot access."""
        rid = self._create_resource_with_access(admin_client, ["admin"])
        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W6REG",
            "password": "password123",
            "email": "w6@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 403

    def test_referee_only_referee_yes(self, admin_client, referee_client):
        """referee: referee user can access."""
        rid = self._create_resource_with_access(admin_client, ["referee"])
        resp = referee_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_referee_only_regular_no(self, admin_client):
        """referee: regular user cannot access."""
        rid = self._create_resource_with_access(admin_client, ["referee"])
        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W7REG",
            "password": "password123",
            "email": "w7@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 403

    def test_sport_entered_user_yes(self, admin_client):
        """sport: user entered in the sport can access."""
        from database import get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active) VALUES ('Test', '2026-01-01', '2026-12-31', 0, 1)"
            )
            conn.execute(
                "INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'DX Challenge', 'continent')"
            )
        rid = self._create_resource_with_access(admin_client, ["sport"], sport_ids=[1])

        # Create user and enter them in the sport
        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W8SPT",
            "password": "password123",
            "email": "w8@example.com",
            "qrz_api_key": "test-api-key"
        })
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES ('W8SPT', 1, ?)",
                (datetime.utcnow().isoformat(),)
            )
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_sport_non_entered_user_no(self, admin_client):
        """sport: user NOT entered in the sport cannot access."""
        from database import get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active) VALUES ('Test', '2026-01-01', '2026-12-31', 0, 1)"
            )
            conn.execute(
                "INSERT INTO sports (olympiad_id, name, target_type) VALUES (1, 'DX Challenge', 'continent')"
            )
        rid = self._create_resource_with_access(admin_client, ["sport"], sport_ids=[1])

        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W9OUT",
            "password": "password123",
            "email": "w9@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 403

    def test_individual_named_user_yes(self, admin_client):
        """individual: named user can access."""
        rid = self._create_resource_with_access(admin_client, ["individual"], callsigns="KJ5IRF")

        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "KJ5IRF",
            "password": "password123",
            "email": "kj5@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200

    def test_individual_other_user_no(self, admin_client):
        """individual: other user cannot access."""
        rid = self._create_resource_with_access(admin_client, ["individual"], callsigns="KJ5IRF")

        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "N0OTH",
            "password": "password123",
            "email": "n0@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 403

    def test_combined_rules_or_logic(self, admin_client):
        """Combined rules: matching any single rule grants access."""
        rid = self._create_resource_with_access(
            admin_client,
            ["referee", "individual"],
            callsigns="KA1XYZ"
        )
        # KA1XYZ is named individually, not a referee — should still get access
        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "KA1XYZ",
            "password": "password123",
            "email": "ka1@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rid}/download")
        assert resp.status_code == 200


class TestResourcePublicPage:
    """Test public resources page."""

    def test_page_loads(self, admin_client):
        """Resources page loads with 200."""
        resp = admin_client.get("/resources")
        assert resp.status_code == 200

    def test_shows_accessible_files(self, admin_client):
        """Logged-in user sees files they can access."""
        _upload_file(admin_client, title="Public Rules", access_types=["public"])
        resp = admin_client.get("/resources")
        assert resp.status_code == 200
        assert "Public Rules" in resp.text

    def test_anonymous_sees_only_public(self, admin_client):
        """Anonymous user sees only public files."""
        _upload_file(admin_client, title="Public Doc", access_types=["public"])
        _upload_file(admin_client, title="Admin Only Doc", access_types=["admin"])

        from fastapi.testclient import TestClient
        from main import app
        anon = TestClient(app)
        resp = anon.get("/resources")
        assert resp.status_code == 200
        assert "Public Doc" in resp.text
        assert "Admin Only Doc" not in resp.text

    def test_logged_in_sees_public_plus_own(self, admin_client):
        """Logged-in user sees public + their accessible files."""
        _upload_file(admin_client, title="For Everyone", access_types=["public"])
        _upload_file(admin_client, title="Competitors Only", access_types=["all_competitors"])

        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W4VIS",
            "password": "password123",
            "email": "w4@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get("/resources")
        assert "For Everyone" in resp.text
        assert "Competitors Only" in resp.text

    def test_empty_state(self, client):
        """Empty state when no accessible files."""
        resp = client.get("/resources")
        assert resp.status_code == 200
        assert "No resources available" in resp.text


class TestResourceDownload:
    """Test resource file download."""

    def test_accessible_file_200(self, admin_client):
        """Accessible file returns 200 with correct headers."""
        _upload_file(admin_client, title="Download Me", access_types=["public"])
        from database import get_db
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()
        resp = admin_client.get(f"/resources/{rf['id']}/download")
        assert resp.status_code == 200
        assert "test.pdf" in resp.headers.get("content-disposition", "")

    def test_inaccessible_file_403(self, admin_client):
        """Inaccessible file returns 403 for logged-in user."""
        _upload_file(admin_client, title="Secret", access_types=["individual"], callsigns="NOBODY")
        from database import get_db
        with get_db() as conn:
            rf = conn.execute("SELECT id FROM resource_files").fetchone()

        from fastapi.testclient import TestClient
        from main import app
        user_client = TestClient(app)
        user_client.post("/signup", json={
            "callsign": "W5DEN",
            "password": "password123",
            "email": "w5d@example.com",
            "qrz_api_key": "test-api-key"
        })
        resp = user_client.get(f"/resources/{rf['id']}/download")
        assert resp.status_code == 403

    def test_missing_file_404(self, admin_client):
        """Non-existent resource returns 404."""
        resp = admin_client.get("/resources/9999/download")
        assert resp.status_code == 404


class TestFormatFileSize:
    """Test format_file_size Jinja2 filter."""

    def test_bytes(self):
        from main import format_file_size
        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        from main import format_file_size
        assert format_file_size(2458) == "2.4 KB"

    def test_megabytes(self):
        from main import format_file_size
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_zero(self):
        from main import format_file_size
        assert format_file_size(0) == "0 B"

    def test_exactly_1kb(self):
        from main import format_file_size
        assert format_file_size(1024) == "1.0 KB"

    def test_large_mb(self):
        from main import format_file_size
        assert format_file_size(10 * 1024 * 1024) == "10.0 MB"

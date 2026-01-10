"""
Tests for audit logging and admin features - TDD: Tests written before implementation.
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta

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
def admin_client(client):
    """Create test client with logged in admin user."""
    from database import get_db

    # Signup auto-logs in
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


class TestAuditModule:
    """Test audit logging module."""

    def test_audit_module_exists(self):
        """Test that audit module exists."""
        import audit
        assert audit is not None

    def test_log_action_function_exists(self):
        """Test that log_action function exists."""
        from audit import log_action
        assert callable(log_action)

    def test_log_action_creates_entry(self):
        """Test log_action creates audit log entry."""
        from audit import log_action, get_audit_logs
        from database import get_db

        log_action(
            actor_callsign="W1ADM",
            action="test_action",
            target_type="test",
            target_id="123",
            details="Test details",
            ip_address="127.0.0.1"
        )

        logs = get_audit_logs(limit=1)
        assert len(logs) == 1
        assert logs[0]["actor_callsign"] == "W1ADM"
        assert logs[0]["action"] == "test_action"
        assert logs[0]["target_type"] == "test"
        assert logs[0]["target_id"] == "123"
        assert logs[0]["details"] == "Test details"
        assert logs[0]["ip_address"] == "127.0.0.1"

    def test_log_action_stores_timestamp(self):
        """Test log_action stores timestamp."""
        from audit import log_action, get_audit_logs

        log_action(
            actor_callsign="W1ADM",
            action="test_action"
        )

        logs = get_audit_logs(limit=1)
        assert logs[0]["timestamp"] is not None
        # Should be recent (within last minute)
        timestamp = datetime.fromisoformat(logs[0]["timestamp"])
        assert datetime.utcnow() - timestamp < timedelta(minutes=1)

    def test_log_action_optional_fields(self):
        """Test log_action works with optional fields."""
        from audit import log_action, get_audit_logs

        # Minimal call - only required fields
        log_action(
            actor_callsign="W1ADM",
            action="minimal_action"
        )

        logs = get_audit_logs(limit=1)
        assert logs[0]["action"] == "minimal_action"
        assert logs[0]["target_type"] is None
        assert logs[0]["target_id"] is None


class TestAuditLogTable:
    """Test audit log database table."""

    def test_audit_log_table_exists(self):
        """Test audit_log table exists in database."""
        from database import get_db

        with get_db() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='audit_log'
            """)
            result = cursor.fetchone()
            assert result is not None

    def test_audit_log_table_has_required_columns(self):
        """Test audit_log table has all required columns."""
        from database import get_db

        with get_db() as conn:
            cursor = conn.execute("PRAGMA table_info(audit_log)")
            columns = {row[1] for row in cursor.fetchall()}

        required = {"id", "timestamp", "actor_callsign", "action",
                    "target_type", "target_id", "details", "ip_address"}
        assert required.issubset(columns)


class TestGetAuditLogs:
    """Test retrieving audit logs."""

    def test_get_audit_logs_function_exists(self):
        """Test get_audit_logs function exists."""
        from audit import get_audit_logs
        assert callable(get_audit_logs)

    def test_get_audit_logs_returns_list(self):
        """Test get_audit_logs returns a list."""
        from audit import get_audit_logs

        logs = get_audit_logs()
        assert isinstance(logs, list)

    def test_get_audit_logs_with_limit(self):
        """Test get_audit_logs respects limit."""
        from audit import log_action, get_audit_logs

        # Create multiple entries
        for i in range(10):
            log_action(actor_callsign="W1ADM", action=f"action_{i}")

        logs = get_audit_logs(limit=5)
        assert len(logs) == 5

    def test_get_audit_logs_ordered_by_timestamp_desc(self):
        """Test logs are ordered by timestamp descending."""
        from audit import log_action, get_audit_logs
        import time

        log_action(actor_callsign="W1ADM", action="first")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        log_action(actor_callsign="W1ADM", action="second")

        logs = get_audit_logs(limit=2)
        assert logs[0]["action"] == "second"  # Most recent first
        assert logs[1]["action"] == "first"

    def test_get_audit_logs_filter_by_action(self):
        """Test filtering logs by action type."""
        from audit import log_action, get_audit_logs

        log_action(actor_callsign="W1ADM", action="login")
        log_action(actor_callsign="W1ADM", action="logout")
        log_action(actor_callsign="W1ADM", action="login")

        logs = get_audit_logs(action="login")
        assert len(logs) == 2
        assert all(log["action"] == "login" for log in logs)

    def test_get_audit_logs_filter_by_actor(self):
        """Test filtering logs by actor callsign."""
        from audit import log_action, get_audit_logs

        log_action(actor_callsign="W1ADM", action="action1")
        log_action(actor_callsign="W2USR", action="action2")
        log_action(actor_callsign="W1ADM", action="action3")

        logs = get_audit_logs(actor_callsign="W1ADM")
        assert len(logs) == 2
        assert all(log["actor_callsign"] == "W1ADM" for log in logs)

    def test_get_audit_logs_filter_by_target_type(self):
        """Test filtering logs by target type."""
        from audit import log_action, get_audit_logs

        log_action(actor_callsign="W1ADM", action="action1", target_type="competitor")
        log_action(actor_callsign="W1ADM", action="action2", target_type="sport")
        log_action(actor_callsign="W1ADM", action="action3", target_type="competitor")

        logs = get_audit_logs(target_type="competitor")
        assert len(logs) == 2
        assert all(log["target_type"] == "competitor" for log in logs)

    def test_get_audit_logs_with_offset(self):
        """Test get_audit_logs respects offset."""
        from audit import log_action, get_audit_logs

        # Create multiple entries
        for i in range(10):
            log_action(actor_callsign="W1ADM", action=f"action_{i}")

        logs = get_audit_logs(limit=5, offset=5)
        assert len(logs) == 5
        # Should skip first 5 (most recent) and return next 5


class TestGetAuditLogCount:
    """Test audit log count function."""

    def test_get_audit_log_count_function_exists(self):
        """Test get_audit_log_count function exists."""
        from audit import get_audit_log_count
        assert callable(get_audit_log_count)

    def test_get_audit_log_count_returns_int(self):
        """Test get_audit_log_count returns an integer."""
        from audit import get_audit_log_count

        count = get_audit_log_count()
        assert isinstance(count, int)

    def test_get_audit_log_count_all_entries(self):
        """Test get_audit_log_count counts all entries."""
        from audit import log_action, get_audit_log_count

        initial = get_audit_log_count()
        for i in range(5):
            log_action(actor_callsign="W1ADM", action=f"action_{i}")

        count = get_audit_log_count()
        assert count == initial + 5

    def test_get_audit_log_count_filter_by_action(self):
        """Test get_audit_log_count filtering by action."""
        from audit import log_action, get_audit_log_count

        log_action(actor_callsign="W1ADM", action="login")
        log_action(actor_callsign="W1ADM", action="logout")
        log_action(actor_callsign="W1ADM", action="login")

        count = get_audit_log_count(action="login")
        assert count == 2

    def test_get_audit_log_count_filter_by_actor(self):
        """Test get_audit_log_count filtering by actor callsign."""
        from audit import log_action, get_audit_log_count

        log_action(actor_callsign="W1CNT", action="action1")
        log_action(actor_callsign="W2CNT", action="action2")
        log_action(actor_callsign="W1CNT", action="action3")

        count = get_audit_log_count(actor_callsign="W1CNT")
        assert count == 2

    def test_get_audit_log_count_filter_by_target_type(self):
        """Test get_audit_log_count filtering by target type."""
        from audit import log_action, get_audit_log_count

        log_action(actor_callsign="W1ADM", action="action1", target_type="competitor")
        log_action(actor_callsign="W1ADM", action="action2", target_type="sport")
        log_action(actor_callsign="W1ADM", action="action3", target_type="competitor")

        count = get_audit_log_count(target_type="competitor")
        assert count == 2


class TestAuditLogEndpoint:
    """Test audit log admin endpoint."""

    def test_audit_log_endpoint_exists(self, admin_client):
        """Test /admin/audit-log endpoint exists."""
        response = admin_client.get("/admin/audit-log")
        assert response.status_code != 404

    def test_audit_log_requires_admin(self, client):
        """Test audit log requires admin access."""
        # Login as regular user
        client.post("/signup", json={
            "callsign": "W1USR",
            "password": "password123",
            "qrz_api_key": "test-key"
        })

        response = client.get("/admin/audit-log")
        assert response.status_code in [401, 403]

    def test_audit_log_displays_entries(self, admin_client):
        """Test audit log page displays log entries."""
        from audit import log_action

        log_action(
            actor_callsign="W1ADM",
            action="test_display",
            details="Should appear on page"
        )

        response = admin_client.get("/admin/audit-log")
        assert response.status_code == 200
        assert "test_display" in response.text

    def test_audit_log_filter_by_action(self, admin_client):
        """Test audit log filtering by action."""
        from audit import log_action

        log_action(actor_callsign="W1ADM", action="login")
        log_action(actor_callsign="W1ADM", action="password_change")

        response = admin_client.get("/admin/audit-log?action=login")
        assert response.status_code == 200
        assert "login" in response.text


class TestActionLogging:
    """Test that specific actions are logged."""

    def test_login_is_logged(self, client):
        """Test successful login is logged."""
        from audit import get_audit_logs
        from auth import register_user

        # Create user directly
        register_user("W1LOG", "password123")

        # Login explicitly
        client.post("/login", json={
            "callsign": "W1LOG",
            "password": "password123"
        })

        logs = get_audit_logs(action="login")
        assert len(logs) >= 1
        assert any(log["actor_callsign"] == "W1LOG" for log in logs)

    def test_logout_is_logged(self, client):
        """Test logout is logged."""
        from audit import get_audit_logs

        # Create user, login, then logout
        client.post("/signup", json={
            "callsign": "W1OUT",
            "password": "password123",
            "qrz_api_key": "test-key"
        })
        client.post("/logout")

        logs = get_audit_logs(action="logout")
        assert len(logs) >= 1

    def test_password_change_is_logged(self, client):
        """Test password change is logged."""
        from audit import get_audit_logs

        # Create user
        client.post("/signup", json={
            "callsign": "W1PWD",
            "password": "password123",
            "qrz_api_key": "test-key"
        })

        # Change password using form data
        client.post("/settings/password", data={
            "current_password": "password123",
            "new_password": "newpassword456"
        })

        logs = get_audit_logs(action="password_change")
        assert len(logs) >= 1

    def test_admin_action_is_logged(self, admin_client):
        """Test admin actions are logged."""
        from audit import get_audit_logs
        from auth import register_user

        register_user("W1TGT", "password123")

        # Disable a user
        admin_client.post("/admin/competitor/W1TGT/disable")

        logs = get_audit_logs(action="competitor_disabled")
        assert len(logs) >= 1
        assert any(log["target_id"] == "W1TGT" for log in logs)


class TestBulkOperations:
    """Test bulk admin operations."""

    def test_bulk_disable_endpoint_exists(self, admin_client):
        """Test bulk disable endpoint exists."""
        response = admin_client.post("/admin/competitors/bulk-disable", json={
            "callsigns": []
        })
        assert response.status_code != 404

    def test_bulk_disable_multiple_competitors(self, admin_client):
        """Test bulk disabling multiple competitors."""
        from auth import register_user
        from database import get_db

        register_user("W1BLK1", "password123")
        register_user("W1BLK2", "password123")
        register_user("W1BLK3", "password123")

        response = admin_client.post("/admin/competitors/bulk-disable", json={
            "callsigns": ["W1BLK1", "W1BLK2"]
        })

        assert response.status_code == 200

        # Verify they're disabled
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT is_disabled FROM competitors WHERE callsign IN (?, ?)",
                ("W1BLK1", "W1BLK2")
            )
            results = cursor.fetchall()
            assert all(r["is_disabled"] == 1 for r in results)

            # W1BLK3 should NOT be disabled
            cursor = conn.execute(
                "SELECT is_disabled FROM competitors WHERE callsign = ?",
                ("W1BLK3",)
            )
            assert cursor.fetchone()["is_disabled"] == 0

    def test_bulk_enable_endpoint_exists(self, admin_client):
        """Test bulk enable endpoint exists."""
        response = admin_client.post("/admin/competitors/bulk-enable", json={
            "callsigns": []
        })
        assert response.status_code != 404

    def test_bulk_enable_multiple_competitors(self, admin_client):
        """Test bulk enabling multiple competitors."""
        from auth import register_user
        from database import get_db

        register_user("W1EN1", "password123")
        register_user("W1EN2", "password123")

        # First disable them
        with get_db() as conn:
            conn.execute(
                "UPDATE competitors SET is_disabled = 1 WHERE callsign IN (?, ?)",
                ("W1EN1", "W1EN2")
            )

        response = admin_client.post("/admin/competitors/bulk-enable", json={
            "callsigns": ["W1EN1", "W1EN2"]
        })

        assert response.status_code == 200

        # Verify they're enabled
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT is_disabled FROM competitors WHERE callsign IN (?, ?)",
                ("W1EN1", "W1EN2")
            )
            results = cursor.fetchall()
            assert all(r["is_disabled"] == 0 for r in results)

    def test_bulk_operations_require_admin(self, client):
        """Test bulk operations require admin access."""
        # Login as regular user
        client.post("/signup", json={
            "callsign": "W1REG",
            "password": "password123",
            "qrz_api_key": "test-key"
        })

        response = client.post("/admin/competitors/bulk-disable", json={
            "callsigns": ["W1ABC"]
        })
        assert response.status_code in [401, 403]

    def test_bulk_operations_logged(self, admin_client):
        """Test bulk operations are logged."""
        from auth import register_user
        from audit import get_audit_logs

        register_user("W1AUD1", "password123")
        register_user("W1AUD2", "password123")

        admin_client.post("/admin/competitors/bulk-disable", json={
            "callsigns": ["W1AUD1", "W1AUD2"]
        })

        logs = get_audit_logs(action="bulk_disable")
        assert len(logs) >= 1


class TestDatabaseBackup:
    """Test database backup functionality."""

    def test_backup_endpoint_exists(self, admin_client):
        """Test backup endpoint exists."""
        response = admin_client.get("/admin/backup")
        assert response.status_code != 404

    def test_backup_requires_admin(self, client):
        """Test backup requires admin access."""
        client.post("/signup", json={
            "callsign": "W1BAK",
            "password": "password123",
            "qrz_api_key": "test-key"
        })

        response = client.get("/admin/backup")
        assert response.status_code in [401, 403]

    def test_backup_returns_sqlite_file(self, admin_client):
        """Test backup returns a SQLite database file."""
        response = admin_client.get("/admin/backup")

        assert response.status_code == 200
        assert "application/octet-stream" in response.headers.get("content-type", "") or \
               "application/x-sqlite3" in response.headers.get("content-type", "")

    def test_backup_has_download_filename(self, admin_client):
        """Test backup has proper download filename."""
        response = admin_client.get("/admin/backup")

        disposition = response.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert ".db" in disposition or ".sqlite" in disposition

    def test_backup_is_logged(self, admin_client):
        """Test backup action is logged."""
        from audit import get_audit_logs

        admin_client.get("/admin/backup")

        logs = get_audit_logs(action="database_backup")
        assert len(logs) >= 1


class TestAdminDashboardEnhancements:
    """Test admin dashboard enhancements."""

    def test_admin_dashboard_shows_recent_audit(self, admin_client):
        """Test admin dashboard shows recent audit entries."""
        from audit import log_action

        log_action(
            actor_callsign="W1ADM",
            action="test_dashboard_audit",
            details="Recent activity"
        )

        response = admin_client.get("/admin")
        assert response.status_code == 200
        # Should show recent activity on dashboard or link to audit log
        assert "test_dashboard_audit" in response.text or "audit" in response.text.lower()

    def test_admin_dashboard_shows_competitor_count(self, admin_client):
        """Test admin dashboard shows competitor count."""
        response = admin_client.get("/admin")
        assert response.status_code == 200
        # Already tested in existing tests, just verify it's still there
        assert "competitor" in response.text.lower()

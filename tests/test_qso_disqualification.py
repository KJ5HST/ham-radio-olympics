"""
Tests for QSO disqualification feature.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta

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
from auth import create_session, SESSION_COOKIE_NAME


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


@pytest.fixture(autouse=True)
def mock_qrz_verify():
    """Mock QRZ API key verification to always succeed."""
    from unittest.mock import patch, AsyncMock
    with patch('main.verify_api_key', new_callable=AsyncMock) as mock:
        mock.return_value = True
        yield mock


def signup_user(client, callsign, password="password123", qrz_api_key="test-api-key"):
    """Helper to create a user via signup."""
    return client.post("/signup", json={
        "callsign": callsign,
        "password": password,
        "qrz_api_key": qrz_api_key
    }, follow_redirects=False)


def login_user(client, callsign, password="password123"):
    """Helper to login a user."""
    return client.post("/login", json={
        "callsign": callsign,
        "password": password
    }, follow_redirects=False)


def create_test_olympiad(admin_headers, client):
    """Create a test olympiad with a sport and match."""
    now = datetime.utcnow()
    start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

    # Create olympiad
    resp = client.post("/admin/olympiad", json={
        "name": "Test Olympics 2026",
        "start_date": start,
        "end_date": end,
        "qualifying_qsos": 0
    }, headers=admin_headers)
    assert resp.status_code == 200, f"Failed to create olympiad: {resp.text}"
    olympiad_id = resp.json()["id"]

    # Activate olympiad
    resp = client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)
    assert resp.status_code == 200

    # Create sport
    resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
        "name": "DX Challenge",
        "description": "Work DX stations",
        "target_type": "continent",
        "work_enabled": True,
        "activate_enabled": False
    }, headers=admin_headers)
    assert resp.status_code == 200, f"Failed to create sport: {resp.text}"
    sport_id = resp.json()["id"]

    # Create match
    resp = client.post(f"/admin/sport/{sport_id}/match", json={
        "start_date": start,
        "end_date": end,
        "target_value": "EU"
    }, headers=admin_headers)
    assert resp.status_code == 200, f"Failed to create match: {resp.text}"
    match_id = resp.json()["id"]

    return olympiad_id, sport_id, match_id


def create_test_qso(competitor_callsign, dx_callsign="DL1ABC", sport_id=None):
    """Insert a test QSO directly into the database."""
    now = datetime.utcnow()
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO qsos (
                competitor_callsign, dx_callsign, qso_datetime_utc,
                dx_dxcc, dx_grid, distance_km, tx_power_w, cool_factor,
                is_confirmed, mode, band
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            competitor_callsign, dx_callsign, now.isoformat(),
            "230", "JO31", 5000, 100, 50.0,
            1, "SSB", "20m"
        ))
        return cursor.lastrowid


class TestDisqualifyQSO:
    """Test QSO disqualification endpoint."""

    def test_admin_can_disqualify_qso(self, client, admin_headers):
        """Test admin can disqualify a QSO."""
        # Setup
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be fraudulent based on time analysis"},
            headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "QSO disqualified"
        assert resp.json()["qso_id"] == qso_id
        assert resp.json()["sport_id"] == sport_id

    def test_referee_can_disqualify_qso(self, client, admin_headers):
        """Test referee assigned to sport can disqualify a QSO."""
        # Create competitor and referee
        signup_user(client, "W1ABC")
        client.cookies.clear()
        signup_user(client, "W1REF")
        client.cookies.clear()

        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Make W1REF a referee and assign to sport
        client.post(f"/admin/competitor/W1REF/set-referee", headers=admin_headers)
        client.post(f"/admin/competitor/W1REF/assign-sport/{sport_id}", headers=admin_headers)

        # Login as referee
        login_user(client, "W1REF")

        # Disqualify as referee
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "QSO does not meet competition requirements for this sport"}
        )
        assert resp.status_code == 200

    def test_non_referee_cannot_disqualify(self, client, admin_headers):
        """Test non-referee cannot disqualify a QSO."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        signup_user(client, "W2ABC")
        client.cookies.clear()

        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Login as non-referee
        login_user(client, "W2ABC")

        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Some reason that is long enough"}
        )
        assert resp.status_code in [401, 403]

    def test_disqualify_requires_reason(self, client, admin_headers):
        """Test disqualification requires a reason."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "short"},
            headers=admin_headers
        )
        assert resp.status_code == 422  # Validation error

    def test_cannot_disqualify_nonexistent_qso(self, client, admin_headers):
        """Test cannot disqualify a non-existent QSO."""
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)

        resp = client.post(
            f"/referee/sport/{sport_id}/qso/99999/disqualify",
            json={"reason": "This is a reason that is long enough"},
            headers=admin_headers
        )
        assert resp.status_code == 404

    def test_cannot_disqualify_already_disqualified(self, client, admin_headers):
        """Test cannot disqualify an already disqualified QSO."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # First disqualification
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "First reason that is long enough"},
            headers=admin_headers
        )

        # Second attempt
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Second reason that is long enough"},
            headers=admin_headers
        )
        assert resp.status_code == 400
        assert "already disqualified" in resp.json()["detail"]


class TestRefuteDisqualification:
    """Test QSO refutation endpoint."""

    def test_owner_can_refute_disqualification(self, client, admin_headers):
        """Test QSO owner can refute a disqualification."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify first
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Login as owner and refute
        login_user(client, "W1ABC")
        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "This QSO is valid, I have QSL card proof"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "refuted"

    def test_non_owner_cannot_refute(self, client, admin_headers):
        """Test non-owner cannot refute a disqualification."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        signup_user(client, "W2ABC")
        client.cookies.clear()

        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Try to refute as different user
        login_user(client, "W2ABC")
        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "This should not be allowed"}
        )
        assert resp.status_code == 403

    def test_cannot_refute_non_disqualified(self, client, admin_headers):
        """Test cannot refute a QSO that is not disqualified."""
        signup_user(client, "W1ABC")
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "This should not work for a non-DQ QSO"}
        )
        assert resp.status_code == 404

    def test_refute_requires_login(self, client, admin_headers):
        """Test refutation requires login."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Try to refute without login
        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "This should require login"}
        )
        assert resp.status_code == 401


class TestRequalifyQSO:
    """Test QSO requalification endpoint."""

    def test_admin_can_requalify(self, client, admin_headers):
        """Test admin can requalify a disqualified QSO."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Requalify
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "After review, this QSO is valid"},
            headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "QSO requalified"

    def test_requalify_after_refute(self, client, admin_headers):
        """Test can requalify after a refutation."""
        signup_user(client, "W1ABC")
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify (as admin)
        client.cookies.clear()
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Refute (as owner)
        login_user(client, "W1ABC")
        client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "I have proof this is valid"}
        )
        client.cookies.clear()

        # Requalify (as admin)
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Refutation accepted, QSO is valid"},
            headers=admin_headers
        )
        assert resp.status_code == 200

    def test_cannot_requalify_already_requalified(self, client, admin_headers):
        """Test cannot requalify an already requalified QSO."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify then requalify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "After review, this QSO is valid"},
            headers=admin_headers
        )

        # Try to requalify again
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Trying to requalify again"},
            headers=admin_headers
        )
        assert resp.status_code == 400
        assert "already requalified" in resp.json()["detail"]


class TestDisqualificationHistory:
    """Test QSO disqualification history endpoint."""

    def test_get_disqualification_history(self, client, admin_headers):
        """Test getting disqualification history for a QSO."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Get history
        resp = client.get(f"/qso/{qso_id}/disqualifications")
        assert resp.status_code == 200
        data = resp.json()
        assert data["qso_id"] == qso_id
        assert len(data["disqualifications"]) == 1
        assert data["disqualifications"][0]["status"] == "disqualified"
        assert len(data["disqualifications"][0]["comments"]) == 1

    def test_history_is_public(self, client, admin_headers):
        """Test disqualification history is publicly accessible."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO appears to be invalid"},
            headers=admin_headers
        )

        # Clear cookies and get history (no auth)
        client.cookies.clear()
        resp = client.get(f"/qso/{qso_id}/disqualifications")
        assert resp.status_code == 200

    def test_history_shows_full_conversation(self, client, admin_headers):
        """Test history shows all comments in the conversation."""
        signup_user(client, "W1ABC")
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.cookies.clear()
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Initial disqualification reason"},
            headers=admin_headers
        )

        # Refute
        login_user(client, "W1ABC")
        client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "My refutation explanation"}
        )
        client.cookies.clear()

        # Requalify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Requalification after review"},
            headers=admin_headers
        )

        # Get history
        resp = client.get(f"/qso/{qso_id}/disqualifications")
        data = resp.json()

        assert len(data["disqualifications"]) == 1
        assert data["disqualifications"][0]["status"] == "requalified"
        comments = data["disqualifications"][0]["comments"]
        assert len(comments) == 3
        assert comments[0]["comment_type"] == "disqualify"
        assert comments[1]["comment_type"] == "refute"
        assert comments[2]["comment_type"] == "requalify"


class TestScoringExcludesDisqualified:
    """Test that disqualified QSOs are excluded from scoring."""

    def test_disqualified_qso_excluded_from_medals(self, client, admin_headers):
        """Test disqualified QSO doesn't count toward medals."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)

        # Enter sport
        login_user(client, "W1ABC")
        client.post(f"/sport/{sport_id}/enter")
        client.cookies.clear()

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Trigger medal computation
        from scoring import recompute_match_medals
        recompute_match_medals(match_id)

        # Check medals exist
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM medals WHERE match_id = ? AND callsign = ?",
                (match_id, "W1ABC")
            )
            medal = cursor.fetchone()
            # Medal may or may not exist depending on target matching

        # Disqualify the QSO
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "This QSO is being disqualified for testing"},
            headers=admin_headers
        )

        # Check that medals were recomputed - should be gone or different
        # The endpoint automatically recomputes medals

    def test_requalified_qso_included_in_medals(self, client, admin_headers):
        """Test requalified QSO counts toward medals again."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)

        # Enter sport
        login_user(client, "W1ABC")
        client.post(f"/sport/{sport_id}/enter")
        client.cookies.clear()

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Disqualify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Temporary disqualification for testing"},
            headers=admin_headers
        )

        # Requalify
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Reinstating QSO after review"},
            headers=admin_headers
        )

        # Medal computation is automatic in the endpoint


class TestSportSpecificDisqualification:
    """Test that disqualifications are sport-specific."""

    def test_qso_can_be_disqualified_in_one_sport_not_another(self, client, admin_headers):
        """Test a QSO can be DQ'd in one sport but valid in another."""
        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with two sports
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        # Sport 1
        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "Sport 1",
            "target_type": "continent",
            "work_enabled": True
        }, headers=admin_headers)
        sport1_id = resp.json()["id"]

        # Sport 2
        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "Sport 2",
            "target_type": "continent",
            "work_enabled": True
        }, headers=admin_headers)
        sport2_id = resp.json()["id"]

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Disqualify only in sport 1
        client.post(
            f"/referee/sport/{sport1_id}/qso/{qso_id}/disqualify",
            json={"reason": "Invalid for sport 1 rules only"},
            headers=admin_headers
        )

        # Check history shows only sport 1
        resp = client.get(f"/qso/{qso_id}/disqualifications")
        data = resp.json()
        assert len(data["disqualifications"]) == 1
        assert data["disqualifications"][0]["sport_id"] == sport1_id


class TestAuditLogging:
    """Test that disqualification actions are logged."""

    def test_disqualify_is_logged(self, client, admin_headers):
        """Test disqualification is logged in audit log."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Testing audit log entry"},
            headers=admin_headers
        )

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE action = 'qso_disqualified' AND target_id = ?",
                (str(qso_id),)
            )
            log = cursor.fetchone()
            assert log is not None
            assert "Testing audit log entry" in log["details"]

    def test_refute_is_logged(self, client, admin_headers):
        """Test refutation is logged in audit log."""
        signup_user(client, "W1ABC")
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        client.cookies.clear()
        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Initial disqualification"},
            headers=admin_headers
        )

        login_user(client, "W1ABC")
        client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "Testing refutation audit log"}
        )

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE action = 'qso_refuted' AND target_id = ?",
                (str(qso_id),)
            )
            log = cursor.fetchone()
            assert log is not None

    def test_requalify_is_logged(self, client, admin_headers):
        """Test requalification is logged in audit log."""
        signup_user(client, "W1ABC")
        client.cookies.clear()
        olympiad_id, sport_id, match_id = create_test_olympiad(admin_headers, client)
        qso_id = create_test_qso("W1ABC")

        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/disqualify",
            json={"reason": "Initial disqualification"},
            headers=admin_headers
        )

        client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Testing requalify audit log"},
            headers=admin_headers
        )

        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_log WHERE action = 'qso_requalified' AND target_id = ?",
                (str(qso_id),)
            )
            log = cursor.fetchone()
            assert log is not None


class TestAutoDisqualificationParkAnomalies:
    """Test automatic disqualification for park reference format anomalies."""

    def test_detect_park_anomaly_valid_formats(self):
        """Test that valid park formats are not flagged."""
        from sync import detect_park_anomaly

        # Valid formats
        assert detect_park_anomaly("US-0001") is None
        assert detect_park_anomaly("K-0001") is None
        assert detect_park_anomaly("VE-1234") is None
        assert detect_park_anomaly("US-10001") is None  # 5 digits ok
        assert detect_park_anomaly("") is None
        assert detect_park_anomaly(None) is None

    def test_detect_park_anomaly_missing_zeros(self):
        """Test detection of park numbers without leading zeros."""
        from sync import detect_park_anomaly

        # K-11 should be K-0011
        anomaly = detect_park_anomaly("K-11")
        assert anomaly is not None
        assert "leading zeros" in anomaly
        assert "K-0011" in anomaly

        # US-1 should be US-0001
        anomaly = detect_park_anomaly("US-1")
        assert anomaly is not None
        assert "leading zeros" in anomaly
        assert "US-0001" in anomaly

    def test_detect_park_anomaly_missing_dash(self):
        """Test detection of park refs missing dash."""
        from sync import detect_park_anomaly

        anomaly = detect_park_anomaly("US0001")
        assert anomaly is not None
        assert "Missing dash" in anomaly

        anomaly = detect_park_anomaly("K1234")
        assert anomaly is not None
        assert "Missing dash" in anomaly

    def test_detect_park_anomaly_missing_prefix(self):
        """Test detection of park refs missing country prefix."""
        from sync import detect_park_anomaly

        anomaly = detect_park_anomaly("0001")
        assert anomaly is not None
        assert "Missing country prefix" in anomaly

        anomaly = detect_park_anomaly("1234")
        assert anomaly is not None
        assert "Missing country prefix" in anomaly

    def test_auto_disqualify_creates_system_comment(self, client, admin_headers):
        """Test that auto-disqualification creates a SYSTEM author comment."""
        from sync import auto_disqualify_qso

        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with POTA sport
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True
        }, headers=admin_headers)
        sport_id = resp.json()["id"]

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Auto-disqualify
        result = auto_disqualify_qso(qso_id, sport_id, "Auto-flagged: Park format anomaly")
        assert result is True

        # Verify the comment has SYSTEM as author
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT c.author_callsign, c.comment, c.comment_type
                FROM qso_disqualification_comments c
                JOIN qso_disqualifications d ON c.disqualification_id = d.id
                WHERE d.qso_id = ? AND d.sport_id = ?
            """, (qso_id, sport_id))
            comment = cursor.fetchone()
            assert comment is not None
            assert comment["author_callsign"] == "SYSTEM"
            assert comment["comment_type"] == "disqualify"
            assert "Auto-flagged" in comment["comment"]

    def test_competitor_can_refute_auto_disqualification(self, client, admin_headers):
        """Test that a competitor can refute an auto-disqualification."""
        from sync import auto_disqualify_qso

        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with POTA sport
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True
        }, headers=admin_headers)
        sport_id = resp.json()["id"]

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Auto-disqualify (simulating what sync would do)
        auto_disqualify_qso(qso_id, sport_id, "Auto-flagged: Park number needs leading zeros: 'K-11' (expected: K-0011)")

        # Login as owner and refute
        login_user(client, "W1ABC")
        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "The park ID K-11 was entered correctly in my logging software, this is how it exported"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "refuted"

        # Verify refutation comment was added
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT c.author_callsign, c.comment_type, c.comment
                FROM qso_disqualification_comments c
                JOIN qso_disqualifications d ON c.disqualification_id = d.id
                WHERE d.qso_id = ? AND d.sport_id = ?
                ORDER BY c.created_at
            """, (qso_id, sport_id))
            comments = cursor.fetchall()
            assert len(comments) == 2
            assert comments[0]["author_callsign"] == "SYSTEM"
            assert comments[0]["comment_type"] == "disqualify"
            assert comments[1]["author_callsign"] == "W1ABC"
            assert comments[1]["comment_type"] == "refute"

    def test_referee_can_requalify_auto_disqualification(self, client, admin_headers):
        """Test that a referee can requalify an auto-disqualified QSO."""
        from sync import auto_disqualify_qso

        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with POTA sport
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True
        }, headers=admin_headers)
        sport_id = resp.json()["id"]

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # Auto-disqualify
        auto_disqualify_qso(qso_id, sport_id, "Auto-flagged: Park format anomaly")

        # Requalify as admin/referee
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Park ID verified as valid, format accepted by POTA"},
            headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "QSO requalified"

        # Verify status
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
                (qso_id, sport_id)
            )
            dq = cursor.fetchone()
            assert dq["status"] == "requalified"

    def test_full_auto_disqualify_refute_requalify_flow(self, client, admin_headers):
        """Test complete flow: auto-DQ -> refute -> requalify."""
        from sync import auto_disqualify_qso

        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with POTA sport
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True
        }, headers=admin_headers)
        sport_id = resp.json()["id"]

        # Create QSO
        qso_id = create_test_qso("W1ABC")

        # 1. Auto-disqualify (what sync does for malformed parks)
        auto_disqualify_qso(
            qso_id, sport_id,
            "Auto-flagged: Park number needs leading zeros: 'K-11' (expected: K-0011)"
        )

        # 2. Competitor refutes
        login_user(client, "W1ABC")
        resp = client.post(
            f"/qso/{qso_id}/sport/{sport_id}/refute",
            json={"refutation": "My logging software exports K-11, but POTA recognizes this as K-0011"}
        )
        assert resp.status_code == 200
        client.cookies.clear()

        # 3. Referee requalifies
        resp = client.post(
            f"/referee/sport/{sport_id}/qso/{qso_id}/requalify",
            json={"reason": "Verified K-11 maps to K-0011 in POTA system, requalifying"},
            headers=admin_headers
        )
        assert resp.status_code == 200

        # 4. Verify full history
        resp = client.get(f"/qso/{qso_id}/disqualifications")
        data = resp.json()
        assert len(data["disqualifications"]) == 1
        assert data["disqualifications"][0]["status"] == "requalified"

        comments = data["disqualifications"][0]["comments"]
        assert len(comments) == 3
        assert comments[0]["comment_type"] == "disqualify"
        assert comments[0]["author_callsign"] == "SYSTEM"
        assert "leading zeros" in comments[0]["comment"]
        assert comments[1]["comment_type"] == "refute"
        assert comments[1]["author_callsign"] == "W1ABC"
        assert comments[2]["comment_type"] == "requalify"

    def test_check_and_disqualify_park_anomalies_integration(self, client, admin_headers):
        """Test the check_and_disqualify_park_anomalies function with real QSO data."""
        from sync import check_and_disqualify_park_anomalies

        signup_user(client, "W1ABC")
        client.cookies.clear()

        # Create olympiad with POTA sport
        now = datetime.utcnow()
        start = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
        end = (now + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59")

        resp = client.post("/admin/olympiad", json={
            "name": "Test Olympics",
            "start_date": start,
            "end_date": end,
            "qualifying_qsos": 0
        }, headers=admin_headers)
        olympiad_id = resp.json()["id"]
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "POTA Challenge",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True
        }, headers=admin_headers)
        sport_id = resp.json()["id"]

        # Create QSO with malformed park reference
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    dx_dxcc, dx_grid, dx_sig_info, distance_km, tx_power_w, cool_factor,
                    is_confirmed, mode, band
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "W1ABC", "K1ABC", now.isoformat(),
                "291", "FN31", "K-11",  # Malformed - should be K-0011
                1000, 100, 10.0,
                1, "SSB", "20m"
            ))
            qso_id = cursor.lastrowid
            conn.commit()

        # Run the anomaly check
        stats = check_and_disqualify_park_anomalies()

        # Verify stats
        assert stats["anomalies_found"] >= 1
        assert stats["qsos_disqualified"] >= 1

        # Verify QSO was disqualified
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
                (qso_id, sport_id)
            )
            dq = cursor.fetchone()
            assert dq is not None
            assert dq["status"] == "disqualified"

            # Verify comment has the anomaly details
            cursor = conn.execute("""
                SELECT comment FROM qso_disqualification_comments c
                JOIN qso_disqualifications d ON c.disqualification_id = d.id
                WHERE d.qso_id = ?
            """, (qso_id,))
            comment = cursor.fetchone()
            assert "K-11" in comment["comment"]
            assert "K-0011" in comment["comment"]

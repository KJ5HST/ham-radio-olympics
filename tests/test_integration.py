"""
Integration tests for full workflow.
"""

import pytest
import os
import tempfile
from datetime import datetime

_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)

os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"
os.environ["TESTING"] = "1"

from fastapi.testclient import TestClient
from main import app
from database import reset_db, get_db
from scoring import recompute_match_medals, get_matching_qsos, update_records


def signup_user(client, callsign, password="password123", qrz_api_key="test-api-key"):
    """Helper to create a user via signup."""
    return client.post("/signup", json={
        "callsign": callsign,
        "password": password,
        "qrz_api_key": qrz_api_key
    })


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_headers():
    return {"X-Admin-Key": "test-admin-key"}


class TestFullWorkflow:
    """Test complete workflow from setup to scoring."""

    def test_complete_competition_workflow(self, client, admin_headers):
        """Test full workflow: create olympiad, sport, match, register, add QSO, score."""
        # 1. Create Olympiad
        resp = client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        assert resp.status_code == 200
        olympiad_id = resp.json()["id"]

        # 2. Activate Olympiad
        client.post(f"/admin/olympiad/{olympiad_id}/activate", headers=admin_headers)

        # 3. Create Sport
        resp = client.post(f"/admin/olympiad/{olympiad_id}/sport", json={
            "name": "DX Challenge",
            "target_type": "continent",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        assert resp.status_code == 200
        sport_id = resp.json()["id"]

        # 4. Create Match
        resp = client.post(f"/admin/sport/{sport_id}/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)
        assert resp.status_code == 200
        match_id = resp.json()["id"]

        # 5. Register competitors and opt into sport
        signup_user(client, "W1ABC")
        signup_user(client, "K2DEF")

        # Opt competitors into the sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1ABC", sport_id, "2026-01-01T00:00:00")
            )
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("K2DEF", sport_id, "2026-01-01T00:00:00")
            )

        # 6. Add QSOs directly to database (simulating sync)
        with get_db() as conn:
            # W1ABC contacts EU at 12:01 - should get gold
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8500.0, 1700.0)
            """, ("W1ABC", "DL1ABC", "2026-01-15T12:01:00", 5.0, "EM12", "JN58", 230))

            # K2DEF contacts EU at 12:05 - should get silver
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8600.0, 860.0)
            """, ("K2DEF", "DL2XYZ", "2026-01-15T12:05:00", 10.0, "FN31", "JO62", 230))

        # 7. Recompute medals
        recompute_match_medals(match_id)

        # 8. Verify medals
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT callsign, qso_race_medal, cool_factor_medal, total_points
                FROM medals WHERE match_id = ?
                ORDER BY total_points DESC
            """, (match_id,))
            medals = [dict(row) for row in cursor.fetchall()]

        assert len(medals) == 2

        # W1ABC should have gold for distance (earlier) and gold for CF (1700 > 860)
        w1 = next(m for m in medals if m["callsign"] == "W1ABC")
        assert w1["qso_race_medal"] == "gold"
        assert w1["cool_factor_medal"] == "gold"
        assert w1["total_points"] == 6  # 3 + 3

        # K2DEF should have silver for both
        k2 = next(m for m in medals if m["callsign"] == "K2DEF")
        assert k2["qso_race_medal"] == "silver"
        assert k2["cool_factor_medal"] == "silver"
        assert k2["total_points"] == 4  # 2 + 2

    def test_medal_reshuffle_on_new_confirmation(self, client, admin_headers):
        """Test that medals reshuffle when new QSO gets confirmed."""
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
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-31T23:59:59",
            "target_value": "EU"
        }, headers=admin_headers)

        signup_user(client, "W1ABC")
        signup_user(client, "K2DEF")

        # Opt competitors into the sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1ABC", 1, "2026-01-01T00:00:00")
            )
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("K2DEF", 1, "2026-01-01T00:00:00")
            )

        # Initial: W1ABC at 12:05 gets gold
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8500.0, 1700.0)
            """, ("W1ABC", "DL1ABC", "2026-01-15T12:05:00", 5.0, "EM12", "JN58", 230))

        recompute_match_medals(1)

        with get_db() as conn:
            cursor = conn.execute("SELECT callsign, qso_race_medal FROM medals")
            medals = {row["callsign"]: row["qso_race_medal"] for row in cursor.fetchall()}
        assert medals["W1ABC"] == "gold"

        # Now K2DEF's earlier QSO gets confirmed
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8600.0, 860.0)
            """, ("K2DEF", "DL2XYZ", "2026-01-15T12:01:00", 10.0, "FN31", "JO62", 230))

        recompute_match_medals(1)

        with get_db() as conn:
            cursor = conn.execute("SELECT callsign, qso_race_medal FROM medals")
            medals = {row["callsign"]: row["qso_race_medal"] for row in cursor.fetchall()}

        # K2DEF should now have gold (earlier)
        assert medals["K2DEF"] == "gold"
        assert medals["W1ABC"] == "silver"


class TestRecords:
    """Test record tracking."""

    def test_world_record_created(self, client, admin_headers):
        """Test world record is created for longest distance."""
        signup_user(client, "W1ABC")

        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 15000.0, 3000.0)
            """, ("W1ABC", "VK3ABC", "2026-01-15T12:00:00", 5.0, "EM12", "QF22", 150))
            qso_id = cursor.lastrowid

        update_records(qso_id, "W1ABC")

        with get_db() as conn:
            cursor = conn.execute("""
                SELECT * FROM records WHERE record_type = 'longest_distance' AND callsign IS NULL
            """)
            record = cursor.fetchone()

        assert record is not None
        assert record["value"] == 15000.0

    def test_personal_best_created(self, client, admin_headers):
        """Test personal best is created."""
        signup_user(client, "W1ABC")

        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 8500.0, 1700.0)
            """, ("W1ABC", "DL1ABC", "2026-01-15T12:00:00", 5.0, "EM12", "JN58", 230))
            qso_id = cursor.lastrowid

        update_records(qso_id, "W1ABC")

        with get_db() as conn:
            cursor = conn.execute("""
                SELECT * FROM records WHERE record_type = 'highest_cool_factor' AND callsign = ?
            """, ("W1ABC",))
            record = cursor.fetchone()

        assert record is not None
        assert record["value"] == 1700.0


class TestPOTABonus:
    """Test POTA bonus in full context."""

    def test_pota_bonus_for_park_target(self, client, admin_headers):
        """Test POTA bonus when working a park."""
        # Setup POTA sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "POTA",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-07T23:59:59",
            "target_value": "K-0001"
        }, headers=admin_headers)

        signup_user(client, "W1ABC")

        # Opt into sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1ABC", 1, "2026-01-01T00:00:00")
            )

        # Work someone at K-0001
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, dx_sig_info, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1000.0, 200.0)
            """, ("W1ABC", "N1POTA", "2026-01-05T12:00:00", 5.0, "EM12", "FN31", 291, "K-0001"))

        recompute_match_medals(1)

        with get_db() as conn:
            cursor = conn.execute("SELECT pota_bonus, total_points FROM medals WHERE callsign = ?", ("W1ABC",))
            medal = cursor.fetchone()

        assert medal["pota_bonus"] == 1
        assert medal["total_points"] == 7  # 3 + 3 + 1

    def test_park_to_park_bonus(self, client, admin_headers):
        """Test park-to-park contact gets +2 bonus."""
        # Setup POTA sport
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "POTA",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": False,
            "separate_pools": False
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-07T23:59:59",
            "target_value": "K-0001"
        }, headers=admin_headers)

        signup_user(client, "W1PTP")

        # Opt into sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1PTP", 1, "2026-01-01T00:00:00")
            )

        # Park-to-park: competitor at K-0002, works someone at K-0001
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, my_sig_info, dx_grid, dx_dxcc, dx_sig_info, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1000.0, 200.0)
            """, ("W1PTP", "N1POTA", "2026-01-05T12:00:00", 5.0, "EM12", "K-0002", "FN31", 291, "K-0001"))

        recompute_match_medals(1)

        with get_db() as conn:
            cursor = conn.execute("SELECT pota_bonus, total_points FROM medals WHERE callsign = ?", ("W1PTP",))
            medal = cursor.fetchone()

        assert medal["pota_bonus"] == 2  # Park-to-park!
        assert medal["total_points"] == 8  # 3 + 3 + 2


class TestSeparatePools:
    """Test separate work/activate pools."""

    def test_separate_pools_scoring(self, client, admin_headers):
        """Test work and activate have separate medal pools."""
        # Setup sport with separate pools
        client.post("/admin/olympiad", json={
            "name": "2026 Olympics",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "qualifying_qsos": 0
        }, headers=admin_headers)
        client.post("/admin/olympiad/1/activate", headers=admin_headers)
        client.post("/admin/olympiad/1/sport", json={
            "name": "POTA Championship",
            "target_type": "park",
            "work_enabled": True,
            "activate_enabled": True,
            "separate_pools": True
        }, headers=admin_headers)
        client.post("/admin/sport/1/match", json={
            "start_date": "2026-01-01T00:00:00",
            "end_date": "2026-01-07T23:59:59",
            "target_value": "K-0001"
        }, headers=admin_headers)

        signup_user(client, "W1ABC")
        signup_user(client, "K2DEF")

        # Opt competitors into sport
        with get_db() as conn:
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("W1ABC", 1, "2026-01-01T00:00:00")
            )
            conn.execute(
                "INSERT INTO sport_entries (callsign, sport_id, entered_at) VALUES (?, ?, ?)",
                ("K2DEF", 1, "2026-01-01T00:00:00")
            )

        with get_db() as conn:
            # W1ABC works K-0001 (hunter)
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, dx_sig_info, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1000.0, 200.0)
            """, ("W1ABC", "N1POTA", "2026-01-05T12:01:00", 5.0, "EM12", "FN31", 291, "K-0001"))

            # K2DEF activates from K-0001 - needs 10+ QSOs on same day for valid POTA activation
            for i in range(10):
                conn.execute("""
                    INSERT INTO qsos (
                        competitor_callsign, dx_callsign, qso_datetime_utc,
                        tx_power_w, my_grid, my_sig_info, dx_grid, dx_dxcc, is_confirmed,
                        distance_km, cool_factor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 500.0, 100.0)
                """, ("K2DEF", f"W{i}XYZ", f"2026-01-05T12:{i+5:02d}:00", 5.0, "FN31", "K-0001", "CM87", 291))

        recompute_match_medals(1)

        with get_db() as conn:
            cursor = conn.execute("SELECT callsign, role, qso_race_medal FROM medals ORDER BY callsign")
            medals = [dict(row) for row in cursor.fetchall()]

        # Both should have gold in their respective pools
        w1 = next(m for m in medals if m["callsign"] == "W1ABC")
        k2 = next(m for m in medals if m["callsign"] == "K2DEF")

        assert w1["role"] == "work"
        assert w1["qso_race_medal"] == "gold"

        assert k2["role"] == "activate"
        assert k2["qso_race_medal"] == "gold"

"""
Tests for scoring engine - medals, cool factor, POTA bonus.
"""

import pytest
from datetime import datetime
import os
import sys
import tempfile

# Create temp file for test database BEFORE importing app modules
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["DATABASE_PATH"] = _test_db_path

from database import init_db, get_db, reset_db
from scoring import (
    matches_target, validate_qso_for_mode, compute_medals,
    should_award_pota_bonus, MatchingQSO, MedalResult
)
from dxcc import get_continent


@pytest.fixture(autouse=True)
def setup_db():
    """Reset database before each test."""
    reset_db()


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    """Cleanup database file after all tests."""
    yield
    if os.path.exists(_test_db_path):
        os.remove(_test_db_path)


class TestTargetMatching:
    """Test QSO matching against targets."""

    def test_work_mode_continent_match(self):
        """Test work mode matching continent target."""
        qso = {
            "dx_dxcc": 230,  # Germany = EU
            "dx_grid": "JN58",
            "my_dxcc": 291,  # USA
            "my_grid": "EM12",
        }

        assert matches_target(qso, "continent", "EU", "work") == True
        assert matches_target(qso, "continent", "NA", "work") == False
        assert matches_target(qso, "continent", "AS", "work") == False

    def test_activate_mode_continent_match(self):
        """Test activate mode matching competitor's continent."""
        qso = {
            "dx_dxcc": 230,  # Germany
            "my_dxcc": 291,  # USA = NA
            "my_grid": "EM12",
        }

        assert matches_target(qso, "continent", "NA", "activate") == True
        assert matches_target(qso, "continent", "EU", "activate") == False

    def test_work_mode_park_match(self):
        """Test work mode matching POTA park target."""
        qso = {
            "dx_sig_info": "K-0001",
            "dx_grid": "FN31",
            "dx_dxcc": 291,
        }

        assert matches_target(qso, "park", "K-0001", "work") == True
        assert matches_target(qso, "park", "K-0002", "work") == False
        assert matches_target(qso, "park", "k-0001", "work") == True  # Case insensitive

    def test_activate_mode_park_match(self):
        """Test activate mode matching competitor's park."""
        qso = {
            "my_sig_info": "K-0001",
            "my_grid": "DN15",
        }

        assert matches_target(qso, "park", "K-0001", "activate") == True
        assert matches_target(qso, "park", "K-9999", "activate") == False

    def test_work_mode_grid_match(self):
        """Test work mode matching grid target."""
        qso = {
            "dx_grid": "FN31ab",
            "dx_dxcc": 291,
        }

        assert matches_target(qso, "grid", "FN31", "work") == True
        assert matches_target(qso, "grid", "FN32", "work") == False
        assert matches_target(qso, "grid", "FN31ab", "work") == True

    def test_activate_mode_grid_match(self):
        """Test activate mode matching competitor's grid."""
        qso = {
            "my_grid": "EM12cd",
        }

        assert matches_target(qso, "grid", "EM12", "activate") == True
        assert matches_target(qso, "grid", "EM13", "activate") == False

    def test_work_mode_call_match(self):
        """Test work mode matching callsign target."""
        qso = {
            "dx_callsign": "W1AW",
            "dx_dxcc": 291,
            "dx_grid": "FN31",
        }

        assert matches_target(qso, "call", "W1AW", "work") == True
        assert matches_target(qso, "call", "W1AW", "work") == True
        assert matches_target(qso, "call", "K1ABC", "work") == False


class TestQSOValidation:
    """Test QSO validation for modes."""

    def test_valid_work_mode_qso(self):
        """Test valid work mode QSO."""
        qso = {
            "tx_power_w": 5.0,
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == True
        assert error is None

    def test_valid_activate_mode_qso(self):
        """Test valid activate mode QSO."""
        qso = {
            "tx_power_w": 10.0,
            "is_confirmed": True,
            "my_grid": "EM12",
        }

        valid, error = validate_qso_for_mode(qso, "activate")
        assert valid == True

    def test_missing_power_valid_for_qso_race(self):
        """Test QSO with missing power is still valid (for QSO Race, not Cool Factor)."""
        qso = {
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        # QSOs without power are valid for QSO Race
        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == True
        assert error is None

    def test_zero_power_valid_for_qso_race(self):
        """Test QSO with zero power is still valid (for QSO Race, not Cool Factor)."""
        qso = {
            "tx_power_w": 0,
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        # QSOs with zero power are valid for QSO Race
        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == True
        assert error is None

    def test_negative_power_valid_for_qso_race(self):
        """Test QSO with negative power is still valid (for QSO Race, not Cool Factor)."""
        qso = {
            "tx_power_w": -5,
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        # QSOs with negative power are valid for QSO Race
        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == True
        assert error is None

    def test_unconfirmed_rejected(self):
        """Test unconfirmed QSO is rejected."""
        qso = {
            "tx_power_w": 5.0,
            "is_confirmed": False,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False
        assert "confirmed" in error.lower()

    def test_work_mode_missing_dxcc_rejected(self):
        """Test work mode without DXCC is rejected."""
        qso = {
            "tx_power_w": 5.0,
            "is_confirmed": True,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False
        assert "DXCC" in error

    def test_work_mode_missing_grid_rejected(self):
        """Test work mode without grid is rejected."""
        qso = {
            "tx_power_w": 5.0,
            "is_confirmed": True,
            "dx_dxcc": 230,
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False
        assert "GRIDSQUARE" in error

    def test_activate_mode_missing_my_grid_rejected(self):
        """Test activate mode without MY_GRIDSQUARE is rejected."""
        qso = {
            "tx_power_w": 5.0,
            "is_confirmed": True,
        }

        valid, error = validate_qso_for_mode(qso, "activate")
        assert valid == False
        assert "MY_GRIDSQUARE" in error


class TestMedalComputation:
    """Test medal computation logic."""

    def test_qso_race_medals_by_time(self):
        """Test distance medals awarded by earliest QSO time."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        # W2XYZ was earliest (12:01), should get gold
        w2 = next(r for r in results if r.callsign == "W2XYZ")
        assert w2.qso_race_medal == "gold"

        # K1ABC was second (12:05), should get silver
        k1 = next(r for r in results if r.callsign == "K1ABC")
        assert k1.qso_race_medal == "silver"

        # N3DEF was third (12:10), should get bronze
        n3 = next(r for r in results if r.callsign == "N3DEF")
        assert n3.qso_race_medal == "bronze"

    def test_cool_factor_medals_by_value(self):
        """Test cool factor medals awarded by highest value."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 10, 800, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 10), 8000, 2, 4000, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        # N3DEF has highest CF (4000), should get gold
        n3 = next(r for r in results if r.callsign == "N3DEF")
        assert n3.cool_factor_medal == "gold"

        # W2XYZ has second highest (1600)
        w2 = next(r for r in results if r.callsign == "W2XYZ")
        assert w2.cool_factor_medal == "silver"

        # K1ABC has lowest (800)
        k1 = next(r for r in results if r.callsign == "K1ABC")
        assert k1.cool_factor_medal == "bronze"

    def test_cool_factor_tiebreak_by_time(self):
        """Test cool factor tie broken by earlier QSO time."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        # Same CF, W2XYZ was earlier - should get gold
        w2 = next(r for r in results if r.callsign == "W2XYZ")
        assert w2.cool_factor_medal == "gold"

        k1 = next(r for r in results if r.callsign == "K1ABC")
        assert k1.cool_factor_medal == "silver"

    def test_qualifying_threshold(self):
        """Test competitors below threshold don't get medals."""
        # Single QSO each, but threshold is 2
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=2, target_type="continent")

        # Neither should be qualified
        for r in results:
            assert r.qualified == False
            assert r.qso_race_medal is None
            assert r.cool_factor_medal is None

    def test_qualifying_with_multiple_qsos(self):
        """Test competitor with multiple QSOs qualifies."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "K1ABC", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
            MatchingQSO(3, "W2XYZ", "DL3", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=2, target_type="continent")

        k1 = next(r for r in results if r.callsign == "K1ABC")
        w2 = next(r for r in results if r.callsign == "W2XYZ")

        assert k1.qualified == True
        assert k1.qso_race_medal == "gold"

        assert w2.qualified == False
        assert w2.qso_race_medal is None

    def test_separate_pools(self):
        """Test separate pools for work/activate modes."""
        qsos = [
            # Work mode entries
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "work", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "work", False),
            # Activate mode entries
            MatchingQSO(3, "N3DEF", "W1", datetime(2026, 1, 1, 12, 3), 1000, 10, 100, "activate", True),
            MatchingQSO(4, "K4GHI", "W2", datetime(2026, 1, 1, 12, 8), 1000, 5, 200, "activate", True),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="park")

        # Work pool medals
        k1 = next(r for r in results if r.callsign == "K1ABC" and r.role == "work")
        w2 = next(r for r in results if r.callsign == "W2XYZ" and r.role == "work")
        assert k1.qso_race_medal == "gold"  # Earlier
        assert w2.qso_race_medal == "silver"

        # Activate pool medals (separate from work)
        n3 = next(r for r in results if r.callsign == "N3DEF" and r.role == "activate")
        k4 = next(r for r in results if r.callsign == "K4GHI" and r.role == "activate")
        assert n3.qso_race_medal == "gold"  # Earlier
        assert k4.qso_race_medal == "silver"

    def test_total_points_calculation(self):
        """Test total points = distance + cool factor + POTA bonus."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", True),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        k1 = results[0]
        # Gold for distance (3) + gold for CF (3) + POTA bonus (1) = 7
        assert k1.qso_race_medal == "gold"
        assert k1.cool_factor_medal == "gold"
        assert k1.pota_bonus == 1
        assert k1.total_points == 7


class TestPOTABonus:
    """Test POTA bonus logic."""

    def test_park_to_park_gets_2_points(self):
        """Test park-to-park contact gets +2 bonus."""
        assert should_award_pota_bonus("park", "work", True) == 2
        assert should_award_pota_bonus("park", "activate", True) == 2

    def test_pota_target_without_park_gets_1_point(self):
        """Test POTA target without competitor at park gets +1."""
        assert should_award_pota_bonus("park", "work", False) == 1
        assert should_award_pota_bonus("park", "activate", False) == 1

    def test_non_pota_target_with_park_gets_1_point(self):
        """Test non-POTA target with competitor at park gets +1."""
        assert should_award_pota_bonus("continent", "work", True) == 1
        assert should_award_pota_bonus("grid", "activate", True) == 1

    def test_non_pota_target_without_park_gets_0_points(self):
        """Test non-POTA target without park gets +0."""
        assert should_award_pota_bonus("continent", "work", False) == 0
        assert should_award_pota_bonus("grid", "activate", False) == 0

    def test_all_target_types(self):
        """Test bonus logic for all target types."""
        non_pota_targets = ["continent", "country", "call", "grid"]

        for target in non_pota_targets:
            # Without park - no bonus
            assert should_award_pota_bonus(target, "work", False) == 0
            assert should_award_pota_bonus(target, "activate", False) == 0

            # With park - +1 bonus
            assert should_award_pota_bonus(target, "work", True) == 1
            assert should_award_pota_bonus(target, "activate", True) == 1

        # POTA target without park - +1
        assert should_award_pota_bonus("park", "work", False) == 1
        assert should_award_pota_bonus("park", "activate", False) == 1

        # Park-to-park - +2
        assert should_award_pota_bonus("park", "work", True) == 2
        assert should_award_pota_bonus("park", "activate", True) == 2

    def test_pota_target_type_gets_bonus(self):
        """Test that 'pota' target type (any park) also awards POTA bonus.

        The 'pota' target type is for 'any park' sports (vs 'park' which is
        for a specific park). Both should award POTA bonus.
        """
        # 'pota' target without competitor at park - +1
        assert should_award_pota_bonus("pota", "work", False) == 1
        assert should_award_pota_bonus("pota", "activate", False) == 1

        # 'pota' target with competitor at park (P2P) - +2
        assert should_award_pota_bonus("pota", "work", True) == 2
        assert should_award_pota_bonus("pota", "activate", True) == 2


class TestDXCCContinent:
    """Test DXCC to continent mapping."""

    def test_known_dxcc_codes(self):
        """Test known DXCC codes map to correct continents."""
        assert get_continent(230) == "EU"  # Germany
        assert get_continent(291) == "NA"  # USA
        assert get_continent(339) == "AS"  # Japan
        assert get_continent(150) == "OC"  # Australia

    def test_unknown_dxcc(self):
        """Test unknown DXCC returns None."""
        assert get_continent(99999) is None


class TestTieBreakers:
    """Test tie-breaking logic for medal awards."""

    def test_qso_race_identical_times_stable_sort(self):
        """Test QSO race with identical timestamps - uses Python's stable sort."""
        # Two competitors with exact same QSO time
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 0, 0), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 0, 0), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        # Both should have medals (one gold, one silver)
        medals = {r.callsign: r.qso_race_medal for r in results}
        assert set(medals.values()) == {"gold", "silver"}

    def test_qso_race_millisecond_difference_wins(self):
        """Test QSO race with very small time difference."""
        # K1ABC is 1 second earlier
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 0, 0), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 0, 1), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        k1 = next(r for r in results if r.callsign == "K1ABC")
        w2 = next(r for r in results if r.callsign == "W2XYZ")

        assert k1.qso_race_medal == "gold"
        assert w2.qso_race_medal == "silver"

    def test_competitor_multiple_qsos_earliest_counts_for_race(self):
        """Test that a competitor's earliest QSO is used for QSO Race."""
        qsos = [
            # K1ABC has QSOs at 12:10 and 12:05 - 12:05 should count
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "K1ABC", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
            # W2XYZ has single QSO at 12:07
            MatchingQSO(3, "W2XYZ", "DL3", datetime(2026, 1, 1, 12, 7), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        k1 = next(r for r in results if r.callsign == "K1ABC")
        w2 = next(r for r in results if r.callsign == "W2XYZ")

        # K1ABC's earliest QSO is 12:05 < W2XYZ's 12:07
        assert k1.qso_race_medal == "gold"
        assert w2.qso_race_medal == "silver"

    def test_competitor_multiple_qsos_best_cf_counts(self):
        """Test that a competitor's highest cool factor QSO is used for CF medal."""
        qsos = [
            # K1ABC has CFs of 800 and 1600 - 1600 should count
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 10, 800, "combined", False),
            MatchingQSO(2, "K1ABC", "DL2", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
            # W2XYZ has single QSO with CF 1200
            MatchingQSO(3, "W2XYZ", "DL3", datetime(2026, 1, 1, 12, 7), 6000, 5, 1200, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        k1 = next(r for r in results if r.callsign == "K1ABC")
        w2 = next(r for r in results if r.callsign == "W2XYZ")

        # K1ABC's best CF is 1600 > W2XYZ's 1200
        assert k1.cool_factor_medal == "gold"
        assert k1.cool_factor_value == 1600
        assert w2.cool_factor_medal == "silver"

    def test_cool_factor_identical_values_and_times(self):
        """Test cool factor tie with identical values and times."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 0, 0), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 0, 0), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        # Both should have medals (one gold, one silver)
        cf_medals = {r.callsign: r.cool_factor_medal for r in results}
        assert set(cf_medals.values()) == {"gold", "silver"}

    def test_cool_factor_same_value_earlier_time_wins(self):
        """Test cool factor tie broken by earlier QSO time."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 7), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        w2 = next(r for r in results if r.callsign == "W2XYZ")
        n3 = next(r for r in results if r.callsign == "N3DEF")
        k1 = next(r for r in results if r.callsign == "K1ABC")

        # Same CF, so earlier time wins
        assert w2.cool_factor_medal == "gold"   # 12:05
        assert n3.cool_factor_medal == "silver"  # 12:07
        assert k1.cool_factor_medal == "bronze"  # 12:10

    def test_four_competitors_only_three_medals(self):
        """Test that only top 3 competitors get medals."""
        qsos = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 2), 8000, 5, 1600, "combined", False),
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 3), 8000, 5, 1600, "combined", False),
            MatchingQSO(4, "K4GHI", "DL4", datetime(2026, 1, 1, 12, 4), 8000, 5, 1600, "combined", False),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="continent")

        k1 = next(r for r in results if r.callsign == "K1ABC")
        w2 = next(r for r in results if r.callsign == "W2XYZ")
        n3 = next(r for r in results if r.callsign == "N3DEF")
        k4 = next(r for r in results if r.callsign == "K4GHI")

        assert k1.qso_race_medal == "gold"
        assert w2.qso_race_medal == "silver"
        assert n3.qso_race_medal == "bronze"
        assert k4.qso_race_medal is None  # 4th place gets nothing

    def test_separate_pools_independent_tiebreaks(self):
        """Test that work and activate pools have independent tie-breaking."""
        qsos = [
            # Work pool
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "work", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "work", False),
            # Activate pool - different times, but same tie-breaking rules
            MatchingQSO(3, "N3DEF", "W1", datetime(2026, 1, 1, 12, 10), 5000, 5, 1000, "activate", True),
            MatchingQSO(4, "K4GHI", "W2", datetime(2026, 1, 1, 12, 3), 5000, 5, 1000, "activate", True),
        ]

        results = compute_medals(qsos, qualifying_qsos=0, target_type="park")

        # Work pool medals
        w2 = next(r for r in results if r.callsign == "W2XYZ" and r.role == "work")
        k1 = next(r for r in results if r.callsign == "K1ABC" and r.role == "work")
        assert w2.qso_race_medal == "gold"  # Earlier (12:01)
        assert k1.qso_race_medal == "silver"

        # Activate pool medals (independent of work)
        k4 = next(r for r in results if r.callsign == "K4GHI" and r.role == "activate")
        n3 = next(r for r in results if r.callsign == "N3DEF" and r.role == "activate")
        assert k4.qso_race_medal == "gold"  # Earlier (12:03)
        assert n3.qso_race_medal == "silver"


class TestMedalReshuffle:
    """Test that new confirmations reshuffle medals."""

    def test_new_earlier_qso_reshuffles(self):
        """Test adding an earlier QSO reshuffles distance medals."""
        # Initial QSOs
        qsos_initial = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 5, 1600, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
        ]

        results_initial = compute_medals(qsos_initial, qualifying_qsos=0, target_type="continent")

        k1_initial = next(r for r in results_initial if r.callsign == "K1ABC")
        assert k1_initial.qso_race_medal == "gold"

        # New QSO added that's earlier
        qsos_with_new = qsos_initial + [
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 1), 8000, 5, 1600, "combined", False),
        ]

        results_new = compute_medals(qsos_with_new, qualifying_qsos=0, target_type="continent")

        # N3DEF should now have gold (earliest)
        n3 = next(r for r in results_new if r.callsign == "N3DEF")
        assert n3.qso_race_medal == "gold"

        # K1ABC should be demoted to silver
        k1_new = next(r for r in results_new if r.callsign == "K1ABC")
        assert k1_new.qso_race_medal == "silver"

    def test_new_higher_cf_reshuffles(self):
        """Test adding higher cool factor QSO reshuffles CF medals."""
        qsos_initial = [
            MatchingQSO(1, "K1ABC", "DL1", datetime(2026, 1, 1, 12, 5), 8000, 10, 800, "combined", False),
            MatchingQSO(2, "W2XYZ", "DL2", datetime(2026, 1, 1, 12, 10), 8000, 5, 1600, "combined", False),
        ]

        results_initial = compute_medals(qsos_initial, qualifying_qsos=0, target_type="continent")

        w2_initial = next(r for r in results_initial if r.callsign == "W2XYZ")
        assert w2_initial.cool_factor_medal == "gold"  # CF=1600 beats 800

        # New QSO with even higher CF
        qsos_with_new = qsos_initial + [
            MatchingQSO(3, "N3DEF", "DL3", datetime(2026, 1, 1, 12, 15), 8000, 2, 4000, "combined", False),
        ]

        results_new = compute_medals(qsos_with_new, qualifying_qsos=0, target_type="continent")

        # N3DEF should now have gold (CF=4000)
        n3 = next(r for r in results_new if r.callsign == "N3DEF")
        assert n3.cool_factor_medal == "gold"

        # W2XYZ demoted to silver
        w2_new = next(r for r in results_new if r.callsign == "W2XYZ")
        assert w2_new.cool_factor_medal == "silver"


class TestTargetMatchingEdgeCases:
    """Test edge cases for target matching."""

    def test_continent_match_no_dxcc_work(self):
        """Test continent match returns False when no DXCC for work mode."""
        qso = {
            "dx_dxcc": None,
            "my_dxcc": 291,
        }
        assert matches_target(qso, "continent", "EU", "work") == False

    def test_continent_match_no_dxcc_activate(self):
        """Test continent match returns False when no DXCC for activate mode."""
        qso = {
            "dx_dxcc": 230,
            "my_dxcc": None,
        }
        assert matches_target(qso, "continent", "NA", "activate") == False

    def test_country_target_work_mode(self):
        """Test country target type for work mode."""
        qso = {
            "dx_dxcc": 230,  # Germany
            "my_dxcc": 291,
        }
        assert matches_target(qso, "country", "230", "work") == True
        assert matches_target(qso, "country", "291", "work") == False

    def test_country_target_activate_mode(self):
        """Test country target type for activate mode."""
        qso = {
            "dx_dxcc": 230,
            "my_dxcc": 291,  # USA
        }
        assert matches_target(qso, "country", "291", "activate") == True
        assert matches_target(qso, "country", "230", "activate") == False

    def test_call_target_activate_mode(self):
        """Test call target type for activate mode."""
        qso = {
            "dx_callsign": "DL1ABC",
        }
        # In activate mode, all QSOs count when targeting a call
        assert matches_target(qso, "call", "W1ABC", "activate") == True

    def test_unknown_target_type(self):
        """Test unknown target type returns False."""
        qso = {"dx_dxcc": 230}
        assert matches_target(qso, "unknown_type", "value", "work") == False


class TestComputeMedalsEdgeCases:
    """Test edge cases in medal computation."""

    def test_empty_matching_qsos(self):
        """Test compute_medals with empty list."""
        results = compute_medals([], qualifying_qsos=0, target_type="continent")
        assert results == []


class TestRecomputeMatchMedals:
    """Test recompute_match_medals function."""

    def test_recompute_nonexistent_match(self):
        """Test recomputing medals for nonexistent match."""
        from scoring import recompute_match_medals
        # Should not raise, just return early
        recompute_match_medals(99999)


class TestRecordUpdates:
    """Test world record and personal best updates."""

    def test_break_world_record(self):
        """Test breaking an existing world record."""
        from scoring import update_records
        from auth import hash_password

        # First register a competitor
        password_hash = hash_password("password123")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("W1FIRST", password_hash, "encrypted", datetime.utcnow().isoformat()))

            # Insert a QSO with 10000km distance
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("W1FIRST", "VK1ABC", "2026-01-15T12:00:00", 5.0, "EM12", "QF22", 150, 10000.0, 2000.0))
            qso_id_1 = cursor.lastrowid

        update_records(qso_id_1, "W1FIRST")

        # Verify world record was created
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT value FROM records WHERE record_type = 'longest_distance' AND callsign IS NULL
            """)
            record = cursor.fetchone()
            assert record["value"] == 10000.0

        # Now another competitor breaks the record
        with get_db() as conn:
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("K2SECOND", password_hash, "encrypted", datetime.utcnow().isoformat()))

            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("K2SECOND", "ZL1XYZ", "2026-01-16T12:00:00", 5.0, "FN31", "RF80", 170, 15000.0, 3000.0))
            qso_id_2 = cursor.lastrowid

        update_records(qso_id_2, "K2SECOND")

        # Verify world record was updated
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT value FROM records WHERE record_type = 'longest_distance' AND callsign IS NULL
            """)
            record = cursor.fetchone()
            assert record["value"] == 15000.0

    def test_break_personal_best(self):
        """Test breaking an existing personal best."""
        from scoring import update_records
        from auth import hash_password

        # Register competitor
        password_hash = hash_password("password123")
        with get_db() as conn:
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, qrz_api_key_encrypted, registered_at)
                VALUES (?, ?, ?, ?)
            """, ("W1PB", password_hash, "encrypted", datetime.utcnow().isoformat()))

            # First QSO
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("W1PB", "DL1ABC", "2026-01-15T12:00:00", 10.0, "EM12", "JN58", 230, 8500.0, 850.0))
            qso_id_1 = cursor.lastrowid

        update_records(qso_id_1, "W1PB")

        # Verify personal best was created
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT value FROM records WHERE record_type = 'highest_cool_factor' AND callsign = ?
            """, ("W1PB",))
            record = cursor.fetchone()
            assert record["value"] == 850.0

        # Second QSO with higher cool factor
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("W1PB", "VK3XYZ", "2026-01-16T12:00:00", 5.0, "EM12", "QF22", 150, 15000.0, 3000.0))
            qso_id_2 = cursor.lastrowid

        update_records(qso_id_2, "W1PB")

        # Verify personal best was updated
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT value FROM records WHERE record_type = 'highest_cool_factor' AND callsign = ?
            """, ("W1PB",))
            record = cursor.fetchone()
            assert record["value"] == 3000.0


class TestTriathlon:
    """Test Triathlon Podium computation."""

    def _setup_olympiad(self):
        """Helper to create an active olympiad."""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Test Olympiad', '2026-01-01', '2026-12-31', 0, 1)
            """)

    def _create_competitor(self, callsign: str, first_name: str = "Test"):
        """Helper to create a competitor."""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, registered_at, first_name)
                VALUES (?, 'hash', '2026-01-01', ?)
            """, (callsign, first_name))

    def _create_qso(self, callsign: str, dx_callsign: str, distance_km: float,
                    tx_power_w: float, my_sig_info: str = None, dx_sig_info: str = None,
                    qso_datetime: str = "2026-01-15T12:00:00"):
        """Helper to create a confirmed QSO with optional POTA info."""
        cool_factor = distance_km / tx_power_w if tx_power_w > 0 else 0
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, my_sig_info, dx_sig_info, mode
                ) VALUES (?, ?, ?, ?, 'EM12', 'JN58', 230, 1, ?, ?, ?, ?, 'SSB')
            """, (callsign, dx_callsign, qso_datetime, tx_power_w,
                  distance_km, cool_factor, my_sig_info, dx_sig_info))

    def test_triathlon_requires_all_three(self):
        """Test that QSO must have distance, power, AND POTA involvement."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1TRI")

        # QSO with distance and power but no POTA - should NOT qualify
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, mode
                ) VALUES ('W1TRI', 'DL1ABC', '2026-01-15T12:00:00', 5.0,
                          'EM12', 'JN58', 230, 1, 8000.0, 1600.0, 'SSB')
            """)

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0  # No POTA = no triathlon

    def test_triathlon_requires_positive_power(self):
        """Test that QSO must have positive power (not zero or missing)."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1TRI")

        # QSO with zero power - should NOT qualify
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, my_sig_info, mode
                ) VALUES ('W1TRI', 'DL1ABC', '2026-01-15T12:00:00', 0,
                          'EM12', 'JN58', 230, 1, 8000.0, 0, 'K-0001', 'SSB')
            """)

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0  # Zero power = no triathlon

    def test_triathlon_requires_positive_distance(self):
        """Test that QSO must have positive distance."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1TRI")

        # QSO with zero distance - should NOT qualify
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, my_sig_info, mode
                ) VALUES ('W1TRI', 'DL1ABC', '2026-01-15T12:00:00', 5.0,
                          'EM12', 'JN58', 230, 1, 0, 0, 'K-0001', 'SSB')
            """)

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0  # Zero distance = no triathlon

    def test_triathlon_p2p_bonus(self):
        """Test P2P gets 100 bonus, single park gets 50."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1P2P")
        self._create_competitor("W1SINGLE")

        # P2P QSO (both parks) - same distance/power for fair comparison
        self._create_qso("W1P2P", "DL1ABC", 8000.0, 5.0,
                        my_sig_info="K-0001", dx_sig_info="DL-0001",
                        qso_datetime="2026-01-15T12:00:00")

        # Single park QSO (only my park) - same distance/power
        self._create_qso("W1SINGLE", "DL2XYZ", 8000.0, 5.0,
                        my_sig_info="K-0002", dx_sig_info=None,
                        qso_datetime="2026-01-15T12:01:00")

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 2

        p2p = next(l for l in leaders if l["callsign"] == "W1P2P")
        single = next(l for l in leaders if l["callsign"] == "W1SINGLE")

        assert p2p["pota_bonus"] == 100  # P2P
        assert single["pota_bonus"] == 50  # Single park

        # P2P should have higher total score due to +50 bonus difference
        assert p2p["total_score"] > single["total_score"]

    def test_triathlon_dx_park_only(self):
        """Test QSO with only DX park (hunt) still qualifies."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1HUNT")

        # Hunt QSO (only DX at park)
        self._create_qso("W1HUNT", "DL1ABC", 8000.0, 5.0,
                        my_sig_info=None, dx_sig_info="DL-0001")

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 1
        assert leaders[0]["pota_bonus"] == 50

    def test_triathlon_percentile_calculation(self):
        """Test percentile calculation is correct."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1LOW")
        self._create_competitor("W1MID")
        self._create_competitor("W1HIGH")

        # Create 3 QSOs with different distances
        # All have POTA so they qualify
        self._create_qso("W1LOW", "DL1", 1000.0, 5.0, my_sig_info="K-0001",
                        qso_datetime="2026-01-15T12:00:00")
        self._create_qso("W1MID", "DL2", 5000.0, 5.0, my_sig_info="K-0002",
                        qso_datetime="2026-01-15T12:01:00")
        self._create_qso("W1HIGH", "DL3", 10000.0, 5.0, my_sig_info="K-0003",
                        qso_datetime="2026-01-15T12:02:00")

        leaders = compute_triathlon_leaders()

        # With 3 QSOs, percentiles should be:
        # - lowest distance (1000): 1/3 = 33.3%
        # - middle distance (5000): 2/3 = 66.7%
        # - highest distance (10000): 3/3 = 100%
        high = next(l for l in leaders if l["callsign"] == "W1HIGH")
        mid = next(l for l in leaders if l["callsign"] == "W1MID")
        low = next(l for l in leaders if l["callsign"] == "W1LOW")

        assert high["distance_percentile"] == 100.0
        assert abs(mid["distance_percentile"] - 66.67) < 1  # ~66.67%
        assert abs(low["distance_percentile"] - 33.33) < 1  # ~33.33%

    def test_triathlon_tie_breaker(self):
        """Test earlier QSO wins ties in total score."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1EARLY")
        self._create_competitor("W1LATE")

        # Two QSOs with identical distance/power/POTA = same score
        # But W1EARLY has earlier timestamp
        self._create_qso("W1EARLY", "DL1", 8000.0, 5.0, my_sig_info="K-0001",
                        qso_datetime="2026-01-15T10:00:00")  # Earlier
        self._create_qso("W1LATE", "DL2", 8000.0, 5.0, my_sig_info="K-0002",
                        qso_datetime="2026-01-15T14:00:00")  # Later

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 2

        # With identical stats, both have same percentiles
        # Tie-breaker: earlier QSO wins
        assert leaders[0]["callsign"] == "W1EARLY"
        assert leaders[1]["callsign"] == "W1LATE"

    def test_triathlon_limit(self):
        """Test only returns top N QSOs."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()

        # Create 5 competitors with QSOs
        for i in range(5):
            call = f"W{i}TEST"
            self._create_competitor(call)
            self._create_qso(call, f"DL{i}", 1000.0 * (i + 1), 5.0,
                            my_sig_info=f"K-{i:04d}",
                            qso_datetime=f"2026-01-15T12:{i:02d}:00")

        # Default limit is 3
        leaders = compute_triathlon_leaders()
        assert len(leaders) == 3

        # Custom limit
        leaders = compute_triathlon_leaders(limit=2)
        assert len(leaders) == 2

        leaders = compute_triathlon_leaders(limit=10)
        assert len(leaders) == 5  # Only 5 exist

    def test_triathlon_no_qualifying_qsos(self):
        """Test returns empty list when no QSOs qualify."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1NOQSO")

        # No QSOs at all
        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0

    def test_triathlon_no_active_olympiad(self):
        """Test returns empty list when no active olympiad."""
        from scoring import compute_triathlon_leaders

        # No olympiad created
        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0

    def test_triathlon_only_confirmed_qsos(self):
        """Test that unconfirmed QSOs are excluded."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1UNCONF")

        # Unconfirmed QSO
        with get_db() as conn:
            conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, my_sig_info, mode
                ) VALUES ('W1UNCONF', 'DL1ABC', '2026-01-15T12:00:00', 5.0,
                          'EM12', 'JN58', 230, 0, 8000.0, 1600.0, 'K-0001', 'SSB')
            """)

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0  # Unconfirmed = excluded

    def test_triathlon_max_score_300(self):
        """Test max theoretical score is 300 (100 + 100 + 100)."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()
        self._create_competitor("W1MAX")

        # Single QSO that's the best in everything = 100% distance + 100% CF + 100 P2P
        self._create_qso("W1MAX", "DL1ABC", 8000.0, 5.0,
                        my_sig_info="K-0001", dx_sig_info="DL-0001")

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 1

        # With single QSO, it's 100% in both distance and CF (100 + 100) + P2P bonus (100)
        assert leaders[0]["total_score"] == 300.0
        assert leaders[0]["distance_percentile"] == 100.0
        assert leaders[0]["cool_factor_percentile"] == 100.0
        assert leaders[0]["pota_bonus"] == 100

    def test_triathlon_qso_outside_olympiad_excluded(self):
        """Test QSOs outside olympiad date range are excluded."""
        from scoring import compute_triathlon_leaders

        self._setup_olympiad()  # 2026-01-01 to 2026-12-31
        self._create_competitor("W1OUT")

        # QSO before olympiad starts
        self._create_qso("W1OUT", "DL1ABC", 8000.0, 5.0,
                        my_sig_info="K-0001",
                        qso_datetime="2025-12-31T23:59:59")  # Before start

        leaders = compute_triathlon_leaders()
        assert len(leaders) == 0  # Outside olympiad period


class TestHonorableMentions:
    """Test honorable mentions for records made outside competition."""

    def _setup_olympiad(self):
        """Helper to create an active olympiad."""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO olympiads (name, start_date, end_date, qualifying_qsos, is_active)
                VALUES ('Test Olympiad', '2026-01-01', '2026-12-31', 0, 1)
            """)

    def _create_competitor(self, callsign: str, first_name: str = "Test"):
        """Helper to create a competitor."""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO competitors (callsign, password_hash, registered_at, first_name)
                VALUES (?, 'hash', '2026-01-01', ?)
            """, (callsign, first_name))

    def _create_qso(self, callsign: str, dx_callsign: str, distance_km: float,
                    tx_power_w: float, qso_datetime: str = "2026-01-15T12:00:00",
                    is_confirmed: int = 1):
        """Helper to create a QSO."""
        cool_factor = distance_km / tx_power_w if tx_power_w > 0 else 0
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, mode
                ) VALUES (?, ?, ?, ?, 'EM12', 'JN58', 230, ?, ?, ?, 'SSB')
            """, (callsign, dx_callsign, qso_datetime, tx_power_w,
                  is_confirmed, distance_km, cool_factor))
            return cursor.lastrowid

    def _create_sport_and_match(self, target_type: str = "continent", target_value: str = "EU"):
        """Helper to create a sport and match."""
        with get_db() as conn:
            # Get olympiad id
            olympiad_id = conn.execute("SELECT id FROM olympiads WHERE is_active = 1").fetchone()['id']

            # Create sport
            cursor = conn.execute("""
                INSERT INTO sports (olympiad_id, name, description, target_type, work_enabled, activate_enabled, separate_pools)
                VALUES (?, 'Test Sport', 'Test', ?, 1, 0, 0)
            """, (olympiad_id, target_type))
            sport_id = cursor.lastrowid

            # Create match
            cursor = conn.execute("""
                INSERT INTO matches (sport_id, start_date, end_date, target_value)
                VALUES (?, '2026-01-01', '2026-01-31', ?)
            """, (sport_id, target_value))
            return sport_id, cursor.lastrowid

    def _create_world_record(self, record_type: str, value: float, qso_id: int):
        """Helper to create a world record."""
        with get_db() as conn:
            conn.execute("""
                INSERT INTO records (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at)
                VALUES (NULL, NULL, NULL, ?, ?, ?, '2026-01-15T12:00:00')
            """, (record_type, value, qso_id))

    def test_honorable_mention_better_outside_olympiad(self):
        """Test that a QSO outside olympiad dates gets honorable mention if better than record."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()  # 2026-01-01 to 2026-12-31
        self._create_competitor("W1RECORD", "Record")
        self._create_competitor("W1OUTSIDE", "Outside")

        # Create a sport and match (needed for competition QSO)
        self._create_sport_and_match("continent", "EU")

        # Competition QSO (inside olympiad, matches target)
        qso_id_1 = self._create_qso("W1RECORD", "DL1ABC", 8000.0, 5.0,
                                     qso_datetime="2026-01-15T12:00:00")

        # Create world record from the competition QSO
        self._create_world_record("longest_distance", 8000.0, qso_id_1)

        # QSO outside olympiad with BETTER distance
        self._create_qso("W1OUTSIDE", "VK3ABC", 15000.0, 5.0,
                        qso_datetime="2025-06-15T12:00:00")  # Before olympiad

        mentions = get_honorable_mentions()

        # Should have an honorable mention for longest distance
        assert mentions['longest_distance'] is not None
        assert mentions['longest_distance']['callsign'] == "W1OUTSIDE"
        assert mentions['longest_distance']['value'] == 15000.0

    def test_honorable_mention_better_no_target_match(self):
        """Test QSO inside olympiad but not matching any target gets honorable mention."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()
        self._create_competitor("W1RECORD", "Record")
        self._create_competitor("W1NOMATCH", "NoMatch")

        # Create a sport targeting EU continent
        self._create_sport_and_match("continent", "EU")

        # Competition QSO (inside olympiad, matches EU target via dx_dxcc=230 Germany)
        qso_id_1 = self._create_qso("W1RECORD", "DL1ABC", 8000.0, 5.0,
                                     qso_datetime="2026-01-15T12:00:00")

        self._create_world_record("longest_distance", 8000.0, qso_id_1)

        # QSO inside olympiad but to Oceania (doesn't match EU target)
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    tx_power_w, my_grid, dx_grid, dx_dxcc, is_confirmed,
                    distance_km, cool_factor, mode
                ) VALUES ('W1NOMATCH', 'VK3XYZ', '2026-01-20T12:00:00', 5.0,
                          'EM12', 'QF22', 150, 1, 15000.0, 3000.0, 'SSB')
            """)

        mentions = get_honorable_mentions()

        # Should have honorable mention (better QSO, but target was OC not EU)
        assert mentions['longest_distance'] is not None
        assert mentions['longest_distance']['callsign'] == "W1NOMATCH"
        assert mentions['longest_distance']['value'] == 15000.0

    def test_no_honorable_mention_when_record_is_best(self):
        """Test no honorable mention when the competition record IS the best QSO."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()
        self._create_competitor("W1RECORD", "Record")
        self._create_competitor("W1OUTSIDE", "Outside")

        self._create_sport_and_match("continent", "EU")

        # Competition QSO with best distance
        qso_id_1 = self._create_qso("W1RECORD", "DL1ABC", 15000.0, 5.0,
                                     qso_datetime="2026-01-15T12:00:00")
        self._create_world_record("longest_distance", 15000.0, qso_id_1)

        # QSO outside olympiad with WORSE distance
        self._create_qso("W1OUTSIDE", "VK3ABC", 8000.0, 5.0,
                        qso_datetime="2025-06-15T12:00:00")

        mentions = get_honorable_mentions()

        # No honorable mention - the record is already the best
        assert mentions['longest_distance'] is None

    def test_honorable_mention_requires_confirmed(self):
        """Test that unconfirmed QSOs don't get honorable mention."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()
        self._create_competitor("W1RECORD", "Record")
        self._create_competitor("W1UNCONF", "Unconfirmed")

        self._create_sport_and_match("continent", "EU")

        # Competition QSO
        qso_id_1 = self._create_qso("W1RECORD", "DL1ABC", 8000.0, 5.0,
                                     qso_datetime="2026-01-15T12:00:00")
        self._create_world_record("longest_distance", 8000.0, qso_id_1)

        # Better QSO outside olympiad but UNCONFIRMED
        self._create_qso("W1UNCONF", "VK3ABC", 15000.0, 5.0,
                        qso_datetime="2025-06-15T12:00:00",
                        is_confirmed=0)  # Unconfirmed

        mentions = get_honorable_mentions()

        # No honorable mention - unconfirmed doesn't count
        assert mentions['longest_distance'] is None

    def test_honorable_mention_cool_factor(self):
        """Test honorable mention for cool factor records."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()
        self._create_competitor("W1RECORD", "Record")
        self._create_competitor("W1OUTSIDE", "Outside")

        self._create_sport_and_match("continent", "EU")

        # Competition QSO with moderate cool factor
        qso_id_1 = self._create_qso("W1RECORD", "DL1ABC", 8000.0, 10.0,  # CF = 800
                                     qso_datetime="2026-01-15T12:00:00")
        self._create_world_record("highest_cool_factor", 800.0, qso_id_1)

        # QSO outside olympiad with BETTER cool factor
        self._create_qso("W1OUTSIDE", "VK3ABC", 15000.0, 5.0,  # CF = 3000
                        qso_datetime="2025-06-15T12:00:00")

        mentions = get_honorable_mentions()

        # Should have honorable mention for cool factor
        assert mentions['highest_cool_factor'] is not None
        assert mentions['highest_cool_factor']['callsign'] == "W1OUTSIDE"
        assert mentions['highest_cool_factor']['value'] == 3000.0

    def test_no_honorable_mention_without_records(self):
        """Test no honorable mention when there are no existing records."""
        from scoring import get_honorable_mentions

        self._setup_olympiad()
        self._create_competitor("W1TEST", "Test")

        # QSO outside olympiad - no record to beat
        self._create_qso("W1TEST", "DL1ABC", 8000.0, 5.0,
                        qso_datetime="2025-06-15T12:00:00")

        mentions = get_honorable_mentions()

        # No honorable mention - but it should be the new record, not honorable mention
        # (Honorable mention only when there's a competition record that's worse)
        assert mentions['longest_distance'] is None
        assert mentions['highest_cool_factor'] is None

    def test_honorable_mention_no_active_olympiad(self):
        """Test honorable mentions work when no olympiad is active."""
        from scoring import get_honorable_mentions

        # No olympiad - just competitors and QSOs
        self._create_competitor("W1TEST", "Test")
        qso_id = self._create_qso("W1TEST", "DL1ABC", 8000.0, 5.0,
                                   qso_datetime="2026-01-15T12:00:00")

        # Create a world record somehow (from previous olympiad)
        self._create_world_record("longest_distance", 5000.0, qso_id)

        self._create_competitor("W1BETTER", "Better")
        self._create_qso("W1BETTER", "VK3ABC", 10000.0, 5.0,
                        qso_datetime="2026-02-15T12:00:00")

        mentions = get_honorable_mentions()

        # With no active olympiad, all QSOs are "outside competition"
        # The better QSO should be an honorable mention
        assert mentions['longest_distance'] is not None
        assert mentions['longest_distance']['value'] == 10000.0

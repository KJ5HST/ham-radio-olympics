"""
Tests for scoring engine - medals, cool factor, POTA bonus.
"""

import pytest
from datetime import datetime
import os
import sys

# Set test database
os.environ["DATABASE_PATH"] = ":memory:"

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

    def test_missing_power_rejected(self):
        """Test QSO with missing power is rejected."""
        qso = {
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False
        assert "power" in error.lower()

    def test_zero_power_rejected(self):
        """Test QSO with zero power is rejected."""
        qso = {
            "tx_power_w": 0,
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False

    def test_negative_power_rejected(self):
        """Test QSO with negative power is rejected."""
        qso = {
            "tx_power_w": -5,
            "is_confirmed": True,
            "dx_dxcc": 230,
            "dx_grid": "JN58",
        }

        valid, error = validate_qso_for_mode(qso, "work")
        assert valid == False

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

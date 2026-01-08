"""
Tests for Maidenhead grid square and distance calculations.
"""

import pytest
import math

from grid_distance import maidenhead_to_latlon, haversine_distance, grid_distance


class TestMaidenheadConversion:
    """Test Maidenhead grid to lat/lon conversion."""

    def test_two_char_grid(self):
        """Test 2-character field conversion."""
        lat, lon = maidenhead_to_latlon("FN")
        # FN is centered at lat 45, lon -70 (center of field)
        assert 40 <= lat <= 50
        assert -80 <= lon <= -60

    def test_four_char_grid(self):
        """Test 4-character grid conversion."""
        lat, lon = maidenhead_to_latlon("FN31")
        # FN31 is Connecticut area
        assert 41.0 <= lat <= 42.0
        assert -73.0 <= lon <= -71.0

    def test_six_char_grid(self):
        """Test 6-character subsquare conversion."""
        lat, lon = maidenhead_to_latlon("FN31pr")
        # More precise location (FN31pr is in Connecticut)
        assert 41.7 <= lat <= 41.8
        assert -72.8 <= lon <= -72.6

    def test_eight_char_grid(self):
        """Test 8-character extended grid conversion."""
        lat, lon = maidenhead_to_latlon("FN31pr55")
        # Very precise location
        assert 41.72 <= lat <= 41.74
        assert -72.8 <= lon <= -72.6

    def test_known_grids(self):
        """Test known grid square locations."""
        test_cases = [
            ("JN58", 48.0, 49.0, 10.0, 12.0),     # Germany
            ("EM12", 32.0, 33.0, -98.0, -96.0),   # Texas
            ("QF22", -38.0, -37.0, 144.0, 146.0), # Victoria, Australia
        ]

        for grid, lat_min, lat_max, lon_min, lon_max in test_cases:
            lat, lon = maidenhead_to_latlon(grid)
            assert lat_min <= lat <= lat_max, f"Grid {grid}: lat {lat} not in range"
            assert lon_min <= lon <= lon_max, f"Grid {grid}: lon {lon} not in range"

    def test_lowercase_grid(self):
        """Test that lowercase grids are handled."""
        lat1, lon1 = maidenhead_to_latlon("fn31")
        lat2, lon2 = maidenhead_to_latlon("FN31")
        assert lat1 == lat2
        assert lon1 == lon2

    def test_mixed_case_grid(self):
        """Test mixed case subsquare (FN31pr vs FN31PR)."""
        lat1, lon1 = maidenhead_to_latlon("FN31pr")
        lat2, lon2 = maidenhead_to_latlon("FN31PR")
        assert lat1 == lat2
        assert lon1 == lon2

    def test_invalid_grid_length(self):
        """Test invalid grid length raises error."""
        with pytest.raises(ValueError):
            maidenhead_to_latlon("F")  # Too short

        with pytest.raises(ValueError):
            maidenhead_to_latlon("FN3")  # Odd length

        with pytest.raises(ValueError):
            maidenhead_to_latlon("FN31pr123")  # Too long

    def test_invalid_field_chars(self):
        """Test invalid field characters raise error."""
        with pytest.raises(ValueError):
            maidenhead_to_latlon("ZZ")  # Z > R

        with pytest.raises(ValueError):
            maidenhead_to_latlon("12")  # Numbers in field

    def test_invalid_square_chars(self):
        """Test invalid square characters raise error."""
        with pytest.raises(ValueError):
            maidenhead_to_latlon("FNAB")  # Letters instead of numbers

    def test_invalid_subsquare_chars(self):
        """Test invalid subsquare characters raise error."""
        with pytest.raises(ValueError):
            maidenhead_to_latlon("FN3199")  # Numbers instead of letters

        with pytest.raises(ValueError):
            maidenhead_to_latlon("FN31YZ")  # Y, Z > X


class TestHaversineDistance:
    """Test great-circle distance calculations."""

    def test_same_point(self):
        """Test distance from point to itself is zero."""
        dist = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert dist == 0.0

    def test_known_distance(self):
        """Test known distance between cities."""
        # New York (40.7128, -74.0060) to London (51.5074, -0.1278)
        # Actual distance ~5570 km
        dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 <= dist <= 5650

    def test_antipodal_points(self):
        """Test distance between antipodal points (half circumference)."""
        # Distance should be approximately half Earth's circumference
        dist = haversine_distance(0, 0, 0, 180)
        # Half circumference ~20015 km
        assert 20000 <= dist <= 20100

    def test_equator_distance(self):
        """Test distance along equator."""
        # 1 degree of longitude at equator ~111 km
        dist = haversine_distance(0, 0, 0, 1)
        assert 110 <= dist <= 112


class TestGridDistance:
    """Test distance between grid squares."""

    def test_same_grid(self):
        """Test distance from grid to itself is zero."""
        dist = grid_distance("FN31", "FN31")
        assert dist == 0.0

    def test_adjacent_grids(self):
        """Test distance between adjacent grids."""
        dist = grid_distance("FN31", "FN32")
        # Adjacent grids are about 1 degree lat apart ~111km
        assert 100 <= dist <= 120

    def test_cross_atlantic(self):
        """Test transatlantic distance."""
        # EM12 (Texas) to JN58 (Germany)
        dist = grid_distance("EM12", "JN58")
        # Should be ~8000-9000 km
        assert 8000 <= dist <= 9500

    def test_cross_pacific(self):
        """Test transpacific distance."""
        # EM12 (Texas) to QF22 (Australia)
        dist = grid_distance("EM12", "QF22")
        # Should be ~14000-16000 km
        assert 14000 <= dist <= 16000

    def test_different_precision_grids(self):
        """Test distance with different precision grids."""
        # 4-char to 6-char should still work
        dist = grid_distance("FN31", "JN58ab")
        assert dist > 0

    def test_cool_factor_calculation(self):
        """Test cool factor = distance / power."""
        # EM12 to JN58 ~8500km
        dist = grid_distance("EM12", "JN58")
        power = 5.0  # 5 watts

        cool_factor = dist / power
        # Should be ~1700
        assert 1600 <= cool_factor <= 1900


class TestMaidenheadExtendedPrecision:
    """Test extended precision grid squares (8+ characters)."""

    def test_eight_char_grid(self):
        """Test 8-character extended precision grid."""
        lat, lon = maidenhead_to_latlon("FN31pr12")
        # Should be more precise than 6-char
        assert 41.0 <= lat <= 42.0
        assert -73.0 <= lon <= -71.0

    def test_invalid_extended_chars(self):
        """Test 8-character grid with invalid chars raises error."""
        with pytest.raises(ValueError) as exc:
            maidenhead_to_latlon("FN31prXY")
        assert "Invalid extended characters" in str(exc.value)

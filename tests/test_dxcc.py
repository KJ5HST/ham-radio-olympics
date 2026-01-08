"""
Tests for DXCC entity to continent mapping.
"""

import pytest
import os
from unittest.mock import patch

import dxcc
from dxcc import (
    get_continent, get_continent_name, _load_dxcc_data, _get_embedded_data,
    get_all_continents, get_all_countries, get_country_name
)


@pytest.fixture(autouse=True)
def reset_dxcc_cache():
    """Reset DXCC cache before each test."""
    dxcc._dxcc_data = None
    yield
    dxcc._dxcc_data = None


class TestGetContinent:
    """Test continent lookup by DXCC code."""

    def test_germany_is_europe(self):
        """Test Germany (230) is in Europe."""
        assert get_continent(230) == "EU"

    def test_usa_is_north_america(self):
        """Test USA (291) is in North America."""
        assert get_continent(291) == "NA"

    def test_japan_is_asia(self):
        """Test Japan (339) is in Asia."""
        assert get_continent(339) == "AS"

    def test_australia_is_oceania(self):
        """Test Australia (150) is in Oceania."""
        assert get_continent(150) == "OC"

    def test_unknown_dxcc_returns_none(self):
        """Test unknown DXCC code returns None."""
        assert get_continent(99999) is None


class TestGetContinentName:
    """Test continent name lookup."""

    def test_europe_name(self):
        """Test EU returns Europe."""
        assert get_continent_name("EU") == "Europe"

    def test_north_america_name(self):
        """Test NA returns North America."""
        assert get_continent_name("NA") == "North America"

    def test_asia_name(self):
        """Test AS returns Asia."""
        assert get_continent_name("AS") == "Asia"

    def test_unknown_continent_returns_none(self):
        """Test unknown continent code returns None."""
        assert get_continent_name("XX") is None


class TestEmbeddedData:
    """Test embedded DXCC data fallback."""

    def test_get_embedded_data(self):
        """Test embedded data has correct structure."""
        data = _get_embedded_data()
        assert "continents" in data
        assert "entities" in data
        assert data["continents"]["EU"] == "Europe"
        assert data["entities"]["230"] == "EU"

    def test_fallback_to_embedded_data(self, monkeypatch, tmp_path):
        """Test fallback to embedded data when JSON file not found."""
        import dxcc
        # Reset cached data
        dxcc._dxcc_data = None

        # Mock os.path.exists to return False for all paths
        original_exists = os.path.exists
        def mock_exists(path):
            if "dxcc_continents.json" in path:
                return False
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", mock_exists)

        # Load data - should use embedded
        data = _load_dxcc_data()
        assert "continents" in data
        assert "entities" in data

        # Reset for other tests
        dxcc._dxcc_data = None


class TestGetAllContinents:
    """Test getting list of all continents."""

    def test_returns_all_seven_continents(self):
        """Test returns all 7 continents."""
        continents = get_all_continents()
        assert len(continents) == 7

    def test_continent_format(self):
        """Test continent entries have code and name."""
        continents = get_all_continents()
        # Should be list of (code, name) tuples
        codes = [c[0] for c in continents]
        assert "NA" in codes
        assert "EU" in codes
        assert "AS" in codes

    def test_sorted_by_name(self):
        """Test continents are sorted by name."""
        continents = get_all_continents()
        names = [c[1] for c in continents]
        assert names == sorted(names)


class TestGetAllCountries:
    """Test getting list of all DXCC countries."""

    def test_returns_countries(self):
        """Test returns list of countries."""
        countries = get_all_countries()
        assert len(countries) > 0

    def test_country_format(self):
        """Test country entries have code and name."""
        countries = get_all_countries()
        # Find USA
        usa = next((c for c in countries if c[0] == "291"), None)
        assert usa is not None
        assert usa[1] == "United States"

    def test_sorted_by_name(self):
        """Test countries are sorted by name."""
        countries = get_all_countries()
        names = [c[1] for c in countries]
        assert names == sorted(names)


class TestGetCountryName:
    """Test country name lookup by DXCC code."""

    def test_usa_name(self):
        """Test USA (291) returns United States."""
        assert get_country_name(291) == "United States"

    def test_germany_name(self):
        """Test Germany (230) returns Germany."""
        assert get_country_name(230) == "Germany"

    def test_unknown_returns_none(self):
        """Test unknown DXCC returns None."""
        assert get_country_name(99999) is None

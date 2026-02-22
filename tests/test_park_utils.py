"""Tests for park_utils module."""

import os
import sys
import tempfile

# Create temp file for test database
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)

# Set test database BEFORE importing app modules
os.environ["DATABASE_PATH"] = _test_db_path
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"
os.environ["TESTING"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from park_utils import is_pota_reference, get_park_name_cached, format_park_display
from database import get_db, reset_db


@pytest.fixture(autouse=True)
def fresh_db():
    reset_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    yield
    try:
        os.unlink(_test_db_path)
    except OSError:
        pass


class TestIsPotaReference:
    """Test is_pota_reference function."""

    def test_valid_references(self):
        assert is_pota_reference("K-0001") is True
        assert is_pota_reference("VE-0123") is True
        assert is_pota_reference("DL-0456") is True
        assert is_pota_reference("US-12607") is True

    def test_invalid_references(self):
        assert is_pota_reference("") is False
        assert is_pota_reference("EU") is False
        assert is_pota_reference("FN31") is False
        assert is_pota_reference("W1AW") is False
        assert is_pota_reference("K-01") is False  # too few digits

    def test_case_insensitive(self):
        assert is_pota_reference("k-0001") is True
        assert is_pota_reference("ve-0123") is True

    def test_none_input(self):
        assert is_pota_reference(None) is False


class TestGetParkNameCached:
    """Test get_park_name_cached function."""

    def test_returns_none_when_not_cached(self):
        result = get_park_name_cached("K-9999")
        assert result is None

    def test_returns_name_when_cached(self):
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("K-0001", "Acadia National Park", "US-ME", "FN44", "2025-01-01T00:00:00")
            )
        result = get_park_name_cached("K-0001")
        assert result == "Acadia National Park"

    def test_case_normalized(self):
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("K-0001", "Acadia National Park", "US-ME", "FN44", "2025-01-01T00:00:00")
            )
        result = get_park_name_cached("k-0001")
        assert result == "Acadia National Park"

    def test_none_input(self):
        result = get_park_name_cached(None)
        assert result is None

    def test_empty_input(self):
        result = get_park_name_cached("")
        assert result is None


class TestFormatParkDisplay:
    """Test format_park_display function."""

    def test_with_cached_name(self):
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("K-0001", "Acadia National Park", "US-ME", "FN44", "2025-01-01T00:00:00")
            )
        result = format_park_display("K-0001")
        assert result == "Acadia National Park (K-0001)"

    def test_without_cached_name(self):
        result = format_park_display("K-9999")
        assert result == "K-9999"

    def test_non_pota_reference(self):
        result = format_park_display("EU")
        assert result == "EU"

    def test_empty_input(self):
        result = format_park_display("")
        assert result == ""

    def test_none_input(self):
        result = format_park_display(None)
        assert result == ""

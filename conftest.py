"""
Pytest configuration and fixtures.
"""

import os
import sys
from unittest.mock import patch, AsyncMock

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set test environment
os.environ["DATABASE_PATH"] = ":memory:"
os.environ["ADMIN_KEY"] = "test-admin-key"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key"

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment."""
    pass


@pytest.fixture(autouse=True)
def mock_qrz_verify():
    """Mock QRZ API key verification to return True for all tests."""
    with patch('main.verify_api_key', new_callable=AsyncMock) as mock:
        mock.return_value = True
        with patch('qrz_client.verify_api_key', new_callable=AsyncMock) as mock2:
            mock2.return_value = True
            yield mock

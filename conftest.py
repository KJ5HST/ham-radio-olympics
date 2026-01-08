"""
Pytest configuration and fixtures.
"""

import os
import sys

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

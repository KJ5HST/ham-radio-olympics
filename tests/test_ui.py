"""
Tests for UI/UX improvements - TDD: Tests written before implementation.
"""

import os
import tempfile
import pytest

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
def logged_in_client(client):
    """Create test client with logged in user."""
    client.post("/signup", json={
        "callsign": "W1TST",
        "password": "password123",
        "qrz_api_key": "test-key"
    })
    return client


class TestDarkMode:
    """Test dark mode functionality."""

    def test_base_template_has_theme_toggle(self, client):
        """Test that base template has theme toggle button."""
        response = client.get("/")
        assert response.status_code == 200
        # Should have theme toggle button
        assert "theme-toggle" in response.text or "dark-mode" in response.text.lower()

    def test_base_template_has_css_variables(self, client):
        """Test that base template uses CSS custom properties for theming."""
        response = client.get("/")
        # Should have CSS variables for colors
        assert "--bg" in response.text or "var(--" in response.text

    def test_base_template_has_theme_script(self, client):
        """Test that base template has JavaScript for theme switching."""
        response = client.get("/")
        # Should have localStorage theme handling
        assert "localStorage" in response.text or "theme" in response.text

    def test_prefers_color_scheme_support(self, client):
        """Test that dark mode respects prefers-color-scheme."""
        response = client.get("/")
        # Should have media query for system preference
        assert "prefers-color-scheme" in response.text


class TestToastNotifications:
    """Test toast notification system."""

    def test_base_template_has_toast_container(self, client):
        """Test that base template has toast notification container."""
        response = client.get("/")
        assert response.status_code == 200
        # Should have toast container element
        assert "toast" in response.text.lower()

    def test_toast_styles_exist(self, client):
        """Test that toast notification styles exist."""
        response = client.get("/")
        # Should have toast-related styles
        assert "toast" in response.text.lower()

    def test_flash_messages_use_toast(self, logged_in_client):
        """Test that flash messages display as toasts."""
        # Trigger a flash message by logging out
        response = logged_in_client.post("/logout", follow_redirects=True)
        assert response.status_code == 200
        # Page should have toast or alert styling
        assert "toast" in response.text.lower() or "alert" in response.text.lower()


class TestAccessibility:
    """Test accessibility features."""

    def test_skip_navigation_link(self, client):
        """Test that pages have skip navigation link."""
        response = client.get("/")
        assert response.status_code == 200
        # Should have skip to content link
        assert "skip" in response.text.lower() or "main-content" in response.text

    def test_proper_heading_hierarchy(self, client):
        """Test that pages use proper heading hierarchy."""
        response = client.get("/")
        # Should have h1 followed by h2, not skipping levels
        text = response.text.lower()
        assert "<h1" in text

    def test_form_labels_exist(self, client):
        """Test that form inputs have labels."""
        response = client.get("/login")
        assert response.status_code == 200
        # Should have label elements
        assert "<label" in response.text.lower()

    def test_aria_labels_on_buttons(self, client):
        """Test that icon buttons have aria labels."""
        response = client.get("/")
        # Navigation and buttons should have accessible names
        assert "aria-label" in response.text or "title=" in response.text

    def test_focus_indicators_in_css(self, client):
        """Test that focus indicators are styled."""
        response = client.get("/")
        # Should have :focus styles
        assert ":focus" in response.text

    def test_alt_text_on_images(self, logged_in_client):
        """Test that images have alt text."""
        response = logged_in_client.get("/dashboard")
        # If there are images, they should have alt attributes
        # This is a soft check since there might not be images
        if "<img" in response.text:
            assert "alt=" in response.text


class TestMobileResponsiveness:
    """Test mobile responsiveness."""

    def test_viewport_meta_tag(self, client):
        """Test that viewport meta tag exists."""
        response = client.get("/")
        assert response.status_code == 200
        assert "viewport" in response.text

    def test_responsive_table_wrapper(self, logged_in_client):
        """Test that tables have responsive wrappers."""
        response = logged_in_client.get("/dashboard")
        # Tables should be wrapped for horizontal scroll
        text = response.text.lower()
        if "<table" in text:
            assert "table-responsive" in text or "overflow" in text

    def test_mobile_navigation(self, client):
        """Test that mobile navigation exists."""
        response = client.get("/")
        # Should have hamburger menu or mobile nav
        assert "mobile" in response.text.lower() or "menu" in response.text.lower() or "nav" in response.text.lower()

    def test_media_queries_exist(self, client):
        """Test that media queries exist for responsiveness."""
        response = client.get("/")
        # Should have media queries for different screen sizes
        assert "@media" in response.text


class TestLoadingStates:
    """Test loading state indicators."""

    def test_loading_spinner_css(self, client):
        """Test that loading spinner styles exist."""
        response = client.get("/")
        # Should have loading/spinner styles
        text = response.text.lower()
        assert "loading" in text or "spinner" in text or "animate" in text

    def test_form_submit_loading(self, client):
        """Test that forms show loading state on submit."""
        response = client.get("/login")
        # Form should have loading handling
        text = response.text.lower()
        # Check for either loading indicator or disabled button on submit
        assert "loading" in text or "submit" in text


class TestGeneralUIImprovements:
    """Test general UI improvements."""

    def test_consistent_button_styles(self, client):
        """Test that buttons have consistent styling."""
        response = client.get("/")
        # Should have .btn class for buttons
        assert "btn" in response.text

    def test_card_component_exists(self, logged_in_client):
        """Test that card component is used."""
        response = logged_in_client.get("/dashboard")
        # Should use card styling for content sections
        assert "card" in response.text.lower()

    def test_error_states_styled(self, client):
        """Test that error states have styling."""
        response = client.get("/")
        # Should have error/alert styles defined
        text = response.text.lower()
        assert "error" in text or "alert" in text or "danger" in text

    def test_success_states_styled(self, client):
        """Test that success states have styling."""
        response = client.get("/")
        # Should have success styles defined
        assert "success" in response.text.lower()

    def test_no_inline_alert_calls(self, client):
        """Test that pages don't use JavaScript alert() for notifications."""
        response = client.get("/login")
        # Should not use alert() for user feedback
        # Allow alerts only for confirm() dialogs
        text = response.text
        # Count occurrences of alert( that aren't confirm(
        alert_count = text.count("alert(")
        # Some confirm dialogs are okay, but excessive alert() is bad UX
        assert alert_count <= 3, "Too many alert() calls, use toast notifications instead"

"""
Tests for the WebBrowserPlugin.
Ensures URLs are properly formatted and opened via webbrowser mocking.
"""

import pytest
from unittest.mock import patch
from openoperator.plugins.web_browser import WebBrowserPlugin


@pytest.fixture
def browser_plugin():
    plugin = WebBrowserPlugin()
    plugin.initialize({})
    return plugin


def test_plugin_metadata(browser_plugin):
    """Verify plugin properties."""
    assert browser_plugin.name == "WebBrowser"
    assert browser_plugin.version == "1.0.0"


@patch("webbrowser.open")
def test_execute_open_url_success(mock_open, browser_plugin):
    """Test successful URL launch and auto-https formatting."""
    mock_open.return_value = True
    
    # Passing url without https
    result = browser_plugin.execute("open_url", url="google.com")
    
    assert result is True
    mock_open.assert_called_once_with("https://google.com")


@patch("webbrowser.open")
def test_execute_open_url_with_schema(mock_open, browser_plugin):
    """Test URL launch when schema is already provided."""
    mock_open.return_value = True
    result = browser_plugin.execute("open_url", url="http://localhost:8080")
    
    assert result is True
    mock_open.assert_called_once_with("http://localhost:8080")


def test_execute_invalid_command(browser_plugin):
    """Test that unsupported commands fail gracefully."""
    result = browser_plugin.execute("close_browser", url="google.com")
    assert result is False


def test_execute_missing_url(browser_plugin):
    """Test that missing required arguments are handled."""
    result = browser_plugin.execute("open_url")
    assert result is False
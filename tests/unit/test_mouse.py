"""
Tests for MouseActionController.
Verifies scroll direction logic and drag-and-drop sequencing.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pynput.mouse import Button

from openoperator.action.mouse import MouseActionController


@pytest.fixture
def mouse_controller():
    return MouseActionController()


@patch('pynput.mouse.Controller.scroll')
def test_scroll_down(mock_scroll, mouse_controller):
    """Test downward scrolling converts to negative delta Y."""
    result = mouse_controller.scroll(5, "down")
    assert result is True
    mock_scroll.assert_called_once_with(0, -5)


@patch('pynput.mouse.Controller.scroll')
def test_scroll_up(mock_scroll, mouse_controller):
    """Test upward scrolling converts to positive delta Y."""
    result = mouse_controller.scroll(3, "up")
    assert result is True
    mock_scroll.assert_called_once_with(0, 3)


@patch('pynput.mouse.Controller.release')
@patch('pynput.mouse.Controller.press')
@patch('time.sleep')
def test_drag_and_drop(mock_sleep, mock_press, mock_release, mouse_controller):
    """Test the complete sequence of a drag and drop action."""
    # We need to mock the property assignment for position
    with patch('pynput.mouse.Controller.position', new_callable=PropertyMock) as mock_position:
        result = mouse_controller.drag_and_drop(100, 100, 500, 500)
        
        assert result is True
        
        # Verify position was set twice (start and end)
        assert mock_position.call_count == 2
        
        # Verify press and release happened with Left button
        mock_press.assert_called_once_with(Button.left)
        mock_release.assert_called_once_with(Button.left)
        
        # Verify sleeps occurred to mimic human timing
        assert mock_sleep.call_count == 3
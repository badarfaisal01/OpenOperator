"""
Mouse Action Controller for OpenOperator.
Handles advanced physical mouse movements like scrolling and dragging.
"""

import logging
import time
from pynput.mouse import Controller, Button

logger = logging.getLogger(__name__)


class MouseActionController:
    """
    Simulates advanced OS-level mouse actions.
    """

    def __init__(self):
        self.mouse = Controller()

    def scroll(self, clicks: int, direction: str = "down") -> bool:
        """
        Scrolls the mouse wheel.
        
        Args:
            clicks (int): Number of "ticks" to scroll.
            direction (str): "up" or "down".
            
        Returns:
            bool: True if successful.
        """
        try:
            # pynput scroll takes (dx, dy). Negative dy is typically scrolling down.
            scroll_amount = -clicks if direction.lower() == "down" else clicks
            self.mouse.scroll(0, scroll_amount)
            logger.info(f"Scrolled {direction} by {clicks} clicks.")
            return True
        except Exception as e:
            logger.error(f"Failed to scroll: {e}", exc_info=True)
            return False

    def drag_and_drop(self, start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """
        Clicks and holds at a start position, moves to an end position, and releases.
        
        Args:
            start_x, start_y: Starting coordinates.
            end_x, end_y: Ending coordinates.
            
        Returns:
            bool: True if successful.
        """
        try:
            # Move to start position
            self.mouse.position = (start_x, start_y)
            time.sleep(0.2)  # Short pause to let OS register movement
            
            # Press and hold left click
            self.mouse.press(Button.left)
            logger.debug(f"Mouse pressed at ({start_x}, {start_y})")
            time.sleep(0.2)
            
            # Drag to end position
            self.mouse.position = (end_x, end_y)
            logger.debug(f"Mouse dragged to ({end_x}, {end_y})")
            time.sleep(0.5)  # Pause to simulate human drag duration
            
            # Release click
            self.mouse.release(Button.left)
            logger.info(f"Successfully Dragged & Dropped from ({start_x}, {start_y}) to ({end_x}, {end_y}).")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drag and drop: {e}", exc_info=True)
            return False
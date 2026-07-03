"""
Web Browser Plugin for OpenOperator.
Allows the agent to autonomously open URLs in the system's default web browser.
"""

import logging
import webbrowser
from typing import Any, Dict

from openoperator.plugins.base import OpenOperatorPlugin

logger = logging.getLogger(__name__)


class WebBrowserPlugin(OpenOperatorPlugin):
    """
    Plugin to execute web browser launches.
    """

    @property
    def name(self) -> str:
        return "WebBrowser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize any required state."""
        logger.debug("WebBrowserPlugin initialized.")

    def execute(self, command: str, **kwargs) -> Any:
        """
        Executes the browser launch command.
        
        Args:
            command (str): The specific plugin action (e.g., 'open_url').
            kwargs: Must contain 'url' (e.g., 'github.com').
            
        Returns:
            bool: True if URL was opened successfully, False otherwise.
        """
        if command.lower() != "open_url":
            logger.warning(f"WebBrowserPlugin does not support command: '{command}'")
            return False

        url = kwargs.get("url")
        if not url:
            logger.error("WebBrowserPlugin requires 'url' keyword argument.")
            return False

        try:
            # Auto-append https if the LLM or user forgets it
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
                
            logger.info(f"Opening URL in default browser: '{url}'")
            # webbrowser.open returns True if browser launch was successful
            success = webbrowser.open(url)
            return success
        except Exception as e:
            logger.error(f"Failed to open URL '{url}': {e}", exc_info=True)
            return False
"""
Terminal Executor Plugin for OpenOperator.
Allows the agent to autonomously run system commands in the background 
(e.g., mkdir, ping, python scripts) and capture their output.
"""

import logging
import subprocess
from typing import Any, Dict

from openoperator.plugins.base import OpenOperatorPlugin

logger = logging.getLogger(__name__)


class TerminalExecutorPlugin(OpenOperatorPlugin):
    """
    Plugin to execute OS-level terminal commands.
    """

    @property
    def name(self) -> str:
        return "TerminalExecutor"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize any required state."""
        logger.debug("TerminalExecutorPlugin initialized.")

    def execute(self, command: str, **kwargs) -> Any:
        """
        Executes the terminal command.
        
        Args:
            command (str): The specific plugin action (e.g., 'run_command').
            kwargs: Must contain 'cmd' (e.g., 'mkdir new_folder').
            
        Returns:
            bool: True if command executed with exit code 0, False otherwise.
        """
        if command.lower() != "run_command":
            logger.warning(f"TerminalExecutorPlugin does not support command: '{command}'")
            return False

        sys_cmd = kwargs.get("cmd")
        if not sys_cmd:
            logger.error("TerminalExecutorPlugin requires 'cmd' keyword argument.")
            return False

        try:
            logger.info(f"Executing system command: '{sys_cmd}'")
            # We use timeout=15.0 to ensure the agent doesn't get stuck on infinite commands like 'ping -t'
            result = subprocess.run(
                sys_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=15.0
            )
            
            if result.returncode == 0:
                logger.info(f"Command successful. Output:\n{result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Command failed (Code {result.returncode}). Error:\n{result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Command '{sys_cmd}' timed out after 15 seconds. Aborting execution.")
            return False
        except Exception as e:
            logger.error(f"Failed to execute command '{sys_cmd}': {e}", exc_info=True)
            return False
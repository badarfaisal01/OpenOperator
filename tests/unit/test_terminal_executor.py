"""
Tests for the TerminalExecutorPlugin.
Ensures system commands are safely passed and outputs are captured.
"""

import pytest
import subprocess
from unittest.mock import patch, MagicMock
from openoperator.plugins.terminal_executor import TerminalExecutorPlugin


@pytest.fixture
def terminal_plugin():
    plugin = TerminalExecutorPlugin()
    plugin.initialize({})
    return plugin


def test_plugin_metadata(terminal_plugin):
    """Verify plugin properties."""
    assert terminal_plugin.name == "TerminalExecutor"
    assert terminal_plugin.version == "1.0.0"


@patch("subprocess.run")
def test_execute_run_command_success(mock_run, terminal_plugin):
    """Test successful terminal command execution."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "hello world"
    mock_run.return_value = mock_result
    
    result = terminal_plugin.execute("run_command", cmd="echo hello world")
    
    assert result is True
    mock_run.assert_called_once_with(
        "echo hello world", shell=True, capture_output=True, text=True, timeout=15.0
    )


@patch("subprocess.run")
def test_execute_run_command_failure(mock_run, terminal_plugin):
    """Test handling of failed terminal commands (non-zero exit code)."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "command not found"
    mock_run.return_value = mock_result
    
    result = terminal_plugin.execute("run_command", cmd="invalid_sys_cmd")
    
    assert result is False


@patch("subprocess.run")
def test_execute_run_command_timeout(mock_run, terminal_plugin):
    """Test that infinite commands do not hang the agent."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ping -t localhost", timeout=15.0)
    
    result = terminal_plugin.execute("run_command", cmd="ping -t localhost")
    assert result is False


def test_execute_invalid_action(terminal_plugin):
    """Test that unsupported plugin actions fail gracefully."""
    result = terminal_plugin.execute("delete_system", cmd="echo test")
    assert result is False


def test_execute_missing_cmd(terminal_plugin):
    """Test that missing required arguments are handled."""
    result = terminal_plugin.execute("run_command")
    assert result is False
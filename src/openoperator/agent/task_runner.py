"""
Task runner module for OpenOperator Vision Planning.
"""

import logging
import time
from typing import Optional

from openoperator.action.keyboard import KeyboardActionController
from openoperator.action.window_controller import WindowController
from openoperator.agent.action_memory_manager import ActionMemoryManager
from openoperator.agent.vision_actor import VisionActor
from openoperator.agent.vision_models import VisionActionType, VisionTaskPlan
from openoperator.core.verification import VerificationEngine
from openoperator.perception.ocr import OCREngine
from openoperator.perception.screenshot import ScreenshotEngine
from openoperator.plugins.manager import PluginManager
from openoperator.plugins.app_launcher import AppLauncherPlugin
from openoperator.plugins.web_browser import WebBrowserPlugin
from openoperator.plugins.terminal_executor import TerminalExecutorPlugin

logger = logging.getLogger(__name__)


class TaskRunner:
    def __init__(
        self,
        window_controller: Optional[WindowController] = None,
        vision_actor: Optional[VisionActor] = None,
        keyboard_controller: Optional[KeyboardActionController] = None,
        screenshot_engine: Optional[ScreenshotEngine] = None,
        ocr_engine: Optional[OCREngine] = None,
        verification_engine: Optional[VerificationEngine] = None,
        action_memory_manager: Optional[ActionMemoryManager] = None,
    ) -> None:
        self.window = window_controller or WindowController()
        self.actor = vision_actor or VisionActor()
        self.keyboard = keyboard_controller or KeyboardActionController()
        self.screenshot = screenshot_engine or ScreenshotEngine()
        self.ocr = ocr_engine or OCREngine()
        self.verification = verification_engine or VerificationEngine()
        self.memory = action_memory_manager or ActionMemoryManager()
        
        self.plugin_manager = PluginManager()
        self.plugin_manager.register_plugin(AppLauncherPlugin)
        self.plugin_manager.register_plugin(WebBrowserPlugin)
        self.plugin_manager.register_plugin(TerminalExecutorPlugin)

    def execute_plan(
        self, plan: VisionTaskPlan, delay_between_steps: float = 1.0, max_retries: int = 3, dynamic_delay: float = 1.5
    ) -> bool:
        if not plan.is_executable or not plan.steps:
            return False

        for step in plan.steps:
            success = False

            for attempt in range(max_retries):
                try:
                    if step.action_type == VisionActionType.RUN_COMMAND:
                        success = self.plugin_manager.execute_plugin_command("TerminalExecutor", "run_command", cmd=step.input_data)
                    
                    elif step.action_type == VisionActionType.OPEN_URL:
                        success = self.plugin_manager.execute_plugin_command("WebBrowser", "open_url", url=step.target_element)
                        if success:
                            time.sleep(3.0)
                            if self.memory: self.memory.remember_window(step.target_element)

                    elif step.action_type == VisionActionType.LAUNCH_APP:
                        success = self.plugin_manager.execute_plugin_command("AppLauncher", "launch", app_name=step.target_element)
                        if success:
                            time.sleep(2.0)
                            if self.memory: self.memory.remember_window(step.target_element)

                    elif step.action_type == VisionActionType.FOCUS_WINDOW:
                        success = self.window.focus_window_by_title(step.target_element)
                        if not success and attempt == max_retries - 1:
                            success = self.plugin_manager.execute_plugin_command("AppLauncher", "launch", app_name=step.target_element)
                            if success:
                                time.sleep(2.0)
                                self.window.focus_window_by_title(step.target_element)
                        if success and self.memory: self.memory.remember_window(step.target_element)

                    elif step.action_type == VisionActionType.CLICK_TEXT:
                        success = self.actor.click_text(step.target_element)
                        if success and self.memory: self.memory.remember_click(step.target_element)

                    elif step.action_type == VisionActionType.TYPE_TEXT:
                        success = self.keyboard.type_text(step.input_data)
                        if success and self.memory: self.memory.remember_type(step.input_data)

                    elif step.action_type == VisionActionType.VERIFY_STATE:
                        image_bytes = self.screenshot.capture_screen()
                        if image_bytes:
                            screen_text = self.ocr.extract_text(image_bytes)
                            if screen_text:
                                result = self.verification.verify_text_present(step.input_data, screen_text)
                                success = result.success

                    if success: break
                    if attempt < max_retries - 1: time.sleep(dynamic_delay)

                except Exception:
                    if attempt < max_retries - 1: time.sleep(dynamic_delay)
                    else: return False

            if not success: return False
            time.sleep(delay_between_steps)

        return True
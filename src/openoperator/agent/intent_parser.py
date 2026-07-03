"""
Intent parser module for OpenOperator.
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any

from openoperator.agent.action_memory_manager import ActionMemoryManager
from openoperator.agent.vision_models import VisionActionType, VisionStep, VisionTaskPlan

logger = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = """You are an AI desktop automation agent. Your task is to translate user natural language commands into a strict JSON array of execution steps.
Users may provide instructions in English, Hindi, or Hinglish.

Allowed Action Types:
- FOCUS_WINDOW: Brings an application to the foreground. Requires 'target_element'.
- LAUNCH_APP: Starts a new application. Requires 'target_element'.
- OPEN_URL: Opens a website. Requires 'target_element'.
- RUN_COMMAND: Executes a background terminal/CMD command. Requires 'input_data' (the command string, e.g., 'mkdir test').
- CLICK_TEXT: Clicks on specific text. Requires 'target_element'.
- TYPE_TEXT: Types strings using the keyboard. Requires 'input_data'.
- VERIFY_STATE: Checks if text exists. Requires 'input_data'.

Context Rule: If the user asks to type or click without explicit focus, do NOT invent a FOCUS_WINDOW step. RUN_COMMAND does not need focus.

Output ONLY a valid JSON array. No explanations.
Format example: [{"action_type": "RUN_COMMAND", "input_data": "ping google.com"}]
"""

class VisionIntentParser:
    def __init__(self, memory: Optional[ActionMemoryManager] = None) -> None:
        self.memory = memory
        self.launch_verbs = {"launch", "start", "run", "shuru", "chalu"}
        self.focus_verbs = {"open", "switch to", "focus", "khol", "kholo"}
        self.browse_verbs = {"visit", "browse", "surf", "go to", "website"}
        self.cmd_verbs = {"execute", "cmd", "terminal", "command"}
        self.click_verbs = {"click", "daba", "dabao", "hit"}
        self.type_verbs = {"type", "enter", "likh", "likho", "darj"}
        self.verify_verbs = {"verify", "check", "confirm", "dekho"}
        
        all_verbs = (
            list(self.launch_verbs) + list(self.focus_verbs) + list(self.browse_verbs) +
            list(self.cmd_verbs) + list(self.click_verbs) + list(self.type_verbs) + list(self.verify_verbs)
        )
        all_verbs.sort(key=len, reverse=True)
        escaped_verbs = [re.escape(v) for v in all_verbs]
        verbs_pattern = "|".join(escaped_verbs)
        
        self.pattern = re.compile(
            rf'\b({verbs_pattern})\b(.*?)(?=\b(?:{verbs_pattern})\b|$)', 
            re.IGNORECASE
        )

    def parse(self, prompt: str) -> VisionTaskPlan:
        logger.debug(f"Parsing vision intent from prompt: '{prompt}'")
        steps: list[VisionStep] = []
        missing_context: list[str] = []
        is_executable = True
        
        matches = list(self.pattern.finditer(prompt))
        step_id = 1
        
        has_explicit_focus = any(
            match.group(1).lower() in self.focus_verbs or 
            match.group(1).lower() in self.launch_verbs or
            match.group(1).lower() in self.browse_verbs 
            for match in matches
        )
        
        if not has_explicit_focus and self.memory and self.memory.last_window:
            has_actionable = any(
                match.group(1).lower() in self.click_verbs or match.group(1).lower() in self.type_verbs 
                for match in matches
            )
            if has_actionable:
                steps.append(
                    VisionStep(step_id=step_id, action_type=VisionActionType.FOCUS_WINDOW, target_element=self.memory.last_window, confidence=1.0)
                )
                step_id += 1

        for match in matches:
            verb = match.group(1).lower()
            arg = match.group(2)
            arg = re.sub(r'(?i)(\b(and|then|next|aur|fir|uske baad|do|karo)\b[\s,]*)+$', '', arg).strip(" ,.")
            
            action_type, target_element, input_data = None, None, None
            
            if verb in self.cmd_verbs:
                action_type = VisionActionType.RUN_COMMAND
                input_data = arg
                if not input_data:
                    missing_context.append(f"Missing command for '{verb}' action.")
                    is_executable = False
            elif verb in self.browse_verbs:
                action_type = VisionActionType.OPEN_URL
                target_element = arg
            elif verb in self.launch_verbs:
                action_type = VisionActionType.LAUNCH_APP
                target_element = arg
            elif verb in self.focus_verbs:
                if "." in arg and (" " not in arg):
                    action_type = VisionActionType.OPEN_URL
                else:
                    action_type = VisionActionType.FOCUS_WINDOW
                target_element = arg
            elif verb in self.click_verbs:
                action_type = VisionActionType.CLICK_TEXT
                if arg.lower().startswith("on ") or arg.lower().startswith("pe "):
                    arg = arg[3:].strip()
                target_element = arg
            elif verb in self.type_verbs:
                action_type = VisionActionType.TYPE_TEXT
                input_data = arg
            elif verb in self.verify_verbs:
                action_type = VisionActionType.VERIFY_STATE
                input_data = arg
            
            if action_type:
                steps.append(VisionStep(step_id=step_id, action_type=action_type, target_element=target_element, input_data=input_data, confidence=1.0))
                step_id += 1
                
        if not steps:
            is_executable = False
            missing_context.append("No valid action keywords found.")
            
        return VisionTaskPlan(original_prompt=prompt, steps=steps, is_executable=is_executable, missing_context=missing_context)


class LLMIntentParser(VisionIntentParser):
    def __init__(self, memory: Optional[ActionMemoryManager] = None, endpoint: str = None, model: str = None, api_key: str = None) -> None:
        super().__init__(memory)
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "llama3")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")

    def parse(self, prompt: str) -> VisionTaskPlan:
        try:
            steps_data = self._call_llm(prompt)
            steps = []
            step_id = 1
            
            has_explicit_focus = any(s.get("action_type") in ["FOCUS_WINDOW", "LAUNCH_APP", "OPEN_URL", "RUN_COMMAND"] for s in steps_data)
            if not has_explicit_focus and self.memory and self.memory.last_window:
                has_actionable = any(s.get("action_type") in ["CLICK_TEXT", "TYPE_TEXT"] for s in steps_data)
                if has_actionable:
                    steps.append(
                        VisionStep(step_id=step_id, action_type=VisionActionType.FOCUS_WINDOW, target_element=self.memory.last_window, confidence=1.0)
                    )
                    step_id += 1

            for data in steps_data:
                action_str = data.get("action_type", "")
                try:
                    action_type = VisionActionType[action_str]
                except KeyError:
                    continue
                    
                steps.append(
                    VisionStep(
                        step_id=step_id,
                        action_type=action_type,
                        target_element=data.get("target_element"),
                        input_data=data.get("input_data"),
                        confidence=0.95 
                    )
                )
                step_id += 1

            if not steps:
                raise ValueError("LLM returned empty sequence.")

            return VisionTaskPlan(original_prompt=prompt, steps=steps, is_executable=True, missing_context=[])

        except Exception as e:
            logger.warning(f"LLM parsing failed ({e}). Falling back to Regex NLP.")
            return super().parse(prompt)

    def _call_llm(self, prompt: str) -> List[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }

        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60.0) as response:
            result = json.loads(response.read().decode('utf-8'))
            
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        if content.startswith("```json"): content = content[7:]
        elif content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        return json.loads(content.strip())
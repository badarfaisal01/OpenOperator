"""
Data models for Natural Language Vision Planning.
"""

from enum import Enum
from pydantic import BaseModel


class VisionActionType(str, Enum):
    """
    Enumeration of supported vision-guided and system actions.
    """
    FOCUS_WINDOW = "FOCUS_WINDOW"
    LAUNCH_APP = "LAUNCH_APP"
    OPEN_URL = "OPEN_URL"
    RUN_COMMAND = "RUN_COMMAND"  # New Action for Terminal execution
    CLICK_TEXT = "CLICK_TEXT"
    TYPE_TEXT = "TYPE_TEXT"
    VERIFY_STATE = "VERIFY_STATE"


class VisionStep(BaseModel):
    step_id: int
    action_type: VisionActionType
    target_element: str | None = None
    input_data: str | None = None
    confidence: float


class VisionTaskPlan(BaseModel):
    original_prompt: str
    steps: list[VisionStep]
    is_executable: bool
    missing_context: list[str]
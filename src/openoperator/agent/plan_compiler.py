"""
Plan Compiler module for OpenOperator.
"""

import logging
from typing import List

from openoperator.agent.vision_models import VisionActionType, VisionStep, VisionTaskPlan

logger = logging.getLogger(__name__)


class VisionPlanCompiler:
    def compile(self, plan: VisionTaskPlan) -> VisionTaskPlan:
        if not plan.is_executable or not plan.steps:
            if not plan.steps:
                plan.is_executable = False
                plan.missing_context.append("Plan contains no executable steps.")
            return plan

        optimized_steps: List[VisionStep] = []
        has_context_established = False

        for step in plan.steps:
            if step.action_type == VisionActionType.CLICK_TEXT and not step.target_element:
                plan.is_executable = False
                return plan
            if step.action_type in (VisionActionType.VERIFY_STATE, VisionActionType.TYPE_TEXT, VisionActionType.RUN_COMMAND) and not step.input_data:
                plan.is_executable = False
                return plan
            if step.action_type in (VisionActionType.LAUNCH_APP, VisionActionType.OPEN_URL) and not step.target_element:
                plan.is_executable = False
                return plan

            if step.action_type in (VisionActionType.FOCUS_WINDOW, VisionActionType.CLICK_TEXT, VisionActionType.LAUNCH_APP, VisionActionType.OPEN_URL, VisionActionType.RUN_COMMAND):
                has_context_established = True
                
            if step.action_type == VisionActionType.TYPE_TEXT and not has_context_established:
                plan.is_executable = False
                return plan

            if optimized_steps:
                last_step = optimized_steps[-1]
                if (step.action_type == VisionActionType.FOCUS_WINDOW and 
                    last_step.action_type == VisionActionType.FOCUS_WINDOW and 
                    step.target_element == last_step.target_element):
                    continue

            step.step_id = len(optimized_steps) + 1
            optimized_steps.append(step)

        plan.steps = optimized_steps
        return plan
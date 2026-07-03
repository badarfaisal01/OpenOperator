"""
End-to-End Autonomous Agent Showcase.

This interactive demo ties together the entire OpenOperator pipeline:
1. LLM Intent Parsing (Understanding Natural Language in English/Hindi/Hinglish)
2. Vision Plan Compilation (Safety Validation & Optimization)
3. Dynamic Task Running (State-Aware Execution & Retries)
"""

import logging

from openoperator.agent.intent_parser import LLMIntentParser
from openoperator.agent.plan_compiler import VisionPlanCompiler
from openoperator.agent.task_runner import TaskRunner

# Configure clean logging for the demo console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("🚀 OpenOperator: Autonomous Agent End-to-End Demo")
    print("=" * 60)
    
    logger.info("Initializing Agent Brain, Eyes, and Hands...")
    
    # 1. Initialize the intelligent parser (LLM with Regex fallback)
    parser = LLMIntentParser()
    
    # 2. Initialize the optimizer/compiler
    compiler = VisionPlanCompiler()
    
    # 3. Initialize the executor (Task Runner)
    runner = TaskRunner()
    
    print("\n✅ Agent is Ready! (Type 'exit' or 'quit' to stop)")
    print("💡 Example command: 'open Notepad and type hello world'")
    print("💡 Hinglish command: 'notepad khol do aur hello type karo'")
    
    while True:
        try:
            print("\n" + "-" * 60)
            user_prompt = input("🤖 Your Command: ").strip()
            
            if user_prompt.lower() in ['exit', 'quit']:
                print("Shutting down Agent. Goodbye! 👋")
                break
                
            if not user_prompt:
                continue
                
            # --- PIPELINE STEP 1: THINK (Parse) ---
            logger.info("Step 1: Parsing Intent...")
            raw_plan = parser.parse(user_prompt)
            
            # --- PIPELINE STEP 2: OPTIMIZE (Compile) ---
            logger.info("Step 2: Compiling & Optimizing Plan...")
            compiled_plan = compiler.compile(raw_plan)
            
            if not compiled_plan.is_executable:
                logger.error("❌ Plan compilation failed. Missing context or safety violation.")
                for err in compiled_plan.missing_context:
                    logger.error(f"  -> {err}")
                continue
            
            # --- PIPELINE STEP 3: ACT (Execute) ---
            logger.info("Step 3: Executing Actions...")
            success = runner.execute_plan(compiled_plan)
            
            if success:
                print("✅ Task Completed Successfully!")
            else:
                print("❌ Task Failed during execution.")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Shutting down...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
"""Agent specialized in testing code."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class CodeTesterAgent(BaseAgent):
    """Agent specialized in writing and revising content."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "code_tester"
    ):
        if instruction is None:
            instruction = """You are a code quality analyst working with the python_repl tool. Your mission is to test code by running it through python_repl to test functionality, identify potential bugs or inefficiencies, and suggest specific, concrete solutions, fixes and improvements with example code. When providing feedback, include both the original and improved code segments for comparison. Use the following format to run and test code:
            
            <tool>
            python_repl(
                code="The code to test",
                test="The test case to run"
            )
            </tool>
            
            Always wrap the tool call in <tool> tags. Put each parameter on a new line with proper indentation. Use proper Python literal syntax for values: Strings in double quotes, numbers without quotes, booleans as true/false, lists in square brackets, and dictionaries in curly braces. Always provide required parameters.
            
            Your response should be in JSON format with the following schema:
            {
                "status": "success" | "error",
                "message": "A description of the test and its outcome",
                "test_results": {
                    "test_name": "Result of the test"
                },
                "error_details": "Details of any errors encountered"
            }
            """
                           
        if model_config is None:
            model_config = {
                "temperature": 0.8,
                "max_tokens": 8000,
                "top_p": 0.9
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="code_tester",
            instruction=instruction,
            tools=["python_repl"],
            model_config=model_config,
            name=name
        )
        
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Test code based on the task."""
        if not self.validate_task(task):
            return {
                "status": "error",
                "message": "Invalid task format",
                "error_details": "Missing required fields in task"
            }
            
        # Extract task components
        code = task["code"]
        test_cases = task.get("test_cases", [])
        
        if not code or not test_cases:
            return {
                "status": "error",
                "message": "Missing code or test cases",
                "error_details": "Both 'code' and 'test_cases' are required"
            }
            
        test_results = self._run_tests(code, test_cases)
        
        return {
            "status": "success",
            "message": "Code testing completed",
            "test_results": test_results,
            "error_details": ""
        }
        
    def _run_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run provided test cases against the code."""
        test_results = {}
        for test_case in test_cases:
            result = self.tool_manager.use_tool(
                "python_repl",
                {"code": code, "test": test_case}
            )
            test_results[test_case["name"]] = result
        return test_results
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate the task has required fields."""
        return "code" in task and "test_cases" in task

"""Agent responsible for implementing features, fixing bugs, and generating code."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class CodeEngineerAgent(BaseAgent):
    """Agent responsible for implementing features, fixing bugs, and generating code."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "code_engineer"
    ):
        if instruction is None:
            instruction = """You are a meticulous and rigorous code engineer. Your primary goal is to create correct, functional, and efficient code that is well-documented, easy to read, and well-structured. You should emulate the best practices of a senior software engineer when generating code. Provide clear, concise, and well-commented code snippets. Focus on getting the details right, since the user will be relying on your output as a reference implementation. You can use the following tools:
            - `python_repl`: To execute and test Python code snippets.
            - `generate_code`: To generate code based on requirements.
            - `refactor_code`: To improve the structure and readability of existing code.
            - `document_code`: To generate documentation for code.
            - `test_code`: To generate unit tests for code.
            
            When using tools, format your requests like this:
            <tool>
            tool_name(
                param1="value1",
                param2=123,
                param3=true
            )
            </tool>
            
            Always wrap the tool call in <tool> tags. Put each parameter on a new line with proper indentation. Use proper Python literal syntax for values: Strings in double quotes, numbers without quotes, booleans as true/false, lists in square brackets, and dictionaries in curly braces. Always provide required parameters.
            
            Your response should be in JSON format with the following schema:
            {
                "task_type": "self_review" | "address_feedback" | "generate_code",
                "status": "success" | "error",
                "message": "A description of the task and its outcome",
                "code": "The generated or modified code",
                "documentation": "Generated documentation if applicable",
                "tests": "Generated tests if applicable",
                "error_details": "Details of any errors encountered"
            }
            """
                           
        if model_config is None:
            model_config = {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 16000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="code_engineer",
            instruction=instruction,
            tools=["python_repl", "generate_code", "refactor_code", "document_code", "test_code"],
            model_config=model_config,
            name=name
        )
        
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task and generate code or address feedback."""
        if not self.validate_task(task):
            return {
                "status": "error",
                "message": "Invalid task format",
                "error_details": "Missing required fields in task"
            }
            
        task_type = task.get("task_type")
        
        if task_type == "self_review":
            return self._perform_self_review(task)
        elif task_type == "address_feedback":
            return self._address_feedback(task)
        elif task_type == "generate_code":
            return self._generate_code(task)
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
                "error_details": "Invalid task_type provided"
            }
            
    def _perform_self_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for self code review."""
        # Placeholder for self-review logic
        return {
            "task_type": "self_review",
            "status": "success",
            "message": "Self-review completed (placeholder)",
            "code": task.get("code", ""),
            "documentation": "",
            "tests": "",
            "error_details": ""
        }
        
    def _address_feedback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for addressing review feedback."""
        # Placeholder for addressing feedback logic
        return {
            "task_type": "address_feedback",
            "status": "success",
            "message": "Feedback addressed (placeholder)",
            "code": task.get("code", ""),
            "documentation": "",
            "tests": "",
            "error_details": ""
        }
        
    def _generate_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for generating new code."""
        requirements = task.get("requirements", "")
        language = task.get("language", "")
        style_guide = task.get("style_guide", "")
        dependencies = task.get("dependencies", [])
        test_requirements = task.get("test_requirements", [])
        
        if not requirements or not language:
            return {
                "task_type": "generate_code",
                "status": "error",
                "message": "Missing requirements or language for code generation",
                "code": "",
                "documentation": "",
                "tests": "",
                "error_details": "Both 'requirements' and 'language' are required"
            }
            
        code_structure = self._plan_code_structure(requirements, language)
        code = self._generate_code_implementation(
            code_structure,
            language,
            style_guide,
            dependencies
        )
        documentation = self._generate_documentation(code, requirements)
        tests = self._generate_tests(code, test_requirements)
        
        return {
            "task_type": "generate_code",
            "status": "success",
            "message": "Code generated successfully",
            "code": code,
            "documentation": documentation,
            "tests": tests,
            "error_details": ""
        }
        
    def _plan_code_structure(
        self,
        requirements: str,
        language: str
    ) -> Dict[str, Any]:
        """Plan the high-level structure of the code."""
        planning_prompt = self._construct_planning_prompt(requirements, language)
        structure = self.generate_response(planning_prompt)
        return self._parse_code_structure(structure)
        
    def _generate_code_implementation(
        self,
        structure: Dict[str, Any],
        language: str,
        style_guide: str,
        dependencies: List[str]
    ) -> str:
        """Generate the actual code implementation."""
        implementation_prompt = self._construct_implementation_prompt(
            structure,
            language,
            style_guide,
            dependencies
        )
        code = self.generate_response(implementation_prompt)
        
        # Verify syntax
        if language.lower() == "python":
            self.tool_manager.use_tool(
                "python_repl",
                {"code": code, "action": "verify_syntax"}
            )
            
        return code
        
    def _generate_documentation(
        self,
        code: str,
        requirements: str
    ) -> Dict[str, Any]:
        """Generate comprehensive documentation for the code."""
        return {
            "overview": self._generate_overview(code, requirements),
            "api_docs": self._generate_api_documentation(code),
            "examples": self._generate_usage_examples(code),
            "dependencies": self._document_dependencies(code)
        }
        
    def _generate_tests(
        self,
        code: str,
        test_requirements: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive test suite."""
        test_cases = []
        
        # Generate unit tests
        unit_tests = self._generate_unit_tests(code, test_requirements)
        test_cases.extend(unit_tests)
        
        # Generate integration tests if needed
        if self._needs_integration_tests(code):
            integration_tests = self._generate_integration_tests(code)
            test_cases.extend(integration_tests)
            
        # Generate edge case tests
        edge_cases = self._generate_edge_case_tests(code)
        test_cases.extend(edge_cases)
        
        return test_cases
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "task_type" in task

"""Coder agent for generating high-quality, well-structured code."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class CoderAgent(BaseAgent):
    """Agent specialized in writing high-quality, well-documented code."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "coder"
    ):
        if instruction is None:
            instruction = """You are a meticulous, rigorous coder. Your primary goal is to create correct, 
                           functional, and efficient code that is well-documented, easy to read, and 
                           well-structured. You should emulate the best practices of a senior software engineer 
                           when generating code. Provide clear, concise, and well-commented code snippets. Focus on 
                           getting the details right, since the user will be relying on your output as a reference 
                           implementation."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.5,  # Lower temperature for more precise output
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 16000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="coder",
            instruction=instruction,
            tools=["python_repl", "generate_code", "refactor_code", "document_code", "test_code"],
            model_config=model_config,
            name=name
        )
        
    def generate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on the task requirements.
        
        Args:
            task: Dictionary containing:
                - requirements: Description of what the code should do
                - language: Programming language to use
                - style_guide: Optional style guide to follow
                - dependencies: Optional list of required dependencies
                - test_requirements: Optional testing requirements
                
        Returns:
            Dictionary containing:
                - code: The generated code
                - documentation: Generated documentation
                - tests: Unit tests for the code
                - metadata: Additional information about the code
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["requirements", "language"]
            }
            
        # Extract task components
        requirements = task["requirements"]
        language = task["language"]
        style_guide = task.get("style_guide", "")
        dependencies = task.get("dependencies", [])
        test_requirements = task.get("test_requirements", [])
        
        # Generate code structure
        code_structure = self._plan_code_structure(requirements, language)
        
        # Generate the code
        code = self._generate_code_implementation(
            code_structure,
            language,
            style_guide,
            dependencies
        )
        
        # Generate documentation
        documentation = self._generate_documentation(code, requirements)
        
        # Generate tests
        tests = self._generate_tests(code, test_requirements)
        
        # Verify the implementation
        verification = self._verify_implementation(code, tests)
        
        # Generate metadata
        metadata = self._compile_code_metadata(
            code,
            documentation,
            tests,
            verification
        )
        
        return {
            "code": code,
            "documentation": documentation,
            "tests": tests,
            "metadata": metadata
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
        
    def _verify_implementation(
        self,
        code: str,
        tests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify the implementation against requirements and tests."""
        verification_results = {
            "syntax_check": self._check_syntax(code),
            "style_check": self._check_style(code),
            "test_results": self._run_test_suite(code, tests),
            "static_analysis": self._run_static_analysis(code)
        }
        
        return verification_results
        
    def _compile_code_metadata(
        self,
        code: str,
        documentation: Dict[str, Any],
        tests: List[Dict[str, Any]],
        verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the generated code."""
        return {
            "complexity_metrics": self._calculate_complexity(code),
            "documentation_coverage": self._analyze_documentation_coverage(
                code,
                documentation
            ),
            "test_coverage": self._calculate_test_coverage(code, tests),
            "verification_summary": self._summarize_verification(verification)
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return all(field in task for field in ["requirements", "language"])
